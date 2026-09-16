"""Preservation verification and QC.

Implements the exact protected-sample test from validated-solution.md.

Enhanced with:
- Chroma-aware verification for YCbCr 4:2:0, 4:2:2, 4:4:4
- Bit-exact floating point comparison (handles NaN, Inf, signed zero)
- Plane-specific masks for subsampled formats
- Post-decode verification support
"""

from dataclasses import dataclass
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from preserve.models import PreservationReport


@dataclass
class FrameVerification:
    frame_index: int
    max_diff: float
    mean_diff: float
    changed_pixels: int
    total_protected_pixels: int
    passed: bool
    # Per-plane details for YCbCr
    plane_details: dict | None = None


def _broadcast_protected(
    protected: NDArray[np.bool_],
    target_shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    """Broadcast 2D mask to target shape."""
    if protected.ndim == 2 and len(target_shape) == 3:
        return np.broadcast_to(protected[..., np.newaxis], target_shape)
    return protected


def _bit_exact_equal(
    a: NDArray,
    b: NDArray,
    dtype: np.dtype,
) -> NDArray[np.bool_]:
    """Bit-exact equality for floating point arrays.

    Handles NaN payloads, signed zero, and Inf correctly.
    For integer dtypes, uses regular equality.
    """
    if np.issubdtype(dtype, np.floating):
        # View as unsigned integers for bit-pattern comparison
        unsigned_dtype = np.dtype(f"u{dtype.itemsize}")
        a_bits = np.ascontiguousarray(a).view(unsigned_dtype)
        b_bits = np.ascontiguousarray(b).view(unsigned_dtype)
        return a_bits == b_bits
    return a == b


def _max_abs_diff_floating(
    a: NDArray,
    b: NDArray,
    dtype: np.dtype,
) -> float:
    """Compute max absolute difference handling NaN/Inf correctly."""
    if np.issubdtype(dtype, np.floating):
        # For floating point, only compare finite values
        finite_mask = np.isfinite(a) & np.isfinite(b)
        if not finite_mask.any():
            return 0.0
        return float(np.abs(a[finite_mask] - b[finite_mask]).max())
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())


def _compute_chroma_masks(
    protected: NDArray[np.bool_],
    chroma_subsampling: str = "4:4:4",
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Compute per-plane protected masks for YCbCr formats.

    Args:
        protected: 2D boolean mask (H, W) for luma plane
        chroma_subsampling: "4:4:4", "4:2:2", or "4:2:0"

    Returns:
        Tuple of (Y_mask, Cb_mask, Cr_mask) for each plane
    """
    h, w = protected.shape

    if chroma_subsampling == "4:4:4":
        y_mask = protected
        cb_mask = protected
        cr_mask = protected
    elif chroma_subsampling == "4:2:2":
        # Chroma is subsampled horizontally: 2 luma pixels share 1 chroma
        y_mask = protected
        cb_mask = protected[:, ::2]
        cr_mask = protected[:, ::2]
    elif chroma_subsampling == "4:2:0":
        # Chroma subsampled both horizontally and vertically
        y_mask = protected
        cb_mask = protected[::2, ::2]
        cr_mask = protected[::2, ::2]
    else:
        raise ValueError(f"Unknown chroma subsampling: {chroma_subsampling}")

    return y_mask, cb_mask, cr_mask


def verify_frame_preservation(
    output: NDArray[np.uint8],
    source: NDArray[np.uint8],
    protected: NDArray[np.bool_],
    tolerance: int = 0,
    bit_exact: bool = False,
    chroma_subsampling: str = "4:4:4",
) -> FrameVerification:
    """Verify protected pixels match source exactly.

    Args:
        output: Output frame (H, W, C) or (H, W)
        source: Source frame (same shape as output)
        protected: Boolean mask of protected pixels (H, W) or broadcastable
        tolerance: Maximum allowed difference (0 for exact match)
        bit_exact: If True, use bit-pattern equality for floats
        chroma_subsampling: Chroma subsampling format for YCbCr

    Returns:
        FrameVerification with detailed results
    """
    if output.shape != source.shape:
        raise ValueError(f"Shape mismatch: {output.shape} vs {source.shape}")
    if output.dtype != source.dtype:
        raise ValueError(f"Dtype mismatch: {output.dtype} vs {source.dtype}")

    # For packed RGB, use simple approach
    if output.ndim == 3 and output.shape[2] == 3:
        protected_bc = _broadcast_protected(protected, output.shape)

        if bit_exact and np.issubdtype(output.dtype, np.floating):
            mismatch = ~_bit_exact_equal(output, source, output.dtype)
            diff = np.abs(output.astype(np.float32) - source.astype(np.float32))
        else:
            diff = np.abs(output.astype(np.int32) - source.astype(np.int32))
            mismatch = diff > tolerance

        protected_diff = diff[protected_bc]
        protected_mismatch = mismatch[protected_bc]

        max_diff = float(protected_diff.max()) if protected_diff.size > 0 else 0.0
        mean_diff = float(protected_diff.mean()) if protected_diff.size > 0 else 0.0
        changed = int(protected_mismatch.sum())
        total = int(protected_bc.sum())

        return FrameVerification(
            frame_index=-1,
            max_diff=max_diff,
            mean_diff=mean_diff,
            changed_pixels=changed,
            total_protected_pixels=total,
            passed=(changed == 0),
        )

    # For YCbCr planar or other formats, verify per-plane
    if output.ndim == 3 and output.shape[2] == 3:
        # Assume YCbCr packed - verify each channel
        plane_details = {}
        total_changed = 0
        total_protected = 0
        max_diff_overall = 0.0
        mean_diff_sum = 0.0

        for plane_idx, plane_name in enumerate(["Y", "Cb", "Cr"]):
            plane_output = output[..., plane_idx]
            plane_source = source[..., plane_idx]

            # Compute plane-specific protected mask
            if chroma_subsampling == "4:4:4":
                plane_protected = protected
            elif chroma_subsampling == "4:2:2":
                plane_protected = protected if plane_idx == 0 else protected[:, ::2]
            elif chroma_subsampling == "4:2:0":
                plane_protected = protected if plane_idx == 0 else protected[::2, ::2]
            else:
                plane_protected = protected

            if plane_protected.shape != plane_output.shape:
                # Resize protected mask if needed
                from cv2 import INTER_NEAREST, resize

                plane_protected = (
                    resize(
                        plane_protected.astype(np.uint8),
                        (plane_output.shape[1], plane_output.shape[0]),
                        interpolation=INTER_NEAREST,
                    )
                    > 0
                )

            plane_protected = plane_protected.astype(bool)

            if bit_exact and np.issubdtype(output.dtype, np.floating):
                plane_mismatch = ~_bit_exact_equal(plane_output, plane_source, output.dtype)
                plane_diff = np.abs(
                    plane_output.astype(np.float32) - plane_source.astype(np.float32)
                )
            else:
                plane_diff = np.abs(plane_output.astype(np.int32) - plane_source.astype(np.int32))
                plane_mismatch = plane_diff > tolerance

            plane_protected_diff = plane_diff[plane_protected]
            plane_protected_mismatch = plane_mismatch[plane_protected]

            plane_changed = int(plane_protected_mismatch.sum())
            plane_total = int(plane_protected.sum())
            plane_max = float(plane_protected_diff.max()) if plane_protected_diff.size > 0 else 0.0
            plane_mean = (
                float(plane_protected_diff.mean()) if plane_protected_diff.size > 0 else 0.0
            )

            plane_details[plane_name] = {
                "changed_pixels": plane_changed,
                "total_protected": plane_total,
                "max_diff": plane_max,
                "mean_diff": plane_mean,
                "passed": plane_changed == 0,
            }

            total_changed += plane_changed
            total_protected += plane_total
            max_diff_overall = max(max_diff_overall, plane_max)
            mean_diff_sum += plane_mean

        return FrameVerification(
            frame_index=-1,
            max_diff=max_diff_overall,
            mean_diff=mean_diff_sum / 3 if total_protected > 0 else 0.0,
            changed_pixels=total_changed,
            total_protected_pixels=total_protected,
            passed=(total_changed == 0),
            plane_details=plane_details,
        )

    # Fallback for other formats
    protected_bc = _broadcast_protected(protected, output.shape)
    diff = np.abs(output.astype(np.int32) - source.astype(np.int32))
    protected_diff = diff[protected_bc]
    changed = int((protected_diff > tolerance).sum())
    total = int(protected_bc.sum())

    return FrameVerification(
        frame_index=-1,
        max_diff=float(protected_diff.max()) if protected_diff.size > 0 else 0.0,
        mean_diff=float(protected_diff.mean()) if protected_diff.size > 0 else 0.0,
        changed_pixels=changed,
        total_protected_pixels=total,
        passed=(changed == 0),
    )


def verify_sequence_preservation(
    outputs: list[NDArray[np.uint8]],
    sources: list[NDArray[np.uint8]],
    protected: NDArray[np.bool_],
    job_id: UUID,
    edited_frame_indices: list[int] | None = None,
    tolerance: int = 0,
    bit_exact: bool = False,
    chroma_subsampling: str = "4:4:4",
) -> PreservationReport:
    """Verify preservation across frame sequence.

    Args:
        outputs: List of output frames
        sources: List of source frames
        protected: Protected mask per frame (T, H, W) boolean
        job_id: Job UUID for report
        edited_frame_indices: Indices of frames that were edited
        tolerance: Maximum allowed difference
        bit_exact: Use bit-pattern equality for floats
        chroma_subsampling: Chroma subsampling format

    Returns:
        PreservationReport with aggregated results
    """
    if len(outputs) != len(sources):
        raise ValueError("Output and source counts must match")
    if len(outputs) != len(protected):
        raise ValueError("Output and protected mask counts must match")

    details: list[dict] = []
    total_max_diff = 0.0
    total_mean_diff = 0.0
    total_changed = 0
    all_passed = True
    all_plane_details = {}

    edited_set = set(edited_frame_indices) if edited_frame_indices else set(range(len(outputs)))

    for i, (out, src, prot) in enumerate(zip(outputs, sources, protected, strict=True)):
        result = verify_frame_preservation(out, src, prot, tolerance, bit_exact, chroma_subsampling)
        result.frame_index = i

        if i in edited_set:
            detail = {
                "frame": i,
                "max_diff": result.max_diff,
                "mean_diff": result.mean_diff,
                "changed_pixels": result.changed_pixels,
                "passed": result.passed,
            }
            if result.plane_details:
                detail["plane_details"] = result.plane_details
            details.append(detail)

            total_max_diff = max(total_max_diff, result.max_diff)
            total_mean_diff += result.mean_diff
            total_changed += result.changed_pixels
            if not result.passed:
                all_passed = False

            if result.plane_details:
                for plane, pdetail in result.plane_details.items():
                    if plane not in all_plane_details:
                        all_plane_details[plane] = {
                            "total_changed": 0,
                            "total_protected": 0,
                            "max_diff": 0.0,
                        }
                    all_plane_details[plane]["total_changed"] += pdetail["changed_pixels"]
                    all_plane_details[plane]["total_protected"] += pdetail["total_protected"]
                    all_plane_details[plane]["max_diff"] = max(
                        all_plane_details[plane]["max_diff"], pdetail["max_diff"]
                    )

    edited_count = len(edited_set)

    # Add plane details to report if available
    report = PreservationReport(
        job_id=job_id,
        total_frames=len(outputs),
        edited_frames=edited_count,
        protected_frames=len(outputs) - edited_count,
        max_diff_outside_mask=total_max_diff,
        mean_diff_outside_mask=total_mean_diff / edited_count if edited_count > 0 else 0.0,
        changed_pixels_outside_mask=total_changed,
        passed=all_passed,
        details=details,
    )

    # Attach plane details for downstream use
    if all_plane_details:
        report.plane_details = all_plane_details

    return report


def verify_preservation_post_decode(
    master_path: str,
    sources: list[NDArray[np.uint8]],
    protected: NDArray[np.bool_],
    job_id: UUID,
    edited_frame_indices: list[int] | None = None,
    tolerance: int = 0,
) -> PreservationReport:
    """Verify preservation on a decoded master file.

    This is the critical validation from validated-solution.md: the guarantee
    must hold on the delivered file , not just pre-encode arrays.
    """
    from preserve.video import extract_frames

    master_frames = extract_frames(master_path)
    if len(master_frames) != len(sources):
        raise ValueError(
            f"Master decode returned {len(master_frames)} frames, expected {len(sources)}"
        )

    return verify_sequence_preservation(
        master_frames, sources, protected, job_id, edited_frame_indices, tolerance
    )


def compute_preservation_metrics(
    output: NDArray[np.uint8],
    source: NDArray[np.uint8],
    protected: NDArray[np.bool_],
) -> dict:
    """Compute detailed preservation metrics for reporting.

    Returns dict with:
    - max_abs_diff: Maximum absolute difference in protected region
    - mean_abs_diff: Mean absolute difference
    - rmse: Root mean square error
    - bit_exact_match_pct: Percentage of exactly matching samples
    - changed_pixel_bbox: Bounding box of changed pixels
    """
    protected_bc = _broadcast_protected(protected, output.shape)
    diff = np.abs(output.astype(np.int32) - source.astype(np.int32))
    protected_diff = diff[protected_bc]

    if protected_diff.size == 0:
        return {
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
            "rmse": 0.0,
            "bit_exact_match_pct": 100.0,
            "changed_pixel_bbox": None,
        }

    # Bit-exact match percentage
    exact_match = (output == source) & protected_bc
    match_pct = 100.0 * exact_match.sum() / protected_bc.sum()

    # Bounding box of changes
    changed = (diff > 0) & protected_bc
    if changed.any():
        coords = np.argwhere(changed)
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        bbox = (int(x1), int(y1), int(x2), int(y2))
    else:
        bbox = None

    return {
        "max_abs_diff": int(protected_diff.max()),
        "mean_abs_diff": float(protected_diff.mean()),
        "rmse": float(np.sqrt((protected_diff**2).mean())),
        "bit_exact_match_pct": float(match_pct),
        "changed_pixel_bbox": bbox,
    }
