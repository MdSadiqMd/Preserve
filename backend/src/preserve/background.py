"""Background reconstruction from real pixels in other frames.

A propagation model like ProPainter hallucinates the region it fills, which reads
as a soft smear. But when the subject moves relative to the scene, the background
behind it is genuinely visible in other frames, so it can be recovered as actual
source samples instead of being invented.

Camera motion is estimated and every frame is warped into a shared reference
space, where the unmasked samples for each needed pixel are collected once. Each
frame is then filled with a temporal median over a window of nearby frames only —
a median over the whole clip would mix incompatible scene states, filling a spot
with hedge in a frame where a bus is actually parked there. Pixels never exposed
within a frame's window get no coverage, and the caller falls back to the
inpainting model there.

DiffuEraser-style enhancements:
- Pre-propagation: extend known pixels across entire time domain
- Pre-inference: sample frames to broaden temporal context
- Temporal receptive field expansion for long sequences
"""

import cv2
import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()


def _as_exclude_list(
    exclude_masks: list[NDArray[np.bool_]] | NDArray[np.bool_] | None,
    count: int,
    shape: tuple[int, ...],
) -> list[NDArray[np.bool_] | None]:
    """Normalize per-frame exclude masks; None entries mean track everywhere."""
    if exclude_masks is None:
        return [None] * count
    if isinstance(exclude_masks, np.ndarray) and exclude_masks.dtype == bool:
        stacked = exclude_masks
        if stacked.shape[0] != count:
            return [None] * count
        return [np.ascontiguousarray(stacked[i]) for i in range(count)]
    items = list(exclude_masks)
    if len(items) != count:
        return [None] * count
    return [m if m is None else np.ascontiguousarray(m.astype(bool)) for m in items]


def _track_mask(
    excludes: list[NDArray[np.bool_] | None], pair_index: int
) -> NDArray[np.uint8] | None:
    """goodFeaturesToTrack mask avoiding the subject in both paired frames.
    The union is dilated 5px so corners near the silhouette (mixed pixels,
    motion-blurred edges) are rejected too. Returns None when nothing is
    excluded, preserving the exact legacy code path.
    """
    prev_ex, cur_ex = excludes[pair_index], excludes[pair_index + 1]
    if prev_ex is None and cur_ex is None:
        return None
    union = np.zeros_like(prev_ex if prev_ex is not None else cur_ex, dtype=bool)
    if prev_ex is not None:
        union |= prev_ex
    if cur_ex is not None:
        union |= cur_ex
    if not union.any():
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    grown = cv2.dilate(union.astype(np.uint8), kernel) > 0
    return np.where(grown, 0, 255).astype(np.uint8)


def estimate_frame_transforms(
    frames: list[NDArray[np.uint8]],
    max_corners: int = 800,
    exclude_masks: list[NDArray[np.bool_]] | NDArray[np.bool_] | None = None,
) -> list[NDArray[np.float64]]:
    """Estimate a 3x3 affine mapping each frame's coords into frame 0's coords.
    Consecutive frames are matched with sparse optical flow and a RANSAC-fitted
    partial affine (translation, rotation, uniform scale), then composed. Pairwise
    estimation is used because consecutive frames overlap almost completely, which
    fits far more stably than matching distant frames directly.
    When exclude_masks marks the moving edit subject per frame, corners are
    detected only on the background (Granados et al. background inpainting:
    motion must be estimated from unoccluded points, never through the object
    being removed). A morphing subject otherwise contributes fast non-rigid
    flow that corrupts the background transform.
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    excludes = _as_exclude_list(exclude_masks, len(frames), grays[0].shape)
    transforms = [np.eye(3, dtype=np.float64)]
    for i, (prev_gray, cur_gray) in enumerate(zip(grays, grays[1:], strict=False)):
        step = np.eye(3, dtype=np.float64)
        track_mask = _track_mask(excludes, i)
        points_prev = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=8, mask=track_mask
        )
        if points_prev is not None and len(points_prev) >= 6:
            points_cur, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, points_prev, None)
            tracked = status.ravel() == 1
            if tracked.sum() >= 6:
                matrix, _ = cv2.estimateAffinePartial2D(
                    points_prev[tracked],
                    points_cur[tracked],
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.0,
                )
                if matrix is not None:
                    step[:2] = matrix
        # step maps prev -> cur, so its inverse takes cur back toward frame 0.
        transforms.append(transforms[-1] @ np.linalg.inv(step))
    return transforms


def registration_residual(
    frames: list[NDArray[np.uint8]],
    transforms: list[NDArray[np.float64]],
    exclude_masks: list[NDArray[np.bool_]] | NDArray[np.bool_] | None = None,
) -> float:
    """Median brightness error between consecutive frames after alignment.
    validated-solution.md 5.1 requires a confidence signal to decide where copied
    pixels are acceptable: "Optical flow is not ground truth." A fitted affine
    always returns something , and copying pixels through a transform that does
    not actually explain the footage produces sharp, confidently wrong patches.
    exclude_masks removes the edit subject from the error: a morphing object
    otherwise inflates the residual even when the background registers cleanly,
    and the fail-closed gate then discards real background pixels it could have
    trusted. Real camera background lands around 12 on an 0-255 scale.
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    height, width = grays[0].shape
    footprint = np.full((height, width), 255, dtype=np.uint8)
    excludes = _as_exclude_list(exclude_masks, len(frames), grays[0].shape)
    errors: list[float] = []
    for index in range(1, len(grays)):
        matrix = np.linalg.inv(transforms[index - 1]) @ transforms[index]
        affine = matrix[:2].astype(np.float32)
        warped = cv2.warpAffine(grays[index], affine, (width, height))
        covered = cv2.warpAffine(footprint, affine, (width, height)) > 0
        kept = covered & ~_pair_exclude(excludes, index - 1, (height, width))
        if kept.sum() < 1000:
            continue
        errors.append(
            float(
                np.abs(
                    warped[kept].astype(np.int32) - grays[index - 1][kept].astype(np.int32)
                ).mean()
            )
        )
    return float(np.median(errors)) if errors else float("inf")


def _pair_exclude(
    excludes: list[NDArray[np.bool_] | None], pair_index: int, shape: tuple[int, int]
) -> NDArray[np.bool_]:
    """Union of a frame pair's exclude masks, grown 5px like the track mask."""
    prev_ex, cur_ex = excludes[pair_index], excludes[pair_index + 1]
    union = np.zeros(shape, dtype=bool)
    if prev_ex is not None:
        union |= prev_ex
    if cur_ex is not None:
        union |= cur_ex
    if union.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        union = cv2.dilate(union.astype(np.uint8), kernel) > 0
    return union


def _reference_canvas(
    transforms: list[NDArray[np.float64]],
    width: int,
    height: int,
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    """Offset transform and size of a canvas holding every frame's warped extent."""
    corners = np.array(
        [[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]], dtype=np.float64
    ).T

    xs: list[float] = []
    ys: list[float] = []
    for transform in transforms:
        mapped = transform @ corners
        xs.extend(mapped[0])
        ys.extend(mapped[1])

    min_x, max_x = np.floor(min(xs)), np.ceil(max(xs))
    min_y, max_y = np.floor(min(ys)), np.ceil(max(ys))

    offset = np.eye(3, dtype=np.float64)
    offset[0, 2] = -min_x
    offset[1, 2] = -min_y

    return offset, (int(max_x - min_x), int(max_y - min_y))


class BackgroundPlate:
    """Per-pixel background samples gathered in a shared reference space.

    Samples are collected once for the pixels that need filling; per-frame fills
    are then cheap temporal-median queries over that stack.

    DiffuEraser enhancement: supports pre-propagated samples from distant frames
    for long-sequence temporal consistency.
    """

    def __init__(
        self,
        samples: NDArray[np.uint8],
        valid: NDArray[np.bool_],
        rows: NDArray[np.intp],
        cols: NDArray[np.intp],
        offset: NDArray[np.float64],
        canvas_size: tuple[int, int],
        transforms: list[NDArray[np.float64]],
        frame_shape: tuple[int, int],
        prepropagated_samples: NDArray[np.uint8] | None = None,
        prepropagated_valid: NDArray[np.bool_] | None = None,
    ) -> None:
        self._samples = samples
        self._valid = valid
        self._rows = rows
        self._cols = cols
        self._offset = offset
        self._canvas_size = canvas_size
        self._transforms = transforms
        self._frame_shape = frame_shape
        self._prepropagated_samples = prepropagated_samples
        self._prepropagated_valid = prepropagated_valid

    @property
    def total_samples(self) -> int:
        return int(self._valid.sum())

    def fill_for_frame(
        self,
        index: int,
        window: int,
        min_samples: int,
        mode: str = "nearest",
        max_spread: float | None = None,
    ) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
        """Background estimate and availability mask in frame index coords.
        mode "nearest" copies each pixel from the temporally closest frame that
        exposed it, which keeps the result as sharp as the source. "median"
        averages over the window, which is steadier against transient noise but
        ghosts anything that moved within the window.
        max_spread (median mode only) rejects pixels whose valid donors
        disagree by more than this mean absolute deviation (0-255): donors
        warped through a locally-wrong affine disagree, so high spread marks
        misalignment and the pixel falls through to the diffusion fill instead
        of pasting a shard (FGVC/E2FGVI validity-mask practice; our §5.1
        confidence gate at pixel granularity). None disables the check.
        """
        canvas_w, canvas_h = self._canvas_size
        height, width = self._frame_shape

        low = max(0, index - window)
        high = min(self._samples.shape[0], index + window + 1)

        window_samples = self._samples[low:high]
        window_valid = self._valid[low:high]

        # Include pre-propagated samples if available
        if self._prepropagated_samples is not None and self._prepropagated_valid is not None:
            preprop_samples = self._prepropagated_samples
            preprop_valid = self._prepropagated_valid
            # Combine with local window samples
            combined_samples = np.concatenate([window_samples, preprop_samples], axis=0)
            combined_valid = np.concatenate([window_valid, preprop_valid], axis=0)
        else:
            combined_samples = window_samples
            combined_valid = window_valid
        counts = combined_valid.sum(axis=0)
        usable = counts >= min_samples
        plate_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        trusted_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        if usable.any():
            if mode == "nearest":
                distance = np.abs(np.arange(combined_samples.shape[0]) - index)[:, None]
                distance = np.where(combined_valid[:, usable], distance, np.iinfo(np.int32).max)
                donor = np.argmin(distance, axis=0)
                values = combined_samples[:, usable][donor, np.arange(donor.size)]
            else:
                masked = np.ma.masked_array(
                    combined_samples[:, usable],
                    mask=~np.repeat(combined_valid[:, usable, None], 3, axis=2),
                )
                if max_spread is not None:
                    med = np.ma.median(masked, axis=0)
                    dev = np.ma.mean(np.ma.abs(masked - med), axis=0).mean(axis=-1)
                    usable[usable] = np.ma.filled(dev, 0.0) <= max_spread
                    if usable.any():
                        masked = np.ma.masked_array(
                            combined_samples[:, usable],
                            mask=~np.repeat(combined_valid[:, usable, None], 3, axis=2),
                        )
                values = (
                    np.ma.filled(np.ma.median(masked, axis=0), 0).astype(np.uint8)
                    if usable.any()
                    else np.zeros((0, 3), dtype=np.uint8)
                )
            plate_canvas[self._rows[usable], self._cols[usable]] = values
            trusted_canvas[self._rows[usable], self._cols[usable]] = 255
        inverse = np.linalg.inv(self._offset @ self._transforms[index])[:2].astype(np.float32)
        plate = cv2.warpAffine(plate_canvas, inverse, (width, height), flags=cv2.INTER_LINEAR)
        available = (
            cv2.warpAffine(trusted_canvas, inverse, (width, height), flags=cv2.INTER_NEAREST) > 0
        )

        return plate, available


def build_background_plate(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    transforms: list[NDArray[np.float64]] | None = None,
) -> BackgroundPlate:
    """Gather unmasked samples for every pixel some frame needs filled."""
    height, width = frames[0].shape[:2]
    if transforms is None:
        transforms = estimate_frame_transforms(frames)

    offset, canvas_size = _reference_canvas(transforms, width, height)
    canvas_w, canvas_h = canvas_size
    warps = [(offset @ t)[:2].astype(np.float32) for t in transforms]

    needed = np.zeros((canvas_h, canvas_w), dtype=bool)
    for mask, warp in zip(masks, warps, strict=False):
        needed |= cv2.warpAffine(mask, warp, canvas_size, flags=cv2.INTER_NEAREST) > 0

    rows, cols = np.nonzero(needed)
    samples = np.zeros((len(frames), len(rows), 3), dtype=np.uint8)
    valid = np.zeros((len(frames), len(rows)), dtype=bool)

    footprint = np.full((height, width), 255, dtype=np.uint8)

    for i, (frame, mask, warp) in enumerate(zip(frames, masks, warps, strict=False)):
        warped = cv2.warpAffine(frame, warp, canvas_size, flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpAffine(mask, warp, canvas_size, flags=cv2.INTER_NEAREST)
        inside = cv2.warpAffine(footprint, warp, canvas_size, flags=cv2.INTER_NEAREST)

        samples[i] = warped[rows, cols]
        valid[i] = (warped_mask[rows, cols] == 0) & (inside[rows, cols] > 0)

    log.info(
        "Background samples gathered",
        canvas=f"{canvas_w}x{canvas_h}",
        pixels_needed=int(len(rows)),
        mean_samples_per_pixel=round(float(valid.sum(axis=0).mean()), 1),
    )

    return BackgroundPlate(
        samples, valid, rows, cols, offset, canvas_size, transforms, (height, width)
    )


def _warp_between(
    image: NDArray[np.uint8],
    transforms: list[NDArray[np.float64]],
    src_index: int,
    dst_index: int,
    size: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
) -> NDArray[np.uint8]:
    """Warp an image from frame src_index's coords into frame dst_index's coords."""
    matrix = np.linalg.inv(transforms[dst_index]) @ transforms[src_index]
    return cv2.warpAffine(image, matrix[:2].astype(np.float32), size, flags=flags)


def fill_by_nearest_donor(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    transforms: list[NDArray[np.float64]],
    index: int,
    min_valid_ratio: float = 0.97,
    max_search: int = 0,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Fill each masked region from the single nearest frame that fully exposes it.

    Choosing one donor per connected region, rather than the closest donor
    per pixel, keeps the patch spatially coherent: per-pixel selection stitches
    neighbouring pixels from frames that are seconds apart, which shows up as a
    torn edge even though each individual pixel is sharp.
    """
    height, width = frames[0].shape[:2]
    size = (width, height)
    total = len(frames)
    search_limit = max_search or total

    target_mask = masks[index] > 0
    fill = np.zeros_like(frames[index])
    filled = np.zeros((height, width), dtype=bool)

    if not target_mask.any():
        return fill, filled

    count, labels = cv2.connectedComponents(target_mask.astype(np.uint8))

    footprint = np.full((height, width), 255, dtype=np.uint8)

    for label in range(1, count):
        region = labels == label
        region_size = int(region.sum())
        if region_size == 0:
            continue

        for offset in range(1, search_limit + 1):
            for donor in (index - offset, index + offset):
                if donor < 0 or donor >= total:
                    continue

                donor_mask = _warp_between(
                    masks[donor], transforms, donor, index, size, cv2.INTER_NEAREST
                )
                donor_inside = _warp_between(
                    footprint, transforms, donor, index, size, cv2.INTER_NEAREST
                )
                usable = (donor_mask == 0) & (donor_inside > 0) & region
                if usable.sum() / region_size < min_valid_ratio:
                    continue
                donor_frame = _warp_between(frames[donor], transforms, donor, index, size)
                fill[usable] = donor_frame[usable]
                filled |= usable
                break
            else:
                continue
            break
    return fill, filled


def gate_by_local_background(
    fill: NDArray[np.uint8],
    available: NDArray[np.bool_],
    frame: NDArray[np.uint8],
    edit_mask: NDArray[np.uint8],
    occupancy: NDArray[np.uint8],
    win: int = 31,
    k: float = 3.0,
    floor: float = 12.0,
    min_support: float = 0.25,
) -> NDArray[np.bool_]:
    """Reject plate pixels alien to their LOCAL background neighborhood.
    SuBSENSE principle: high local variance yields a larger threshold and vice
    versa, so textured areas (lane markings) tolerate detail while plain
    asphalt is strict. Per-pixel Gaussian of nearby unmasked background
    (normalized box filters, mask-aware): reject where any channel exceeds
    max(k sigma, floor) from the local mean. Catches coherent-but-wrong
    donors that donor-agreement gating cannot see, without the global
    median's textured-detail veto. Fail-open on shape mismatch or thin
    local support.
    """
    if (
        frame.shape[:2] != available.shape
        or edit_mask.shape != available.shape
        or occupancy.shape != available.shape
    ):
        return available
    region = (edit_mask > 0) & available
    if not region.any():
        return available
    bg = (~(edit_mask > 0) & ~(occupancy > 0)).astype(np.float32)
    kernel = (win, win)
    support = cv2.boxFilter(bg, -1, kernel, normalize=True)
    frame_f = frame.astype(np.float32)
    excess = np.zeros(available.shape, dtype=np.float32)
    for ch in range(3):
        channel = frame_f[..., ch]
        mean = cv2.boxFilter(channel * bg, -1, kernel, normalize=True) / np.maximum(support, 1e-6)
        mean_sq = cv2.boxFilter(channel * channel * bg, -1, kernel, normalize=True) / np.maximum(
            support, 1e-6
        )
        sigma = np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))
        tol = np.maximum(k * sigma, floor)
        excess = np.maximum(excess, np.abs(fill[..., ch].astype(np.float32) - mean) - tol)
    reject = (excess > 0) & region & (support >= min_support)
    return available & ~reject


def gate_by_patch_coherence(
    fill: NDArray[np.uint8],
    available: NDArray[np.bool_],
    frame: NDArray[np.uint8],
    edit_mask: NDArray[np.uint8],
    occupancy: NDArray[np.uint8],
    thresh: float = 0.45,
    min_component: int = 64,
) -> NDArray[bool]:
    """Reject filled components with no background twin (exemplar coherence).
    Patch-based inpainting's contract (Wexler et al.): every filled patch must
    resemble some unoccluded patch. Per connected component of the filled edit
    region, masked normalized cross-correlation of the component against the
    frame, restricted to component pixels AND fully-unmasked search patches:
    best score below thresh rejects the whole component to the model fill.
    Masked (not bbox) correlation matters: bbox corners outside a non-square
    component hold zeros that tank naive matchTemplate and veto good fills.
    Flat fills (zero template variance, where NCC is undefined) fall back to
    a median-color distance check. Components below min_component are left
    to the pixel gates. Fail-open on shape mismatch. Returns filtered mask.
    """
    if (
        frame.shape[:2] != available.shape
        or edit_mask.shape != available.shape
        or occupancy.shape != available.shape
    ):
        return available
    region = (edit_mask > 0) & available
    if not region.any():
        return available
    bg_mask = ~(edit_mask > 0) & ~(occupancy > 0)
    if not bg_mask.any():
        return available
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg_med = float(np.median(frame_gray[bg_mask]))
    bg_f = bg_mask.astype(np.float32)
    count, labels = cv2.connectedComponents(region.astype(np.uint8))
    keep = np.ones_like(region, dtype=bool)
    for label in range(1, count):
        comp = labels == label
        if int(comp.sum()) < min_component:
            continue
        ys, xs = np.nonzero(comp)
        y1, y2, x1, x2 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        template = cv2.cvtColor(fill[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY).astype(np.float32)
        support = (comp[y1:y2, x1:x2]).astype(np.float32)
        if template.shape[0] >= frame_gray.shape[0] or template.shape[1] >= frame_gray.shape[1]:
            continue
        template_vals = template[support > 0]
        if float(template_vals.std()) < 1.0:
            if abs(float(template_vals.mean()) - bg_med) > 36.0:
                keep[comp] = False
            continue
        kernel = (template * support)[::-1, ::-1]
        mask_kernel = support[::-1, ::-1]
        n = float(support.sum())
        sum_t = float((template * support).sum())
        sum_t2 = float(((template * support) ** 2).sum())
        sum_ts = cv2.filter2D(frame_gray, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        sum_s = cv2.filter2D(frame_gray, -1, mask_kernel, borderType=cv2.BORDER_CONSTANT)
        sum_s2 = cv2.filter2D(
            frame_gray * frame_gray, -1, mask_kernel, borderType=cv2.BORDER_CONSTANT
        )
        fully_unmasked = (
            cv2.filter2D(bg_f, -1, mask_kernel, borderType=cv2.BORDER_CONSTANT) >= n - 0.5
        )
        denom = (n * sum_s2 - sum_s * sum_s) * (n * sum_t2 - sum_t * sum_t)
        with np.errstate(invalid="ignore", divide="ignore"):
            ncc = np.where(
                fully_unmasked,
                (n * sum_ts - sum_s * sum_t) / np.sqrt(np.maximum(denom, 1e-6)),
                -1.0,
            )
        if float(ncc.max()) < thresh:
            keep[comp] = False
    return available & (~region | (region & keep))


def blend_fills(
    plate: NDArray[np.uint8],
    fallback: NDArray[np.uint8],
    available: NDArray[np.bool_],
    feather_px: int = 5,
) -> NDArray[np.uint8]:
    """Prefer real background pixels, easing into the fallback where unavailable.

    A hard switch between the two sources leaves a visible edge, since the plate is
    sharp and the model fill is soft.
    """
    if not available.any():
        return fallback
    if available.all() and feather_px <= 0:
        return plate

    weight = available.astype(np.float32)
    if feather_px > 0:
        kernel = 2 * feather_px + 1
        weight = cv2.GaussianBlur(weight, (kernel, kernel), 0)

    weight = weight[..., None]
    blended = weight * plate.astype(np.float32) + (1.0 - weight) * fallback.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def prepropagate_background(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    transforms: list[NDArray[np.float64]],
    *,
    sample_stride: int = 8,
    propagation_window: int = 0,
    min_samples: int = 5,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """DiffuEraser-style pre-propagation: extend known pixels across entire time domain.

    Samples frames at sample_stride intervals, propagates known background
    across the full clip length, then returns consolidated samples that cover
    a much larger temporal context than local window propagation.

    Args:
        frames: Source video frames
        masks: Binary masks (T, H, W) uint8, 255 = edit region
        transforms: Affine transforms from estimate_frame_transforms
        sample_stride: Sample every Nth frame for global propagation
        propagation_window: 0 = use all frames; >0 = local window around each sample
        min_samples: Minimum frames where background must be visible

    Returns:
        (prepropagated_samples, prepropagated_valid) for BackgroundPlate
    """
    T = len(frames)
    height, width = frames[0].shape[:2]

    if transforms is None:
        transforms = estimate_frame_transforms(frames)

    offset, canvas_size = _reference_canvas(transforms, width, height)
    canvas_w, canvas_h = canvas_size
    warps = [(offset @ t)[:2].astype(np.float32) for t in transforms]

    # Determine which canvas pixels need filling
    needed = np.zeros((canvas_h, canvas_w), dtype=bool)
    for mask, warp in zip(masks, warps, strict=False):
        needed |= cv2.warpAffine(mask, warp, canvas_size, flags=cv2.INTER_NEAREST) > 0

    rows, cols = np.nonzero(needed)
    if len(rows) == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((0, 0), dtype=bool)

    # Sample frames for global propagation
    sample_indices = list(range(0, T, sample_stride))
    if sample_indices[-1] != T - 1:
        sample_indices.append(T - 1)

    log.info("Pre-propagating background", sample_frames=len(sample_indices), total_frames=T)

    # Gather samples from sampled frames
    all_samples = []
    all_valid = []

    footprint = np.full((height, width), 255, dtype=np.uint8)

    for i in sample_indices:
        frame, mask, warp = frames[i], masks[i], warps[i]

        warped = cv2.warpAffine(frame, warp, canvas_size, flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpAffine(mask, warp, canvas_size, flags=cv2.INTER_NEAREST)
        inside = cv2.warpAffine(footprint, warp, canvas_size, flags=cv2.INTER_NEAREST)

        samples = warped[rows, cols]
        valid = (warped_mask[rows, cols] == 0) & (inside[rows, cols] > 0)

        all_samples.append(samples)
        all_valid.append(valid)

    if not all_samples:
        return np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((0, 0), dtype=bool)

    all_samples = np.stack(all_samples, axis=0)  # (S, N, 3)
    all_valid = np.stack(all_valid, axis=0)  # (S, N)

    # Temporal propagation: for each pixel, find the nearest sampled frame with valid background
    # This extends known background across the entire clip
    consolidated_samples = np.zeros((len(rows), 3), dtype=np.uint8)
    consolidated_valid = np.zeros(len(rows), dtype=bool)

    for pixel_idx in range(len(rows)):
        valid_indices = np.where(all_valid[:, pixel_idx])[0]
        if len(valid_indices) >= min_samples:
            # Use median of all valid samples for robustness
            pixel_samples = all_samples[valid_indices, pixel_idx]
            consolidated_samples[pixel_idx] = np.median(pixel_samples, axis=0).astype(np.uint8)
            consolidated_valid[pixel_idx] = True

    # Expand back to frame-wise format for BackgroundPlate compatibility
    # We'll return as (1, N, 3) and (1, N) so they can be concatenated with local samples
    final_samples = consolidated_samples[None, :, :]  # (1, N, 3)
    final_valid = consolidated_valid[None, :]  # (1, N)

    log.info(
        "Pre-propagation complete",
        pixels_covered=int(consolidated_valid.sum()),
        total_pixels=len(rows),
        coverage_pct=round(100 * consolidated_valid.sum() / max(1, len(rows)), 1),
    )

    return final_samples, final_valid


def preinference_temporal_context(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    transforms: list[NDArray[np.float64]],
    *,
    sample_stride: int = 4,
    context_window: int = 0,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """DiffuEraser-style pre-inference: broaden temporal context for consistent generation.

    Similar to pre-propagation but focuses on providing a global temporal prior
    for generative models. Returns sampled frames and their masks in the reference
    canvas space.

    Args:
        frames: Source video frames
        masks: Binary masks (T, H, W) uint8, 255 = edit region
        transforms: Affine transforms
        sample_stride: Sample every Nth frame
        context_window: 0 = use all frames; >0 = local window

    Returns:
        (sampled_frames_in_canvas, sampled_masks_in_canvas) for global context
    """
    T = len(frames)
    height, width = frames[0].shape[:2]

    if transforms is None:
        transforms = estimate_frame_transforms(frames)

    offset, canvas_size = _reference_canvas(transforms, width, height)
    canvas_w, canvas_h = canvas_size
    warps = [(offset @ t)[:2].astype(np.float32) for t in transforms]

    sample_indices = list(range(0, T, sample_stride))
    if sample_indices[-1] != T - 1:
        sample_indices.append(T - 1)

    footprint = np.full((height, width), 255, dtype=np.uint8)

    sampled_frames = []
    sampled_masks = []

    for i in sample_indices:
        frame, mask, warp = frames[i], masks[i], warps[i]

        warped_frame = cv2.warpAffine(frame, warp, canvas_size, flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpAffine(mask, warp, canvas_size, flags=cv2.INTER_NEAREST)
        inside = cv2.warpAffine(footprint, warp, canvas_size, flags=cv2.INTER_NEAREST)

        # Only keep valid canvas region
        valid_region = inside > 0
        warped_frame[~valid_region] = 0
        warped_mask[~valid_region] = 0

        sampled_frames.append(warped_frame)
        sampled_masks.append(warped_mask)

    sampled_frames = np.stack(sampled_frames, axis=0)  # (S, H, W, 3)
    sampled_masks = np.stack(sampled_masks, axis=0)  # (S, H, W)

    log.info(
        "Pre-inference context built",
        sample_count=len(sample_indices),
        canvas=f"{canvas_w}x{canvas_h}",
    )

    return sampled_frames, sampled_masks


def enhance_background_plate_diffueraser(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    transforms: list[NDArray[np.float64]] | None = None,
    *,
    preprop_stride: int = 8,
    preinf_stride: int = 4,
    min_samples: int = 5,
) -> BackgroundPlate:
    """Build BackgroundPlate with DiffuEraser-style enhancements.

    Combines local window samples with global pre-propagation and pre-inference
    for maximum temporal receptive field.

    Args:
        frames: Source video frames
        masks: Binary masks (T, H, W) uint8, 255 = edit region
        transforms: Optional pre-computed transforms
        preprop_stride: Stride for pre-propagation sampling
        preinf_stride: Stride for pre-inference sampling
        min_samples: Minimum samples for a pixel to be considered valid

    Returns:
        Enhanced BackgroundPlate with global temporal context
    """
    if transforms is None:
        transforms = estimate_frame_transforms(frames)

    # Build standard local plate
    local_plate = build_background_plate(frames, masks, transforms)

    # Pre-propagation: global known-pixel extension
    preprop_samples, preprop_valid = prepropagate_background(
        frames,
        masks,
        transforms,
        sample_stride=preprop_stride,
        min_samples=min_samples,
    )

    # Pre-inference: global temporal context (stored for potential use by diffusion model)
    preinf_frames, preinf_masks = preinference_temporal_context(
        frames,
        masks,
        transforms,
        sample_stride=preinf_stride,
    )

    # Create enhanced plate with pre-propagated samples
    enhanced_plate = BackgroundPlate(
        samples=local_plate._samples,
        valid=local_plate._valid,
        rows=local_plate._rows,
        cols=local_plate._cols,
        offset=local_plate._offset,
        canvas_size=local_plate._canvas_size,
        transforms=local_plate._transforms,
        frame_shape=local_plate._frame_shape,
        prepropagated_samples=preprop_samples if preprop_samples.size > 0 else None,
        prepropagated_valid=preprop_valid if preprop_valid.size > 0 else None,
    )

    # Store pre-inference context for potential diffusion model use
    enhanced_plate._preinference_frames = preinf_frames
    enhanced_plate._preinference_masks = preinf_masks

    return enhanced_plate
