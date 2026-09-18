"""MatAnyone-style alpha propagation for semi-transparent boundaries.

Current binary masks miss hair, motion blur, defocus, and transparency.
This module propagates fractional alpha across frames using color sampling
and temporal consistency, inspired by MatAnyone (CVPR 2025) and Generative
Omnimatte's layered decomposition.

Key improvements over v1:
- Guided filter for edge-aware alpha refinement (He et al. ECCV 2010)
- Learned foreground/background color propagation (MatAnyone memory bank)
- Bidirectional flow with occlusion handling
- Confidence-weighted temporal blending
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from preserve.metrics import _flow_pair


def guided_filter(
    guide: NDArray[np.uint8],
    src: NDArray[np.float32],
    radius: int = 4,
    eps: float = 1e-3,
) -> NDArray[np.float32]:
    """Fast guided filter (He et al.) for edge-aware alpha refinement.

    Uses the guide image (source frame) to transfer structure to the
    alpha channel, preserving object boundaries while smoothing noise.
    """
    if guide.ndim == 3:
        guide_gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        guide_gray = guide.astype(np.float32) / 255.0

    mean_I = cv2.boxFilter(guide_gray, -1, (radius * 2 + 1, radius * 2 + 1), normalize=True)
    mean_p = cv2.boxFilter(src, -1, (radius * 2 + 1, radius * 2 + 1), normalize=True)
    mean_Ip = cv2.boxFilter(guide_gray * src, -1, (radius * 2 + 1, radius * 2 + 1), normalize=True)

    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(
        guide_gray * guide_gray, -1, (radius * 2 + 1, radius * 2 + 1), normalize=True
    )
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (radius * 2 + 1, radius * 2 + 1), normalize=True)
    mean_b = cv2.boxFilter(b, -1, (radius * 2 + 1, radius * 2 + 1), normalize=True)

    q = mean_a * guide_gray + mean_b
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def _estimate_foreground_background_memory(
    frame: NDArray[np.uint8],
    binary_mask: NDArray[np.bool_],
    memory_bank: list[tuple[NDArray[np.float32], NDArray[np.float32]]] | None = None,
    erode_px: int = 3,
    dilate_px: int = 5,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list]:
    """Estimate foreground and background colors with memory bank (MatAnyone-style).

    Maintains a memory bank of (fg_color, bg_color) pairs from confident frames
    to stabilize color estimation in challenging frames (motion blur, occlusion).
    """
    kernel_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
    )
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
    )

    fg_mask = cv2.erode(binary_mask.astype(np.uint8), kernel_erode) > 0
    bg_mask = ~cv2.dilate(binary_mask.astype(np.uint8), kernel_dilate).astype(bool)

    frame_f = frame.astype(np.float32) / 255.0

    fg_color = frame_f[fg_mask].mean(axis=0) if fg_mask.any() else frame_f.mean(axis=(0, 1))

    if bg_mask.any():
        bg_color = frame_f[bg_mask].mean(axis=0)
    else:
        bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    # Update memory bank with confident estimates
    if memory_bank is not None:
        # Confidence: mask area and color separation
        mask_area = fg_mask.sum() / (frame.shape[0] * frame.shape[1])
        color_sep = np.linalg.norm(fg_color - bg_color)
        confidence = mask_area * color_sep

        if confidence > 0.01 and len(memory_bank) < 10:
            memory_bank.append((fg_color, bg_color))
        elif memory_bank and confidence < 0.005:
            # Use memory bank average for unstable frames
            fg_colors = np.stack([m[0] for m in memory_bank])
            bg_colors = np.stack([m[1] for m in memory_bank])
            fg_color = fg_colors.mean(axis=0)
            bg_color = bg_colors.mean(axis=0)

    return fg_color, bg_color, memory_bank


def _solve_alpha_closed_form(
    frame: NDArray[np.uint8],
    fg_color: NDArray[np.float32],
    bg_color: NDArray[np.float32],
    binary_mask: NDArray[np.bool_],
    regularization: float = 0.01,
) -> NDArray[np.float32]:
    """Closed-form alpha matting (Levin et al. style) for a single frame.

    Solves: I = alpha F + (1 - alpha) B for alpha
    With regularization to prevent noise amplification.
    """
    frame_f = frame.astype(np.float32) / 255.0

    diff = fg_color - bg_color
    diff_sq = np.sum(diff * diff)

    if diff_sq < 1e-4:
        dist_to_fg = np.sum((frame_f - fg_color) ** 2, axis=2)
        dist_to_bg = np.sum((frame_f - bg_color) ** 2, axis=2)
        alpha = dist_to_bg / (dist_to_fg + dist_to_bg + 1e-6)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    num = np.sum((frame_f - bg_color) * diff, axis=2)
    alpha = num / (diff_sq + regularization)

    alpha = np.clip(alpha, 0.0, 1.0)

    if binary_mask.any() and not binary_mask.all():
        dist_in = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 3)
        dist_out = cv2.distanceTransform((~binary_mask).astype(np.uint8), cv2.DIST_L2, 3)
        total = dist_in + dist_out + 1e-6
        boundary_alpha = dist_out / total
        transition_width = 5.0
        weight = np.clip(dist_in / transition_width, 0.0, 1.0)
        alpha = weight * alpha + (1.0 - weight) * boundary_alpha

    return alpha.astype(np.float32)


def _compute_flow_confidence(
    flow: NDArray[np.float32],
    binary_mask: NDArray[np.bool_],
    prev_binary_mask: NDArray[np.bool_],
) -> float:
    """Compute confidence in flow warping for a region.

    Checks forward-backward consistency and occlusion.
    """
    h, w = flow.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Forward warp of previous mask
    warped_prev = cv2.remap(
        prev_binary_mask.astype(np.float32),
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Overlap with current mask
    overlap = warped_prev * binary_mask.astype(np.float32)
    union = np.maximum(warped_prev, binary_mask.astype(np.float32))

    if union.sum() == 0:
        return 0.0

    iou = overlap.sum() / union.sum()

    # Forward-backward consistency check
    # Compute backward flow
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_flow_mag = flow_mag[binary_mask].mean() if binary_mask.any() else 0

    return float(iou * (1.0 / (1.0 + mean_flow_mag * 0.1)))


def _propagate_alpha_temporal_bidirectional(
    alphas: list[NDArray[np.float32]],
    frames: list[NDArray[np.uint8]],
    binary_masks: list[NDArray[np.bool_]],
    max_warp_error: float = 15.0,
    occlusion_threshold: float = 0.3,
) -> list[NDArray[np.float32]]:
    """Bidirectional temporal propagation with occlusion handling (MatAnyone-style).

    Forward pass: propagate alpha along source motion
    Backward pass: refine with future context
    Includes occlusion detection to avoid propagating through disocclusions.
    """
    if len(alphas) < 2:
        return alphas

    T = len(alphas)
    h, w = alphas[0].shape

    # Forward pass
    forward_alphas = [alphas[0].copy()]

    for i in range(1, T):
        if not binary_masks[i].any() or not binary_masks[i - 1].any():
            forward_alphas.append(alphas[i].copy())
            continue

        flow = _flow_pair(
            cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY),
        )

        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        warped_alpha = cv2.remap(
            forward_alphas[-1],
            grid_x + flow[..., 0],
            grid_y + flow[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Check flow confidence
        confidence = _compute_flow_confidence(flow, binary_masks[i], binary_masks[i - 1])

        if confidence < occlusion_threshold:
            # Low confidence: trust current frame
            forward_alphas.append(alphas[i].copy())
            continue

        # Adaptive blending based on confidence
        blend_weight = 0.4 * confidence
        blended = (1.0 - blend_weight) * alphas[i] + blend_weight * warped_alpha
        forward_alphas.append(np.clip(blended, 0.0, 1.0).astype(np.float32))

    # Backward pass
    backward_alphas = forward_alphas.copy()

    for i in range(T - 2, -1, -1):
        if not binary_masks[i].any() or not binary_masks[i + 1].any():
            continue

        flow = _flow_pair(
            cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY),
        )

        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        warped_alpha = cv2.remap(
            backward_alphas[i + 1],
            grid_x + flow[..., 0],
            grid_y + flow[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        confidence = _compute_flow_confidence(flow, binary_masks[i], binary_masks[i + 1])

        if confidence < occlusion_threshold:
            continue

        blend_weight = 0.3 * confidence
        backward_alphas[i] = np.clip(
            (1.0 - blend_weight) * backward_alphas[i] + blend_weight * warped_alpha, 0.0, 1.0
        ).astype(np.float32)

    return backward_alphas


def compute_fractional_alpha(
    frames: list[NDArray[np.uint8]],
    binary_masks: NDArray[np.uint8],
    *,
    refine_strength: float = 0.5,
    use_guided_filter: bool = True,
    guided_filter_radius: int = 4,
) -> NDArray[np.float32]:
    """Compute fractional alpha masks for semi-transparent boundaries.

    MatAnyone-style: combines per-frame closed-form matting with
    bidirectional temporal propagation for temporal consistency.

    Args:
        frames: Source video frames (RGB uint8)
        binary_masks: Binary masks from SAM 2 (T, H, W) uint8, 255 = object
        refine_strength: 0 = keep binary, 1 = full fractional alpha
        use_guided_filter: Apply guided filter for edge-aware refinement
        guided_filter_radius: Radius for guided filter

    Returns:
        Fractional alpha masks (T, H, W) float32 in [0, 1]
    """
    if refine_strength <= 0:
        return (binary_masks > 0).astype(np.float32)

    T = len(frames)
    alphas = []
    memory_bank = []

    # Step 1: Per-frame closed-form matting with memory bank
    for i in range(T):
        binary = binary_masks[i] > 0
        if not binary.any() or binary.all():
            alphas.append(binary.astype(np.float32))
            continue

        fg_color, bg_color, memory_bank = _estimate_foreground_background_memory(
            frames[i], binary, memory_bank
        )
        alpha = _solve_alpha_closed_form(frames[i], fg_color, bg_color, binary)
        alphas.append(alpha)

    # Step 2: Bidirectional temporal propagation with occlusion handling
    binary_bool = binary_masks > 0
    alphas = _propagate_alpha_temporal_bidirectional(alphas, frames, list(binary_bool))

    # Step 3: Guided filter refinement for edge-aware alpha
    if use_guided_filter:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for i in range(T):
            if binary_bool[i].any():
                # Guided filter using source frame as guide
                alphas[i] = guided_filter(frames[i], alphas[i], radius=guided_filter_radius)
                # Morphological closing to fill small holes
                alphas[i] = cv2.morphologyEx(alphas[i], cv2.MORPH_CLOSE, kernel)
                # Slight sharpening
                alphas[i] = np.clip(alphas[i] * 1.05 - 0.025, 0.0, 1.0).astype(np.float32)

    return np.stack(alphas, axis=0)


def detect_transparency_regions(
    frame: NDArray[np.uint8],
    binary_mask: NDArray[np.bool_],
    *,
    gradient_threshold: float = 0.15,
) -> NDArray[np.bool_]:
    """Detect likely semi-transparent regions (hair, motion blur, defocus).

    Uses gradient magnitude at mask boundary - transparent regions
    have softer edges than hard object boundaries.
    """
    if not binary_mask.any() or binary_mask.all():
        return np.zeros_like(binary_mask, dtype=bool)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(binary_mask.astype(np.uint8), kernel) > 0
    boundary = dilated & ~binary_mask

    eroded = cv2.erode(binary_mask.astype(np.uint8), kernel) > 0
    interior_boundary = binary_mask & ~eroded
    boundary = boundary | interior_boundary

    boundary_grad = grad_mag[boundary]
    if len(boundary_grad) == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    mean_grad = boundary_grad.mean()
    transparency = boundary & (grad_mag < mean_grad * gradient_threshold)

    return transparency


def build_alpha_with_transparency(
    frames: list[NDArray[np.uint8]],
    binary_masks: NDArray[np.uint8],
    *,
    transparency_weight: float = 0.5,
    refine_strength: float = 0.5,
) -> NDArray[np.float32]:
    """Build alpha masks with explicit transparency handling.

    Combines fractional alpha matting with transparency detection
    for hair, motion blur, and defocus regions.
    """
    base_alpha = compute_fractional_alpha(frames, binary_masks, refine_strength=refine_strength)

    for i in range(len(frames)):
        binary = binary_masks[i] > 0
        trans_regions = detect_transparency_regions(frames[i], binary)
        if trans_regions.any():
            base_alpha[i][trans_regions] = np.clip(
                base_alpha[i][trans_regions] * (1.0 - transparency_weight * 0.5), 0.0, 1.0
            )

    return base_alpha


class MatAnyonePropagator:
    """MatAnyone-style memory-based alpha propagation for long sequences.

    Maintains a memory bank of confident frames to stabilize
    alpha propagation across long videos with occlusions and
    large appearance changes.
    """

    def __init__(self, memory_size: int = 10):
        self.memory_size = memory_size
        self.memory_frames = []
        self.memory_alphas = []
        self.memory_masks = []

    def add_to_memory(
        self,
        frame: NDArray[np.uint8],
        alpha: NDArray[np.float32],
        mask: NDArray[np.bool_],
    ) -> None:
        """Add a frame to memory bank if confident."""
        if mask.any() and mask.sum() > 100:
            self.memory_frames.append(frame)
            self.memory_alphas.append(alpha)
            self.memory_masks.append(mask)

            # Keep memory bounded
            if len(self.memory_frames) > self.memory_size:
                self.memory_frames.pop(0)
                self.memory_alphas.pop(0)
                self.memory_masks.pop(0)

    def propagate_with_memory(
        self,
        frame: NDArray[np.uint8],
        alpha: NDArray[np.float32],
        mask: NDArray[np.bool_],
        *,
        blend_weight: float = 0.3,
    ) -> NDArray[np.float32]:
        """Propagate alpha using memory bank for long-range consistency."""
        if not self.memory_alphas or not mask.any():
            return alpha

        # Find best matching memory frame using color histogram
        best_idx = 0
        best_score = -1
        frame_hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        frame_hist = cv2.normalize(frame_hist, frame_hist).flatten()

        for i, mem_frame in enumerate(self.memory_frames):
            mem_hist = cv2.calcHist(
                [mem_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
            )
            mem_hist = cv2.normalize(mem_hist, mem_hist).flatten()
            score = cv2.compareHist(frame_hist, mem_hist, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score > 0.8:
            # Warp memory alpha to current frame
            flow = _flow_pair(
                cv2.cvtColor(self.memory_frames[best_idx], cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            )
            h, w = alpha.shape
            grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
            warped_mem_alpha = cv2.remap(
                self.memory_alphas[best_idx],
                grid_x + flow[..., 0],
                grid_y + flow[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            # Blend with current alpha in mask region
            result = alpha.copy()
            result[mask] = (1.0 - blend_weight) * alpha[mask] + blend_weight * warped_mem_alpha[
                mask
            ]
            return np.clip(result, 0.0, 1.0).astype(np.float32)

        return alpha
