"""Long video support with overlapping windows and ID resampling (VideoPainter-style).

VideoPainter (SIGGRAPH 2025) handles arbitrary-length video by:
1. Splitting into overlapping windows
2. Using middle frame as identity reference
3. Blending overlapping regions
4. ID resampling adapters for cross-window identity consistency
"""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class WindowConfig:
    """Configuration for windowed video processing."""

    window_size: int = 24  # Model's native context window
    overlap: int = 8  # Overlap between windows
    min_window: int = 8  # Minimum window size
    id_reference_stride: int = 12  # Use middle frame as ID reference


def split_into_windows(
    length: int,
    config: WindowConfig,
) -> list[tuple[int, int, int, int]]:
    """Split video into overlapping windows.

    Returns list of (window_start, window_end, valid_start, valid_end)
    where valid region excludes overlap for blending.
    """
    if length <= config.window_size:
        return [(0, length, 0, length)]

    windows = []
    stride = config.window_size - config.overlap

    for start in range(0, length, stride):
        end = min(start + config.window_size, length)
        if end - start < config.min_window and windows:
            # Merge with previous window if too small
            prev = windows[-1]
            windows[-1] = (prev[0], end, prev[2], end)
            break

        # Valid region: exclude overlap from blending
        valid_start = start + config.overlap // 2 if start > 0 else start
        valid_end = end - config.overlap // 2 if end < length else end

        windows.append((start, end, valid_start, valid_end))

    return windows


def blend_windows(
    window_results: list[tuple[list[NDArray[np.uint8]], tuple[int, int, int, int]]],
    total_length: int,
    blend_width: int = 4,
) -> list[NDArray[np.uint8]]:
    """Blend overlapping window results with cross-fade.

    Args:
        window_results: List of (frames, (start, end, valid_start, valid_end))
        total_length: Total number of frames
        blend_width: Width of cross-fade region in frames

    Returns:
        Blended frame sequence
    """
    if len(window_results) == 1:
        return window_results[0][0]

    h, w = window_results[0][0][0].shape[:2]
    blended = [None] * total_length
    weight_sum = np.zeros(total_length, dtype=np.float32)

    for frames, (start, end, valid_start, valid_end) in window_results:
        for i, frame in enumerate(frames):
            global_idx = start + i
            if global_idx >= total_length:
                continue

            # Compute blend weight
            if global_idx < valid_start:
                # Fade in
                if blend_width > 0 and valid_start > start:
                    weight = (global_idx - start) / (valid_start - start)
                else:
                    weight = 1.0
            elif global_idx > valid_end:
                # Fade out
                if blend_width > 0 and end > valid_end:
                    weight = (end - global_idx) / (end - valid_end)
                else:
                    weight = 1.0
            else:
                weight = 1.0

            if blended[global_idx] is None:
                blended[global_idx] = frame.astype(np.float32) * weight
                weight_sum[global_idx] = weight
            else:
                blended[global_idx] += frame.astype(np.float32) * weight
                weight_sum[global_idx] += weight

    # Normalize
    for i in range(total_length):
        if weight_sum[i] > 0:
            blended[i] = np.clip(blended[i] / weight_sum[i], 0, 255).astype(np.uint8)
        else:
            # Fallback
            blended[i] = np.zeros((h, w, 3), dtype=np.uint8)

    return blended


def compute_id_reference_frame(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    window_start: int,
    window_end: int,
) -> tuple[int, NDArray[np.uint8]]:
    """Find the best identity reference frame in a window (VideoPainter-style).

    Uses the middle frame of the window where the object is most visible
    and least occluded as the identity anchor.
    """
    best_score = -1
    best_idx = window_start
    best_frame = frames[window_start]

    for i in range(window_start, window_end):
        mask = masks[i] > 0
        if not mask.any():
            continue

        # Score: visible area sharpness
        area = mask.sum()
        gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray[mask], cv2.CV_64F).var() if mask.any() else 0

        score = area * (1.0 + sharpness / 1000.0)

        if score > best_score:
            best_score = score
            best_idx = i
            best_frame = frames[i]

    return best_idx, best_frame


def apply_id_resampling(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    reference_frame: NDArray[np.uint8],
    reference_mask: NDArray[np.bool_],
    reference_idx: int,
    current_idx: int,
    *,
    strength: float = 0.3,
) -> list[NDArray[np.uint8]]:
    """Apply ID resampling to maintain identity consistency across windows (VideoPainter).

    Warps the reference frame's object appearance to current frame's pose
    using optical flow, then blends with generated result.

    This is a lightweight approximation of VideoPainter's ID-Resample Adapter.
    """
    import cv2

    from preserve.metrics import _flow_pair

    if current_idx == reference_idx:
        return list(frames)
    # Explicit list copy plus copy-on-write: callers may hand us tuples or
    # reuse the arrays elsewhere, and in-place blending would alias them.
    result = list(frames)
    h, w = frames[0].shape[:2]
    # Estimate flow from reference to current
    flow = _flow_pair(
        cv2.cvtColor(reference_frame, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(frames[current_idx], cv2.COLOR_RGB2GRAY),
    )
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    warped_ref = cv2.remap(
        reference_frame,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # Blend warped reference with generated frame in mask region
    mask = masks[current_idx] > 0
    if mask.any():
        fresh = result[current_idx].copy()
        fresh[mask] = (
            (1.0 - strength) * frames[current_idx][mask].astype(np.float32)
            + strength * warped_ref[mask].astype(np.float32)
        ).astype(np.uint8)
        result[current_idx] = fresh
    return result


def process_long_video(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    process_fn: Callable[[list[NDArray[np.uint8]], NDArray[np.uint8]], list[NDArray[np.uint8]]],
    config: WindowConfig | None = None,
    id_resample_strength: float = 0.3,
) -> list[NDArray[np.uint8]]:
    """Process arbitrarily long video using overlapping windows with ID resampling.

    Args:
        frames: Input video frames
        masks: Binary masks for each frame
        process_fn: Function that processes a window (frames, masks) -> output_frames
        config: Window configuration
        id_resample_strength: Strength of ID resampling (0 = off, 1 = full reference)

    Returns:
        Processed frames for entire video
    """
    if config is None:
        config = WindowConfig()
    if len(frames) <= config.window_size:
        return process_fn(frames, masks)
    windows = split_into_windows(len(frames), config)
    window_results = []

    # First pass: process each window
    for start, end, valid_start, valid_end in windows:
        window_frames = frames[start:end]
        window_masks = masks[start:end]

        # Find ID reference frame in this window
        ref_idx, ref_frame = compute_id_reference_frame(frames, masks, start, end)
        ref_mask = masks[ref_idx] > 0

        # Process window
        output = process_fn(window_frames, window_masks)

        # Apply ID resampling to maintain identity
        if id_resample_strength > 0:
            for i in range(len(output)):
                global_idx = start + i
                if global_idx < len(frames):
                    output = apply_id_resampling(
                        output,
                        window_masks,
                        ref_frame,
                        ref_mask,
                        ref_idx - start,
                        i,
                        strength=id_resample_strength,
                    )

        window_results.append((output, (start, end, valid_start, valid_end)))

    # Blend overlapping windows
    return blend_windows(window_results, len(frames))


def videosplit_scene_boundaries(
    frames: list[NDArray[np.uint8]],
    threshold: float = 30.0,
) -> list[int]:
    """Detect scene boundaries using color histogram difference.

    Returns frame indices where scene cuts occur. Processing should not
    cross scene boundaries (VideoPainter uses PySceneDetect).
    """
    if len(frames) < 2:
        return [0, len(frames)]

    boundaries = [0]
    for i in range(1, len(frames)):
        hist1 = cv2.calcHist([frames[i - 1]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frames[i]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        if diff > threshold / 100.0:
            boundaries.append(i)

    boundaries.append(len(frames))
    return boundaries
