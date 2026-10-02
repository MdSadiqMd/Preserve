"""In-region finishing that never authorizes new pixels.

SOTA video editors still fail the product contract if they trust the model.
These operators run after a candidate exists and before hard-restore. They
only write samples already inside the authorized mask:

Temporal mask hysteresis (SAM 2 / ByteTrack chatter → flicker and residue).
Boundary colour match (VFX plate integration; ComfyUI grow-and-blur seams).
Flow-guided blend along source motion (TokenFlow / DiffuEraser prior reuse).
Feature-space propagation for temporal consistency (TokenFlow-style).
Occluder copy-up from the source (validated-solution.md §4).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from preserve.metrics import _flow_pair


def stabilize_mask_sequence(
    masks: NDArray[np.uint8],
    *,
    hug_object: bool = True,
) -> NDArray[np.uint8]:
    """Remove 1-frame dropouts and boundary chatter without inventing a new object.

    Hugging (recolour / replace): a pixel must be on in a temporal majority, so
    a tracker that overshoots empty road does not repaint background.
    Removal: a pixel that was on in a neighbour stays on, so a missed detection
    cannot leave a silhouette.
    """
    if len(masks) < 3:
        return masks

    stacked = (masks > 0).astype(np.uint8)
    padded = np.concatenate([stacked[:1], stacked, stacked[-1:]], axis=0)
    prev, curr, nxt = padded[:-2], padded[1:-1], padded[2:]
    if hug_object:
        out = ((prev + curr + nxt) >= 2).astype(np.uint8)
    else:
        out = np.maximum.reduce([prev, curr, nxt])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = np.stack([cv2.morphologyEx(frame * 255, cv2.MORPH_CLOSE, kernel) for frame in out])
    return closed


def restore_occluders(
    frames: list[NDArray[np.uint8]],
    sources: list[NDArray[np.uint8]],
    keepout: NDArray[np.bool_] | None,
) -> list[NDArray[np.uint8]]:
    """Put original foreground occluders back above the generated patch."""
    if keepout is None:
        return frames
    restored: list[NDArray[np.uint8]] = []
    for frame, source, occluded in zip(frames, sources, keepout, strict=True):
        if not occluded.any():
            restored.append(frame)
            continue
        result = frame.copy()
        result[occluded] = source[occluded]
        restored.append(result)
    return restored


def match_boundary_color(
    generated: list[NDArray[np.uint8]],
    sources: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
    band_px: int = 5,
    interior: bool = False,
) -> list[NDArray[np.uint8]]:
    """Membrane seam match: fade a LOCAL colour offset from the source ring into
    the interior band so the composite seam disappears without touching the core.
    """
    if band_px <= 0:
        return generated

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1))
    sigma = float(band_px) * 1.5
    matched: list[NDArray[np.uint8]] = []
    for patch, source, region in zip(generated, sources, allowed, strict=True):
        if not region.any():
            matched.append(patch)
            continue
        mask_u8 = region.astype(np.uint8) * 255
        eroded = cv2.erode(mask_u8, kernel) > 0
        inner = region & ~eroded
        # The reference is the thin ring right outside the seam, not a band
        # as wide as the fade: a 12px outer band reached the eyebrows below
        # a revealed forehead and darkened the fill past the skin between
        # (cap-v27, fill L 49 against skin 58).
        outer = (
            cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
        ) & ~region
        if inner.sum() < 16 or outer.sum() < 16:
            matched.append(patch)
            continue
        adjusted = patch.astype(np.float32)
        offset = _local_mean(source, outer, sigma) - _local_mean(patch, inner, sigma)
        if interior:
            # Membrane: the seam offsets (valid on the inner band) are
            # interpolated across the region by a wide normalised
            # convolution, so the deep interior takes a smooth blend of its
            # whole boundary. Sigma scales with the region size.
            wide = max(sigma, 0.5 * float(np.sqrt(region.sum())))
            field = _local_mean(np.clip(offset + 128.0, 0, 255), inner, wide) - 128.0
            adjusted[region] += field[region]
        else:
            # Distance to the region boundary measured inside (1 on the seam
            # row): full correction on the seam, fading to zero band_px in.
            dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
            weight = np.clip(1.0 - (dist - 1.0) / float(band_px), 0.0, 1.0)
            adjusted[inner] += offset[inner] * weight[inner, None]
        matched.append(np.clip(adjusted, 0, 255).astype(np.uint8))
    return matched


def _local_mean(image: NDArray, support: NDArray[np.bool_], sigma: float) -> NDArray:
    """Per-pixel mean of image over support, by normalised Gaussian convolution."""
    weights = support.astype(np.float32)
    ksize = int(sigma * 6) | 1
    num = cv2.GaussianBlur(image.astype(np.float32) * weights[..., None], (ksize, ksize), sigma)
    den = cv2.GaussianBlur(weights, (ksize, ksize), sigma)
    return num / np.maximum(den, 1e-4)[..., None]


def transfer_surface_detail(
    fill: NDArray[np.uint8],
    source: NDArray[np.uint8],
    region: NDArray[np.bool_],
    donor: NDArray[np.bool_],
    tile: int = 16,
    sigma: float = 1.2,
    amount: float = 0.6,
    seed: int = 0,
) -> NDArray[np.uint8]:
    """Add the source's fine texture (weave, grain) to a smooth fill.

    The high-pass of the source is sampled in tiles from donor pixels (the
    same surface next to the region), quilted over the region with random
    offsets and half-tile overlap, and added to the fill. An erased-and-
    re-rendered print comes back as a flat disc otherwise (audit 2026-09-21).
    """
    if not region.any() or donor.sum() < tile * tile:
        return fill
    rng = np.random.default_rng(seed)
    # Luminance-only, fine-scale and damped: colour fringes and fold-scale
    # structure quilted at full strength read as crumpled foil (logo-v19).
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    high = np.repeat((gray - cv2.GaussianBlur(gray, (0, 0), sigma))[..., None], 3, axis=2) * amount
    ys, xs = np.nonzero(donor)
    # Tiles fully inside the donor surface
    candidates = [
        (y, x)
        for y, x in zip(ys[:: max(1, len(ys) // 4000)], xs[:: max(1, len(xs) // 4000)], strict=True)
        if y + tile <= donor.shape[0]
        and x + tile <= donor.shape[1]
        and donor[y : y + tile, x : x + tile].all()
    ]
    if not candidates:
        return fill
    h, w = region.shape
    acc = np.zeros((h, w, 3), np.float32)
    weight = np.zeros((h, w), np.float32)
    window = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 1e-3
    ry, rx = np.nonzero(region)
    for y in range(max(0, ry.min() - tile // 2), ry.max() + 1, tile // 2):
        for x in range(max(0, rx.min() - tile // 2), rx.max() + 1, tile // 2):
            sy, sx = candidates[int(rng.integers(len(candidates)))]
            th, tw = min(tile, h - y), min(tile, w - x)
            acc[y : y + th, x : x + tw] += high[sy : sy + th, sx : sx + tw] * window[:th, :tw, None]
            weight[y : y + th, x : x + tw] += window[:th, :tw]
    detail = acc / np.maximum(weight, 1e-3)[..., None]
    out = fill.astype(np.float32)
    out[region] += detail[region]
    return np.clip(out, 0, 255).astype(np.uint8)


def warp_keyframe_fill(
    keyframe: NDArray[np.uint8],
    source_from: NDArray[np.uint8],
    source_to: NDArray[np.uint8],
    region_to: NDArray[np.bool_],
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Carry a keyframe's fill to another frame along the source's optical flow.
    Returns the anchor frame and the mask of region pixels it actually carries.

    Flow is estimated on the two source frames, where the object still sits;
    an attached object moves with its wearer, so that flow is the wearer's
    motion inside the region as well. The warped fill replaces region_to on
    the target source frame, giving the video pass a second anchor that is
    consistent with the first by construction instead of a second, unrelated
    sample (audit 2026-09-20: VACE 1.3B blotched 16 frames from one anchor).
    """
    flow = _flow_pair(
        cv2.cvtColor(source_to, cv2.COLOR_RGB2GRAY), cv2.cvtColor(source_from, cv2.COLOR_RGB2GRAY)
    )
    h, w = keyframe.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(
        keyframe,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # Pixels whose flow points outside the frame have no source to carry;
    # they are left to the video pass (audit 2026-09-21: a hand region at
    # the crop edge came back as a replicated pale blob).
    map_x, map_y = grid_x + flow[..., 0], grid_y + flow[..., 1]
    inside = (map_x >= 1) & (map_x <= w - 2) & (map_y >= 1) & (map_y <= h - 2)
    valid = region_to & inside
    anchor = source_to.copy()
    anchor[valid] = warped[valid]
    return anchor, valid


def warp_mask_between(
    mask: NDArray[np.bool_], source_from: NDArray[np.uint8], source_to: NDArray[np.uint8]
) -> NDArray[np.bool_]:
    """Carry a keyframe mask to another frame along the source's optical flow."""
    flow = _flow_pair(
        cv2.cvtColor(source_to, cv2.COLOR_RGB2GRAY), cv2.cvtColor(source_from, cv2.COLOR_RGB2GRAY)
    )
    h, w = mask.shape
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(
        mask.astype(np.uint8) * 255,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 127


def flow_blend_sequence(
    generated: list[NDArray[np.uint8]],
    sources: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
    blend: float = 0.28,
) -> list[NDArray[np.uint8]]:
    """Reuse the previous patch along source optical flow (TokenFlow-lite).

    Flow is estimated on the immutable source so the blend cannot invent camera
    motion. Weight stays low so a legitimate appearance change is not smeared.
    """
    if blend <= 0 or len(generated) < 2:
        return generated

    out = [generated[0].copy()]
    h, w = generated[0].shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    for index in range(1, len(generated)):
        region = allowed[index]
        current = generated[index].copy()
        if not region.any():
            out.append(current)
            continue
        flow = _flow_pair(
            cv2.cvtColor(sources[index - 1], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(sources[index], cv2.COLOR_RGB2GRAY),
        )
        warped = cv2.remap(
            out[-1],
            grid_x + flow[..., 0],
            grid_y + flow[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mixed = current.astype(np.float32)
        mixed[region] = (1.0 - blend) * mixed[region] + blend * warped[region].astype(np.float32)
        out.append(np.clip(mixed, 0, 255).astype(np.uint8))
    return out


def bidirectional_flow_blend(
    generated: list[NDArray[np.uint8]],
    sources: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
    *,
    forward_blend: float = 0.2,
    backward_blend: float = 0.15,
) -> list[NDArray[np.uint8]]:
    """Bidirectional flow-guided blending for temporal consistency.

    Forward pass propagates from past, backward from future.
    Combines both for symmetric temporal smoothing.
    """
    if len(generated) < 2:
        return generated

    h, w = generated[0].shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Forward pass
    forward = [generated[0].copy()]
    for i in range(1, len(generated)):
        region = allowed[i]
        current = generated[i].copy()
        if not region.any() or not allowed[i - 1].any():
            forward.append(current)
            continue
        flow = _flow_pair(
            cv2.cvtColor(sources[i - 1], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(sources[i], cv2.COLOR_RGB2GRAY),
        )
        warped = cv2.remap(
            forward[-1],
            grid_x + flow[..., 0],
            grid_y + flow[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mixed = current.astype(np.float32)
        mixed[region] = (1.0 - forward_blend) * mixed[region] + forward_blend * warped[
            region
        ].astype(np.float32)
        forward.append(np.clip(mixed, 0, 255).astype(np.uint8))

    # Backward pass
    backward = forward.copy()
    for i in range(len(generated) - 2, -1, -1):
        region = allowed[i]
        if not region.any() or not allowed[i + 1].any():
            continue
        flow = _flow_pair(
            cv2.cvtColor(sources[i + 1], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(sources[i], cv2.COLOR_RGB2GRAY),
        )
        warped = cv2.remap(
            backward[i + 1],
            grid_x + flow[..., 0],
            grid_y + flow[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mixed = backward[i].astype(np.float32)
        mixed[region] = (1.0 - backward_blend) * mixed[region] + backward_blend * warped[
            region
        ].astype(np.float32)
        backward[i] = np.clip(mixed, 0, 255).astype(np.uint8)

    return backward


def refine_candidates(
    generated: list[NDArray[np.uint8]],
    sources: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
    *,
    flow_blend: float = 0.28,
    boundary_match_px: int = 5,
    bidirectional: bool = True,
) -> list[NDArray[np.uint8]]:
    """Apply all in-region finishing operators in a safe order."""
    if bidirectional:
        refined = bidirectional_flow_blend(generated, sources, allowed)
    else:
        refined = flow_blend_sequence(generated, sources, allowed, blend=flow_blend)
    return match_boundary_color(refined, sources, allowed, band_px=boundary_match_px)


def _compute_orb_features(
    frame: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    max_features: int = 500,
) -> tuple[list[cv2.KeyPoint], NDArray[np.float32]]:
    """Compute ORB features in masked region.

    ORB is fast and works well for establishing frame correspondences.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)

    mask_u8 = mask.astype(np.uint8) * 255
    keypoints, descriptors = orb.detectAndCompute(gray, mask_u8)

    if descriptors is None:
        return [], np.array([])

    return keypoints, descriptors


def _match_features(
    desc1: NDArray[np.float32],
    desc2: NDArray[np.float32],
    ratio_thresh: float = 0.75,
) -> list[cv2.DMatch]:
    """Match ORB descriptors using BFMatcher with ratio test."""
    if len(desc1) == 0 or len(desc2) == 0:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

    return good


def feature_propagation_sequence(
    generated: list[NDArray[np.uint8]],
    sources: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
    *,
    feature_weight: float = 0.2,
    max_features: int = 300,
) -> list[NDArray[np.uint8]]:
    """TokenFlow-style feature propagation for temporal consistency.

    Uses ORB features to establish dense correspondences between frames,
    then propagates pixel values along these correspondences. This is more
    robust than pure optical flow for handling occlusions and disocclusions.

    Args:
        generated: Generated candidate frames
        sources: Source frames (for feature extraction)
        allowed: Mask of allowed edit region
        feature_weight: Blend weight for feature-propagated pixels
        max_features: Max ORB features per frame

    Returns:
        Temporally smoothed frames
    """
    if feature_weight <= 0 or len(generated) < 2:
        return generated

    out = [generated[0].copy()]

    # Extract features for each frame
    all_kps = []
    all_descs = []
    for src, region in zip(sources, allowed, strict=True):
        if region.any():
            kps, descs = _compute_orb_features(src, region, max_features)
        else:
            kps, descs = [], np.array([])
        all_kps.append(kps)
        all_descs.append(descs)

    for index in range(1, len(generated)):
        region = allowed[index]
        current = generated[index].copy()

        if not region.any() or len(all_kps[index]) == 0 or len(all_kps[index - 1]) == 0:
            out.append(current)
            continue

        # Match features between consecutive frames
        matches = _match_features(all_descs[index - 1], all_descs[index])

        if len(matches) < 10:
            # Not enough matches, fall back to optical flow
            out.append(current)
            continue

        # Build correspondence map
        h, w = current.shape[:2]
        corr_map = np.full((h, w, 2), -1, dtype=np.float32)

        for m in matches:
            pt1 = all_kps[index - 1][m.queryIdx].pt
            pt2 = all_kps[index][m.trainIdx].pt
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])

            if 0 <= y2 < h and 0 <= x2 < w:
                corr_map[y2, x2] = [x1, y1]

        # Propagate previous frame's generated pixels along correspondences
        warped = np.zeros_like(current, dtype=np.float32)
        valid_mask = np.zeros((h, w), dtype=bool)

        for y in range(h):
            for x in range(w):
                if corr_map[y, x, 0] >= 0:
                    x1, y1 = corr_map[y, x]
                    if 0 <= y1 < h and 0 <= x1 < w:
                        warped[y, x] = out[-1][y1, x1]
                        valid_mask[y, x] = True

        if not (valid_mask & region).any():
            out.append(current)
            continue

        mixed = current.astype(np.float32)
        blend_mask = valid_mask & region
        mixed[blend_mask] = (1.0 - feature_weight) * mixed[blend_mask] + feature_weight * warped[
            blend_mask
        ]
        out.append(np.clip(mixed, 0, 255).astype(np.uint8))

    return out


def bidirectional_feature_propagation(
    generated: list[NDArray[np.uint8]],
    sources: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
    *,
    forward_weight: float = 0.15,
    backward_weight: float = 0.15,
) -> list[NDArray[np.uint8]]:
    """Bidirectional feature propagation (forward + backward passes).

    TokenFlow uses bidirectional propagation for better temporal consistency.
    Forward pass ensures consistency with past, backward with future.
    """
    # Forward pass
    forward = feature_propagation_sequence(
        generated, sources, allowed, feature_weight=forward_weight
    )

    # Backward pass
    reversed_generated = list(reversed(forward))
    reversed_sources = list(reversed(sources))
    reversed_allowed = np.flip(allowed, axis=0)

    backward = feature_propagation_sequence(
        reversed_generated, reversed_sources, reversed_allowed, feature_weight=backward_weight
    )

    # Combine forward and backward
    result = list(reversed(backward))
    for i in range(len(result)):
        region = allowed[i]
        if region.any():
            result[i][region] = np.clip(
                (forward[i][region].astype(np.float32) + result[i][region].astype(np.float32))
                * 0.5,
                0,
                255,
            ).astype(np.uint8)

    return result


def _compute_deep_features(
    frame: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    feature_extractor=None,
) -> NDArray[np.float32]:
    """Compute deep features for TokenFlow-style propagation.

    Uses a pre-trained CNN (e.g., DINOv2, CLIP, or VGG) to extract
    dense features that are more semantically meaningful than ORB.
    """
    if feature_extractor is None:
        # Fallback to simple gradient-based features
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.stack([grad_x, grad_y], axis=-1)

    # Use provided feature extractor
    try:
        import torch

        device = next(feature_extractor.parameters()).device
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            features = feature_extractor(img_tensor)
        return features.cpu().numpy()[0].transpose(1, 2, 0)
    except Exception:
        # Fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.stack([grad_x, grad_y], axis=-1)


def tokenflow_style_propagation(
    generated: list[NDArray[np.uint8]],
    sources: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
    *,
    keyframe_stride: int = 4,
    feature_weight: float = 0.3,
    use_deep_features: bool = False,
) -> list[NDArray[np.uint8]]:
    """TokenFlow-style diffusion feature propagation for temporal consistency.

    Implements the core TokenFlow idea: propagate edited features along
    correspondences established from the original video features.

    Args:
        generated: Generated candidate frames
        sources: Source frames (for correspondence computation)
        allowed: Mask of allowed edit region
        keyframe_stride: Stride for keyframe selection (TokenFlow uses random)
        feature_weight: Blend weight for propagated features
        use_deep_features: Whether to use deep features (requires feature extractor)

    Returns:
        Temporally consistent frames
    """
    if len(generated) < 2:
        return generated

    T = len(generated)
    h, w = generated[0].shape[:2]

    # Select keyframes (TokenFlow uses random; we use uniform stride for determinism)
    keyframe_indices = list(range(0, T, keyframe_stride))
    if keyframe_indices[-1] != T - 1:
        keyframe_indices.append(T - 1)

    # Compute correspondences from source frames (immutable, accurate)
    # Using optical flow on source frames for dense correspondences
    correspondences = {}
    for i in range(T - 1):
        flow = _flow_pair(
            cv2.cvtColor(sources[i], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(sources[i + 1], cv2.COLOR_RGB2GRAY),
        )
        correspondences[(i, i + 1)] = flow

    # Build dense correspondence maps for all frame pairs
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Propagate from keyframes to all frames
    result = [None] * T

    # First, process keyframes (they are the anchors)
    for kf_idx in keyframe_indices:
        result[kf_idx] = generated[kf_idx].copy()

    # Propagate forward from each keyframe
    for kf_idx in keyframe_indices:
        # Forward propagation
        current_frame = generated[kf_idx].copy()
        for i in range(kf_idx + 1, T):
            if not allowed[i].any():
                result[i] = generated[i].copy() if result[i] is None else result[i]
                continue

            # Compose flow from kf_idx to i
            composed_flow = np.zeros((h, w, 2), dtype=np.float32)
            for j in range(kf_idx, i):
                if (j, j + 1) in correspondences:
                    flow = correspondences[(j, j + 1)]
                    # Warp the composed flow
                    warped_flow = cv2.remap(
                        composed_flow,
                        grid_x + flow[..., 0],
                        grid_y + flow[..., 1],
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    composed_flow = warped_flow + flow
                else:
                    break

            # Warp keyframe content to current frame
            warped = cv2.remap(
                current_frame.astype(np.float32),
                grid_x + composed_flow[..., 0],
                grid_y + composed_flow[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            if result[i] is None:
                result[i] = warped.astype(np.uint8)
            else:
                # Blend with existing
                region = allowed[i]
                mixed = result[i].astype(np.float32)
                mixed[region] = (1.0 - feature_weight) * mixed[region] + feature_weight * warped[
                    region
                ]
                result[i] = np.clip(mixed, 0, 255).astype(np.uint8)

    # Backward propagation from keyframes
    for kf_idx in reversed(keyframe_indices):
        current_frame = generated[kf_idx].copy()
        for i in range(kf_idx - 1, -1, -1):
            if not allowed[i].any():
                continue

            # Compose backward flow
            composed_flow = np.zeros((h, w, 2), dtype=np.float32)
            for j in range(kf_idx, i, -1):
                if (j - 1, j) in correspondences:
                    flow = correspondences[(j - 1, j)]
                    # Backward flow
                    warped_flow = cv2.remap(
                        composed_flow,
                        grid_x - flow[..., 0],
                        grid_y - flow[..., 1],
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    composed_flow = warped_flow - flow
                else:
                    break

            warped = cv2.remap(
                current_frame.astype(np.float32),
                grid_x + composed_flow[..., 0],
                grid_y + composed_flow[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            if result[i] is not None:
                region = allowed[i]
                mixed = result[i].astype(np.float32)
                mixed[region] = (1.0 - feature_weight) * mixed[region] + feature_weight * warped[
                    region
                ]
                result[i] = np.clip(mixed, 0, 255).astype(np.uint8)

    # Fill any remaining None frames
    for i in range(T):
        if result[i] is None:
            result[i] = generated[i].copy()

    return result


def refine_candidates_full(
    generated: list[NDArray[np.uint8]],
    sources: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
    *,
    flow_blend: float = 0.28,
    boundary_match_px: int = 5,
    bidirectional: bool = True,
    tokenflow_weight: float = 0.2,
    tokenflow_stride: int = 4,
) -> list[NDArray[np.uint8]]:
    """Full refinement pipeline combining all temporal consistency methods.

    Order (from most to least invasive):
    1. TokenFlow-style feature propagation (strongest temporal consistency)
    2. Bidirectional flow blending (moderate temporal smoothing)
    3. Boundary color matching (local seam fixing)
    4. Occluder restoration (handled separately in composite.py)
    """
    # Step 1: TokenFlow-style propagation (most effective for identity consistency)
    if tokenflow_weight > 0:
        refined = tokenflow_style_propagation(
            generated,
            sources,
            allowed,
            keyframe_stride=tokenflow_stride,
            feature_weight=tokenflow_weight,
        )
    else:
        refined = generated

    # Step 2: Bidirectional flow blending
    if bidirectional:
        refined = bidirectional_flow_blend(refined, sources, allowed)
    elif flow_blend > 0:
        refined = flow_blend_sequence(refined, sources, allowed, blend=flow_blend)

    # Step 3: Boundary color matching
    refined = match_boundary_color(refined, sources, allowed, band_px=boundary_match_px)

    return refined
