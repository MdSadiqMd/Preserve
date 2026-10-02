"""Shadow and reflection detection for Generative Omnimatte-style effect handling.

Generative Omnimatte (CVPR 2025) decomposes video into layers where each object
layer includes its associated effects (shadows, reflections). Current binary
masks miss these effects, leaving halos when objects are removed/replaced.

This module detects and includes shadow/reflection regions in the edit mask.
"""

import cv2
import numpy as np
from numpy.typing import NDArray


def detect_shadow_regions(
    frame: NDArray[np.uint8],
    object_mask: NDArray[np.bool_],
    *,
    dilation_px: int = 30,
    darkness_threshold: float = 0.25,
    saturation_threshold: float = 0.4,
    direction_hint: tuple[float, float] | None = None,
) -> NDArray[np.bool_]:
    """Detect cast shadow regions extending from an object.

    Shadows are typically:
    - Darker than surroundings (lower value in HSV)
    - Desaturated (lower saturation)
    - Connected to the object base
    - In a consistent direction (opposite light source)
    """
    if not object_mask.any():
        return np.zeros_like(object_mask, dtype=bool)

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    _, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Dilate object mask to search region around it
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )
    search_region = cv2.dilate(object_mask.astype(np.uint8), kernel) > 0
    search_region = search_region & ~object_mask

    if not search_region.any():
        return np.zeros_like(object_mask, dtype=bool)

    # A shadow is darker than the SURROUNDING surface, not than the object:
    # a car with black tyres, grille and glass averages darker than the
    # asphalt shadow it casts, so an object-relative threshold found nothing
    # (audit 2026-09-19: dark blob left on the road after removal). Reference
    # the unoccluded search ring instead; shadows keep the surface's hue, so
    # saturation is bounded relative to the ring rather than the object.
    ring_v = float(np.median(v[search_region]))
    ring_s = float(np.median(s[search_region]))
    darker = v < ring_v * (1.0 - darkness_threshold)
    desaturated = s < ring_s + 255 * saturation_threshold * 0.5
    shadow_candidate = darker & desaturated & search_region

    # Keep only connected components touching object base
    # Find object bottom (where shadow typically starts)
    obj_coords = np.argwhere(object_mask)
    if len(obj_coords) > 0:
        # A ground shadow starts at the contact line, which for a car or a
        # standing figure lies well above the silhouette's lowest pixel
        # (wheels, feet). Restricting to rows below the bottom dropped the
        # whole cast shadow (audit 2026-09-19: dark blob left after removal).
        top_y, bottom_y = obj_coords[:, 0].min(), obj_coords[:, 0].max()
        shadow_below = np.zeros_like(object_mask, dtype=bool)
        shadow_below[(top_y + bottom_y) // 2 :, :] = True
        shadow_candidate = shadow_candidate & shadow_below

    # Surface texture (asphalt grain, grass) breaks a shadow into specks that
    # would each fail the area test: close the candidate map first so a real
    # shadow becomes one component before the connectivity filter runs.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    shadow_candidate = (
        cv2.morphologyEx(shadow_candidate.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel) > 0
    ) & search_region

    # Connected components - keep only those touching dilated object
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        shadow_candidate.astype(np.uint8), connectivity=8
    )

    result = np.zeros_like(object_mask, dtype=bool)
    obj_dilated = (
        cv2.dilate(
            object_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        )
        > 0
    )

    for label in range(1, num_labels):
        component = labels == label
        # Touching the object boundary and substantial in size.
        if (component & obj_dilated).any() and stats[label, cv2.CC_STAT_AREA] > 50:
            result |= component

    return result


def detect_reflection_regions(
    frame: NDArray[np.uint8],
    object_mask: NDArray[np.bool_],
    *,
    dilation_px: int = 40,
    brightness_threshold: float = 0.15,
    vertical_flip: bool = True,
) -> NDArray[np.bool_]:
    """Detect specular reflection regions (e.g., on wet ground, glass).

    Reflections are typically:
    - Vertically aligned with object (for ground reflections)
    - Similar color to object but brighter/more saturated
    - In a region below the object (for ground plane reflections)
    """
    if not object_mask.any():
        return np.zeros_like(object_mask, dtype=bool)

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Search region below object
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )
    search_region = cv2.dilate(object_mask.astype(np.uint8), kernel) > 0
    search_region = search_region & ~object_mask

    if vertical_flip:
        # Reflection is typically below object
        obj_coords = np.argwhere(object_mask)
        if len(obj_coords) > 0:
            bottom_y = obj_coords[:, 0].max()
            search_region = search_region & (np.arange(frame.shape[0])[:, None] >= bottom_y)

    if not search_region.any():
        return np.zeros_like(object_mask, dtype=bool)

    # Object color statistics
    obj_hue = h[object_mask]
    obj_sat = s[object_mask]
    obj_val = v[object_mask]

    if len(obj_hue) == 0:
        return np.zeros_like(object_mask, dtype=bool)

    # Reflections share hue with object but may be brighter
    hue_mean = np.mean(obj_hue)
    hue_std = np.std(obj_hue)
    sat_mean = np.mean(obj_sat)

    # Handle hue circularity
    hue_diff = np.abs(h.astype(np.float32) - hue_mean)
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)  # HSV hue is 0-179 in OpenCV

    similar_hue = hue_diff < (hue_std * 2 + 10)
    similar_sat = s > (sat_mean * 0.5)
    brighter = v > (np.mean(obj_val) * (1.0 + brightness_threshold))

    reflection_candidate = similar_hue & similar_sat & brighter & search_region

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        reflection_candidate.astype(np.uint8), connectivity=8
    )

    result = np.zeros_like(object_mask, dtype=bool)
    for label in range(1, num_labels):
        component = labels == label
        if stats[label, cv2.CC_STAT_AREA] > 30:
            # Check vertical alignment with object
            obj_x_center = (
                np.mean(np.argwhere(object_mask)[:, 1]) if object_mask.any() else frame.shape[1] / 2
            )
            comp_x_center = np.mean(np.argwhere(component)[:, 1])
            if abs(comp_x_center - obj_x_center) < frame.shape[1] * 0.3:
                result |= component

    return result


def detect_contact_shadow(
    frame: NDArray[np.uint8],
    object_mask: NDArray[np.bool_],
    *,
    band_px: int = 8,
) -> NDArray[np.bool_]:
    """Detect contact shadow - the dark band directly under an object.

    This is the most reliable shadow cue and is essential for realistic
    object removal (otherwise object appears to float).
    """
    if not object_mask.any():
        return np.zeros_like(object_mask, dtype=bool)

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]

    # Thin band at object bottom
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, 3))
    dilated_horiz = cv2.dilate(object_mask.astype(np.uint8), kernel) > 0
    contact_band = dilated_horiz & ~object_mask

    if not contact_band.any():
        return np.zeros_like(object_mask, dtype=bool)

    # Contact shadow is darker than surroundings
    obj_v = v[object_mask].mean() if object_mask.any() else 128
    band_v = v[contact_band]
    is_darker = band_v < (obj_v * 0.85)

    result = np.zeros_like(object_mask, dtype=bool)
    result[contact_band] = is_darker

    # Morphological cleanup
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE, kernel_clean) > 0
    result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean) > 0

    return result


def detect_wearer_shadow(
    frame: NDArray[np.uint8],
    object_mask: NDArray[np.bool_],
    parent_mask: NDArray[np.bool_],
    *,
    band_px: int = 20,
    darker_than: float = 0.9,
    not_darker_than: float = 0.55,
    stats: dict | None = None,
) -> NDArray[np.bool_]:
    """Shadow an attached object casts on its wearer (a cap brim on a forehead).

    Ring of band_px around the object restricted to the parent, kept where
    brightness falls below darker_than x the parent's own brightness just
    beyond the ring. Left outside the region, that band survives the removal
    as a dark line across the wearer (audit 2026-09-20).
    """
    if not object_mask.any() or not parent_mask.any():
        return np.zeros_like(object_mask, dtype=bool)
    v = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[..., 2]
    obj = object_mask.astype(np.uint8)
    # A worn object's shadow falls below it (overhead light): grow the band
    # downward only, so dark hair beside a cap is not mistaken for shadow.
    below = cv2.dilate(obj, np.ones((band_px + 1, 1), np.uint8), anchor=(0, band_px)) > 0
    ring = cv2.dilate(obj, _ellipse(band_px)) > 0
    outer = cv2.dilate(obj, _ellipse(band_px * 3)) > 0
    band = below & ~object_mask & parent_mask
    reference = outer & ~ring & parent_mask
    if stats is not None:
        stats["band_px"] = int(band.sum())
        stats["reference_px"] = int(reference.sum())
    if band.sum() < 16 or reference.sum() < 64:
        return np.zeros_like(object_mask, dtype=bool)
    reference_v = float(np.median(v[reference]))
    threshold = darker_than * reference_v
    if stats is not None:
        stats["band_v"] = float(np.median(v[band]))
        stats["threshold_v"] = threshold
    result = np.zeros_like(object_mask, dtype=bool)
    # Shadowed skin is dimmer than lit skin; eyebrows and hair are far darker
    # still and are features, not shadow (audit: a second eyebrow painted
    # above the real one when the band swallowed the brows). The contact
    # shadow right under a brim is nearly as dark as a brow, so the floor
    # only applies beyond the first few pixels from the object (audit
    # 2026-09-21: dark brim line left on the forehead at floor 0.55).
    # The deep contact shadow under a brim runs 0.3-0.4x of the lit skin for
    # 10px or more (cap-v20 dumps: band median 40 vs reference 118, only a
    # sliver kept at a 4px contact zone), while brows sit farther down.
    contact = cv2.dilate(obj, _ellipse(10)) > 0
    # No floor at all in the contact zone: the deepest brim shadow sits
    # under 0.3x and was still cut (cap-v21 dumps); brows are farther down.
    floor = np.where(contact, 0.0, not_darker_than * reference_v)
    # The whole strip below the object on the wearer is taken (its penumbra
    # fades too softly for a brightness cut: cap-v19 still met a shadowed
    # tail at the region edge); only far darker features (brows, hair) stay out.
    result[band] = v[band] > floor[band]
    if stats is not None:
        stats["darker_share"] = float((v[band] < threshold).mean())
    result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(1)) > 0
    # Only shadow that touches the object counts; dark eyebrows further away do not
    touching = cv2.dilate(obj, _ellipse(2)) > 0
    n, labels = cv2.connectedComponents(result.astype(np.uint8))
    keep = {lab for lab in np.unique(labels[touching]) if lab != 0}
    return np.isin(labels, list(keep)) if keep else np.zeros_like(object_mask, dtype=bool)


def _ellipse(radius: int) -> NDArray[np.uint8]:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))


def build_effect_mask(
    frames: list[NDArray[np.uint8]],
    object_masks: NDArray[np.uint8],
    *,
    include_cast_shadow: bool = True,
    include_reflection: bool = True,
    include_contact_shadow: bool = True,
) -> NDArray[np.uint8]:
    """Build combined effect mask (shadows + reflections) for all frames.

    This extends the object mask to include associated effects, which is
    critical for clean object removal/replacement (Generative Omnimatte approach).

    Args:
        frames: Source video frames
        object_masks: Binary object masks (T, H, W) uint8, 255 = object
        include_cast_shadow: Detect cast shadows
        include_reflection: Detect specular reflections
        include_contact_shadow: Detect contact shadows under object

    Returns:
        Combined effect mask (T, H, W) uint8, 255 = effect region
    """
    T = len(frames)
    effect_masks = np.zeros_like(object_masks)

    for i in range(T):
        obj_mask = object_masks[i] > 0
        if not obj_mask.any():
            effect_masks[i] = obj_mask.astype(np.uint8) * 255
            continue

        combined = np.zeros_like(obj_mask, dtype=bool)

        if include_contact_shadow:
            contact = detect_contact_shadow(frames[i], obj_mask)
            combined |= contact

        if include_cast_shadow:
            cast = detect_shadow_regions(frames[i], obj_mask)
            combined |= cast

        if include_reflection:
            reflection = detect_reflection_regions(frames[i], obj_mask)
            combined |= reflection

        effect_masks[i] = combined.astype(np.uint8) * 255

    return effect_masks


def extend_mask_with_effects(
    core_mask: NDArray[np.uint8],
    effect_mask: NDArray[np.uint8],
    *,
    max_extension_px: int = 50,
) -> NDArray[np.uint8]:
    """Extend core mask to include detected effects, with size limit.

    Prevents runaway effect detection from consuming the whole frame.
    """
    if not effect_mask.any():
        return core_mask

    # Process each frame individually since cv2.dilate only works on 2D
    T = core_mask.shape[0]
    extended = core_mask.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max_extension_px * 2 + 1, max_extension_px * 2 + 1)
    )

    for i in range(T):
        core_dilated = cv2.dilate(core_mask[i], kernel) > 0
        constrained_effects = (effect_mask[i] > 0) & core_dilated
        extended[i][constrained_effects] = 255

    return extended
