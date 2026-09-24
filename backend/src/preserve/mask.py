"""Mask creation, tracking, and manipulation."""

import re
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from preserve.config import settings
from preserve.models import BoundingBox, MaskDefinition, MaskType, Point

if TYPE_CHECKING:
    from preserve.prompt import TargetSpec


def create_mask_from_points(
    points: list[Point],
    width: int,
    height: int,
    frame_count: int,
) -> NDArray[np.uint8]:
    """Create binary masks from point annotations.

    Points define a polygon; interpolates between keyframes.
    """
    masks = np.zeros((frame_count, height, width), dtype=np.uint8)

    keyframes: dict[int, list[tuple[int, int]]] = {}
    for p in points:
        if p.frame not in keyframes:
            keyframes[p.frame] = []
        keyframes[p.frame].append((int(p.x * width), int(p.y * height)))

    sorted_frames = sorted(keyframes.keys())

    for i, frame_idx in enumerate(sorted_frames):
        pts = np.array(keyframes[frame_idx], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(masks[frame_idx], [pts], 255)

        if i < len(sorted_frames) - 1:
            next_frame = sorted_frames[i + 1]
            next_pts = np.array(keyframes[next_frame], dtype=np.int32)
            if len(pts) == len(next_pts) and len(pts) >= 3:
                for f in range(frame_idx + 1, next_frame):
                    t = (f - frame_idx) / (next_frame - frame_idx)
                    interp_pts = (pts * (1 - t) + next_pts * t).astype(np.int32)
                    cv2.fillPoly(masks[f], [interp_pts], 255)

    return masks


def create_mask_from_box(
    box: BoundingBox,
    width: int,
    height: int,
    frame_count: int,
) -> NDArray[np.uint8]:
    """Create binary masks from bounding box (single keyframe, no tracking yet)."""
    masks = np.zeros((frame_count, height, width), dtype=np.uint8)

    x1 = int(box.x * width)
    y1 = int(box.y * height)
    x2 = int((box.x + box.width) * width)
    y2 = int((box.y + box.height) * height)

    masks[box.frame, y1:y2, x1:x2] = 255
    for f in range(box.frame + 1, frame_count):
        masks[f] = masks[box.frame]

    return masks


def dilate_mask(mask: NDArray[np.uint8], pixels: int) -> NDArray[np.uint8]:
    """Dilate mask by specified pixels."""
    if pixels <= 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))

    if mask.ndim == 3:
        return np.stack([cv2.dilate(m, kernel) for m in mask])
    return cv2.dilate(mask, kernel)


def feather_mask(mask: NDArray[np.uint8], pixels: int) -> NDArray[np.float32]:
    """Soft alpha whose support stays inside the binary mask.

    Distance-transform ramp (not Gaussian blur): a blur leaks alpha outside
    the authorized region, wasting the composite and inviting seam math to
    touch pixels that hard-restore must then put back. Inside the mask, alpha
    reaches 1 after pixels from the boundary.
    """
    binary = mask > 0
    if pixels <= 0:
        return binary.astype(np.float32)

    def _feather_one(plane: NDArray[np.uint8]) -> NDArray[np.float32]:
        on = (plane > 0).astype(np.uint8)
        if not on.any():
            return np.zeros(plane.shape, dtype=np.float32)
        dist = cv2.distanceTransform(on, cv2.DIST_L2, 3)
        alpha = np.clip(dist / float(pixels), 0.0, 1.0)
        alpha[on == 0] = 0.0
        return alpha.astype(np.float32)

    if mask.ndim == 3:
        return np.stack([_feather_one(m) for m in mask])
    return _feather_one(mask)


def create_mask_from_prompt(
    prompt: str,
    frames: list[NDArray[np.uint8]],
    target: "TargetSpec | None" = None,
    hug_object: bool = False,
    target_phrase: str | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_], dict]:
    """Locate the edit's subject and return per-frame masks.

    Three grounding paths, tried in order:

    1. COCO instance segmentation, when the subject is a class the detector knows.
       This gives tight, tracked, per-instance masks and colour filtering.
    2. Open-vocabulary text grounding (CLIPSeg) on target_phrase, for any subject
       the detector does not cover — "her dress", "the girl on the left".
    3. SAM 2 refinement of whichever mask the first two produced, recovering the
       object extent detectors under-cover (window lines, skirts, wheels).

    A miss on the first two is reported as "could not locate <subject>" — an
    honest not-found, never a "couldn't parse your phrasing". The caller then
    declines the edit rather than touching the whole frame, which is the point
    of the project.

    Additionally, detects shadows and reflections (Generative Omnimatte approach)
    to extend the mask for clean object removal/replacement.
    """
    from preserve.prompt import parse_prompt
    from preserve.segment import get_segmenter

    spec = target if target is not None else parse_prompt(prompt)
    phrase = _bare_noun_phrase(target_phrase) if target_phrase else _grounding_phrase(spec)

    stats: dict = {"mask_source": "none"}
    masks = None
    keepout = None

    if not spec.is_empty:
        segmenter = get_segmenter()
        masks, keepout, stats = segmenter.segment_frames(frames, spec, hug_object=hug_object)
        stats["spec"] = spec.describe()
        stats["mask_source"] = "instance-segmentation"

    # Fall back to open-vocabulary grounding when the detector knows no matching
    # class, or knew the class but found no instance in this footage.
    if masks is None or stats.get("instances_matched", 0) == 0:
        from preserve.groundseg import get_grounder

        from preserve.edits.classify import wants_multiple

        grounded, grounding_stats = get_grounder().segment_frames(
            frames, phrase, multiple=wants_multiple(target.raw_prompt if target else phrase)
        )
        if grounding_stats["frames_with_detections"] > 0:
            masks = grounded
            keepout = np.zeros(masks.shape, dtype=bool)
            stats.update(grounding_stats)
            stats["mask_source"] = "text-grounding"
            stats["instances_matched"] = grounding_stats["frames_with_detections"]
            stats["tracks_matched"] = 1
            stats["tracks_total"] = 1

    if masks is None:
        height, width = frames[0].shape[:2]
        masks = np.zeros((len(frames), height, width), dtype=np.uint8)
        keepout = np.zeros(masks.shape, dtype=bool)
        stats.setdefault("instances_matched", 0)

    masks = _refine_with_sam(frames, masks, stats)
    from preserve.edits.coherence import stabilize_mask_sequence

    masks = stabilize_mask_sequence(masks, hug_object=hug_object)
    stats["mask_stabilized"] = True
    if not hug_object:
        # Removal: the matte must reach the object's outer rim, or the rim
        # survives the fill as an outline (audit 2026-09-20: red frame edge
        # after glasses removal, dark stick edge after popsicle removal).
        masks, grown_pixels = grow_by_object_colour(frames, masks)
        stats["colour_grown_pixels"] = grown_pixels

    attached = False
    if not hug_object and (masks > 0).any():
        # Removal: is the target attached to another detected object (a cap on
        # a head, a straw at a mouth)? Then what it hides is that object's
        # surface, never background: donors from other frames are invalid, the
        # object's own dark features are not cast shadows, and the fill needs
        # a prompted semantic model rather than a background remover.
        from preserve.background import attachment_parent, attachment_ratio

        occluders, per_class = get_segmenter().instance_union(frames)
        ratio = attachment_ratio(masks, occluders)
        stats["attachment_ratio"] = round(ratio, 3)
        stats["_occluders"] = occluders
        attached_max = float(settings.get_background_config().get("attached_max_ring_ratio", 0.5))
        # A known wearable counts as attached on far less contact: a fedora's
        # ring is mostly sky, only its underside touches the head (audit
        # 2026-09-21: ratio 0.30 sent a hat to the background remover, which
        # painted a beige blob where the head was).
        from preserve.edits.classify import is_wearable

        wearable_min = float(settings.get_background_config().get("wearable_min_ring_ratio", 0.15))
        wearable = is_wearable(target_phrase or phrase)
        if ratio >= attached_max or (wearable and ratio >= wearable_min):
            attached = True
            stats["attachment_parent"] = attachment_parent(masks, per_class) or "object"
            # The object's shadow on its wearer belongs to the edit (a brim's
            # band on the forehead), bounded to a ring on the parent.
            from preserve.effects import detect_wearer_shadow, extend_mask_with_effects

            shadow_stats: dict = {}
            shadow = (
                np.stack(
                    [
                        detect_wearer_shadow(frame, mask > 0, parent, stats=shadow_stats)
                        for frame, mask, parent in zip(frames, masks, occluders, strict=True)
                    ]
                ).astype(np.uint8)
                * 255
            )
            shadow_pixels = int((shadow > 0).sum())
            stats["wearer_shadow"] = {**shadow_stats, "pixels": shadow_pixels}
            if shadow_pixels:
                before_shadow = masks > 0
                masks = extend_mask_with_effects(masks, shadow, max_extension_px=16)
                # Shadow pixels carry SHADOW_VALUE so the keyframe editor can
                # erase just that band before rendering: an image editor
                # otherwise keeps the brim's shadow on the forehead as a
                # structure prior (audit 2026-09-21: dark strip inside the fill).
                masks[(masks > 0) & ~before_shadow] = SHADOW_VALUE
                stats["wearer_shadow_pixels"] = shadow_pixels
            # The grown matte's outline is jagged at the 5-10px scale (SAM
            # edge + colour growth + shadow strip); on a reveal that outline
            # becomes the seam across skin, so it is smoothed first.
            masks = _smooth_outline(masks)
            beyond = np.stack(
                [
                    _beyond_wearer(mask > 0, parent)
                    for mask, parent in zip(masks, occluders, strict=True)
                ]
            )
            masks[beyond & (masks == 255)] = BACKGROUND_SIDE_VALUE
            stats["background_side_pixels"] = int((beyond & (masks > 0)).sum())
            if bool(settings.get_background_config().get("interaction_removal", True)):
                # A hand holding the target is its interaction (VOID, 2026):
                # left in place it grips nothing. Hand pixels touching the
                # target join the edit so the fill can lower or hide them.
                hand, hand_pixels = _touching_hand(frames, masks, occluders)
                hand_frames = [i for i in range(len(frames)) if (hand[i] > 0).any()]
                stats["interaction_hand_frames"] = (
                    f"{len(hand_frames)}/{len(frames)} ({hand_frames[0]}-{hand_frames[-1]})"
                    if hand_frames
                    else "0"
                )
                if hand_pixels:
                    # Interaction pixels carry the value 128 (still > 0 for
                    # every region test) so the anchor builder can tell them
                    # apart: a hand moves with itself, not with the wearer,
                    # so its fill is never carried along source flow.
                    masks = np.maximum(masks, (hand > 0).astype(np.uint8) * INTERACTION_VALUE)
                    stats["interaction_hand_pixels"] = hand_pixels

    # Detect and include shadow/reflection effects (Generative Omnimatte approach)
    # Only for removal of free-standing objects, whose shadows fall on background
    if not hug_object and not attached:
        from preserve.effects import build_effect_mask, extend_mask_with_effects

        effect_mask = build_effect_mask(frames, masks)
        effect_pixels = int((effect_mask > 0).sum())
        if effect_pixels > 0:
            original_pixels = int((masks > 0).sum())
            masks = extend_mask_with_effects(masks, effect_mask)
            stats["effect_pixels_added"] = effect_pixels
            stats["original_pixels"] = original_pixels
            stats["effect_extension_pct"] = round(100 * effect_pixels / max(1, original_pixels), 1)
            stats["effects_detected"] = True

    stats.setdefault("spec", spec.describe())
    stats["target_phrase"] = phrase
    return masks, keepout, stats


def _refine_with_sam(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    stats: dict,
) -> NDArray[np.uint8]:
    """Grow masks to full object extent with SAM 2, when enabled and grounded."""
    config = settings.get_refinement_config()
    if not config.get("settings", {}).get("enabled", False):
        return masks
    if stats.get("mask_source") not in ("instance-segmentation", "text-grounding"):
        return masks

    from preserve.refine import get_refiner

    refined, refine_stats = get_refiner().refine_sequence(frames, masks)
    stats.update(refine_stats)
    stats["mask_source"] += "+sam2"
    return refined


def _bare_noun_phrase(phrase: str) -> str:
    """No possessive or determiner: "their caps" lights the people up for
    CLIPSeg, "caps" lights the caps (same failure as the residue metric)."""
    return re.sub(
        r"^(?:his|her|their|its|my|our|your|the|a|an|this|that|these|those)\s+",
        "",
        phrase.strip(),
        flags=re.IGNORECASE,
    )


def _grounding_phrase(spec: "TargetSpec") -> str:
    """Describe the target as a noun phrase for a text-conditioned segmenter."""
    colour = " ".join(sorted(spec.colors))
    subject = ", ".join(sorted(spec.targets))
    return f"{colour} {subject}".strip() or _bare_noun_phrase(spec.raw_prompt)


def process_mask(
    definition: MaskDefinition,
    width: int,
    height: int,
    frame_count: int,
    frames: list[NDArray[np.uint8]] | None = None,
    target: "TargetSpec | None" = None,
    hug_object: bool = False,
    target_phrase: str | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.bool_] | None, dict]:
    """Process a mask definition into core mask, soft alpha, keepout, and stats.

    keepout marks pixels that belong to other detected objects and must not be
    altered even though dilation would otherwise reach them; it is None for mask
    types that carry no object semantics.

    For SEGMENTATION masks with frames available, uses MatAnyone-style fractional
    alpha propagation to handle hair, motion blur, and transparency boundaries.
    """
    stats: dict = {}
    keepout: NDArray[np.bool_] | None = None

    if definition.mask_type == MaskType.POINTS and definition.points:
        core_mask = create_mask_from_points(definition.points, width, height, frame_count)
    elif definition.mask_type == MaskType.BOX and definition.box:
        core_mask = create_mask_from_box(definition.box, width, height, frame_count)
    elif definition.mask_type == MaskType.SEGMENTATION and definition.segmentation_prompt:
        if frames is None:
            raise ValueError("Segmentation masks require decoded frames")
        core_mask, keepout, stats = create_mask_from_prompt(
            definition.segmentation_prompt,
            frames,
            target=target,
            hug_object=hug_object,
            target_phrase=target_phrase,
        )
    else:
        raise ValueError(f"Unsupported mask type: {definition.mask_type}")

    # core_mask may carry marks (SHADOW_VALUE, INTERACTION_VALUE, ...); every
    # geometric consumer sees the binary region.
    dilated = dilate_mask((core_mask > 0).astype(np.uint8) * 255, definition.dilation_px)
    if keepout is not None:
        dilated[keepout] = 0

    # Fractional (MatAnyone-style) alpha only when the object stays in frame
    # (recolour/replace): soft hair and motion-blur edges then blend naturally.
    # For removal every partially-covered pixel would mix the old object back
    # in at its alpha (audit 2026-09-19: a 50% translucent car over a clean
    # fill), so removal takes the binary region with an outer feather only.
    if definition.mask_type == MaskType.SEGMENTATION and frames is not None and hug_object:
        from preserve.alpha_propagation import build_alpha_with_transparency

        alpha = build_alpha_with_transparency(frames, dilated)
        # Clamp alpha to the dilated region to prevent leakage into protected areas.
        # The fractional alpha matting can spread beyond the binary mask boundary;
        # we restrict it to the authorized edit region (dilated minus keepout).
        alpha[dilated == 0] = 0
        # Transparency belongs to the boundary band only: inside the object
        # the replacement must be opaque, or the old object shows through
        # its own glossy panels (audit 2026-09-20: red patches on a yellow car).
        band = max(2, int(definition.feather_px or 0) + 2)
        interior = np.stack(
            [
                cv2.erode((m > 0).astype(np.uint8), np.ones((2 * band + 1, 2 * band + 1), np.uint8))
                for m in core_mask
            ]
        )
        alpha[interior > 0] = 1.0
        stats["alpha_method"] = "matanyone-fractional"
    else:
        alpha = feather_mask(dilated, definition.feather_px)
        if definition.mask_type == MaskType.SEGMENTATION:
            stats["alpha_method"] = "distance-feather"

    return core_mask, alpha, keepout, stats


# Matte value marking interaction pixels (a hand holding the target); every
# region test uses mask > 0, so they behave as part of the edit everywhere
# except where the anchor builder asks which pixels move with the wearer.
INTERACTION_VALUE = 128
# Matte value for the object's shadow on its wearer (erased before the
# keyframe editor sees the frame, generated like any other region pixel).
SHADOW_VALUE = 64
# Matte value for an attached target's pixels beyond its wearer (a brim over
# the wall behind): erased classically before the keyframe editor, which
# otherwise keeps the object's outline as a pale ghost on the background.
BACKGROUND_SIDE_VALUE = 32
# Every value the keyframe editor erases before rendering. A holding hand is
# erased too: rendered from its reference the editor kept it as a pale blob
# (popsicle-v19 frame 0); painted over from the jacket around it, the editor
# only has to make the surface plausible.
ERASE_VALUES = (SHADOW_VALUE, BACKGROUND_SIDE_VALUE, INTERACTION_VALUE)


def _smooth_outline(masks: NDArray[np.uint8], radius: int = 5) -> NDArray[np.uint8]:
    """Round the region outline (values kept) by a blurred-threshold of its support."""
    out = masks.copy()
    for i, mask in enumerate(masks):
        support = (mask > 0).astype(np.float32)
        rounded = cv2.GaussianBlur(support, (0, 0), radius) > 0.5
        added = rounded & ~(mask > 0)
        out[i][~rounded] = 0
        out[i][added] = 255
    return out


def _beyond_wearer(
    region: NDArray[np.bool_], parent: NDArray[np.bool_], margin_px: int = 10
) -> NDArray[np.bool_]:
    """Region pixels whose nearest pixel outside the region is not on the wearer.

    The detector's person mask includes worn items, so "inside the parent"
    says nothing about a brim over the wall behind; what the pixel borders
    does (audit 2026-09-21: 150 px/frame marked on a fedora, ghost stayed).
    """
    if not region.any():
        return np.zeros_like(region)
    # The detector's person mask overshoots the hat by a few pixels, so the
    # wall right above a brim still read as "wearer" (capwalk-v18: only the
    # brim's rim was marked). Judge against the parent's core.
    parent = (
        cv2.erode(
            parent.astype(np.uint8), np.ones((2 * margin_px + 1, 2 * margin_px + 1), np.uint8)
        )
        > 0
    )
    outside = (~region).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        outside * 0 + region.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    # labels index the nearest zero pixel (outside the region), 1-based in
    # row-major order over zero pixels
    zero_index = np.flatnonzero(outside.ravel())
    nearest = zero_index[np.clip(labels.ravel() - 1, 0, len(zero_index) - 1)]
    nearest_on_parent = parent.ravel()[nearest].reshape(region.shape)
    beyond = region & ~nearest_on_parent
    # Nearest-neighbour boundaries are pixel-jagged; a smoothed mark keeps the
    # classical erase from leaving blocky bokeh at the hair edge (cap-v20).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    smoothed = cv2.morphologyEx(beyond.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel) > 0
    return smoothed & region


def _touching_hand(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    occluders: NDArray[np.bool_],
    threshold: float = 0.45,
    touch_px: int = 12,
    max_gap: int = 16,
) -> tuple[NDArray[np.uint8], int]:
    """Hand regions (open-vocabulary) whose pixels touch the target, per frame."""
    from preserve.groundseg import get_grounder

    # "fingers" fires where "a hand" does not (a hand half out of frame:
    # 0.76 vs 0.31 on the popsicle clip's first frame); take the stronger.
    grounder = get_grounder()
    heat = np.maximum(grounder.relevance(frames, "a hand"), grounder.relevance(frames, "fingers"))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch_px + 1, 2 * touch_px + 1))
    clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    result = np.zeros_like(masks)
    added = 0
    components: list[tuple[NDArray[np.int32], int]] = []
    for i, (h, mask, parent) in enumerate(zip(heat, masks, occluders, strict=True)):
        # Not restricted to the detected person: an arm entering from the
        # frame edge is often outside that mask (popsicle frames 0-4); the
        # touch test and neighbour propagation bound the candidates instead.
        hand = (h > threshold) & ~(mask > 0)
        hand_u8 = cv2.morphologyEx(hand.astype(np.uint8), cv2.MORPH_OPEN, clean)
        n, labels = cv2.connectedComponents(hand_u8)
        components.append((labels, n))
        if not hand.any():
            continue
        touching = cv2.dilate(mask, kernel) > 0
        keep = {lab for lab in np.unique(labels[touching]) if lab != 0}
        if keep:
            chosen = np.isin(labels, list(keep))
            result[i][chosen] = 255
            added += int(chosen.sum())
    # A hand that stops touching the target's matte (SAM ends the stick above
    # the fingers in some frames) is the same hand: keep components that
    # overlap a marked neighbour frame, sweeping forward then backward, in
    # every frame (a frame's own small mark must not block the sweep:
    # popsicle-v25 frame 0 kept the hand while frame 1 was marked).
    for order in (range(1, len(frames)), range(len(frames) - 2, -1, -1)):
        for i in order:
            j = i - 1 if order.step > 0 else i + 1
            if not (result[j] > 0).any():
                continue
            labels, n = components[i]
            keep = {lab for lab in np.unique(labels[result[j] > 0]) if lab != 0}
            if keep:
                chosen = np.isin(labels, list(keep)) & ~(result[i] > 0)
                result[i][chosen] = 255
                added += int(chosen.sum())
    # A frame where the target's matte ends above the hand (SAM stops at the
    # stick) gets no mark of its own; carry the nearest marked frame's hand
    # along source flow (popsicle-v21: frame 0 kept the hand).
    from preserve.edits.coherence import warp_mask_between

    areas = [int((result[i] > 0).sum()) for i in range(len(frames))]
    marked = [i for i in range(len(frames)) if areas[i] > 0]
    if marked and min(areas) < 0.3 * max(areas):
        for i in range(len(frames)):
            nearest = min(marked, key=lambda j: abs(j - i))
            # Frames with far less hand than their marked neighbour (the
            # relevance map dropped out) take the neighbour's hand by flow
            if areas[i] >= 0.3 * areas[nearest]:
                continue
            if abs(nearest - i) > max_gap:
                continue
            carried = warp_mask_between(result[nearest] > 0, frames[nearest], frames[i])
            carried &= ~(masks[i] > 0) & ~(result[i] > 0)
            if carried.any():
                result[i][carried] = 255
                added += int(carried.sum())
    return result, added


def grow_by_object_colour(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    ring_px: int = 10,
    delta_e: float = 14.0,
    clusters: int = 4,
) -> tuple[NDArray[np.uint8], int]:
    """Extend each frame's matte to ring pixels that continue the object's colours.

    Ring pixels (within ring_px) join the matte when they sit within delta_e
    (CIE76 Lab) of one of the object's dominant colours and closer to it than
    to any colour of the surrounding annulus, so a red spectacle frame is
    followed to its edge while the wearer's skin next to it is not. Only
    components touching the matte are kept; growth is bounded by the ring.
    """
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * ring_px + 1, 4 * ring_px + 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    grown = masks.copy()
    added = 0
    for i, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
        region = mask > 0
        if region.sum() < 64:
            continue
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        ring = (cv2.dilate(mask, ring_kernel) > 0) & ~region
        context = (cv2.dilate(mask, outer_kernel) > 0) & ~ring & ~region
        if ring.sum() < 16 or context.sum() < 16:
            continue

        def centroids(pixels: NDArray[np.float32]) -> NDArray[np.float32]:
            sample = pixels[:: max(1, len(pixels) // 4000)]
            k = min(clusters, len(sample))
            _, _, centres = cv2.kmeans(sample, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            return centres

        object_colours = centroids(lab[region])
        context_colours = centroids(lab[context])
        ring_lab = lab[ring]
        to_object = np.linalg.norm(ring_lab[:, None, :] - object_colours[None], axis=2).min(1)
        to_context = np.linalg.norm(ring_lab[:, None, :] - context_colours[None], axis=2).min(1)
        candidate = np.zeros(region.shape, bool)
        candidate[ring] = (to_object < delta_e) & (to_object < to_context)
        if not candidate.any():
            continue
        n, labels = cv2.connectedComponents((candidate | region).astype(np.uint8))
        keep = np.isin(labels, np.unique(labels[region])) & candidate
        added += int(keep.sum())
        grown[i][keep] = 255
    return grown, added


def compute_allowed_region(
    core_mask: NDArray[np.uint8],
    dilation_px: int,
    keepout: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Compute R_allowed: every sample authorized to change."""
    dilated = dilate_mask(core_mask, dilation_px)
    allowed = dilated > 0
    if keepout is not None:
        allowed &= ~keepout
    return allowed


def compute_protected_region(allowed: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Compute Q_protected: complement of R_allowed."""
    return ~allowed
