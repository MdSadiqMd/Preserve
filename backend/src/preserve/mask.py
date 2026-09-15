"""Mask creation, tracking, and manipulation."""

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
    phrase = target_phrase or _grounding_phrase(spec)

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

        grounded, grounding_stats = get_grounder().segment_frames(frames, phrase)
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

    # Detect and include shadow/reflection effects (Generative Omnimatte approach)
    # Only for removal operations where effects must also be removed
    if not hug_object:  # hug_object=False means REMOVE operation
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


def _grounding_phrase(spec: "TargetSpec") -> str:
    """Describe the target as a noun phrase for a text-conditioned segmenter."""
    colour = " ".join(sorted(spec.colors))
    subject = ", ".join(sorted(spec.targets))
    return f"{colour} {subject}".strip() or spec.raw_prompt


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

    dilated = dilate_mask(core_mask, definition.dilation_px)
    if keepout is not None:
        dilated[keepout] = 0

    # Use MatAnyone-style fractional alpha for segmentation masks with frames
    if definition.mask_type == MaskType.SEGMENTATION and frames is not None:
        from preserve.alpha_propagation import build_alpha_with_transparency

        alpha = build_alpha_with_transparency(frames, dilated)
        # Clamp alpha to the dilated region to prevent leakage into protected areas.
        # The fractional alpha matting can spread beyond the binary mask boundary;
        # we restrict it to the authorized edit region (dilated minus keepout).
        alpha[dilated == 0] = 0
        stats["alpha_method"] = "matanyone-fractional"
    else:
        alpha = feather_mask(dilated, definition.feather_px)
        # Only add alpha_method to stats for segmentation to avoid breaking
        # existing tests that expect empty stats for box/point masks
        if definition.mask_type == MaskType.SEGMENTATION:
            stats["alpha_method"] = "distance-feather"

    return core_mask, alpha, keepout, stats


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
