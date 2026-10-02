"""Recolour a tracked region without a generative model.

validated-solution.md lists "Color, exposure, blur, redaction" under "Tracked
matte plus conventional effect", and 6.1 notes conventional finishing is the
highest-reliability path. A hue rotation driven by a tracked matte changes only
the pixels it is handed, so the preservation invariant is satisfied by
construction rather than by hoping the model behaved.

Shading is preserved by rewriting hue and scaling saturation while leaving the
value channel alone, so highlights, shadows, and material texture survive.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

# Representative OpenCV hue (0-179) for each colour name.
TARGET_HUES: dict[str, int] = {
    "red": 0,
    "orange": 12,
    "yellow": 27,
    "green": 60,
    "cyan": 90,
    "teal": 90,
    "blue": 115,
    "purple": 140,
    "violet": 140,
    "magenta": 160,
    "pink": 168,
    "brown": 15,
}

ACHROMATIC_VALUES: dict[str, int] = {
    "black": 35,
    "white": 235,
    "gray": 130,
    "grey": 130,
    "silver": 175,
}


def _circular_mean_hue(hues: NDArray[np.float32], weights: NDArray[np.float32]) -> float:
    """Saturation-weighted circular mean of OpenCV hues (0-179).

    Hue is an angle: a red object straddles the 179/0 wrap, so the arithmetic
    median of {178, 2} is 90 — green — instead of ~0. Doubling maps the wrap to
    360°, where a mean is well defined.
    """
    angles = hues.astype(np.float64) * (2.0 * np.pi / 180.0)
    sin_mean = float(np.sum(np.sin(angles) * weights))
    cos_mean = float(np.sum(np.cos(angles) * weights))
    if sin_mean == 0 and cos_mean == 0:
        return 0.0
    return (np.degrees(np.arctan2(sin_mean, cos_mean)) / 2.0) % 180.0


def pooled_source_hue(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
) -> float | None:
    """Clip-wide saturation-weighted circular mean hue of the masked regions."""
    pooled_hues: list[NDArray[np.float32]] = []
    pooled_weights: list[NDArray[np.float32]] = []

    for frame, mask in zip(frames, masks, strict=True):
        region = mask > 0
        if not region.any():
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[..., 1][region]
        chromatic = sat > 30
        if not chromatic.any():
            continue
        pooled_hues.append(hsv[..., 0][region][chromatic])
        pooled_weights.append(sat[chromatic])

    if not pooled_hues:
        return None

    return _circular_mean_hue(np.concatenate(pooled_hues), np.concatenate(pooled_weights))


def estimate_hue_rotation(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    target_hue: int,
) -> float:
    """One hue rotation for the whole clip, from pooled chromatic pixels.

    A per-frame delta would wobble with the mask and lighting each frame — and
    near the red wrap it can jump by ±90° outright, which reads as flicker in
    the output even though the source itself was steady. Pooling every frame's
    chromatic pixels makes the rotation constant across the clip, so the output
    inherits only the source's own temporal variation.
    """
    source_hue = pooled_source_hue(frames, masks)
    if source_hue is None:
        return 0.0
    return float((target_hue - source_hue) % 180.0)


def recolor_region(
    frame: NDArray[np.uint8],
    region: NDArray[np.bool_],
    color: str,
    saturation_boost: float = 1.25,
    delta: float | None = None,
    gate_center: float | None = None,
    hue_gate: float = 26.0,
) -> NDArray[np.uint8]:
    """Return frame with region shifted toward color.

    Only pixels inside region are touched; the rest of the array is returned
    unchanged so the caller's composite still owns the preservation guarantee.

    Two strategies, chosen by how chromatic the region already is:

     Chromatic source: rotate hue by delta instead of assigning the target
      flatly, and scale saturation proportionally. Relative hue structure
      (two-tone panels, tinted reflections) survives, and neutral pixels —
      glass, tyres, shadows — stay near-neutral because their own low
      saturation carries through, rather than being repainted at a hard floor.
     Near-grey source: rotation has nothing to rotate, so assign the target
      hue directly and lift saturation to make the new colour read.

    The rotation is gated to pixels near the object's own dominant hue
    (gate_center, in OpenCV hue units): a car's glass reflects the sky and
    its lights are amber, and an ungated rotation repaints the windshield green
    and the headlamps pink. Pixels outside hue_gate of the dominant hue are
    left exactly as filmed — "make the car blue" means the paint, not the glass.

    delta should come from estimate_hue_rotation for sequence work; when
    omitted it is derived from this frame alone, which is fine for stills but
    wobbles across a clip.
    """
    if not region.any():
        return frame.copy()

    result = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    if color in ACHROMATIC_VALUES:
        # Desaturate toward neutral and retarget brightness, keeping the local
        # contrast that makes the surface read as a material.
        target = ACHROMATIC_VALUES[color]
        local = val[region]
        centred = local - local.mean()
        sat[region] = 0
        val[region] = np.clip(target + centred * 0.85, 0, 255)
    else:
        target_hue = TARGET_HUES.get(color)
        if target_hue is None:
            raise ValueError(f"No hue defined for colour {color!r}")

        regional_sat = sat[region]
        chromatic_mask = regional_sat > 30
        median_sat = float(np.median(regional_sat[chromatic_mask])) if chromatic_mask.any() else 0.0

        if median_sat >= 40:
            # Rotating preserves within-region variation; hue wraps at 180.
            if delta is None or gate_center is None:
                frame_source = (
                    _circular_mean_hue(hue[region][chromatic_mask], regional_sat[chromatic_mask])
                    if chromatic_mask.any()
                    else None
                )
                if delta is None and frame_source is not None:
                    delta = float((target_hue - frame_source) % 180.0)
                if gate_center is None:
                    gate_center = frame_source
            if gate_center is not None:
                distance = np.abs((hue - gate_center + 90.0) % 180.0 - 90.0)
                rotate = region & (distance <= hue_gate)
            else:
                rotate = region
            hue[rotate] = np.mod(hue[rotate] + delta, 180.0)
            sat[rotate] = np.clip(sat[rotate] * saturation_boost, 0, 255)
        else:
            hue[region] = target_hue
            # Lift saturation so a near-grey source still reads as the new colour,
            # but keep it proportional so shading variation is not flattened.
            lifted = np.maximum(regional_sat * saturation_boost, 90)
            sat[region] = np.clip(lifted, 0, 255)

    hsv[..., 0], hsv[..., 1], hsv[..., 2] = hue, sat, val
    recoloured = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    result[region] = recoloured[region]
    return result


def recolor_sequence(
    frames: list[NDArray[np.uint8]],
    masks: NDArray[np.uint8],
    color: str,
) -> list[NDArray[np.uint8]]:
    """Recolour a clip with one clip-wide rotation, keeping frames consistent."""
    target_hue = TARGET_HUES.get(color)
    source_hue = (
        pooled_source_hue(frames, masks)
        if target_hue is not None and color not in ACHROMATIC_VALUES
        else None
    )
    delta = float((int(target_hue) - source_hue) % 180.0) if source_hue is not None else None
    return [
        recolor_region(frame, mask > 0, color, delta=delta, gate_center=source_hue)
        for frame, mask in zip(frames, masks, strict=True)
    ]
