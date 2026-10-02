"""Replace a masked object with something else via masked diffusion.

This is the last resort in the hierarchy from validated-solution.md 3.3 — used
only when the requested pixels genuinely do not exist in the source and cannot be
expressed as a colour transform or recovered from another frame.

Two things matter for it to be usable on video:

Diffusers' own documentation warns that inpainting checkpoints may alter
  unmasked content, and ships apply_overlay() for exactly that reason. We do
  not rely on the model respecting the mask — the caller composites and then hard
  restores protected samples. This module's output is a candidate patch .
Independent per-frame sampling causes texture crawl and identity drift
  (5.2). One fixed seed and one shared latent noise tensor is a cheap, real
  mitigation: it makes the sampler start every frame from the same point, so the
  replacement keeps a consistent identity.

For shape-changing replacements (e.g., car -> truck, person -> dog), uses
VideoSwap-style semantic point correspondence to transfer motion trajectory
while allowing shape change.
"""

import re

import cv2
import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from PIL import Image

from preserve.config import settings
from preserve.prompt import ACHROMATIC_COLORS, COLOR_HUE_RANGES
from preserve.segment import _color_pixel_mask
from preserve.semantic_points import (
    videoswap_deform,
)

log = structlog.get_logger()

# Optical flow for propagating anchor patches runs on a downscale: Farneback is
# O(area), and flow fields are smooth enough that a quarter-scale estimate,
# resampled up, is accurate to well under a pixel at editing resolutions.
FLOW_MAX_DIM = 256

# SD 1.5 operates on multiples of 8. The checkpoint was trained at 512, but
# upsampling a smaller frame to 512 only to downsample the result costs ~1.8x the
# compute for detail the source never had, so the frame's own size is used when it
# is already a sane multiple of 8.
MODEL_TILE = 512
MIN_TILE = 320


def _tile_size(width: int, height: int) -> int:
    longest = max(width, height)
    if longest >= MODEL_TILE:
        return MODEL_TILE
    return max(MIN_TILE, (longest // 8) * 8)


def _gray(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _downscale_gray(gray: NDArray[np.uint8]) -> tuple[NDArray[np.float32], float]:
    height, width = gray.shape
    scale = min(1.0, FLOW_MAX_DIM / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )
    # Kept at the native 0-255 scale: Farneback's polynomial expansion returns
    # near-zero fields when handed [0,1]-normalised frames on this cv2 build.
    return gray.astype(np.float32), scale


def _flow_to_frame(from_gray: NDArray[np.uint8], to_gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Dense displacement field taking from_gray's coords onto to_gray.

    Estimated on a downscale, then returned at the full frame size. Resizing
    the field up spreads each value across more pixels but leaves magnitudes in
    small-frame units, so they are scaled back by 1/scale.
    """
    a, scale = _downscale_gray(from_gray)
    b, _ = _downscale_gray(to_gray)
    height, width = a.shape
    # Pyramid halvings must not collapse the frame: keep the coarsest level at
    # roughly 16px on a side, or the estimate degenerates on small frames.
    levels = max(1, min(4, int(np.log2(min(height, width) / 16))))
    flow = cv2.calcOpticalFlowFarneback(
        a,
        b,
        None,
        pyr_scale=0.5,
        levels=levels,
        winsize=21,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    height, width = from_gray.shape
    if scale < 1.0:
        flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow /= scale
    return flow.astype(np.float32)


def _warp_by_flow(image: NDArray[np.uint8], flow: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Sample image displaced by flow: out(y, x) = image(y + fy, x + fx).

    This pulls each destination pixel from where its content came from, so a
    patch rendered at the anchor appears at the tracked position in-between.
    """
    h, w = flow.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(
        image,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _anchor_indices(edited: list[int], stride: int) -> list[int]:
    """Pick which edited frames are rendered by diffusion directly."""
    if stride <= 1 or len(edited) <= 2:
        return list(edited)
    picked = [i for pos, i in enumerate(edited) if pos % stride == 0]
    if edited[-1] not in picked:
        picked.append(edited[-1])
    return picked


class SDInpaintReplacer:
    """Masked SD 1.5 inpainting, anchored and propagated across frames.

    Independent per-frame sampling re-invents the replacement every frame, which
    reads as texture crawl and identity drift (validated-solution.md 5.2; the
    mechanism TokenFlow and DiffuEraser both correct by propagating information
    along inter-frame correspondences rather than re-sampling). Two measures,
    cheap enough for unified memory, approximate that here:

     every anchor frame starts from the same seed, so the sampler begins each
      render from the identical noise realization;
     in-between frames are not sampled at all. Their patch is the flow-warped
      blend of the two bracketing anchors, so the replacement moves with the
      scene instead of being regenerated, and identity can only change at
      anchor boundaries.
    """

    def __init__(self) -> None:
        self._pipe = None
        self._config = settings.get_replacement_config()
        self._settings = self._config.get("settings", {})
        self.name = self._config.get("name", "SD1.5 Inpainting")

    def is_available(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        from diffusers import StableDiffusionInpaintPipeline

        model = self._config.get("model", "stable-diffusion-v1-5/stable-diffusion-inpainting")
        device = settings.get_device()
        dtype = torch.float32 if device.type == "mps" else torch.float16

        log.info("Loading replacement model", model=model)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            # See animatediff.py: eager load avoids meta-tensor materialization
            # failures when memory is tight.
            low_cpu_mem_usage=False,
        )
        pipe.set_progress_bar_config(disable=True)

        # Memory optimizations
        enable_tiling = self._settings.get("enable_vae_tiling", True)
        enable_slicing = self._settings.get("enable_attention_slicing", True)
        enable_offload = self._settings.get("enable_model_offload", False)

        if enable_tiling and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
            log.info("VAE tiling enabled for memory efficiency")

        if enable_slicing:
            pipe.enable_attention_slicing()
            log.info("Attention slicing enabled for memory efficiency")

        if enable_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            log.info("Model CPU offload enabled")
        else:
            pipe.to(device)

        self._pipe = pipe
        self._device = device

    def unload(self) -> None:
        self._pipe = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def replace_sequence(
        self,
        frames: list[NDArray[np.uint8]],
        masks: NDArray[np.uint8],
        prompt: str,
        negative_prompt: str | None = None,
        task: str = "replace",
    ) -> tuple[list[NDArray[np.uint8]], dict]:
        """Generate a replacement candidate for each frame's masked region.
        task is accepted for replacer-interface parity (VACE honors
        removal_ overrides); this backend currently ignores it.

        For shape-changing replacements (car->truck, person->dog), applies
        VideoSwap-style semantic point correspondence before diffusion.
        """
        if self._pipe is None:
            self.load()

        steps = int(self._settings.get("num_inference_steps", 25))
        guidance = float(self._settings.get("guidance_scale", 8.0))
        strength = float(self._settings.get("strength", 0.85))
        seed = int(self._settings.get("fixed_seed", 12345))
        stride = int(self._settings.get("anchor_stride", 1))
        color_evidence_min = float(self._settings.get("color_evidence_min", 0.10))
        color = _prompt_color(prompt)

        # Detect shape change request (e.g., "car into a truck", "person to dog")
        source_class, target_class = _parse_shape_change(prompt)
        shape_change = (
            source_class is not None and target_class is not None and source_class != target_class
        )

        negative = negative_prompt or (
            "blurry, low quality, distorted, deformed, extra objects, watermark, text"
        )

        height, width = frames[0].shape[:2]
        tile = _tile_size(width, height)

        edited_indices = [i for i, m in enumerate(masks) if (m > 0).any()]
        anchors = _anchor_indices(edited_indices, stride)

        # Apply VideoSwap deformation for shape-changing replacements
        if shape_change and edited_indices:
            log.info(
                "Applying VideoSwap semantic point deformation",
                source=source_class,
                target=target_class,
            )
            deformed_frames = videoswap_deform(frames, masks, source_class, target_class)
            # Use deformed frames as initialization for diffusion
            # This gives the model the correct shape while it fills in texture
            init_frames = deformed_frames
        else:
            init_frames = frames

        # Same seed plus strength 1.0 means every anchor starts from the identical
        # noise latent, so renders differ only through their surrounding context —
        # the strongest identity sharing this pipeline can get without a
        # video-native model.
        renders: dict[int, NDArray[np.uint8]] = {}
        for index in anchors:
            renders[index] = self._render_anchor(
                init_frames[index],
                masks[index],
                prompt,
                negative,
                tile,
                steps,
                guidance,
                strength,
                seed,
            )
            log.info("Replacement anchor rendered", frame=index + 1, total=len(frames))

        rerendered = 0
        borrowed = 0
        if color is not None and color_evidence_min > 0:
            rerendered = self._repair_missing_color(
                frames,
                renders,
                masks,
                color,
                prompt,
                negative,
                tile,
                steps,
                guidance,
                strength,
                seed,
                color_evidence_min,
            )

            # An anchor still failing after retries is dropped as a propagation
            # source; its segment is served by the nearest passing anchor
            # instead. Only when no anchor ever produced the object does the
            # job fail closed (validated-solution.md 11).
            failed = [
                i
                for i, r in renders.items()
                if self._colour_evidence(r, masks[i], color) < color_evidence_min
            ]
            if failed and len(failed) == len(renders):
                raise RuntimeError(
                    "The model could not produce the requested replacement "
                    f"(no anchor rendered the colour '{color}'). Try again, or "
                    "rephrase with a plainer subject and colour."
                )
            for index in failed:
                donor = min(
                    (a for a in renders if a not in failed),
                    key=lambda a: abs(a - index),
                    default=None,
                )
                if donor is not None:
                    del renders[index]
                    log.info(
                        "Anchor segment served by nearest passing anchor",
                        frame=index + 1,
                        donor_frame=donor + 1,
                    )
                    borrowed += 1

        candidates: list[NDArray[np.uint8] | None] = [None] * len(frames)
        for index, render in renders.items():
            candidates[index] = render

        propagated = self._fill_propagated(candidates, frames, edited_indices, set(renders))

        # Propagation decays on footage that morphs rather than moves (generated
        # clips register poorly — the same reason background reconstruction skips
        # them), so a segment that left its anchor with a weak patch stays weak
        # no matter how good the anchor was. The closed loop: verify every edited
        # frame's evidence and re-render failures natively — same seed, so the
        # rescue shares identity with the anchors instead of re-rolling it.
        rescued = 0
        if color is not None and color_evidence_min > 0:
            rescued = self._rescue_weak_frames(
                candidates,
                frames,
                masks,
                edited_indices,
                prompt,
                negative,
                tile,
                steps,
                guidance,
                strength,
                seed,
                color,
                color_evidence_min,
            )

            # Still-weak frames after a native retry: borrow the nearest strong
            # frame's candidate, flow-warped. Best effort — the alternative is a
            # removal-looking hole, which is the one outcome this pipeline never
            # ships.
            weak = [
                i
                for i in edited_indices
                if self._colour_evidence(candidates[i], masks[i], color) < color_evidence_min
            ]
            strong = [i for i in edited_indices if i not in weak]
            for index in weak:
                if not strong:
                    raise RuntimeError(
                        "The model could not produce the requested replacement "
                        f"(the colour '{color}' never appeared). Try again, or "
                        "rephrase with a plainer subject and colour."
                    )
                donor = min(strong, key=lambda a: abs(a - index))
                flow = _flow_to_frame(_gray(frames[donor]), _gray(frames[index]))
                candidates[index] = _warp_by_flow(candidates[donor], flow)
                borrowed += 1
                log.info(
                    "Weak frame borrowed nearest strong frame",
                    frame=index + 1,
                    donor_frame=donor + 1,
                )

        final = [
            candidates[index] if candidates[index] is not None else frame
            for index, frame in enumerate(frames)
        ]

        provenance = {
            "backend": self.name,
            "model": self._config.get("model"),
            "prompt": prompt,
            "negative_prompt": negative,
            "seed": seed,
            "strength": strength,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "frames_generated": len(edited_indices),
            "anchors_rendered": len(anchors),
            "anchors_rerendered": rerendered,
            "frames_rescued": rescued,
            "frames_borrowed": borrowed,
            "frames_propagated": propagated,
            "anchor_stride": stride if stride > 1 else 1,
            "propagation": "farneback-flow" if propagated else None,
            "tile": tile,
        }
        return final, provenance

    def _render_anchor(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8],
        prompt: str,
        negative: str,
        tile: int,
        steps: int,
        guidance: float,
        strength: float,
        seed: int,
    ) -> NDArray[np.uint8]:
        """One masked diffusion render at an anchor frame."""
        height, width = frame.shape[:2]
        generator = torch.Generator("cpu").manual_seed(seed)
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=Image.fromarray(frame).resize((tile, tile), Image.LANCZOS),
            mask_image=Image.fromarray(mask).resize((tile, tile), Image.NEAREST),
            height=tile,
            width=tile,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        return cv2.resize(
            np.asarray(result, dtype=np.uint8),
            (width, height),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def _repair_missing_color(
        self,
        frames: list[NDArray[np.uint8]],
        renders: dict[int, NDArray[np.uint8]],
        masks: NDArray[np.uint8],
        color: str,
        prompt: str,
        negative: str,
        tile: int,
        steps: int,
        guidance: float,
        strength: float,
        seed: int,
        min_evidence: float,
        max_retries: int = 2,
    ) -> int:
        """Re-render anchors that ignored the instruction's colour.

        An anchor occasionally fills its region with background — road where the
        taxi belongs — and propagation would carry that failure across its whole
        segment. Masked-region histograms cannot catch this: legitimate renders
        of the same object at different poses correlate near zero (measured
        0.0-0.18 across anchors), so there is no consensus to deviate from. What
         is checkable is the instruction itself: when the prompt names a
        colour, a render that obeyed it shows that colour across the object's
        interior — measured on the eroded core, taxi fills run 12-43% and road
        fills 0-6%. Failing anchors are re-rolled and the render with the most
        colour evidence wins. This is the automated slice of the failure policy
        in validated-solution.md 11.
        """
        rerendered = 0
        for index, render in list(renders.items()):
            best = self._colour_evidence(render, masks[index], color)
            if best >= min_evidence:
                continue

            log.info(
                "Anchor missing requested colour, re-rendering",
                frame=index + 1,
                evidence=round(best, 3),
                color=color,
            )
            for attempt in range(max_retries):
                alternative = self._render_anchor(
                    frames[index],
                    masks[index],
                    prompt,
                    negative,
                    tile,
                    steps,
                    guidance,
                    strength,
                    seed + 1000 * (attempt + 1),
                )
                alt_evidence = self._colour_evidence(alternative, masks[index], color)
                if alt_evidence > best:
                    renders[index] = alternative
                    best = alt_evidence
                if best >= min_evidence:
                    break
            rerendered += 1

        return rerendered

    def _colour_evidence(
        self, render: NDArray[np.uint8], mask: NDArray[np.uint8], color: str
    ) -> float:
        """Fraction of the object's interior matching the requested colour.

        Measured on the eroded mask, not the full edit region: the region
        includes a dilation margin of genuine background, which dilutes the
        fraction and lets a mostly-background fill scrape past a lenient
        threshold (measured: taxi fills run 12-43% yellow in the core, road
        fills 0-6%, but the same road fill scored 5-6% once the margin was
        included — enough to pass a 5% gate and poison every frame that
        propagated from it).
        """
        region = mask > 0
        if not region.any():
            return 1.0

        eroded = cv2.erode(region.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if not eroded.any():
            eroded = region
        hsv = cv2.cvtColor(render, cv2.COLOR_RGB2HSV)
        return float((_color_pixel_mask(hsv, color) & eroded).sum() / eroded.sum())

    def _rescue_weak_frames(
        self,
        candidates: list[NDArray[np.uint8] | None],
        frames: list[NDArray[np.uint8]],
        masks: NDArray[np.uint8],
        edited_indices: list[int],
        prompt: str,
        negative: str,
        tile: int,
        steps: int,
        guidance: float,
        strength: float,
        seed: int,
        color: str,
        min_evidence: float,
    ) -> int:
        """Repair edited frames whose propagated patch lost the object.

        Flow-warped propagation assumes the scene moves; generated footage
        morphs, so warps degrade and a taxi can dissolve a few frames from its
        anchor even when every anchor was fine. But a full-strength native
        re-render re-rolls the identity, which shows as flicker against the
        frames that propagated fine.

        The repair is therefore DiffuEraser's propagation-prior principle: the
        nearest strong frame's patch is warped in and initialised under the
        mask at reduced strength, so the sampler keeps the neighbour's object
        layout while re-rendering it into this frame's context. Identity flows
        from the neighbour; the model fixes what the warp got wrong.
        """
        refine_strength = min(0.7, strength)
        rescued = 0

        def evidence_of(index: int) -> float:
            candidate = candidates[index]
            if candidate is None:
                return 0.0
            return self._colour_evidence(candidate, masks[index], color)

        for index in edited_indices:
            best = evidence_of(index)
            if best >= min_evidence:
                continue

            # Nearest frame whose content still shows the object; it is both
            # the identity source and the layout prior. Sweeping in index order
            # and preferring the closest preceding strong frame keeps warp
            # spans at 1-2 frames — the only range where Farneback stays
            # trustworthy on morphing footage — so each repair chains off the
            # last good frame instead of reaching across the clip.
            donors = [i for i in edited_indices if i != index and evidence_of(i) >= min_evidence]

            attempts: list[tuple[NDArray[np.uint8], float, float]] = []
            if donors:
                preceding = [d for d in donors if d < index]
                donor = min(preceding or donors, key=lambda a: abs(a - index))
                flow = _flow_to_frame(_gray(frames[donor]), _gray(frames[index]))
                warped = _warp_by_flow(candidates[donor], flow)
                init = frames[index].copy()
                region = masks[index] > 0
                init[region] = warped[region]
                refined = self._render_anchor(
                    init,
                    masks[index],
                    prompt,
                    negative,
                    tile,
                    steps,
                    guidance,
                    refine_strength,
                    seed,
                )
                attempts.append(
                    (refined, self._colour_evidence(refined, masks[index], color), refine_strength)
                )

            # No strong neighbour (or as a fallback): a native full-strength
            # render from the shared seed.
            native = self._render_anchor(
                frames[index],
                masks[index],
                prompt,
                negative,
                tile,
                steps,
                guidance,
                strength,
                seed,
            )
            attempts.append((native, self._colour_evidence(native, masks[index], color), strength))

            patch, ev, used = max(attempts, key=lambda a: a[1])
            if ev > best:
                candidates[index] = patch
                rescued += 1
                log.info(
                    "Weak frame repaired",
                    frame=index + 1,
                    evidence=round(ev, 3),
                    mode="propagation-prior" if used < strength else "native",
                )

        if rescued:
            log.info("Weak frames repaired", frames=rescued)
        return rescued

    def _fill_propagated(
        self,
        candidates: list[NDArray[np.uint8] | None],
        frames: list[NDArray[np.uint8]],
        edited_indices: list[int],
        anchors: set[int],
    ) -> int:
        """Fill in-between edited frames from the nearest anchor, flow-warped.

        Each anchor's render owns its segment: an in-between frame is that
        segment's patch pulled along the measured motion, nothing else.
        Blending two anchors was tried and rejected — the renders are different
        objects, and cross-dissolving them ghosts one object out of existence
        while the other fades in. Flow fields are smoothed along time before
        use because per-frame Farneback on soft generated footage jitters.
        """
        ordered = sorted(anchors)
        if not ordered:
            return 0

        raw_fields: dict[tuple[int, int], NDArray[np.float32]] = {}
        for index in edited_indices:
            if index in anchors:
                continue
            before = max((a for a in ordered if a < index), default=None)
            after = min((a for a in ordered if a > index), default=None)
            donor = (
                before
                if after is None
                else after
                if before is None
                else (before if index - before <= after - index else after)
            )
            raw_fields[(donor, index)] = _flow_to_frame(_gray(frames[donor]), _gray(frames[index]))

        served_by_donor: dict[int, list[int]] = {}
        for donor, index in raw_fields:
            served_by_donor.setdefault(donor, []).append(index)

        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        for donor, indices in served_by_donor.items():
            indices.sort()
            if len(indices) < 3:
                continue
            stack = np.stack([raw_fields[(donor, i)] for i in indices])
            padded = np.pad(stack, ((1, 1), (0, 0), (0, 0), (0, 0)), mode="edge")
            blurred = sum(kernel[k] * padded[k : k + len(indices)] for k in range(3))
            for pos, i in enumerate(indices):
                raw_fields[(donor, i)] = blurred[pos]

        filled = 0
        for (donor, index), flow in sorted(raw_fields.items(), key=lambda kv: kv[0][1]):
            candidates[index] = _warp_by_flow(candidates[donor], flow)
            filled += 1

        if filled:
            log.info("Replacement patches propagated", frames=filled)
        return filled


def _prompt_color(prompt: str) -> str | None:
    """The first named colour word in a prompt, if any."""
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    for word in sorted(words):
        if word in COLOR_HUE_RANGES or word in ACHROMATIC_COLORS:
            return word
    return None


def _parse_shape_change(prompt: str) -> tuple[str | None, str | None]:
    """Parse shape change request from prompt.

    Detects patterns like:
    - "car into a truck" -> ("car", "truck")
    - "person to dog" -> ("person", "dog")
    - "replace car with bus" -> ("car", "bus")
    - "swap person for cat" -> ("person", "cat")
    - "change car into motorcycle" -> ("car", "motorcycle")

    Returns (source_class, target_class) or (None, None) if no shape change detected.
    """
    import re

    prompt_lower = prompt.lower()

    # COCO class names for reference
    coco_classes = {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    }

    # Pattern 1: "X into Y" or "X to Y"
    match = re.search(r"\b(\w+)\s+(?:into|to)\s+(?:a\s+)?(\w+)\b", prompt_lower)
    if match:
        src, tgt = match.groups()
        if src in coco_classes and tgt in coco_classes:
            return src, tgt

    # Pattern 2: "replace X with Y" or "replace X by Y"
    match = re.search(r"replace\s+(\w+)\s+(?:with|by)\s+(?:a\s+)?(\w+)", prompt_lower)
    if match:
        src, tgt = match.groups()
        if src in coco_classes and tgt in coco_classes:
            return src, tgt

    # Pattern 3: "swap X for Y" or "swap X with Y"
    match = re.search(r"swap\s+(\w+)\s+(?:for|with)\s+(?:a\s+)?(\w+)", prompt_lower)
    if match:
        src, tgt = match.groups()
        if src in coco_classes and tgt in coco_classes:
            return src, tgt

    # Pattern 4: "change X into Y" or "change X to Y"
    match = re.search(r"change\s+(\w+)\s+(?:into|to)\s+(?:a\s+)?(\w+)", prompt_lower)
    if match:
        src, tgt = match.groups()
        if src in coco_classes and tgt in coco_classes:
            return src, tgt

    # Pattern 5: "turn X into Y" or "turn X to Y"
    match = re.search(r"turn\s+(\w+)\s+(?:into|to)\s+(?:a\s+)?(\w+)", prompt_lower)
    if match:
        src, tgt = match.groups()
        if src in coco_classes and tgt in coco_classes:
            return src, tgt

    # Pattern 6: "make X a Y" or "make X into Y"
    match = re.search(r"make\s+(\w+)\s+(?:a|into)\s+(\w+)", prompt_lower)
    if match:
        src, tgt = match.groups()
        if src in coco_classes and tgt in coco_classes:
            return src, tgt

    return None, None


_replacer: SDInpaintReplacer | None = None


def get_replacer():
    """Process-wide replacer so the checkpoint loads only once.

    The backend follows replacement.default in models_config.yaml. VACE renders
    the whole clip in one pass with attention spanning frames, which removes the
    per-frame identity drift the SD path can only mitigate; sd_inpaint stays
    available as a fallback.
    """
    global _replacer
    if _replacer is None:
        default = settings.get_replacement_default()
        if default == "vace":
            from preserve.edits.vace import get_vace_replacer

            _replacer = get_vace_replacer()
        elif default == "lucy":
            from preserve.edits.lucy import get_lucy_replacer

            _replacer = get_lucy_replacer()
        else:
            _replacer = SDInpaintReplacer()
    return _replacer
