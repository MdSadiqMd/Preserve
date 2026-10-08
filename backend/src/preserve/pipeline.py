"""Main editing pipeline orchestrating the full workflow."""

import re
from datetime import datetime

import cv2
import numpy as np
import structlog
from numpy.typing import NDArray

from preserve.background import (
    blend_fills,
    build_background_plate,
    estimate_frame_transforms,
    gate_by_local_background,
    gate_by_patch_coherence,
    registration_residual,
)
from preserve.color import ColorMetadata, create_color_metadata_from_ffprobe
from preserve.composite import composite_sequence
from preserve.config import settings
from preserve.edits.classify import (
    EditOperation,
    EditSpec,
    classify_edit,
    is_flat_surface_target,
    reveal_surface,
)
from preserve.edits.coherence import match_boundary_color
from preserve.edits.recolor import recolor_sequence
from preserve.edits.replace import get_replacer
from preserve.inpaint import InpaintBackend, InpaintRequest
from preserve.inpaint.minimax_remover import create_minimax_backend
from preserve.inpaint.propainter import create_propainter_backend
from preserve.long_video import (
    WindowConfig,
    process_long_video,
)
from preserve.mask import (
    BACKGROUND_SIDE_VALUE,
    ERASE_VALUES,
    INTERACTION_VALUE,
    SHADOW_VALUE,
    compute_allowed_region,
    compute_protected_region,
    dilate_mask,
    feather_mask,
    process_mask,
)
from preserve.models import EditJob, JobStatus, MaskType, VideoMetadata
from preserve.verify import verify_sequence_preservation
from preserve.video import (
    extract_frames,
    write_lossless_master,
    write_playable,
)

log = structlog.get_logger()


def frame_range(
    start_ms: int, end_ms: int, fps: float, frame_count: int, duration_ms: int
) -> tuple[int, int]:
    """Half-open [start, end) frame interval for a millisecond range.

    A request that reaches the clip's end means every frame: truncating
    duration_ms * fps / 1000 dropped the last frame (33 frames at 16fps probe
    as 2062ms -> 32.99 -> 32), so the output was one frame short of the
    source and the audit could not compare them (validated-solution 3.2:
    compare timing with rational integers, not rounded decimal seconds).
    """
    start = max(0, min(frame_count, round(start_ms * fps / 1000)))
    end = frame_count if end_ms >= duration_ms else min(frame_count, round(end_ms * fps / 1000))
    return start, max(start, end)


class EditPipeline:
    """Orchestrates the full localized video editing pipeline."""

    def __init__(self):
        self._backends: dict[str, InpaintBackend] = {}

    def get_backend(self, backend_name: str | None = None) -> InpaintBackend:
        """Get or create inpainting backend."""
        name = backend_name or settings.inpaint_backend

        if name not in self._backends:
            if name == "propainter":
                self._backends[name] = create_propainter_backend()
            elif name == "minimax":
                self._backends[name] = create_minimax_backend()
            else:
                raise ValueError(f"Unknown backend: {name}")

        backend = self._backends[name]
        if not backend.is_available():
            log.info("Loading backend", backend=backend.name)
            backend.load()

        return backend

    def _resize_frames(self, frames: list, target_width: int) -> tuple:
        """Resize frames for processing, return (resized_frames, scale)."""
        h, w = frames[0].shape[:2]
        if w <= target_width:
            return frames, 1.0

        scale = target_width / w
        new_h = int(h * scale)
        new_w = target_width

        resized = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
        return resized, scale

    def _upscale_frames(self, frames: list, target_size: tuple) -> list:
        """Upscale frames to target size (width, height).

        Lanczos rather than cubic: only the inpainted fill is upscaled, and it is
        already the softest part of the frame, so the sharper kernel matters.
        """
        return [cv2.resize(f, target_size, interpolation=cv2.INTER_LANCZOS4) for f in frames]

    def _upscale_masks(self, masks: NDArray, target_size: tuple) -> NDArray:
        """Upscale a (T, H, W) binary mask stack with a smooth boundary.

        Linear interpolation thresholded at half keeps the region binary but
        turns the processing-grid staircase into a straight edge (nearest
        left a 1.33x jagged seam along every composite boundary).
        """
        return np.stack(
            [
                np.where(
                    cv2.GaussianBlur(
                        cv2.resize(
                            (m > 0).astype(np.uint8) * 255,
                            target_size,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                        (0, 0),
                        1.5,
                    )
                    > 127,
                    255,
                    0,
                ).astype(np.uint8)
                for m in masks
            ]
        )

    def _build_full_res_regions(
        self,
        core_mask: NDArray[np.uint8],
        keepout: NDArray[np.bool_] | None,
        target_size: tuple,
        dilation_px: int,
        feather_px: int,
        scale: float,
        processing_frames: list[NDArray[np.uint8]] | None = None,
        processing_alpha: NDArray[np.float32] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.uint8], NDArray[np.uint8]]:
        """Recompute alpha, allowed region, and object occupancy at output size.

        If processing_alpha (fractional alpha from MatAnyone-style propagation) is
        provided, it is upscaled and used instead of recomputing from binary masks.
        This preserves hair/transparency boundaries that binary dilation would lose.
        """
        background_config = settings.get_background_config()
        core_full = self._upscale_masks(core_mask, target_size)

        dilation_full = max(1, int(round(dilation_px / scale)))
        dilated = dilate_mask(core_full, dilation_full)
        occupancy = core_full

        if keepout is not None:
            keepout_full = self._upscale_masks(keepout.astype(np.uint8) * 255, target_size)
            occupancy = np.maximum(occupancy, keepout_full)

            # Instance masks routinely miss wheels and lower panels, so grow the
            # keepout before subtracting it or dilation reaches those pixels.
            grow_px = int(background_config.get("keepout_grow_px", 8) / scale)
            grown = dilate_mask(keepout_full, grow_px) if grow_px > 0 else keepout_full
            dilated[grown > 0] = 0

        occupancy_px = int(background_config.get("occupancy_dilate_px", 12) / scale)
        if occupancy_px > 0:
            occupancy = dilate_mask(occupancy, occupancy_px)

        # Use fractional alpha if available (MatAnyone-style), otherwise recompute
        if processing_alpha is not None:
            # Upscale the fractional alpha to full resolution
            alpha = self._upscale_alpha(processing_alpha, target_size)
            # Recompute allowed from upscaled alpha (non-zero alpha = allowed)
            allowed = alpha > 0
            edit_mask = (allowed * 255).astype(np.uint8)
        else:
            feather_full = max(0, int(round(feather_px / scale)))
            alpha = feather_mask(dilated, feather_full)
            allowed = dilated > 0
            edit_mask = dilated

        return alpha, allowed, edit_mask, occupancy

    def _upscale_alpha(self, alpha: NDArray[np.float32], target_size: tuple) -> NDArray[np.float32]:
        """Upscale fractional alpha with Lanczos to preserve soft transitions."""
        return np.stack(
            [cv2.resize(a, target_size, interpolation=cv2.INTER_LANCZOS4) for a in alpha]
        )

    def _reconstruct_background(
        self,
        frames: list[NDArray[np.uint8]],
        edit_mask: NDArray[np.uint8],
        occupancy: NDArray[np.uint8],
        fallback: list[NDArray[np.uint8]],
        refine_prompt: str | None = None,
        refine_negative: str | None = None,
        refine_enabled: bool | None = None,
    ) -> tuple[list[NDArray[np.uint8]], dict]:
        """Replace the model's fill with real background pixels where available.
        DiffuEraser-style hierarchy:
        1. Same-frame clone / clean plate
        2. Temporal pixel propagation (real background from other frames)
        3. Feature propagation
        4. Generative diffusion completion only for genuinely unknown pixels
        Step 4 is the cascade: pixels with no trusted plate coverage are
        re-rendered by the video-diffusion backend conditioned on the
        plate-filled surroundings, instead of showing the propagation
        model's hallucination. Without refine_prompt the ProPainter
        fallback shows through unchanged (legacy behavior).
        """
        config = settings.get_background_config()
        # Mask-aware registration: estimate background motion from unoccluded
        # points only, and score the fit on the background too. The morphing
        # subject otherwise corrupts the transform and inflates the residual,
        # and the fail-closed gate then discards real background pixels.
        exclude = (occupancy > 0) | (edit_mask > 0)
        exclude_list = [m for m in exclude]
        transforms = estimate_frame_transforms(frames, exclude_masks=exclude_list)
        # Fail closed rather than paste confidently-wrong pixels: if the rigid
        # motion model does not explain this footage, copied background cannot be
        # trusted anywhere in the clip.
        residual = registration_residual(frames, transforms, exclude_masks=exclude_list)
        max_residual = float(config.get("max_registration_error", 18.0))
        if residual > max_residual:
            log.warning(
                "Background reconstruction skipped: camera motion does not fit",
                residual=round(residual, 1),
                threshold=max_residual,
            )
            return fallback, {
                "background_skipped": "registration-unreliable",
                "registration_residual": round(residual, 1),
                "real_background_pct": 0.0,
            }

        plate = build_background_plate(frames, occupancy, transforms)
        window = int(config.get("window", 0)) or len(frames)
        min_samples = int(config.get("min_samples", 5))
        mode = str(config.get("mode", "median"))
        feather = int(config.get("blend_feather_px", 5))
        max_spread = config.get("max_donor_spread", 28.0)
        max_spread = None if max_spread is None else float(max_spread)
        local_enabled = bool(config.get("local_gate_enabled", True))
        local_win = int(config.get("local_gate_win", 31))
        local_k = float(config.get("local_gate_k", 3.0))
        local_floor = float(config.get("local_gate_floor", 12.0))
        patch_enabled = bool(config.get("patch_coherence_enabled", True))
        patch_thresh = float(config.get("patch_coherence_thresh", 0.45))
        # DiffuEraser-style: build confidence map of propagated background
        # Pixels with high confidence (seen in many frames) use real background
        # Pixels with low confidence fall back to model fill
        fills: list[tuple[NDArray[np.uint8], NDArray[np.bool_]]] = []
        needed = 0
        recovered = 0
        generative_filled = 0
        local_rejected = 0
        patch_rejected = 0
        for index in range(len(frames)):
            fill, available = plate.fill_for_frame(
                index,
                window=window,
                min_samples=min_samples,
                mode=mode,
                max_spread=max_spread,
            )
            region = edit_mask[index] > 0
            needed += int(region.sum())
            if local_enabled:
                before = int((region & available).sum())
                available = gate_by_local_background(
                    fill,
                    available,
                    frames[index],
                    edit_mask[index],
                    occupancy[index],
                    win=local_win,
                    k=local_k,
                    floor=local_floor,
                )
                local_rejected += before - int((region & available).sum())
            if patch_enabled:
                before = int((region & available).sum())
                available = gate_by_patch_coherence(
                    fill,
                    available,
                    frames[index],
                    edit_mask[index],
                    occupancy[index],
                    thresh=patch_thresh,
                )
                patch_rejected += before - int((region & available).sum())
            recovered += int((region & available).sum())
            fills.append((fill, available))
        # Step 4 cascade: unknown-only pixels (no trusted plate coverage) are
        # re-rendered by video diffusion conditioned on the plate surroundings,
        # instead of showing the propagation model's hallucination. Fail-soft:
        # any diffusion error keeps the ProPainter fallback for those pixels.
        refined_fallback = list(fallback)
        cascade_unknown_pct = 0.0
        cascade_ran = False
        cascade_enabled = (
            refine_enabled
            if refine_enabled is not None
            else bool(config.get("refine_unknown_with_vace", False))
        )
        if refine_prompt is not None:
            unknown = np.stack([(edit_mask[i] > 0) & ~fills[i][1] for i in range(len(frames))])
            cascade_unknown_pct = round(100 * int(unknown.sum()) / max(1, needed), 1)
            min_ratio = float(config.get("refine_min_unknown_ratio", 0.05))
            # Single-shot capacity: windowed VACE over long clips OOM-kills
            # this hardware (SIGKILL, uncatchable) — measured on 171 frames.
            # The cascade stays a short-clip tool until windowed rendering
            # fits the memory budget.
            vace_settings = settings.get_replacement_config("vace").get("settings", {})
            max_window = int(vace_settings.get("max_window_frames", 81))
            if len(frames) > max_window:
                log.info(
                    "Cascade refinement skipped: clip exceeds single-shot capacity",
                    frames=len(frames),
                    max_window=max_window,
                )
            elif (
                cascade_enabled
                and unknown.any()
                and (int(unknown.sum()) / max(1, needed)) >= min_ratio
            ):
                try:
                    refined, _ = get_replacer().replace_sequence(
                        frames,
                        (unknown.astype(np.uint8) * 255),
                        refine_prompt,
                        refine_negative,
                        task="remove",
                    )
                    refined_fallback = refined
                    cascade_ran = True
                except Exception:
                    log.warning(
                        "Cascade refinement failed, keeping propagation fill", exc_info=True
                    )
        results: list[NDArray[np.uint8]] = []
        for index, model_fill in enumerate(refined_fallback):
            fill, available = fills[index]
            # Blend: real background where available, model fill where not
            # Use confidence-weighted blending
            blended = blend_fills(fill, model_fill, available, feather)
            results.append(blended)
        stats = {
            "pixels_needing_fill": needed,
            "pixels_from_real_background": recovered,
            "real_background_pct": round(100 * recovered / max(1, needed), 1),
            "registration_residual": round(residual, 1),
            "generative_filled_pct": round(100 * generative_filled / max(1, needed), 1),
            "max_donor_spread": max_spread,
            "local_rejected": local_rejected,
            "patch_rejected": patch_rejected,
            "cascade_ran": cascade_ran,
            "cascade_unknown_pct": cascade_unknown_pct,
        }
        return results, stats

    @staticmethod
    def _roi_box(
        core_mask: NDArray[np.uint8],
        shape: tuple[int, ...],
        pad_ratio: float = 0.25,
        min_pad: int = 64,
        min_side: int = 448,
        max_area_ratio: float = 0.6,
    ) -> tuple[int, int, int, int] | None:
        """Native-resolution crop around the edit region for the generative backends.

        Union bounding box of the region over time, padded for context, grown
        to at least min_side per axis and snapped to multiples of 16. None when
        the region already spans most of the frame (a full-frame render is no
        worse). The 480p-class models otherwise see a cap or a face as a few
        dozen latent tokens after the whole frame is downscaled (audit
        2026-09-20: VACE reveal smeared a 150px cap; the same model has
        ~9x the tokens for it inside a crop).
        """
        ys, xs = np.nonzero(core_mask.max(axis=0) > 0)
        if ys.size == 0:
            return None
        h, w = shape[:2]
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

        def grow(a: int, b: int, limit: int) -> tuple[int, int]:
            # Context padding per axis, so a wide, flat region (a cap) is not
            # padded vertically by its width and pushed past the area cap.
            pad = int(max(min_pad, pad_ratio * (b - a)))
            size = max(b - a + 2 * pad, min_side)
            size = min(((size + 15) // 16) * 16, (limit // 16) * 16)
            start = (a + b) // 2 - size // 2
            start = max(0, min(start, limit - size))
            return start, start + size

        y0, y1 = grow(y0, y1, h)
        x0, x1 = grow(x0, x1, w)
        if (y1 - y0) * (x1 - x0) >= max_area_ratio * h * w:
            return None
        return y0, y1, x0, x1

    def _render_in_roi(
        self,
        frames: list[NDArray[np.uint8]],
        core_mask: NDArray[np.uint8],
        render,
    ) -> tuple[list[NDArray[np.uint8]], dict]:
        """Run render(frames, masks) on the ROI crop and paste the result back.

        Pixels outside the crop are the source's own; the composite and the
        hard restore downstream never see the difference.
        """
        box = self._roi_box(core_mask, frames[0].shape)
        if box is None:
            return render(frames, core_mask, None), {}
        y0, y1, x0, x1 = box
        log.info("Rendering in ROI crop", roi=f"{x1 - x0}x{y1 - y0}+{x0}+{y0}")
        crop_frames = [np.ascontiguousarray(f[y0:y1, x0:x1]) for f in frames]
        crop_masks = np.ascontiguousarray(core_mask[:, y0:y1, x0:x1])
        rendered = render(crop_frames, crop_masks, box)
        out = []
        for frame, patch in zip(frames, rendered, strict=True):
            full = frame.copy()
            full[y0:y1, x0:x1] = patch
            out.append(full)
        return out, {"roi": [x0, y0, x1, y1]}

    def _carry_prior_anchors(
        self,
        prior: dict[int, NDArray[np.uint8]],
        fr: list[NDArray[np.uint8]],
        mk: NDArray[np.uint8],
        box: tuple[int, int, int, int] | None,
    ) -> dict:
        """Anchors for a later window: the previous window's finished overlap
        frames (cropped to this window's ROI) plus the last overlap frame's
        fill carried by flow to the window's last frame, so no frame is far
        from an anchor and the identity chosen once never re-rolls."""
        from preserve.edits.coherence import warp_keyframe_fill

        grow_px = int(
            settings.get_replacement_config("vace").get("settings", {}).get("mask_grow_px", 8)
        )
        anchors: dict = {}
        for index, frame in prior.items():
            if 0 <= index < len(fr):
                if box is not None:
                    y0, y1, x0, x1 = box
                    frame = np.ascontiguousarray(frame[y0:y1, x0:x1])
                anchors[index] = frame
        last = len(fr) - 1
        if anchors and last not in anchors:
            source_index = max(anchors)
            region_last = dilate_mask(mk[last : last + 1], grow_px)[0] > 0
            warped, carried = warp_keyframe_fill(
                anchors[source_index], fr[source_index], fr[last], region_last
            )
            carried &= ~(mk[last] == INTERACTION_VALUE)
            anchors[last] = (warped, ~(region_last & ~carried))
        return anchors

    def _keyframe_anchors(
        self,
        fr: list[NDArray[np.uint8]],
        mk: NDArray[np.uint8],
        prompts: dict[str, str],
        negative: str | None,
        provenance: dict,
        replacement: str | None = None,
        erase_first: bool = False,
        texture_from_ring: bool = False,
    ) -> tuple[dict[int, NDArray[np.uint8]], NDArray[np.uint8] | None]:
        """One image-edited keyframe plus flow-warped copies at both ends.

        With replacement (a phrase such as "a yellow taxi"), the keyframe's
        proposal is segmented for that object and any footprint beyond the
        source matte is carried to every frame along source flow and returned
        as an extension mask (T, H, W): a taller taxi is not clipped to the
        car's silhouette.

        An image editor has the semantics (hair under a cap, a taxi in place
        of a car) and works at native resolution; the video pass then
        propagates the anchored frames (mask 0) instead of inventing the fill
        (audit 2026-09-20). prompts maps keyframe backend -> prompt.
        """
        from preserve.edits.coherence import warp_keyframe_fill
        from preserve.edits.keyframe import get_keyframe_editor

        editor = get_keyframe_editor()
        # Fill the same grown region the video pass regenerates: anchoring a
        # frame whose ring still holds the object's rim propagates that rim
        # to every frame (red cap edge, glasses outline, popsicle stick edge).
        grow_px = int(
            settings.get_replacement_config("vace").get("settings", {}).get("mask_grow_px", 8)
        )
        # The middle frame: VACE drifts with distance from an anchor, so no
        # frame is farther than half the clip from one.
        anchor_index = len(fr) // 2
        region = dilate_mask(mk[anchor_index : anchor_index + 1], grow_px)[0]
        prompt = prompts.get(editor.backend, next(iter(prompts.values())))
        erase = np.isin(mk[anchor_index], ERASE_VALUES).astype(np.uint8)
        rendered = editor.render(
            fr[anchor_index], region, prompt, negative or "", erase_first, erase=erase
        )
        min_carried = float(settings.get_keyframe_config().get("min_carried_ratio", 0.6))
        extension = None
        if replacement and editor.backend == "klein":
            footprint = self._replacement_footprint(rendered, region > 0, replacement)
            if footprint.any():
                from preserve.edits.coherence import warp_mask_between

                extension = np.zeros((len(fr), *region.shape), np.uint8)
                extension[anchor_index][footprint] = 255
                for t in range(len(fr)):
                    if t != anchor_index:
                        carried = warp_mask_between(footprint, fr[anchor_index], fr[t])
                        extension[t][carried & ~(mk[t] > 0)] = 255
                region = np.maximum(region, extension[anchor_index])
                provenance["shape_extension_px"] = int((extension > 0).sum())
        if erase_first and texture_from_ring:
            # A flat surface (fabric under a print) is the same surface as
            # its ring: quilt the ring's fine texture onto the smooth render.
            # Never for anatomy: high-pass tiles from sunglasses and hair
            # edges landed as dark rectangles on a face (capwalk-v24).
            from preserve.edits.coherence import transfer_surface_detail

            ring = (dilate_mask(region[None], 40)[0] > 0) & ~(region > 0)
            rendered = transfer_surface_detail(rendered, fr[anchor_index], region > 0, ring)
        keyframe = fr[anchor_index].copy()
        keyframe[region > 0] = rendered[region > 0]
        self._debug_dump("keyframe", keyframe)
        matte = fr[anchor_index] // 2
        for value, colour in (
            (255, (255, 0, 0)),
            (SHADOW_VALUE, (0, 0, 255)),
            (BACKGROUND_SIDE_VALUE, (0, 255, 0)),
            (INTERACTION_VALUE, (255, 255, 0)),
        ):
            matte[mk[anchor_index] == value] = colour
        self._debug_dump("matte", matte.astype(np.uint8))
        self._debug_dump(
            "keyframe-erase", np.where(erase[..., None] > 0, 255, fr[anchor_index]).astype(np.uint8)
        )
        anchors = {anchor_index: keyframe}
        if bool(settings.get_keyframe_config().get("end_anchors", True)) and len(fr) > 2:
            # First and last frames take the same fill carried along source
            # flow, so the video pass interpolates between consistent anchors.
            for end in (0, len(fr) - 1):
                matte_end = fr[end] // 2
                for value, colour in (
                    (255, (255, 0, 0)),
                    (SHADOW_VALUE, (0, 0, 255)),
                    (BACKGROUND_SIDE_VALUE, (0, 255, 0)),
                    (INTERACTION_VALUE, (255, 255, 0)),
                ):
                    matte_end[mk[end] == value] = colour
                self._debug_dump(f"matte-end{end}", matte_end.astype(np.uint8))
                region_end = dilate_mask(mk[end : end + 1], grow_px)[0] > 0
                if extension is not None:
                    region_end |= extension[end] > 0
                warped, carried = warp_keyframe_fill(
                    keyframe, fr[anchor_index], fr[end], region_end
                )
                # Anchored only where the warp had a source and the pixel
                # moves with the wearer (never a held hand, INTERACTION_VALUE);
                # the rest of the region stays open for the video pass.
                carried &= ~(mk[end] == INTERACTION_VALUE)
                movable = region_end & ~(mk[end] == INTERACTION_VALUE)
                held = (mk[end] == INTERACTION_VALUE).any()
                if held or (movable.any() and carried.sum() < min_carried * movable.sum()):
                    # Flow could not carry the fill this far (fast motion, a
                    # turn): an open end lets the video pass paint the object
                    # back (logo-v12 frame 0). Edit this frame directly.
                    rendered_end = editor.render(
                        fr[end],
                        region_end.astype(np.uint8) * 255,
                        prompt,
                        negative or "",
                        erase_first,
                        erase=np.isin(mk[end], ERASE_VALUES).astype(np.uint8),
                    )
                    # A held hand never carries by flow (it is not the
                    # wearer's surface), so this frame's whole region comes
                    # from the image editor (popsicle-v17 frame 0: a pale
                    # hand blob where the video pass filled it alone).
                    warped = fr[end].copy()
                    warped[region_end] = rendered_end[region_end]
                    carried = region_end.copy()
                    self._debug_dump(f"keyframe-end{end}", warped)
                    self._debug_dump(
                        f"keyframe-end{end}-erase",
                        np.where(np.isin(mk[end], ERASE_VALUES)[..., None], 255, fr[end]).astype(
                            np.uint8
                        ),
                    )
                    provenance["end_keyframes"] = provenance.get("end_keyframes", 0) + 1
                anchors[end] = (warped, ~(region_end & ~carried))
        editor.unload()
        provenance["keyframe_backend"] = editor.backend
        provenance["anchor_indices"] = sorted(anchors)
        return anchors, extension

    @staticmethod
    def _debug_dump(name: str, image: NDArray[np.uint8]) -> None:
        """Write an intermediate to $PRESERVE_DEBUG_DIR/<name>.png when set."""
        import os

        directory = os.environ.get("PRESERVE_DEBUG_DIR")
        if directory:
            os.makedirs(directory, exist_ok=True)
            cv2.imwrite(
                os.path.join(directory, f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )

    @staticmethod
    def _replacement_footprint(
        rendered: NDArray[np.uint8], region: NDArray[np.bool_], replacement: str
    ) -> NDArray[np.bool_]:
        """Pixels of the keyframe proposal that belong to the replacement object
        but lie outside the source matte, bounded to a zone around it."""
        from preserve.groundseg import get_grounder

        threshold = float(settings.get_grounding_config().get("settings", {}).get("threshold", 0.4))
        heat = get_grounder().relevance([rendered], replacement)[0]
        ys, xs = np.nonzero(region)
        reach = int(0.35 * max(ys.max() - ys.min(), xs.max() - xs.min())) + 8
        zone = dilate_mask(region[None].astype(np.uint8) * 255, reach)[0] > 0
        candidate = (heat > threshold) & zone
        candidate = (
            cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            > 0
        )
        # Only the object attached to the source matte, not a second one nearby
        n, labels = cv2.connectedComponents((candidate | region).astype(np.uint8))
        keep = np.isin(labels, np.unique(labels[region])) & candidate
        return keep & ~region

    def _produce_candidate(
        self,
        job: EditJob,
        edit_spec: EditSpec,
        frames: list[NDArray[np.uint8]],
        core_mask: NDArray[np.uint8],
        request,
        video_meta: VideoMetadata | None = None,
        reveal_parent: str | None = None,
        interaction_hand: bool = False,
        prior_anchors: dict[int, NDArray[np.uint8]] | None = None,
    ) -> tuple[list[NDArray[np.uint8]], dict]:
        """Produce candidate pixels for the work region.

        prior_anchors (window index -> finished frame at processing size) come
        from the previous window of a long clip: they replace the keyframe so
        the identity chosen once propagates through every window.

        Routing follows the hierarchy in validated-solution.md 3.3 — the least
        generative method that can express the edit wins. The candidate is never
        authoritative; the caller composites it and hard-restores protected samples.

        reveal_parent names the detected object an attached removal target sits
        on ("person" for a cap); the fill must then show that object's surface.
        """
        scene_prompt = None
        if video_meta is not None and video_meta.generation:
            scene_prompt = video_meta.generation.get("prompt")

        if edit_spec.operation is EditOperation.RECOLOR:
            job.message = f"Recolouring to {edit_spec.new_color}"
            log.info("Using conventional recolour", color=edit_spec.new_color)
            candidates = recolor_sequence(frames, core_mask, edit_spec.new_color)
            return candidates, {"method": "tracked-matte-recolor", "generative": False}

        if edit_spec.operation in (EditOperation.REPLACE, EditOperation.EDIT):
            is_edit = edit_spec.operation is EditOperation.EDIT
            job.message = (
                f"Applying '{edit_spec.instruction}'"
                if is_edit
                else f"Generating {edit_spec.replacement}"
            )
            replacer = get_replacer()
            if getattr(replacer, "prompt_style", "description") == "instruction":
                # An instruction editor takes the user's own sentence.
                prompt = edit_spec.raw_prompt
            else:
                prompt = self._diffusion_prompt(edit_spec, scene_prompt)
            log.info("Using masked diffusion", prompt=prompt, operation=edit_spec.operation.value)
            provenance: dict = {}
            anchor_keyframe = bool(settings.get_keyframe_config().get("enabled", True)) and getattr(
                replacer, "supports_anchors", False
            )

            extension_crop: dict = {}

            def render_replace(fr, mk, box):
                anchors = None
                if prior_anchors:
                    anchors = self._carry_prior_anchors(prior_anchors, fr, mk, box)
                elif anchor_keyframe:
                    keyframe_prompts = {
                        "klein": edit_spec.raw_prompt,
                        "sd_inpaint": self._diffusion_prompt(edit_spec, scene_prompt),
                    }
                    anchors, extension = self._keyframe_anchors(
                        fr,
                        mk,
                        keyframe_prompts,
                        request.negative_prompt,
                        provenance,
                        replacement=None if is_edit else edit_spec.replacement,
                    )
                    if extension is not None:
                        extension_crop["mask"] = extension
                        mk = np.maximum(mk, extension)
                extra = {"anchors": anchors} if anchors else {}
                cands, info = replacer.replace_sequence(
                    fr, mk, prompt, request.negative_prompt, **extra
                )
                provenance.update(info)
                return cands

            candidates, roi = self._render_in_roi(frames, core_mask, render_replace)
            stats = {
                "method": "masked-diffusion-edit" if is_edit else "masked-diffusion-replace",
                "generative": True,
                "provenance": provenance,
                **roi,
            }
            if "mask" in extension_crop:
                # Map the crop-space footprint back to the processing frame.
                full = np.zeros_like(core_mask)
                x0, y0, x1, y1 = roi.get("roi", [0, 0, core_mask.shape[2], core_mask.shape[1]])
                full[:, y0:y1, x0:x1] = extension_crop["mask"]
                stats["region_extension"] = full
            return candidates, stats

        reveal_enabled = bool(settings.get_background_config().get("attached_reveal", True))
        if edit_spec.operation is EditOperation.REMOVE and reveal_parent and reveal_enabled:
            # Attached target: a background remover paints "background" where a
            # mouth or hair belongs (audit 2026-09-19: whitish smear). Prompt the
            # video-native editor with the subject as generated, minus the target.
            job.message = f"Revealing the {reveal_parent} behind {edit_spec.target_phrase}"
            replacer = get_replacer()
            if getattr(replacer, "prompt_style", "description") == "instruction":
                prompt = self._reveal_instruction(edit_spec, reveal_parent)
            else:
                prompt = self._reveal_prompt(edit_spec, scene_prompt, reveal_parent)
            negative = self._reveal_negative(edit_spec, request.negative_prompt)
            log.info("Attached target, using prompted reveal fill", prompt=prompt)
            provenance = {}
            anchor_keyframe = bool(settings.get_keyframe_config().get("enabled", True)) and getattr(
                replacer, "supports_anchors", False
            )

            def render_reveal(fr, mk, box):
                anchors = None
                if prior_anchors:
                    anchors = self._carry_prior_anchors(prior_anchors, fr, mk, box)
                elif anchor_keyframe:
                    keyframe_prompts = {
                        "klein": self._reveal_instruction(
                            edit_spec, reveal_parent, interaction_hand=interaction_hand
                        ),
                        "sd_inpaint": self._keyframe_prompt(edit_spec, scene_prompt, reveal_parent),
                    }
                    # Erase the whole object before the editor sees the frame:
                    # with the brim still in its reference klein repainted the
                    # brim's shadow inside the fill whatever the band erase did
                    # (cap-v22 dumps). The instruction and the erased surround
                    # carry what to reveal.
                    anchors, _ = self._keyframe_anchors(
                        fr,
                        mk,
                        keyframe_prompts,
                        negative,
                        provenance,
                        erase_first=True,
                        texture_from_ring=is_flat_surface_target(edit_spec.target_phrase)
                        and bool(settings.get_keyframe_config().get("texture_quilt", False)),
                    )
                extra = {"anchors": anchors} if anchors else {}
                cands, info = replacer.replace_sequence(
                    fr, mk, prompt, negative, task="reveal", **extra
                )
                provenance.update(info)
                return cands

            candidates, roi = self._render_in_roi(frames, core_mask, render_reveal)
            return candidates, {
                "method": "masked-diffusion-reveal",
                "generative": True,
                "reveal_parent": reveal_parent,
                "provenance": provenance,
                **roi,
            }

        # REMOVE operation: try background plate first; if registration is unreliable
        # (generated/morphing footage), fall back to VACE which conditions on the
        # real surroundings rather than hallucinating unconditioned.
        job.message = "Reconstructing occluded background"
        config = settings.get_background_config()
        # Same mask-aware registration as the plate builder: score the
        # background fit, not the morphing subject, or every removal on
        # generated footage routes to diffusion by construction.
        routing_exclude = [m for m in (core_mask > 0)]
        transforms = estimate_frame_transforms(frames, exclude_masks=routing_exclude)
        residual = registration_residual(frames, transforms, exclude_masks=routing_exclude)
        threshold = float(config.get("max_registration_error", 18.0))
        # Only the propagation backend needs the VACE detour: it can only copy
        # pixels it can register. A generative remover (MiniMax) conditions on
        # the surroundings itself, and unlike VACE was trained not to
        # regenerate the masked object, so it owns removal on any footage.
        if residual > threshold and settings.inpaint_backend == "propainter":
            # Morphing footage: propagation cannot register, so fill with the
            # video-native model conditioned on the real surroundings instead of
            # letting ProPainter hallucinate unconditioned pixels.
            replacer = get_replacer()
            prompt = self._removal_prompt(edit_spec, scene_prompt)
            negative = self._removal_negative(request.negative_prompt)
            log.info(
                "Registration unreliable, using VACE for removal",
                residual=round(residual, 1),
                threshold=threshold,
                prompt=prompt,
            )
            provenance = {}

            def render_remove(fr, mk, box):
                cands, info = replacer.replace_sequence(fr, mk, prompt, negative, task="remove")
                provenance.update(info)
                return cands

            candidates, roi = self._render_in_roi(frames, core_mask, render_remove)
            return candidates, {
                "method": "masked-diffusion-remove",
                "generative": True,
                "registration_residual": round(residual, 1),
                "provenance": provenance,
                **roi,
            }

        backend = self.get_backend()
        log.info("Using inpainting backend", backend=backend.name, residual=round(residual, 1))
        metadata: dict = {}

        def render_inpaint(fr, mk, box):
            result = backend.inpaint(
                InpaintRequest(
                    frames=fr,
                    masks=mk,
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    denoise_strength=request.denoise_strength,
                    seed=request.seed,
                )
            )
            metadata.update(result.metadata or {})
            return result.frames

        if settings.inpaint_backend == "propainter":
            # Propagation needs the whole frame: its donors live outside any crop.
            candidates, roi = render_inpaint(frames, core_mask, None), {}
            method = "temporal-propagation-remove"
        else:
            candidates, roi = self._render_in_roi(frames, core_mask, render_inpaint)
            method = "masked-diffusion-remove"
        return candidates, {
            "method": method,
            "generative": True,
            "registration_residual": round(residual, 1),
            "provenance": metadata,
            **roi,
        }

    @staticmethod
    def _removal_prompt(edit_spec: EditSpec, scene_prompt: str | None) -> str:
        """Build a descriptive prompt for object removal via VACE.
        When registration fails on morphing footage, VACE conditioned on the
        surrounding context fills the hole more plausibly than ProPainter's
        unconditioned fill. The prompt should describe the output scene:
        the surroundings without the object.
        """
        if scene_prompt:
            lower = scene_prompt.lower()
            surface = next(
                (
                    w
                    for w in (
                        "highway",
                        "road",
                        "street",
                        "track",
                        "runway",
                        "field",
                        "desert",
                        "floor",
                        "ground",
                    )
                    if w in lower
                ),
                None,
            )
            tail = scene_prompt.split(",", 1)
            tail = tail[1].strip() if len(tail) > 1 else ""
            parts = []
            if surface:
                parts.append(f"empty {surface} with no car, no vehicle, no object")
            if tail:
                parts.append(tail)
            if parts:
                return ", ".join(parts)
        # Never name the target here: an inpainting model fills WITH the
        # positive prompt, so "blue cars" paints blue cars. The target lives
        # in the negative prompt only (audit: blue ghost blob, 2026-09-04).
        return (
            "empty scene, continuous background with no vehicle, no object, "
            "matching lighting, perspective and camera motion, photorealistic"
        )

    @staticmethod
    def _reveal_surface(edit_spec: EditSpec, parent: str) -> str:
        return reveal_surface(edit_spec.target_phrase, parent)

    @classmethod
    def _reveal_prompt(cls, edit_spec: EditSpec, scene_prompt: str | None, parent: str) -> str:
        """Describe the subject as generated, minus the attached target.

        "a young boy wearing a red baseball cap, standing in a park" with target
        "cap" becomes "a young boy, standing in a park, the person without any
        cap ...": the clause that introduces the target is cut at the participle
        so the subject survives, and clauses that still name it are dropped.
        """
        noun = (edit_spec.target_phrase or "object").split()[-1].lower()
        kept: list[str] = []
        for clause in re.split(r",\s*", scene_prompt or ""):
            trimmed = re.sub(
                rf"\b(wearing|holding|with|eating|carrying|in)\b[^,]*?\b{re.escape(noun)}s?\b",
                "",
                clause,
                flags=re.IGNORECASE,
            ).strip(" .")
            if trimmed and noun not in trimmed.lower():
                kept.append(trimmed)
        head = ", ".join(kept) or f"a {parent}"
        surface = cls._reveal_surface(edit_spec, parent)
        return (
            f"{head}, the {parent} without any {noun}, the {parent}'s {surface} "
            f"fully visible where the {noun} was, same lighting, photorealistic, sharp"
        )

    @classmethod
    def _keyframe_prompt(cls, edit_spec: EditSpec, scene_prompt: str | None, parent: str) -> str:
        """Region-focused description for a masked keyframe fill.

        A masked inpainter paints what the positive prompt names, so the
        target noun never appears (the reveal prompt's "without any cap" put
        a white cap back, audit 2026-09-20); the scene's lighting clause is
        kept so the fill matches the exposure.
        """
        surface = cls._reveal_surface(edit_spec, parent)
        lighting = next(
            (
                clause.strip()
                for clause in re.split(r",\s*", scene_prompt or "")
                if re.search(r"light|sun|golden|overcast|night|studio", clause, re.IGNORECASE)
            ),
            "natural light",
        )
        return (
            f"close-up photo of a {parent}'s {surface}, bare, nothing worn, "
            f"{lighting}, photorealistic, sharp, natural skin and hair texture"
        )

    @classmethod
    def _reveal_instruction(
        cls, edit_spec: EditSpec, parent: str, interaction_hand: bool = False
    ) -> str:
        """Instruction-style reveal for editors that take commands (Lucy, klein)."""
        target = edit_spec.target_phrase or "the object"
        surface = cls._reveal_surface(edit_spec, parent)
        hand = (
            " and the hand that was holding it, showing what is behind the hand"
            if interaction_hand
            else ""
        )
        return (
            f"Remove {target}{hand} completely, showing the {parent}'s {surface} underneath, "
            f"natural and photorealistic. Keep the {parent}'s face, pose, clothes, "
            "lighting and the background exactly the same."
        )

    @staticmethod
    def _reveal_negative(edit_spec: EditSpec, request_negative: str | None) -> str:
        noun = (edit_spec.target_phrase or "object").split()[-1].lower()
        extra = f"{noun}, {noun}s, hat, accessory, object in front"
        return f"{request_negative}, {extra}" if request_negative else extra

    @staticmethod
    def _removal_negative(request_negative: str | None) -> str:
        """Object terms that suppress VACE reinsertion (shared by the removal
        fallback and the unknown-pixel cascade so both speak identically)."""
        removal_neg = "car, vehicle, automobile, taxi, truck, person, object"
        return f"{request_negative}, {removal_neg}" if request_negative else removal_neg

    @staticmethod
    def _diffusion_prompt(edit_spec: EditSpec, scene_prompt: str | None = None) -> str:
        """Build a diffusion prompt describing the desired contents of the region.

        For REPLACE, VACE's inpainting mode works best with a prompt describing
        only the replacement subject — the unmasked src_video already carries
        the scene context. Over-specifying the surroundings in the prompt can
        cause the model to reimagine the background instead of copying it from
        the source, increasing temporal drift.
        """
        if edit_spec.operation is EditOperation.EDIT:
            desired = edit_spec.instruction or edit_spec.raw_prompt
            return (
                f"{desired}, matching the original lighting, perspective, "
                "shadows and camera motion, photorealistic, sharp"
            )
        replacement = edit_spec.replacement or "the same scene"
        # Result clauses usually carry their own article ("a yellow taxi").
        replacement = re.sub(r"^(?:a|an|the)\s+", "", replacement)
        # Use minimal prompt for REPLACE: the src_video conditions the surroundings.
        return f"a {replacement}"

    def _produce_candidate_windowed(
        self,
        job: EditJob,
        edit_spec: EditSpec,
        frames: list[NDArray[np.uint8]],
        core_mask: NDArray[np.uint8],
        request,
        video_meta: VideoMetadata | None = None,
        reveal_parent: str | None = None,
        interaction_hand: bool = False,
    ) -> tuple[list[NDArray[np.uint8]], dict]:
        """Produce candidate with windowed processing for long videos.

        Uses VideoPainter-style overlapping windows with ID resampling for
        clips exceeding model's native context window.
        """
        # Check if we need windowed processing
        # Both video-native backends (VACE, MiniMax-Remover) take 81 frames in
        # one pass; splitting shorter clips into 24-frame windows re-rendered
        # the overlap twice and blended it (audit 2026-09-19: 32-frame clip
        # went through two windows for no reason).
        native_window = int(
            settings.get_replacement_config("vace").get("settings", {}).get("max_window_frames", 81)
        )

        if len(frames) <= native_window:
            return self._produce_candidate(
                job,
                edit_spec,
                frames,
                core_mask,
                request,
                video_meta,
                reveal_parent,
                interaction_hand,
            )

        log.info(
            "Using windowed processing for long video", frames=len(frames), window=native_window
        )

        anchored_ops = edit_spec.operation in (EditOperation.REPLACE, EditOperation.EDIT) or (
            edit_spec.operation is EditOperation.REMOVE and reveal_parent
        )
        if anchored_ops and getattr(get_replacer(), "supports_anchors", False):
            return self._produce_candidate_sequential(
                job,
                edit_spec,
                frames,
                core_mask,
                request,
                video_meta,
                reveal_parent,
                interaction_hand,
                native_window,
            )

        # Configure windowed processing
        window_config = WindowConfig(
            window_size=native_window,
            overlap=native_window // 3,
            id_reference_stride=native_window // 2,
        )

        def process_window(window_frames: list[NDArray[np.uint8]], window_masks: NDArray[np.uint8]):
            # Create a temporary job for this window
            from copy import deepcopy

            temp_job = deepcopy(job)
            temp_job.message = f"Processing window {len(window_frames)} frames"
            # _produce_candidate returns (frames, stats); the windowed runner
            # consumes frame lists only (a tuple here crashed long clips).
            window_candidates, _ = self._produce_candidate(
                temp_job,
                edit_spec,
                window_frames,
                window_masks,
                request,
                video_meta,
                reveal_parent,
                interaction_hand,
            )
            return window_candidates

        # Process with windowed approach
        candidate_frames = process_long_video(
            frames, core_mask, process_window, window_config, id_resample_strength=0.3
        )
        method_stats = {
            "method": "windowed-candidate",
            "generative": edit_spec.operation is not EditOperation.RECOLOR,
            "windowed_processing": True,
            "num_windows": (len(frames) - 1) // (native_window - native_window // 3) + 1,
        }
        return candidate_frames, method_stats

    def _produce_candidate_sequential(
        self,
        job: EditJob,
        edit_spec: EditSpec,
        frames: list[NDArray[np.uint8]],
        core_mask: NDArray[np.uint8],
        request,
        video_meta: VideoMetadata | None,
        reveal_parent: str | None,
        interaction_hand: bool,
        window: int,
    ) -> tuple[list[NDArray[np.uint8]], dict]:
        """Long clips for the anchored backends: windows run in order, and each
        window after the first is anchored on the previous window's finished
        overlap frames (mask 0), so the keyframe made once for the first window
        is what every later window continues. No cross-window blend is needed:
        anchored frames come back unchanged."""
        overlap = max(3, window // 4)
        step = window - overlap
        outputs: list[NDArray[np.uint8] | None] = [None] * len(frames)
        start = 0
        windows = 0
        while start < len(frames):
            end = min(len(frames), start + window)
            if end - start < overlap + 2 and start > 0:
                start = max(0, len(frames) - window)
                end = len(frames)
            prior = None
            if start > 0:
                prior = {
                    j: outputs[start + j]
                    for j in range(min(overlap, end - start))
                    if outputs[start + j] is not None
                }
            job.message = f"Window {windows + 1}: frames {start}-{end - 1}"
            candidates, _ = self._produce_candidate(
                job,
                edit_spec,
                frames[start:end],
                core_mask[start:end],
                request,
                video_meta,
                reveal_parent,
                interaction_hand,
                prior_anchors=prior,
            )
            for j, candidate in enumerate(candidates):
                if outputs[start + j] is None:
                    outputs[start + j] = candidate
            windows += 1
            if end == len(frames):
                break
            start += step
        return [o if o is not None else f for o, f in zip(outputs, frames, strict=True)], {
            "method": "windowed-anchored",
            "generative": True,
            "windowed_processing": True,
            "num_windows": windows,
        }

    def process(self, job: EditJob, video_meta: VideoMetadata) -> EditJob:
        """Execute the full editing pipeline."""
        request = job.request
        pipeline_config = settings.get_pipeline_config()
        log.info("Starting edit pipeline", job_id=str(job.id), edit_type=request.edit_type)

        try:
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow()
            job.message = "Extracting frames"
            job.progress = 0.1

            max_frames = int(pipeline_config.get("max_frames", 60))
            processing_width = int(pipeline_config.get("processing_width", 960))

            start_frame, end_frame = frame_range(
                request.time_range.start_ms,
                request.time_range.end_ms,
                video_meta.fps,
                video_meta.frame_count,
                video_meta.duration_ms,
            )
            total_frames = end_frame - start_frame

            step = max(1, total_frames // max_frames)
            source_frames = extract_frames(
                video_meta.path,
                start_frame=start_frame,
                end_frame=end_frame,
                step=step,
            )
            log.info("Extracted frames", count=len(source_frames), step=step)

            orig_h, orig_w = source_frames[0].shape[:2]
            processing_frames, scale = self._resize_frames(source_frames, processing_width)
            proc_h, proc_w = processing_frames[0].shape[:2]

            if scale < 1.0:
                log.info(
                    "Resized for processing",
                    original=f"{orig_w}x{orig_h}",
                    processing=f"{proc_w}x{proc_h}",
                )

            job.progress = 0.15
            job.message = "Interpreting edit"

            # Classify before masking: the operation decides which backend runs,
            # and for a recolour the requested output colour must not be treated
            # as a filter on which objects to select.
            edit_spec = classify_edit(request.prompt or request.mask.segmentation_prompt or "")
            log.info(
                "Edit classified", operation=edit_spec.operation.value, spec=edit_spec.describe()
            )

            job.progress = 0.2
            job.message = f"Locating {edit_spec.target_phrase}"

            frame_count = len(processing_frames)
            mask_definition = request.mask
            if mask_definition.mask_type == MaskType.SEGMENTATION:
                # Feed the classifier's cleaned target to the segmenter rather than
                # the raw sentence.
                mask_definition = mask_definition.model_copy(
                    update={
                        "segmentation_prompt": (
                            edit_spec.target.raw_prompt or edit_spec.target_phrase
                        )
                    }
                )
            # Generative ops (replace/edit/remove) need room for both the old
            # object residue and the new shape (validated-solution 4): the audit
            # showed a red roof surviving because YOLO+SAM stopped at the window
            # line and the tight matte protected it. 8px matches VACE mask_grow
            # so generation and composite regions coincide (no VAE halo ring).
            # 12px was tried and rejected (pink blend band, color 8.3 -> 6.3).
            # Recolour stays tight: it must not extend past the object.
            if (
                edit_spec.operation
                in (
                    EditOperation.REPLACE,
                    EditOperation.EDIT,
                )
                or edit_spec.operation is EditOperation.REMOVE
            ):
                mask_definition = mask_definition.model_copy(
                    update={"dilation_px": max(mask_definition.dilation_px or 0, 8)}
                )
            hug_object = edit_spec.operation is not EditOperation.REMOVE
            core_mask, alpha, keepout, mask_stats = process_mask(
                mask_definition,
                width=proc_w,
                height=proc_h,
                frame_count=frame_count,
                frames=processing_frames,
                target=edit_spec.target,
                hug_object=hug_object,
                target_phrase=edit_spec.target_phrase,
            )
            occluders_proc = mask_stats.pop("_occluders", None)
            attached_parent = mask_stats.get("attachment_parent")
            mask_stats["operation"] = edit_spec.operation.value
            mask_stats["edit"] = edit_spec.describe()
            mask_coverage = float(np.mean(core_mask > 0) * 100)
            # mask_stats may already carry a coverage_pct (from text grounding), so
            # log the edit-region coverage under a distinct key to avoid a collision.
            log.info("Mask created", edit_coverage_pct=round(mask_coverage, 2), **mask_stats)
            if mask_stats.get("instances_matched", 1) == 0:
                # Could not locate the subject anywhere in the clip. Fail closed
                # rather than touch the whole frame — preserving the surroundings
                # is the entire guarantee, so a localized edit with no location to
                # apply to is declined, not silently widened.
                subject = edit_spec.target_phrase or mask_stats.get("spec", "that")
                job.status = JobStatus.FAILED
                job.error = (
                    f"Couldn't find '{subject}' in this video to edit. "
                    "Try naming it differently, or regenerate the video instead."
                )
                log.warning("Subject not located", error=job.error)
                return job
            # Use the overridden definition so allowed matches the alpha region.
            dilation_px = mask_definition.dilation_px or pipeline_config.get(
                "default_dilation_px", 5
            )
            allowed = compute_allowed_region(core_mask, dilation_px, keepout)
            job.progress = 0.3
            candidate_frames, method_stats = self._produce_candidate_windowed(
                job,
                edit_spec,
                processing_frames,
                core_mask,
                request,
                video_meta,
                attached_parent,
                bool(mask_stats.get("interaction_hand_pixels")),
            )
            extension = method_stats.pop("region_extension", None)
            if extension is not None and (extension > 0).any():
                # The replacement's footprint beyond the source matte (a
                # taller taxi) is part of the requested change: grow the
                # authorized region and its alpha to cover it.
                core_mask = np.maximum(core_mask, extension)
                allowed = compute_allowed_region(core_mask, dilation_px, keepout)
                alpha = np.maximum(
                    alpha,
                    feather_mask(
                        dilate_mask(extension, dilation_px),
                        mask_definition.feather_px or pipeline_config.get("default_feather_px", 3),
                    ),
                )
                mask_stats["shape_extension_pct"] = round(
                    100 * int((extension > 0).sum()) / max(1, int((core_mask > 0).sum())), 1
                )
                log.info(
                    "Edit region grown to the replacement's footprint",
                    pct=mask_stats["shape_extension_pct"],
                )
            mask_stats.update(method_stats)
            # In-region finishing (VideoPainter/InVi): affine-match the interior
            # seam to the source ring so VAE boundary discontinuity does not
            # read as a halo. Only touches the allowed region; the hard-restore
            # after compositing still owns the exterior.
            if method_stats.get("generative", False):
                # Removal fills are the surrounding surface, so the whole
                # interior follows the ring (interior=True). A reveal fill is
                # a different surface (hair between skin and sky) and must not
                # be pulled toward its ring: seam band only.
                # A flat-surface reveal (a print on fabric) is the same
                # surface as its ring, so it takes the interior harmonisation
                # too (audit 2026-09-21: bright disc where the logo was).
                flat = is_flat_surface_target(edit_spec.target_phrase)
                reveal = bool(method_stats.get("reveal_parent"))
                # A reveal takes a short fading seam band only: pulling the
                # fill toward its ring at full weight dragged a revealed
                # forehead into the brow shadow below it (cap-v27/v28), and
                # a 12px fade showed as a strip; 8px reads as a soft edge.
                candidate_frames = match_boundary_color(
                    candidate_frames,
                    processing_frames,
                    allowed,
                    band_px=8 if reveal else 5,
                    interior=edit_spec.operation is EditOperation.REMOVE and (not reveal or flat),
                )
                self._debug_dump("harmonised", candidate_frames[len(candidate_frames) // 2])
            job.progress = 0.7
            job.message = "Compositing with preservation"
            # Composite at the original resolution so untouched pixels are the
            # source's own full-detail samples. Compositing at processing size and
            # upscaling afterwards softened the entire frame and verified the
            # preservation guarantee at a resolution that was never delivered.
            # Generative composites use a 2px blend band (not 3): the audit halo
            # was original red rim mixed with generated fill across a 3px feather.
            feather_px = mask_definition.feather_px or pipeline_config.get("default_feather_px", 3)
            if method_stats.get("reveal_parent"):
                # A reveal seam runs across skin: a wider blend (still inside
                # the 8px grown region) hides the tonal step a 2px band shows
                # as a jagged line along the old brim (audit 2026-09-20).
                feather_px = 5
            elif method_stats.get("generative", False):
                feather_px = min(feather_px, 2)

            if scale < 1.0:
                target_size = (orig_w, orig_h)
                generated = self._upscale_frames(candidate_frames, target_size)
                composite_sources = source_frames
                (
                    composite_alpha,
                    composite_allowed,
                    edit_mask,
                    occupancy,
                ) = self._build_full_res_regions(
                    core_mask,
                    keepout,
                    target_size,
                    dilation_px,
                    feather_px,
                    scale,
                    processing_frames,
                    alpha,
                )
            else:
                generated = candidate_frames
                composite_sources = processing_frames
                composite_alpha, composite_allowed = alpha, allowed
                edit_mask = (composite_allowed * 255).astype(np.uint8)
                occupancy = dilate_mask(core_mask, 12)
                if keepout is not None:
                    occupancy = np.maximum(occupancy, keepout.astype(np.uint8) * 255)

            wants_background = (
                edit_spec.operation is EditOperation.REMOVE
                and settings.get_background_config().get("enabled", True)
            )
            if wants_background:
                job.message = "Reconstructing background from other frames"
                scene_prompt = None
                if video_meta is not None and video_meta.generation:
                    scene_prompt = video_meta.generation.get("prompt")
                # Every detected instance of any class is donor-occupied, not
                # just same-class keepout: a target with no detector class (a
                # cap) otherwise lets the wearer's head pixels from other
                # frames pass as "background" (audit: whitish smear on the
                # head after cap removal).
                if occluders_proc is not None:
                    occluders = self._upscale_masks(
                        occluders_proc.astype(np.uint8) * 255,
                        (composite_sources[0].shape[1], composite_sources[0].shape[0]),
                    )
                    # Instance masks stop short of hair and skirts; grown like
                    # the keepout, or those pixels donate into the plate as a
                    # dark contour along the person (audit 2026-09-21).
                    grow = int(settings.get_background_config().get("keepout_grow_px", 8) / scale)
                    if grow > 0:
                        occluders = dilate_mask(occluders, grow)
                    occupancy = np.maximum(occupancy, occluders)
            if wants_background and attached_parent:
                # An attached object moves with its wearer, so the region it
                # hides (wearer's surface and the background beyond it) is
                # occluded in every frame: no donor is valid. A plate here
                # returned the cap's own defocused edge as "background"
                # (audit 2026-09-21: dark band). Only the prompted fill applies.
                log.info(
                    "Background reconstruction skipped: target attached to another object",
                    parent=attached_parent,
                    attachment_ratio=mask_stats.get("attachment_ratio"),
                )
                mask_stats["background_skipped"] = "attached-object"
            elif wants_background:
                generated, background_stats = self._reconstruct_background(
                    composite_sources,
                    edit_mask,
                    occupancy,
                    generated,
                    refine_prompt=self._removal_prompt(edit_spec, scene_prompt),
                    refine_negative=self._removal_negative(request.negative_prompt),
                )
                mask_stats.update(background_stats)
                log.info("Background reconstruction", **background_stats)

            # Build color metadata for compositing
            # Use ffprobe data from video_meta if available, otherwise defaults.
            # Extracted frames are RGB24 full range (0-255), so override color_range.
            color_meta = ColorMetadata()
            if video_meta.ffprobe_data:
                color_meta = create_color_metadata_from_ffprobe(video_meta.ffprobe_data)
                color_meta.color_range = "full"

            # Use legacy sRGB-space compositing to guarantee exact pixel preservation.
            # Linear compositing with color management introduces rounding errors
            # that break the 0-tolerance preservation guarantee.
            output_frames = composite_sequence(
                sources=composite_sources,
                generated=generated,
                alphas=composite_alpha,
                allowed=composite_allowed,
                source_meta=color_meta,
                generated_meta=color_meta,
                out_meta=color_meta,
                working_space="linear_srgb",
                use_linear_compositing=False,
            )
            job.progress = 0.85

            job.message = "Verifying preservation"
            verification_config = pipeline_config.get("verification", {})
            tolerance = verification_config.get("tolerance", 0)

            report = verify_sequence_preservation(
                outputs=output_frames,
                sources=composite_sources,
                protected=compute_protected_region(composite_allowed),
                job_id=job.id,
                edited_frame_indices=list(range(frame_count)),
                tolerance=tolerance,
            )

            if not report.passed:
                log.error(
                    "Preservation verification failed",
                    changed_pixels=report.changed_pixels_outside_mask,
                    max_diff=report.max_diff_outside_mask,
                )
                job.status = JobStatus.FAILED
                job.error = (
                    f"Preservation failed: {report.changed_pixels_outside_mask} "
                    "pixels changed outside mask"
                )
                return job

            log.info("Preservation verified", passed=report.passed)
            job.progress = 0.9
            job.message = "Writing output"

            settings.ensure_dirs()
            # Sampling every stepth frame keeps total duration unchanged, so the
            # original audio track still lines up.
            effective_fps = video_meta.fps / step

            # Two artefacts, as validated-solution.md 8.2/8.3 prescribe. The master
            # is lossless, so decoding it returns the composited samples exactly and
            # the preservation guarantee is demonstrable on a delivered file. The
            # derivative is an ordinary H.264 encode for playback and download, and
            # is explicitly perceptual — a lossy encode changes samples everywhere,
            # so it can never serve as the preservation proof.
            master_path = settings.output_dir / f"{job.id}.master.mp4"
            output_path = settings.output_dir / f"{job.id}.mp4"

            write_lossless_master(
                output_frames, master_path, effective_fps, audio_source=video_meta.path
            )
            write_playable(output_frames, output_path, effective_fps, audio_source=video_meta.path)

            # The invariant is only proved if it holds on the delivered file, not
            # just on pre-encode arrays (validated-solution.md 8.2/10.3). The
            # master was written losslessly so its decode should be bit-exact;
            # this closes the loop by actually decoding it back and re-checking.
            job.message = "Verifying delivered master"
            master_frames = extract_frames(master_path)
            if len(master_frames) != len(output_frames):
                job.status = JobStatus.FAILED
                job.error = (
                    f"Master decode returned {len(master_frames)} frames, "
                    f"expected {len(output_frames)}"
                )
                return job

            master_report = verify_sequence_preservation(
                outputs=master_frames,
                sources=composite_sources,
                protected=compute_protected_region(composite_allowed),
                job_id=job.id,
                edited_frame_indices=list(range(frame_count)),
                tolerance=tolerance,
            )
            if not master_report.passed:
                log.error(
                    "Post-decode verification failed on master",
                    changed_pixels=master_report.changed_pixels_outside_mask,
                    max_diff=master_report.max_diff_outside_mask,
                )
                job.status = JobStatus.FAILED
                job.error = (
                    "Delivered master failed post-decode preservation: "
                    f"{master_report.changed_pixels_outside_mask} pixels changed outside mask"
                )
                return job

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 1.0
            # Persist the exact allowed region this output was composited and
            # verified against. SAM/YOLO re-derivation is nondeterministic
            # across processes (MPS), so an independent audit must judge the
            # master against THIS mask, not a re-derived lookalike whose
            # fringe pixels differ by ~200px and false-fail the gate.
            # (validated-solution.md 4: store final mattes as versioned assets.)
            allowed_path = settings.output_dir / f"{job.id}.allowed.npy"
            np.save(allowed_path, np.ascontiguousarray(composite_allowed))
            job.result = {
                "output_path": str(output_path),
                "master_path": str(master_path),
                "allowed_path": str(allowed_path),
                "master_is_lossless": True,
                "master_verified": master_report.passed,
                "preservation_report": report.model_dump(),
                "mask_stats": mask_stats,
                "mask_coverage_pct": round(mask_coverage, 2),
            }
            log.info(
                "Edit completed",
                job_id=str(job.id),
                master=str(master_path),
                derivative=str(output_path),
            )

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            log.exception("Pipeline failed", error=str(e))

        return job
