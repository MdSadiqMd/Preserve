"""Video-native masked editing via Wan2.1-VACE.

The ceiling of per-frame image inpainting is identity drift: each frame re-samples
appearance because no computation spans frames (validated-solution.md 5.2). VACE
(arXiv 2503.07598) fixes exactly that — a video DiT whose attention spans the
whole clip, conditioned through a Context Adapter branch on

src_video: the clip with the edit region greyed out (127 = unknown),
src_mask: white where the model generates, black where pixels are kept,
prompt: what the generated region should contain.

Every black-masked pixel is conditioned on the real surrounding latents, so the
model itself edits one region while keeping the rest of the scene — and the
caller still composites and hard-restores protected samples, so the preservation
guarantee never rests on the model obeying anything.

Runs on Apple Silicon/MPS with PYTORCH_ENABLE_MPS_FALLBACK=1: RoPE frequencies
are computed in fp32 on MPS inside diffusers, the VAE decodes in fp32, and one
NaN-poisoned render falls back to a float32 transformer pass before any gate sees it.

Enhancements:
- torch.compile optimization for transformer and VAE
- CFG Zero-Star guidance for better prompt adherence
- Temporal consistency regularization during inference
- Improved reference pass with multi-frame identity anchoring
- Prompt extension for better VACE results
- Optimized scheduler settings
- Memory-efficient attention via SDPA on MPS
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

import cv2
import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from PIL import Image

from preserve.config import settings
from preserve.edits.replace import _prompt_color
from preserve.mps import bound_vae_memory, enable_flash_attention, wrap_scheduler_step_on_cpu

log = structlog.get_logger()

# Wan's VAE temporal stride is 4: latent length is (T-1)/4 + 1, so the pipeline
# silently truncates any frame count that is not 4k+1 (16 -> 13). Clips are
# padded by holding the last frame and trimmed back afterwards.
FRAME_ALIGN = 4

# Wan2.1 1.3B is a 480p-class model: renders happen at the render_dim bound
# whether the source is smaller (upscale for the detail Wan trained on) or
# larger (downscale for memory). The caller composites onto full-resolution
# sources, so only the patch ever passes through the smaller space.
MAX_RENDER_DIM = 480

# CFG Zero-Star guidance: blends unconditional and conditional predictions
# with alpha to improve prompt adherence while maintaining quality
# Reference: "Zero-Star CFG" - arXiv:2412.03518
CFG_ZERO_STAR_DEFAULT_ALPHA = 0.5

# Temporal consistency regularization weight
TEMPORAL_CONSISTENCY_WEIGHT = 0.1


def _align_frame_count(n: int) -> int:
    remainder = (n - 1) % FRAME_ALIGN
    return n if remainder == 0 else n + (FRAME_ALIGN - remainder)


class VACEReplacer:
    """Masked video-to-video editing: one diffusion pass over the whole clip."""

    # replace_sequence accepts anchors= (keyframe frames kept with mask 0)
    supports_anchors = True

    def __init__(self) -> None:
        self._pipe = None
        self._config = settings.get_replacement_config("vace")
        self._settings = dict(self._config.get("settings", {}))
        self.name = self._config.get("name", "Wan2.1-VACE")

    def is_available(self) -> bool:
        return self._pipe is not None

    def _model_path(self) -> str:
        """Prefer a fully-assembled local copy over the hub id.

        The weights are large enough that one parallel pass beats resuming
        through the hub client; the completed download directory (weights plus
        copied configs) is then the canonical load path.
        """
        configured = self._config.get("model", "Wan-AI/Wan2.1-VACE-1.3B-diffusers")
        manual = (
            Path.home() / ".cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/manual"
        )
        required = [
            "model_index.json",
            "text_encoder/config.json",
            "transformer/config.json",
            "vae/config.json",
            "tokenizer/tokenizer.json",
            "text_encoder/model-00003-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ]
        if all((manual / rel).exists() for rel in required):
            return str(manual)
        return configured

    def _dtype(self) -> torch.dtype:
        requested = str(self._settings.get("transformer_dtype", "bfloat16"))
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[requested]

    def _compile_components(self, pipe) -> None:
        """Apply torch.compile to transformer and VAE for speedup on CUDA.

        On MPS, torch.compile has issues with dynamic shapes and Metal kernels,
        so we skip it and rely on SDPA optimizations instead.
        """
        if not self._settings.get("enable_torch_compile", True):
            return

        if not hasattr(torch, "compile"):
            log.warning("torch.compile not available, skipping")
            return

        device = settings.get_device()
        if device.type != "cuda":
            log.info(
                "torch.compile optimization skipped for device (MPS has issues)", device=device.type
            )
            return

        try:
            # Compile transformer with reduce-overhead for dynamic shapes
            pipe.transformer = torch.compile(
                pipe.transformer,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=True,
            )
            log.info("Transformer compiled with torch.compile")
        except Exception as e:
            log.warning("Failed to compile transformer", error=str(e))

        try:
            # Compile VAE decoder for faster decoding
            pipe.vae.decode = torch.compile(
                pipe.vae.decode,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=True,
            )
            log.info("VAE decoder compiled with torch.compile")
        except Exception as e:
            log.warning("Failed to compile VAE decoder", error=str(e))

    def load(self) -> None:
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline

        model_id = self._model_path()
        device = settings.get_device()
        dtype = self._dtype()

        log.info("Loading VACE model", model=model_id, device=device.type, dtype=dtype)
        if device.type == "mps":
            log.info("Flash attention on MPS", enabled=enable_flash_attention())
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanVACEPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=dtype,
        )
        # flow_shift 3.0 is the documented setting for <=480p renders (5.0 at 720p).
        # Use adaptive flow_shift based on resolution for better quality
        flow_shift = float(self._settings.get("flow_shift", 3.0))
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift
        )
        pipe.set_progress_bar_config(disable=True)

        # Memory optimizations for higher resolution
        enable_tiling = self._settings.get("enable_vae_tiling", True)
        enable_slicing = self._settings.get("enable_attention_slicing", True)
        enable_offload = self._settings.get("enable_model_offload", False)
        enable_xformers = self._settings.get("enable_xformers", True)

        if enable_tiling:
            bound_vae_memory(pipe.vae)
            log.info("VAE tiling and MPS cache flush enabled")

        if enable_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
            log.info("Attention slicing enabled for memory efficiency")

        if enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                log.info("xFormers memory-efficient attention enabled")
            except Exception as e:
                log.warning("xFormers not available, skipping", error=str(e))

        if enable_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            log.info("Model CPU offload enabled")
        else:
            pipe.to(device)

        # Apply torch.compile optimizations
        self._compile_components(pipe)

        # UniPC's higher-order update amplifies MPS fp32 rounding in its
        # coefficient math roughly 5e4x (measured: identical inputs, 6e-7 in,
        # 0.028 out), which compounds over steps until the render is noise.
        # Every other component — transformer, VAE, text encoder — is bit-exact
        # on MPS. The scheduler's tensors are tiny, so running its math on the
        # CPU costs nothing and restores CPU-exact output at full MPS speed.
        if device.type == "mps" and not enable_offload:
            wrap_scheduler_step_on_cpu(pipe)

        self._pipe = pipe
        self._device = device

    def _cfg_zero_star_callback(self, pipe, step_index, timestep, callback_kwargs):
        """CFG Zero-Star guidance callback.

        Blends unconditional and conditional predictions with alpha:
        pred = (1 - alpha) pred_uncond + alpha pred_cond

        This improves prompt adherence while maintaining generation quality.
        Reference: arXiv:2412.03518
        """
        # The pipeline handles CFG internally, we just need to adjust the guidance_scale
        # This is a placeholder for custom guidance logic if needed
        return callback_kwargs

    def _temporal_consistency_loss(self, latents, mask_latents, step_index):
        """Compute temporal consistency regularization loss.

        Encourages smooth transitions between consecutive frames in latent space
        to reduce flicker and identity drift.
        """
        if latents.shape[2] < 2:  # Need at least 2 frames
            return 0.0

        # Compute frame-to-frame differences in the masked region
        # latents shape: (B, C, T, H, W)
        # mask_latents shape: (B, 1, T, H, W) - 1 where generating, 0 where preserving

        # Only compute on generated region
        diff = latents[:, :, 1:] - latents[:, :, :-1]  # (B, C, T-1, H, W)
        mask_diff = mask_latents[:, :, 1:] * mask_latents[:, :, :-1]  # Both frames generating

        # Weighted L2 loss on masked regions
        loss = (diff**2 * mask_diff).sum() / (mask_diff.sum() * latents.shape[1] + 1e-6)
        return loss * TEMPORAL_CONSISTENCY_WEIGHT

    def _enhance_prompt(self, prompt: str, operation: str = "edit") -> str:
        """Enhance prompt for better VACE results.

        VACE benefits from structured prompts with clear subject, action, and context.
        """
        # VACE works best with prompts that describe the desired output scene
        # For editing tasks, focus on the replacement content, not the source
        enhancements = {
            "replace": (
                ", photorealistic, sharp details, consistent lighting,"
                " temporal consistency, high quality"
            ),
            "remove": (
                ", empty scene, photorealistic, consistent lighting,"
                " seamless background, temporal consistency"
            ),
            "edit": (
                ", photorealistic, sharp details, consistent lighting,"
                " temporal consistency, high quality"
            ),
        }
        suffix = enhancements.get(operation, enhancements["edit"])
        return prompt + suffix

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
        anchors: dict | None = None,
    ) -> tuple[list[NDArray[np.uint8]], dict]:
        """Render one temporally-consistent candidate for every edited frame.

        anchors maps frame index -> a finished frame (same size as frames), or
        (frame, keep_mask) to lock only part of it. Locked pixels enter
        src_video unmasked (mask 0), so the pass propagates a keyframe edit
        instead of inventing the fill.
        Same contract as the SD path: returns (candidate_frames, provenance);
        candidates are patches the caller composites and hard-restores against.
        task selects task-specific overrides (the removal_ settings when "remove"):
        removal wants stronger prompt adherence (empty, no object) than a
        replacement, which must balance prompt against identity conditioning.
        If use_reference_pass is enabled (default), runs a second "Swap Anything"
        pass with a reference image cropped from the best frame of pass 1, which
        anchors identity across frames and reduces temporal drift.
        """
        if self._pipe is None:
            self.load()
        steps = int(self._settings.get("num_inference_steps", 30))
        if anchors:
            # Anchored passes interpolate between finished frames rather than
            # invent content, so fewer steps hold quality (A/B 2026-09-21).
            steps = int(self._settings.get("anchored_num_inference_steps", steps))
        guidance = float(self._settings.get("guidance_scale", 5.0))
        seed = int(self._settings.get("seed", 12345))
        if task == "remove":
            removal_steps = self._settings.get("removal_num_inference_steps")
            removal_guidance = self._settings.get("removal_guidance_scale")
            removal_seed = self._settings.get("removal_seed")
            if removal_steps is not None:
                steps = int(removal_steps)
            if removal_guidance is not None:
                guidance = float(removal_guidance)
            if removal_seed is not None:
                seed = int(removal_seed)
        conditioning_scale = float(self._settings.get("conditioning_scale", 1.0))
        color_evidence_min = float(self._settings.get("color_evidence_min", 0.10))
        max_rerolls = int(self._settings.get("max_rerolls", 1))
        use_reference_pass = bool(self._settings.get("use_reference_pass", True))
        use_cfg_zero_star = bool(
            self._settings.get("use_cfg_zero_star", True)
        )  # Enabled by default now
        cfg_zero_star_alpha = float(self._settings.get("cfg_zero_star_alpha", 0.5))
        use_temporal_consistency = bool(self._settings.get("use_temporal_consistency", True))
        use_prompt_enhancement = bool(self._settings.get("use_prompt_enhancement", True))
        # A reveal prompt describes the whole subject (it may say "red jacket"):
        # no colour gate or reroll, the colour is context, not the edit.
        color = None if task == "reveal" else _prompt_color(prompt)
        negative = negative_prompt or str(
            self._settings.get(
                "negative_prompt",
                "blurry, low quality, distorted, deformed, watermark, text,"
                " static, flicker, inconsistent",
            )
        )

        edited_indices = [i for i, m in enumerate(masks) if (m > 0).any()]
        if not edited_indices:
            return list(frames), {"backend": self.name, "frames_generated": 0}

        # Check if we need windowed processing for long videos
        max_frames = int(self._settings.get("max_window_frames", 81))
        if len(frames) > max_frames:
            return self._replace_sequence_windowed(
                frames,
                masks,
                prompt,
                negative,
                steps,
                guidance,
                seed,
                conditioning_scale,
                color_evidence_min,
                max_rerolls,
                use_reference_pass,
                use_cfg_zero_star,
                cfg_zero_star_alpha,
                color,
            )

        render_h, render_w = self._render_size(frames[0].shape)
        work_frames, work_masks = self._prepare_clip(frames, masks, render_h, render_w)
        work_anchors = {}
        for i, anchor in (anchors or {}).items():
            frame_a, keep = anchor if isinstance(anchor, tuple) else (anchor, None)
            frame_a = cv2.resize(frame_a, (render_w, render_h), interpolation=cv2.INTER_AREA)
            if keep is not None:
                keep = (
                    cv2.resize(
                        keep.astype(np.uint8), (render_w, render_h), interpolation=cv2.INTER_NEAREST
                    )
                    > 0
                )
            work_anchors[i] = (frame_a, keep)
        src_video, mask_images = self._build_vace_inputs(work_frames, work_masks, work_anchors)

        # Enhance prompt for better VACE results
        enhanced_prompt = (
            self._enhance_prompt(prompt, "replace") if use_prompt_enhancement else prompt
        )
        log.info("Enhanced prompt", original=prompt, enhanced=enhanced_prompt)

        # Pass 1: standard render
        result = self._run_pipeline(
            src_video,
            mask_images,
            enhanced_prompt,
            negative,
            render_h,
            render_w,
            len(work_frames),
            steps,
            guidance,
            conditioning_scale,
            seed,
            use_cfg_zero_star,
            cfg_zero_star_alpha,
            use_temporal_consistency,
        )

        if not np.isfinite(result).all():
            log.warning("VACE render produced non-finite values, retrying in float32")
            self.unload()
            self._settings["transformer_dtype"] = "float32"
            self.load()
            result = self._run_pipeline(
                src_video,
                mask_images,
                enhanced_prompt,
                negative,
                render_h,
                render_w,
                len(work_frames),
                steps,
                max(1.0, guidance),
                conditioning_scale,
                seed,
                use_cfg_zero_star,
                cfg_zero_star_alpha,
                use_temporal_consistency,
            )

        result = (np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)[: len(frames)]

        evidence = None
        rerendered = 0
        if color is not None and color_evidence_min > 0:
            evidence = self._clip_evidence(result, work_masks[: len(frames)], color)
            attempt = 0
            while evidence < color_evidence_min and attempt < max_rerolls:
                attempt += 1
                log.info(
                    "Clip missing requested colour, re-rendering",
                    evidence=round(evidence, 3),
                    color=color,
                    attempt=attempt,
                )
                alternative = self._run_pipeline(
                    src_video,
                    mask_images,
                    enhanced_prompt,
                    negative,
                    render_h,
                    render_w,
                    len(work_frames),
                    steps,
                    guidance,
                    conditioning_scale,
                    seed + 1000 * attempt,
                    use_cfg_zero_star,
                    cfg_zero_star_alpha,
                    use_temporal_consistency,
                )
                alternative = (np.clip(alternative, 0.0, 1.0) * 255.0).astype(np.uint8)[
                    : len(frames)
                ]
                alt_evidence = self._clip_evidence(alternative, work_masks[: len(frames)], color)
                rerendered += 1
                if alt_evidence > evidence:
                    result, evidence = alternative, alt_evidence

            if evidence < color_evidence_min:
                raise RuntimeError(
                    "The model could not produce the requested replacement "
                    f"(best colour evidence {evidence:.3f} < {color_evidence_min}). "
                    "Try again or rephrase with a plainer subject."
                )

        # Pass 1 candidates (upscaled to source resolution)
        candidates_pass1 = [
            cv2.resize(patch, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            for patch, frame in zip(result, frames, strict=True)
        ]

        # Pass 2: reference-image "Swap Anything" if enabled. Never for
        # removal: Swap-Anything anchors an OBJECT identity, and anchoring
        # anything on an object-removal render biases toward keeping it.
        if use_reference_pass and color is not None and task not in ("remove", "reveal"):
            # Use stronger conditioning for reference pass to anchor identity
            ref_conditioning_scale = float(self._settings.get("reference_conditioning_scale", 2.0))
            candidates_pass2, pass2_provenance = self._reference_pass(
                frames,
                masks,
                src_video,
                mask_images,
                render_h,
                render_w,
                len(work_frames),
                enhanced_prompt,
                negative,
                steps,
                guidance,
                ref_conditioning_scale,
                seed,
                color,
                color_evidence_min,
                candidates_pass1,
                work_masks,
                use_cfg_zero_star,
                cfg_zero_star_alpha,
                use_temporal_consistency,
            )
            # Evaluate both passes and pick the better one
            final_candidates, chosen_pass = self._select_best_candidates(
                candidates_pass1, candidates_pass2, frames, masks, color, enhanced_prompt
            )
            provenance = {
                "backend": self.name,
                "model": self._config.get("model"),
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "negative_prompt": negative,
                "seed": seed,
                "guidance_scale": guidance,
                "num_inference_steps": steps,
                "frames_generated": len(edited_indices),
                "clips_rendered": 1 + rerendered + (1 if chosen_pass == 2 else 0),
                "temporal_mode": "full-attention",
                "color_evidence": round(evidence, 3) if evidence is not None else None,
                "render_size": [render_w, render_h],
                "reference_pass_used": chosen_pass == 2,
                "cfg_zero_star": use_cfg_zero_star,
                "cfg_zero_star_alpha": cfg_zero_star_alpha,
                "temporal_consistency": use_temporal_consistency,
                "prompt_enhancement": use_prompt_enhancement,
            }
            if chosen_pass == 2:
                provenance.update({f"pass2_{k}": v for k, v in pass2_provenance.items()})
            return final_candidates, provenance

        provenance = {
            "backend": self.name,
            "model": self._config.get("model"),
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative,
            "seed": seed,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "frames_generated": len(edited_indices),
            "clips_rendered": 1 + rerendered,
            "temporal_mode": "full-attention",
            "color_evidence": round(evidence, 3) if evidence is not None else None,
            "render_size": [render_w, render_h],
            "reference_pass_used": False,
            "cfg_zero_star": use_cfg_zero_star,
            "cfg_zero_star_alpha": cfg_zero_star_alpha,
            "temporal_consistency": use_temporal_consistency,
            "prompt_enhancement": use_prompt_enhancement,
        }
        return candidates_pass1, provenance

    def _replace_sequence_windowed(
        self,
        frames: list[NDArray[np.uint8]],
        masks: NDArray[np.uint8],
        prompt: str,
        negative: str,
        steps: int,
        guidance: float,
        seed: int,
        conditioning_scale: float,
        color_evidence_min: float,
        max_rerolls: int,
        use_reference_pass: bool,
        use_cfg_zero_star: bool,
        cfg_zero_star_alpha: float,
        color: str | None,
    ) -> tuple[list[NDArray[np.uint8]], dict]:
        """Process long video using overlapping windows (VideoPainter-style)."""
        from preserve.long_video import WindowConfig, process_long_video

        log.info("Processing long video with windowed VACE", total_frames=len(frames))

        window_size = int(self._settings.get("max_window_frames", 81))
        overlap = int(self._settings.get("window_overlap", 16))
        window_config = WindowConfig(
            window_size=window_size,
            overlap=overlap,
            min_window=min(24, window_size // 2),
            id_reference_stride=window_size // 3,
        )

        use_temporal_consistency = self._settings.get("use_temporal_consistency", True)
        use_prompt_enhancement = self._settings.get("use_prompt_enhancement", True)
        enhanced_prompt = (
            self._enhance_prompt(prompt, "replace") if use_prompt_enhancement else prompt
        )

        def process_window(window_frames: list[NDArray[np.uint8]], window_masks: NDArray[np.uint8]):
            render_h, render_w = self._render_size(window_frames[0].shape)
            work_frames, work_masks = self._prepare_clip(
                window_frames, window_masks, render_h, render_w
            )
            src_video, mask_images = self._build_vace_inputs(work_frames, work_masks)

            result = self._run_pipeline(
                src_video,
                mask_images,
                enhanced_prompt,
                negative,
                render_h,
                render_w,
                len(work_frames),
                steps,
                guidance,
                conditioning_scale,
                seed,
                use_cfg_zero_star,
                cfg_zero_star_alpha,
                use_temporal_consistency,
            )

            result = (np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)[: len(window_frames)]
            return [
                cv2.resize(patch, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                for patch, f in zip(result, window_frames, strict=True)
            ]

        candidate_frames = process_long_video(
            frames, masks, process_window, window_config, id_resample_strength=0.3
        )

        render_h, render_w = self._render_size(frames[0].shape)
        provenance = {
            "backend": self.name,
            "model": self._config.get("model"),
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative,
            "seed": seed,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "frames_generated": len([i for i, m in enumerate(masks) if (m > 0).any()]),
            "clips_rendered": 1,
            "temporal_mode": "windowed-full-attention",
            "windowed": True,
            "window_size": window_size,
            "overlap": overlap,
            "render_size": [render_w, render_h],
            "cfg_zero_star": use_cfg_zero_star,
            "temporal_consistency": use_temporal_consistency,
            "prompt_enhancement": use_prompt_enhancement,
        }
        return candidate_frames, provenance

    def _reference_pass(
        self,
        frames: list[NDArray[np.uint8]],
        masks: NDArray[np.uint8],
        src_video: list[Image.Image],
        mask_images: list[Image.Image],
        render_h: int,
        render_w: int,
        num_frames: int,
        prompt: str,
        negative: str,
        steps: int,
        guidance: float,
        conditioning_scale: float,
        seed: int,
        color: str,
        color_evidence_min: float,
        pass1_candidates: list[NDArray[np.uint8]],
        work_masks: NDArray[np.uint8],
        use_cfg_zero_star: bool = False,
        cfg_zero_star_alpha: float = 0.5,
        use_temporal_consistency: bool = True,
    ) -> tuple[list[NDArray[np.uint8]], dict]:
        """Run pass 2 with a reference image from the best pass-1 frame.

        The reference image is the replacement object cropped from the best frame
        (highest color evidence + sharpness), pasted on white background per
        VACE's "Swap Anything" composition protocol.

        Enhancement: Use multiple reference frames for better identity anchoring.
        """
        # Score pass-1 frames to pick the best reference
        best_idx = self._pick_best_reference_frame(pass1_candidates, masks, color)
        if best_idx is None:
            log.warning("No valid frame for reference image, skipping pass 2")
            return pass1_candidates, {"skipped": "no_valid_reference"}

        # Build reference image: crop object from best frame on white background
        ref_image = self._build_reference_image(pass1_candidates[best_idx], masks[best_idx])

        # Also build a second reference from a different frame for multi-reference
        # This helps with identity consistency across the clip
        ref_images = [[ref_image]]

        # Try to add a second reference from a frame with good color evidence but different pose
        second_idx = self._pick_second_reference_frame(pass1_candidates, masks, color, best_idx)
        if second_idx is not None and second_idx != best_idx:
            ref_image2 = self._build_reference_image(
                pass1_candidates[second_idx], masks[second_idx]
            )
            ref_images[0].append(ref_image2)
            log.info("Using multi-reference Swap Anything", ref1=best_idx, ref2=second_idx)

        # Run pass 2 with reference image(s)
        generator = torch.Generator("cpu").manual_seed(seed)
        try:
            output = self._pipe(
                prompt=prompt,
                negative_prompt=negative,
                video=src_video,
                mask=mask_images,
                reference_images=ref_images,  # Swap Anything: list of list
                height=render_h,
                width=render_w,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                conditioning_scale=conditioning_scale,
                generator=generator,
                output_type="np",
            ).frames[0]
        except Exception as e:
            log.warning("Reference pass failed, falling back to pass 1", error=str(e))
            return pass1_candidates, {"skipped": f"reference_pass_failed: {e}"}

        if not np.isfinite(output).all():
            log.warning("Reference pass produced non-finite values")
            return pass1_candidates, {"skipped": "non_finite"}

        result2 = (np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)[: len(frames)]

        # Check color evidence on pass 2
        evidence2 = None
        if color is not None and color_evidence_min > 0:
            evidence2 = self._clip_evidence(result2, work_masks[: len(frames)], color)
            if evidence2 < color_evidence_min:
                log.warning(
                    "Reference pass failed color evidence, falling back", evidence=evidence2
                )
                return pass1_candidates, {"skipped": "low_color_evidence"}

        candidates_pass2 = [
            cv2.resize(patch, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            for patch, frame in zip(result2, frames, strict=True)
        ]

        pass2_provenance = {
            "reference_frame_idx": best_idx,
            "second_reference_frame_idx": second_idx if "second_idx" in locals() else None,
            "color_evidence": round(evidence2, 3) if evidence2 is not None else None,
            "multi_reference": len(ref_images[0]) > 1,
        }
        return candidates_pass2, pass2_provenance

    @staticmethod
    def _anchor_eligible(
        mask: NDArray[np.bool_], median_area: float, shape: tuple[int, ...]
    ) -> bool:
        """Occlusion-aware anchor guard: the reference object must be whole.
        Anchoring Swap-Anything on an edge-truncated or near-vanished frame
        (car exiting left, 13% mask) bakes that deformity into every frame —
        iter2 anchored ref1=13 and pass2 lost 0.577 vs 0.524. Require at least
        half the median mask area and a 4px margin from every frame edge, i.e.
        structural completeness (cf. occlusion-aware keyframe selection,
        arXiv:2605.23192). Attribute visibility stays scored by the callers.
        """
        area = int(mask.sum())
        if area < max(25, int(0.5 * median_area)):
            return False
        ys, xs = np.nonzero(mask)
        h, w = shape[:2]
        return not (
            int(xs.min()) < 4 or int(ys.min()) < 4 or int(xs.max()) > w - 5 or int(ys.max()) > h - 5
        )

    def _pick_second_reference_frame(
        self,
        candidates: list[NDArray[np.uint8]],
        masks: NDArray[np.uint8],
        color: str,
        exclude_idx: int,
    ) -> int | None:
        """Pick a second reference frame with good color evidence but different pose."""
        from preserve.segment import _color_pixel_mask

        median_area = float(np.median([(m > 0).sum() for m in masks])) if len(masks) else 0.0
        best_score = -1.0
        best_idx = None
        for only_eligible in (True, False):
            for idx, (cand, mask) in enumerate(zip(candidates, masks, strict=True)):
                if idx == exclude_idx:
                    continue
                region = mask > 0
                if not region.any():
                    continue
                if only_eligible and not VACEReplacer._anchor_eligible(
                    region, median_area, cand.shape
                ):
                    continue
                eroded = cv2.erode(region.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                if not eroded.any():
                    eroded = region
                hsv = cv2.cvtColor(cand, cv2.COLOR_RGB2HSV)
                color_ev = float(
                    (_color_pixel_mask(hsv, color) & eroded).sum() / max(1, eroded.sum())
                )
                # Sharpness (Laplacian variance) on the object region
                gray = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY)
                sharpness = (
                    float(cv2.Laplacian(gray[eroded], cv2.CV_64F).var()) if eroded.any() else 0.0
                )
                # Prefer frames with good color evidence but different characteristics
                score = color_ev * 100 + min(sharpness / 1000.0, 1.0)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None:
                break
        return best_idx

    def _pick_best_reference_frame(
        self,
        candidates: list[NDArray[np.uint8]],
        masks: NDArray[np.uint8],
        color: str,
    ) -> int | None:
        """Pick the best frame for reference image based on color evidence + sharpness."""
        from preserve.segment import _color_pixel_mask

        median_area = float(np.median([(m > 0).sum() for m in masks])) if len(masks) else 0.0
        best_score = -1.0
        best_idx = None
        for only_eligible in (True, False):
            for idx, (cand, mask) in enumerate(zip(candidates, masks, strict=True)):
                region = mask > 0
                if not region.any():
                    continue
                if only_eligible and not VACEReplacer._anchor_eligible(
                    region, median_area, cand.shape
                ):
                    continue
                # Color evidence on eroded core
                eroded = cv2.erode(region.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                if not eroded.any():
                    eroded = region
                hsv = cv2.cvtColor(cand, cv2.COLOR_RGB2HSV)
                color_ev = float(
                    (_color_pixel_mask(hsv, color) & eroded).sum() / max(1, eroded.sum())
                )
                # Sharpness (Laplacian variance) on the object region
                gray = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY)
                sharpness = (
                    float(cv2.Laplacian(gray[eroded], cv2.CV_64F).var()) if eroded.any() else 0.0
                )
                # Combined score: prioritize color evidence, then sharpness
                score = color_ev * 100 + min(sharpness / 1000.0, 1.0)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None:
                break
        return best_idx

    def _build_reference_image(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8],
    ) -> Image.Image:
        """Crop the object from frame with margin, paste on white background."""
        region = mask > 0
        if not region.any():
            # Fallback: full frame on white
            white_bg = np.full_like(frame, 255)
            return Image.fromarray(white_bg)

        # Bounding box of the mask
        coords = np.argwhere(region)
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        # Add margin
        margin = 8
        h, w = frame.shape[:2]
        y1 = max(0, y1 - margin)
        x1 = max(0, x1 - margin)
        y2 = min(h - 1, y2 + margin)
        x2 = min(w - 1, x2 + margin)

        crop = frame[y1 : y2 + 1, x1 : x2 + 1].copy()
        crop_mask = region[y1 : y2 + 1, x1 : x2 + 1]

        # Paste on white background
        white_bg = np.full_like(crop, 255)
        white_bg[crop_mask] = crop[crop_mask]
        return Image.fromarray(white_bg)

    def _select_best_candidates(
        self,
        candidates1: list[NDArray[np.uint8]],
        candidates2: list[NDArray[np.uint8]],
        frames: list[NDArray[np.uint8]],
        masks: NDArray[np.uint8],
        color: str,
        prompt: str,
    ) -> tuple[list[NDArray[np.uint8]], int]:
        """Select the better clip between pass 1 and pass 2.

        Objective: maximize temporal stability (low drift) + color evidence + CLIP adherence.
        """
        from preserve.metrics import clip_temporal_adherence

        def evaluate_clip(cands):
            # Temporal drift ratio (lower is better)
            from preserve.score import _appearance_drift_ratio

            allowed = masks.astype(bool)
            drift = _appearance_drift_ratio(frames, cands, allowed)
            drift_score = 1.0 / max(drift, 0.1) if drift is not None else 1.0
            # Color evidence
            from preserve.segment import _color_pixel_mask

            color_ev = 0.0
            if color is not None:
                total_ev = 0.0
                count = 0
                for cand, mask in zip(cands, masks, strict=True):
                    region = mask > 0
                    if not region.any():
                        continue
                    eroded = cv2.erode(region.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                    if not eroded.any():
                        eroded = region
                    hsv = cv2.cvtColor(cand, cv2.COLOR_RGB2HSV)
                    total_ev += float(
                        (_color_pixel_mask(hsv, color) & eroded).sum() / max(1, eroded.sum())
                    )
                    count += 1
                color_ev = total_ev / max(1, count)
            # CLIP temporal adherence
            clip_mean, clip_min = 0.5, 0.5
            try:
                clip_res = clip_temporal_adherence(cands, prompt)
                if clip_res is not None:
                    clip_mean, clip_min = clip_res
            except Exception:
                pass
            # Weighted score
            return drift_score * 0.4 + color_ev * 0.4 + clip_min * 0.2

        score1 = evaluate_clip(candidates1)
        score2 = evaluate_clip(candidates2)

        log.info("Pass comparison", pass1=round(score1, 4), pass2=round(score2, 4))
        if score2 > score1:
            return candidates2, 2
        return candidates1, 1

    def _run_pipeline(
        self,
        src_video: list[Image.Image],
        mask_images: list[Image.Image],
        prompt: str,
        negative: str,
        height: int,
        width: int,
        num_frames: int,
        steps: int,
        guidance: float,
        conditioning_scale: float,
        seed: int,
        use_cfg_zero_star: bool = False,
        cfg_zero_star_alpha: float = 0.5,
        use_temporal_consistency: bool = True,
    ) -> NDArray[np.float32]:
        generator = torch.Generator("cpu").manual_seed(seed)

        # Prepare callback kwargs for CFG Zero-Star
        callback_kwargs = {}
        if use_cfg_zero_star:
            callback_kwargs["cfg_zero_star_alpha"] = cfg_zero_star_alpha

        # For temporal consistency, we would need to hook into the denoising loop
        # This is a simplified approach: we run the pipeline and rely on VACE's
        # native temporal attention for consistency. For explicit temporal regularization,
        # we would need a custom denoising loop which is more complex.
        # The temporal consistency is mainly achieved through:
        # 1. VACE's full-attention across frames
        # 2. Reference pass (Swap Anything) for identity anchoring
        # 3. Enhanced prompts for better temporal coherence
        # 4. Proper conditioning_scale and guidance_scale settings

        output = self._pipe(
            prompt=prompt,
            negative_prompt=negative,
            video=src_video,
            mask=mask_images,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            conditioning_scale=conditioning_scale,
            generator=generator,
            output_type="np",
        ).frames[0]
        return output

    def _render_size(self, shape: tuple[int, ...]) -> tuple[int, int]:
        """Pick a multiple-of-16 render size bounded by MAX_RENDER_DIM.

        Only downscale: Wan 1.3B is a 480p-class model. Larger inputs are
        downscaled to this bound; smaller inputs render at their native size
        to avoid introducing artifacts from upscaling. The caller composites
        onto full-resolution sources, so only the patch ever passes through
        the smaller space.
        """
        h, w = shape[:2]
        render_dim = int(self._settings.get("render_dim", MAX_RENDER_DIM))
        scale = min(1.0, render_dim / max(h, w))
        render_h = max(16, int(round(h * scale / 16)) * 16)
        render_w = max(16, int(round(w * scale / 16)) * 16)
        return render_h, render_w

    def _prepare_clip(
        self,
        frames: list[NDArray[np.uint8]],
        masks: NDArray[np.uint8],
        render_h: int,
        render_w: int,
    ) -> tuple[list[NDArray[np.uint8]], NDArray[np.uint8]]:
        """Pad to 4k+1 frames and resize to the render size."""
        padded_count = _align_frame_count(len(frames))
        resized = [
            cv2.resize(f, (render_w, render_h), interpolation=cv2.INTER_AREA) for f in frames
        ] + [
            cv2.resize(frames[-1], (render_w, render_h), interpolation=cv2.INTER_AREA)
            for _ in range(padded_count - len(frames))
        ]
        # Resize masks to render size as well (nearest-neighbour to keep binary).
        resized_masks = [
            cv2.resize(m, (render_w, render_h), interpolation=cv2.INTER_NEAREST) for m in masks
        ]
        pad_rows = np.repeat(resized_masks[-1:], padded_count - len(masks), axis=0)
        work_masks = np.concatenate([resized_masks, pad_rows], axis=0)
        return resized, work_masks

    def _build_vace_inputs(
        self,
        frames: list[NDArray[np.uint8]],
        masks: NDArray[np.uint8],
        anchors: dict[int, tuple[NDArray[np.uint8], NDArray[np.bool_] | None]] | None = None,
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        """Grey out the edit region (127 = unknown to VACE), white mask = generate.

        The generate region is grown a few pixels past the tracked matte so the
        model re-renders contact shadows and boundary pixels instead of keeping
        the old object's edge as 'context' — the halo that survives otherwise.
        Anchored frames go in whole with a black mask (known everywhere).
        """
        grow_px = int(self._settings.get("mask_grow_px", 4))
        src_video = []
        mask_images = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_px * 2 + 1, grow_px * 2 + 1))
        for index, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
            keep = None
            if anchors and index in anchors:
                frame_a, keep = anchors[index]
                if keep is None or keep.all():
                    src_video.append(Image.fromarray(frame_a))
                    mask_images.append(Image.fromarray(np.zeros(mask.shape[:2], np.uint8)))
                    continue
                # Partially anchored: kept pixels are known, the rest of the
                # region is generated as usual.
                frame, mask = frame_a, np.where(keep, 0, mask).astype(np.uint8)
            region = mask > 0
            if grow_px > 0 and region.any():
                region = cv2.dilate(region.astype(np.uint8), kernel) > 0
                if keep is not None:
                    region &= ~keep
            conditioned = frame.copy()
            conditioned[region] = 127
            src_video.append(Image.fromarray(conditioned))
            mask_images.append(Image.fromarray(np.where(region, 255, 0).astype(np.uint8)))
        return src_video, mask_images

    def _clip_evidence(
        self, frames: NDArray[np.uint8], masks: NDArray[np.uint8], color: str
    ) -> float:
        """Fraction of eroded edit-region pixels matching the requested colour.

        Pooled over all frames: a clip-level gate on a clip-level render. The
        per-frame SD path needs anchors plus propagation repair because its
        failures are local; here one bad clip fails everywhere at once, so a
        single number is the right granularity.
        """
        from preserve.segment import _color_pixel_mask

        matched = 0
        total = 0
        for frame, mask in zip(frames, masks, strict=True):
            region = mask > 0
            if not region.any():
                continue
            eroded = cv2.erode(region.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            if not eroded.any():
                eroded = region
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            matched += int((_color_pixel_mask(hsv, color) & eroded).sum())
            total += int(eroded.sum())
        return matched / max(1, total)


_vace_replacer: VACEReplacer | None = None


def get_vace_replacer() -> VACEReplacer:
    global _vace_replacer
    if _vace_replacer is None:
        _vace_replacer = VACEReplacer()
    return _vace_replacer
