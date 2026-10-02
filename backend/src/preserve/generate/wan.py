"""Text-to-video generation via Wan 2.2 TI2V-5B (Diffusers WanPipeline).

Replaces AnimateDiff/SD1.5 as the default source-asset generator: a 2025
video DiT (5B, Apache 2.0, T2V+I2V hybrid) instead of a 2023 motion adapter
over an image UNet. Same Wan family as the VACE edit path, so prompt
conventions transfer between generation and editing.

Memory: ~10GB DiT in bf16 + umT5 text encoder. Fits the 48GB unified-memory
Mac via sequential CPU offload; attention/VAE slicing cap the 720p peak.
"""

import numpy as np
import structlog
import torch
from numpy.typing import NDArray

from preserve.config import settings
from preserve.generate.base import GenerationRequest, GenerationResult
from preserve.mps import bound_vae_memory, enable_flash_attention, wrap_scheduler_step_on_cpu

log = structlog.get_logger()


def _snap_frames(n: int) -> int:
    """Wan's 3D VAE compresses time 4x: frame counts must be 4k+1."""
    # Floor is 17, not 9: 13-frame renders produce unstructured mush on this
    # model (measured 2026-09-05), 17 is the smallest count that forms a scene.
    n = max(17, int(n))
    return ((n - 1) // 4) * 4 + 1


def _snap_size(v: int) -> int:
    """Dims must be multiples of 32 for patchification (diffusers adjusts
    anything smaller silently, e.g. 720 becomes 704)."""
    return max(256, (int(v) // 32) * 32)


class WanBackend:
    """Wan 2.2 TI2V-5B in text-to-video mode."""

    def __init__(self) -> None:
        self._pipe = None
        self._config = settings.get_generation_config()
        self._settings = self._config.get("settings", {})
        self.name = self._config.get("name", "Wan2.2-TI2V-5B")

    def is_available(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        from diffusers import AutoencoderKLWan, WanPipeline

        model_id = self._config.get("model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        device = settings.get_device()
        dtype = torch.bfloat16
        log.info("Loading generation model", model=model_id, dtype=str(dtype))
        if device.type == "mps":
            log.info("Flash attention on MPS", enabled=enable_flash_attention())

        # The model card loads the VAE in fp32 while the DiT runs in bf16. The
        # 16x-compression Wan2.2 VAE posterises in bf16 (audit 2026-09-19:
        # blotchy, banded renders), and it is cheap next to the transformer.
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
        pipe.set_progress_bar_config(disable=True)
        if bool(self._settings.get("enable_model_offload", True)):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        if bool(self._settings.get("enable_attention_slicing", True)):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                log.warning("Attention slicing unavailable, continuing without it")
        if bool(self._settings.get("enable_vae_tiling", True)):
            bound_vae_memory(vae)

        # Same UniPC-on-MPS rounding blow-up the VACE path fixed; model offload
        # leaves the latents on MPS, so the wrap is needed with or without it.
        # Audit 2026-09-19: the hazy, blown-out generations came from here.
        if device.type == "mps" and bool(self._settings.get("scheduler_on_cpu", True)):
            wrap_scheduler_step_on_cpu(pipe)

        self._pipe = pipe
        self._device = device

    def unload(self) -> None:
        self._pipe = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._pipe is None:
            self.load()

        seed = request.seed if request.seed is not None else 0
        generator = torch.Generator("cpu").manual_seed(seed)
        negative = request.negative_prompt or self._settings.get(
            "default_negative_prompt",
            "Bright tones, overexposed, static, blurred details, subtitles, "
            "low quality, worst quality, JPEG compression residue, ugly, "
            "deformed, disfigured, still picture, messy background",
        )
        num_frames = _snap_frames(request.num_frames)
        height = _snap_size(request.height)
        width = _snap_size(request.width)

        log.info(
            "Generating video",
            prompt=request.prompt,
            frames=num_frames,
            size=f"{width}x{height}",
            steps=request.num_inference_steps,
        )
        output = self._pipe(
            prompt=request.prompt,
            negative_prompt=negative,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        video = output.frames[0]
        if isinstance(video, np.ndarray) and video.max() <= 1.0:
            video = (video * 255).round().clip(0, 255).astype(np.uint8)
        frames: list[NDArray[np.uint8]] = [np.asarray(f, dtype=np.uint8) for f in video]

        provenance = {
            "backend": self.name,
            "model_id": self._config.get("model_id"),
            "pipeline": type(self._pipe).__name__,
            "dtype": "bfloat16",
            "device": str(self._device),
            "prompt": request.prompt,
            "negative_prompt": negative,
            "seed": seed,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "num_frames": num_frames,
            "resolution": f"{width}x{height}",
            "torch_version": torch.__version__,
        }
        return GenerationResult(frames=frames, fps=request.fps, provenance=provenance)


def create_wan_backend() -> WanBackend:
    return WanBackend()
