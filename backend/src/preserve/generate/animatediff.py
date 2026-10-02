"""Text-to-video generation via AnimateDiff on top of Stable Diffusion 1.5.

Chosen over the larger video transformers because it fits comfortably in unified
memory on Apple Silicon and shares the SD 1.5 latent space with the inpainting
model used for replacement edits — one VAE and one text encoder for both the
generate and edit halves of the product, instead of two unrelated ecosystems.

The output of this module is a source asset , not a master: it is hashed and
written to disk, and every subsequent edit treats it as immutable.
"""

import numpy as np
import structlog
import torch
from numpy.typing import NDArray

from preserve.config import settings
from preserve.generate.base import GenerationRequest, GenerationResult

log = structlog.get_logger()


class AnimateDiffBackend:
    """AnimateDiff motion adapter over an SD 1.5 checkpoint."""

    def __init__(self) -> None:
        self._pipe = None
        self._config = settings.get_generation_config()
        self._settings = self._config.get("settings", {})
        self.name = self._config.get("name", "AnimateDiff")

    def is_available(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter

        base = self._config.get("base_model", "stable-diffusion-v1-5/stable-diffusion-v1-5")
        adapter_id = self._config.get("motion_adapter", "guoyww/animatediff-motion-adapter-v1-5-2")

        device = settings.get_device()
        # float32 on MPS: half precision produces black frames in the motion
        # modules on some Metal builds, and the model is small enough not to need it.
        dtype = torch.float32 if device.type == "mps" else torch.float16

        log.info("Loading generation model", base=base, adapter=adapter_id, dtype=str(dtype))

        # low_cpu_mem_usage loads weights lazily onto the meta device and
        # materializes them during .to(). Under memory pressure that leaves
        # tensors still on meta and the move fails with "Cannot copy out of meta
        # tensor", so weights are read eagerly instead.
        adapter = MotionAdapter.from_pretrained(
            adapter_id, torch_dtype=dtype, low_cpu_mem_usage=False
        )
        pipe = AnimateDiffPipeline.from_pretrained(
            base, motion_adapter=adapter, torch_dtype=dtype, low_cpu_mem_usage=False
        )

        # AnimateDiff was trained against a linear beta schedule; the SD default
        # scaled_linear yields washed-out, low-motion output.
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1,
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)

        # AnimateDiff runs the UNet over all frames as one batch, so peak
        # activation memory scales with frames H W. Slicing attention and the
        # VAE decode trades a little speed for a much lower peak, which is the
        # difference between running and swapping on unified memory.
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        self._pipe = pipe
        self._device = device
        self._dtype = dtype

    def unload(self) -> None:
        self._pipe = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._pipe is None:
            self.load()

        seed = request.seed if request.seed is not None else 0
        # CPU generator keeps sampling identical regardless of the compute device.
        generator = torch.Generator("cpu").manual_seed(seed)

        negative = request.negative_prompt or self._settings.get(
            "default_negative_prompt", "blurry, low quality, distorted, watermark"
        )

        log.info(
            "Generating video",
            prompt=request.prompt,
            frames=request.num_frames,
            size=f"{request.width}x{request.height}",
            steps=request.num_inference_steps,
        )

        output = self._pipe(
            prompt=request.prompt,
            negative_prompt=negative,
            num_frames=request.num_frames,
            height=request.height,
            width=request.width,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            generator=generator,
        )

        frames: list[NDArray[np.uint8]] = [
            np.asarray(image, dtype=np.uint8) for image in output.frames[0]
        ]

        provenance = {
            "backend": self.name,
            "base_model": self._config.get("base_model"),
            "motion_adapter": self._config.get("motion_adapter"),
            "scheduler": "DDIMScheduler(beta_schedule=linear)",
            "dtype": str(self._dtype),
            "device": str(self._device),
            "prompt": request.prompt,
            "negative_prompt": negative,
            "seed": seed,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "num_frames": request.num_frames,
            "resolution": f"{request.width}x{request.height}",
            "torch_version": torch.__version__,
        }

        return GenerationResult(frames=frames, fps=request.fps, provenance=provenance)


def create_animatediff_backend() -> AnimateDiffBackend:
    return AnimateDiffBackend()
