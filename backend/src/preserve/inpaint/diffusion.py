"""Diffusion-based video inpainting backend.

Uses Wan2.1-VACE or similar diffusion models for cases where
the hidden content is never visible in any frame.
"""

from pathlib import Path

import torch

from preserve.config import settings
from preserve.inpaint.base import InpaintBackend, InpaintRequest, InpaintResult


class DiffusionInpaintBackend(InpaintBackend):
    """Diffusion model backend for video inpainting."""

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: Path | None = None,
    ):
        self.model_id = model_id or settings.video_inpaint_model
        self.model_dir = model_dir or settings.model_dir
        self._pipe = None
        self._device = None
        self._dtype = None

    @property
    def name(self) -> str:
        return f"Diffusion ({self.model_id.split('/')[-1]})"

    @property
    def supports_mps(self) -> bool:
        return True

    def is_available(self) -> bool:
        return self._pipe is not None

    def _get_device(self) -> torch.device:
        if settings.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif settings.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _get_dtype(self) -> torch.dtype:
        if settings.dtype == "bfloat16":
            return torch.bfloat16
        elif settings.dtype == "float16":
            return torch.float16
        return torch.float32

    def load(self) -> None:
        """Load diffusion pipeline.

        NOTE: Actual loading requires model download.
        """
        self._device = self._get_device()
        self._dtype = self._get_dtype()

        # Check for local model first
        local_path = self.model_dir / self.model_id.replace("/", "--")
        model_path = local_path if local_path.exists() else self.model_id

        # Placeholder for actual pipeline loading
        # from diffusers import WanVideoPipeline # or similar
        # self._pipe = WanVideoPipeline.from_pretrained(
        # model_path,
        # torch_dtype=self._dtype,
        # ).to(self._device)

        raise NotImplementedError(
            f"Diffusion model loading not yet implemented for {model_path}. "
            "Run download script first: python -m preserve.scripts.download_models"
        )

    def unload(self) -> None:
        self._pipe = None
        if self._device and self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._device = None
        self._dtype = None

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        """Run diffusion-based video inpainting.

        Pipeline:
        1. Encode frames to latent space
        2. Add noise to masked latent regions
        3. Denoise with temporal attention
        4. Decode back to pixel space
        """
        if not self.is_available():
            raise RuntimeError("Diffusion model not loaded. Call load() first.")

        # Placeholder implementation
        raise NotImplementedError("Diffusion inpainting not yet implemented")


def create_diffusion_backend(model_id: str | None = None) -> DiffusionInpaintBackend:
    """Factory function for diffusion backend."""
    return DiffusionInpaintBackend(model_id=model_id)
