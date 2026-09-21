"""MiniMax-Remover (NeurIPS 2025, arXiv:2505.24873) video object removal.

A Wan-2.1-class 1.3B DiT with text input and cross-attention removed, trained
with minimax optimisation against "bad noise" so the masked region is filled
with background instead of a regenerated copy of the object. This is the
failure mode VACE removal showed on every audit (dark ghost car): a general
inpainting model completes the silhouette its mask hands it, a purpose-trained
remover does not. 6-12 steps, no CFG, no prompt.

Conditioning is channel concatenation in latent space: [noisy latent (16),
VAE(masked video) (16), VAE(mask as video) (16)] -> 48 input channels.

Weights: zibojia/minimax-remover (CC BY-NC 4.0, non-commercial).
"""

from typing import Any

import cv2
import numpy as np
import scipy.ndimage
import structlog
import torch

from preserve.config import settings
from preserve.inpaint.base import InpaintBackend, InpaintRequest, InpaintResult
from preserve.mps import bound_vae_memory, enable_flash_attention

log = structlog.get_logger()


def _align_frame_count(n: int) -> int:
    """Wan's causal VAE compresses time 4x: clip length must be 4k+1."""
    return ((max(n, 1) - 1 + 3) // 4) * 4 + 1


class MiniMaxRemoverBackend(InpaintBackend):
    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or settings.get_backend_config("minimax")
        self._settings = self._config.get("settings", {})
        self._vae = None
        self._transformer = None
        self._scheduler = None
        self._device: torch.device | None = None
        self._dtype = torch.float16

    @property
    def name(self) -> str:
        return self._config.get("name", "MiniMax-Remover")

    @property
    def supports_mps(self) -> bool:
        return True

    def is_available(self) -> bool:
        return self._transformer is not None

    def load(self) -> None:
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler

        from preserve.inpaint.minimax_transformer import Transformer3DModel

        model_dir = settings.model_dir / self._config.get("model_dir", "minimax-remover")
        if not (model_dir / "transformer").exists():
            raise FileNotFoundError(f"MiniMax-Remover weights not found: {model_dir}")
        self._device = settings.get_device()
        dtype_name = str(self._settings.get("transformer_dtype", "float16"))
        self._dtype = getattr(torch, dtype_name)
        log.info("Loading MiniMax-Remover", model=str(model_dir), dtype=dtype_name)
        if self._device.type == "mps":
            log.info("Flash attention on MPS", enabled=enable_flash_attention())
        # VAE in fp32: the Wan VAE is bit-exact on MPS in fp32 and its cost is
        # small next to the DiT; fp16 VAE decode is where banding shows up.
        self._vae = AutoencoderKLWan.from_pretrained(
            model_dir, subfolder="vae", torch_dtype=torch.float32
        ).to(self._device)
        self._vae.eval()
        if bool(self._settings.get("enable_vae_tiling", True)):
            bound_vae_memory(self._vae)
        self._transformer = Transformer3DModel.from_pretrained(
            model_dir, subfolder="transformer", torch_dtype=self._dtype
        ).to(self._device)
        self._transformer.eval()
        self._scheduler = UniPCMultistepScheduler.from_pretrained(model_dir, subfolder="scheduler")

    def unload(self) -> None:
        self._vae = None
        self._transformer = None
        self._scheduler = None
        if self._device is not None and self._device.type == "mps":
            torch.mps.empty_cache()
        self._device = None

    def _render_size(self, h: int, w: int) -> tuple[int, int]:
        """Multiple-of-16 size bounded by render_dim on the long side (480p-class model)."""
        render_dim = int(self._settings.get("render_dim", 480))
        scale = min(1.0, render_dim / max(h, w))
        return max(16, round(h * scale / 16) * 16), max(16, round(w * scale / 16) * 16)

    @torch.no_grad()
    def _encode(self, video: torch.Tensor) -> torch.Tensor:
        """video: (1, 3, T, H, W) in [-1, 1] -> normalised latents (1, 16, t, h, w)."""
        assert self._vae is not None
        cfg = self._vae.config
        mean = torch.tensor(cfg.latents_mean, device=video.device).view(1, cfg.z_dim, 1, 1, 1)
        inv_std = 1.0 / torch.tensor(cfg.latents_std, device=video.device).view(
            1, cfg.z_dim, 1, 1, 1
        )
        latents = self._vae.encode(video.float()).latent_dist.mode()
        return (latents - mean) * inv_std

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        assert self._vae is not None
        cfg = self._vae.config
        mean = torch.tensor(cfg.latents_mean, device=latents.device).view(1, cfg.z_dim, 1, 1, 1)
        std = torch.tensor(cfg.latents_std, device=latents.device).view(1, cfg.z_dim, 1, 1, 1)
        return self._vae.decode(latents.float() * std + mean, return_dict=False)[0]

    @torch.no_grad()
    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        if not self.is_available():
            self.load()
        assert self._transformer is not None and self._scheduler is not None
        assert self._device is not None
        steps = int(self._settings.get("num_inference_steps", 12))
        iterations = int(self._settings.get("mask_dilate_iterations", 6))
        seed = request.seed if request.seed is not None else int(self._settings.get("seed", 42))

        frames = request.frames
        n = len(frames)
        h, w = frames[0].shape[:2]
        render_h, render_w = self._render_size(h, w)
        padded = _align_frame_count(n)
        # Dilation iterations are specified for the 480p reference resolution;
        # scale so the grown band covers the same fraction of the frame.
        scale = max(render_h, render_w) / 480.0
        grow = round(iterations * scale)

        work = [cv2.resize(f, (render_w, render_h), interpolation=cv2.INTER_AREA) for f in frames]
        work += [work[-1]] * (padded - n)
        masks = [
            cv2.resize(m, (render_w, render_h), interpolation=cv2.INTER_NEAREST) > 127
            for m in request.masks
        ]
        masks += [masks[-1]] * (padded - n)
        if grow > 0:
            masks = [scipy.ndimage.binary_dilation(m, iterations=grow) for m in masks]

        images = torch.from_numpy(np.stack(work)).permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0
        mask_t = torch.from_numpy(np.stack(masks)).float()[None, None].expand(1, 3, -1, -1, -1)
        images = images.to(self._device)
        mask_t = mask_t.to(self._device)

        masked_latents = self._encode(images * (1 - mask_t))
        mask_latents = self._encode(2 * mask_t - 1)
        generator = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn(masked_latents.shape, generator=generator).to(self._device)

        # Scheduler math runs on CPU in fp32: UniPC's higher-order update
        # amplifies MPS rounding into noise (measured on the VACE path).
        self._scheduler.set_timesteps(steps, device="cpu")
        latents_cpu = latents.float().cpu()
        cond = torch.cat([masked_latents, mask_latents], dim=1).to(self._dtype)
        for t in self._scheduler.timesteps:
            model_input = torch.cat([latents_cpu.to(self._device, self._dtype), cond], dim=1)
            timestep = t.to(self._device).expand(1)
            noise_pred = self._transformer(hidden_states=model_input, timestep=timestep)[0]
            latents_cpu = self._scheduler.step(
                noise_pred.float().cpu(), t, latents_cpu, return_dict=False
            )[0]

        video = self._decode(latents_cpu.to(self._device))[0]
        if not torch.isfinite(video).all():
            raise RuntimeError("MiniMax-Remover produced non-finite output")
        video = ((video.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()

        out = [
            cv2.resize(video[i], (w, h), interpolation=cv2.INTER_LANCZOS4)
            if (render_h, render_w) != (h, w)
            else video[i]
            for i in range(n)
        ]
        return InpaintResult(
            frames=out,
            metadata={
                "backend": self.name,
                "steps": steps,
                "seed": seed,
                "render_size": f"{render_w}x{render_h}",
                "mask_dilate_iterations": grow,
            },
        )


def create_minimax_backend() -> MiniMaxRemoverBackend:
    return MiniMaxRemoverBackend()
