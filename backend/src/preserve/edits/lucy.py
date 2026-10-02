"""Lucy Edit 1.1 (Decart, Wan2.2-5B instruction video editor) as a replacement backend.

An instruction-following editor: given the source clip and "remove his cap"
or "replace the car with a yellow taxi", it re-renders the whole clip with the
edit applied while keeping motion and identity. It has the semantics the
removal path lacks for attached objects (a cap hides hair, a straw hides a
mouth), which a background remover cannot invent and VACE-1.3B shape-leaks.

Only the transformer is Lucy's own (fp16 single-file ComfyUI checkpoint,
10GB); the VAE, umT5 text encoder, tokenizer and scheduler are the local
Wan2.2-TI2V-5B components. The pipeline never trusts its output outside the
mask: the caller composites the region and hard-restores the rest.

Weights: decart-ai/Lucy-Edit-1.1-Dev (non-commercial license).
"""

from pathlib import Path

import cv2
import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from PIL import Image

from preserve.config import settings
from preserve.mps import bound_vae_memory, enable_flash_attention, wrap_scheduler_step_on_cpu

log = structlog.get_logger()


def _align_frame_count(n: int) -> int:
    return ((max(n, 1) - 1 + 3) // 4) * 4 + 1


def _snap32(v: int) -> int:
    return max(32, int(round(v / 32)) * 32)


class LucyEditReplacer:
    prompt_style = "instruction"

    def __init__(self) -> None:
        self._config = settings.get_replacement_config("lucy")
        self._settings = self._config.get("settings", {})
        self.name = self._config.get("name", "Lucy-Edit-1.1")
        self._pipe = None
        self._device: torch.device | None = None

    def is_available(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        from diffusers import (
            AutoencoderKLWan,
            LucyEditPipeline,
            UniPCMultistepScheduler,
            WanTransformer3DModel,
        )
        from transformers import AutoTokenizer, UMT5EncoderModel

        base = str(self._config.get("base_model"))
        model_dir = settings.model_dir / self._config.get("model_dir", "lucy-edit-1.1")
        weights = model_dir / "transformer" / self._config.get("weights", "")
        if not weights.exists():
            raise FileNotFoundError(f"Lucy Edit weights not found: {weights}")
        self._device = settings.get_device()
        dtype = torch.bfloat16
        log.info("Loading Lucy Edit", weights=str(weights), base=base)
        if self._device.type == "mps":
            log.info("Flash attention on MPS", enabled=enable_flash_attention())
        transformer = WanTransformer3DModel.from_single_file(
            str(weights), config=str(model_dir / "transformer"), torch_dtype=dtype
        )
        vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae", torch_dtype=torch.float32)
        bound_vae_memory(vae)
        text_encoder = UMT5EncoderModel.from_pretrained(
            base, subfolder="text_encoder", torch_dtype=dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
        scheduler = UniPCMultistepScheduler.from_pretrained(base, subfolder="scheduler")
        pipe = LucyEditPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
            expand_timesteps=True,
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.to(self._device)
        if self._device.type == "mps":
            wrap_scheduler_step_on_cpu(pipe)
        self._pipe = pipe

    def unload(self) -> None:
        self._pipe = None
        if self._device is not None and self._device.type == "mps":
            torch.mps.empty_cache()

    def _render_size(self, h: int, w: int) -> tuple[int, int]:
        render_dim = int(self._settings.get("render_dim", 832))
        scale = min(1.0, render_dim / max(h, w))
        return _snap32(h * scale), _snap32(w * scale)

    @torch.no_grad()
    def replace_sequence(
        self,
        frames: list[NDArray[np.uint8]],
        masks: NDArray[np.uint8],
        prompt: str,
        negative_prompt: str | None = None,
        task: str = "replace",
    ) -> tuple[list[NDArray[np.uint8]], dict]:
        """Render the instructed edit over the whole clip; caller masks it.

        Same contract as the other replacers. masks are unused by the model
        (Lucy is mask-free) but kept for interface parity.
        """
        if self._pipe is None:
            self.load()
        assert self._pipe is not None
        steps = int(self._settings.get("num_inference_steps", 30))
        guidance = float(self._settings.get("guidance_scale", 5.0))
        seed = int(self._settings.get("seed", 42))
        negative = negative_prompt or str(self._settings.get("negative_prompt", ""))

        n = len(frames)
        h, w = frames[0].shape[:2]
        render_h, render_w = self._render_size(h, w)
        padded = _align_frame_count(n)
        work = [
            Image.fromarray(cv2.resize(f, (render_w, render_h), interpolation=cv2.INTER_AREA))
            for f in frames
        ]
        work += [work[-1]] * (padded - n)
        log.info(
            "Lucy Edit render",
            prompt=prompt,
            size=f"{render_w}x{render_h}",
            frames=padded,
            steps=steps,
        )
        video = self._pipe(
            video=work,
            prompt=prompt,
            negative_prompt=negative,
            height=render_h,
            width=render_w,
            num_frames=padded,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).frames[0]
        if not np.isfinite(video).all():
            raise RuntimeError("Lucy Edit produced non-finite output")
        rendered = (np.asarray(video) * 255).round().clip(0, 255).astype(np.uint8)
        out = [
            cv2.resize(rendered[i], (w, h), interpolation=cv2.INTER_LANCZOS4)
            if (render_h, render_w) != (h, w)
            else rendered[i]
            for i in range(n)
        ]
        return out, {
            "backend": self.name,
            "task": task,
            "prompt": prompt,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "render_size": f"{render_w}x{render_h}",
            "weights": str(Path(self._config.get("weights", ""))),
        }


_lucy: LucyEditReplacer | None = None


def get_lucy_replacer() -> LucyEditReplacer:
    global _lucy
    if _lucy is None:
        _lucy = LucyEditReplacer()
    return _lucy
