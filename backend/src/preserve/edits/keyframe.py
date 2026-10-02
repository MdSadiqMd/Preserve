"""Keyframe amodal completion: an image editor proposes one frame's fill.

Removing an attached object (cap, glasses, popsicle) exposes a surface no
frame of the clip ever shows, so the fill has to be invented with semantics
(hair under the cap, eyes behind the glasses). The video models available
here do not do that (audit 2026-09-20); image editors do. This module edits a
single crop, and the caller hands the result to VACE as an anchored frame so
the video pass propagates it instead of inventing its own.

Backends: SD1.5 inpainting (cached, masked fill from a description) and
FLUX.2 klein 4B (instruction editing, reference image in, no mask).
"""

import cv2
import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from PIL import Image

from preserve.config import settings

log = structlog.get_logger()


def _snap(v: int, m: int) -> int:
    return max(m, int(round(v / m)) * m)


def _shapeless_hole(region: NDArray[np.bool_], grow_ratio: float = 0.06) -> NDArray[np.bool_]:
    """Convex hull of the region, dilated: the hole an inpainting model fills.

    A masked model reads the hole's silhouette as the object to paint (audit
    2026-09-20: SD1.5 filled a cap-shaped hole with three different caps
    whatever the prompt said). A rounded hull with the brim outline gone gives
    the prompt a say. Only the true region is pasted back, so this never
    widens the edit.
    """
    points = np.argwhere(region)[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(points)
    canvas = np.zeros(region.shape, np.uint8)
    cv2.fillConvexPoly(canvas, hull, 255)
    _, _, w, h = cv2.boundingRect(points)
    grow = max(4, int(grow_ratio * max(w, h)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    return cv2.dilate(canvas, kernel) > 0


def _align_to(
    rendered: NDArray[np.uint8], frame: NDArray[np.uint8], support: NDArray[np.bool_]
) -> NDArray[np.uint8]:
    """Shift rendered so its unedited pixels register with frame (sub-pixel).

    An instruction editor re-renders the whole image and may return it a
    pixel or two off; pasting a shifted region back leaves a jagged seam at
    the boundary (audit 2026-09-20). Phase correlation over the untouched
    area gives the global shift; anything over 6px is treated as a failed
    estimate and ignored.
    """
    weight = support.astype(np.float32)
    a = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) * weight
    b = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY).astype(np.float32) * weight
    (dx, dy), response = cv2.phaseCorrelate(b, a)
    if response < 0.1 or max(abs(dx), abs(dy)) > 6.0 or max(abs(dx), abs(dy)) < 0.05:
        return rendered
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
    return cv2.warpAffine(
        rendered,
        matrix,
        (rendered.shape[1], rendered.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


class KeyframeEditor:
    def __init__(self) -> None:
        config = settings.get_keyframe_config()
        self.backend = str(config.get("default", "sd_inpaint"))
        self._config = config.get("backend", {})
        self._settings = self._config.get("settings", {})
        self._pipe = None

    def unload(self) -> None:
        self._pipe = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _load_sd(self):
        from diffusers import StableDiffusionInpaintPipeline

        device = settings.get_device()
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self._config.get("model", "stable-diffusion-v1-5/stable-diffusion-inpainting"),
            torch_dtype=torch.float32 if device.type == "mps" else torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=False,
        )
        pipe.set_progress_bar_config(disable=True)
        return pipe.to(device)

    def _load_klein(self):
        from diffusers import Flux2KleinPipeline

        model_dir = settings.model_dir / self._config.get("model_dir", "flux2-klein-4b")
        pipe = Flux2KleinPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
        pipe.set_progress_bar_config(disable=True)
        return pipe.to(settings.get_device())

    def complete(
        self, frame: NDArray[np.uint8], mask: NDArray[np.uint8], prompt: str, negative: str = ""
    ) -> NDArray[np.uint8]:
        """Fill mask>0 on this frame with prompt semantics; pixels outside are the frame's own."""
        rendered = self.render(frame, mask, prompt, negative)
        region = mask > 0
        result = frame.copy()
        result[region] = rendered[region]
        return result

    @torch.no_grad()
    def render(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8],
        prompt: str,
        negative: str = "",
        erase_first: bool | None = None,
        erase: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """The editor's full-frame proposal, registered to frame; the caller picks the region.

        erase: pixels to paint over classically regardless of erase_first (a
        brim's shadow on the forehead, marked SHADOW_VALUE in the matte).
        erase_first: paint the region over with a classical inpaint before the
        reference-conditioned editor sees it. A removal instruction otherwise
        keeps the object's structure as an embossed ghost (audit 2026-09-21:
        logo ring on a plain tee); with the structure gone the editor only has
        to make the surface plausible. Default from settings.erase_first.
        """
        if erase_first is None:
            erase_first = bool(self._settings.get("erase_first", False))
        to_erase = (mask > 0) if erase_first else (erase > 0 if erase is not None else None)
        if to_erase is not None and to_erase.any():
            frame = cv2.inpaint(frame, to_erase.astype(np.uint8), 7, cv2.INPAINT_TELEA)
        if self._pipe is None:
            log.info("Loading keyframe editor", backend=self.backend)
            self._pipe = self._load_klein() if self.backend == "klein" else self._load_sd()
        h, w = frame.shape[:2]
        seed = int(self._settings.get("seed", 7))
        generator = torch.Generator("cpu").manual_seed(seed)
        region = mask > 0
        if self.backend == "klein":
            render_h, render_w = _snap(h, 16), _snap(w, 16)
            image = Image.fromarray(frame).resize((render_w, render_h), Image.LANCZOS)
            out = self._pipe(
                image=[image],
                prompt=prompt,
                height=render_h,
                width=render_w,
                num_inference_steps=int(self._settings.get("num_inference_steps", 4)),
                guidance_scale=float(self._settings.get("guidance_scale", 1.0)),
                generator=generator,
            ).images[0]
        else:
            render_dim = int(self._settings.get("render_dim", 512))
            scale = render_dim / max(h, w)
            render_h, render_w = _snap(int(h * scale), 8), _snap(int(w * scale), 8)
            image = Image.fromarray(frame).resize((render_w, render_h), Image.LANCZOS)
            hole = _shapeless_hole(region)
            mask_image = Image.fromarray(np.where(hole, 255, 0).astype(np.uint8)).resize(
                (render_w, render_h), Image.NEAREST
            )
            out = self._pipe(
                prompt=prompt,
                negative_prompt=negative or None,
                image=image,
                mask_image=mask_image,
                height=render_h,
                width=render_w,
                strength=1.0,
                num_inference_steps=int(self._settings.get("num_inference_steps", 30)),
                guidance_scale=float(self._settings.get("guidance_scale", 7.5)),
                generator=generator,
            ).images[0]
        rendered = cv2.resize(
            np.asarray(out.convert("RGB")), (w, h), interpolation=cv2.INTER_LANCZOS4
        )
        if self.backend == "klein":
            rendered = _align_to(rendered, frame, ~region)
        return rendered


_editor: KeyframeEditor | None = None


def get_keyframe_editor() -> KeyframeEditor:
    global _editor
    if _editor is None:
        _editor = KeyframeEditor()
    return _editor
