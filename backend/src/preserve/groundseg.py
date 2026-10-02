"""Text-grounded segmentation for subjects a closed-vocabulary detector misses.

YOLO is fast and precise but only knows COCO's 80 classes, and its confidence
drops on stylized or soft generated footage. Since the user names the subject in
free text, the mask builder needs a path that accepts arbitrary phrases.

CLIPSeg produces a coarse per-pixel relevance map for any text prompt. It is used
strictly as a fallback: when the detector finds the subject, its instance masks are
tighter and better temporally behaved. When it finds nothing, a coarse mask that
covers the right object beats failing the edit outright.

SAM 2 would give better boundaries, and validated-solution.md 4 cites it for
exactly this role; CLIPSeg is used here because it needs no point prompts and
resolves the subject from the text alone.
"""

import cv2
import numpy as np
import structlog
import torch
from numpy.typing import NDArray

from preserve.config import settings

log = structlog.get_logger()


class ClipSegGrounder:
    """Text-prompted coarse segmentation."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._config = settings.get_grounding_config()
        self._settings = self._config.get("settings", {})
        self.name = self._config.get("name", "CLIPSeg")

    def is_available(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        model_id = self._config.get("model", "CIDAS/clipseg-rd64-refined")
        log.info("Loading grounding model", model=model_id)

        self._processor = CLIPSegProcessor.from_pretrained(model_id)
        self._model = CLIPSegForImageSegmentation.from_pretrained(model_id)
        # CLIPSeg is small and its decoder is numerically touchy in half
        # precision, so it stays on CPU float32 rather than competing with the
        # video models for accelerator memory.
        self._model.eval()

    def relevance(
        self, frames: list[NDArray[np.uint8]], phrase: str, batch_size: int = 8
    ) -> NDArray[np.float32]:
        """Per-pixel relevance of phrase in every frame, (T, H, W) in [0, 1]."""
        if self._model is None:
            self.load()
        height, width = frames[0].shape[:2]
        heats = np.zeros((len(frames), height, width), dtype=np.float32)
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            inputs = self._processor(
                text=[phrase] * len(batch),
                images=batch,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits
            if logits.ndim == 2:
                logits = logits.unsqueeze(0)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for offset, heat in enumerate(probabilities):
                heats[start + offset] = cv2.resize(
                    heat, (width, height), interpolation=cv2.INTER_CUBIC
                )
        return heats

    def segment_frames(
        self,
        frames: list[NDArray[np.uint8]],
        phrase: str,
        multiple: bool = False,
    ) -> tuple[NDArray[np.uint8], dict]:
        """Build per-frame masks for a free-text phrase.

        Returns (masks, stats) with masks as (T, H, W) uint8, 255 where the
        subject was found.
        """
        if self._model is None:
            self.load()

        threshold = float(self._settings.get("threshold", 0.4))
        min_area_ratio = float(self._settings.get("min_area_ratio", 0.002))
        batch_size = int(self._settings.get("batch_size", 8))
        close_px = int(self._settings.get("close_px", 7))

        height, width = frames[0].shape[:2]
        min_area = max(1, int(min_area_ratio * height * width))
        masks = np.zeros((len(frames), height, width), dtype=np.uint8)
        found_frames = 0

        heat = self.relevance(frames, phrase, batch_size)
        if len(heat) >= 3:
            # Relevance flickers frame to frame; a max over the neighbours
            # keeps a subject that dips for one frame (audit 2026-09-21:
            # two caps traded places across frames).
            padded = np.concatenate([heat[:1], heat, heat[-1:]], axis=0)
            heat = np.maximum.reduce([padded[:-2], padded[1:-1], padded[2:]])
        components_kept = 0
        previous: NDArray[np.bool_] | None = None
        # Track outward from the frame with the strongest response, so the
        # instance chosen there holds in both directions; a forward-only
        # pass started on whichever instance frame 0 happened to favour
        # (logo-v14: frame 0 edited the other woman).
        start = int(np.argmax([float(h.max()) for h in heat])) if not multiple else 0
        order = list(range(start, len(heat))) + list(range(start - 1, -1, -1))
        for position, index in enumerate(order):
            resized = heat[index]
            if index == start - 1:
                previous = masks[start] > 0 if (masks[start] > 0).any() else None
            # Relevance maps are relative, so an absolute cut alone either
            # selects everything or nothing depending on the scene. Gate on
            # both the absolute threshold and the map's own dynamic range.
            peak = float(resized.max())
            if peak < threshold:
                continue
            binary = resized >= max(threshold, peak * 0.55)

            if close_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
                )
                binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0

            # Keep every blob that is large enough and lights up nearly as
            # strongly as the brightest one: "their caps" has two subjects,
            # while scattered background patches sharing the texture peak low.
            count, labels, statistics, _ = cv2.connectedComponentsWithStats(
                binary.astype(np.uint8), connectivity=8
            )
            candidates = [
                labels == label
                for label in range(1, count)
                if statistics[label, cv2.CC_STAT_AREA] >= min_area
            ]
            strong = [
                c for c in candidates if float(resized[c].max()) >= max(threshold, 0.75 * peak)
            ]
            if not candidates:
                continue
            if not multiple:
                # One subject asked for ("his cap" with two boys in frame):
                # follow the one chosen in the previous frame among every
                # candidate (it may have dipped below "strong" this frame,
                # which is how the logo edit jumped to the other woman), else
                # take the strongest.
                best = None
                if previous is not None:
                    overlaps = [float((c & previous).sum()) for c in candidates]
                    if max(overlaps) > 0:
                        best = candidates[int(np.argmax(overlaps))]
                if best is None:
                    pool = strong or candidates
                    best = pool[int(np.argmax([float(resized[c].max()) for c in pool]))]
                strong = [best]
            elif not strong:
                continue
            for component in strong:
                masks[index][component] = 255
            previous = masks[index] > 0
            found_frames += 1
            components_kept += len(strong)

        stats = {
            "grounding_backend": self.name,
            "grounding_phrase": phrase,
            "frames_with_detections": found_frames,
            "components_per_frame": round(components_kept / max(1, found_frames), 2),
            "coverage_pct": round(float(np.mean(masks > 0) * 100), 3),
        }
        log.info("Grounded segmentation", **stats)
        return masks, stats


_grounder: ClipSegGrounder | None = None


def get_grounder() -> ClipSegGrounder:
    """Process-wide grounder so the checkpoint loads only once."""
    global _grounder
    if _grounder is None:
        _grounder = ClipSegGrounder()
    return _grounder
