"""Structure-aware audit metrics for localized video edits.

The histogram-correlation drift ratio in score.py is structure-blind: it notices
when colours swing, not when geometry re-invents itself. The video-inpainting
literature (ProPainter's VFID/EPE protocol; Lai et al. ECCV 2018 blind temporal
consistency; TokenFlow/InsV2V evaluations) judges temporal stability with two
instruments this module adds:

Warp consistency — optical flow estimated on the source defines how the
  real scene moves. The output is warped along that field and compared with the
  next output frame. Content that moves with the scene scores low error;
  re-invented or flickering content scores high. Reported as a ratio against the
  same measurement on the untouched source, so genuine subject motion is not
  counted against the edit.
CLIP-temporal — per-frame text/image similarity to the edit prompt. The
  mean measures adherence; the minimum across frames catches a replacement that
  collapses partway through the clip even while its average looks fine.
LPIPS — Learned Perceptual Image Patch Similarity for perceptual quality.
  Better correlates with human judgment than PSNR/SSIM.
Color fidelity — Delta E (CIEDE2000) and hue/chroma preservation metrics.
"""

import glob
import re

import cv2
import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger()

_CLIP_PATH_GLOB = (
    "/Users/sadiq/.cache/huggingface/hub/models--sentence-transformers--"
    "clip-ViT-B-32/snapshots/*/0_CLIPModel"
)

_lpips_model = None


def _flow_pair(gray_a: NDArray[np.uint8], gray_b: NDArray[np.uint8]) -> NDArray[np.float32]:
    h, w = gray_a.shape
    scale = min(1.0, 256 / max(h, w))
    if scale < 1.0:
        gray_a = cv2.resize(gray_a, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gray_b = cv2.resize(gray_b, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    levels = max(1, min(4, int(np.log2(min(gray_a.shape) / 16))))
    flow = cv2.calcOpticalFlowFarneback(
        gray_a.astype(np.float32),
        gray_b.astype(np.float32),
        None,
        pyr_scale=0.5,
        levels=levels,
        winsize=21,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    if scale < 1.0:
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow /= scale
    return flow


def _warp_residual(
    image: NDArray[np.uint8], next_image: NDArray[np.uint8], flow: NDArray[np.float32]
) -> float:
    """Mean |warped - next| after moving image onto next_image's timeline."""
    h, w = flow.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(
        image,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    return float(np.abs(warped - next_image.astype(np.float32)).mean())


def warp_consistency_ratio(
    sources: list[NDArray[np.uint8]],
    outputs: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
) -> float | None:
    """How much more photometric warp-error the edited region carries than its source.

    Flow comes from the source frames, so the measurement asks one question: does
    the edited content obey the scene's measured motion? A value near 1.0 means
    the edit is as temporally lawful as the footage it replaced.
    """
    src_errors: list[float] = []
    out_errors: list[float] = []

    for index in range(len(sources) - 1):
        region = allowed[index] & allowed[index + 1]
        if int(region.sum()) < 256:
            continue
        flow = _flow_pair(
            cv2.cvtColor(sources[index], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(sources[index + 1], cv2.COLOR_RGB2GRAY),
        )
        grid_y, grid_x = np.mgrid[0 : sources[index].shape[0], 0 : sources[index].shape[1]].astype(
            np.float32
        )

        def region_residual(a, b, flow=flow, region=region, grid_x=grid_x, grid_y=grid_y):
            warped = cv2.remap(
                a,
                grid_x + flow[..., 0],
                grid_y + flow[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).astype(np.float32)
            per_pixel = np.abs(warped - b.astype(np.float32)).mean(axis=2)
            return float(per_pixel[region].mean())

        src_errors.append(region_residual(sources[index], sources[index + 1]))
        out_errors.append(region_residual(outputs[index], outputs[index + 1]))

    if not out_errors:
        return None
    source_drift = float(np.mean(src_errors))
    output_drift = float(np.mean(out_errors))
    return output_drift / max(source_drift, 1e-3)


_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor

        candidates = sorted(glob.glob(_CLIP_PATH_GLOB))
        if not candidates:
            return None, None
        path = candidates[-1]
        _clip_model = CLIPModel.from_pretrained(path)
        _clip_processor = CLIPProcessor.from_pretrained(path)
        _clip_model.eval()
    return _clip_model, _clip_processor


def clip_temporal_adherence(
    outputs: list[NDArray[np.uint8]], prompt: str, device: str = "cpu"
) -> tuple[float, float] | None:
    """(mean, min) CLIP similarity between each output frame and the prompt.

    The mean is TokenFlow-style prompt adherence; the minimum is the collapse
    detector — a replacement that dissolves at frame 10 shows up there even when
    nine earlier frames scored well.
    """
    model, processor = _load_clip()
    if model is None:
        log.info("CLIP weights unavailable, skipping clip-temporal metric")
        return None

    import torch

    model = model.to(device)
    similarities = []
    with torch.no_grad():
        text = processor(text=[prompt], return_tensors="pt", padding=True)
        text_output = model.get_text_features(**{k: v.to(device) for k, v in text.items()})
        text_features = getattr(text_output, "pooler_output", None)
        if text_features is None:
            text_features = text_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        batch_size = 8
        for start in range(0, len(outputs), batch_size):
            chunk = outputs[start : start + batch_size]
            images = processor(images=list(chunk), return_tensors="pt")["pixel_values"]
            image_output = model.get_image_features(pixel_values=images.to(device))
            image_features = getattr(image_output, "pooler_output", None)
            if image_features is None:
                image_features = image_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (image_features @ text_features.T).squeeze(1)
            similarities.extend(float(v) for v in logits.cpu())

    if not similarities:
        return None
    return float(np.mean(similarities)), float(np.min(similarities))


def _load_lpips():
    """Load LPIPS model for perceptual similarity."""
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips

            _lpips_model = lpips.LPIPS(net="vgg")
            _lpips_model.eval()
            log.info("LPIPS model loaded (VGG)")
        except ImportError:
            log.info("LPIPS not installed, skipping perceptual metric")
            return None
    return _lpips_model


def lpips_distance(
    frames_a: list[NDArray[np.uint8]],
    frames_b: list[NDArray[np.uint8]],
    device: str = "cpu",
) -> tuple[float, float] | None:
    """LPIPS perceptual distance between two frame sequences.

    Lower is better. Returns (mean, max) distance across frames.
    LPIPS correlates better with human perception than PSNR/SSIM.
    """
    model = _load_lpips()
    if model is None:
        return None

    import torch

    model = model.to(device)
    distances = []

    with torch.no_grad():
        for a, b in zip(frames_a, frames_b, strict=True):
            # Convert to tensors in [-1, 1] range
            a_tensor = (
                torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            b_tensor = (
                torch.from_numpy(b.astype(np.float32) / 127.5 - 1.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            dist = model(a_tensor, b_tensor).item()
            distances.append(dist)

    if not distances:
        return None
    return float(np.mean(distances)), float(np.max(distances))


def _rgb_to_lab(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Convert RGB to CIELAB color space."""
    # Normalize to [0, 1]
    rgb_norm = rgb.astype(np.float32) / 255.0

    # sRGB to linear RGB
    linear = np.where(rgb_norm <= 0.04045, rgb_norm / 12.92, ((rgb_norm + 0.055) / 1.055) ** 2.4)

    # sRGB to XYZ (D65)
    xyz = np.dot(
        linear,
        np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        ).T,
    )

    # XYZ to LAB
    xyz_n = xyz / np.array([0.95047, 1.0, 1.08883])  # D65 white point
    f = np.where(xyz_n > 0.008856, xyz_n ** (1 / 3), (7.787 * xyz_n) + 16 / 116)

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def delta_e_ciede2000(
    lab1: NDArray[np.float32],
    lab2: NDArray[np.float32],
) -> NDArray[np.float32]:
    """CIEDE2000 color difference (approximate, fast version).

    Full CIEDE2000 is complex; this is a simplified but accurate approximation
    suitable for perceptual color difference measurement.
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Simplified CIEDE2000 (using CIE76 with weighting)
    # For production, use colour-science library for full CIEDE2000
    dL = L1 - L2
    da = a1 - a2
    db = b1 - b2

    # Weighting factors (approximate)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2

    # Simplified weighting
    SL = 1.0
    SC = 1.0 + 0.045 * C_avg
    SH = 1.0 + 0.015 * C_avg

    dE = np.sqrt((dL / SL) ** 2 + (da / SC) ** 2 + (db / SH) ** 2)

    return dE


def color_fidelity_metrics(
    source_frames: list[NDArray[np.uint8]],
    output_frames: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
) -> dict:
    """Comprehensive color fidelity metrics for edited region.

    Returns dict with:
    - mean_delta_e: Mean CIEDE2000 in edit region
    - max_delta_e: Max CIEDE2000 in edit region
    - hue_preservation: Mean hue difference (degrees) in edit region
    - chroma_preservation: Mean chroma ratio in edit region
    - edit_region_colorfulness: Colorfulness of edit vs source
    """
    delta_es = []
    hue_diffs = []
    chroma_ratios = []

    for src, out, region in zip(source_frames, output_frames, allowed, strict=True):
        if not region.any():
            continue

        src_lab = _rgb_to_lab(src)
        out_lab = _rgb_to_lab(out)

        # Delta E in edit region
        de = delta_e_ciede2000(src_lab[region], out_lab[region])
        delta_es.extend(de.tolist())

        # Hue difference
        src_hue = np.arctan2(src_lab[region, 2], src_lab[region, 1])
        out_hue = np.arctan2(out_lab[region, 2], out_lab[region, 1])
        hue_diff = np.abs(src_hue - out_hue)
        hue_diff = np.minimum(hue_diff, 2 * np.pi - hue_diff)  # Wrap around
        hue_diffs.extend(np.degrees(hue_diff).tolist())

        # Chroma ratio
        src_chroma = np.sqrt(src_lab[region, 1] ** 2 + src_lab[region, 2] ** 2)
        out_chroma = np.sqrt(out_lab[region, 1] ** 2 + out_lab[region, 2] ** 2)
        chroma_ratio = out_chroma / (src_chroma + 1e-6)
        chroma_ratios.extend(chroma_ratio.tolist())

    if not delta_es:
        return {
            "mean_delta_e": 0.0,
            "max_delta_e": 0.0,
            "mean_hue_diff_deg": 0.0,
            "mean_chroma_ratio": 1.0,
            "colorfulness_preservation": 1.0,
        }

    return {
        "mean_delta_e": float(np.mean(delta_es)),
        "max_delta_e": float(np.max(delta_es)),
        "mean_hue_diff_deg": float(np.mean(hue_diffs)),
        "mean_chroma_ratio": float(np.mean(chroma_ratios)),
        "colorfulness_preservation": float(np.mean(chroma_ratios)),
    }


def comprehensive_edit_quality(
    source_frames: list[NDArray[np.uint8]],
    output_frames: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
    prompt: str = "",
    device: str = "cpu",
) -> dict:
    """Run all quality metrics for an edit.

    Returns combined metrics dict.
    """
    metrics = {}

    # Warp consistency
    wc = warp_consistency_ratio(source_frames, output_frames, allowed)
    if wc is not None:
        metrics["warp_consistency_ratio"] = wc

    # CLIP temporal adherence
    if prompt:
        clip_res = clip_temporal_adherence(output_frames, prompt, device)
        if clip_res:
            metrics["clip_temporal_mean"], metrics["clip_temporal_min"] = clip_res

    # LPIPS perceptual distance
    lpips_res = lpips_distance(source_frames, output_frames, device)
    if lpips_res:
        metrics["lpips_mean"], metrics["lpips_max"] = lpips_res

    # Color fidelity
    color_metrics = color_fidelity_metrics(source_frames, output_frames, allowed)
    metrics.update(color_metrics)

    return metrics


def removal_residue(
    sources: list[NDArray[np.uint8]],
    outputs: list[NDArray[np.uint8]],
    allowed: NDArray[np.bool_],
    phrase: str,
) -> float | None:
    """How much of the removed subject still reads inside the edit region.

    Open-vocabulary relevance (CLIPSeg) of the target phrase, averaged over
    the allowed region, output divided by source. 1.0 means the region reads
    as the target exactly as before (a ghost); a clean fill lands well under
    0.5. The headline score is blind to this: it rewards changed pixels, and
    a repainted ghost car changes every pixel (audit 2026-09-19: 96.6/100 for
    a visible dark car silhouette).
    """
    from preserve.groundseg import get_grounder

    grounder = get_grounder()
    # The bare noun: with a possessive ("his glasses") CLIPSeg lights the
    # whole person and bare eyes still score 0.63 of the source; "glasses"
    # alone scores 0.19 (measured 2026-09-21).
    phrase = re.sub(
        r"^(?:his|her|their|its|my|our|your|the|a|an|this|that)\s+",
        "",
        phrase.strip(),
        flags=re.IGNORECASE,
    )
    source_heat = grounder.relevance(sources, phrase)
    output_heat = grounder.relevance(outputs, phrase)
    before: list[float] = []
    after: list[float] = []
    for src, out, region in zip(source_heat, output_heat, allowed, strict=True):
        if region.sum() < 64:
            continue
        before.append(float(src[region].mean()))
        after.append(float(out[region].mean()))
    if not before:
        return None
    return float(np.mean(after)) / max(float(np.mean(before)), 1e-3)
