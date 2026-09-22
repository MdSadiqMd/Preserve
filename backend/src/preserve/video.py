"""Video I/O and metadata extraction."""

import hashlib
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from preserve.models import VideoMetadata


def compute_file_hash(path: Path, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def probe_video(path: Path) -> VideoMetadata:
    """Extract video metadata without decoding frames."""
    file_hash = compute_file_hash(path)

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 24)
        duration_ms = int((stream.duration or 0) * stream.time_base * 1000)
        if duration_ms == 0 and container.duration:
            duration_ms = int(container.duration / 1000)

        # Extract full ffprobe data for color management
        ffprobe_data = {}
        try:
            codec_context = stream.codec_context
            ffprobe_data = {
                "codec_name": codec_context.name,
                "codec_long_name": getattr(codec_context, "long_name", None),
                "profile": getattr(codec_context, "profile", None),
                "level": getattr(codec_context, "level", None),
                "pix_fmt": stream.pix_fmt,
                "color_range": getattr(codec_context, "color_range", None),
                "color_space": getattr(codec_context, "color_space", None),
                "color_primaries": getattr(codec_context, "color_primaries", None),
                "color_trc": getattr(codec_context, "color_trc", None),
                "chroma_location": getattr(codec_context, "chroma_location", None),
                "field_order": getattr(codec_context, "field_order", None),
                "refs": getattr(codec_context, "refs", None),
                "has_b_frames": getattr(codec_context, "has_b_frames", None),
            }
        except Exception:
            pass

        meta = VideoMetadata(
            id=file_hash[:16],
            filename=path.name,
            path=path,
            width=stream.width,
            height=stream.height,
            fps=fps,
            frame_count=stream.frames or int(duration_ms * fps / 1000),
            duration_ms=duration_ms,
            codec=stream.codec_context.name,
            pixel_format=stream.pix_fmt or "unknown",
            file_hash=file_hash,
            ffprobe_data=ffprobe_data if ffprobe_data else None,
        )

    # Load generation provenance from sidecar if available
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        import contextlib
        import json

        with contextlib.suppress(Exception):
            meta.generation = json.loads(sidecar.read_text())

    return meta


def extract_frames(
    path: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    step: int = 1,
) -> list[NDArray[np.uint8]]:
    """Extract frames as RGB numpy arrays."""
    frames: list[NDArray[np.uint8]] = []

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for i, frame in enumerate(container.decode(video=0)):
            if i < start_frame:
                continue
            if end_frame is not None and i >= end_frame:
                break
            if (i - start_frame) % step != 0:
                continue

            rgb = frame.to_ndarray(format="rgb24")
            frames.append(rgb)

    return frames


def extract_frame_at_pts(path: Path, pts_ms: int) -> NDArray[np.uint8]:
    """Extract a single frame at approximate presentation timestamp."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        target_pts = int(pts_ms / 1000 / time_base)

        container.seek(target_pts, stream=stream)
        for frame in container.decode(video=0):
            return frame.to_ndarray(format="rgb24")

    raise ValueError(f"Could not extract frame at {pts_ms}ms")


def has_audio_stream(path: Path) -> bool:
    try:
        with av.open(str(path)) as container:
            return len(container.streams.audio) > 0
    except av.AVError:
        return False


def _mux_audio(video_path: Path, audio_source: Path, output_path: Path) -> None:
    """Copy the audio stream from audio_source alongside video_path's video.

    Audio carries its own timestamps, so this stays in sync even when the edited
    video was written at a different frame rate, as long as total duration is
    unchanged.
    """
    with (
        av.open(str(video_path)) as vin,
        av.open(str(audio_source)) as ain,
        av.open(str(output_path), mode="w") as out,
    ):
        in_video = vin.streams.video[0]
        in_audio = ain.streams.audio[0]

        out_video = out.add_stream_from_template(in_video)
        out_audio = out.add_stream_from_template(in_audio)

        for packet in vin.demux(in_video):
            if packet.dts is None:
                continue
            packet.stream = out_video
            out.mux(packet)

        for packet in ain.demux(in_audio):
            if packet.dts is None:
                continue
            packet.stream = out_audio
            out.mux(packet)


def write_lossless_master(
    frames: list[NDArray[np.uint8]],
    output_path: Path,
    fps: float,
    audio_source: Path | None = None,
) -> None:
    """Write frames so that decoding them returns the exact same samples.

    validated-solution.md 8.2 requires an exact flattened master for a
    decoded-sample guarantee: an ordinary H.264 encode changes pixels everywhere
    through quantisation and RGB->YCbCr conversion, so the preservation invariant
    cannot be demonstrated on it no matter how careful the compositing was.

    libx264rgb at qp=0 codes RGB directly and losslessly, avoiding the chroma
    conversion that makes even "lossless" yuv420p/yuv444p round trips inexact.
    Browsers cannot decode RGB H.264, which is why a separate playable derivative
    is written alongside this master.
    """
    write_frames(
        frames,
        output_path,
        fps,
        codec="libx264rgb",
        pixel_format="rgb24",
        encoder_options={"qp": "0", "preset": "veryfast"},
        audio_source=audio_source,
    )


def write_playable(
    frames: list[NDArray[np.uint8]],
    output_path: Path,
    fps: float,
    audio_source: Path | None = None,
) -> None:
    """Write a browser-playable file that is visually lossless.

    The preservation guarantee lives on the RGB master, but the master is not
    browser-decodable, so this is what a user actually watches. An ordinary CRF
    encode re-quantises the whole frame — measured here, CRF 18 perturbs ~24%
    of pixels even when the input is byte-identical to the source, which reads as
    the untouched background shifting in colour and contrast.

    qp=0 is mathematically lossless in the coded 4:2:0 space, so the only
    residual is the RGB->YCbCr 4:2:0 chroma subsample itself, which is
    imperceptible and confined to coloured edges. yuv420p keeps it playable
    everywhere, unlike the 4:4:4 the RGB master needs.
    """
    write_frames(
        frames,
        output_path,
        fps,
        codec="libx264",
        pixel_format="yuv420p",
        encoder_options={"qp": "0", "preset": "veryfast"},
        audio_source=audio_source,
    )


def write_frames(
    frames: list[NDArray[np.uint8]],
    output_path: Path,
    fps: float,
    codec: str = "libx264",
    crf: int = 18,
    pixel_format: str = "yuv420p",
    audio_source: Path | None = None,
    encoder_options: dict[str, str] | None = None,
) -> None:
    """Write frames to a video file, optionally carrying over source audio.

    When audio_source is given and has an audio track, video is encoded to a
    temporary file first and then remuxed with the original audio, which avoids
    interleaving encoded video with copied audio packets in one pass.
    """
    if not frames:
        raise ValueError("No frames to write")

    copy_audio = audio_source is not None and has_audio_stream(audio_source)
    target = output_path.with_suffix(".video.mp4") if copy_audio else output_path

    h, w = frames[0].shape[:2]

    with av.open(str(target), mode="w") as container:
        stream = container.add_stream(codec, rate=Fraction(fps).limit_denominator(10000))
        stream.width = w
        stream.height = h
        stream.pix_fmt = pixel_format
        stream.options = encoder_options or {"crf": str(crf)}

        for rgb in frames:
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    if copy_audio:
        try:
            _mux_audio(target, audio_source, output_path)
        finally:
            target.unlink(missing_ok=True)


def write_frames_lossless(
    frames: list[NDArray[np.uint8]],
    output_dir: Path,
    prefix: str = "frame",
) -> list[Path]:
    """Write frames as lossless PNG sequence."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for i, rgb in enumerate(frames):
        path = output_dir / f"{prefix}_{i:06d}.png"
        Image.fromarray(rgb).save(path, compress_level=1)
        paths.append(path)

    return paths
