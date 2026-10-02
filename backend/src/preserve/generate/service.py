"""Generate a video and register it as an immutable source asset.

validated-solution.md 3.1 requires the source to be locked and hashed before any
edit touches it. A generated clip is no different: once written it is the source
of truth for every protected pixel, so it is hashed, probed, and stored read-only,
with the full generation provenance kept alongside it (5.6).

The canonical source is the decoded generated file, not the model's in-memory
tensor. validated-solution.md 2 defines the practical contract as "protected
decoded samples unchanged", so the file is written once at high quality and every
later comparison decodes it back. That keeps the asset browser-playable while
leaving the invariant exact, because both sides of the comparison come from the
same decode.
"""

import os
from datetime import datetime
from pathlib import Path

import structlog

from preserve.config import _load_models_config, settings
from preserve.generate.animatediff import create_animatediff_backend
from preserve.generate.base import GenerationBackend, GenerationRequest
from preserve.generate.wan import create_wan_backend
from preserve.models import GenerateRequest, GenerationJob, JobStatus, VideoMetadata
from preserve.video import probe_video, write_playable

log = structlog.get_logger()

# Near-transparent quality for the source asset; the edit path never re-encodes
# protected pixels, it copies them from this file's decode.
SOURCE_CRF = 12


class GenerationService:
    """Runs text-to-video jobs and locks their output as source assets."""

    def __init__(self) -> None:
        self._backend: GenerationBackend | None = None

    def get_backend(self) -> GenerationBackend:
        if self._backend is None:
            default = _load_models_config().get("generation", {}).get("default", "animatediff")
            if default == "wan":
                self._backend = create_wan_backend()
            else:
                self._backend = create_animatediff_backend()
        if not self._backend.is_available():
            log.info("Loading generation backend", backend=self._backend.name)
            self._backend.load()
        return self._backend

    def process(self, job: GenerationJob) -> tuple[GenerationJob, VideoMetadata | None]:
        request = job.request
        log.info("Starting generation", job_id=str(job.id), prompt=request.prompt)

        try:
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow()
            job.message = "Loading generation model"
            job.progress = 0.1

            backend = self.get_backend()

            job.message = "Generating video"
            job.progress = 0.25

            result = backend.generate(
                GenerationRequest(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_frames=request.num_frames,
                    height=request.height,
                    width=request.width,
                    fps=request.fps,
                    guidance_scale=request.guidance_scale,
                    num_inference_steps=request.num_inference_steps,
                    seed=request.seed,
                )
            )

            job.progress = 0.85
            job.message = "Writing source asset"

            settings.ensure_dirs()
            path = settings.upload_dir / f"generated_{job.id.hex[:16]}.mp4"

            # Visually lossless and browser-playable. This clip is the reference
            # every edit preserves against, so its background must not carry
            # compression artefacts that a later edit would then be blamed for.
            write_playable(result.frames, path, float(result.fps))

            meta = probe_video(path)
            meta.generation = {
                "job_id": str(job.id),
                "created_at": job.started_at.isoformat(),
                **result.provenance,
            }

            # Lock the source: the immutability requirement is only meaningful if
            # the file cannot be rewritten in place afterwards.
            os.chmod(path, 0o444)

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 1.0
            job.message = "Generation complete"
            job.result = {
                "video_id": meta.id,
                "output_path": str(path),
                "width": meta.width,
                "height": meta.height,
                "fps": meta.fps,
                "frame_count": meta.frame_count,
                "duration_ms": meta.duration_ms,
                "file_hash": meta.file_hash,
                "provenance": result.provenance,
            }

            log.info(
                "Generation complete",
                job_id=str(job.id),
                video_id=meta.id,
                frames=meta.frame_count,
                path=str(path),
            )
            return job, meta

        except Exception as exc:
            log.exception("Generation failed", job_id=str(job.id))
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error = str(exc)
            return job, None


def generate_to_path(request: GenerateRequest, path: Path) -> VideoMetadata:
    """Generate straight to a chosen path. Used by tests and scripts."""
    service = GenerationService()
    backend = service.get_backend()
    result = backend.generate(
        GenerationRequest(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=request.num_frames,
            height=request.height,
            width=request.width,
            fps=request.fps,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
        path.unlink()
    write_playable(result.frames, path, float(result.fps))
    meta = probe_video(path)
    meta.generation = result.provenance
    # Persist provenance for reuse mode
    _write_generation_provenance(path, result.provenance)
    return meta


def _write_generation_provenance(video_path: Path, provenance: dict) -> None:
    """Write generation provenance as a sidecar JSON for reuse."""
    sidecar = video_path.with_suffix(".json")
    import json

    sidecar.write_text(json.dumps(provenance, indent=2))
