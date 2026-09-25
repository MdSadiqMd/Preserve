"""FastAPI application."""

import io
from pathlib import Path
from uuid import UUID

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image

from preserve.config import settings
from preserve.generate.service import GenerationService
from preserve.models import (
    EditJob,
    EditRequest,
    GenerateRequest,
    GenerationJob,
    JobStatus,
    VideoMetadata,
)
from preserve.pipeline import EditPipeline
from preserve.video import extract_frame_at_pts

log = structlog.get_logger()

app = FastAPI(
    title="Preserve",
    description="Localized AI video editing with pixel-level preservation guarantees",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

videos: dict[str, VideoMetadata] = {}
jobs: dict[UUID, EditJob] = {}
generations: dict[UUID, GenerationJob] = {}
pipeline = EditPipeline()
generator = GenerationService()


@app.on_event("startup")
async def startup():
    settings.ensure_dirs()
    device = settings.get_device()
    log.info("Preserve API starting", device=str(device))


@app.get("/health")
async def health():
    device = settings.get_device()
    return {
        "status": "ok",
        "device": str(device),
        "backend": settings.inpaint_backend,
    }


def _run_generation(job_id: UUID) -> None:
    job = generations[job_id]
    finished, meta = generator.process(job)
    generations[job_id] = finished
    if meta is not None:
        # A generated clip becomes an immutable source asset that edits address by id.
        videos[meta.id] = meta


@app.post("/generate")
async def create_generation(
    request: GenerateRequest, background_tasks: BackgroundTasks
) -> GenerationJob:
    """Generate a video from a text prompt."""
    job = GenerationJob(request=request)
    generations[job.id] = job
    background_tasks.add_task(_run_generation, job.id)
    log.info("Generation queued", job_id=str(job.id), prompt=request.prompt)
    return job


@app.get("/generate/{job_id}")
async def get_generation(job_id: UUID) -> GenerationJob:
    if job_id not in generations:
        raise HTTPException(404, "Generation job not found")
    return generations[job_id]


@app.get("/generate")
async def list_generations(limit: int = 20) -> list[GenerationJob]:
    ordered = sorted(generations.values(), key=lambda j: j.created_at, reverse=True)
    return ordered[:limit]


@app.get("/videos/{video_id}")
async def get_video(video_id: str) -> VideoMetadata:
    """Get video metadata."""
    if video_id not in videos:
        raise HTTPException(404, "Video not found")
    return videos[video_id]


@app.get("/videos/{video_id}/stream")
async def stream_video(video_id: str):
    """Stream the original video file."""
    if video_id not in videos:
        raise HTTPException(404, "Video not found")

    meta = videos[video_id]
    if not meta.path.exists():
        raise HTTPException(404, "Video file not found")

    return FileResponse(
        meta.path,
        media_type="video/mp4",
        filename=meta.filename,
    )


@app.get("/videos/{video_id}/frame/{frame_ms}")
async def get_frame(video_id: str, frame_ms: int):
    """Get a single frame as JPEG."""
    if video_id not in videos:
        raise HTTPException(404, "Video not found")

    meta = videos[video_id]
    frame = extract_frame_at_pts(meta.path, frame_ms)

    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg",
    )


@app.post("/jobs")
async def create_job(request: EditRequest, background_tasks: BackgroundTasks) -> EditJob:
    """Create a new edit job."""
    if request.video_id not in videos:
        raise HTTPException(404, "Video not found")

    video_meta = videos[request.video_id]

    if request.time_range.end_ms > video_meta.duration_ms:
        raise HTTPException(400, "Time range exceeds video duration")

    job = EditJob(request=request)
    jobs[job.id] = job

    background_tasks.add_task(run_job, job.id, video_meta)

    log.info("Job created", job_id=str(job.id))
    return job


def run_job(job_id: UUID, video_meta: VideoMetadata):
    """Run job in background (sync, called from background task)."""
    job = jobs[job_id]

    try:
        pipeline.process(job, video_meta)
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        log.exception("Job failed", job_id=str(job_id))


@app.get("/jobs/{job_id}")
async def get_job(job_id: UUID) -> EditJob:
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: UUID):
    """Download job result video."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job not completed: {job.status}")

    if not job.result or "output_path" not in job.result:
        raise HTTPException(500, "No output path")

    output_path = Path(job.result["output_path"])
    if not output_path.exists():
        raise HTTPException(500, "Output file missing")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"preserve_{job_id}.mp4",
    )


@app.get("/jobs")
async def list_jobs(limit: int = 20) -> list[EditJob]:
    """List recent jobs."""
    sorted_jobs = sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)
    return sorted_jobs[:limit]


@app.get("/config")
async def get_config():
    """Get current configuration."""
    backend_config = settings.get_backend_config()
    return {
        "device": str(settings.get_device()),
        "dtype": settings.dtype,
        "inpaint_backend": settings.inpaint_backend,
        "backend_name": backend_config.get("name", "Unknown"),
        "backend_description": backend_config.get("description", ""),
        "max_video_duration_seconds": settings.max_video_duration_seconds,
        "max_video_resolution": settings.max_video_resolution,
    }
