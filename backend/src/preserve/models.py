from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EditType(StrEnum):
    REMOVAL = "removal"
    REPLACEMENT = "replacement"
    INPAINT = "inpaint"


class MaskType(StrEnum):
    POINTS = "points"
    BOX = "box"
    SEGMENTATION = "segmentation"
    UPLOADED = "uploaded"


class TimeRange(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class Point(BaseModel):
    x: float
    y: float
    frame: int


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    frame: int


class MaskDefinition(BaseModel):
    mask_type: MaskType
    points: list[Point] | None = None
    box: BoundingBox | None = None
    segmentation_prompt: str | None = None
    mask_file_id: str | None = None
    dilation_px: int = Field(default=5, ge=0, le=50)
    feather_px: int = Field(default=3, ge=0, le=20)


class EditRequest(BaseModel):
    video_id: str
    edit_type: EditType
    time_range: TimeRange
    mask: MaskDefinition
    prompt: str | None = None
    negative_prompt: str | None = None
    reference_image_id: str | None = None
    denoise_strength: float = Field(default=0.8, ge=0.1, le=1.0)
    seed: int | None = None
    context_frames: int = Field(default=8, ge=2, le=24)


class EditJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    status: JobStatus = JobStatus.PENDING
    request: EditRequest
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float = 0.0
    message: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class VideoMetadata(BaseModel):
    id: str
    filename: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_ms: int
    codec: str
    pixel_format: str
    file_hash: str
    # Full ffprobe/av metadata for color management and verification
    ffprobe_data: dict[str, Any] | None = None
    # Set for videos this system generated, so the UI can show where a clip came
    # from and edits can record what they were applied to.
    generation: dict[str, Any] | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    num_frames: int = Field(default=16, ge=8, le=128)
    fps: int = Field(default=8, ge=4, le=30)
    height: int = Field(default=384, ge=256, le=1280)
    width: int = Field(default=384, ge=256, le=1280)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=20, ge=8, le=60)
    seed: int | None = None


class GenerationJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    status: JobStatus = JobStatus.PENDING
    request: GenerateRequest
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float = 0.0
    message: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class PreservationReport(BaseModel):
    job_id: UUID
    total_frames: int
    edited_frames: int
    protected_frames: int
    max_diff_outside_mask: float
    mean_diff_outside_mask: float
    changed_pixels_outside_mask: int
    passed: bool
    details: list[dict[str, Any]]
