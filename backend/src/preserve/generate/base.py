"""Interface for text-to-video generation backends."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str | None = None
    num_frames: int = 16
    height: int = 512
    width: int = 512
    fps: int = 8
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    seed: int | None = None


@dataclass
class GenerationResult:
    frames: list[NDArray[np.uint8]]
    fps: int
    # Everything needed to reproduce or audit the run, per validated-solution.md 5.6.
    provenance: dict = field(default_factory=dict)


class GenerationBackend(Protocol):
    """A text-to-video model."""

    name: str

    def is_available(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...
