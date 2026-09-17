"""Base interface for inpainting backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class InpaintRequest:
    frames: list[NDArray[np.uint8]]
    masks: NDArray[np.uint8]
    prompt: str | None = None
    negative_prompt: str | None = None
    reference_image: NDArray[np.uint8] | None = None
    denoise_strength: float = 0.8
    seed: int | None = None


@dataclass
class InpaintResult:
    frames: list[NDArray[np.uint8]]
    confidence: NDArray[np.float32] | None = None
    metadata: dict | None = None


class InpaintBackend(ABC):
    """Base class for video inpainting backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (model loaded, etc)."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model to free memory."""
        ...

    @abstractmethod
    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        """Perform video inpainting."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging/display."""
        ...

    @property
    @abstractmethod
    def supports_mps(self) -> bool:
        """Whether this backend supports Apple MPS."""
        ...
