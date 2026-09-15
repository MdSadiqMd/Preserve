from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_models_config() -> dict[str, Any]:
    config_path = Path(__file__).parent / "models_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRESERVE_", env_file=".env")

    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    upload_dir: Path = Field(default=Path("./data/uploads"))
    output_dir: Path = Field(default=Path("./data/outputs"))
    cache_dir: Path = Field(default=Path("./data/cache"))
    model_dir: Path = Field(default=Path("./models"))

    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"

    inpaint_backend: str = "propainter"
    segmentation_backend: str = "yolo"

    max_video_duration_seconds: float = 30.0
    max_video_resolution: int = 1280
    default_fps: float = 24.0

    def ensure_dirs(self) -> None:
        for d in [self.upload_dir, self.output_dir, self.cache_dir, self.model_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_device(self) -> torch.device:
        if self.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device)

    def get_dtype(self) -> torch.dtype:
        if self.dtype == "bfloat16":
            return torch.bfloat16
        elif self.dtype == "float16":
            return torch.float16
        return torch.float32

    def get_backend_config(self, backend_name: str | None = None) -> dict[str, Any]:
        name = backend_name or self.inpaint_backend
        models_config = _load_models_config()
        backends = models_config.get("inpainting", {}).get("backends", {})
        return backends.get(name, {})

    def get_segmentation_config(self, backend_name: str | None = None) -> dict[str, Any]:
        name = backend_name or self.segmentation_backend
        models_config = _load_models_config()
        backends = models_config.get("segmentation", {}).get("backends", {})
        return backends.get(name, {})

    def get_pipeline_config(self) -> dict[str, Any]:
        models_config = _load_models_config()
        return models_config.get("pipeline", {})

    def get_background_config(self) -> dict[str, Any]:
        models_config = _load_models_config()
        return models_config.get("background", {})

    def get_grounding_config(self, backend_name: str | None = None) -> dict[str, Any]:
        models_config = _load_models_config()
        section = models_config.get("grounding", {})
        name = backend_name or section.get("default", "clipseg")
        return section.get("backends", {}).get(name, {})

    def get_generation_config(self, backend_name: str | None = None) -> dict[str, Any]:
        models_config = _load_models_config()
        section = models_config.get("generation", {})
        name = backend_name or section.get("default", "animatediff")
        return section.get("backends", {}).get(name, {})

    def get_replacement_config(self, backend_name: str | None = None) -> dict[str, Any]:
        models_config = _load_models_config()
        section = models_config.get("replacement", {})
        name = backend_name or section.get("default", "sd_inpaint")
        return section.get("backends", {}).get(name, {})

    def get_replacement_default(self) -> str:
        models_config = _load_models_config()
        return models_config.get("replacement", {}).get("default", "sd_inpaint")

    def get_refinement_config(self, backend_name: str | None = None) -> dict[str, Any]:
        models_config = _load_models_config()
        section = models_config.get("refinement", {})
        name = backend_name or section.get("default", "sam2")
        return section.get("backends", {}).get(name, {})


settings = Settings()
