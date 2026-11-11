from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CATCHVISION_",
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = Path(__file__).resolve().parents[2]
    device: str | None = None
    skip_model_load: bool = False

    detector_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_weights: str = "sam2.1_b.pt"
    bioclip_model_id: str = "hf-hub:imageomics/bioclip"

    box_threshold: float = 0.28
    text_threshold: float = 0.25
    text_prompt: str = "fish."
    bioclip_text_weight: float = 0.45
    min_cosine_sim: float = 0.22
    min_conf_for_auto: float = 0.70
    min_margin_for_auto: float = 0.25

    aruco_dict_name: str = "DICT_4X4_50"
    aruco_min_markers: int = 2
    board_width_mm: float = 297.0
    board_height_mm: float = 210.0
    marker_size_mm: float = 35.0
    pixels_per_mm: float = 4.0
    camera_height_mm: float | None = None

    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    segmentor_backend: Literal["gdino_sam2", "sam3"] = "gdino_sam2"
    sam3_model_id: str = "facebook/sam3"

    @computed_field
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @computed_field
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field
    @property
    def db_path(self) -> Path:
        return self.data_dir / "catchvision.duckdb"

    @computed_field
    @property
    def eval_redsea_dir(self) -> Path:
        return self.processed_dir / "eval_redsea"

    @computed_field
    @property
    def eval_zero_shot_csv(self) -> Path:
        return self.eval_redsea_dir / "eval_zero_shot.csv"

    @computed_field
    @property
    def stress_test_dir(self) -> Path:
        return self.processed_dir / "stress_test"

    @computed_field
    @property
    def stress_benchmark_csv(self) -> Path:
        return self.stress_test_dir / "stress_benchmark_results.csv"


class SegmentedFish(BaseModel):
    fish_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask_area_px: int
    touches_border: bool = False
    crop_path: Path | None = None
    masked_crop_path: Path | None = None
