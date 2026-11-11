from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool


class FishInspectionRow(BaseModel):
    fish_id: int
    species: str
    common_name: str
    local_ar: str
    cls_conf: float
    raw_cos: float
    margin: float
    needs_review: bool
    tl_cm: float | None = None
    depth_cm: float | None = None
    weight_g: float | None = None
    undersized: bool | None = None


class InspectResponse(BaseModel):
    inspection_id: str
    image_name: str
    calibrated: bool
    detected_markers: list[int] = Field(default_factory=list)
    reproj_error_mm: float | None = None
    rows: list[FishInspectionRow] = Field(default_factory=list)
    overlay_b64: str | None = None
    rectified_b64: str | None = None


class CommitResponse(BaseModel):
    frame_id: str
    fish_count: int


class LedgerResponse(BaseModel):
    species_summary: list[dict[str, Any]]
    recent_catches: list[dict[str, Any]]


class BenchmarkRequest(BaseModel):
    image_path: str = "data/processed/test_run/synthetic_tilted.jpg"
    ground_truth_length_mm: float = 150.0
    scenarios: list[str] | None = None
    severities: list[float] = Field(default_factory=lambda: [0.4, 0.8])


class BenchmarkSummaryResponse(BaseModel):
    summary: list[dict[str, Any]]
    sample_images: list[str]
