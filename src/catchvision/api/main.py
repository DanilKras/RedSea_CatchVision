from __future__ import annotations

import base64
import json
import os
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cv2
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from catchvision.api.schemas import (
    BenchmarkRequest,
    BenchmarkSummaryResponse,
    CommitResponse,
    FishInspectionRow,
    HealthResponse,
    InspectResponse,
    LedgerResponse,
)
from catchvision.benchmark.stress import DEFAULT_SCENARIOS, FieldStressBenchmark
from catchvision.config import SegmentedFish, Settings
from catchvision.geometry.calibration import ArucoCalibrator, CalibrationResult, FishMeasurement
from catchvision.mlops.tracking import ExperimentTracker
from catchvision.storage.catch_db import CatchDatabase
from catchvision.vision.classifier import (
    DEFAULT_RED_SEA_SPECIES,
    ClassificationResult,
    FishClassifier,
    SpeciesSpec,
)
from catchvision.vision.segmentor import FishSegmentor

SETTINGS = Settings()
DB_PATH = Path(os.getenv("CATCHVISION_DB_PATH", str(SETTINGS.db_path)))
PROCESSED_DIR = Path(os.getenv("CATCHVISION_PROCESSED_DIR", str(SETTINGS.processed_dir)))
API_WORK_DIR = PROCESSED_DIR / "api_runs"
SAMPLE_IMAGE_PATH = PROCESSED_DIR / "test_run/synthetic_tilted.jpg"


@dataclass
class PendingInspection:
    image_path: Path
    calib: CalibrationResult | None
    segmented: list[SegmentedFish]
    measurements: list[FishMeasurement | None]
    classifications: list[ClassificationResult]
    site_name: str
    min_conf_auto: float
    min_cos_auto: float


def _encode_bgr_b64(img_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("Failed to encode image to JPEG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _resolve_input(file: UploadFile | None, use_sample: bool, run_dir: Path) -> tuple[Path, str]:
    if file is None and not use_sample:
        raise HTTPException(status_code=400, detail="Provide either `file` or `use_sample=true`.")

    if use_sample:
        if not SAMPLE_IMAGE_PATH.exists():
            raise HTTPException(status_code=404, detail=f"Sample not found: {SAMPLE_IMAGE_PATH}")
        return SAMPLE_IMAGE_PATH, SAMPLE_IMAGE_PATH.name

    assert file is not None
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    dst = run_dir / f"input{suffix}"
    dst.write_bytes(file.file.read())
    return dst, (file.filename or dst.name)


def _measure_fish(calibrator: ArucoCalibrator, img_bgr, items, masks):
    try:
        calib = calibrator.calibrate(img_bgr)
    except Exception:
        return None, [None] * len(items)

    measurements = []
    for item, mask in zip(items, masks, strict=True):
        try:
            measurements.append(calibrator.measure_mask(item.fish_id, mask, calib))
        except Exception:
            measurements.append(None)
    return calib, measurements


@asynccontextmanager
async def lifespan(app: FastAPI):
    API_WORK_DIR.mkdir(parents=True, exist_ok=True)
    app.state.gpu_lock = threading.Lock()
    app.state.db_lock = threading.Lock()
    app.state.pending = OrderedDict()
    app.state.include_jewel_demo = False

    if not SETTINGS.skip_model_load and os.getenv("CATCHVISION_SKIP_MODEL_LOAD", "0") != "1":
        logger.info("Initializing CV models and DuckDB...")
        app.state.segmentor = FishSegmentor(settings=SETTINGS)
        app.state.calibrator = ArucoCalibrator(
            board_size_mm=(SETTINGS.board_width_mm, SETTINGS.board_height_mm),
            marker_size_mm=SETTINGS.marker_size_mm,
            px_per_mm=SETTINGS.pixels_per_mm,
            min_markers=SETTINGS.aruco_min_markers,
        )
        app.state.classifier = FishClassifier(settings=SETTINGS)
        app.state.db = CatchDatabase(DB_PATH, settings=SETTINGS)
        app.state.tracker = ExperimentTracker(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", SETTINGS.mlflow_tracking_uri)
        )

    yield

    db = getattr(app.state, "db", None)
    if db is not None:
        db.close()


app = FastAPI(
    title="RedSea-CatchVision Inference & Ledger API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    seg = getattr(app.state, "segmentor", None)
    return HealthResponse(
        status="ok",
        device=getattr(seg, "device", "unknown"),
        models_loaded=seg is not None,
    )


@app.post("/api/v1/inspect", response_model=InspectResponse)
def inspect_image(
    file: Annotated[UploadFile | None, File()] = None,
    use_sample: bool = Form(False),
    site_name: str = Form("Thuwal_KAUST_Dock"),
    box_threshold: float = Form(SETTINGS.box_threshold),
    min_conf_auto: float = Form(SETTINGS.min_conf_for_auto),
    min_cos_auto: float = Form(SETTINGS.min_cosine_sim),
    include_jewel_demo: bool = Form(False),
    save_db: bool = Form(False),
) -> InspectResponse:
    run_dir = API_WORK_DIR / uuid.uuid4().hex[:10]
    run_dir.mkdir(parents=True, exist_ok=True)

    target_path, image_name = _resolve_input(file, use_sample, run_dir)
    img_bgr = cv2.imread(str(target_path))
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    segmentor: FishSegmentor = app.state.segmentor
    calibrator: ArucoCalibrator = app.state.calibrator
    classifier: FishClassifier = app.state.classifier

    with app.state.gpu_lock:
        segmentor.settings.box_threshold = box_threshold
        classifier.min_cosine_sim = min_cos_auto

        if include_jewel_demo != getattr(app.state, "include_jewel_demo", False):
            catalog = list(DEFAULT_RED_SEA_SPECIES)
            if include_jewel_demo:
                catalog.append(
                    SpeciesSpec(
                        scientific_name="Hemichromis guttatus",
                        common_name="Jewel cichlid",
                        family="Cichlidae",
                    )
                )
            classifier.set_species_catalog(catalog)
            app.state.include_jewel_demo = include_jewel_demo

        items, masks = segmentor.segment_image(target_path, run_dir, save_overlay=True)
        calib, measurements = _measure_fish(calibrator, img_bgr, items, masks)
        cls_results = classifier.classify_batch(
            [it.masked_crop_path for it in items],
            fish_ids=[it.fish_id for it in items],
            lengths_cm=[(m.midline_length_mm / 10.0) if m else None for m in measurements],
        )

    overlay_path = run_dir / f"{target_path.stem}_overlay.jpg"
    overlay_img = cv2.imread(str(overlay_path)) if overlay_path.exists() else img_bgr
    overlay_b64 = _encode_bgr_b64(overlay_img)

    rectified_b64 = None
    if calib is not None:
        valid_m = [m for m in measurements if m is not None]
        rect_img = calibrator.warp_image(img_bgr, calib)
        rect_overlay = calibrator.draw_measurements_overlay(rect_img, valid_m, calib)
        rectified_b64 = _encode_bgr_b64(rect_overlay)

    rows = []
    for it, meas, cls_res in zip(items, measurements, cls_results, strict=True):
        needs_rev = (
            cls_res.confidence < min_conf_auto
            or cls_res.margin_top1_top2 < SETTINGS.min_margin_for_auto
            or cls_res.raw_cosine_sim < min_cos_auto
        )
        rows.append(
            FishInspectionRow(
                fish_id=it.fish_id,
                species=cls_res.top_species,
                common_name=cls_res.common_name,
                local_ar=cls_res.local_name_ar,
                cls_conf=cls_res.confidence,
                raw_cos=cls_res.raw_cosine_sim,
                margin=cls_res.margin_top1_top2,
                needs_review=needs_rev,
                tl_cm=round(meas.midline_length_mm / 10.0, 2) if meas else None,
                depth_cm=round(meas.body_depth_mm / 10.0, 2) if meas else None,
                weight_g=cls_res.estimated_weight_g,
                undersized=cls_res.below_min_legal_size,
            )
        )

    insp_id = uuid.uuid4().hex[:12]
    app.state.pending[insp_id] = PendingInspection(
        image_path=target_path,
        calib=calib,
        segmented=list(items),
        measurements=measurements,
        classifications=list(cls_results),
        site_name=site_name,
        min_conf_auto=min_conf_auto,
        min_cos_auto=min_cos_auto,
    )
    while len(app.state.pending) > 20:
        app.state.pending.popitem(last=False)

    if save_db and items:
        with app.state.db_lock:
            db: CatchDatabase = app.state.db
            db.min_conf_for_auto = min_conf_auto
            db.min_cosine_for_auto = min_cos_auto
            db.save_frame_results(
                image_path=target_path,
                calib=calib,
                segmented=items,
                measurements=measurements,
                classifications=cls_results,
                site_name=site_name,
            )

    return InspectResponse(
        inspection_id=insp_id,
        image_name=image_name,
        calibrated=calib is not None,
        detected_markers=calib.detected_marker_ids if calib else [],
        reproj_error_mm=round(calib.reprojection_error_mm, 3) if calib else None,
        rows=rows,
        overlay_b64=overlay_b64,
        rectified_b64=rectified_b64,
    )


@app.post("/api/v1/inspect/{inspection_id}/commit", response_model=CommitResponse)
def commit_inspection(inspection_id: str) -> CommitResponse:
    pending = app.state.pending.get(inspection_id)
    if pending is None:
        raise HTTPException(status_code=404, detail="Inspection ID expired or not found.")

    with app.state.db_lock:
        db: CatchDatabase = app.state.db
        db.min_conf_for_auto = pending.min_conf_auto
        db.min_cosine_for_auto = pending.min_cos_auto
        frame_id = db.save_frame_results(
            image_path=pending.image_path,
            calib=pending.calib,
            segmented=pending.segmented,
            measurements=pending.measurements,
            classifications=pending.classifications,
            site_name=pending.site_name,
        )
    return CommitResponse(frame_id=frame_id, fish_count=len(pending.segmented))


@app.get("/api/v1/ledger", response_model=LedgerResponse)
def get_ledger(limit: int = Query(100, ge=1, le=500)) -> LedgerResponse:
    with app.state.db_lock:
        db: CatchDatabase = app.state.db
        sp_df = db.get_species_summary()
        recent_df = db.get_recent_catches(limit=limit)
    return LedgerResponse(
        species_summary=_df_to_records(sp_df),
        recent_catches=_df_to_records(recent_df),
    )


@app.post("/api/v1/benchmark", response_model=BenchmarkSummaryResponse)
def run_benchmark(req: BenchmarkRequest) -> BenchmarkSummaryResponse:
    bench_path = Path(req.image_path) if req.image_path else SAMPLE_IMAGE_PATH
    if not bench_path.exists():
        raise HTTPException(status_code=404, detail=f"Reference image not found: {bench_path}")

    scenarios = tuple(req.scenarios) if req.scenarios else DEFAULT_SCENARIOS
    out_dir = PROCESSED_DIR / "stress_api"

    with app.state.gpu_lock:
        suite = FieldStressBenchmark(
            app.state.segmentor,
            app.state.calibrator,
            app.state.classifier,
        )
        df = suite.run_suite(
            image_path=bench_path,
            output_dir=out_dir,
            ground_truth_length_mm=req.ground_truth_length_mm,
            scenarios=scenarios,
            severities=req.severities or [0.4],
            db=None,
        )

    run_id = uuid.uuid4().hex[:10]
    with app.state.db_lock:
        db: CatchDatabase = app.state.db
        db.save_benchmark_rows(run_id, bench_path.name, df.to_dict(orient="records"))
        summary_df = db.get_benchmark_summary()

    tracker = getattr(app.state, "tracker", None)
    if tracker is not None:
        valid_err = df["length_pct_error"].dropna()
        tracker.log_run(
            run_name="api_stress_benchmark",
            params={"image": bench_path.name, "severities": req.severities},
            metrics={
                "calibration_rate": float(df["calibrated"].mean()),
                "mean_length_pct_error": float(valid_err.mean()) if not valid_err.empty else 0.0,
            },
            artifacts=[out_dir / "stress_benchmark_results.csv"],
        )

    samples = sorted(p.name for p in out_dir.glob("*_sev*.jpg"))[:6]
    return BenchmarkSummaryResponse(
        summary=_df_to_records(summary_df),
        sample_images=[f"stress_api/{name}" for name in samples],
    )


@app.get("/api/v1/benchmark", response_model=BenchmarkSummaryResponse)
def get_benchmark_summary() -> BenchmarkSummaryResponse:
    with app.state.db_lock:
        db: CatchDatabase = app.state.db
        summary_df = db.get_benchmark_summary()

    samples = []
    for folder in ("stress_api", "stress_test"):
        d = PROCESSED_DIR / folder
        if d.exists():
            samples = [f"{folder}/{p.name}" for p in sorted(d.glob("*_sev*.jpg"))[:6]]
            if samples:
                break

    return BenchmarkSummaryResponse(
        summary=_df_to_records(summary_df),
        sample_images=samples,
    )


@app.get("/api/v1/artifacts/{rel_path:path}")
def get_artifact(rel_path: str) -> FileResponse:
    root = PROCESSED_DIR.resolve()
    target = (root / rel_path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(target)
