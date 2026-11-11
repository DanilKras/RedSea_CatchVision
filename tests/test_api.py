from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from catchvision.api import main as api_main
from catchvision.config import SegmentedFish
from catchvision.storage.catch_db import CatchDatabase
from catchvision.vision.classifier import ClassificationResult


def test_api_inspect_commit_and_ledger(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CATCHVISION_SKIP_MODEL_LOAD", "1")
    crop_p = tmp_path / "masked.jpg"
    img = np.full((64, 128, 3), 140, dtype=np.uint8)
    cv2.imwrite(str(crop_p), img)

    class FakeSegmentor:
        device = "cpu"
        settings = SimpleNamespace(box_threshold=0.28)

        def segment_image(self, image_path, output_dir, save_overlay=True):
            output_dir.mkdir(parents=True, exist_ok=True)
            if save_overlay:
                cv2.imwrite(str(output_dir / f"{image_path.stem}_overlay.jpg"), img)
            item = SegmentedFish(
                fish_id=0,
                confidence=0.82,
                bbox_xyxy=(1.0, 1.0, 100.0, 50.0),
                mask_area_px=3000,
                crop_path=crop_p,
                masked_crop_path=crop_p,
            )
            return [item], [np.ones((64, 128), dtype=bool)]

    class FakeCalibrator:
        def calibrate(self, image_bgr):
            raise ValueError("No markers in mock")

    class FakeClassifier:
        min_cosine_sim = 0.22

        def set_species_catalog(self, catalog):
            return None

        def classify_batch(self, images, fish_ids=None, lengths_cm=None, top_k=5):
            return [
                ClassificationResult(
                    fish_id=0,
                    top_species="Lutjanus bohar",
                    common_name="Two-spot red snapper",
                    local_name_ar="Bohar",
                    family="Lutjanidae",
                    confidence=0.91,
                    margin_top1_top2=0.75,
                    raw_cosine_sim=0.30,
                )
            ]

    db = CatchDatabase(tmp_path / "api_test.duckdb")
    monkeypatch.setattr(api_main, "DB_PATH", tmp_path / "lifespan.duckdb")
    monkeypatch.setattr(api_main, "API_WORK_DIR", tmp_path / "api_runs")

    with TestClient(api_main.app) as client:
        api_main.app.state.segmentor = FakeSegmentor()
        api_main.app.state.calibrator = FakeCalibrator()
        api_main.app.state.classifier = FakeClassifier()
        api_main.app.state.db = db

        health_r = client.get("/health")
        assert health_r.status_code == 200
        assert health_r.json()["models_loaded"] is True

        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        insp_r = client.post(
            "/api/v1/inspect",
            files={"file": ("snap.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
            data={"site_name": "Test_Dock"},
        )
        assert insp_r.status_code == 200
        payload = insp_r.json()
        assert len(payload["rows"]) == 1
        assert payload["rows"][0]["species"] == "Lutjanus bohar"
        assert payload["overlay_b64"] is not None

        commit_r = client.post(f"/api/v1/inspect/{payload['inspection_id']}/commit")
        assert commit_r.status_code == 200
        assert commit_r.json()["fish_count"] == 1

        ledger_r = client.get("/api/v1/ledger")
        assert ledger_r.status_code == 200
        assert ledger_r.json()["species_summary"][0]["top_species"] == "Lutjanus bohar"

    db.close()
