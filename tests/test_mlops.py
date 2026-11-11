from __future__ import annotations

import json
from pathlib import Path

from catchvision.mlops.tracking import ExperimentTracker


def test_experiment_tracker_writes_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("scenario,err\nclean,0.4\n", encoding="utf-8")

    tracker = ExperimentTracker(
        experiment_name="TestExp",
        manifest_dir=tmp_path / "runs",
        tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    manifest_file = tracker.log_run(
        run_name="unit_eval",
        params={"min_cosine_sim": 0.22},
        metrics={"selective_top1": 0.65, "ood_fp_rate": 0.0},
        artifacts=[artifact],
        tags={"stage": "ci"},
    )

    assert manifest_file.exists()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["params"]["min_cosine_sim"] == 0.22
    assert data["metrics"]["selective_top1"] == 0.65
