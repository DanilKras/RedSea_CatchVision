from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


def _git_revision() -> str:
    """Return the short Git commit hash for run reproducibility."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


class ExperimentTracker:
    """Unified MLOps tracker writing local JSON manifests and optional MLflow runs."""

    def __init__(
        self,
        experiment_name: str = "RedSea-CatchVision",
        manifest_dir: Path | str = Path("data/artifacts/runs"),
        tracking_uri: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.manifest_dir = Path(manifest_dir)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")

    def log_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[Path] | None = None,
        tags: dict[str, str] | None = None,
    ) -> Path:
        """Persist run parameters, metrics, and artifact paths locally and to MLflow."""
        run_id = uuid.uuid4().hex[:10]
        git_sha = _git_revision()
        timestamp = datetime.now(UTC).isoformat()
        artifact_paths = [Path(p) for p in (artifacts or [])]

        manifest = {
            "run_id": run_id,
            "run_name": run_name,
            "experiment": self.experiment_name,
            "timestamp_utc": timestamp,
            "git_commit": git_sha,
            "params": params,
            "metrics": {k: round(float(v), 5) for k, v in metrics.items()},
            "artifacts": [str(p) for p in artifact_paths],
            "tags": tags or {},
        }
        manifest_path = self.manifest_dir / f"{run_name}_{run_id}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if mlflow is not None:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                with mlflow.start_run(run_name=f"{run_name}-{run_id}"):
                    mlflow.set_tags({"git_commit": git_sha, **(tags or {})})
                    mlflow.log_params({k: str(v) for k, v in params.items()})
                    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
                    mlflow.log_artifact(str(manifest_path))
                    for art in artifact_paths:
                        if art.exists():
                            mlflow.log_artifact(str(art))
            except Exception as exc:
                logger.warning(f"MLflow logging skipped ({exc}); manifest saved locally.")

        logger.info(f"Logged MLOps run '{run_name}' ({run_id}) -> {manifest_path}")
        return manifest_path
