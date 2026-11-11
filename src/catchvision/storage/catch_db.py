from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

from catchvision.config import SegmentedFish
from catchvision.geometry.calibration import CalibrationResult, FishMeasurement
from catchvision.vision.classifier import ClassificationResult
from catchvision.config import SegmentedFish, Settings


class CatchDatabase:
    """Analytical DuckDB store for catch records, metric measurements, and stress benchmarks."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        min_conf_for_auto: float | None = None,
        min_margin_for_auto: float | None = None,
        min_cosine_for_auto: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or Settings()
        self.db_path = Path(db_path) if db_path is not None else cfg.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_conf_for_auto = (cfg.min_conf_for_auto if min_conf_for_auto is None else min_conf_for_auto)
        self.min_margin_for_auto = (
        cfg.min_margin_for_auto if min_margin_for_auto is None else min_margin_for_auto)
        self.min_cosine_for_auto = (cfg.min_cosine_sim if min_cosine_for_auto is None else min_cosine_for_auto)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS survey_frames
            (
                frame_id
                VARCHAR
                PRIMARY
                KEY,
                created_at
                TIMESTAMP,
                site_name
                VARCHAR,
                image_path
                VARCHAR,
                detected_markers
                VARCHAR,
                reproj_error_mm
                DOUBLE,
                fish_count
                INTEGER,
                total_biomass_g
                DOUBLE
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catch_records
            (
                record_id
                VARCHAR
                PRIMARY
                KEY,
                frame_id
                VARCHAR,
                fish_id
                INTEGER,
                det_confidence
                DOUBLE,
                top_species
                VARCHAR,
                common_name
                VARCHAR,
                local_name_ar
                VARCHAR,
                family
                VARCHAR,
                cls_confidence
                DOUBLE,
                margin_top1_top2
                DOUBLE,
                needs_review
                BOOLEAN,
                straight_length_mm
                DOUBLE,
                midline_length_mm
                DOUBLE,
                body_depth_mm
                DOUBLE,
                area_cm2
                DOUBLE,
                curvature_ratio
                DOUBLE,
                estimated_weight_g
                DOUBLE,
                below_min_legal_size
                BOOLEAN,
                crop_path
                VARCHAR,
                masked_crop_path
                VARCHAR,
                raw_cosine_sim
                DOUBLE
            );
            """
        )
        self.conn.execute(
            "ALTER TABLE catch_records ADD COLUMN IF NOT EXISTS raw_cosine_sim DOUBLE;"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs
            (
                run_id
                VARCHAR,
                created_at
                TIMESTAMP,
                image_name
                VARCHAR,
                scenario
                VARCHAR,
                severity
                DOUBLE,
                calibrated
                BOOLEAN,
                detected_markers
                VARCHAR,
                reproj_error_mm
                DOUBLE,
                detected_fish_count
                INTEGER,
                measured_length_mm
                DOUBLE,
                length_abs_error_mm
                DOUBLE,
                length_pct_error
                DOUBLE,
                top_species
                VARCHAR,
                cls_confidence
                DOUBLE,
                species_matches_ref
                BOOLEAN,
                runtime_ms
                DOUBLE
            );
            """
        )

    def _needs_review(self, cls_res: ClassificationResult) -> bool:
        return (
            cls_res.confidence < self.min_conf_for_auto
            or cls_res.margin_top1_top2 < self.min_margin_for_auto
            or cls_res.raw_cosine_sim < self.min_cosine_for_auto
        )

    def save_frame_results(
        self,
        image_path: Path,
        calib: CalibrationResult | None,
        segmented: Sequence[SegmentedFish],
        measurements: Sequence[FishMeasurement | None],
        classifications: Sequence[ClassificationResult],
        site_name: str = "Thuwal_Landing",
    ) -> str:
        """Persist a processed frame and all detected fish specimens."""
        if not (len(segmented) == len(measurements) == len(classifications)):
            raise ValueError("segmented, measurements, and classifications must align.")

        frame_id = uuid.uuid4().hex[:12]
        now = datetime.now(UTC).replace(tzinfo=None)

        total_biomass = sum(
            c.estimated_weight_g for c in classifications if c.estimated_weight_g is not None
        )
        markers_json = json.dumps(calib.detected_marker_ids) if calib else "[]"
        reproj_err = calib.reprojection_error_mm if calib else None

        self.conn.execute(
            """
            INSERT INTO survey_frames
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                frame_id,
                now,
                site_name,
                str(image_path),
                markers_json,
                reproj_err,
                len(segmented),
                round(total_biomass, 2),
            ],
        )

        rows: list[list[object]] = []
        for seg, meas, cls_res in zip(segmented, measurements, classifications, strict=True):
            rows.append(
                [
                    uuid.uuid4().hex[:12],
                    frame_id,
                    seg.fish_id,
                    seg.confidence,
                    cls_res.top_species,
                    cls_res.common_name,
                    cls_res.local_name_ar,
                    cls_res.family,
                    cls_res.confidence,
                    cls_res.margin_top1_top2,
                    self._needs_review(cls_res),
                    meas.straight_length_mm if meas else None,
                    meas.midline_length_mm if meas else None,
                    meas.body_depth_mm if meas else None,
                    meas.area_cm2 if meas else None,
                    meas.curvature_ratio if meas else None,
                    cls_res.estimated_weight_g,
                    cls_res.below_min_legal_size,
                    str(seg.crop_path),
                    str(seg.masked_crop_path),
                    cls_res.raw_cosine_sim,
                ]
            )

        if rows:
            self.conn.executemany(
                """
                INSERT INTO catch_records (record_id, frame_id, fish_id, det_confidence, top_species,
                                           common_name, local_name_ar, family, cls_confidence,
                                           margin_top1_top2, needs_review, straight_length_mm,
                                           midline_length_mm, body_depth_mm, area_cm2, curvature_ratio,
                                           estimated_weight_g, below_min_legal_size, crop_path,
                                           masked_crop_path, raw_cosine_sim)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        logger.info(
            f"Saved frame={frame_id} ({len(rows)} fish, biomass={total_biomass:.1f}g) "
            f"-> {self.db_path}"
        )
        return frame_id

    def save_benchmark_rows(
        self,
        run_id: str,
        image_name: str,
        rows: Sequence[dict[str, object]],
    ) -> None:
        """Insert rows from a field stress benchmark run."""
        now = datetime.now(UTC).replace(tzinfo=None)
        payload = [
            [
                run_id,
                now,
                image_name,
                r["scenario"],
                r["severity"],
                r["calibrated"],
                json.dumps(r["detected_markers"]),
                r["reproj_error_mm"],
                r["detected_fish_count"],
                r["measured_length_mm"],
                r["length_abs_error_mm"],
                r["length_pct_error"],
                r["top_species"],
                r["cls_confidence"],
                r["species_matches_ref"],
                r["runtime_ms"],
            ]
            for r in rows
        ]
        if payload:
            self.conn.executemany(
                """
                INSERT INTO benchmark_runs
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            logger.info(f"Saved {len(payload)} benchmark rows (run_id={run_id}) -> {self.db_path}")

    def get_species_summary(self) -> pd.DataFrame:
        """Aggregate catch counts, mean TL (cm), total biomass (kg), and undersized share by species."""
        return self.conn.execute(
            """
            SELECT top_species,
                   ANY_VALUE(common_name)                                    AS common_name,
                   ANY_VALUE(local_name_ar)                                  AS local_name_ar,
                   COUNT(*)                                                  AS fish_count,
                   ROUND(AVG(midline_length_mm) / 10.0, 2)                   AS mean_tl_cm,
                   ROUND(SUM(COALESCE(estimated_weight_g, 0.0)) / 1000.0, 3) AS total_biomass_kg,
                   ROUND(100.0 * AVG(CASE WHEN below_min_legal_size THEN 1.0 ELSE 0.0 END), 1)
                                                                             AS pct_undersized,
                   ROUND(100.0 * AVG(CASE WHEN needs_review THEN 1.0 ELSE 0.0 END), 1)
                                                                             AS pct_needs_review
            FROM catch_records
            GROUP BY top_species
            ORDER BY fish_count DESC, total_biomass_kg DESC
            """
        ).df()

    def get_recent_catches(self, limit: int = 100) -> pd.DataFrame:
        """Return recent catch records joined with landing site metadata."""
        return self.conn.execute(
            """
            SELECT f.created_at,
                   f.site_name,
                   c.frame_id,
                   c.fish_id,
                   c.top_species,
                   c.local_name_ar,
                   ROUND(c.midline_length_mm / 10.0, 2) AS tl_cm,
                   c.estimated_weight_g,
                   c.cls_confidence,
                   ROUND(c.raw_cosine_sim, 4)           AS raw_cosine_sim,
                   c.below_min_legal_size,
                   c.needs_review
            FROM catch_records c
                     JOIN survey_frames f ON c.frame_id = f.frame_id
            ORDER BY f.created_at DESC, c.fish_id ASC LIMIT ?
            """,
            [limit],
        ).df()

    def get_benchmark_summary(self) -> pd.DataFrame:
        """Aggregate pipeline robustness metrics across perturbation scenarios."""
        return self.conn.execute(
            """
            SELECT scenario,
                   severity,
                   COUNT(*)                           AS n_trials,
                   ROUND(100.0 * AVG(CASE WHEN calibrated THEN 1.0 ELSE 0.0 END), 1)
                                                      AS calibration_rate_pct,
                   ROUND(AVG(length_abs_error_mm), 2) AS mae_length_mm,
                   ROUND(AVG(length_pct_error), 2)    AS mape_length_pct,
                   ROUND(100.0 * AVG(CASE WHEN species_matches_ref THEN 1.0 ELSE 0.0 END), 1)
                                                      AS species_consistency_pct,
                   ROUND(AVG(cls_confidence), 3)      AS mean_cls_conf,
                   ROUND(AVG(runtime_ms), 1)          AS mean_runtime_ms
            FROM benchmark_runs
            GROUP BY scenario, severity
            ORDER BY scenario, severity
            """
        ).df()

    def close(self) -> None:
        self.conn.close()
