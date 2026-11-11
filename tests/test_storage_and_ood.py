from __future__ import annotations

from pathlib import Path

import pytest

from catchvision.config import SegmentedFish
from catchvision.storage.catch_db import CatchDatabase
from catchvision.vision.classifier import (
    DEFAULT_RED_SEA_SPECIES,
    ClassificationResult,
    FishClassifier,
)


def _make_cls_result(
    fish_id: int = 0,
    species: str = "Plectropomus pessuliferus",
    confidence: float = 0.90,
    margin: float = 0.70,
    raw_cos: float = 0.28,
    weight_g: float | None = 450.0,
    undersized: bool | None = False,
) -> ClassificationResult:
    return ClassificationResult(
        fish_id=fish_id,
        top_species=species,
        common_name="Roving coral grouper",
        local_name_ar="Najil",
        family="Serranidae",
        confidence=confidence,
        margin_top1_top2=margin,
        raw_cosine_sim=raw_cos,
        low_cosine_similarity=raw_cos < 0.22,
        estimated_weight_g=weight_g,
        below_min_legal_size=undersized,
    )


def test_lwr_biomass_and_legal_size() -> None:
    najil = DEFAULT_RED_SEA_SPECIES[0]
    weight_large, under_large = FishClassifier._estimate_biomass(najil, length_cm=40.0)
    weight_small, under_small = FishClassifier._estimate_biomass(najil, length_cm=25.0)

    expected_40 = round(float(najil.lwr_a * (40.0**najil.lwr_b)), 2)
    assert weight_large == pytest.approx(expected_40, rel=1e-4)
    assert under_large is False
    assert weight_small is not None and weight_small > 0
    assert under_small is True


def test_ood_gating_flags_high_softmax_low_cosine(tmp_path: Path) -> None:
    db = CatchDatabase(
        db_path=tmp_path / "test.duckdb",
        min_conf_for_auto=0.70,
        min_margin_for_auto=0.25,
        min_cosine_for_auto=0.22,
    )
    ood_pred = _make_cls_result(confidence=0.9901, margin=0.9814, raw_cos=0.2010)
    in_dist_pred = _make_cls_result(confidence=0.8800, margin=0.6500, raw_cos=0.2760)

    assert db._needs_review(ood_pred) is True
    assert db._needs_review(in_dist_pred) is False
    db.close()


def test_duckdb_save_and_aggregations(tmp_path: Path) -> None:
    db = CatchDatabase(db_path=tmp_path / "ledger.duckdb")
    seg = SegmentedFish(
        fish_id=0,
        confidence=0.58,
        bbox_xyxy=(10.0, 10.0, 210.0, 90.0),
        mask_area_px=12000,
        crop_path=tmp_path / "crop.jpg",
        masked_crop_path=tmp_path / "masked.jpg",
    )
    cls_res = _make_cls_result(confidence=0.85, margin=0.60, raw_cos=0.26, weight_g=510.0)

    frame_id = db.save_frame_results(
        image_path=tmp_path / "dock.jpg",
        calib=None,
        segmented=[seg],
        measurements=[None],
        classifications=[cls_res],
        site_name="Thuwal_Dock",
    )
    assert len(frame_id) == 12

    summary = db.get_species_summary()
    recent = db.get_recent_catches()
    assert int(summary.loc[0, "fish_count"]) == 1
    assert float(summary.loc[0, "total_biomass_kg"]) == pytest.approx(0.51, abs=1e-3)
    assert float(recent.loc[0, "raw_cosine_sim"]) == pytest.approx(0.26, abs=1e-4)

    db.save_benchmark_rows(
        run_id="run01",
        image_name="sample.jpg",
        rows=[
            {
                "scenario": "sun_glare",
                "severity": 0.8,
                "calibrated": True,
                "detected_markers": [0, 1, 2],
                "reproj_error_mm": 0.086,
                "detected_fish_count": 1,
                "measured_length_mm": 149.93,
                "length_abs_error_mm": 0.07,
                "length_pct_error": 0.05,
                "top_species": "Cephalopholis miniata",
                "cls_confidence": 0.51,
                "species_matches_ref": True,
                "runtime_ms": 1850.0,
            }
        ],
    )
    bench = db.get_benchmark_summary()
    assert len(bench) == 1
    assert float(bench.loc[0, "calibration_rate_pct"]) == 100.0
    db.close()
