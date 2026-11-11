from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from catchvision.benchmark.stress import DEFAULT_SCENARIOS, FieldStressBenchmark


def test_all_stress_perturbations_preserve_shape_and_dtype() -> None:
    dummy = SimpleNamespace()
    bench = FieldStressBenchmark(
        segmentor=dummy,
        calibrator=dummy,
        classifier=dummy,
    )
    base = np.full((240, 320, 3), 120, dtype=np.uint8)
    cv2.rectangle(base, (90, 80), (230, 150), (200, 180, 160), -1)
    fake_markers = {
        0: np.array([[10, 10], [40, 10], [40, 40], [10, 40]], dtype=np.int32),
        1: np.array([[270, 10], [300, 10], [300, 40], [270, 40]], dtype=np.int32),
    }

    for scenario in DEFAULT_SCENARIOS:
        out = bench.apply_perturbation(
            base,
            scenario=scenario,
            severity=0.6,
            seed=42,
            marker_boxes=fake_markers,
        )
        assert out.shape == base.shape
        assert out.dtype == np.uint8
        if scenario == "clean":
            assert np.array_equal(out, base)
        else:
            assert not np.array_equal(out, base)


def test_unknown_stress_scenario_raises() -> None:
    dummy = SimpleNamespace()
    bench = FieldStressBenchmark(
        segmentor=dummy,
        calibrator=dummy,
        classifier=dummy,
    )
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown stress scenario"):
        bench.apply_perturbation(base, scenario="typhoon")
