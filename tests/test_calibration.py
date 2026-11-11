from __future__ import annotations

import cv2
import numpy as np
import pytest

from catchvision.geometry.calibration import DEFAULT_MARKER_CENTERS_MM, ArucoCalibrator


def _build_synthetic_aruco_scene(
    px_per_mm: float = 4.0,
    marker_size_mm: float = 35.0,
    occlude_ids: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Создает синтетический A4-стенд с ArUco-маркерами и маской рыбы 160 мм."""
    w_px = int(round(297.0 * px_per_mm))
    h_px = int(round(210.0 * px_per_mm))
    img = np.full((h_px, w_px, 3), 245, dtype=np.uint8)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side_px = int(round(marker_size_mm * px_per_mm))
    half = side_px // 2

    for mid, (cx_mm, cy_mm) in DEFAULT_MARKER_CENTERS_MM.items():
        if mid in occlude_ids:
            continue
        marker = cv2.aruco.generateImageMarker(dictionary, mid, side_px)
        cx_px = int(round(cx_mm * px_per_mm))
        cy_px = int(round(cy_mm * px_per_mm))
        y1, y2 = cy_px - half, cy_px - half + side_px
        x1, x2 = cx_px - half, cx_px - half + side_px
        img[y1:y2, x1:x2] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    mask_u8 = np.zeros((h_px, w_px), dtype=np.uint8)
    fish_w = int(round(160.0 * px_per_mm))
    fish_h = int(round(40.0 * px_per_mm))
    cv2.ellipse(
        mask_u8,
        (w_px // 2, h_px // 2),
        (fish_w // 2, fish_h // 2),
        0,
        0,
        360,
        1,
        -1,
    )
    return img, mask_u8 > 0


def test_calibration_and_sizing_under_perspective() -> None:
    img, mask = _build_synthetic_aruco_scene()
    h, w = img.shape[:2]

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[90, 50], [w - 40, 25], [w - 20, h - 35], [35, h - 60]])
    h_mat = cv2.getPerspectiveTransform(src, dst)

    tilted_img = cv2.warpPerspective(img, h_mat, (w, h), borderValue=(210, 210, 210))
    tilted_mask = (
        cv2.warpPerspective(
            mask.astype(np.uint8) * 255,
            h_mat,
            (w, h),
            flags=cv2.INTER_NEAREST,
        )
        > 127
    )

    cal = ArucoCalibrator(marker_size_mm=35.0)
    calib = cal.calibrate(tilted_img)
    meas = cal.measure_mask(fish_id=0, mask=tilted_mask, calib=calib)

    assert calib.detected_marker_ids == [0, 1, 2, 3]
    assert calib.reprojection_error_mm < 0.2
    assert meas.straight_length_mm == pytest.approx(160.0, abs=2.5)
    assert meas.midline_length_mm == pytest.approx(160.0, abs=3.0)
    assert meas.body_depth_mm == pytest.approx(40.0, abs=2.0)


def test_calibration_fallback_with_two_occluded_markers() -> None:
    img, mask = _build_synthetic_aruco_scene(occlude_ids=(0, 3))
    cal = ArucoCalibrator(marker_size_mm=35.0, min_markers=2)
    calib = cal.calibrate(img)
    meas = cal.measure_mask(fish_id=1, mask=mask, calib=calib)

    assert calib.detected_marker_ids == [1, 2]
    assert meas.midline_length_mm == pytest.approx(160.0, abs=3.0)


def test_calibration_fails_when_too_few_markers() -> None:
    img, _ = _build_synthetic_aruco_scene(occlude_ids=(0, 1, 2))
    cal = ArucoCalibrator(min_markers=2)
    with pytest.raises(ValueError, match="Detected only 1 calibration markers"):
        cal.calibrate(img)
