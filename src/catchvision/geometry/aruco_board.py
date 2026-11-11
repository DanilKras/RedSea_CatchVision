from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def generate_a4_calibration_mat(
    output_path: Path,
    dpi: int = 300,
    marker_size_mm: float = 35.0,
    margin_mm: float = 15.0,
) -> dict[int, tuple[float, float]]:
    """Generate a printable A4 landscape calibration mat with 4 corner ArUco markers."""
    a4_w_mm, a4_h_mm = 297.0, 210.0
    mm_to_px = dpi / 25.4

    canvas_w = int(round(a4_w_mm * mm_to_px))
    canvas_h = int(round(a4_h_mm * mm_to_px))
    marker_px = int(round(marker_size_mm * mm_to_px))
    margin_px = int(round(margin_mm * mm_to_px))

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    corners_px = {
        0: (margin_px, margin_px),
        1: (canvas_w - margin_px - marker_px, margin_px),
        2: (canvas_w - margin_px - marker_px, canvas_h - margin_px - marker_px),
        3: (margin_px, canvas_h - margin_px - marker_px),
    }

    centers_mm: dict[int, tuple[float, float]] = {}

    for marker_id, (x, y) in corners_px.items():
        marker_gray = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        canvas[y : y + marker_px, x : x + marker_px] = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)

        cx_mm = (x + marker_px / 2.0) / mm_to_px
        cy_mm = (y + marker_px / 2.0) / mm_to_px
        centers_mm[marker_id] = (round(cx_mm, 2), round(cy_mm, 2))

        cv2.putText(
            canvas,
            f"ID:{marker_id}",
            (x, y + marker_px + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (80, 80, 80),
            2,
        )

    pad = margin_px + marker_px + 20
    cv2.rectangle(canvas, (pad, pad), (canvas_w - pad, canvas_h - pad), (210, 210, 210), 2)

    footer = (
        f"RedSea-CatchVision | A4 Landscape | DICT_4X4_50 | "
        f"{marker_size_mm:.0f}mm Markers | Print at 100% Scale"
    )
    cv2.putText(
        canvas,
        footer,
        (margin_px + marker_px + 30, canvas_h - margin_px - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (120, 120, 120),
        2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    logger.info(f"Calibration mat saved to {output_path} | Marker centers (mm): {centers_mm}")
    return centers_mm


if __name__ == "__main__":
    generate_a4_calibration_mat(Path("data/calibration/aruco_mat_a4.png"))
