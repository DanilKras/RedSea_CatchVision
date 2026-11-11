from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter1d
from scipy.spatial.distance import cdist
from skimage.graph import route_through_array

DEFAULT_MARKER_CENTERS_MM: dict[int, tuple[float, float]] = {
    0: (32.47, 32.47),
    1: (264.54, 32.47),
    2: (264.54, 177.50),
    3: (32.47, 177.50),
}

ARUCO_DICT_CANDIDATES: tuple[int, ...] = (
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_6X6_50,
)


class FishMeasurement(BaseModel):
    """Metric measurements of a single fish in millimeters and square centimeters."""

    fish_id: int
    straight_length_mm: float = Field(description="Straight length (max Feret diameter)")
    midline_length_mm: float = Field(description="Curved midline total length (TL)")
    body_depth_mm: float = Field(description="Maximum body depth perpendicular to axis")
    area_cm2: float = Field(description="Projected mask area in cm^2")
    curvature_ratio: float = Field(description="Ratio of midline length to straight length")
    head_tail_pts_rect: tuple[tuple[float, float], tuple[float, float]]
    midline_pts_rect: list[tuple[float, float]] = Field(default_factory=list)


@dataclass(slots=True)
class CalibrationResult:
    """Result of frame calibration against the ArUco reference mat."""

    homography_img_to_mm: np.ndarray
    homography_img_to_rect: np.ndarray
    rect_size_px: tuple[int, int]
    px_per_mm: float
    detected_marker_ids: list[int]
    reprojection_error_mm: float
    aruco_dict_id: int


class ArucoCalibrator:
    """Perspective calibration via an ArUco mat and metric sizing of SAM 2.1 masks."""

    def __init__(
        self,
        marker_centers_mm: dict[int, tuple[float, float]] | None = None,
        board_size_mm: tuple[float, float] = (297.0, 210.0),
        marker_size_mm: float | None = 35.0,
        px_per_mm: float = 4.0,
        canvas_padding_mm: float = 40.0,
        aruco_dict_id: int = cv2.aruco.DICT_4X4_50,
        min_markers: int = 2,
    ) -> None:
        self.marker_centers_mm = marker_centers_mm or dict(DEFAULT_MARKER_CENTERS_MM)
        self.board_width_mm, self.board_height_mm = board_size_mm
        self.marker_size_mm = marker_size_mm
        self.px_per_mm = px_per_mm
        self.canvas_padding_mm = canvas_padding_mm
        self.aruco_dict_id = aruco_dict_id
        self.min_markers = min_markers
        self._detector_params = cv2.aruco.DetectorParameters()

    def _detect_known_markers(
        self,
        image_gray: np.ndarray,
    ) -> tuple[dict[int, np.ndarray], int]:
        """Detect known markers from `marker_centers_mm`, trying fallback ArUco dicts."""
        dicts_to_try = [self.aruco_dict_id] + [
            d for d in ARUCO_DICT_CANDIDATES if d != self.aruco_dict_id
        ]

        best_matched: dict[int, np.ndarray] = {}
        best_dict = self.aruco_dict_id

        for dict_id in dicts_to_try:
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            detector = cv2.aruco.ArucoDetector(dictionary, self._detector_params)
            corners, ids, _ = detector.detectMarkers(image_gray)
            if ids is None or len(ids) == 0:
                continue

            matched: dict[int, np.ndarray] = {}
            for marker_corners, marker_id in zip(corners, ids.flatten(), strict=True):
                mid = int(marker_id)
                if mid in self.marker_centers_mm:
                    matched[mid] = marker_corners.reshape(4, 2).astype(np.float64)

            if len(matched) > len(best_matched):
                best_matched = matched
                best_dict = dict_id

            if len(best_matched) == len(self.marker_centers_mm):
                break

        return best_matched, best_dict

    def calibrate(self, image_bgr: np.ndarray) -> CalibrationResult:
        """Compute homography matrices from image pixels to mm and rectified canvas."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        matched, dict_id = self._detect_known_markers(gray)

        if len(matched) < self.min_markers:
            raise ValueError(
                f"Detected only {len(matched)} calibration markers ({sorted(matched.keys())}); "
                f"need at least {self.min_markers}."
            )

        if len(matched) >= 4:
            # Four marker centers give an exact homography independent of marker size
            src_pts = np.array([pts.mean(axis=0) for pts in matched.values()], dtype=np.float64)
            dst_pts = np.array(
                [self.marker_centers_mm[mid] for mid in matched],
                dtype=np.float64,
            )
            method = 0
        else:
            if self.marker_size_mm is None:
                raise ValueError(
                    f"Only {len(matched)} markers visible; set marker_size_mm to use corners."
                )
            logger.warning(
                f"Visible markers {sorted(matched.keys())} < 4; "
                "falling back to marker corner correspondences."
            )
            half = self.marker_size_mm / 2.0
            src_list: list[np.ndarray] = []
            dst_list: list[list[float]] = []
            for mid, corners_px in matched.items():
                cx, cy = self.marker_centers_mm[mid]
                src_list.append(corners_px)
                dst_list.extend(
                    [
                        [cx - half, cy - half],
                        [cx + half, cy - half],
                        [cx + half, cy + half],
                        [cx - half, cy + half],
                    ]
                )
            src_pts = np.vstack(src_list).astype(np.float64)
            dst_pts = np.array(dst_list, dtype=np.float64)
            method = cv2.RANSAC

        h_img_to_mm, _ = cv2.findHomography(
            src_pts, dst_pts, method=method, ransacReprojThreshold=2.0
        )
        if h_img_to_mm is None:
            raise RuntimeError("cv2.findHomography failed to find a valid perspective matrix.")

        projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), h_img_to_mm).reshape(-1, 2)
        reproj_error_mm = float(np.linalg.norm(projected - dst_pts, axis=1).mean())

        shift = self.canvas_padding_mm * self.px_per_mm
        t_mm_to_rect = np.array(
            [
                [self.px_per_mm, 0.0, shift],
                [0.0, self.px_per_mm, shift],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        h_img_to_rect = t_mm_to_rect @ h_img_to_mm

        rect_w = int(round((self.board_width_mm + 2 * self.canvas_padding_mm) * self.px_per_mm))
        rect_h = int(round((self.board_height_mm + 2 * self.canvas_padding_mm) * self.px_per_mm))

        logger.info(
            f"Calibrated frame with markers={sorted(matched.keys())} | "
            f"reproj_err={reproj_error_mm:.3f} mm | rect={rect_w}x{rect_h}px"
        )
        return CalibrationResult(
            homography_img_to_mm=h_img_to_mm,
            homography_img_to_rect=h_img_to_rect,
            rect_size_px=(rect_w, rect_h),
            px_per_mm=self.px_per_mm,
            detected_marker_ids=sorted(matched.keys()),
            reprojection_error_mm=reproj_error_mm,
            aruco_dict_id=dict_id,
        )

    def warp_image(self, image_bgr: np.ndarray, calib: CalibrationResult) -> np.ndarray:
        """Perspective-rectify a BGR image to a metric top-down view."""
        return cv2.warpPerspective(
            image_bgr,
            calib.homography_img_to_rect,
            calib.rect_size_px,
            flags=cv2.INTER_LINEAR,
        )

    def warp_mask(self, mask: np.ndarray, calib: CalibrationResult) -> np.ndarray:
        """Project a binary SAM 2.1 mask onto the isotropic metric plane."""
        mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
        warped = cv2.warpPerspective(
            mask_u8,
            calib.homography_img_to_rect,
            calib.rect_size_px,
            flags=cv2.INTER_NEAREST,
        )
        return warped > 127

    @staticmethod
    def _clean_binary_mask(mask: np.ndarray) -> np.ndarray:
        """Keep the largest connected component, fill holes, and smooth boundaries."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if num_labels <= 1:
            return np.zeros_like(mask, dtype=bool)

        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cleaned = labels == largest_label
        cleaned = binary_fill_holes(cleaned)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_u8 = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return cleaned_u8 > 0

    @staticmethod
    def _geodesic_midline(
        cleaned_mask: np.ndarray,
        p1_xy: np.ndarray,
        p2_xy: np.ndarray,
    ) -> np.ndarray:
        """
        Trace a smooth midline between endpoints (p1, p2) inside the mask,
        biasing the path toward the spine via distance transform.
        """
        ys, xs = np.where(cleaned_mask)
        y_min = max(0, int(ys.min()) - 2)
        y_max = min(cleaned_mask.shape[0], int(ys.max()) + 3)
        x_min = max(0, int(xs.min()) - 2)
        x_max = min(cleaned_mask.shape[1], int(xs.max()) + 3)
        sub_mask = cleaned_mask[y_min:y_max, x_min:x_max]

        edt = distance_transform_edt(sub_mask)
        max_edt = float(edt.max()) if edt.max() > 0 else 1.0
        cost = np.where(sub_mask, 1.0 + 0.35 * (1.0 - edt / max_edt), 1e6)

        fg_yx = np.column_stack(np.where(sub_mask))

        def snap_to_mask(pt_xy: np.ndarray) -> tuple[int, int]:
            rel_yx = np.array([pt_xy[1] - y_min, pt_xy[0] - x_min], dtype=np.float64)
            nearest = fg_yx[int(np.argmin(np.sum((fg_yx - rel_yx) ** 2, axis=1)))]
            return int(nearest[0]), int(nearest[1])

        start_yx = snap_to_mask(p1_xy)
        end_yx = snap_to_mask(p2_xy)

        indices, _ = route_through_array(
            cost,
            start=start_yx,
            end=end_yx,
            fully_connected=True,
            geometric=True,
        )
        path_xy = np.array([(c + x_min, r + y_min) for r, c in indices], dtype=np.float64)

        if len(path_xy) >= 9:
            smooth_x = gaussian_filter1d(path_xy[:, 0], sigma=3.0, mode="nearest")
            smooth_y = gaussian_filter1d(path_xy[:, 1], sigma=3.0, mode="nearest")
            smooth_x[0], smooth_y[0] = p1_xy[0], p1_xy[1]
            smooth_x[-1], smooth_y[-1] = p2_xy[0], p2_xy[1]
            path_xy = np.column_stack((smooth_x, smooth_y))

        return path_xy

    def measure_mask(
        self,
        fish_id: int,
        mask: np.ndarray,
        calib: CalibrationResult,
        mask_already_warped: bool = False,
    ) -> FishMeasurement:
        """Compute metric fish measurements from a SAM 2.1 mask."""
        warped_mask = mask if mask_already_warped else self.warp_mask(mask, calib)
        cleaned = self._clean_binary_mask(warped_mask)
        area_px = int(cleaned.sum())
        if area_px < 10:
            raise ValueError(f"Fish #{fish_id} warped mask is too small ({area_px} px).")

        area_cm2 = area_px / (calib.px_per_mm**2) / 100.0

        contours, _ = cv2.findContours(
            cleaned.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        largest_cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        hull = cv2.convexHull(largest_cnt.astype(np.float32)).reshape(-1, 2).astype(np.float64)

        dists = cdist(hull, hull)
        i_max, j_max = np.unravel_index(np.argmax(dists), dists.shape)
        p1, p2 = hull[i_max], hull[j_max]
        straight_len_px = float(dists[i_max, j_max])

        axis_vec = p2 - p1
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm > 1e-6:
            perp_unit = np.array([-axis_vec[1], axis_vec[0]]) / axis_norm
            projs = largest_cnt @ perp_unit
            body_depth_px = float(projs.max() - projs.min())
        else:
            body_depth_px = 0.0

        path_xy = self._geodesic_midline(cleaned, p1, p2)
        stride = max(1, len(path_xy) // 60)
        sampled = path_xy[::stride]
        if not np.allclose(sampled[-1], path_xy[-1]):
            sampled = np.vstack([sampled, path_xy[-1]])

        midline_len_px = max(
            float(np.linalg.norm(np.diff(sampled, axis=0), axis=1).sum()),
            straight_len_px,
        )
        midline_pts = [(float(x), float(y)) for x, y in sampled]

        straight_length_mm = straight_len_px / calib.px_per_mm
        midline_length_mm = midline_len_px / calib.px_per_mm
        body_depth_mm = body_depth_px / calib.px_per_mm

        return FishMeasurement(
            fish_id=fish_id,
            straight_length_mm=round(straight_length_mm, 2),
            midline_length_mm=round(midline_length_mm, 2),
            body_depth_mm=round(body_depth_mm, 2),
            area_cm2=round(area_cm2, 2),
            curvature_ratio=round(midline_length_mm / max(straight_length_mm, 1e-6), 3),
            head_tail_pts_rect=((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))),
            midline_pts_rect=midline_pts,
        )

    @staticmethod
    def draw_measurements_overlay(
        rect_bgr: np.ndarray,
        measurements: list[FishMeasurement],
        calib: CalibrationResult,
    ) -> np.ndarray:
        """Draw fish midlines and metric length labels onto a rectified frame."""
        canvas = rect_bgr.copy()
        for m in measurements:
            if len(m.midline_pts_rect) >= 2:
                pts = np.round(np.array(m.midline_pts_rect)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

            p1, p2 = m.head_tail_pts_rect
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0])), int(round(p2[1])))
            cv2.circle(canvas, pt1, 5, (0, 128, 255), -1)
            cv2.circle(canvas, pt2, 5, (0, 128, 255), -1)

            mid_x = int(round((p1[0] + p2[0]) / 2))
            mid_y = int(round((p1[1] + p2[1]) / 2)) - 12
            label = (f"#{m.fish_id} TL:{m.midline_length_mm / 10.0:.1f}"
                     f"cm H:{m.body_depth_mm / 10.0:.1f}cm")
            cv2.putText(
                canvas,
                label,
                (max(10, mid_x - 80), max(25, mid_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            canvas,
            f"Markers: {calib.detected_marker_ids} "
            f"Reproj err: {calib.reprojection_error_mm:.2f} mm",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        return canvas
