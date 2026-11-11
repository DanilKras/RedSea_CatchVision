from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from catchvision.geometry.calibration import ARUCO_DICT_CANDIDATES, ArucoCalibrator
from catchvision.storage.catch_db import CatchDatabase
from catchvision.vision.classifier import FishClassifier
from catchvision.vision.segmentor import FishSegmentor

DEFAULT_SCENARIOS: tuple[str, ...] = (
    "clean",
    "sun_glare",
    "water_droplets",
    "low_light",
    "motion_blur",
    "marker_occlusion_1",
    "marker_occlusion_2",
)


class BenchmarkRow(BaseModel):
    """Metrics recorded for a single stress-test scenario run."""

    scenario: str
    severity: float
    calibrated: bool
    detected_markers: list[int]
    reproj_error_mm: float | None
    detected_fish_count: int
    measured_length_mm: float | None
    length_abs_error_mm: float | None
    length_pct_error: float | None
    top_species: str | None
    cls_confidence: float | None
    species_matches_ref: bool
    runtime_ms: float


class FieldStressBenchmark:
    """Evaluates pipeline stability under simulated dockside degradations."""

    def __init__(
        self,
        segmentor: FishSegmentor | None = None,
        calibrator: ArucoCalibrator | None = None,
        classifier: FishClassifier | None = None,
    ) -> None:
        self.segmentor = segmentor or FishSegmentor()
        self.calibrator = calibrator or ArucoCalibrator()
        self.classifier = classifier or FishClassifier()

    @staticmethod
    def _find_aruco_boxes(image_bgr: np.ndarray) -> dict[int, np.ndarray]:
        """Locate ArUco marker polygons so occlusion patches can target them accurately."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        params = cv2.aruco.DetectorParameters()
        best: dict[int, np.ndarray] = {}
        for dict_id in ARUCO_DICT_CANDIDATES:
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue
            found = {
                int(mid): c.reshape(4, 2).astype(np.int32)
                for c, mid in zip(corners, ids.flatten(), strict=True)
            }
            if len(found) > len(best):
                best = found
        return best

    @staticmethod
    def _apply_sun_glare(
        image_bgr: np.ndarray, severity: float, rng: np.random.Generator
    ) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        glare = np.zeros((h, w), dtype=np.float32)
        n_spots = 2 + int(round(3 * severity))
        for _ in range(n_spots):
            cx = int(rng.integers(int(0.15 * w), int(0.85 * w)))
            cy = int(rng.integers(int(0.15 * h), int(0.85 * h)))
            ax1 = int(rng.integers(int(0.05 * w), int(0.16 * w)))
            ax2 = int(rng.integers(int(0.03 * h), int(0.10 * h)))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(glare, (cx, cy), (ax1, ax2), angle, 0, 360, 1.0, -1)

        k = max(31, int(min(h, w) * 0.14) | 1)
        glare = cv2.GaussianBlur(glare, (k, k), 0)
        if glare.max() > 0:
            glare /= glare.max()

        intensity = 0.45 + 0.50 * severity
        out = image_bgr.astype(np.float32) + glare[..., None] * (255.0 * intensity)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_water_droplets(
        image_bgr: np.ndarray, severity: float, rng: np.random.Generator
    ) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        blur_k = max(15, int(min(h, w) * 0.04) | 1)
        blurred = cv2.GaussianBlur(image_bgr, (blur_k, blur_k), 0)

        droplet_mask = np.zeros((h, w), dtype=np.float32)
        n_drops = int(12 + 35 * severity)
        for _ in range(n_drops):
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))
            r = int(rng.integers(max(6, min(h, w) // 70), max(14, min(h, w) // 22)))
            cv2.circle(droplet_mask, (cx, cy), r, 1.0, -1)

        droplet_mask = cv2.GaussianBlur(droplet_mask, (9, 9), 0)[..., None]
        hazed = cv2.addWeighted(image_bgr, 1.0 - 0.15 * severity, blurred, 0.15 * severity, 0)
        out = hazed.astype(np.float32) * (1.0 - 0.85 * droplet_mask) + (
            blurred.astype(np.float32) + 22.0 * severity
        ) * (0.85 * droplet_mask)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_low_light(
        image_bgr: np.ndarray, severity: float, rng: np.random.Generator
    ) -> np.ndarray:
        gamma = 1.0 + 1.8 * severity
        gain = 1.0 - 0.55 * severity
        norm = image_bgr.astype(np.float32) / 255.0
        dark = np.power(norm, gamma) * gain * 255.0
        noise_std = 4.0 + 14.0 * severity
        noise = rng.normal(0.0, noise_std, size=image_bgr.shape).astype(np.float32)
        return np.clip(dark + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_motion_blur(
        image_bgr: np.ndarray, severity: float, rng: np.random.Generator
    ) -> np.ndarray:
        k = max(5, int(5 + 24 * severity) | 1)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        angle = float(rng.uniform(-35, 35))
        rot = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        kernel /= max(float(kernel.sum()), 1e-6)
        return cv2.filter2D(image_bgr, -1, kernel)

    @staticmethod
    def _apply_marker_occlusion(
        image_bgr: np.ndarray,
        marker_boxes: dict[int, np.ndarray],
        n_occlude: int,
        severity: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = image_bgr.copy()
        if not marker_boxes:
            return out

        ids = sorted(marker_boxes.keys())
        chosen = rng.choice(ids, size=min(n_occlude, len(ids)), replace=False)
        for mid in chosen:
            pts = marker_boxes[int(mid)]
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            pad = int(max(10, 0.25 * max(x_max - x_min, y_max - y_min) * (0.5 + severity)))
            x1 = max(0, int(x_min - pad))
            y1 = max(0, int(y_min - pad))
            x2 = min(out.shape[1], int(x_max + pad))
            y2 = min(out.shape[0], int(y_max + pad))
            cv2.rectangle(out, (x1, y1), (x2, y2), (45, 50, 58), -1)
        return out

    def apply_perturbation(
        self,
        image_bgr: np.ndarray,
        scenario: str,
        severity: float = 0.6,
        seed: int = 42,
        marker_boxes: dict[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Apply a synthetic field degradation scenario to a BGR image."""
        sev = float(np.clip(severity, 0.0, 1.0))
        rng = np.random.default_rng(seed)

        if scenario == "clean":
            return image_bgr.copy()
        if scenario == "sun_glare":
            return self._apply_sun_glare(image_bgr, sev, rng)
        if scenario == "water_droplets":
            return self._apply_water_droplets(image_bgr, sev, rng)
        if scenario == "low_light":
            return self._apply_low_light(image_bgr, sev, rng)
        if scenario == "motion_blur":
            return self._apply_motion_blur(image_bgr, sev, rng)
        if scenario in ("marker_occlusion_1", "marker_occlusion_2"):
            n_occlude = 1 if scenario == "marker_occlusion_1" else 2
            boxes = marker_boxes if marker_boxes is not None else self._find_aruco_boxes(image_bgr)
            return self._apply_marker_occlusion(image_bgr, boxes, n_occlude, sev, rng)

        raise ValueError(f"Unknown stress scenario: {scenario}")

    def _evaluate_single(
        self,
        image_bgr: np.ndarray,
        scenario: str,
        severity: float,
        work_dir: Path,
        ref_length_mm: float | None,
        ref_species: str | None,
    ) -> BenchmarkRow:
        perturbed_path = work_dir / f"{scenario}_sev{severity:.2f}.jpg"
        cv2.imwrite(str(perturbed_path), image_bgr)

        t0 = time.perf_counter()

        calib = None
        calibrated = False
        detected_markers: list[int] = []
        reproj_err: float | None = None
        try:
            calib = self.calibrator.calibrate(image_bgr)
            calibrated = True
            detected_markers = calib.detected_marker_ids
            reproj_err = round(calib.reprojection_error_mm, 3)
        except Exception as exc:
            logger.warning(f"[{scenario} | sev={severity}] Calibration failed: {exc}")

        items, masks = self.segmentor.segment_image(
            perturbed_path,
            output_dir=work_dir / f"{scenario}_sev{severity:.2f}",
            save_overlay=True,
        )

        measured_mm: float | None = None
        top_species: str | None = None
        cls_conf: float | None = None

        if items:
            best_idx = max(range(len(items)), key=lambda i: items[i].confidence)
            best_item = items[best_idx]
            best_mask = masks[best_idx]

            if calib is not None:
                try:
                    meas = self.calibrator.measure_mask(best_item.fish_id, best_mask, calib)
                    measured_mm = meas.midline_length_mm
                except Exception as exc:
                    logger.warning(f"[{scenario} | sev={severity}] Measurement failed: {exc}")

            cls_res = self.classifier.classify(
                best_item.masked_crop_path,
                fish_id=best_item.fish_id,
                length_cm=(measured_mm / 10.0) if measured_mm else None,
            )
            top_species = cls_res.top_species
            cls_conf = cls_res.confidence

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        abs_err = None
        pct_err = None
        if measured_mm is not None and ref_length_mm is not None and ref_length_mm > 0:
            abs_err = round(abs(measured_mm - ref_length_mm), 2)
            pct_err = round(100.0 * abs_err / ref_length_mm, 2)

        species_match = bool(
            top_species is not None and ref_species is not None and top_species == ref_species
        )

        return BenchmarkRow(
            scenario=scenario,
            severity=severity,
            calibrated=calibrated,
            detected_markers=detected_markers,
            reproj_error_mm=reproj_err,
            detected_fish_count=len(items),
            measured_length_mm=measured_mm,
            length_abs_error_mm=abs_err,
            length_pct_error=pct_err,
            top_species=top_species,
            cls_confidence=cls_conf,
            species_matches_ref=species_match,
            runtime_ms=elapsed_ms,
        )

    def run_suite(
        self,
        image_path: Path,
        output_dir: Path,
        ground_truth_length_mm: float | None = None,
        ground_truth_species: str | None = None,
        scenarios: Sequence[str] = DEFAULT_SCENARIOS,
        severities: Sequence[float] = (0.4, 0.8),
        db: CatchDatabase | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Run the stress benchmark suite and return a metrics summary table."""
        base_bgr = cv2.imread(str(image_path))
        if base_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        marker_boxes = self._find_aruco_boxes(base_bgr)

        baseline = self._evaluate_single(
            base_bgr,
            scenario="clean",
            severity=0.0,
            work_dir=output_dir,
            ref_length_mm=ground_truth_length_mm,
            ref_species=ground_truth_species,
        )
        ref_len = ground_truth_length_mm or baseline.measured_length_mm
        ref_sp = ground_truth_species or baseline.top_species

        if ground_truth_length_mm is None and baseline.measured_length_mm is not None:
            baseline.length_abs_error_mm = 0.0
            baseline.length_pct_error = 0.0
        if ground_truth_species is None and baseline.top_species is not None:
            baseline.species_matches_ref = True

        results: list[BenchmarkRow] = [baseline]

        for scenario in scenarios:
            if scenario == "clean":
                continue
            for sev in severities:
                perturbed = self.apply_perturbation(
                    base_bgr,
                    scenario=scenario,
                    severity=sev,
                    seed=seed,
                    marker_boxes=marker_boxes,
                )
                row = self._evaluate_single(
                    perturbed,
                    scenario=scenario,
                    severity=sev,
                    work_dir=output_dir,
                    ref_length_mm=ref_len,
                    ref_species=ref_sp,
                )
                results.append(row)

        df = pd.DataFrame([r.model_dump() for r in results])
        csv_path = output_dir / "stress_benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Stress benchmark finished ({len(df)} runs) -> {csv_path}")

        if db is not None:
            run_id = uuid.uuid4().hex[:10]
            db.save_benchmark_rows(run_id, image_path.name, [r.model_dump() for r in results])

        return df
