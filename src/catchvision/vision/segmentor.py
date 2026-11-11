from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import SAM

from catchvision.config import SegmentedFish, Settings


class FishSegmentor:
    """Zero-shot fish detection and instance segmentation (GroundingDINO + SAM 2.1 or SAM 3)."""

    def __init__(self, settings: Settings | None = None, device: str | None = None) -> None:
        self.settings = settings or Settings()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = self.settings.segmentor_backend
        logger.info(f"Initializing FishSegmentor (backend={self.backend}, device={self.device})")

        if self.backend == "sam3":
            try:
                from transformers import Sam3Model, Sam3Processor
            except ImportError as exc:
                raise ImportError(
                    "Sam3Model/Sam3Processor require a newer `transformers` release. "
                    "Upgrade with `uv sync --upgrade-package transformers`."
                ) from exc

            self.sam3_processor = Sam3Processor.from_pretrained(self.settings.sam3_model_id)
            self.sam3_model = Sam3Model.from_pretrained(self.settings.sam3_model_id).to(
                self.device
            )
            self.sam3_model.eval()
        else:
            if self.device.startswith("cuda"):
                torch.backends.cudnn.enabled = False

            self.processor = AutoProcessor.from_pretrained(self.settings.detector_model_id)
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.settings.detector_model_id,
                attn_implementation="eager",
            ).to(self.device)
            self.detector.eval()

            self.sam = SAM(self.settings.sam_weights)

    @staticmethod
    def _suppress_contained_boxes(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_thresh: float = 0.5,
        containment_thresh: float = 0.65,
    ) -> list[int]:
        """Suppress overlapping duplicates (IoU) and smaller sub-boxes (e.g., fins) inside larger detections."""
        if len(boxes) == 0:
            return []

        order = np.argsort(-scores)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep: list[int] = []

        for idx in order:
            box_i = boxes[idx]
            drop = False
            for kept_idx in keep:
                box_k = boxes[kept_idx]
                ix1 = max(box_i[0], box_k[0])
                iy1 = max(box_i[1], box_k[1])
                ix2 = min(box_i[2], box_k[2])
                iy2 = min(box_i[3], box_k[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter == 0.0:
                    continue

                union = areas[idx] + areas[kept_idx] - inter
                iou = inter / max(union, 1e-6)
                containment = inter / max(areas[idx], 1e-6)
                if iou > iou_thresh or containment > containment_thresh:
                    drop = True
                    break

            if not drop:
                keep.append(int(idx))
        return keep

    @torch.inference_mode()
    def detect_fish_boxes(self, image_pil: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Run GroundingDINO and filter duplicate or nested bounding boxes."""
        if self.backend != "gdino_sam2":
            raise RuntimeError("detect_fish_boxes is only available for backend='gdino_sam2'.")

        inputs = self.processor(
            images=image_pil,
            text=self.settings.text_prompt,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.detector(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.settings.box_threshold,
            text_threshold=self.settings.text_threshold,
            target_sizes=[(image_pil.height, image_pil.width)],
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        keep = self._suppress_contained_boxes(
            boxes, scores, iou_thresh=0.5, containment_thresh=0.65
        )
        return boxes[keep], scores[keep]

    @torch.inference_mode()
    def _predict_sam3(
        self,
        image_pil: Image.Image,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run single-pass Promptable Concept Segmentation with SAM 3."""
        prompt = self.settings.text_prompt.rstrip(".").strip() or "fish"
        inputs = self.sam3_processor(
            images=image_pil,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.sam3_model(**inputs)
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=self.settings.box_threshold,
            mask_threshold=0.5,
            target_sizes=[(image_pil.height, image_pil.width)],
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        masks = results["masks"].cpu().numpy().astype(bool)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        if len(boxes) == 0:
            return boxes, scores, masks

        keep = self._suppress_contained_boxes(
            boxes, scores, iou_thresh=0.5, containment_thresh=0.65
        )
        return boxes[keep], scores[keep], masks[keep]

    @torch.inference_mode()
    def segment_image(
        self,
        image_path: Path,
        output_dir: Path,
        save_overlay: bool = True,
    ) -> tuple[list[SegmentedFish], list[np.ndarray]]:
        """Detect fish boxes, extract instance masks, and save raw/neutral-background crops."""
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        if self.backend == "sam3":
            boxes, scores, masks_np = self._predict_sam3(image_pil)
            logger.info(
                f"SAM 3 detected {len(boxes)} instances (scores={np.round(scores, 3).tolist()})"
            )
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

            if len(boxes) == 0 or len(masks_np) == 0:
                logger.warning(
                    f"No fish detected in {image_path.name} "
                    f"(backend=sam3, box_threshold={self.settings.box_threshold})"
                )
                return [], []
        else:
            boxes, scores = self.detect_fish_boxes(image_pil)
            logger.info(
                f"GroundingDINO detected {len(boxes)} boxes (scores={np.round(scores, 3).tolist()})"
            )
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

            if len(boxes) == 0:
                logger.warning(
                    f"No fish detected in {image_path.name} "
                    f"(box_threshold={self.settings.box_threshold})"
                )
                return [], []

            sam_results = self.sam.predict(
                source=image_bgr,
                bboxes=boxes.tolist(),
                device=self.device,
                verbose=False,
            )[0]

            if sam_results.masks is None or len(sam_results.masks.data) == 0:
                logger.warning(f"SAM 2.1 produced no masks for {image_path.name}")
                return [], []

            masks_np = sam_results.masks.data.cpu().numpy().astype(bool)

        output_dir.mkdir(parents=True, exist_ok=True)
        overlay = image_bgr.copy()

        segmented_items: list[SegmentedFish] = []
        valid_masks: list[np.ndarray] = []

        for idx, (box, score, mask) in enumerate(zip(boxes, scores, masks_np, strict=True)):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            h, w = image_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            area_px = int(mask.sum())
            if area_px == 0 or x2 <= x1 or y2 <= y1:
                continue

            raw_crop = image_bgr[y1:y2, x1:x2]
            raw_crop_path = output_dir / f"{image_path.stem}_fish_{idx}_raw.jpg"
            cv2.imwrite(str(raw_crop_path), raw_crop)

            neutral_bg = np.full_like(image_bgr, 128)
            neutral_bg[mask] = image_bgr[mask]
            masked_crop = neutral_bg[y1:y2, x1:x2]
            masked_crop_path = output_dir / f"{image_path.stem}_fish_{idx}_masked.jpg"
            cv2.imwrite(str(masked_crop_path), masked_crop)

            if save_overlay:
                color = np.random.default_rng(idx + 42).integers(60, 255, size=3).tolist()
                overlay[mask] = (overlay[mask] * 0.55 + np.array(color) * 0.45).astype(np.uint8)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    overlay,
                    f"#{idx} {score:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            segmented_items.append(
                SegmentedFish(
                    fish_id=idx,
                    confidence=float(score),
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    mask_area_px=area_px,
                    crop_path=raw_crop_path,
                    masked_crop_path=masked_crop_path,
                )
            )
            valid_masks.append(mask)

        if save_overlay and segmented_items:
            overlay_path = output_dir / f"{image_path.stem}_overlay.jpg"
            cv2.imwrite(str(overlay_path), overlay)
            logger.info(
                f"Segmented {len(segmented_items)} fish in {image_path.name} -> {overlay_path}"
            )
        elif not segmented_items:
            logger.warning(f"All {len(boxes)} masks were filtered out (zero area or invalid bbox)")

        return segmented_items, valid_masks