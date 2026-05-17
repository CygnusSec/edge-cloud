"""
Edge inference module for the Edge-Cloud Demo system.
Loads a lightweight YOLOv8 model and runs object detection on images.
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from models import BoundingBox, DetectedObject, EdgeInferResult

# Standard YOLO input size
_YOLO_INPUT_SIZE = (640, 640)

# Performance threshold in milliseconds
_LATENCY_THRESHOLD_MS = 500.0


class EdgeInference:
    """
    Lightweight edge inference using YOLOv8n/s.

    Loads the model once at construction time and exposes
    ``preprocess()`` for image preparation and ``infer()`` for
    end-to-end detection on a single image.
    """

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        """
        Initialise the edge inference engine.

        Parameters
        ----------
        model_path:
            Path to (or name of) the YOLOv8 model weights file.
            Defaults to ``"yolov8n.pt"`` (nano variant).
            Ultralytics will download the weights automatically if the
            file is not found locally.
        """
        self.model_path = model_path
        self.model: YOLO = YOLO(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Read and pre-process an image for YOLO inference.

        Steps:
        1. Read the image from *image_path* using ``cv2.imread``.
        2. Resize to 640×640 pixels.
        3. Return the resulting numpy array (BGR, uint8).

        Parameters
        ----------
        image_path:
            Absolute or relative path to the source image file.

        Returns
        -------
        np.ndarray
            Pre-processed image array with shape ``(640, 640, 3)``.

        Raises
        ------
        ValueError
            If the image cannot be read (file missing, corrupt, or not
            a recognised image format).
        """
        img = cv2.imread(image_path)
        if img is None:
            print(
                f"[EdgeInference] ERROR: Cannot read image: {image_path}",
                file=sys.stderr,
            )
            raise ValueError(
                f"Cannot read image at path: '{image_path}'. "
                "The file may be missing, corrupt, or not a valid image format."
            )

        resized = cv2.resize(img, _YOLO_INPUT_SIZE)
        return resized

    def infer(self, image_path: str) -> EdgeInferResult:
        """
        Run YOLO inference on a single image and return structured results.

        The method:
        * Measures end-to-end latency with ``time.perf_counter()``.
        * Passes *image_path* directly to the YOLO model (Ultralytics
          handles its own I/O and pre-processing internally).
        * Extracts per-detection class name, confidence, and bounding box.
        * Computes ``max_confidence`` (0.0 when no objects are detected).
        * Logs a performance-violation warning to stderr when latency
          exceeds 500 ms.

        Parameters
        ----------
        image_path:
            Absolute or relative path to the source image file.

        Returns
        -------
        EdgeInferResult
            Structured inference result including detected objects,
            max confidence, object count, latency, and class list.

        Raises
        ------
        ValueError
            If the image cannot be read (propagated from ``preprocess``
            when called for validation, or raised directly here).
        """
        # Validate that the image is readable before running inference
        img = cv2.imread(image_path)
        if img is None:
            print(
                f"[EdgeInference] ERROR: Cannot read image: {image_path}",
                file=sys.stderr,
            )
            raise ValueError(
                f"Cannot read image at path: '{image_path}'. "
                "The file may be missing, corrupt, or not a valid image format."
            )

        # --- Run inference and measure latency ---
        t_start = time.perf_counter()
        results = self.model(image_path, verbose=False)
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000.0

        # --- Log performance violation if latency exceeds threshold ---
        if latency_ms > _LATENCY_THRESHOLD_MS:
            print(
                f"[EdgeInference] PERFORMANCE VIOLATION: "
                f"Inference on '{image_path}' took {latency_ms:.1f} ms "
                f"(threshold: {_LATENCY_THRESHOLD_MS:.0f} ms)",
                file=sys.stderr,
            )

        # --- Extract detections from the first result ---
        detected_objects: list[DetectedObject] = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes  # ultralytics Boxes object

            if boxes is not None and len(boxes) > 0:
                # xyxy: shape (N, 4) — absolute pixel coordinates
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                names = result.names  # dict[int, str]

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])
                    confidence = float(confs[i])
                    class_name = names[cls_ids[i]]

                    detected_objects.append(
                        DetectedObject(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        )
                    )

        # --- Compute summary statistics ---
        if detected_objects:
            max_confidence = max(obj.confidence for obj in detected_objects)
        else:
            max_confidence = 0.0

        object_count = len(detected_objects)
        detected_classes = list({obj.class_name for obj in detected_objects})

        return EdgeInferResult(
            image_path=image_path,
            objects=detected_objects,
            max_confidence=max_confidence,
            object_count=object_count,
            latency_ms=latency_ms,
            detected_classes=detected_classes,
        )
