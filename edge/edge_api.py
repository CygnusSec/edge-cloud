"""
Edge Node API - FastAPI application for lightweight YOLOv8n inference.

Provides the same interface as the Cloud Node but uses a smaller model.
Used by the Dashboard to perform real edge inference.

Provides:
  POST /predict  - Accept image upload, run YOLOv8n inference, return JSON
  GET  /health   - Health check
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model: YOLO | None = None
_MODEL_NAME = "yolov8n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the YOLOv8n model at startup and release at shutdown."""
    global _model
    _model = YOLO(f"{_MODEL_NAME}.pt")
    yield
    _model = None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Edge-Cloud Demo - Edge Node API",
    description="Lightweight YOLOv8n inference API for the edge node.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class BoundingBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedObjectSchema(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBoxSchema


class EdgePredictResponse(BaseModel):
    objects: list[DetectedObjectSchema]
    confidence: float           # Highest confidence score (0.0 if no objects)
    object_count: int           # Number of detected objects
    latency_ms: float           # Edge inference time in milliseconds
    detected_classes: list[str] # Unique detected class names
    processing_place: str       # Always "edge"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=EdgePredictResponse)
async def predict(file: UploadFile = File(...)) -> EdgePredictResponse:
    """
    Accept an image via multipart/form-data, run YOLOv8n inference, return JSON.

    Raises:
        HTTPException(400): Image is invalid or cannot be decoded.
        HTTPException(500): Internal error during inference.
    """
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Cannot read image: empty file")

    np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Cannot read image: file is not a valid image format",
        )

    try:
        t_start = time.perf_counter()
        results = _model(image, verbose=False)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Internal inference error: {exc}",
        ) from exc

    detected_objects: list[DetectedObjectSchema] = []
    detected_classes: list[str] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name: str = result.names[cls_id]
            conf: float = float(box.conf[0].item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())

            detected_objects.append(
                DetectedObjectSchema(
                    class_name=class_name,
                    confidence=conf,
                    bbox=BoundingBoxSchema(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
            if class_name not in detected_classes:
                detected_classes.append(class_name)

    max_confidence = (
        max(obj.confidence for obj in detected_objects) if detected_objects else 0.0
    )

    return EdgePredictResponse(
        objects=detected_objects,
        confidence=max_confidence,
        object_count=len(detected_objects),
        latency_ms=latency_ms,
        detected_classes=detected_classes,
        processing_place="edge",
    )


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": _MODEL_NAME,
        "model_loaded": _model is not None,
    }
