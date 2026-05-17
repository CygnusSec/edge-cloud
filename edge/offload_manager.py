"""
Offload Manager for the Edge-Cloud Demo system.

Decides whether to process an image at the edge or offload it to the cloud
based on a configurable confidence threshold. Handles cloud communication
with retry logic.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import requests

from models import CloudResponse, DetectedObject, BoundingBox, EdgeInferResult, ProcessingDecision

# Default offload threshold
DEFAULT_THRESHOLD = 0.6

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF = [1.0, 2.0, 4.0]  # seconds


class OffloadManager:
    """
    Decides whether to process an image at the edge or offload to cloud.

    Usage:
        manager = OffloadManager(threshold=0.6, cloud_url="http://cloud_node:8000")
        decision = manager.decide(edge_result, image_path)
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        cloud_url: str = "http://cloud_node:8000",
    ) -> None:
        """
        Initialise the Offload Manager.

        Args:
            threshold: Confidence threshold (0.0–1.0). Images with
                max_confidence < threshold are offloaded to cloud.
                Defaults to 0.6. Must be in (0.0, 1.0).
            cloud_url: Base URL of the Cloud Node API.
        """
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"threshold must be in (0.0, 1.0), got {threshold}"
            )
        self.threshold = threshold
        self.cloud_url = cloud_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self, edge_result: EdgeInferResult, image_path: str
    ) -> ProcessingDecision:
        """
        Decide whether to use edge result or offload to cloud.

        If edge max_confidence >= threshold: keep edge result.
        If edge max_confidence < threshold: offload to cloud.

        Args:
            edge_result: Result from EdgeInference.infer().
            image_path: Path to the original image file.

        Returns:
            ProcessingDecision with final result and metadata.
        """
        if edge_result.max_confidence >= self.threshold:
            # Keep edge result
            return ProcessingDecision(
                processing_place="edge",
                offloaded=False,
                uploaded_bytes=0,
                final_confidence=edge_result.max_confidence,
                final_object_count=edge_result.object_count,
                final_objects=edge_result.objects,
                total_latency_ms=edge_result.latency_ms,
            )
        else:
            # Offload to cloud
            t_start = time.perf_counter()
            cloud_response = self.offload_to_cloud(image_path)
            total_latency_ms = (time.perf_counter() - t_start) * 1000.0

            if cloud_response is None:
                # Fallback to edge result on cloud failure
                print(
                    f"[OffloadManager] Cloud unavailable, falling back to edge result for: {image_path}",
                    file=sys.stderr,
                )
                return ProcessingDecision(
                    processing_place="edge",
                    offloaded=False,
                    uploaded_bytes=0,
                    final_confidence=edge_result.max_confidence,
                    final_object_count=edge_result.object_count,
                    final_objects=edge_result.objects,
                    total_latency_ms=edge_result.latency_ms,
                )

            uploaded_bytes = os.path.getsize(image_path)

            return ProcessingDecision(
                processing_place="cloud",
                offloaded=True,
                uploaded_bytes=uploaded_bytes,
                final_confidence=cloud_response.confidence,
                final_object_count=cloud_response.object_count,
                final_objects=cloud_response.objects,
                total_latency_ms=total_latency_ms,
            )

    def offload_to_cloud(self, image_path: str) -> CloudResponse | None:
        """
        Send image to Cloud Node via POST /predict with retry logic.

        Retries up to MAX_RETRIES times with exponential backoff.
        Returns None if all retries fail.

        Args:
            image_path: Path to the image file to upload.

        Returns:
            CloudResponse on success, None on failure.
        """
        url = f"{self.cloud_url}/predict"

        for attempt in range(MAX_RETRIES):
            try:
                with open(image_path, "rb") as f:
                    response = requests.post(
                        url,
                        files={"file": (os.path.basename(image_path), f, "image/jpeg")},
                        timeout=30,
                    )

                if response.status_code == 200:
                    data = response.json()
                    # Parse objects from response
                    objects = []
                    for obj in data.get("objects", []):
                        bbox_data = obj.get("bbox", {})
                        objects.append(
                            DetectedObject(
                                class_name=obj["class_name"],
                                confidence=obj["confidence"],
                                bbox=BoundingBox(
                                    x1=bbox_data["x1"],
                                    y1=bbox_data["y1"],
                                    x2=bbox_data["x2"],
                                    y2=bbox_data["y2"],
                                ),
                            )
                        )
                    return CloudResponse(
                        objects=objects,
                        confidence=data.get("confidence", 0.0),
                        object_count=data.get("object_count", 0),
                        latency_ms=data.get("latency_ms", 0.0),
                        detected_classes=data.get("detected_classes", []),
                    )
                else:
                    print(
                        f"[OffloadManager] Cloud returned HTTP {response.status_code} "
                        f"for {image_path} (attempt {attempt + 1}/{MAX_RETRIES})",
                        file=sys.stderr,
                    )

            except (requests.ConnectionError, requests.Timeout) as exc:
                print(
                    f"[OffloadManager] Connection error on attempt {attempt + 1}/{MAX_RETRIES}: {exc}",
                    file=sys.stderr,
                )

            # Wait before retry (except on last attempt)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])

        print(
            f"[OffloadManager] All {MAX_RETRIES} attempts failed for: {image_path}",
            file=sys.stderr,
        )
        return None
