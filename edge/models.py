"""
Shared data models for the Edge-Cloud Demo system.
Defines dataclasses used across edge, cloud, and dashboard components.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    """Bounding box coordinates for a detected object."""
    x1: float  # Top-left x coordinate (pixels)
    y1: float  # Top-left y coordinate (pixels)
    x2: float  # Bottom-right x coordinate (pixels)
    y2: float  # Bottom-right y coordinate (pixels)


@dataclass
class DetectedObject:
    """A single detected object with class, confidence, and bounding box."""
    class_name: str       # Object class label (e.g., "person", "car")
    confidence: float     # Detection confidence score (0.0–1.0)
    bbox: BoundingBox     # Bounding box around the detected object


@dataclass
class EdgeInferResult:
    """Result of running inference on the edge node."""
    image_path: str
    objects: list[DetectedObject]    # List of detected objects
    max_confidence: float            # Highest confidence score (0.0 if no objects)
    object_count: int                # Number of detected objects
    latency_ms: float                # Inference time in milliseconds
    detected_classes: list[str]      # List of detected class names


@dataclass
class ProcessingDecision:
    """Decision made by the Offload Manager for a single image."""
    processing_place: str            # "edge" or "cloud"
    offloaded: bool                  # True if image was sent to cloud
    uploaded_bytes: int              # Bytes uploaded to cloud (0 if processed at edge)
    final_confidence: float          # Final confidence score after processing
    final_object_count: int          # Final number of detected objects
    final_objects: list[DetectedObject]  # Final list of detected objects
    total_latency_ms: float          # Total latency including network time if offloaded


@dataclass
class CloudResponse:
    """Response returned by the Cloud Node API."""
    objects: list[DetectedObject]    # List of detected objects
    confidence: float                # Highest confidence score
    object_count: int                # Number of detected objects
    latency_ms: float                # Cloud inference time in milliseconds
    detected_classes: list[str]      # List of detected class names


@dataclass
class LogRecord:
    """A single log entry for one processed image."""
    image_name: str          # Image filename (e.g., "easy/img001.jpg")
    latency_ms: float        # Total processing time (ms)
    uploaded_bytes: int      # Bytes uploaded to cloud (0 for edge-only)
    processing_place: str    # "edge" or "cloud"
    confidence: float        # Highest confidence score (0.0–1.0)
    object_count: int        # Number of detected objects
    offloaded: bool          # True if image was offloaded to cloud
    detected_classes: str    # JSON string of detected class names list
    cpu_percent: float       # CPU usage at time of processing (%)
    ram_mb: float            # RAM usage at time of processing (MB)
    difficulty: str          # Image difficulty: "easy", "medium", or "hard"
    scenario: str            # Scenario: "cloud_only", "edge_only", or "edge_cloud"
    timestamp: str           # ISO 8601 timestamp


@dataclass
class ExperimentSummary:
    """Aggregated summary statistics for a completed experiment scenario."""
    scenario: str                    # "cloud_only", "edge_only", or "edge_cloud"
    total_images: int                # Total number of images processed
    avg_latency_ms: float            # Average latency per image (ms)
    total_uploaded_bytes: int        # Total bytes uploaded to cloud
    throughput_img_per_sec: float    # Images processed per second
    avg_confidence: float            # Average confidence score
    avg_cpu_percent: float           # Average CPU usage (%)
    avg_ram_mb: float                # Average RAM usage (MB)
    offload_ratio: float             # Fraction of images offloaded (0.0 if N/A)
    start_time: str                  # ISO 8601 experiment start timestamp
    end_time: str                    # ISO 8601 experiment end timestamp
    total_duration_sec: float        # Total experiment duration in seconds
