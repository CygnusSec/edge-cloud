"""
Edge-Cloud scenario: edge processes first, offloads to cloud if confidence is low.

Results are saved to results/edge_cloud.csv.

Usage:
    python run_edge_cloud.py [--model MODEL] [--threshold FLOAT]
                             [--cloud-url URL] [--dataset-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from models import LogRecord
from logger import ExperimentLogger
from edge_infer import EdgeInference
from offload_manager import OffloadManager

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(dataset_dir: str) -> list[tuple[str, str]]:
    """Collect all images from dataset/images/easy, medium, hard."""
    images = []
    for difficulty in ("easy", "medium", "hard"):
        subdir = os.path.join(dataset_dir, "images", difficulty)
        if not os.path.isdir(subdir):
            continue
        for fname in sorted(os.listdir(subdir)):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                images.append((os.path.join(subdir, fname), difficulty))

    if not images:
        print(
            f"[run_edge_cloud] ERROR: No images found in {dataset_dir}/images/{{easy,medium,hard}}",
            file=sys.stderr,
        )
        sys.exit(1)

    return images


def run_edge_cloud(
    model_path: str,
    threshold: float,
    cloud_url: str,
    dataset_dir: str,
    output_dir: str,
) -> None:
    """Run the edge-cloud experiment scenario."""
    output_path = os.path.join(output_dir, "edge_cloud.csv")
    logger = ExperimentLogger(output_path, "edge_cloud")

    print(f"[run_edge_cloud] Loading model: {model_path}")
    inferencer = EdgeInference(model_path=model_path)
    offload_mgr = OffloadManager(threshold=threshold, cloud_url=cloud_url)

    images = collect_images(dataset_dir)
    print(
        f"[run_edge_cloud] Processing {len(images)} images "
        f"(threshold={threshold}, cloud={cloud_url})..."
    )

    for image_path, difficulty in images:
        image_name = os.path.join(difficulty, os.path.basename(image_path))
        timestamp = datetime.now().isoformat()

        try:
            edge_result = inferencer.infer(image_path)
        except ValueError as exc:
            print(f"[run_edge_cloud] Skipping {image_name}: {exc}", file=sys.stderr)
            continue

        decision = offload_mgr.decide(edge_result, image_path)
        detected_classes = list({obj.class_name for obj in decision.final_objects})

        record = LogRecord(
            image_name=image_name,
            latency_ms=decision.total_latency_ms,
            uploaded_bytes=decision.uploaded_bytes,
            processing_place=decision.processing_place,
            confidence=decision.final_confidence,
            object_count=decision.final_object_count,
            offloaded=decision.offloaded,
            detected_classes=json.dumps(detected_classes),
            cpu_percent=0.0,
            ram_mb=0.0,
            difficulty=difficulty,
            scenario="edge_cloud",
            timestamp=timestamp,
        )
        logger.log_image_result(record)

    logger.save()
    summary = logger.compute_summary()

    print("\n=== Edge-Cloud Summary ===")
    print(f"  Total images    : {summary.total_images}")
    print(f"  Avg latency     : {summary.avg_latency_ms:.2f} ms")
    print(f"  Total upload    : {summary.total_uploaded_bytes:,} bytes")
    print(f"  Throughput      : {summary.throughput_img_per_sec:.4f} img/s")
    print(f"  Avg confidence  : {summary.avg_confidence:.4f}")
    print(f"  Offload ratio   : {summary.offload_ratio:.2f}%")
    print(f"  Avg CPU         : {summary.avg_cpu_percent:.2f}%")
    print(f"  Avg RAM         : {summary.avg_ram_mb:.2f} MB")
    print(f"  Results saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run edge-cloud experiment scenario")
    parser.add_argument(
        "--model",
        default=os.environ.get("EDGE_MODEL", "yolov8n.pt"),
        help="YOLOv8 edge model path (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6")),
        help="Confidence threshold for offloading (default: 0.6)",
    )
    parser.add_argument(
        "--cloud-url",
        default=os.environ.get("CLOUD_URL", "http://localhost:8000"),
        help="Cloud Node base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("DATASET_DIR", "dataset"),
        help="Path to dataset directory (default: dataset/)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "results"),
        help="Path to output directory (default: results/)",
    )
    args = parser.parse_args()
    run_edge_cloud(args.model, args.threshold, args.cloud_url, args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
