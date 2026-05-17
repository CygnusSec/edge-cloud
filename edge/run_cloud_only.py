"""
Cloud-Only scenario: send all images to the Cloud Node for processing.

Baseline for comparison against edge-only and edge-cloud scenarios.
Results are saved to results/cloud_only.csv.

Usage:
    python run_cloud_only.py [--cloud-url URL] [--dataset-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import requests

from models import LogRecord
from logger import ExperimentLogger

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(dataset_dir: str) -> list[tuple[str, str]]:
    """
    Collect all images from dataset/images/easy, medium, hard.
    Returns list of (image_path, difficulty) tuples sorted alphabetically.
    """
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
            f"[run_cloud_only] ERROR: No images found in {dataset_dir}/images/{{easy,medium,hard}}",
            file=sys.stderr,
        )
        sys.exit(1)

    return images


def run_cloud_only(cloud_url: str, dataset_dir: str, output_dir: str) -> None:
    """Run the cloud-only experiment scenario."""
    output_path = os.path.join(output_dir, "cloud_only.csv")
    logger = ExperimentLogger(output_path, "cloud_only")

    images = collect_images(dataset_dir)
    print(f"[run_cloud_only] Processing {len(images)} images via cloud-only...")

    predict_url = f"{cloud_url.rstrip('/')}/predict"

    for image_path, difficulty in images:
        image_name = os.path.join(difficulty, os.path.basename(image_path))
        timestamp = datetime.now().isoformat()

        t_start = time.perf_counter()
        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    predict_url,
                    files={"file": (os.path.basename(image_path), f, "image/jpeg")},
                    timeout=30,
                )
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            if response.status_code == 200:
                data = response.json()
                confidence = data.get("confidence", 0.0)
                object_count = data.get("object_count", 0)
                detected_classes = json.dumps(data.get("detected_classes", []))
                uploaded_bytes = os.path.getsize(image_path)
            else:
                print(
                    f"[run_cloud_only] HTTP {response.status_code} for {image_name}",
                    file=sys.stderr,
                )
                continue

        except (requests.ConnectionError, requests.Timeout) as exc:
            print(
                f"[run_cloud_only] Connection error for {image_name}: {exc}",
                file=sys.stderr,
            )
            continue

        record = LogRecord(
            image_name=image_name,
            latency_ms=latency_ms,
            uploaded_bytes=uploaded_bytes,
            processing_place="cloud",
            confidence=confidence,
            object_count=object_count,
            offloaded=False,
            detected_classes=detected_classes,
            cpu_percent=0.0,
            ram_mb=0.0,
            difficulty=difficulty,
            scenario="cloud_only",
            timestamp=timestamp,
        )
        logger.log_image_result(record)

    logger.save()
    summary = logger.compute_summary()

    print("\n=== Cloud-Only Summary ===")
    print(f"  Total images    : {summary.total_images}")
    print(f"  Avg latency     : {summary.avg_latency_ms:.2f} ms")
    print(f"  Total upload    : {summary.total_uploaded_bytes:,} bytes")
    print(f"  Throughput      : {summary.throughput_img_per_sec:.4f} img/s")
    print(f"  Avg confidence  : {summary.avg_confidence:.4f}")
    print(f"  Avg CPU         : {summary.avg_cpu_percent:.2f}%")
    print(f"  Avg RAM         : {summary.avg_ram_mb:.2f} MB")
    print(f"  Results saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cloud-only experiment scenario")
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
    run_cloud_only(args.cloud_url, args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
