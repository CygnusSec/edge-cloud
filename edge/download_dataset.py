"""
Download sample images for the edge-cloud demo dataset.

Uses publicly accessible image URLs and organizes them into:
  dataset/images/easy/    — clear images, few objects
  dataset/images/medium/  — multiple objects
  dataset/images/hard/    — complex/crowded scenes

Usage:
    python download_dataset.py [--dataset-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

# Public domain images from picsum.photos (Lorem Picsum) and other open sources
# These are reliable, no auth required
SAMPLE_IMAGES = {
    "easy": [
        ("https://picsum.photos/id/237/640/480", "easy_001.jpg"),
        ("https://picsum.photos/id/1025/640/480", "easy_002.jpg"),
        ("https://picsum.photos/id/1074/640/480", "easy_003.jpg"),
        ("https://picsum.photos/id/200/640/480", "easy_004.jpg"),
        ("https://picsum.photos/id/219/640/480", "easy_005.jpg"),
        ("https://picsum.photos/id/250/640/480", "easy_006.jpg"),
        ("https://picsum.photos/id/292/640/480", "easy_007.jpg"),
        ("https://picsum.photos/id/306/640/480", "easy_008.jpg"),
        ("https://picsum.photos/id/326/640/480", "easy_009.jpg"),
        ("https://picsum.photos/id/338/640/480", "easy_010.jpg"),
    ],
    "medium": [
        ("https://picsum.photos/id/400/640/480", "medium_001.jpg"),
        ("https://picsum.photos/id/401/640/480", "medium_002.jpg"),
        ("https://picsum.photos/id/402/640/480", "medium_003.jpg"),
        ("https://picsum.photos/id/403/640/480", "medium_004.jpg"),
        ("https://picsum.photos/id/404/640/480", "medium_005.jpg"),
        ("https://picsum.photos/id/405/640/480", "medium_006.jpg"),
        ("https://picsum.photos/id/406/640/480", "medium_007.jpg"),
        ("https://picsum.photos/id/407/640/480", "medium_008.jpg"),
        ("https://picsum.photos/id/408/640/480", "medium_009.jpg"),
        ("https://picsum.photos/id/409/640/480", "medium_010.jpg"),
    ],
    "hard": [
        ("https://picsum.photos/id/500/640/480", "hard_001.jpg"),
        ("https://picsum.photos/id/501/640/480", "hard_002.jpg"),
        ("https://picsum.photos/id/502/640/480", "hard_003.jpg"),
        ("https://picsum.photos/id/503/640/480", "hard_004.jpg"),
        ("https://picsum.photos/id/504/640/480", "hard_005.jpg"),
        ("https://picsum.photos/id/505/640/480", "hard_006.jpg"),
        ("https://picsum.photos/id/506/640/480", "hard_007.jpg"),
        ("https://picsum.photos/id/507/640/480", "hard_008.jpg"),
        ("https://picsum.photos/id/508/640/480", "hard_009.jpg"),
        ("https://picsum.photos/id/509/640/480", "hard_010.jpg"),
    ],
}


def download_images(dataset_dir: str) -> None:
    """Download sample images into dataset/images/easy|medium|hard/."""
    total = sum(len(v) for v in SAMPLE_IMAGES.values())
    downloaded = 0
    failed = 0

    for difficulty, images in SAMPLE_IMAGES.items():
        out_dir = os.path.join(dataset_dir, "images", difficulty)
        os.makedirs(out_dir, exist_ok=True)

        for url, filename in images:
            out_path = os.path.join(out_dir, filename)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                print(f"  [skip] {difficulty}/{filename} already exists")
                downloaded += 1
                continue

            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; edge-cloud-demo/1.0)"},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                with open(out_path, "wb") as f:
                    f.write(data)
                print(f"  [ok]   {difficulty}/{filename} ({len(data):,} bytes)")
                downloaded += 1
            except Exception as exc:
                print(f"  [fail] {difficulty}/{filename}: {exc}", file=sys.stderr)
                failed += 1

    print(f"\nDownload complete: {downloaded}/{total} images, {failed} failed")
    print(f"Dataset location: {os.path.abspath(dataset_dir)}/images/")

    if downloaded < total:
        print(
            "\nTip: You can also manually copy your own images into:\n"
            f"  {os.path.abspath(dataset_dir)}/images/easy/\n"
            f"  {os.path.abspath(dataset_dir)}/images/medium/\n"
            f"  {os.path.abspath(dataset_dir)}/images/hard/\n"
            "At least 10 images per folder recommended."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sample dataset images")
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("DATASET_DIR", "dataset"),
        help="Path to dataset directory (default: dataset/)",
    )
    args = parser.parse_args()

    print(f"Downloading sample images to: {args.dataset_dir}/images/")
    download_images(args.dataset_dir)


if __name__ == "__main__":
    main()
