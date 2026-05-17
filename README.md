# Edge-Cloud Computing for Multimedia Transmission and Large-Scale Image Semantic Analysis

A demonstration system for edge-cloud computing architecture applied to object detection in images and video. The system proves three core research claims by comparing three experiment scenarios: **cloud-only**, **edge-only**, and **edge-cloud**.

## Research Objectives

1. **Reduce bandwidth** — edge-cloud sends only difficult images to the cloud, significantly less than cloud-only
2. **Reduce latency** — simple images processed at the edge avoid the cloud round-trip
3. **Improve quality** — difficult images are offloaded to a stronger cloud model for better detection accuracy

## System Architecture

```
Dataset (easy / medium / hard)
        │
        ▼
  Edge Node  ──── YOLOv8n/s (lightweight, CPU)
        │
        │  confidence < 0.6 → offload
        ▼
  Cloud Node ──── YOLOv8m (stronger, CPU) via FastAPI POST /predict
        │
        ▼
  Logger ──── CSV results (latency, bandwidth, confidence, CPU/RAM)
        │
        ▼
  Dashboard ── Streamlit (charts, comparison table, bounding boxes)
```

## Project Structure

```
edge-cloud/
├── edge/
│   ├── models.py             # Shared dataclasses (DetectedObject, LogRecord, ...)
│   ├── edge_infer.py         # EdgeInference class — YOLOv8n/s + OpenCV
│   ├── offload_manager.py    # OffloadManager — confidence threshold decision
│   ├── logger.py             # ExperimentLogger — Pandas CSV + psutil metrics
│   ├── run_cloud_only.py     # Scenario 1: all images sent to cloud
│   ├── run_edge_only.py      # Scenario 2: all images processed at edge
│   ├── run_edge_cloud.py     # Scenario 3: adaptive offloading
│   ├── metrics.py            # Compute summary metrics and comparison table
│   ├── Dockerfile
│   └── requirements.txt
├── cloud/
│   ├── cloud_api.py          # FastAPI app — POST /predict, GET /health
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/
│   ├── app.py                # Streamlit dashboard — 3 pages, 4 charts
│   ├── Dockerfile
│   └── requirements.txt
├── dataset/
│   └── images/
│       ├── easy/             # Clear images, few objects
│       ├── medium/           # Multiple objects
│       └── hard/             # Blurry, dark, occluded objects
├── results/                  # Output CSV files
├── tests/
│   ├── unit/
│   ├── property/
│   └── integration/
├── docker-compose.yml
└── README.md
```

## Quick Start

### Option 1: Docker Compose (recommended)

```bash
# 1. Add images to dataset/images/easy/, medium/, hard/
#    (JPEG/PNG, at least 10 images per folder)

# 2. Start all services
docker-compose up --build

# 3. Run experiment scenarios (in a new terminal)
docker-compose exec edge_node python run_cloud_only.py
docker-compose exec edge_node python run_edge_only.py
docker-compose exec edge_node python run_edge_cloud.py

# 4. Compute summary metrics
docker-compose exec edge_node python metrics.py

# 5. Open dashboard
# http://localhost:8501
```

### Option 2: Run locally (without Docker)

```bash
# Install dependencies
pip install -r edge/requirements.txt
pip install -r cloud/requirements.txt
pip install -r dashboard/requirements.txt

# Terminal 1 — start Cloud Node
cd cloud
uvicorn cloud_api:app --host 0.0.0.0 --port 8000

# Terminal 2 — run experiment scenarios
cd edge
python run_cloud_only.py --cloud-url http://localhost:8000
python run_edge_only.py
python run_edge_cloud.py --threshold 0.6 --cloud-url http://localhost:8000
python metrics.py

# Terminal 3 — start Dashboard
cd dashboard
streamlit run app.py
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `--threshold` | `0.6` | Confidence threshold for offloading to cloud |
| `--model` | `yolov8n.pt` | Edge model (`yolov8n.pt` or `yolov8s.pt`) |
| `--cloud-url` | `http://localhost:8000` | Cloud Node base URL |
| `--dataset-dir` | `dataset/` | Dataset root directory |
| `--output-dir` | `results/` | Output directory for CSV files |

Environment variables (used in Docker): `CLOUD_URL`, `CONFIDENCE_THRESHOLD`, `SCENARIO`, `DATASET_DIR`, `OUTPUT_DIR`.

## Experiment Scenarios

### Scenario 1 — Cloud-Only (baseline)
All images are sent to the cloud regardless of content. Measures maximum bandwidth usage and cloud-dependent latency.

```bash
python edge/run_cloud_only.py
# Output: results/cloud_only.csv
```

### Scenario 2 — Edge-Only
All images are processed locally at the edge. Measures minimum bandwidth (zero upload) and edge-only inference quality.

```bash
python edge/run_edge_only.py
# Output: results/edge_only.csv
```

### Scenario 3 — Edge-Cloud (adaptive offloading)
Edge processes each image first. If `max_confidence >= threshold`, the edge result is kept. Otherwise the image is offloaded to the cloud.

```bash
python edge/run_edge_cloud.py --threshold 0.6
# Output: results/edge_cloud.csv
```

## Output Files

| File | Description |
|---|---|
| `results/cloud_only.csv` | Per-image results for cloud-only scenario |
| `results/edge_only.csv` | Per-image results for edge-only scenario |
| `results/edge_cloud.csv` | Per-image results for edge-cloud scenario |
| `results/metrics_summary.csv` | Aggregated comparison across all scenarios |

### CSV Columns

| Column | Type | Description |
|---|---|---|
| `image_name` | str | Relative image path (e.g. `easy/img001.jpg`) |
| `latency_ms` | float | Total processing time (ms) |
| `uploaded_bytes` | int | Bytes sent to cloud (0 for edge-only) |
| `processing_place` | str | `"edge"` or `"cloud"` |
| `confidence` | float | Highest detection confidence (0.0–1.0) |
| `object_count` | int | Number of detected objects |
| `offloaded` | bool | Whether image was offloaded to cloud |
| `detected_classes` | str | JSON list of detected class names |
| `cpu_percent` | float | CPU usage at processing time (%) |
| `ram_mb` | float | RAM usage at processing time (MB) |
| `difficulty` | str | `"easy"`, `"medium"`, or `"hard"` |
| `timestamp` | str | ISO 8601 timestamp |

## Evaluation Metrics

- **Average latency (ms)** — mean processing time per image
- **Total upload bandwidth (bytes)** — total data sent to cloud
- **Offload ratio (%)** — percentage of images offloaded in edge-cloud scenario
- **Throughput (img/s)** — images processed per second
- **Average confidence** — mean detection confidence score
- **CPU / RAM usage** — resource consumption per scenario

### Key Formulas

```
Offload ratio     = N_cloud / N_total × 100%
Bandwidth savings = (1 - B_edge_cloud / B_cloud_only) × 100%
Throughput        = N_images / total_duration_sec
```

## Cloud Node API

**`POST /predict`** — accepts a JPEG/PNG image, returns detection results.

```json
{
  "objects": [
    {"class_name": "car", "confidence": 0.87, "bbox": {"x1": 100, "y1": 80, "x2": 220, "y2": 200}}
  ],
  "confidence": 0.87,
  "object_count": 1,
  "latency_ms": 145.3,
  "detected_classes": ["car"]
}
```

**`GET /health`** — Docker health check endpoint.

```json
{"status": "healthy", "model": "yolov8m", "model_loaded": true}
```

## Dashboard

Open `http://localhost:8501` after starting the dashboard service.

| Page | Description |
|---|---|
| **Single Image Analysis** | Upload an image, choose a scenario, view bounding boxes and metrics |
| **Scenario Comparison** | Summary table + 4 charts (latency, bandwidth, offload ratio, confidence) |
| **Video Analysis** | Upload a video, process frames using edge-cloud mechanism |

## Models

| Component | Model | Size | Purpose |
|---|---|---|---|
| Edge Node | YOLOv8n | ~6 MB | Fast lightweight inference |
| Cloud Node | YOLOv8m | ~50 MB | Higher accuracy for difficult images |

Models run on **CPU only** — no GPU required. YOLOv8 weights are downloaded automatically by Ultralytics on first run.

## Tech Stack

| Component | Technology |
|---|---|
| Object detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Image processing | OpenCV |
| Cloud API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Data logging | Pandas CSV |
| System metrics | psutil |
| Containerization | Docker + Docker Compose |
| Language | Python 3.10+ |

## Notes

- Dataset folders (`easy/`, `medium/`, `hard/`) must contain at least 10 images each for statistically meaningful results.
- The confidence threshold (default `0.6`) directly controls the offload ratio — lower threshold → more offloading → higher quality but more bandwidth.
- All three scenarios must be run on the **same dataset** for a fair comparison.
- The `results/` directory is mounted as a shared volume between the edge container and the dashboard container.
