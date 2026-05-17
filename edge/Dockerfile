FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install torch first (heaviest, cached separately)
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights so startup is fast
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy application code
COPY models.py .
COPY edge_infer.py .
COPY edge_api.py .
COPY offload_manager.py .
COPY logger.py .
COPY run_cloud_only.py .
COPY run_edge_only.py .
COPY run_edge_cloud.py .
COPY metrics.py .

ENV SCENARIO=edge_cloud
ENV CLOUD_URL=http://cloud_node:8000
ENV CONFIDENCE_THRESHOLD=0.6
ENV DATASET_DIR=/app/dataset
ENV OUTPUT_DIR=/app/results

EXPOSE 8001

# Default: run edge API server. Override SCENARIO env var to run experiment scripts.
CMD ["sh", "-c", "uvicorn edge_api:app --host 0.0.0.0 --port 8001"]
