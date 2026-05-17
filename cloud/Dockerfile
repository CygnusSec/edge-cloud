FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch first (heaviest, cached separately)
# CPU-only torch to keep image smaller (~800MB vs ~2GB for GPU)
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8m weights so startup is fast
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Copy application code
COPY cloud_api.py .

EXPOSE 8000

CMD ["uvicorn", "cloud_api:app", "--host", "0.0.0.0", "--port", "8000"]
