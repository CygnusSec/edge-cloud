FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

EXPOSE 8501

ENV RESULTS_DIR=/app/results
ENV DATASET_DIR=/app/dataset
ENV CLOUD_URL=http://cloud_node:8000
ENV CONFIDENCE_THRESHOLD=0.6

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.maxUploadSize=500"]
