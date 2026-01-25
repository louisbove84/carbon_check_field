# Vertex AI Skew Audit Container - BASE IMAGE ONLY
# =================================================
# Container for monthly data skew detection and drift monitoring.
# Code is downloaded at runtime from GCS, so no rebuild needed for code changes!

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib (critical for TensorBoard image rendering)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpng-dev \
    libfreetype6-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies ONLY
COPY skew_job/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules directory (from parent ml_pipeline/)
COPY shared /app/shared

# Vertex AI will set AIP_TENSORBOARD_LOG_DIR environment variable
ENV PYTHONUNBUFFERED=1

# Copy entrypoint script that downloads code at runtime
COPY skew_job/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Entrypoint downloads code from GCS staging bucket, then runs skew audit
ENTRYPOINT ["/app/entrypoint.sh"]
