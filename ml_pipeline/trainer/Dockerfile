# Vertex AI Training Container (ML-Optimized) - BASE IMAGE ONLY
# Code is downloaded at runtime, so no rebuild needed for code changes!
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# Note: matplotlib requires these libraries for image rendering (critical for TensorBoard)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpng-dev \
    libfreetype6-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies ONLY
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Vertex AI will set AIP_MODEL_DIR environment variable
ENV PYTHONUNBUFFERED=1

# Copy entrypoint script that downloads code at runtime
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Entrypoint downloads code from GCS staging bucket, then runs training
ENTRYPOINT ["/app/entrypoint.sh"]

