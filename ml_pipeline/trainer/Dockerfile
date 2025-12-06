# Vertex AI Training Container (ML-Optimized)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training scripts and modules
COPY vertex_ai_training.py .
COPY feature_engineering.py .
COPY tensorboard_utils.py .

# Vertex AI will set AIP_MODEL_DIR environment variable
ENV PYTHONUNBUFFERED=1

# Run training script
ENTRYPOINT ["python", "vertex_ai_training.py"]

