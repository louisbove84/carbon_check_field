# Skew Audit Container
# ====================
# Container for monthly data skew detection and drift monitoring.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpng-dev \
    libfreetype6-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY skew_job/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules directory (from parent ml_pipeline/)
COPY shared /app/shared

# Copy skew job application files
COPY skew_job/main.py .
COPY skew_job/skew_detector.py .

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run Flask app with gunicorn
# Timeout set high for EE data collection
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 3600 main:app
