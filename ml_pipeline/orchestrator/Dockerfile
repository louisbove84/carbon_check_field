# Orchestrator Container (Lightweight)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies (no ML libraries - keeps image small!)
COPY orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules directory first (from parent ml_pipeline/)
COPY shared /app/shared

# Copy orchestrator application files
COPY orchestrator/main.py .
COPY orchestrator/orchestrator.py .
COPY orchestrator/earth_engine_collector.py .
COPY orchestrator/config.yaml .

# Copy trainer code so orchestrator can upload it to GCS at runtime
COPY trainer /app/trainer

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run Flask app
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 3600 main:app

