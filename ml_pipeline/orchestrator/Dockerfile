# Orchestrator Container (Lightweight)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies (no ML libraries - keeps image small!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY orchestrator.py .
COPY earth_engine_collector.py .
COPY config.yaml .

# Copy shared modules directory
COPY ../shared /app/shared

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run Flask app
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 3600 main:app

