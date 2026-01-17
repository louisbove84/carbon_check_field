# GCP Deployment Structure

## Overview

This project deploys to GCP with **separate services**, each with its own container and requirements:

1. **Backend API** → Cloud Run (FastAPI)
2. **ML Orchestrator** → Cloud Run (Flask, lightweight)
3. **ML Trainer** → Vertex AI (ML training job)

## Service-Specific Requirements

Each service has its own `requirements.txt` that matches its Dockerfile:

### Backend (`backend/requirements.txt`)
- Used by: `backend/Dockerfile`
- Deploys to: Cloud Run
- Includes: FastAPI, Earth Engine, Vertex AI client

### Orchestrator (`ml_pipeline/orchestrator/requirements.txt`)
- Used by: `ml_pipeline/orchestrator/Dockerfile`
- Deploys to: Cloud Run
- **Lightweight** - No ML libraries (keeps image small!)
- Includes: Flask, Google Cloud SDKs, Earth Engine

### Trainer (`ml_pipeline/trainer/requirements.txt`)
- Used by: `ml_pipeline/trainer/Dockerfile`
- Deploys to: Vertex AI
- Includes: scikit-learn, TensorBoard, matplotlib, all ML libraries

## Dockerfiles

Each Dockerfile installs **only** what that service needs:

```dockerfile
# Example: orchestrator/Dockerfile
COPY orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

This keeps container images small and deployment fast.

## Local Development

For **local development and testing**, use:

- **`requirements-all.txt`** - Consolidated file with all dependencies
- **`pyproject.toml`** - Modern Python project config (optional)

These are **NOT used in GCP deployments** - they're only for local setup.

** See [Quick Start Guide](docs/QUICK_START.md) for detailed local setup instructions.**

## Building and Deploying

### Orchestrator (Cloud Run)
```bash
cd ml_pipeline
gcloud builds submit --config cloudbuild.yaml
```

### Trainer (Vertex AI)
The trainer container is built separately and uploaded to Container Registry.

### Backend (Cloud Run)
Build and deploy using the backend Dockerfile.

## Why Separate Requirements?

1. **Smaller containers** - Each service only includes what it needs
2. **Faster deployments** - Less to download and install
3. **Clear separation** - Easy to see what each service depends on
4. **Security** - Minimal attack surface per service

## Updating Dependencies

When adding a new dependency:

1. **Identify which service needs it**
2. **Add to that service's `requirements.txt`**
3. **Rebuild that service's Docker image**
4. **For local dev**: Also add to `requirements-all.txt` (optional)

## Local Testing

To test locally with all dependencies:

```bash
# Install everything
pip install -r requirements-all.txt

# Or use pyproject.toml
pip install -e ".[all]"
```

But remember: **GCP uses the individual requirements.txt files**, not the consolidated one!

** See [Quick Start Guide](docs/QUICK_START.md) for complete setup instructions.**
