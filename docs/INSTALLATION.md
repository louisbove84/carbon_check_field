# Installation Guide

Complete guide for setting up the CarbonCheck Field development environment.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Python Setup](#python-setup)
4. [Troubleshooting](#troubleshooting)

---

## Overview

### GCP vs Local Development

**For GCP Deployment**: Each service uses its own `requirements.txt`:
- `backend/requirements.txt` → Backend Dockerfile
- `ml_pipeline/orchestrator/requirements.txt` → Orchestrator Dockerfile  
- `ml_pipeline/trainer/requirements.txt` → Trainer Dockerfile

**For Local Development**: Use the consolidated files below.

See [DEPLOYMENT.md](../DEPLOYMENT.md) for GCP deployment details.

---

## Quick Start

### Option 1: Using the Setup Script (Recommended)

```bash
cd carbon_check_field
./setup_venv.sh
```

This will:
- Create a virtual environment with Python 3.13.9
- Install all dependencies from `requirements-all.txt`

### Option 2: Using pyproject.toml (Modern Python)

```bash
cd carbon_check_field

# Create venv if needed
/opt/homebrew/bin/python3.13 -m venv venv
source venv/bin/activate

# Install with all optional dependencies
pip install -e ".[all]"
```

### Option 3: Manual Installation

```bash
cd carbon_check_field

# Create venv
/opt/homebrew/bin/python3.13 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements-all.txt
```

---

## Python Setup

### Python 3.13 Requirements

Python 3.13 requires:
1. **GEOS library** for `shapely` (system dependency)
2. **Newer pydantic** version (2.5.3 is too old for Python 3.13)

### Step 1: Install GEOS (required for shapely)

```bash
brew install geos
```

### Step 2: Install Python dependencies

The `requirements-all.txt` has been updated with Python 3.13-compatible versions:

```bash
cd carbon_check_field
source venv/bin/activate
pip install -r requirements-all.txt
```

### Alternative: Use Python 3.11 (matches Dockerfiles)

If you prefer to match the Dockerfiles exactly (which use Python 3.11):

```bash
# Install Python 3.11 via Homebrew
brew install python@3.11

# Create venv with Python 3.11
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements-all.txt
```

**Note:** The Dockerfiles use Python 3.11, so using Python 3.11 locally ensures exact compatibility with production.

---

## What Gets Installed

The installation includes:

### Core Dependencies
- `numpy`, `pandas` - Data processing
- `PyYAML` - Configuration management
- Google Cloud SDKs (BigQuery, Storage, AI Platform, Earth Engine)

### Backend API (`[backend]`)
- `fastapi`, `uvicorn` - Web framework
- `pydantic` - Data validation
- `shapely` - Geometry processing

### ML Training (`[trainer]`)
- `scikit-learn` - Machine learning
- `joblib` - Model serialization
- `matplotlib`, `seaborn` - Visualization
- `tensorboard`, `torch` - Training monitoring

### Orchestrator (`[orchestrator]`)
- `flask`, `gunicorn` - Lightweight web server

---

## File Structure

- **`pyproject.toml`** - Modern Python project configuration with optional dependencies
- **`requirements-all.txt`** - Consolidated requirements file (all dependencies)
- **`backend/requirements.txt`** - Backend-specific dependencies
- **`ml_pipeline/trainer/requirements.txt`** - ML training dependencies
- **`ml_pipeline/orchestrator/requirements.txt`** - Orchestrator dependencies

---

## Using the Virtual Environment

### Activate

```bash
source venv/bin/activate
```

### Run Training

```bash
# Option 1: Use the helper script
./run_training.sh --sample-size 300

# Option 2: Activate venv manually
source venv/bin/activate
python ml_pipeline/tools/test_training.py --sample-size 300
```

### Deactivate

```bash
deactivate
```

### Development Mode

For development, install with dev dependencies:

```bash
pip install -e ".[all,dev]"
```

---

## Troubleshooting

### SSL Certificate Errors

If you see SSL certificate errors, try:

```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements-all.txt
```

### Python Crashes

The virtual environment uses Python 3.13.9 from Homebrew, which should be stable. If you still experience crashes:

1. Make sure you're using the venv Python: `which python` should show `venv/bin/python`
2. Try reinstalling: `rm -rf venv && ./setup_venv.sh`

### Missing Dependencies

If a specific module is missing, install it:

```bash
source venv/bin/activate
pip install <package-name>
```

### Viewing Results

After training completes, view TensorBoard:

```bash
source venv/bin/activate
tensorboard --logdir test_output/tensorboard_logs
```

The confusion matrix will show all classes including 'Other' if it's in your BigQuery data.

---

## Next Steps

- See [DEPLOYMENT.md](../DEPLOYMENT.md) for GCP deployment structure
- See [README.md](../README.md) for project overview
- See [docs/IOS_DEVELOPMENT.md](./IOS_DEVELOPMENT.md) for iOS development
