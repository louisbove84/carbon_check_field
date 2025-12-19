# ML Pipeline Tools

This directory contains testing, evaluation, and utility tools for the ML pipeline.

## Tools Overview

### Test Training

**`test_training.py`** - Run a test training job locally with a small dataset

This script mimics the production training process (orchestrator training) but:
- Uses a small sample of BigQuery data (default: 500 samples)
- Trains a smaller model (fewer estimators, shallower trees)
- Runs locally without Vertex AI
- Produces TensorBoard logs and model outputs

**Usage:**
```bash
# Basic test training
python tools/test_training.py

# Custom sample size
python tools/test_training.py --sample-size 200

# Upload TensorBoard logs to GCS
python tools/test_training.py --upload-tensorboard

# Upload model to GCS
python tools/test_training.py --upload-model

# Full options
python tools/test_training.py \
    --sample-size 500 \
    --min-samples-per-crop 20 \
    --output-dir test_output \
    --upload-tensorboard \
    --upload-model
```

**Output:**
- Model saved to `test_output/model/`
- TensorBoard logs saved to `test_output/tensorboard_logs/`
- View TensorBoard: `tensorboard --logdir test_output/tensorboard_logs/`

### Model Evaluation

**`evaluate_model.py`** - Evaluate an existing trained model

Loads a pre-trained model (from GCS or local) and runs comprehensive evaluation
without retraining. Useful for evaluating production models, comparing versions,
or debugging performance.

**Usage:**
```bash
# Evaluate latest model from GCS
python tools/evaluate_model.py

# Evaluate specific archived model
python tools/evaluate_model.py --model-path models/crop_classifier_archive/crop_classifier_20251205_2255

# Evaluate local model
python tools/evaluate_model.py --model-path ./local_model --local

# Use custom test data
python tools/evaluate_model.py --model-path models/crop_classifier_latest --test-data ./test_data.csv
```

**Output:**
- TensorBoard logs saved to `evaluation_output/tensorboard_logs/`
- View TensorBoard: `tensorboard --logdir evaluation_output/tensorboard_logs/`

### Verification

**`verify_tensorboard.py`** - TensorBoard verification and diagnostics

Helps diagnose TensorBoard issues by:
- Inspecting event files for images, scalars, and other data
- Checking TensorBoard instance configuration
- Verifying GCS paths and file structure

**Usage:**
```bash
# Verify latest training run
python tools/verify_tensorboard.py

# Check specific run
python tools/verify_tensorboard.py --run run_20251215_202351

# Check TensorBoard instance
python tools/verify_tensorboard.py --check-instance

# List available runs
python tools/verify_tensorboard.py --list-runs
```

### Endpoint Testing

**`test_endpoint.py`** - Test Vertex AI endpoint predictions

Tests the deployed crop classification model endpoint with sample data.

**Usage:**
```bash
python tools/test_endpoint.py
```

### Local Testing

**`test_tensorboard_local.py`** - Local TensorBoard testing

Creates test TensorBoard logs locally to verify image rendering works.

**Usage:**
```bash
python tools/test_tensorboard_local.py
```

### Utilities

**`view_tensorboard.sh`** - Launch TensorBoard viewer

Quick script to view TensorBoard logs from GCS locally.

**Usage:**
```bash
./tools/view_tensorboard.sh [run_directory]
```

**`setup_evaluation_tables.sql`** - BigQuery table setup

SQL script to create evaluation and tracking tables in BigQuery.

**Usage:**
```bash
bq query --use_legacy_sql=false < tools/setup_evaluation_tables.sql
```

## Requirements

All tools require:
- Python 3.9+
- Google Cloud SDK authentication
- Access to the GCP project
- Required Python packages (see `requirements.txt`)

## Configuration

All scripts automatically load configuration from:
- `orchestrator/config.yaml` (primary)
- Falls back to GCS: `gs://carboncheck-data/config/config.yaml`

## Common Workflows

### Test Training Pipeline Locally

**Use `test_training.py` to train a NEW model with a small dataset:**
```bash
# 1. Train a new model with small dataset
python tools/test_training.py --sample-size 500

# 2. View TensorBoard results
tensorboard --logdir test_output/tensorboard_logs/

# 3. Verify images are rendering correctly
python tools/verify_tensorboard.py --check-instance
```

### Evaluate Existing Models

**Use `evaluate_model.py` to evaluate an EXISTING trained model:**
```bash
# 1. Evaluate latest production model
python tools/evaluate_model.py

# 2. Compare two model versions
python tools/evaluate_model.py --model-path models/crop_classifier_archive/v1 --output-dir results_v1
python tools/evaluate_model.py --model-path models/crop_classifier_archive/v2 --output-dir results_v2

# 3. View results
tensorboard --logdir evaluation_output/tensorboard_logs/
```

### Debug TensorBoard Issues

```bash
# 1. Check TensorBoard instance
python tools/verify_tensorboard.py --check-instance

# 2. Verify latest run has images
python tools/verify_tensorboard.py

# 3. Test image rendering locally
python tools/test_tensorboard_local.py
```

