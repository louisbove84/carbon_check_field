# Configuration Management Guide

## Overview

All configuration is centralized in a **YAML file stored on Cloud Storage**. This allows you to change configuration without modifying code or redeploying functions!

## How It Works (Recommended: Cloud Storage YAML)

```
config.yaml (single source of truth)
    ↓ upload via ./upload_config.sh
Cloud Storage (gs://carboncheck-data/config/config.yaml)
    ↓ loaded at runtime by
config.py
    ↓ imports from
All Python files
```

**Priority:** 
1. **Cloud Storage YAML** (primary, recommended)
2. Environment Variables (fallback)
3. Hardcoded defaults (last resort)

---

## Quick Start: Using YAML Configuration (Recommended)

### 1. Edit Configuration

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline

# Edit config.yaml
nano config.yaml  # or open in your editor
```

### 2. Upload to Cloud Storage

```bash
./upload_config.sh
```

**That's it!** Changes take effect on next function execution. No redeployment needed!

### 3. Verify

```bash
# View current config on GCP
gsutil cat gs://carboncheck-data/config/config.yaml

# Test immediately
./test_pipeline.sh retrain
```

---

## Why YAML on Cloud Storage?

### ✅ Benefits Over Environment Variables

| Feature | YAML on GCS | Environment Variables |
|---------|-------------|----------------------|
| **Update without redeployment** | ✅ Yes | ❌ No |
| **Single source of truth** | ✅ Yes | ⚠️ Per-function |
| **Version control** | ✅ Yes (GCS versioning) | ❌ No |
| **Easy rollback** | ✅ Yes (backups) | ❌ No |
| **View/edit easily** | ✅ Yes (YAML file) | ⚠️ CLI only |
| **Change quality gates instantly** | ✅ Yes | ❌ Requires redeploy |

**Result:** Change quality gates at 11 PM without redeploying! 🚀

---

## Configuration Files

### `config.yaml` - Main Configuration File

Single source of truth for all settings:

```yaml
quality_gates:
  absolute_min_accuracy: 0.75  # 75%
  min_per_crop_f1: 0.70         # 70%
  improvement_margin: 0.02       # 2%

data_collection:
  samples_per_crop: 100

model:
  hyperparameters:
    n_estimators: 100
    max_depth: 10
```

### `config.py` - Configuration Loader

Loads from Cloud Storage, then falls back to env vars:

```python
from config import PROJECT_ID, REGION, BUCKET_NAME, etc.
```

**You don't need to touch this file!** It automatically loads from Cloud Storage.

---

## Available Configuration Variables

### Google Cloud Project
| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ID` | `ml-pipeline-477612` | GCP project ID |
| `REGION` | `us-central1` | GCP region |
| `BUCKET_NAME` | `carboncheck-data` | GCS bucket name |
| `DATASET_ID` | `crop_ml` | BigQuery dataset |

### Quality Gates (Most commonly adjusted!)
| Variable | Default | Description |
|----------|---------|-------------|
| `ABSOLUTE_MIN_ACCURACY` | `0.75` | Minimum accuracy (75%) |
| `MIN_PER_CROP_F1` | `0.70` | Min F1 per crop (70%) |
| `IMPROVEMENT_MARGIN` | `0.02` | Required improvement (2%) |

### Data Collection
| Variable | Default | Description |
|----------|---------|-------------|
| `SAMPLES_PER_CROP` | `100` | Samples collected per crop monthly |

### Model Training
| Variable | Default | Description |
|----------|---------|-------------|
| `N_ESTIMATORS` | `100` | RandomForest estimators |
| `MAX_DEPTH` | `10` | Max tree depth |
| `MIN_SAMPLES_SPLIT` | `5` | Min samples for split |

---

## Deployment with Configuration

### Option 1: Use the Deployment Script (Recommended)

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline

# Edit deploy_with_config.sh to change values
# Then deploy:
./deploy_with_config.sh
```

### Option 2: Manual Deployment with Environment Variables

```bash
gcloud functions deploy auto-retrain-model \
  --gen2 --runtime=python311 --region=us-central1 \
  --source=. --entry-point=retrain_model \
  --trigger-http --allow-unauthenticated \
  --timeout=3600s --memory=4Gi \
  --set-env-vars "PROJECT_ID=ml-pipeline-477612,REGION=us-central1,BUCKET_NAME=carboncheck-data,ABSOLUTE_MIN_ACCURACY=0.75,MIN_PER_CROP_F1=0.70,IMPROVEMENT_MARGIN=0.02"
```

---

## Updating Configuration (Without Redeploying Code!)

### Method 1: YAML File (Recommended)

**Example: Make gates more strict (require 80% accuracy, 3% improvement)**

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline

# Edit config.yaml
nano config.yaml

# Change:
quality_gates:
  absolute_min_accuracy: 0.80  # Was 0.75
  improvement_margin: 0.03      # Was 0.02

# Upload
./upload_config.sh
```

**Done!** Next function execution uses new config. No redeployment!

### Method 2: Environment Variables (Legacy)

**Example: Same change via environment variables**

```bash
gcloud functions deploy auto-retrain-model \
  --region=us-central1 \
  --gen2 \
  --update-env-vars ABSOLUTE_MIN_ACCURACY=0.80,IMPROVEMENT_MARGIN=0.03
```

**Note:** This requires redeployment and only updates one function at a time.

### Update Data Collection Rate (YAML)

**Example: Collect 200 samples per crop instead of 100**

```bash
# Edit config.yaml
nano config.yaml

# Change:
data_collection:
  samples_per_crop: 200  # Was 100

# Upload
./upload_config.sh
```

### Update Multiple Variables at Once

```bash
gcloud functions deploy auto-retrain-model \
  --region=us-central1 \
  --gen2 \
  --update-env-vars \
    ABSOLUTE_MIN_ACCURACY=0.80,\
    MIN_PER_CROP_F1=0.75,\
    IMPROVEMENT_MARGIN=0.03,\
    N_ESTIMATORS=200
```

---

## View Current Configuration

### View YAML on Cloud Storage

```bash
# View current production config
gsutil cat gs://carboncheck-data/config/config.yaml

# View specific section
gsutil cat gs://carboncheck-data/config/config.yaml | grep -A 5 "quality_gates"
```

### View Backups

```bash
# List all backups
gsutil ls gs://carboncheck-data/config/backups/

# View specific backup
gsutil cat gs://carboncheck-data/config/backups/config_20251126_143000.yaml
```

### Test Configuration Locally

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline

# Print current config (loads from GCS if available)
python3 config.py
```

### Rollback to Previous Version

```bash
# List backups
gsutil ls gs://carboncheck-data/config/backups/

# Rollback to specific backup
gsutil cp gs://carboncheck-data/config/backups/config_20251126_143000.yaml \
  gs://carboncheck-data/config/config.yaml

# Verify
gsutil cat gs://carboncheck-data/config/config.yaml | head -20
```

**Output:**
```
============================================================
CURRENT CONFIGURATION
============================================================
Project ID:          ml-pipeline-477612
Region:              us-central1
Bucket:              carboncheck-data
Dataset:             crop_ml
Model Name:          crop-classifier-latest
Endpoint Name:       crop-endpoint

Quality Gates:
  Min Accuracy:      75%
  Min Crop F1:       70%
  Improvement:       2%

Data Collection:
  Samples/Crop:      100
============================================================
```

---

## Common Configuration Changes

### Make Quality Gates More Strict

**When:** After collecting 6+ months of data and model is consistently passing gates

```bash
gcloud functions deploy auto-retrain-model \
  --region=us-central1 --gen2 \
  --update-env-vars \
    ABSOLUTE_MIN_ACCURACY=0.80,\
    MIN_PER_CROP_F1=0.75,\
    IMPROVEMENT_MARGIN=0.03
```

### Make Quality Gates More Lenient

**When:** Model keeps getting blocked due to strict gates in early months

```bash
gcloud functions deploy auto-retrain-model \
  --region=us-central1 --gen2 \
  --update-env-vars \
    ABSOLUTE_MIN_ACCURACY=0.70,\
    MIN_PER_CROP_F1=0.65,\
    IMPROVEMENT_MARGIN=0.01
```

### Increase Data Collection

**When:** Want more training data per month

```bash
gcloud functions deploy monthly-data-collection \
  --region=us-central1 --gen2 \
  --update-env-vars SAMPLES_PER_CROP=200
```

### Tune RandomForest Hyperparameters

**When:** Want to experiment with model architecture

```bash
gcloud functions deploy auto-retrain-model \
  --region=us-central1 --gen2 \
  --update-env-vars \
    N_ESTIMATORS=200,\
    MAX_DEPTH=15,\
    MIN_SAMPLES_SPLIT=10
```

---

## Environment-Specific Configuration

### Development Environment

```bash
# Deploy with lower gates for testing
gcloud functions deploy auto-retrain-model \
  --region=us-central1 --gen2 \
  --set-env-vars \
    PROJECT_ID=ml-pipeline-dev-123456,\
    ABSOLUTE_MIN_ACCURACY=0.60,\
    MIN_PER_CROP_F1=0.55,\
    IMPROVEMENT_MARGIN=0.01
```

### Production Environment

```bash
# Deploy with strict gates
gcloud functions deploy auto-retrain-model \
  --region=us-central1 --gen2 \
  --set-env-vars \
    PROJECT_ID=ml-pipeline-477612,\
    ABSOLUTE_MIN_ACCURACY=0.80,\
    MIN_PER_CROP_F1=0.75,\
    IMPROVEMENT_MARGIN=0.03
```

---

## Configuration Best Practices

### 1. Always Test Configuration Changes

```bash
# After updating config, test immediately:
./test_pipeline.sh retrain
```

### 2. Document Configuration Changes

Keep track of quality gate changes in BigQuery:

```sql
-- View recent deployment decisions to see if gates are too strict/lenient
SELECT 
  deployment_time,
  deployment_decision,
  accuracy,
  gates_failed
FROM `ml-pipeline-477612.crop_ml.deployment_history`
ORDER BY deployment_time DESC
LIMIT 10;
```

### 3. Gradual Gate Tightening

Don't jump from 70% to 90% accuracy requirement. Increase gradually:
- Month 1-3: 70% min accuracy
- Month 4-6: 75% min accuracy  
- Month 7+: 80% min accuracy

### 4. Seasonal Adjustments

Consider adjusting gates based on season:
- Growing season (Apr-Sep): Higher accuracy expected
- Off-season (Oct-Mar): Lower accuracy acceptable

---

## Troubleshooting

### Environment Variables Not Applied

**Check current values:**
```bash
gcloud functions describe auto-retrain-model \
  --region=us-central1 --gen2 \
  --format='value(serviceConfig.environmentVariables)'
```

**Force update:**
```bash
gcloud functions deploy auto-retrain-model \
  --region=us-central1 --gen2 \
  --update-env-vars KEY=VALUE
```

### Configuration Not Taking Effect

Cloud Functions cache environment variables. To force reload:

```bash
# Redeploy with same config (forces restart)
gcloud functions deploy auto-retrain-model \
  --region=us-central1 --gen2 \
  --update-env-vars ABSOLUTE_MIN_ACCURACY=0.75
```

### Wrong Configuration File Used

Make sure you're deploying from the correct directory:

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline
gcloud functions deploy auto-retrain-model --source=. ...
```

---

## Summary

### Centralized Configuration Benefits

✅ **Single Source of Truth:** `config.py` contains all configuration  
✅ **No Hardcoded Values:** All values read from environment variables  
✅ **Easy Updates:** Change config without code changes  
✅ **Environment-Specific:** Different configs for dev/prod  
✅ **Audit Trail:** Environment variables tracked in Cloud Console  

### Quick Commands

```bash
# View current config
python3 config.py

# Deploy with config
./deploy_with_config.sh

# Update quality gates
gcloud functions deploy auto-retrain-model \
  --region=us-central1 --gen2 \
  --update-env-vars ABSOLUTE_MIN_ACCURACY=0.80

# View deployed config
gcloud functions describe auto-retrain-model \
  --region=us-central1 --gen2 \
  --format='table(serviceConfig.environmentVariables)'
```

**Result:** You can now change configuration (especially quality gates) without touching code! 🎉

