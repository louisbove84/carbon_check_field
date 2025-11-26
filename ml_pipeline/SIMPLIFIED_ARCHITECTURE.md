# Simplified ML Pipeline Architecture

## Overview

**Simple Cloud Run service with one entry point that orchestrates 3 clear steps.**

## Structure

```
ml_pipeline/
├── main.py                  # Orchestrator (calls the 3 steps)
├── collect_data.py          # Step 1: Earth Engine → BigQuery
├── retrain.py               # Step 2: Train model
├── evaluate.py              # Step 3: Evaluate & deploy if gates pass
├── config.yaml              # Configuration (loaded directly by each file)
├── deploy.sh                # Single deployment script
├── Dockerfile               # Cloud Run container
└── requirements.txt         # Python dependencies
```

## How It Works

### 1. Main Orchestrator (`main.py`)

```python
# Run full pipeline:
python main.py

# Run specific step:
python main.py --step=1  # Data collection only
python main.py --step=2  # Retraining only
python main.py --step=3  # Evaluation only
```

**What it does:**
- Imports collect_data, retrain, evaluate
- Calls them in sequence
- Logs results
- Provides HTTP endpoint for Cloud Run

### 2. Data Collection (`collect_data.py`)

```python
def collect_training_data():
    # Load config.yaml directly
    # Query Earth Engine for Sentinel-2 imagery
    # Extract NDVI features
    # Push to BigQuery
    return {'status': 'success', 'samples_collected': 400}
```

### 3. Model Retraining (`retrain.py`)

```python
def retrain_model():
    # Load config.yaml directly
    # Load data from BigQuery (excluding holdout)
    # Train RandomForest pipeline
    # Save to GCS
    return {'status': 'success', 'accuracy': 0.85}
```

### 4. Evaluation & Deployment (`evaluate.py`)

```python
def evaluate_and_deploy():
    # Load config.yaml directly
    # Load holdout test set
    # Evaluate champion vs challenger
    # Check quality gates
    # Deploy if gates pass
    return {'status': 'success', 'deployed': True}
```

## Configuration

**Single YAML file loaded by all scripts:**

```yaml
# config.yaml
quality_gates:
  absolute_min_accuracy: 0.75
  min_per_crop_f1: 0.70
  improvement_margin: 0.02

data_collection:
  samples_per_crop: 100

model:
  hyperparameters:
    n_estimators: 100
    max_depth: 10
```

**Update configuration:**
```bash
# 1. Edit config.yaml
nano config.yaml

# 2. Upload to Cloud Storage
./upload_config.sh

# 3. Next execution uses new config (no redeployment!)
```

## Deployment

**Single script deploys everything:**

```bash
./deploy.sh
```

**What it does:**
1. Builds Docker container
2. Pushes to Google Container Registry
3. Deploys to Cloud Run
4. Sets up Cloud Scheduler (optional)

## Usage

### Deploy to Cloud Run

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline
./deploy.sh
```

### Run Full Pipeline

```bash
# Via HTTP
curl -X POST https://ml-pipeline-xxxxx-uc.a.run.app

# Via Cloud Scheduler (automated monthly)
# Runs automatically on schedule
```

### Run Specific Step

```bash
# Data collection only
curl -X POST https://ml-pipeline-xxxxx-uc.a.run.app?step=1

# Retraining only  
curl -X POST https://ml-pipeline-xxxxx-uc.a.run.app?step=2

# Evaluation only
curl -X POST https://ml-pipeline-xxxxx-uc.a.run.app?step=3
```

### Local Testing

```bash
# Run full pipeline locally
python main.py --step=1  # Data collection
python main.py --step=2  # Retraining
python main.py --step=3  # Evaluation

# Or start Flask server
python main.py --port=8080
curl http://localhost:8080
```

## Benefits of Simplified Architecture

### ✅ Before (Complex)
- ❌ 2 separate Cloud Functions
- ❌ Multiple deployment scripts
- ❌ Cloud Scheduler to coordinate
- ❌ config.py wrapper module
- ❌ Confusing for newcomers

### ✅ After (Simple)
- ✅ **1 Cloud Run service**
- ✅ **1 deployment script** (`./deploy.sh`)
- ✅ **3 clear Python files** (collect, retrain, evaluate)
- ✅ **Direct config.yaml loading** (no wrapper)
- ✅ **Easy to understand** and modify

## File Details

### `main.py` - 150 lines
- Imports the 3 pipeline steps
- Calls them in sequence
- HTTP endpoint for Cloud Run
- Error handling and logging

### `collect_data.py` - 200 lines
- Loads config.yaml directly
- Earth Engine queries
- BigQuery insertion
- Returns simple dict

### `retrain.py` - 250 lines  
- Loads config.yaml directly
- BigQuery data loading
- Model training
- GCS model saving
- Returns simple dict

### `evaluate.py` - 200 lines
- Loads config.yaml directly
- Holdout set evaluation
- Champion vs challenger comparison
- Quality gate checking
- Vertex AI deployment
- Returns simple dict

### `deploy.sh` - 50 lines
- Build Docker image
- Push to Container Registry
- Deploy to Cloud Run
- Setup Cloud Scheduler

### `Dockerfile` - 15 lines
- Python 3.11 base image
- Install requirements
- Copy files
- Run main.py

## Cost

**Simplified Cloud Run:**
- Runs on-demand (pay per execution)
- ~$2-5/month for monthly runs
- Same functionality as before
- Much simpler to maintain

## Next Steps

1. ✅ Create `main.py` (orchestrator) - DONE
2. ✅ Create `collect_data.py` - DONE
3. ⏳ Create `retrain.py` (in progress)
4. ⏳ Create `evaluate.py` (in progress)
5. ⏳ Create `deploy.sh`
6. ⏳ Create `Dockerfile`
7. ⏳ Test locally
8. ⏳ Deploy to Cloud Run

**Result:** One simple service that does everything! 🎉

