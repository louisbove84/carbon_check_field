# Quick Start Guide

## TL;DR - Get Pipeline Running in 10-15 Minutes

### 1. One-time setup + first run (local)
This script provisions the required GCP resources and runs the ML pipeline once from your machine:
```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline
python setup_and_run_pipeline.py
```

**What it does:**
- Creates GCS buckets
- Uploads `config.yaml` to GCS
- Creates TensorBoard instance
- Creates BigQuery dataset (tables are created during export)
- Runs the full pipeline (Earth Engine → Training → Deploy)

### 2. Re-run the pipeline only (local)
Requires one-time setup to have been completed (or resources created manually). This runs the pipeline from your machine:
```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline/orchestrator
python orchestrator.py
```

### 3. Optional: Deploy the orchestrator to Cloud Run
`setup_and_run_pipeline.py` does not deploy the orchestrator. Deploying Cloud Run is only needed if you want scheduled or remote runs (Cloud Scheduler/HTTP) without your machine:
```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline/orchestrator
./deploy.sh
```

### 4. Optional: Schedule monthly runs (Cloud Scheduler)
Once you have a Cloud Run URL, schedule a POST to it:
```bash
gcloud scheduler jobs create http carboncheck-monthly-retrain \
  --schedule="0 3 1 * *" \
  --uri="<CLOUD_RUN_SERVICE_URL>" \
  --http-method=POST \
  --time-zone="America/Chicago" \
  --location="us-central1"
```

---

## Optional: Create Evaluation Tables
Evaluation still happens in the pipeline; these tables are only needed if you want to run the extra evaluation queries/tools:
```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline/tools
bq query --use_legacy_sql=false < setup_evaluation_tables.sql
```

---

## Monitoring Commands

```bash
# Orchestrator logs (Cloud Run)
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ml-pipeline" \
  --project=ml-pipeline-477612 \
  --limit=50

# Vertex AI training jobs
gcloud ai custom-jobs list \
  --region=us-central1 \
  --project=ml-pipeline-477612

# Scheduled jobs
gcloud scheduler jobs list --location=us-central1
```

---

## Full Documentation

- **Deployment Guide:** [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)
- **Evaluation System:** [`MODEL_EVALUATION_GUIDE.md`](MODEL_EVALUATION_GUIDE.md)
- **Main README:** [`../README.md`](../README.md)

---

**That's it!** The pipeline is now set up and can be rerun locally or scheduled via Cloud Run.

