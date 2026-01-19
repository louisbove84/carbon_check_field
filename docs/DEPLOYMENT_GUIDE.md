# ML Pipeline Deployment Guide
## Cloud Run (Orchestrator) + Vertex AI (Trainer)

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Folder Structure](#folder-structure)
- [Service Separation and Dependencies](#service-separation-and-dependencies)
- [Deployment Steps](#deployment-steps)
  - [Step 1: One-time setup + first run (local)](#step-1-one-time-setup--first-run-local)
  - [Step 2: Build trainer container (Cloud Run/Vertex AI)](#step-2-build-trainer-container-cloud-runvertex-ai)
  - [Step 3: Deploy orchestrator (Cloud Run)](#step-3-deploy-orchestrator-cloud-run)
  - [Step 4: Run pipeline (Cloud Run)](#step-4-run-pipeline-cloud-run)
- [Scheduling (Cloud Scheduler)](#scheduling-cloud-scheduler)
- [Monitoring](#monitoring)
- [Configuration](#configuration)
- [Cost Breakdown](#cost-breakdown)
- [Testing](#testing)
- [Next Steps](#next-steps)

---

## ️ Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│ HTTP POST Request                                         │
└────────────────────┬──────────────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────────────┐
│ Cloud Run Orchestrator (Lightweight)                       │
│ - Export Earth Engine data to GCS                          │
│ - Trigger Vertex AI Training Job                           │
│   - Vertex AI Custom Training (inside trainer image)       │
│     - Load data from BigQuery/GCS                          │
│     - Train RandomForest model                             │
│     - Save model to GCS                                    │
│     - Return metrics                                       │
│ - Monitor training                                         │
│ - Evaluate & deploy if gates pass                          │
└────────────────────────────────────────────────────────────┘

```

---

##  Folder Structure

```
ml_pipeline/
├── orchestrator/              # Cloud Run orchestrator
│   ├── main.py               # Flask HTTP endpoint
│   ├── orchestrator.py       # Orchestration logic
│   ├── config.yaml           # Configuration
│   ├── Dockerfile            # Lightweight image
│   ├── requirements.txt      # Minimal deps (no ML libs)
│   └── deploy.sh             # Deploy to Cloud Run
│
├── trainer/                  # Vertex AI training container
│   ├── vertex_ai_training.py             # Training script
│   ├── Dockerfile           # ML-optimized image
│   ├── requirements.txt     # ML libraries
│   └── build_docker.sh             # Build & push to Artifact Registry
│
└── config.yaml              # Shared configuration
```

---

## Service Separation and Dependencies

This project deploys **separate services**, each with its own container and requirements:
- **Backend API** → Cloud Run (FastAPI)
- **ML Orchestrator** → Cloud Run (Flask, lightweight)
- **ML Trainer** → Vertex AI (training job)

Each service has its own `requirements.txt` that matches its Dockerfile:
- `backend/requirements.txt` → `backend/Dockerfile`
- `ml_pipeline/orchestrator/requirements.txt` → `ml_pipeline/orchestrator/Dockerfile`
- `ml_pipeline/trainer/requirements.txt` → `ml_pipeline/trainer/Dockerfile`

For **local development**, use:
- `requirements-all.txt` for a consolidated install
- `pyproject.toml` for editable installs (optional)

GCP deployments use the **service-specific** requirements, not the consolidated one.

---

##  Deployment Steps

#### Step 1: One-time setup + first run (local)
Provision required GCP resources and run the pipeline once from your machine:
```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline
python setup_and_run_pipeline.py
```

**What this does:**
- Creates GCS buckets
- Uploads `config.yaml` to GCS
- Creates TensorBoard instance
- Creates BigQuery dataset (tables are created during export)
- Runs the full pipeline (Earth Engine → Training → Deploy)

---

#### Step 2: Build trainer container (Cloud Run/Vertex AI)
Build and push the training container to Artifact Registry:
```bash
cd trainer
./build_docker.sh
```

---

#### Step 3: Deploy orchestrator (Cloud Run)
Deploy the orchestrator so it can be triggered remotely/scheduled:
```bash
cd orchestrator
./deploy.sh
```

---

#### Step 4: Run pipeline (Cloud Run)
Trigger the complete pipeline with one HTTP request:
```bash
curl -X POST <CLOUD_RUN_SERVICE_URL>
```

---

## Scheduling (Cloud Scheduler)
Create a monthly Cloud Scheduler job that calls the orchestrator endpoint:
```bash
gcloud scheduler jobs create http carboncheck-monthly-retrain \
  --schedule="0 3 1 * *" \
  --uri="<CLOUD_RUN_SERVICE_URL>" \
  --http-method=POST \
  --time-zone="America/Chicago" \
  --location="us-central1"
```

Notes:
- This uses unauthenticated access (current Cloud Run config allows it).
- If you restrict Cloud Run, add `--oidc-service-account-email` and grant the Scheduler SA access.

##  Monitoring

### View Orchestrator Logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ml-pipeline" \
  --project=ml-pipeline-477612 \
  --limit=50 \
  --format=json
```

### View Training Logs
```bash
gcloud ai custom-jobs list \
  --region=us-central1 \
  --project=ml-pipeline-477612
```

---

## ️ Configuration

Edit `config.yaml` to change settings:

```yaml
# Training infrastructure
training:
  machine_type: n1-standard-4      # Or n1-highmem-8, etc.
  accelerator_type: null           # Or "NVIDIA_TESLA_T4" for GPU
  accelerator_count: 0

# Quality gates
quality_gates:
  absolute_min_accuracy: 0.75
  min_per_crop_f1: 0.70
```

**To update config:**
```bash
gsutil cp config.yaml gs://carboncheck-data/config/config.yaml
```

**No redeployment needed!** Next pipeline run will use the new config.

---

##  Cost Breakdown

| Component                         | Duration   | Cost per Run     |
|-----------------------------------|------------|------------------|
| **Cloud Run orchestrator**        | 1-2 min    | ~$0.10           |
| **Vertex AI training (n1-std-4)** | 5-10 min   | ~$0.50-$2.00     |
| **Cloud Storage**                 | -          | ~$0.01           |
| **BigQuery**                      | -          | ~$0.05           |
| **Total per run**                 | ~10-15 min | **~$0.66-$2.16** |

**Monthly cost (1 run/month):** ~$0.66-$2.16

**vs Old Cloud Functions:** ~$5-7/month  
**Savings: 60-70%!** 

---

##  Testing

### Test Orchestrator Locally
```bash
cd orchestrator
python main.py
# In another terminal:
curl -X POST http://localhost:8080
```

### Test Trainer Locally
```bash
cd trainer
python vertex_ai_training.py
```

---

##  Next Steps

1.  Build trainer container (`cd trainer && ./build_docker.sh`)
2.  Deploy orchestrator (`cd orchestrator && ./deploy.sh`)
3.  Run pipeline (`curl -X POST <service-url>`)
4.  Set up Cloud Scheduler for monthly runs
5.  Monitor BigQuery for metrics

**That's it!** Your ML pipeline is now fully automated. 

