# ML Pipeline Deployment Guide
## Cloud Run (Orchestrator) + Vertex AI (Trainer)

---

## ️ Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│ HTTP POST Request                                        │
└────────────────────┬──────────────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────────────┐
│ Cloud Run Orchestrator (Lightweight)                      │
│ - Export Earth Engine data to GCS                         │
│ - Trigger Vertex AI Training Job                          │
│ - Monitor training                                         │
│ - Evaluate & deploy if gates pass                         │
└────────────────────┬───────────────────────────────────────┘
                     ↓ triggers
┌────────────────────────────────────────────────────────────┐
│ Vertex AI Custom Training (Heavy ML)                      │
│ - Load data from BigQuery/GCS                             │
│ - Train RandomForest model                                │
│ - Save model to GCS                                       │
│ - Return metrics                                          │
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

##  Deployment Steps

### Step 1: Build Trainer Container

Build and push the training container to Artifact Registry:

```bash
cd trainer
./build_docker.sh
```

**What this does:**
- Creates Artifact Registry repository if needed
- Builds Docker image with ML libraries
- Pushes to `us-central1-docker.pkg.dev/ml-pipeline-477612/ml-containers/crop-trainer:latest`

**Duration:** ~3-5 minutes

---

### Step 2: Deploy Orchestrator

Deploy the orchestrator to Cloud Run:

```bash
cd orchestrator
./deploy.sh
```

**What this does:**
- Uploads `config.yaml` to Cloud Storage
- Builds lightweight Docker image
- Deploys to Cloud Run
- Returns service URL

**Duration:** ~2-3 minutes

---

### Step 3: Run Pipeline

Trigger the complete pipeline with one HTTP request:

```bash
curl -X POST https://ml-pipeline-6by67xpgga-uc.a.run.app
```

**What happens:**
1.  Cloud Run orchestrator starts
2.  Exports Earth Engine data to GCS
3.  Triggers Vertex AI training job
4. ⏳ Monitors training completion
5.  Evaluates and deploys if gates pass

**Duration:** ~10-15 minutes total

---

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

##  Scheduled Runs

Set up monthly automated runs with Cloud Scheduler:

```bash
# Create Cloud Scheduler job
gcloud scheduler jobs create http ml-pipeline-monthly \
  --location=us-central1 \
  --schedule="0 0 1 * *" \
  --uri="https://ml-pipeline-6by67xpgga-uc.a.run.app" \
  --http-method=POST \
  --project=ml-pipeline-477612
```

**Schedule:** Runs on the 1st of every month at midnight

---

##  Cost Breakdown

| Component | Duration | Cost per Run |
|-----------|----------|--------------|
| **Cloud Run orchestrator** | 1-2 min | ~$0.10 |
| **Vertex AI training (n1-standard-4)** | 5-10 min | ~$0.50-$2.00 |
| **Cloud Storage** | - | ~$0.01 |
| **BigQuery** | - | ~$0.05 |
| **Total per run** | ~10-15 min | **~$0.66-$2.16** |

**Monthly cost (1 run/month):** ~$0.66-$2.16

**vs Old Cloud Functions:** ~$5-7/month  
**Savings: 60-70%!** 

---

## ️ Troubleshooting

### Orchestrator fails to trigger training
**Error:** `Permission denied`  
**Fix:** Ensure service account has `aiplatform.customJobs.create` permission:
```bash
gcloud projects add-iam-policy-binding ml-pipeline-477612 \
  --member="serviceAccount:ml-pipeline-sa@ml-pipeline-477612.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### Training job fails
**Error:** `Container image not found`  
**Fix:** Rebuild trainer container:
```bash
cd trainer && ./build_docker.sh
```

### Model doesn't deploy
**Error:** `Quality gates failed`  
**Fix:** Check logs for accuracy. Lower thresholds in `config.yaml` if needed.

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

