# Simplified ML Pipeline Architecture (Cloud Run + Vertex AI)

## Overview

Split workload between **Cloud Run (orchestrator)** and **Vertex AI Custom Training (heavy ML)**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Cloud Run (Lightweight Orchestrator)                       │
│                                                             │
│ Container: orchestrator/                                   │
│ ├── main.py              Flask app (HTTP endpoint)         │
│ ├── orchestrator.py      Orchestration logic               │
│ ├── config.yaml          Configuration                     │
│ └── Dockerfile           Lightweight image                 │
│                                                             │
│ What it does:                                              │
│ 1. Authenticate to Earth Engine + GCS                      │
│ 2. Trigger Earth Engine export → GCS                       │
│ 3. Call Vertex AI Training Job API                         │
│ 4. Monitor training completion                             │
│ 5. Evaluate results & deploy if gates pass                 │
└─────────────────────────────────────────────────────────────┘
                           ↓ triggers
┌─────────────────────────────────────────────────────────────┐
│ Vertex AI Custom Training (Heavy ML Workload)              │
│                                                             │
│ Container: trainer/                                        │
│ ├── train.py             Training script                   │
│ ├── Dockerfile           ML-optimized image                │
│ └── requirements.txt     ML libraries (sklearn, pandas)    │
│                                                             │
│ What it does:                                              │
│ 1. Load training data from GCS                             │
│ 2. Train RandomForest model                                │
│ 3. Save model artifact to GCS                              │
│ 4. Return metrics to Vertex AI                             │
│                                                             │
│ Runs on: Managed infrastructure (can use GPUs/TPUs!)      │
└─────────────────────────────────────────────────────────────┘
```

## Folder Structure

```
ml_pipeline/
├── orchestrator/           # Cloud Run orchestrator
│   ├── main.py            # Flask app + HTTP endpoint
│   ├── orchestrator.py    # Orchestration logic
│   ├── config.yaml        # Configuration
│   ├── Dockerfile         # Lightweight image
│   ├── requirements.txt   # Minimal deps (no ML libs)
│   └── deploy.sh          # Deploy orchestrator
│
├── trainer/               # Vertex AI training container
│   ├── train.py          # Training script
│   ├── Dockerfile        # ML-optimized image
│   ├── requirements.txt  # ML libraries
│   └── build.sh          # Build & push to Artifact Registry
│
└── config.yaml           # Shared configuration
```

## How It Works

### Step 1: Cloud Run Orchestrator Starts

```bash
curl -X POST https://ml-pipeline-xxxxx.run.app
```

**Orchestrator does:**
1. Load `config.yaml` from Cloud Storage
2. Authenticate to Earth Engine
3. Export training data from Earth Engine → Cloud Storage
4. Call Vertex AI Training Job API

### Step 2: Vertex AI Runs Training

**Vertex AI does:**
1. Spins up training container on managed infrastructure
2. Runs `train.py` script
3. Script loads data from GCS
4. Trains model
5. Saves model to GCS
6. Returns metrics

### Step 3: Orchestrator Completes

**Orchestrator does:**
1. Waits for training completion
2. Loads model metrics
3. Evaluates against quality gates
4. Deploys to Vertex AI endpoint if gates pass
5. Returns final status

## Benefits

### ✅ Separation of Concerns
- Cloud Run: Orchestration only (lightweight, cheap)
- Vertex AI: Heavy ML training (optimized infrastructure)

### ✅ Cost Optimization
- Cloud Run: Pay only for orchestration time (~1-2 minutes)
- Vertex AI: Pay only for training time (~5-10 minutes)
- No idle time costs

### ✅ Scalability
- Can use GPUs for training if needed
- Vertex AI handles resource management
- Training container can scale independently

### ✅ Simplicity
- One API call triggers everything
- Clear separation between orchestration and training
- Easy to debug (separate logs for each component)

## Deployment

### 1. Build & Push Training Container

```bash
cd trainer
./build.sh  # Builds and pushes to Artifact Registry
```

### 2. Deploy Orchestrator to Cloud Run

```bash
cd orchestrator
./deploy.sh  # Deploys orchestrator
```

### 3. Run Pipeline

```bash
curl -X POST https://ml-pipeline-xxxxx.run.app
```

**That's it!** One HTTP call triggers the entire pipeline.

## Configuration

**Single `config.yaml` for both containers:**

```yaml
project:
  id: ml-pipeline-477612
  region: us-central1

training:
  machine_type: n1-standard-4  # Or n1-highmem-8, n1-standard-16, etc.
  accelerator_type: null       # Or "NVIDIA_TESLA_T4" for GPU
  accelerator_count: 0

quality_gates:
  absolute_min_accuracy: 0.75
  min_per_crop_f1: 0.70
```

## Cost Estimate

- **Cloud Run orchestrator:** ~$0.10/run (1-2 minutes)
- **Vertex AI training:** ~$0.50-2.00/run (5-10 minutes on n1-standard-4)
- **Total per month:** ~$1-3/month (runs once monthly)

**vs Old Architecture:**
- Cloud Functions: ~$5-7/month
- **Savings: 60-70%!**

## Next Steps

1. Create `orchestrator/` folder with lightweight orchestration code
2. Create `trainer/` folder with ML training code  
3. Build and deploy both containers
4. Test end-to-end pipeline

**Much cleaner and more scalable!** 🚀

