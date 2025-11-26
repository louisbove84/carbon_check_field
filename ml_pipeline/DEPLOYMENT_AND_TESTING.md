# Deployment & Testing Guide

## How the Automated Pipeline Runs on GCP

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Cloud Scheduler (Cron Jobs)                                │
│                                                             │
│  • Schedule 1: "0 0 1 * *" (1st of every month at midnight)│
│    ↓ HTTP POST                                             │
│    └─> Cloud Function: monthly-data-collection             │
│                                                             │
│  • Schedule 2: "0 0 5 * *" (5th of every month at midnight)│
│    ↓ HTTP POST                                             │
│    └─> Cloud Function: auto-retrain-model                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Cloud Functions (Serverless Python)                        │
│                                                             │
│  1. monthly-data-collection                                │
│     • Runs on 1st of month                                 │
│     • Queries Earth Engine for Sentinel-2 imagery          │
│     • Extracts NDVI features                               │
│     • Validates against CDL                                │
│     • Inserts 400 samples into BigQuery                    │
│                                                             │
│  2. auto-retrain-model                                     │
│     • Runs on 5th of month                                 │
│     • Loads training data (excluding holdout)              │
│     • Trains challenger model                              │
│     • Evaluates vs champion on holdout set                 │
│     • Checks quality gates                                 │
│     • Deploys to Vertex AI if gates pass                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Data Storage & Model Serving                               │
│                                                             │
│  • BigQuery: Training data & metrics                       │
│  • Cloud Storage: Model artifacts                          │
│  • Vertex AI: Model endpoint (production serving)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Complete Setup Instructions

### Step 1: Setup BigQuery Tables

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline

# Create evaluation tables
bq query --use_legacy_sql=false < setup_evaluation_tables.sql
```

**Verify:**
```bash
bq ls ml-pipeline-477612:crop_ml
```

You should see:
- `training_features` (existing)
- `holdout_test_set` (new)
- `model_performance` (new)
- `deployment_history` (new)

---

### Step 2: Deploy Cloud Functions

#### 2a. Deploy Data Collection Function

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline

gcloud functions deploy monthly-data-collection \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=collect_training_data \
  --trigger-http \
  --allow-unauthenticated \
  --timeout=3600s \
  --memory=4Gi \
  --max-instances=1 \
  --service-account=ml-pipeline-sa@ml-pipeline-477612.iam.gserviceaccount.com
```

**Wait for deployment** (takes 2-3 minutes)

#### 2b. Deploy Model Retraining Function

```bash
gcloud functions deploy auto-retrain-model \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=retrain_model \
  --trigger-http \
  --allow-unauthenticated \
  --timeout=3600s \
  --memory=4Gi \
  --max-instances=1 \
  --service-account=ml-pipeline-sa@ml-pipeline-477612.iam.gserviceaccount.com
```

**Wait for deployment** (takes 2-3 minutes)

---

### Step 3: Setup Cloud Scheduler (Automated Triggers)

#### 3a. Schedule Monthly Data Collection

```bash
# Runs on 1st of every month at midnight UTC
gcloud scheduler jobs create http monthly-data-collection-trigger \
  --schedule="0 0 1 * *" \
  --uri="https://us-central1-ml-pipeline-477612.cloudfunctions.net/monthly-data-collection" \
  --http-method=POST \
  --location=us-central1 \
  --time-zone="America/Chicago" \
  --description="Monthly crop data collection from Earth Engine"
```

#### 3b. Schedule Monthly Model Retraining

```bash
# Runs on 5th of every month at midnight UTC
gcloud scheduler jobs create http auto-retrain-trigger \
  --schedule="0 0 5 * *" \
  --uri="https://us-central1-ml-pipeline-477612.cloudfunctions.net/auto-retrain-model" \
  --http-method=POST \
  --location=us-central1 \
  --time-zone="America/Chicago" \
  --description="Monthly model retraining with champion/challenger evaluation"
```

**Verify Schedules:**
```bash
gcloud scheduler jobs list --location=us-central1
```

---

## Manual Testing (Before Automation)

### Test 1: Create Holdout Test Set

**Purpose:** Verify holdout set creation works

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline

# Run Python script to create holdout
python3 << 'EOF'
from model_evaluation import create_or_load_holdout_set
import json

# Create holdout set (20% of all data)
df = create_or_load_holdout_set(force_recreate=False)

print(f"\n✅ Holdout set created: {len(df)} samples")
print(f"Crops: {df['crop'].value_counts().to_dict()}")
EOF
```

**Expected Output:**
```
📊 Managing holdout test set...
🔨 Creating new holdout test set (20% of all data)...
   • Corn: 150 samples reserved for testing
   • Soybeans: 145 samples reserved for testing
   • Alfalfa: 142 samples reserved for testing
   • Winter Wheat: 138 samples reserved for testing
✅ Holdout set created: 575 samples
```

**Verify in BigQuery:**
```bash
bq query --use_legacy_sql=false \
"SELECT crop, COUNT(*) as count FROM \`ml-pipeline-477612.crop_ml.holdout_test_set\` GROUP BY crop"
```

---

### Test 2: Manual Data Collection

**Purpose:** Test Earth Engine data collection

```bash
# Trigger data collection manually
curl -X POST \
  https://us-central1-ml-pipeline-477612.cloudfunctions.net/monthly-data-collection

# View logs
gcloud functions logs read monthly-data-collection \
  --region=us-central1 \
  --limit=100
```

**Expected in logs:**
```
🌾 MONTHLY CROP DATA COLLECTION PIPELINE
📅 Collection period: 2025-10-27 to 2025-11-26
🛰️  Creating Sentinel-2 composites...
📊 Using CDL year: 2025
🌱 Sampling 100 points for Corn (CDL code 1)
  ✅ Collected 100/100 samples for Corn
... (repeat for other crops)
📤 Loading 400 samples to BigQuery
✅ COLLECTION COMPLETE
```

**Verify in BigQuery:**
```bash
bq query --use_legacy_sql=false \
"SELECT crop, COUNT(*) as count, MAX(collection_date) as latest 
FROM \`ml-pipeline-477612.crop_ml.training_features\` 
GROUP BY crop 
ORDER BY crop"
```

---

### Test 3: Manual Model Retraining (Full Pipeline Test)

**Purpose:** Test complete retraining pipeline with evaluation

```bash
# Trigger retraining manually
curl -X POST \
  https://us-central1-ml-pipeline-477612.cloudfunctions.net/auto-retrain-model

# View logs in real-time
gcloud functions logs read auto-retrain-model \
  --region=us-central1 \
  --limit=200 \
  --format=json
```

**Expected in logs:**

```
============================================================
🌾 AUTOMATED MODEL RETRAINING PIPELINE
============================================================
📊 Step 0: Ensuring holdout test set exists...
✅ Using existing holdout set: 575 samples

📥 Step 1: Loading training data (excluding holdout)...
✅ Loaded 2300 training samples

🔍 Step 2: Checking data quality...
📊 Total samples: 2300
   • Corn: 575 samples
   • Soybeans: 575 samples
   • Alfalfa: 575 samples
   • Winter Wheat: 575 samples

🔧 Step 3: Engineering features...
✅ Created 17 total features

🤖 Step 4: Training challenger model...
   Train samples: 1840
   Test samples: 460
✅ Training accuracy: 94.12%
✅ Test accuracy: 87.39%

💾 Step 5: Saving challenger model to GCS...
✅ Model archived to gs://carboncheck-data/models/...
✅ Latest model updated at gs://carboncheck-data/models/crop_classifier_latest

⚖️  Step 6: Evaluating challenger vs champion...
============================================================
🎯 MODEL EVALUATION & DEPLOYMENT DECISION
============================================================
📊 Managing holdout test set...
✅ Using existing holdout set: 575 samples
🧪 Evaluating challenger...
   Accuracy: 85.22%
   • Corn: F1=0.87, n=150
   • Soybeans: F1=0.84, n=145
   • Alfalfa: F1=0.83, n=142
   • Winter Wheat: F1=0.86, n=138

⚠️  Could not find champion model (first deployment)

⚖️  Comparing models and making deployment decision...
============================================================
DEPLOYMENT DECISION
============================================================
✅ Accuracy 85.22% >= minimum 75%
✅ Corn F1=87.00% >= 70%
✅ Soybeans F1=84.00% >= 70%
✅ Alfalfa F1=83.00% >= 70%
✅ Winter Wheat F1=86.00% >= 70%
✅ No champion exists (first deployment)
🎉 All gates passed - deploying challenger!

🚀 Step 7: Deploying challenger (passed all gates)...
✅ Model registered: crop-classifier-latest (ID: 1234567890123456789)
✅ Endpoint created: crop-endpoint (ID: 2450616804754587648)
✅ Model deployed successfully!

============================================================
✅ RETRAINING PIPELINE COMPLETE
============================================================
⏱️  Duration: 8.3 minutes
📊 Training samples: 2300
🎯 Deployment: DEPLOYED
```

**Verify Deployment:**
```bash
# Check Vertex AI endpoint
gcloud ai endpoints list --region=us-central1

# Check deployed models
gcloud ai models list --region=us-central1

# View model performance in BigQuery
bq query --use_legacy_sql=false \
"SELECT * FROM \`ml-pipeline-477612.crop_ml.model_performance\` 
ORDER BY evaluation_time DESC LIMIT 5"
```

---

### Test 4: Verify Endpoint Works

**Purpose:** Test predictions from deployed endpoint

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline

python3 test_endpoint.py
```

**Expected Output:**
```
🧪 Testing Vertex AI Endpoint
Endpoint ID: 2450616804754587648
✅ Test 1: Corn features
   Prediction: Corn
   Confidence: High

✅ Test 2: Soybeans features
   Prediction: Soybeans
   Confidence: High

... etc
```

---

## Monitoring & Verification

### View Cloud Scheduler Status

```bash
# List all scheduled jobs
gcloud scheduler jobs list --location=us-central1

# View specific job details
gcloud scheduler jobs describe monthly-data-collection-trigger \
  --location=us-central1

# View recent executions
gcloud scheduler jobs describe monthly-data-collection-trigger \
  --location=us-central1 \
  --format="table(status.lastAttemptTime, status.code)"
```

### View Function Execution History

```bash
# Data collection logs
gcloud functions logs read monthly-data-collection \
  --region=us-central1 \
  --limit=50

# Retraining logs
gcloud functions logs read auto-retrain-model \
  --region=us-central1 \
  --limit=200
```

### Query Metrics in BigQuery

#### Recent Model Performance
```sql
SELECT 
  model_type,
  model_name,
  accuracy,
  corn_f1,
  soybeans_f1,
  alfalfa_f1,
  winter_wheat_f1,
  evaluation_time
FROM `ml-pipeline-477612.crop_ml.model_performance`
ORDER BY evaluation_time DESC
LIMIT 10;
```

#### Deployment History
```sql
SELECT 
  deployment_time,
  deployment_decision,
  accuracy,
  training_samples,
  gates_failed
FROM `ml-pipeline-477612.crop_ml.deployment_history`
ORDER BY deployment_time DESC
LIMIT 10;
```

#### Training Data Growth
```sql
SELECT 
  DATE_TRUNC(collection_date, MONTH) as month,
  crop,
  COUNT(*) as samples
FROM `ml-pipeline-477612.crop_ml.training_features`
GROUP BY month, crop
ORDER BY month DESC, crop;
```

---

## Manually Trigger Scheduled Jobs

### Trigger Data Collection Now
```bash
gcloud scheduler jobs run monthly-data-collection-trigger \
  --location=us-central1
```

### Trigger Model Retraining Now
```bash
gcloud scheduler jobs run auto-retrain-trigger \
  --location=us-central1
```

**View execution:**
```bash
# Wait 30 seconds, then check logs
gcloud functions logs read auto-retrain-model \
  --region=us-central1 \
  --limit=100
```

---

## Troubleshooting

### Issue: Cloud Function Timeout

**Symptoms:** Function fails after 9 minutes

**Solutions:**
```bash
# Increase timeout to 1 hour (max)
gcloud functions deploy auto-retrain-model \
  --timeout=3600s \
  --update
```

### Issue: Out of Memory

**Symptoms:** Function crashes during training

**Solutions:**
```bash
# Increase memory to 8GB
gcloud functions deploy auto-retrain-model \
  --memory=8Gi \
  --update
```

### Issue: Earth Engine Auth Failed

**Symptoms:** "Earth Engine initialization failed"

**Solutions:**
```bash
# Verify service account has Earth Engine access
gcloud projects get-iam-policy ml-pipeline-477612 \
  --flatten="bindings[].members" \
  --filter="bindings.members:ml-pipeline-sa@ml-pipeline-477612.iam.gserviceaccount.com"

# Add Earth Engine permissions if missing
gcloud projects add-iam-policy-binding ml-pipeline-477612 \
  --member="serviceAccount:ml-pipeline-sa@ml-pipeline-477612.iam.gserviceaccount.com" \
  --role="roles/earthengine.writer"
```

### Issue: Deployment Always Blocked

**Check why:**
```sql
SELECT 
  deployment_time,
  accuracy,
  gates_failed,
  reasoning
FROM `ml-pipeline-477612.crop_ml.deployment_history`
WHERE deployment_decision = 'blocked'
ORDER BY deployment_time DESC
LIMIT 5;
```

**Adjust gates if too strict:**
Edit `ml_pipeline/model_evaluation.py`:
```python
ABSOLUTE_MIN_ACCURACY = 0.70  # Lower from 0.75
MIN_PER_CROP_F1 = 0.65        # Lower from 0.70
IMPROVEMENT_MARGIN = 0.01      # Lower from 0.02
```

Then redeploy:
```bash
gcloud functions deploy auto-retrain-model \
  --source=. \
  --update
```

---

## Cost Monitoring

### View Cloud Function Costs
```bash
# Go to GCP Console > Billing > Reports
# Filter by SKU: "Cloud Functions"
```

**Expected monthly costs:**
- Data collection: ~$1-2/month (runs once/month, 5-10 min)
- Model retraining: ~$3-5/month (runs once/month, 10-15 min)
- Total: **~$5-7/month**

### View Vertex AI Costs
```bash
# Go to GCP Console > Billing > Reports
# Filter by SKU: "Vertex AI Prediction"
```

**Expected monthly costs:**
- Endpoint hosting: ~$50-100/month (n1-standard-2, 1 replica)
- Predictions: ~$0.10 per 1,000 predictions

---

## Summary

### Automated Setup Complete When:
- ✅ BigQuery tables created
- ✅ Cloud Functions deployed
- ✅ Cloud Scheduler jobs configured
- ✅ Initial holdout set created
- ✅ First model deployed

### Monthly Automated Flow:
- **Day 1:** Data collection runs automatically (adds 400 samples)
- **Day 5:** Retraining runs automatically (trains, evaluates, deploys if better)
- **You:** Monitor BigQuery metrics and deployment decisions

### Manual Testing Checklist:
1. ✅ Create BigQuery tables
2. ✅ Deploy both Cloud Functions
3. ✅ Create holdout test set manually
4. ✅ Trigger data collection manually
5. ✅ Trigger retraining manually
6. ✅ Verify endpoint predictions work
7. ✅ Setup Cloud Scheduler
8. ✅ Monitor first automated run

**You're all set! The pipeline will run on its own every month.** 🎉

