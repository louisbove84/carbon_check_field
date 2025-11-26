# Quick Start Guide

## TL;DR - Get Pipeline Running in 15 Minutes

### 1. Create BigQuery Tables (1 minute)
```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline
bq query --use_legacy_sql=false < setup_evaluation_tables.sql
```

### 2. Deploy Cloud Functions (5 minutes)
```bash
# Data collection function
gcloud functions deploy monthly-data-collection \
  --gen2 --runtime=python311 --region=us-central1 \
  --source=. --entry-point=collect_training_data \
  --trigger-http --allow-unauthenticated \
  --timeout=3600s --memory=4Gi --max-instances=1 \
  --service-account=ml-pipeline-sa@ml-pipeline-477612.iam.gserviceaccount.com

# Model retraining function (with evaluation)
gcloud functions deploy auto-retrain-model \
  --gen2 --runtime=python311 --region=us-central1 \
  --source=. --entry-point=retrain_model \
  --trigger-http --allow-unauthenticated \
  --timeout=3600s --memory=4Gi --max-instances=1 \
  --service-account=ml-pipeline-sa@ml-pipeline-477612.iam.gserviceaccount.com
```

### 3. Test Manually (5 minutes)
```bash
# Test holdout set creation
./test_pipeline.sh holdout

# Test retraining (takes 5-10 min)
./test_pipeline.sh retrain

# View metrics
./test_pipeline.sh metrics
```

### 4. Setup Automation (2 minutes)
```bash
# Schedule data collection (1st of month)
gcloud scheduler jobs create http monthly-data-collection-trigger \
  --schedule="0 0 1 * *" \
  --uri="https://us-central1-ml-pipeline-477612.cloudfunctions.net/monthly-data-collection" \
  --http-method=POST --location=us-central1 \
  --time-zone="America/Chicago"

# Schedule retraining (5th of month)
gcloud scheduler jobs create http auto-retrain-trigger \
  --schedule="0 0 5 * *" \
  --uri="https://us-central1-ml-pipeline-477612.cloudfunctions.net/auto-retrain-model" \
  --http-method=POST --location=us-central1 \
  --time-zone="America/Chicago"
```

### 5. Verify & Monitor (2 minutes)
```bash
# Check scheduled jobs
gcloud scheduler jobs list --location=us-central1

# View recent metrics
./test_pipeline.sh metrics

# View function logs
gcloud functions logs read auto-retrain-model --region=us-central1 --limit=50
```

---

## Testing Commands

```bash
# Test specific steps
./test_pipeline.sh holdout      # Create holdout test set
./test_pipeline.sh collection   # Trigger data collection
./test_pipeline.sh retrain      # Trigger retraining
./test_pipeline.sh endpoint     # Test predictions
./test_pipeline.sh metrics      # View BigQuery metrics

# Run all tests interactively
./test_pipeline.sh
```

---

## Monitoring Commands

```bash
# View recent model performance
bq query --use_legacy_sql=false \
"SELECT model_type, accuracy, corn_f1, soybeans_f1, evaluation_time 
FROM \`ml-pipeline-477612.crop_ml.model_performance\` 
ORDER BY evaluation_time DESC LIMIT 5"

# View deployment history
bq query --use_legacy_sql=false \
"SELECT deployment_time, deployment_decision, accuracy, gates_failed 
FROM \`ml-pipeline-477612.crop_ml.deployment_history\` 
ORDER BY deployment_time DESC LIMIT 5"

# View function logs
gcloud functions logs read auto-retrain-model --region=us-central1 --limit=100

# View Cloud Scheduler execution history
gcloud scheduler jobs describe auto-retrain-trigger --location=us-central1
```

---

## Manual Triggers (Testing)

```bash
# Trigger data collection now
gcloud scheduler jobs run monthly-data-collection-trigger --location=us-central1

# Trigger retraining now
gcloud scheduler jobs run auto-retrain-trigger --location=us-central1

# Or use HTTP:
curl -X POST https://us-central1-ml-pipeline-477612.cloudfunctions.net/auto-retrain-model
```

---

## How It Works

### Automated Monthly Schedule

**Day 1 (1st of month):**
- Cloud Scheduler triggers `monthly-data-collection`
- Collects 400 new samples from Earth Engine
- Adds to BigQuery `training_features` table

**Day 5 (5th of month):**
- Cloud Scheduler triggers `auto-retrain-model`
- Creates holdout set (first run only, 20% of data)
- Trains challenger model on remaining 80%
- Evaluates challenger vs champion on holdout set
- **Deploys only if:**
  - Accuracy ≥ 75%
  - Each crop F1 ≥ 0.70
  - Beats champion by ≥ 2%
- Logs decision to BigQuery

### Quality Gates

✅ **Gate 1:** Absolute minimum accuracy (75%)  
✅ **Gate 2:** Per-crop F1 score (0.70 each)  
✅ **Gate 3:** Beat champion by margin (2%)

All gates must pass to deploy!

---

## Troubleshooting

### Function fails with timeout
```bash
gcloud functions deploy auto-retrain-model --timeout=3600s --update
```

### Out of memory
```bash
gcloud functions deploy auto-retrain-model --memory=8Gi --update
```

### Deployment always blocked
Check why:
```sql
SELECT deployment_time, gates_failed, reasoning
FROM `ml-pipeline-477612.crop_ml.deployment_history`
WHERE deployment_decision = 'blocked'
ORDER BY deployment_time DESC LIMIT 3
```

Adjust gates in `model_evaluation.py` if too strict.

---

## Cost Estimate

- **Data Collection:** ~$1-2/month (runs once, 5-10 min)
- **Model Retraining:** ~$3-5/month (runs once, 10-15 min)
- **Vertex AI Endpoint:** ~$50-100/month (always-on serving)
- **BigQuery Storage:** ~$1/month (few GB)
- **Total:** ~$55-110/month

---

## Full Documentation

- **Complete Setup:** [`DEPLOYMENT_AND_TESTING.md`](DEPLOYMENT_AND_TESTING.md)
- **Evaluation System:** [`MODEL_EVALUATION_GUIDE.md`](MODEL_EVALUATION_GUIDE.md)
- **Main README:** [`../README.md`](../README.md)

---

## Quick Checks

```bash
# Are functions deployed?
gcloud functions list --region=us-central1

# Are schedules active?
gcloud scheduler jobs list --location=us-central1

# Is endpoint running?
gcloud ai endpoints list --region=us-central1

# Recent training data?
bq query --use_legacy_sql=false \
"SELECT crop, COUNT(*) FROM \`ml-pipeline-477612.crop_ml.training_features\` 
GROUP BY crop"

# Is holdout set created?
bq query --use_legacy_sql=false \
"SELECT COUNT(*) as holdout_samples FROM \`ml-pipeline-477612.crop_ml.holdout_test_set\`"
```

---

**🎉 That's it! Your pipeline is now automated and runs monthly on GCP.**

