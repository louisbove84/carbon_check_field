# Model Evaluation & Deployment Gating Guide

## Overview

This guide explains the **champion/challenger evaluation system** that ensures only high-quality models are deployed to production.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Monthly Data Collection (1st of month)                  │
│    • Collect 400 new samples (100 per crop)                │
│    • Push to BigQuery: training_features table             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Create/Load Holdout Test Set (one-time setup)           │
│    • Reserve 20% of all data for testing                   │
│    • Stratified by crop (balanced)                         │
│    • Stored in BigQuery: holdout_test_set table            │
│    • NEVER used for training (prevents data leakage)       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Train Challenger Model (5th of month)                   │
│    • Load training data EXCLUDING holdout                  │
│    • Train RandomForest pipeline (scaler + model)          │
│    • Save to GCS (versioned + latest)                      │
│    • DO NOT deploy yet                                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Evaluate Challenger vs Champion                         │
│    • Load current production model (champion)              │
│    • Load newly trained model (challenger)                 │
│    • Test BOTH on same holdout set                         │
│    • Compare: accuracy, per-crop F1, confusion matrix      │
│    • Store metrics in BigQuery: model_performance table    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Quality Gates (deployment decision)                     │
│                                                             │
│    Gate 1: Absolute Minimum Accuracy                       │
│    ✅ Challenger accuracy >= 75%                            │
│                                                             │
│    Gate 2: Per-Crop Performance                            │
│    ✅ Each crop F1 score >= 0.70                            │
│                                                             │
│    Gate 3: Beat Champion (if exists)                       │
│    ✅ Challenger accuracy > champion accuracy + 2%          │
│                                                             │
│    ALL gates must pass to deploy                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Deploy or Block                                          │
│                                                             │
│    IF all gates passed:                                    │
│    ✅ Deploy challenger to Vertex AI endpoint               │
│    ✅ Challenger becomes new champion                       │
│    ✅ Log deployment to deployment_history table            │
│                                                             │
│    IF any gate failed:                                     │
│    ❌ Block deployment                                      │
│    ❌ Champion remains in production                        │
│    ❌ Log decision with failure reasons                     │
│    ❌ Send alert notification                               │
└─────────────────────────────────────────────────────────────┘
```

## Quality Gates Explained

### Gate 1: Absolute Minimum Accuracy
- **Threshold:** 75%
- **Purpose:** Ensures model is production-ready
- **Failure:** Model is too inaccurate for real-world use

### Gate 2: Per-Crop F1 Score
- **Threshold:** 0.70 for each crop
- **Purpose:** Prevents models that are good overall but fail on specific crops
- **Failure:** One or more crops have poor performance

### Gate 3: Beat Champion
- **Threshold:** +2% improvement over current production model
- **Purpose:** Only deploy models that are meaningfully better
- **Failure:** Challenger is not significantly better than champion

## BigQuery Tables

### 1. `holdout_test_set`
**Purpose:** Permanent test set for unbiased evaluation

| Column | Type | Description |
|--------|------|-------------|
| sample_id | STRING | Unique sample identifier |
| crop | STRING | Crop name |
| ndvi_* | FLOAT64 | NDVI features |
| reserved_date | TIMESTAMP | When sample was reserved for testing |

**Key Points:**
- Created once with 20% of all data
- Never used for training
- Stratified by crop
- Enables consistent comparison across model versions

### 2. `model_performance`
**Purpose:** Track all model evaluations

| Column | Type | Description |
|--------|------|-------------|
| model_type | STRING | "champion" or "challenger" |
| model_name | STRING | Model identifier |
| accuracy | FLOAT64 | Overall accuracy on holdout set |
| corn_f1 | FLOAT64 | F1 score for Corn |
| soybeans_f1 | FLOAT64 | F1 score for Soybeans |
| alfalfa_f1 | FLOAT64 | F1 score for Alfalfa |
| winter_wheat_f1 | FLOAT64 | F1 score for Winter Wheat |
| evaluation_time | TIMESTAMP | When evaluation was performed |
| metrics_json | STRING | Full metrics as JSON |

**Key Points:**
- Tracks every model evaluation
- Enables historical comparison
- Powers monitoring dashboards

### 3. `deployment_history`
**Purpose:** Log all deployment decisions

| Column | Type | Description |
|--------|------|-------------|
| deployment_time | TIMESTAMP | When decision was made |
| model_id | STRING | Vertex AI model ID (if deployed) |
| deployment_decision | STRING | "deployed", "blocked", or "rollback" |
| accuracy | FLOAT64 | Model accuracy |
| gates_passed | STRING | Which gates passed (JSON) |
| gates_failed | STRING | Which gates failed (JSON) |
| decision_json | STRING | Full decision details |

**Key Points:**
- Audit trail of all deployments
- Tracks why models were blocked
- Enables rollback decisions

## Setup Instructions

### 1. Create BigQuery Tables

```bash
cd ml_pipeline
bq query --use_legacy_sql=false < setup_evaluation_tables.sql
```

### 2. Create Holdout Test Set

```bash
python -c "from model_evaluation import create_or_load_holdout_set; create_or_load_holdout_set(force_recreate=True)"
```

This will:
- Load all data from `training_features` table
- Sample 20% stratified by crop
- Save to `holdout_test_set` table

### 3. Deploy Updated Cloud Functions

```bash
# Deploy data collection (unchanged)
gcloud functions deploy monthly-data-collection \
  --gen2 \
  --runtime=python310 \
  --region=us-central1 \
  --source=. \
  --entry-point=collect_training_data \
  --trigger-http \
  --allow-unauthenticated

# Deploy retraining with evaluation
gcloud functions deploy auto-retrain-model \
  --gen2 \
  --runtime=python310 \
  --region=us-central1 \
  --source=. \
  --entry-point=retrain_model \
  --trigger-http \
  --allow-unauthenticated \
  --timeout=3600s \
  --memory=4GB
```

### 4. Test the Pipeline

```bash
# Trigger retraining manually
curl -X POST https://us-central1-ml-pipeline-477612.cloudfunctions.net/auto-retrain-model
```

Check the logs to see evaluation results and deployment decision.

## Monitoring & Queries

### View Recent Model Performance

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
WHERE evaluation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
ORDER BY evaluation_time DESC;
```

### View Deployment History

```sql
SELECT 
  deployment_time,
  deployment_decision,
  accuracy,
  model_id,
  JSON_EXTRACT_SCALAR(gates_failed, '$[0]') as first_failure_reason
FROM `ml-pipeline-477612.crop_ml.deployment_history`
ORDER BY deployment_time DESC
LIMIT 10;
```

### Compare Champion vs Challenger Over Time

```sql
WITH latest_evaluations AS (
  SELECT 
    model_type,
    accuracy,
    (corn_f1 + soybeans_f1 + alfalfa_f1 + winter_wheat_f1) / 4 as avg_f1,
    evaluation_time,
    ROW_NUMBER() OVER (PARTITION BY model_type ORDER BY evaluation_time DESC) as rn
  FROM `ml-pipeline-477612.crop_ml.model_performance`
)
SELECT 
  model_type,
  accuracy,
  avg_f1,
  evaluation_time
FROM latest_evaluations
WHERE rn = 1;
```

## Customizing Quality Gates

Edit `model_evaluation.py`:

```python
# Deployment thresholds
ABSOLUTE_MIN_ACCURACY = 0.75  # Must be at least 75% accurate
MIN_PER_CROP_F1 = 0.70        # Each crop must have F1 > 0.70
IMPROVEMENT_MARGIN = 0.02      # Challenger must beat champion by 2%
```

**Recommendations:**
- Start with **loose gates** (75% accuracy, +2% improvement)
- **Tighten gradually** as you collect more data
- **Monitor blocked deployments** to find appropriate thresholds
- Consider **seasonal adjustments** for crop-specific gates

## Troubleshooting

### Challenger Always Gets Blocked

**Possible Causes:**
1. Gates are too strict
2. Not enough training data
3. Data quality issues
4. Model overfitting

**Solutions:**
```sql
-- Check recent blocked deployments
SELECT 
  deployment_time,
  accuracy,
  gates_failed
FROM `ml-pipeline-477612.crop_ml.deployment_history`
WHERE deployment_decision = 'blocked'
ORDER BY deployment_time DESC
LIMIT 5;
```

### Holdout Set Is Too Small

**Solution:**
```python
# Recreate with more data after collecting more samples
from model_evaluation import create_or_load_holdout_set
create_or_load_holdout_set(force_recreate=True)
```

### Model Performance Degrading

**Check:**
1. Data drift (new crop patterns)
2. Seasonal changes
3. CDL data quality
4. Satellite imagery availability

**Query:**
```sql
-- Track accuracy over time
SELECT 
  DATE(evaluation_time) as date,
  AVG(accuracy) as avg_accuracy
FROM `ml-pipeline-477612.crop_ml.model_performance`
WHERE model_type = 'champion'
GROUP BY date
ORDER BY date DESC;
```

## Best Practices

1. **Never modify holdout set** after creation
2. **Monitor gate pass rates** monthly
3. **Review blocked deployments** to tune thresholds
4. **Track per-crop performance** for seasonal adjustments
5. **Set up alerts** for deployment decisions
6. **Keep 6+ months of history** in BigQuery
7. **Document gate threshold changes** in deployment notes

## Next Steps

- Set up **Slack/email alerts** for deployment decisions
- Create **monitoring dashboard** in Looker/Data Studio
- Implement **A/B testing** for gradual rollouts
- Add **shadow mode** to compare predictions before full deployment
- Set up **automatic rollback** if production metrics degrade

