# Training Job Monitoring Guide

## Current Training Job

**Job:** `crop-training-20251206_1332-custom-job`  
**Started:** 2025-12-06 13:32:59 UTC  
**Expected Duration:** ~10 minutes  
**ETA:** ~13:43 UTC  

## Monitor Job Status

### Quick Status Check
```bash
gcloud ai custom-jobs list \
  --region=us-central1 \
  --project=ml-pipeline-477612 \
  --limit=1 \
  --sort-by=~create_time \
  --format="table(displayName,createTime,state,endTime)"
```

### Job States
- `JOB_STATE_PENDING` - Job queued, waiting for resources
- `JOB_STATE_RUNNING` - Training in progress
- `JOB_STATE_SUCCEEDED` - Completed successfully ✅
- `JOB_STATE_FAILED` - Error occurred ❌

## View Live Logs

### Cloud Console
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=ml-pipeline-477612

### Command Line
```bash
# Get recent logs
gcloud logging read "resource.type=ml_job AND timestamp>='2025-12-06T13:32:00Z'" \
  --limit=100 \
  --format="value(textPayload)" \
  --project=ml-pipeline-477612 | tail -50
```

### Key Log Messages to Look For

✅ **Success Indicators:**
```
✅ Loaded 1227 training samples
✅ Created 19 features
✅ Test accuracy: 96.75%
📊 Logging confusion matrices to TensorBoard...
✅ Logged confusion matrices to TensorBoard
✅ Logged feature importance to TensorBoard
✅ SummaryWriter closed
📂 Local TensorBoard files created:
   /tmp/tensorboard_logs/events.out.tfevents... (800+ KB)
✅ Uploaded 15+ TensorBoard log files
```

❌ **Error Indicators:**
```
❌ Training failed
ModuleNotFoundError
Permission denied
Failed to upload
```

## After Training Completes

### 1. Verify Files Uploaded
```bash
# Check GCS for new files (no double slashes!)
gsutil ls -lh gs://ml-pipeline-477612-training/training_output/logs/

# Look for files from timestamp ~1765029179 (13:32 UTC)
# Should see:
# - events.out.tfevents.1765029179... (~800 KB)
# - Subdirectory with additional events
```

### 2. Check TensorBoard

**URL:** https://console.cloud.google.com/vertex-ai/tensorboard/instances/8764605208211750912?project=ml-pipeline-477612

**Steps:**
1. Select experiment: `crop-training-20251206_1332-custom-job`
2. Click **IMAGES** tab
3. Should see:
   - `confusion_matrix/counts_and_percent`
   - `confusion_matrix/percentage`  
   - `confusion_matrix/normalized`
   - `feature_importance/top_10`
   - `feature_importance/all_features`
   - `per_crop_metrics/precision_recall_f1`
   - `per_crop_metrics/support`
   - `misclassifications/error_matrix`

4. Click **SCALARS** tab
5. Should see 5 data points for each metric:
   - `evaluation/accuracy`
   - `evaluation/cohens_kappa`
   - `evaluation/matthews_corr`
   - `evaluation/macro_f1`
   - `evaluation/weighted_f1`
   - Per-crop metrics (precision, recall, f1_score)

6. Click **TEXT** tab
7. Should see `training_summary` with complete metrics

### 3. Verify GCS Path Structure

**Expected (No Double Slashes):**
```
gs://ml-pipeline-477612-training/training_output/logs/
├── events.out.tfevents.1765029179...  (~800 KB)
└── 1765029190.../
    └── events.out.tfevents...  (~700 B)
```

**Bad (Old Issue - Should NOT See):**
```
gs://ml-pipeline-477612-training/training_output/logs//
                                                      ^^
```

## Troubleshooting

### If Training Job Fails

1. **Check error logs:**
   ```bash
   gcloud logging read "resource.type=ml_job AND severity>=ERROR AND timestamp>='2025-12-06T13:32:00Z'" \
     --limit=20 \
     --project=ml-pipeline-477612
   ```

2. **Common issues:**
   - Out of memory → Increase machine_type in config.yaml
   - Permission denied → Check service account IAM roles
   - Module not found → Rebuild Docker container

### If TensorBoard IMAGES Tab is Empty

1. **Check if files exist:**
   ```bash
   gsutil ls -lh gs://ml-pipeline-477612-training/training_output/logs/
   ```

2. **Look for double slashes:**
   - If you see `logs//`, the fix didn't apply
   - Verify Docker container was rebuilt after fix

3. **Check file sizes:**
   - Main events file should be ~800 KB
   - If only a few bytes, no data was logged

4. **Hard refresh TensorBoard:**
   - Press Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)

### If Still Having Issues

```bash
# Run local evaluation to test TensorBoard logging
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline/trainer
python evaluate_model.py \
  --model-path gs://carboncheck-data/models/crop_classifier_latest/model.joblib \
  --output-dir ./test_tensorboard \
  --num-runs 2

# View locally
tensorboard --logdir=./test_tensorboard/tensorboard_logs
# Open: http://localhost:6006
```

## Timeline Reference

| Time (UTC) | Job | Status |
|------------|-----|--------|
| 12:04 | crop-training-20251206_1204 | ❌ gsutil not found |
| 12:41 | crop-training-20251206_1241 | ❌ Double slash issue |
| 13:07 | crop-training-20251206_1307 | ✅ Fixed (but used old container) |
| **13:32** | **crop-training-20251206_1332** | **🏃 RUNNING (latest fixes)** |

## Success Criteria

Training is successful when:
- ✅ Job state: `JOB_STATE_SUCCEEDED`
- ✅ Test accuracy: ~96-97%
- ✅ TensorBoard files uploaded: 15+ files, ~800 KB total
- ✅ GCS paths: NO double slashes
- ✅ TensorBoard IMAGES tab: 8+ visualizations visible
- ✅ TensorBoard SCALARS tab: 5 data points per metric
- ✅ Logs show: "✅ Uploaded X TensorBoard log files" (X > 0)

