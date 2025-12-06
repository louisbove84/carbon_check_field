# TensorBoard Double Slash Issue - Resolution

## Problem

Files were being uploaded to GCS successfully, but TensorBoard UI showed empty IMAGES tab.

### Root Cause

**Double slash in GCS path:**
```
gs://ml-pipeline-477612-training/training_output/logs//events.out.tfevents...
                                                      ^^
                                                   Double slash!
```

This happened because:
1. Vertex AI sets `AIP_TENSORBOARD_LOG_DIR` to `gs://bucket/training_output/logs/` (with trailing slash)
2. Training code extracted the prefix: `training_output/logs/`
3. Code concatenated: `f"{gcs_prefix}/{relative_path}"` → `training_output/logs//file.tfevents`

### Why It Breaks TensorBoard

- TensorBoard indexes files based on their full GCS path
- The double slash creates an invalid or unexpected path structure
- TensorBoard's file watcher doesn't recognize the files as valid event logs
- Files exist (802 KB uploaded) but UI can't display them

## Fix

### Code Change

**Before:**
```python
if gcs_prefix:
    gcs_blob_path = f"{gcs_prefix}/{relative_path}".replace('\\', '/')
```

**After:**
```python
if gcs_prefix:
    # Remove trailing slash from prefix to avoid double slashes
    clean_prefix = gcs_prefix.rstrip('/')
    gcs_blob_path = f"{clean_prefix}/{relative_path}".replace('\\', '/')
```

### Files Changed

- `/ml_pipeline/trainer/train.py` (line 441)

## Verification

### Before Fix
```bash
$ gsutil ls -lh gs://ml-pipeline-477612-training/training_output/logs//
802.09 KiB  2025-12-06T12:46:50Z  gs://...logs//events.out.tfevents...
                                           ^^
```

### After Fix (Expected)
```bash
$ gsutil ls -lh gs://ml-pipeline-477612-training/training_output/logs/
802.09 KiB  2025-12-06T13:17:50Z  gs://...logs/events.out.tfevents...
                                          ^
```

## Timeline

| Time  | Training Job | Issue |
|-------|--------------|-------|
| 12:04 | `crop-training-20251206_1204` | gsutil not found |
| 12:41 | `crop-training-20251206_1241` | Double slash in path |
| 13:07 | `crop-training-20251206_1307` | **Fixed** ✅ |

## Testing

After training completes (~13:17):

1. **Check GCS path:**
   ```bash
   gsutil ls -lhr gs://ml-pipeline-477612-training/training_output/logs/
   ```
   - Should NOT see double slashes
   - Should see files around 800+ KB

2. **Check TensorBoard:**
   - Open: [TensorBoard Instance](https://console.cloud.google.com/vertex-ai/tensorboard/instances/8764605208211750912?project=ml-pipeline-477612)
   - Select experiment: `crop-training-20251206_1307-custom-job`
   - Click **IMAGES** tab
   - Should see:
     - `confusion_matrix/counts_and_percent`
     - `confusion_matrix/percentage`
     - `confusion_matrix/normalized`
     - `feature_importance/top_10`
     - `feature_importance/all_features`
     - `per_crop_metrics/precision_recall_f1`
     - `misclassifications/error_matrix`

3. **Check logs:**
   ```bash
   gcloud logging read "resource.type=ml_job AND timestamp>='2025-12-06T13:07:00Z'" \
     --limit=100 --format="value(textPayload)" --project=ml-pipeline-477612 | \
     grep -E "(Uploaded|TensorBoard)"
   ```
   - Should see: `✅ Uploaded 15+ TensorBoard log files`
   - Should NOT see double slashes in paths

## Related Issues

This fix also addresses:
- [TENSORBOARD_DEBUGGING_SUMMARY.md](./TENSORBOARD_DEBUGGING_SUMMARY.md) - Previous gsutil issue
- [TENSORBOARD_VIEWING_GUIDE.md](./TENSORBOARD_VIEWING_GUIDE.md) - How to view results

## Lessons Learned

1. **Always strip trailing slashes** when building paths programmatically
2. **Check GCS paths visually** - double slashes are easy to miss in logs
3. **TensorBoard is sensitive to path structure** - files must be in expected format
4. **Upload success ≠ TensorBoard success** - files can upload but not be indexed

