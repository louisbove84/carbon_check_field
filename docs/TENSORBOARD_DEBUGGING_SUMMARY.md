# TensorBoard Upload Debugging - Issue Resolution

## Timeline of Issues and Fixes

### Issue 1: `gsutil` Command Not Found ❌

**Problem:**
```
ERROR: [Errno 2] No such file or directory: 'gsutil'
```

**Root Cause:**
- Training container based on `python:3.11-slim`
- Google Cloud SDK (which includes `gsutil`) not installed
- Code was trying to call `subprocess.run(['gsutil', ...])`

**Fix:**
- Replaced `gsutil` subprocess call with Python `google.cloud.storage` library
- Already installed via `google-cloud-storage` dependency
- Used `storage.Client()` to upload files programmatically

### Issue 2: Zero Files Uploaded ❌

**Problem:**
```
✅ Uploaded 0 TensorBoard log files to gs://...
```

**Root Cause:**
- `AIP_MODEL_DIR` environment variable set by Vertex AI can be a **GCS path**
- Example: `AIP_MODEL_DIR=gs://bucket/training_output/model`
- Code was using: `local_tensorboard_dir = os.path.join(output_dir, 'tensorboard_logs')`
- This resulted in: `gs://bucket/training_output/model/tensorboard_logs` (a GCS path!)
- `SummaryWriter` cannot write directly to GCS paths
- GCS filesystem library (`gcsfs`) showed warnings: `"Append mode 'a' is not supported in GCS"`
- Files were never created, so there was nothing to upload

**Fix:**
- Changed to **hardcoded local path**: `/tmp/tensorboard_logs`
- Now `SummaryWriter` always writes to a true local filesystem
- Files are created successfully
- Then uploaded to GCS using `storage.Client()`

## Final Working Solution

```python
# ✅ CORRECT: Always use a local path for TensorBoard
local_tensorboard_dir = '/tmp/tensorboard_logs'
os.makedirs(local_tensorboard_dir, exist_ok=True)

# Create writer (writes to local filesystem)
writer = SummaryWriter(log_dir=local_tensorboard_dir)

# ... log metrics ...

writer.close()

# Upload to GCS using Python storage client
if tensorboard_gcs_path:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for root, dirs, files in os.walk(local_tensorboard_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_tensorboard_dir)
            gcs_blob_path = f"{gcs_prefix}/{relative_path}".replace('\\', '/')
            
            blob = bucket.blob(gcs_blob_path)
            blob.upload_from_filename(local_file_path)
```

## Key Learnings

1. **Never assume environment variables are local paths** - Vertex AI sets many env vars to GCS paths
2. **`SummaryWriter` requires local filesystem** - It cannot write to `gs://` paths
3. **Use Python libraries over CLI tools** - More reliable in containerized environments
4. **Add verbose logging** - Critical for debugging in remote training environments
5. **Test assumptions** - The `AIP_MODEL_DIR` assumption was wrong

## Verification Steps

After training completes, verify:

1. **Check logs for file count:**
   ```
   ✅ Uploaded X TensorBoard log files  # Should be > 0
   ```

2. **Check GCS directly:**
   ```bash
   gsutil ls -lh gs://ml-pipeline-477612-training/training_output/logs/
   ```

3. **View in TensorBoard UI:**
   - Navigate to Vertex AI → TensorBoard
   - Should see new experiment with:
     - Confusion matrices under Images tab
     - Feature importance charts
     - Per-crop metrics (5 data points for progression)
     - Advanced metrics (Cohen's Kappa, Matthews Correlation)

## Training Job Timeline

- **12:04** - Training job (with gsutil issue)
- **12:41** - Training job (with local path fix) ← **Current, in progress**
- Expected completion: **~12:52**

