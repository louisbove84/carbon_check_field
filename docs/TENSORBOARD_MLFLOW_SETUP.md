# TensorBoard & MLflow Setup

## Overview

We've implemented both TensorBoard and MLflow for experiment tracking and visualization. This allows you to compare both tools and choose which works best for your workflow.

## TensorBoard Fixes

### Image Display Issue
The titles of images were appearing in TensorBoard but the actual images were not displaying. We've made the following improvements:

1. **Temporary File Method**: Changed from `BytesIO` to temporary file approach for more reliable image handling on GCP
2. **RGBA to RGB Conversion**: Improved handling of RGBA images by creating a white background
3. **Image Format Validation**: Added better validation and error handling for image conversion

### Current Status
- Images are being logged correctly (logs confirm this)
- Files are being uploaded to GCS successfully
- TensorBoard on GCP may still have display issues (this is a known limitation)

## MLflow Setup

### Why MLflow?
MLflow provides:
- **Better artifact handling**: Images are stored as files, not embedded in event logs
- **Easier debugging**: Can view artifacts directly in GCS or MLflow UI
- **More reliable**: Less prone to format issues
- **Better UI**: More intuitive interface for viewing results

### Implementation

1. **MLflow Tracking**: Uses local file store (`/tmp/mlruns`) during training
2. **Artifact Storage**: Artifacts are uploaded to GCS after training completes
3. **Parallel Logging**: Both TensorBoard and MLflow log the same metrics/visualizations

### What Gets Logged to MLflow

- **Hyperparameters**: Model configuration (n_estimators, max_depth, etc.)
- **Metrics**: Accuracy, precision, recall, F1-scores
- **Visualizations**:
  - Confusion matrices (3 views)
  - Per-crop metrics comparison
  - Feature importance charts
  - Overall metrics summary
- **Model Artifacts**: Trained model saved as MLflow model

## Usage

### During Training

Both TensorBoard and MLflow logging happen automatically during training. The training script will:

1. Log to TensorBoard (for Vertex AI TensorBoard integration)
2. Log to MLflow (for better artifact handling)
3. Upload both to GCS after training completes

### Viewing Results

#### TensorBoard (GCP)
1. Go to Vertex AI → TensorBoard in GCP Console
2. Select your TensorBoard instance
3. View experiments and runs
4. Check the "IMAGES" tab for visualizations

#### MLflow (Local or GCS)
1. **Local**: MLflow artifacts are in `/tmp/mlruns/` on the training container
2. **GCS**: After training, artifacts are uploaded to `gs://{bucket}/mlflow/`
3. **View in MLflow UI**:
   ```bash
   # Download MLflow artifacts from GCS
   gsutil -m cp -r gs://{bucket}/mlflow /tmp/
   
   # Start MLflow UI
   mlflow ui --backend-store-uri file:///tmp/mlflow
   ```

### Comparing Results

Both tools log the same information, so you can:
- Use TensorBoard for scalar metrics and progression
- Use MLflow for images and artifacts
- Compare which tool displays your data better

## File Locations

### TensorBoard
- **Local**: `/tmp/tensorboard_logs/` (during training)
- **GCS**: `gs://{bucket}/training_output/logs/` (after upload)
- **Vertex AI**: Automatically synced to TensorBoard instance

### MLflow
- **Local**: `/tmp/mlruns/` (during training)
- **GCS**: `gs://{bucket}/mlflow/` (after upload)

## Troubleshooting

### TensorBoard Images Not Showing
1. Check that files were uploaded to GCS
2. Verify the TensorBoard instance is reading from the correct path
3. Try refreshing the TensorBoard UI
4. Check logs for any upload errors

### MLflow Artifacts Not Uploading
1. Check GCS permissions for the service account
2. Verify the bucket exists and is accessible
3. Check training logs for upload errors

### Viewing MLflow Locally
If you want to view MLflow results locally:

```bash
# Download from GCS
gsutil -m cp -r gs://carboncheck-data/mlflow /tmp/mlflow

# Start MLflow UI
mlflow ui --backend-store-uri file:///tmp/mlflow
```

Then open `http://localhost:5000` in your browser.

## Next Steps

1. **Run a training job** to test both systems
2. **Compare the results** in both TensorBoard and MLflow
3. **Choose your preferred tool** based on which displays your data better
4. **Consider using both**: TensorBoard for scalars, MLflow for artifacts

## Configuration

Both tools use the same configuration from `config.yaml`:
- Storage bucket: `config['storage']['bucket']`
- Model parameters: `config['model']`

No additional configuration needed - both tools are set up automatically during training.

