# How to View TensorBoard Results in Vertex AI

## ⚠️ Common Mistake

**Problem:** "I only see test accuracy, no confusion matrices!"

**Solution:** You need to switch to the **IMAGES** tab!

## Where Everything Is Located

### 1. Scalars Tab (What you're seeing now)
- ✅ `evaluation/accuracy` - Overall accuracy
- ✅ `evaluation/cohens_kappa` - Cohen's Kappa score
- ✅ `evaluation/matthews_corr` - Matthews Correlation Coefficient  
- ✅ `evaluation/macro_f1` - Macro-averaged F1 score
- ✅ `evaluation/weighted_f1` - Weighted F1 score
- ✅ `per_crop/Alfalfa/precision` - Per-crop precision scores
- ✅ `per_crop/Alfalfa/recall` - Per-crop recall scores
- ✅ `per_crop/Alfalfa/f1_score` - Per-crop F1 scores
- ✅ `training/train_accuracy` - Training set accuracy
- ✅ `hparam/train_accuracy` - Hyperparameter metrics

### 2. **IMAGES Tab** (Click this! 📸)

This is where you'll find:

#### Confusion Matrices
- 🖼️ `confusion_matrix/counts_and_percent` - Matrix showing counts + percentages
- 🖼️ `confusion_matrix/percentage` - Color-coded percentage matrix
- 🖼️ `confusion_matrix/normalized` - Normalized values (0-1)

#### Per-Crop Detailed Metrics
- 🖼️ `per_crop_metrics/precision_recall_f1` - Bar charts for each crop
- 🖼️ `per_crop_metrics/support` - Sample counts per crop

#### Feature Importance
- 🖼️ `feature_importance/top_10` - Top 10 most important features
- 🖼️ `feature_importance/all_features` - All features ranked

#### Misclassification Analysis
- 🖼️ `misclassifications/error_matrix` - Heat map of common errors

### 3. Text Tab
- 📄 `training_summary` - Overall training summary with all metrics

### 4. Hparams Tab
- ⚙️ Hyperparameter comparison across runs

## Step-by-Step Instructions

1. **Open TensorBoard:**
   ```
   https://console.cloud.google.com/vertex-ai/tensorboard/instances/8764605208211750912?project=ml-pipeline-477612
   ```

2. **Select Latest Experiment:**
   - Look for experiment with recent timestamp (e.g., `2025-12-06 12:41`)
   - Click on it to open

3. **Check IMAGES Tab:**
   - Click "IMAGES" in the top navigation
   - You should see ALL the visualizations listed above
   - If you see NOTHING, the images weren't uploaded (see troubleshooting below)

4. **Check Scalars Tab:**
   - Click "SCALARS" in the top navigation
   - Expand categories on the left (evaluation/, per_crop/, etc.)
   - Should see multiple data points if `num_runs=5` worked

5. **Check Text Tab:**
   - Click "TEXT" in the top navigation
   - Look for `training_summary`
   - Should show complete metrics table

## Troubleshooting

### If IMAGES tab is empty:

1. **Check if files were uploaded:**
   ```bash
   gsutil ls -lh gs://ml-pipeline-477612-training/training_output/logs/
   ```
   - Should see files from recent timestamp
   - Should see multiple `.tfevents` files

2. **Check training logs:**
   - Look for: `✅ Uploaded X TensorBoard log files`
   - X should be > 0 (typically 10-20 files)

3. **Check for errors:**
   ```bash
   gcloud logging read "resource.type=ml_job AND severity>=ERROR AND timestamp>='2025-12-06T12:41:00Z'" --limit=50 --project=ml-pipeline-477612
   ```

### If you see old data:

1. **Hard refresh TensorBoard:**
   - Press `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows)
   - This clears the cache

2. **Check experiment timestamp:**
   - Make sure you selected the LATEST experiment
   - Experiments are listed by creation time

3. **Delete old experiments:**
   ```bash
   cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline/scripts
   python cleanup_tensorboard_experiments.py --keep-latest --execute
   ```

## Quick Test

Run this locally to verify TensorBoard is working:

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/ml_pipeline/trainer
python local_evaluation.py --model-path gs://carboncheck-data/models/crop_classifier_latest/model.joblib --output-dir ./test_tensorboard --num-runs 2

# Then view it:
tensorboard --logdir=./test_tensorboard/tensorboard_logs
# Open: http://localhost:6006
```

You should see:
- ✅ IMAGES tab with confusion matrices
- ✅ SCALARS tab with metrics (2 data points)
- ✅ TEXT tab with summary

