# Model Evaluation Guide

## Overview

The `carbon_check_field` ML pipeline now includes comprehensive model evaluation beyond basic TensorBoard metrics. The new evaluation module provides actionable insights for model performance analysis.

## What's Included

### 1. **Enhanced Confusion Matrix** (`confusion_matrix_enhanced.png`)

Three side-by-side visualizations:
- **Count + Percentage**: Raw counts with row-wise percentages
- **Percentage Only**: Easier-to-read percentage heatmap (0-100%)
- **Normalized (0-1)**: Normalized confusion matrix for academic reporting

**Why it's useful**: Quickly identify which crops are being confused with each other. Percentages make it easy to spot patterns even when class sizes differ.

---

### 2. **Per-Crop Metrics Chart** (`per_crop_metrics.png`)

Four subplots:
- **Grouped Bar Chart**: Precision, Recall, F1 side-by-side for each crop
- **Sorted F1 Scores**: Identifies best and worst-performing crops
- **Test Set Distribution**: Shows sample sizes per crop (important for interpreting metrics)
- **Precision vs Recall Scatter**: Reveals trade-offs (size = test sample count)

**Why it's useful**: See at a glance which crops need more training data or are inherently difficult to classify. The scatter plot reveals if the model is biased toward precision or recall.

---

### 3. **Feature Importance** (`feature_importance.png`, `feature_importance.json`)

Two plots:
- **Top 20 Features**: Bar chart of most important features for classification
- **Cumulative Importance**: How many features needed to reach 80%/90% of total importance

**Why it's useful**: 
- Understand which NDVI stats, elevation bins, or geographic features matter most
- Identify redundant features that can be removed
- Validate that engineered features (sin/cos, elevation bins) are actually useful

**JSON Output**: Full ranked list of all features with importance scores for further analysis.

---

### 4. **Misclassification Analysis** (`misclassification_analysis.png`, `misclassifications.json`)

- Bar chart of top 10 most common misclassification pairs (e.g., "Corn → Soybeans")
- Includes both count and percentage of total samples for each true class

**Why it's useful**: 
- Prioritize which crop pairs need better separation
- Informs feature engineering (e.g., if Corn/Soybeans are confused, maybe add temporal features to capture planting dates)
- Helps understand when model is "reasonably wrong" vs "wildly wrong"

**JSON Output**: Complete list of all misclassification pairs for programmatic analysis.

---

### 5. **Advanced Metrics Report** (`advanced_metrics.png`, `advanced_metrics.json`)

**Visualization**: Bar chart of 5 key metrics
**Metrics Included**:
- **Accuracy**: Overall correctness
- **Cohen's Kappa**: Agreement beyond chance (accounts for class imbalance)
- **Matthews Correlation Coefficient (MCC)**: Balanced metric even with class imbalance (-1 to +1)
- **Macro F1**: Unweighted average F1 (treats all crops equally)
- **Weighted F1**: Sample-weighted average F1

**Why it's useful**: 
- **Cohen's Kappa** and **MCC** are more reliable than accuracy when classes are imbalanced
- **Macro vs Weighted F1**: If they differ significantly, it means performance varies wildly across crops
- Great for reporting to stakeholders (more rigorous than just "accuracy")

**JSON Output**: Complete breakdown with per-crop metrics for export to reports/papers.

---

## How to Access Results

### During Training

The evaluation runs automatically during training and outputs are saved to:

```
gs://ml-pipeline-artifacts/models/[VERSION]/evaluation/
```

### After Training

1. **GCS Bucket**: Download from Cloud Storage
   ```bash
   gsutil -m cp -r gs://ml-pipeline-artifacts/models/latest/evaluation ./local_eval
   ```

2. **Vertex AI Artifacts**: Automatically attached to training jobs in Vertex AI
   - Go to: [Vertex AI Training](https://console.cloud.google.com/vertex-ai/training/training-pipelines)
   - Select your training job → "Artifacts" tab

3. **TensorBoard**: Basic metrics still available in TensorBoard
   - View: [Vertex AI TensorBoard](https://console.cloud.google.com/vertex-ai/tensorboard)

---

## Interpreting Results

### Good Model Characteristics

 **Confusion Matrix**: Diagonal is bright, off-diagonals are dark  
 **Per-Crop Metrics**: All F1 scores > 0.85  
 **Feature Importance**: Top features make sense (e.g., NDVI stats dominate)  
 **Misclassifications**: Few patterns, evenly distributed  
 **Advanced Metrics**: Cohen's Kappa > 0.8, MCC > 0.8

### Warning Signs

️ **Confusion Matrix**: One crop predicts everything or gets predicted as everything  
️ **Per-Crop Metrics**: One crop has F1 << 0.7 (needs more/better data)  
️ **Feature Importance**: Top feature has >50% importance (model relies too heavily on one signal)  
️ **Misclassifications**: One pair dominates (e.g., 80% of errors are Corn → Soybeans)  
️ **Advanced Metrics**: Accuracy is high but Cohen's Kappa is low (model is just guessing majority class)

---

## Example Workflow

### Scenario: Model shows 95% accuracy but Cohen's Kappa is only 0.65

**Investigation Steps**:

1. **Check Per-Crop Metrics**: 
   - Result: Cotton has F1=0.30, others are >0.95
   - **Diagnosis**: Class imbalance - Cotton is under-represented

2. **Check Test Set Distribution**:
   - Result: Cotton has 2 test samples, others have 40+
   - **Action**: Collect more Cotton samples (increase `num_fields_per_crop` in `config.yaml`)

3. **Check Misclassifications**:
   - Result: Cotton is always predicted as "Corn"
   - **Action**: Add more distinctive features (e.g., Cotton has different seasonal NDVI patterns)

4. **Check Feature Importance**:
   - Result: `ndvi_mean` accounts for 60% of importance
   - **Action**: Add temporal features (early vs late season NDVI difference) to better separate crops

---

## Comparison to TensorBoard

| Metric Type | TensorBoard | New Evaluation Module |
|-------------|-------------|----------------------|
| **Confusion Matrix** | Basic heatmap | 3 views (count, %, normalized) |
| **Per-Crop Metrics** | Scalars (hard to compare) | Visual charts + scatter plots |
| **Feature Importance** |  Not available |  Top 20 + cumulative |
| **Misclassification Analysis** |  Not available |  Ranked pairs with % |
| **Advanced Metrics** |  Not available |  Cohen's Kappa, MCC |
| **Export Format** | TFEvents (proprietary) | PNG + JSON (portable) |
| **Usefulness** | Good for training curves | **Better for model diagnostics** |

**Recommendation**: Use TensorBoard for monitoring training progress over time (if you run multiple experiments). Use the new evaluation module for deep-dive analysis after training completes.

---

## Files Generated

```
evaluation/
├── confusion_matrix_enhanced.png    # 3-panel confusion matrix
├── per_crop_metrics.png             # 4-panel per-crop analysis
├── feature_importance.png           # Top features + cumulative
├── feature_importance.json          # All features ranked (programmatic)
├── misclassification_analysis.png   # Top 10 error pairs
├── misclassifications.json          # All error pairs (programmatic)
├── advanced_metrics.png             # 5 key metrics bar chart
└── advanced_metrics.json            # Full metrics report (export-ready)
```

---

## Next Steps

1. **Retrain with New Data**: Trigger a training run to test the new evaluation
   ```bash
   curl -X POST https://ml-pipeline-[PROJECT_ID].us-central1.run.app/train
   ```

2. **Download Results**: Wait ~15-20 minutes, then download evaluation artifacts
   ```bash
   gsutil ls gs://ml-pipeline-artifacts/models/latest/evaluation/
   gsutil cp -r gs://ml-pipeline-artifacts/models/latest/evaluation/ ./results/
   ```

3. **Analyze**: Open the PNG files and JSON reports

4. **Iterate**: Based on findings, adjust:
   - **Data collection**: Increase samples for low-performing crops
   - **Feature engineering**: Add/remove features based on importance
   - **Hyperparameters**: Tune based on precision/recall trade-offs

---

## Questions?

If the evaluation module reveals unexpected patterns or you need help interpreting results, check:
- **Confusion Matrix**: Which crops are confused?
- **Feature Importance**: Are the top features reasonable?
- **Test Set Distribution**: Do all crops have enough test samples (>20)?

Most issues stem from class imbalance or insufficient training data for specific crops.

