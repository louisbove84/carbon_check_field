# Standalone Model Evaluation

## Overview

The `local_evaluation.py` script allows you to run comprehensive model evaluation **without retraining or data collection**. This is useful for:

- ✅ Quickly generating confusion matrices and metrics for existing models
- ✅ Comparing different model versions
- ✅ Debugging model performance issues
- ✅ Generating reports for stakeholders

## Quick Start

### Evaluate Latest Model from GCS

```bash
cd ml_pipeline/trainer
python local_evaluation.py
```

This will:
1. Load the latest model from `gs://carboncheck-data/models/crop_classifier_latest`
2. Load test data from BigQuery
3. Generate all evaluation artifacts in `./evaluation_output/`

### Evaluate Specific Model Version

```bash
python local_evaluation.py \
  --model-path models/crop_classifier_archive/crop_classifier_20251205_2255 \
  --output-dir ./results_v1
```

### Evaluate Local Model

```bash
python local_evaluation.py \
  --model-path ./my_local_model \
  --local \
  --output-dir ./local_results
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path` | Path to model (GCS or local) | `models/crop_classifier_latest` |
| `--output-dir` | Directory for evaluation results | `./evaluation_output` |
| `--bucket` | GCS bucket name | From config.yaml |
| `--test-data` | Path to CSV with test data | Uses BigQuery |
| `--test-split` | Test split ratio (if using training data) | `0.2` |
| `--local` | Treat model-path as local (not GCS) | `False` |
| `--config` | Path to config.yaml | Auto-detect |

## Examples

### Example 1: Quick Evaluation

```bash
# Evaluate latest production model
python local_evaluation.py
```

**Output:**
```
📁 Results saved to: ./evaluation_output
✅ confusion_matrix_enhanced.png (153 KB)
✅ per_crop_metrics.png (164 KB)
✅ feature_importance.png (125 KB)
✅ misclassification_analysis.png (49 KB)
✅ advanced_metrics.json
```

### Example 2: Compare Two Model Versions

```bash
# Evaluate version 1
python local_evaluation.py \
  --model-path models/crop_classifier_archive/crop_classifier_20251205_2200 \
  --output-dir ./results_v1

# Evaluate version 2
python local_evaluation.py \
  --model-path models/crop_classifier_archive/crop_classifier_20251205_2255 \
  --output-dir ./results_v2

# Compare results
diff results_v1/advanced_metrics.json results_v2/advanced_metrics.json
```

### Example 3: Use Custom Test Data

```bash
# Export test data from BigQuery first
bq query --use_legacy_sql=false --format=csv \
  "SELECT * FROM \`ml-pipeline-477612.crop_ml.training_features\` LIMIT 100" \
  > test_data.csv

# Evaluate with custom test data
python local_evaluation.py \
  --model-path models/crop_classifier_latest \
  --test-data test_data.csv
```

### Example 4: Evaluate with Different Test Split

```bash
# Use 30% of data for testing (instead of default 20%)
python local_evaluation.py \
  --model-path models/crop_classifier_latest \
  --test-split 0.3
```

## Output Files

The script generates the same comprehensive evaluation artifacts as training:

```
evaluation_output/
├── confusion_matrix_enhanced.png      # 3-panel confusion matrix
├── per_crop_metrics.png               # 4-panel per-crop analysis
├── feature_importance.png             # Top 20 features + cumulative
├── feature_importance.json            # All features ranked
├── misclassification_analysis.png     # Top 10 error pairs
├── misclassifications.json            # All misclassifications
├── advanced_metrics.png               # 5 key metrics bar chart
└── advanced_metrics.json              # Complete metrics report
```

## Requirements

The script requires:
- ✅ Trained model saved as `model.joblib` (and optionally `feature_cols.json`)
- ✅ Access to BigQuery (for loading test data) OR a CSV file with test data
- ✅ Python packages: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `google-cloud-bigquery`, `google-cloud-storage`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Error: "Model file not found"

**Solution**: Check the model path. For GCS paths, ensure you have:
- Correct bucket name
- Correct path structure: `models/crop_classifier_latest/model.joblib`

**Debug**:
```bash
gsutil ls gs://carboncheck-data/models/crop_classifier_latest/
```

### Error: "Feature columns mismatch"

**Solution**: The model expects different features than your data. This can happen if:
- Model was trained with different feature engineering
- Config.yaml has different elevation quantiles

**Fix**: Ensure you're using the same `config.yaml` that was used during training, or load it from GCS:
```bash
python local_evaluation.py --config config/config.yaml
```

### Error: "No module named 'seaborn'"

**Solution**: Install missing dependencies:
```bash
pip install seaborn matplotlib scikit-learn pandas numpy
```

### Error: "Permission denied" (BigQuery/GCS)

**Solution**: Authenticate with Google Cloud:
```bash
gcloud auth application-default login
```

## Integration with Training Pipeline

The evaluation script uses the **same evaluation module** (`tensorboard_utils.py`) as the training pipeline, ensuring consistency. The only difference is:

- **Training**: Evaluation runs automatically after training
- **Standalone**: You can run evaluation anytime on any saved model

## Use Cases

### 1. **Quick Model Check**
After deploying a new model, quickly verify it's performing well:
```bash
python local_evaluation.py
open evaluation_output/confusion_matrix_enhanced.png
```

### 2. **Model Comparison**
Compare performance across multiple model versions:
```bash
for version in v1 v2 v3; do
  python local_evaluation.py \
    --model-path models/archive/model_$version \
    --output-dir results_$version
done
```

### 3. **Debugging Performance Issues**
If production metrics look off, evaluate the deployed model:
```bash
python local_evaluation.py \
  --model-path models/crop_classifier_latest \
  --output-dir debug_results
```

Then check:
- `misclassification_analysis.png` - Which crops are confused?
- `feature_importance.png` - Are the right features being used?
- `advanced_metrics.json` - Is Cohen's Kappa low? (indicates class imbalance issues)

### 4. **Stakeholder Reports**
Generate professional visualizations for presentations:
```bash
python local_evaluation.py \
  --model-path models/crop_classifier_latest \
  --output-dir stakeholder_report

# All PNG files are ready to include in presentations
```

## Performance

- **Time**: ~30-60 seconds (depends on test set size)
- **Data Loading**: ~5-10 seconds from BigQuery
- **Evaluation**: ~20-40 seconds (generating visualizations)
- **Output**: ~500 KB total (PNG + JSON files)

## Next Steps

After running evaluation:

1. **View Results**: Open the PNG files to see visualizations
2. **Check Metrics**: Review `advanced_metrics.json` for detailed numbers
3. **Compare**: If evaluating multiple models, compare the JSON files
4. **Action**: Based on findings, decide if model needs retraining or more data

---

**See Also**:
- `MODEL_EVALUATION_GUIDE.md` - How to interpret evaluation results
- `vertex_ai_training.py` - Training script (includes automatic evaluation)

