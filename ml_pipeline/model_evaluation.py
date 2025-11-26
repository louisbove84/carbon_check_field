"""
Model Evaluation & Champion/Challenger Comparison
==================================================
Evaluates trained models and determines if a new model (challenger)
should replace the current production model (champion).

Key Features:
- Permanent holdout test set for unbiased evaluation
- Champion vs Challenger comparison
- Quality gates for deployment decisions
- Metrics tracking in BigQuery
"""

import logging
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from google.cloud import bigquery, storage, aiplatform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "ml-pipeline-477612"
REGION = "us-central1"
BUCKET_NAME = "carboncheck-data"
DATASET_ID = "crop_ml"
TRAINING_TABLE_ID = "training_features"
METRICS_TABLE_ID = "model_performance"
HOLDOUT_TABLE_ID = "holdout_test_set"

# Deployment thresholds
ABSOLUTE_MIN_ACCURACY = 0.75  # Must be at least 75% accurate
MIN_PER_CROP_F1 = 0.70  # Each crop must have F1 > 0.70
IMPROVEMENT_MARGIN = 0.02  # Challenger must beat champion by 2%
MIN_TEST_SAMPLES = 50  # Minimum holdout samples required

# Feature columns (must match training)
BASE_FEATURE_COLUMNS = [
    'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max',
    'ndvi_p25', 'ndvi_p50', 'ndvi_p75',
    'ndvi_early', 'ndvi_late',
    'elevation_m', 'longitude', 'latitude'
]

# ============================================================
# HOLDOUT TEST SET MANAGEMENT
# ============================================================

def create_or_load_holdout_set(force_recreate: bool = False) -> pd.DataFrame:
    """
    Create a permanent holdout test set or load existing one.
    This ensures consistent evaluation across model versions.
    
    Args:
        force_recreate: If True, recreate holdout set from scratch
    
    Returns:
        DataFrame with holdout test samples
    """
    logger.info("📊 Managing holdout test set...")
    
    client = bigquery.Client(project=PROJECT_ID)
    holdout_table_ref = f"{PROJECT_ID}.{DATASET_ID}.{HOLDOUT_TABLE_ID}"
    
    # Check if holdout set exists
    try:
        query = f"SELECT COUNT(*) as count FROM `{holdout_table_ref}`"
        result = client.query(query).result()
        existing_count = list(result)[0].count
        
        if existing_count >= MIN_TEST_SAMPLES and not force_recreate:
            logger.info(f"✅ Using existing holdout set: {existing_count} samples")
            
            # Load holdout set
            query = f"""
            SELECT *
            FROM `{holdout_table_ref}`
            WHERE ndvi_mean IS NOT NULL
            """
            return client.query(query).to_dataframe()
    
    except Exception as e:
        logger.info(f"   Holdout set not found, creating new one: {e}")
    
    # Create new holdout set (20% of all data, stratified by crop)
    logger.info("🔨 Creating new holdout test set (20% of all data)...")
    
    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.{TRAINING_TABLE_ID}`
    WHERE ndvi_mean IS NOT NULL
    """
    
    df_all = client.query(query).to_dataframe()
    
    # Stratified sample: 20% from each crop
    holdout_samples = []
    for crop in df_all['crop'].unique():
        crop_df = df_all[df_all['crop'] == crop]
        n_holdout = int(len(crop_df) * 0.20)
        holdout_crop = crop_df.sample(n=n_holdout, random_state=42)
        holdout_samples.append(holdout_crop)
        logger.info(f"   • {crop}: {n_holdout} samples reserved for testing")
    
    df_holdout = pd.concat(holdout_samples, ignore_index=True)
    
    # Create holdout table if it doesn't exist
    schema = [
        bigquery.SchemaField("field_id", "STRING"),
        bigquery.SchemaField("crop", "STRING"),
        bigquery.SchemaField("crop_code", "INTEGER"),
        bigquery.SchemaField("sample_id", "STRING"),
        bigquery.SchemaField("collection_date", "TIMESTAMP"),
        bigquery.SchemaField("cdl_code", "INTEGER"),
        # NDVI features
        bigquery.SchemaField("ndvi_mean", "FLOAT"),
        bigquery.SchemaField("ndvi_std", "FLOAT"),
        bigquery.SchemaField("ndvi_min", "FLOAT"),
        bigquery.SchemaField("ndvi_max", "FLOAT"),
        bigquery.SchemaField("ndvi_p25", "FLOAT"),
        bigquery.SchemaField("ndvi_p50", "FLOAT"),
        bigquery.SchemaField("ndvi_p75", "FLOAT"),
        bigquery.SchemaField("ndvi_early", "FLOAT"),
        bigquery.SchemaField("ndvi_late", "FLOAT"),
        bigquery.SchemaField("elevation_m", "FLOAT"),
        bigquery.SchemaField("longitude", "FLOAT"),
        bigquery.SchemaField("latitude", "FLOAT"),
        bigquery.SchemaField("reserved_date", "TIMESTAMP"),
    ]
    
    # Add timestamp
    df_holdout['reserved_date'] = datetime.now()
    
    # Upload to BigQuery
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition='WRITE_TRUNCATE'
    )
    
    job = client.load_table_from_dataframe(
        df_holdout, holdout_table_ref, job_config=job_config
    )
    job.result()
    
    logger.info(f"✅ Holdout set created: {len(df_holdout)} samples")
    logger.info(f"   Saved to: {holdout_table_ref}")
    
    return df_holdout


def get_training_set_excluding_holdout() -> pd.DataFrame:
    """
    Get training data excluding the holdout test set.
    This ensures no data leakage.
    
    Returns:
        DataFrame with training samples only
    """
    logger.info("📥 Loading training data (excluding holdout)...")
    
    client = bigquery.Client(project=PROJECT_ID)
    
    query = f"""
    SELECT t.*
    FROM `{PROJECT_ID}.{DATASET_ID}.{TRAINING_TABLE_ID}` t
    LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.{HOLDOUT_TABLE_ID}` h
        ON t.sample_id = h.sample_id
    WHERE h.sample_id IS NULL
        AND t.ndvi_mean IS NOT NULL
    """
    
    df = client.query(query).to_dataframe()
    
    logger.info(f"✅ Loaded {len(df)} training samples")
    logger.info(f"   Crops: {df['crop'].value_counts().to_dict()}")
    
    return df


# ============================================================
# MODEL EVALUATION
# ============================================================

def engineer_features_for_eval(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features (must match training pipeline).
    
    Args:
        df: DataFrame with base features
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    df['ndvi_range'] = df['ndvi_max'] - df['ndvi_min']
    df['ndvi_iqr'] = df['ndvi_p75'] - df['ndvi_p25']
    df['ndvi_change'] = df['ndvi_late'] - df['ndvi_early']
    df['ndvi_early_ratio'] = df['ndvi_early'] / (df['ndvi_mean'] + 0.001)
    df['ndvi_late_ratio'] = df['ndvi_late'] / (df['ndvi_mean'] + 0.001)
    
    return df


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, Any]:
    """
    Evaluate model on test set and compute comprehensive metrics.
    
    Args:
        model: Trained sklearn pipeline
        X_test: Test features
        y_test: Test labels
        model_name: Model identifier (e.g., "champion", "challenger")
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"🧪 Evaluating {model_name}...")
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=sorted(y_test.unique())
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
    
    # Detailed report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Compile metrics
    crops = sorted(y_test.unique())
    per_crop_metrics = {}
    
    for i, crop in enumerate(crops):
        per_crop_metrics[crop] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'per_crop_metrics': per_crop_metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'test_samples': len(X_test),
        'evaluation_time': datetime.now().isoformat()
    }
    
    # Log results
    logger.info(f"   Accuracy: {accuracy:.2%}")
    for crop, m in per_crop_metrics.items():
        logger.info(f"   • {crop}: F1={m['f1_score']:.2%}, n={m['support']}")
    
    return metrics


def load_model_from_gcs(model_path: str):
    """
    Load model from GCS.
    
    Args:
        model_path: GCS path (gs://bucket/path/model.joblib)
    
    Returns:
        Loaded sklearn pipeline
    """
    logger.info(f"📦 Loading model from {model_path}")
    
    # Parse GCS path
    parts = model_path.replace('gs://', '').split('/')
    bucket_name = parts[0]
    blob_path = '/'.join(parts[1:])
    
    # Download model
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    # Load to memory
    model_bytes = blob.download_as_bytes()
    model = joblib.loads(model_bytes)
    
    logger.info("✅ Model loaded")
    return model


def get_current_champion_path() -> Optional[str]:
    """
    Get the GCS path of the current production model (champion).
    
    Returns:
        GCS path or None if no champion exists
    """
    try:
        # Try to get latest deployed model from Vertex AI
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        endpoints = aiplatform.Endpoint.list(filter='display_name="crop-endpoint"')
        
        if endpoints and endpoints[0].list_models():
            # Champion is currently deployed
            return f'gs://{BUCKET_NAME}/models/crop_classifier_latest/model.joblib'
        
        return None
    
    except Exception as e:
        logger.warning(f"⚠️  Could not find champion model: {e}")
        return None


# ============================================================
# CHAMPION VS CHALLENGER COMPARISON
# ============================================================

def compare_models_and_decide(
    challenger_metrics: Dict[str, Any],
    champion_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compare challenger vs champion and make deployment decision.
    
    Args:
        challenger_metrics: Metrics for new trained model
        champion_metrics: Metrics for current production model (None if first model)
    
    Returns:
        Decision dictionary with 'should_deploy' boolean and reasoning
    """
    logger.info("⚖️  Comparing models and making deployment decision...")
    
    decision = {
        'should_deploy': False,
        'reasoning': [],
        'gates_passed': [],
        'gates_failed': [],
        'challenger_metrics': challenger_metrics,
        'champion_metrics': champion_metrics,
        'decision_time': datetime.now().isoformat()
    }
    
    challenger_acc = challenger_metrics['accuracy']
    
    # Gate 1: Absolute minimum accuracy
    if challenger_acc >= ABSOLUTE_MIN_ACCURACY:
        decision['gates_passed'].append(
            f"✅ Accuracy {challenger_acc:.2%} >= minimum {ABSOLUTE_MIN_ACCURACY:.0%}"
        )
    else:
        decision['gates_failed'].append(
            f"❌ Accuracy {challenger_acc:.2%} < minimum {ABSOLUTE_MIN_ACCURACY:.0%}"
        )
    
    # Gate 2: Per-crop F1 scores
    all_crops_pass = True
    for crop, metrics in challenger_metrics['per_crop_metrics'].items():
        f1 = metrics['f1_score']
        if f1 >= MIN_PER_CROP_F1:
            decision['gates_passed'].append(
                f"✅ {crop} F1={f1:.2%} >= {MIN_PER_CROP_F1:.0%}"
            )
        else:
            decision['gates_failed'].append(
                f"❌ {crop} F1={f1:.2%} < {MIN_PER_CROP_F1:.0%}"
            )
            all_crops_pass = False
    
    # Gate 3: Beat champion (if exists)
    if champion_metrics:
        champion_acc = champion_metrics['accuracy']
        improvement = challenger_acc - champion_acc
        
        if improvement >= IMPROVEMENT_MARGIN:
            decision['gates_passed'].append(
                f"✅ Beats champion by {improvement:.2%} (>= {IMPROVEMENT_MARGIN:.0%})"
            )
        else:
            decision['gates_failed'].append(
                f"❌ Only beats champion by {improvement:.2%} (need {IMPROVEMENT_MARGIN:.0%})"
            )
        
        decision['reasoning'].append(
            f"Champion accuracy: {champion_acc:.2%}, "
            f"Challenger accuracy: {challenger_acc:.2%}, "
            f"Improvement: {improvement:+.2%}"
        )
    else:
        decision['gates_passed'].append("✅ No champion exists (first deployment)")
        decision['reasoning'].append("First model deployment - no champion to compare")
    
    # Final decision
    if len(decision['gates_failed']) == 0:
        decision['should_deploy'] = True
        decision['reasoning'].append("🎉 All gates passed - deploying challenger!")
    else:
        decision['should_deploy'] = False
        decision['reasoning'].append(
            f"⛔ Deployment blocked - {len(decision['gates_failed'])} gate(s) failed"
        )
    
    # Log decision
    logger.info("=" * 60)
    logger.info("DEPLOYMENT DECISION")
    logger.info("=" * 60)
    for gate in decision['gates_passed']:
        logger.info(gate)
    for gate in decision['gates_failed']:
        logger.warning(gate)
    for reason in decision['reasoning']:
        logger.info(reason)
    logger.info("=" * 60)
    
    return decision


# ============================================================
# METRICS STORAGE
# ============================================================

def save_metrics_to_bigquery(metrics: Dict[str, Any], model_type: str) -> None:
    """
    Save model metrics to BigQuery for tracking.
    
    Args:
        metrics: Model evaluation metrics
        model_type: "champion" or "challenger"
    """
    logger.info(f"💾 Saving {model_type} metrics to BigQuery...")
    
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{METRICS_TABLE_ID}"
    
    # Flatten metrics for BigQuery
    row = {
        'model_type': model_type,
        'model_name': metrics['model_name'],
        'accuracy': metrics['accuracy'],
        'test_samples': metrics['test_samples'],
        'evaluation_time': metrics['evaluation_time'],
        'metrics_json': json.dumps(metrics),
    }
    
    # Add per-crop F1 scores as columns
    for crop, crop_metrics in metrics['per_crop_metrics'].items():
        row[f'{crop.lower().replace(" ", "_")}_f1'] = crop_metrics['f1_score']
    
    # Insert row
    errors = client.insert_rows_json(table_ref, [row])
    
    if errors:
        logger.error(f"❌ BigQuery insert failed: {errors}")
    else:
        logger.info(f"✅ Metrics saved to {table_ref}")


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================

def evaluate_and_decide(challenger_model_path: str) -> Dict[str, Any]:
    """
    Complete evaluation pipeline:
    1. Load/create holdout test set
    2. Evaluate challenger model
    3. Evaluate champion model (if exists)
    4. Compare and make deployment decision
    5. Save metrics to BigQuery
    
    Args:
        challenger_model_path: GCS path to new trained model
    
    Returns:
        Decision dictionary
    """
    logger.info("=" * 60)
    logger.info("🎯 MODEL EVALUATION & DEPLOYMENT DECISION")
    logger.info("=" * 60)
    
    # Step 1: Get holdout test set
    df_holdout = create_or_load_holdout_set(force_recreate=False)
    
    if len(df_holdout) < MIN_TEST_SAMPLES:
        raise Exception(
            f"Insufficient holdout samples: {len(df_holdout)} < {MIN_TEST_SAMPLES}"
        )
    
    # Engineer features
    df_holdout = engineer_features_for_eval(df_holdout)
    
    # Prepare test data
    feature_cols = BASE_FEATURE_COLUMNS + [
        'ndvi_range', 'ndvi_iqr', 'ndvi_change',
        'ndvi_early_ratio', 'ndvi_late_ratio'
    ]
    
    X_test = df_holdout[feature_cols]
    y_test = df_holdout['crop']
    
    # Step 2: Evaluate challenger
    challenger_model = load_model_from_gcs(challenger_model_path)
    challenger_metrics = evaluate_model(
        challenger_model, X_test, y_test, "challenger"
    )
    
    # Step 3: Evaluate champion (if exists)
    champion_metrics = None
    champion_path = get_current_champion_path()
    
    if champion_path:
        try:
            champion_model = load_model_from_gcs(champion_path)
            champion_metrics = evaluate_model(
                champion_model, X_test, y_test, "champion"
            )
        except Exception as e:
            logger.warning(f"⚠️  Could not evaluate champion: {e}")
    
    # Step 4: Compare and decide
    decision = compare_models_and_decide(challenger_metrics, champion_metrics)
    
    # Step 5: Save metrics
    save_metrics_to_bigquery(challenger_metrics, "challenger")
    if champion_metrics:
        save_metrics_to_bigquery(champion_metrics, "champion")
    
    return decision


if __name__ == '__main__':
    # For local testing
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = f'gs://{BUCKET_NAME}/models/crop_classifier_latest/model.joblib'
    
    decision = evaluate_and_decide(model_path)
    print(json.dumps(decision, indent=2, default=str))

