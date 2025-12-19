#!/usr/bin/env python3
"""
Test Training Script
====================
This script runs a test training job using a small subset of data.
It uses the same code from trainer/ directory but with:
- Limited BigQuery data (small sample)
- Smaller model (fewer estimators)
- Local TensorBoard logs (optional GCS upload)
- Local model saving (optional GCS upload)

This allows testing the full training pipeline without:
- Running expensive Vertex AI jobs
- Using large datasets
- Waiting for long training times

Usage:
    # Run test training locally
    python tools/test_training.py

    # Run with custom sample size
    python tools/test_training.py --sample-size 200

    # Upload TensorBoard logs to GCS
    python tools/test_training.py --upload-tensorboard

    # Upload model to GCS
    python tools/test_training.py --upload-model
"""

import os
import sys
import json
import logging
import joblib
import yaml
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

# Add trainer directory to path so we can import modules
trainer_dir = Path(__file__).parent.parent / 'trainer'
sys.path.insert(0, str(trainer_dir))

from google.cloud import bigquery, storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter

# Import trainer modules
from feature_engineering import engineer_features_dataframe
from tensorboard_logging import log_training_metrics_to_tensorboard

# Import shared evaluation function from evaluate_model.py
# Add tools directory to path for import
tools_dir = Path(__file__).parent
sys.path.insert(0, str(tools_dir))
from evaluate_model import evaluate_model_with_tensorboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_crop_labels(crop_name):
    """Standardize crop names (remove underscores, title case)."""
    return crop_name.replace('_', ' ').title()


def load_config():
    """Load configuration from orchestrator config."""
    config_path = Path(__file__).parent.parent / 'orchestrator' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_training_data_sample(config, sample_size=500, min_samples_per_crop=20):
    """
    Load a small sample of training data from BigQuery.
    
    Args:
        config: Configuration dict
        sample_size: Total number of samples to load
        min_samples_per_crop: Minimum samples per crop (for stratification)
    
    Returns:
        DataFrame with sampled data
    """
    logger.info("📥 Loading test training data from BigQuery...")
    logger.info(f"   Sample size: {sample_size}")
    logger.info(f"   Min samples per crop: {min_samples_per_crop}")
    
    project_id = config['project']['id']
    dataset = config['bigquery']['dataset']
    table = config['bigquery']['tables']['training']
    
    # First, get crop distribution to ensure we have enough samples per crop
    count_query = f"""
        SELECT crop, COUNT(*) as count
        FROM `{project_id}.{dataset}.{table}`
        WHERE crop IS NOT NULL
        GROUP BY crop
        ORDER BY count DESC
    """
    
    client = bigquery.Client(project=project_id)
    crop_counts = client.query(count_query).to_dataframe()
    
    logger.info(f"   Available crops: {crop_counts.to_dict('records')}")
    
    # Build stratified query - get samples from each crop
    # Use RAND() for random sampling within each crop
    crop_samples = []
    samples_per_crop = max(min_samples_per_crop, sample_size // len(crop_counts))
    
    for _, row in crop_counts.iterrows():
        crop = row['crop']
        available = row['count']
        take = min(samples_per_crop, available)
        
        crop_query = f"""
            SELECT 
                crop,
                ndvi_mean, ndvi_std, ndvi_min, ndvi_max,
                ndvi_p25, ndvi_p50, ndvi_p75,
                ndvi_early, ndvi_late
            FROM `{project_id}.{dataset}.{table}`
            WHERE crop = '{crop}'
            ORDER BY RAND()
            LIMIT {take}
        """
        
        crop_df = client.query(crop_query).to_dataframe()
        crop_samples.append(crop_df)
        logger.info(f"   ✅ Loaded {len(crop_df)} samples for {crop}")
    
    # Combine all crops
    df = pd.concat(crop_samples, ignore_index=True)
    
    # If we have more than sample_size, randomly sample down
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"   Randomly sampled down to {sample_size} samples")
    
    logger.info(f"✅ Loaded {len(df)} total samples")
    logger.info(f"   Crops: {df['crop'].value_counts().to_dict()}")
    
    return df


def train_model(df, feature_cols, config, test_size=0.2):
    """
    Train RandomForest model with smaller hyperparameters for testing.
    
    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        config: Configuration dict
        test_size: Fraction of data to use for testing
    
    Returns:
        pipeline, metrics, X_test, y_test, y_pred
    """
    logger.info("🤖 Training test model...")
    
    # Prepare data
    X = df[feature_cols]
    y = df['crop']
    
    logger.info(f"   Training samples: {len(X)}")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Classes: {list(y.unique())}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Use smaller model for testing (fewer estimators, shallower trees)
    hyperparams = config.get('model', {}).get('hyperparameters', {})
    n_estimators = hyperparams.get('n_estimators', 100)
    max_depth = hyperparams.get('max_depth', 10)
    
    # Reduce for testing
    test_n_estimators = min(20, n_estimators // 5)  # Much smaller
    test_max_depth = min(5, max_depth // 3)  # Shallower
    
    logger.info(f"   Model: RandomForest")
    logger.info(f"   Estimators: {test_n_estimators} (production: {n_estimators})")
    logger.info(f"   Max depth: {test_max_depth} (production: {max_depth})")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=test_n_estimators,
            max_depth=test_max_depth,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    logger.info(f"✅ Training complete")
    logger.info(f"   Train accuracy: {train_acc:.2%}")
    logger.info(f"   Test accuracy: {test_acc:.2%}")
    
    # Compile metrics
    metrics = {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'classification_report': classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    }
    
    return pipeline, metrics, X_test, y_test, test_pred


def save_model_local(pipeline, feature_cols, metrics, output_dir):
    """Save model locally."""
    logger.info(f"💾 Saving model to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'model.joblib')
    features_path = os.path.join(output_dir, 'feature_cols.json')
    metrics_path = os.path.join(output_dir, 'metrics.json')
    
    joblib.dump(pipeline, model_path, protocol=4)
    
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, default=str)
    
    logger.info(f"✅ Model saved locally")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Features: {features_path}")
    logger.info(f"   Metrics: {metrics_path}")


def upload_model_to_gcs(pipeline, feature_cols, metrics, config, prefix='models/test_training'):
    """Upload model to GCS (optional)."""
    logger.info(f"📤 Uploading model to GCS...")
    
    bucket_name = config['storage']['bucket']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Save to temp directory first
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'model.joblib')
        features_path = os.path.join(temp_dir, 'feature_cols.json')
        metrics_path = os.path.join(temp_dir, 'metrics.json')
        
        joblib.dump(pipeline, model_path, protocol=4)
        
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, default=str)
        
        # Upload to GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        gcs_prefix = f'{prefix}/{timestamp}'
        bucket.blob(f'{gcs_prefix}/model.joblib').upload_from_filename(model_path)
        bucket.blob(f'{gcs_prefix}/feature_cols.json').upload_from_filename(features_path)
        bucket.blob(f'{gcs_prefix}/metrics.json').upload_from_filename(metrics_path)
        
        logger.info(f"✅ Model uploaded to gs://{bucket_name}/{gcs_prefix}")


def upload_tensorboard_logs(local_dir, config, prefix='training_output/logs/test'):
    """Upload TensorBoard logs to GCS (optional)."""
    logger.info(f"📤 Uploading TensorBoard logs to GCS...")
    
    bucket_name = config['storage']['bucket']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'test_run_{timestamp}'
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    uploaded_count = 0
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_path = f'{prefix}/{run_name}/{rel_path}'
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            uploaded_count += 1
    
    logger.info(f"✅ Uploaded {uploaded_count} TensorBoard files")
    logger.info(f"   GCS location: gs://{bucket_name}/{prefix}/{run_name}/")


def main():
    parser = argparse.ArgumentParser(
        description='Run test training with small dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=500,
        help='Total number of samples to load from BigQuery (default: 500)'
    )
    parser.add_argument(
        '--min-samples-per-crop',
        type=int,
        default=20,
        help='Minimum samples per crop for stratification (default: 20)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_output',
        help='Local directory for model and logs (default: test_output)'
    )
    parser.add_argument(
        '--upload-tensorboard',
        action='store_true',
        help='Upload TensorBoard logs to GCS'
    )
    parser.add_argument(
        '--upload-model',
        action='store_true',
        help='Upload model to GCS'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("🧪 TEST TRAINING")
    logger.info("=" * 70)
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    start_time = datetime.now()
    
    try:
        # Load config
        config = load_config()
        
        # Load data sample
        df = load_training_data_sample(
            config,
            sample_size=args.sample_size,
            min_samples_per_crop=args.min_samples_per_crop
        )
        
        # Normalize crop labels
        df['crop'] = df['crop'].apply(normalize_crop_labels)
        logger.info(f"✅ Normalized crop labels: {df['crop'].unique().tolist()}")
        
        # Engineer features
        logger.info("🔧 Engineering features...")
        df_enhanced, feature_cols = engineer_features_dataframe(df)
        logger.info(f"✅ Created {len(feature_cols)} features")
        
        # Train model
        pipeline, metrics, X_test, y_test, y_pred = train_model(df_enhanced, feature_cols, config)
        
        # Setup TensorBoard logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f'test_run_{timestamp}'
        local_tb_dir = os.path.join(args.output_dir, 'tensorboard_logs', run_name)
        os.makedirs(local_tb_dir, exist_ok=True)
        
        logger.info("")
        logger.info("📊 Setting up TensorBoard logging...")
        logger.info(f"   Local directory: {local_tb_dir}")
        
        writer = SummaryWriter(log_dir=local_tb_dir)
        
        # Run evaluation using shared evaluation function
        logger.info("")
        logger.info("🔍 Running model evaluation...")
        eval_results = evaluate_model_with_tensorboard(
            model=pipeline,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            writer=writer,
            step=0,
            num_runs=1
        )
        
        # Log training-specific metrics (not part of general evaluation)
        log_training_metrics_to_tensorboard(writer, config, metrics, y_test, y_pred)
        
        # Flush and close
        writer.flush()
        writer.close()
        logger.info("✅ TensorBoard logging complete")
        
        # Save model locally
        model_dir = os.path.join(args.output_dir, 'model')
        save_model_local(pipeline, feature_cols, metrics, model_dir)
        
        # Optional uploads
        if args.upload_tensorboard:
            upload_tensorboard_logs(local_tb_dir, config)
        
        if args.upload_model:
            upload_model_to_gcs(pipeline, feature_cols, metrics, config)
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ TEST TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.1f} minutes")
        logger.info(f"Test accuracy: {metrics['test_accuracy']:.2%}")
        logger.info(f"Train accuracy: {metrics['train_accuracy']:.2%}")
        logger.info(f"Training samples: {metrics['n_train_samples']}")
        logger.info(f"Test samples: {metrics['n_test_samples']}")
        logger.info("")
        logger.info("📁 Output files:")
        logger.info(f"   Model: {model_dir}/")
        logger.info(f"   TensorBoard logs: {local_tb_dir}/")
        logger.info("")
        logger.info("📊 View TensorBoard locally:")
        logger.info(f"   tensorboard --logdir {local_tb_dir}")
        logger.info("")
        
        # Print metrics JSON for programmatic access
        output_metrics = {
            'status': 'success',
            'accuracy': metrics['test_accuracy'],
            'duration_minutes': round(duration, 2),
            'training_samples': metrics['n_train_samples'],
            'test_samples': metrics['n_test_samples'],
            'metrics': metrics
        }
        
        print("\n" + "=" * 70)
        print("TRAINING METRICS (JSON):")
        print("=" * 70)
        print(json.dumps(output_metrics, indent=2, default=str))
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Test training failed: {e}", exc_info=True)
        
        print(json.dumps({
            'status': 'error',
            'error': str(e)
        }, indent=2))
        
        sys.exit(1)


if __name__ == '__main__':
    main()

