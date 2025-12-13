"""
SIMPLIFIED Vertex AI Training Script
======================================
This version fixes TensorBoard image logging by:
1. Writing directly to AIP_TENSORBOARD_LOG_DIR (no manual upload)
2. Simplified image conversion
3. Removed unnecessary progression/num_runs complexity
4. Reduced excessive flushes
"""

import os
import sys
import json
import logging
import joblib
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import bigquery, storage, aiplatform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter

# Import local modules (all files are in /app)
from feature_engineering import engineer_features_dataframe
from tensorboard_utils import run_comprehensive_evaluation, log_training_metrics_to_tensorboard

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
    """Load configuration from GCS or local file."""
    try:
        client = storage.Client()
        bucket = client.bucket('carboncheck-data')
        blob = bucket.blob('config/config.yaml')
        config_text = blob.download_as_text()
        logger.info("✅ Loaded config from GCS")
        return yaml.safe_load(config_text)
    except Exception as e:
        logger.warning(f"⚠️  Could not load from GCS: {e}, using local config")
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)


def load_training_data(config):
    """Load training data from BigQuery."""
    logger.info("📥 Loading training data from BigQuery...")
    
    project_id = config['project']['id']
    dataset = config['bigquery']['dataset']
    table = config['bigquery']['tables']['training']
    
    query = f"""
        SELECT 
            crop,
            ndvi_mean, ndvi_std, ndvi_min, ndvi_max,
            ndvi_p25, ndvi_p50, ndvi_p75,
            ndvi_early, ndvi_late
        FROM `{project_id}.{dataset}.{table}`
        WHERE crop IS NOT NULL
    """
    
    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()
    
    logger.info(f"✅ Loaded {len(df)} samples")
    logger.info(f"   Crops: {df['crop'].value_counts().to_dict()}")
    
    return df


def engineer_features(df, config):
    """Engineer features using shared module."""
    logger.info("🔧 Engineering features...")
    
    # Use shared feature engineering (no elevation quantiles needed anymore)
    df_enhanced, all_features = engineer_features_dataframe(df)
    
    logger.info(f"✅ Created {len(all_features)} features")
    
    return df_enhanced, all_features


def train_model(df, feature_cols, config):
    """Train RandomForest model."""
    logger.info("🤖 Training model...")
    
    # Prepare data
    X = df[feature_cols]
    y = df['crop']
    
    logger.info(f"   Training samples: {len(X)}")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Classes: {list(y.unique())}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=config.get('training', {}).get('n_estimators', 100),
            max_depth=config.get('training', {}).get('max_depth', 15),
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


def save_model(pipeline, feature_cols, metrics, config):
    """Save model to GCS."""
    logger.info("💾 Saving model...")
    
    bucket_name = config['storage']['bucket']
    output_dir = os.environ.get('AIP_MODEL_DIR', '/tmp/model')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save locally
    model_path = os.path.join(output_dir, 'model.joblib')
    features_path = os.path.join(output_dir, 'feature_cols.json')
    metrics_path = os.path.join(output_dir, 'metrics.json')
    
    joblib.dump(pipeline, model_path, protocol=4)
    
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, default=str)
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Archive
    archive_prefix = f'models/crop_classifier_archive/crop_classifier_{timestamp}'
    bucket.blob(f'{archive_prefix}/model.joblib').upload_from_filename(model_path)
    bucket.blob(f'{archive_prefix}/feature_cols.json').upload_from_filename(features_path)
    bucket.blob(f'{archive_prefix}/metrics.json').upload_from_filename(metrics_path)
    
    # Latest
    latest_prefix = 'models/crop_classifier_latest'
    bucket.blob(f'{latest_prefix}/model.joblib').upload_from_filename(model_path)
    bucket.blob(f'{latest_prefix}/feature_cols.json').upload_from_filename(features_path)
    bucket.blob(f'{latest_prefix}/metrics.json').upload_from_filename(metrics_path)
    
    logger.info(f"✅ Model saved to gs://{bucket_name}/{latest_prefix}")


if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("🎯 VERTEX AI TRAINING (SIMPLIFIED)")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # Load config
        config = load_config()
        
        # Load data
        df = load_training_data(config)
        
        # Normalize crop labels
        df['crop'] = df['crop'].apply(normalize_crop_labels)
        logger.info(f"✅ Normalized crop labels: {df['crop'].unique().tolist()}")
        
        # Engineer features
        df_enhanced, feature_cols = engineer_features(df, config)
        
        # Train model
        pipeline, metrics, X_test, y_test, y_pred = train_model(df_enhanced, feature_cols, config)
        
        # === TENSORBOARD LOGGING ===
        # Following the official GCP guide for Vertex AI TensorBoard integration
        # KEY INSIGHT: Vertex AI automatically syncs logs from a LOCAL directory
        # to the managed TensorBoard bucket when tensorboard= is set in the training job
        
        # Check if AIP_TENSORBOARD_LOG_DIR is set (it should be for managed TensorBoard)
        managed_tb_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
        
        # CRITICAL: Write to a SIMPLE local directory structure
        # Vertex AI will automatically sync this to the managed bucket
        # Structure: /tmp/tensorboard/ (flat, no subdirectories needed)
        local_tb_dir = '/tmp/tensorboard'
        os.makedirs(local_tb_dir, exist_ok=True)
        
        logger.info(f"📊 TensorBoard Configuration:")
        logger.info(f"   AIP_TENSORBOARD_LOG_DIR: {managed_tb_dir if managed_tb_dir else 'NOT SET'}")
        logger.info(f"   Local directory: {local_tb_dir}")
        logger.info(f"   Vertex AI will auto-sync logs to managed TensorBoard bucket")
        
        # Create writer pointing to LOCAL directory (Vertex AI handles the rest)
        writer = SummaryWriter(log_dir=local_tb_dir)
        
        # CRITICAL: Initialize Vertex AI Experiments to register in UI
        experiment_name = os.environ.get('AIP_TENSORBOARD_EXPERIMENT_NAME', 'crop_training')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f'run_{timestamp}'
        
        try:
            logger.info(f"🔗 Initializing Vertex AI Experiment: {experiment_name}")
            aiplatform.init(
                project=config['project']['id'],
                location=config['project']['region'],
                experiment=experiment_name,
                experiment_tensorboard='projects/303566498201/locations/us-central1/tensorboards/3503556418512879616'
            )
            
            # Start run - this creates it in the Experiments UI
            aiplatform.start_run(run=run_name)
            logger.info(f"✅ Started run: {run_name}")
        except Exception as e:
            logger.warning(f"⚠️  Could not start Vertex AI Experiment: {e}")
        
        # Run evaluation (SIMPLIFIED: no num_runs, just log once)
        logger.info("\n🔍 Running model evaluation...")
        eval_results = run_comprehensive_evaluation(
            model=pipeline,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            writer=writer,
            step=0,
            num_runs=1  # SIMPLIFIED: Just log once, no progression
        )
        
        # Log training metrics
        log_training_metrics_to_tensorboard(writer, config, metrics, y_test, y_pred)
        
        # Log key metrics to Vertex AI Experiments (for UI display)
        try:
            aiplatform.log_metrics({
                'accuracy': metrics['test_accuracy'],
                'train_accuracy': metrics['train_accuracy'],
                'num_features': len(feature_cols),
                'num_samples': len(df_enhanced)
            })
            logger.info("✅ Logged metrics to Vertex AI Experiments")
        except Exception as e:
            logger.warning(f"⚠️  Could not log metrics to Vertex AI: {e}")
        
        # CRITICAL: Explicit flush and delay to ensure all data is written
        logger.info("💾 Flushing TensorBoard writer...")
        writer.flush()
        
        # IMPORTANT: Longer delay to ensure all images are fully written to disk
        import time
        logger.info("⏳ Waiting 5 seconds for images to write...")
        time.sleep(5)
        
        # End Vertex AI Experiment run (registers it in UI)
        try:
            aiplatform.end_run()
            logger.info("✅ Ended Vertex AI Experiment run")
        except Exception as e:
            logger.warning(f"⚠️  Could not end Vertex AI run: {e}")
        
        # Close writer
        writer.close()
        logger.info("✅ TensorBoard writer closed")
        
        # DEBUG: Check what files were created locally
        logger.info(f"🔍 Checking local TensorBoard files in {local_tb_dir}")
        local_files = []
        for root, dirs, files in os.walk(local_tb_dir):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                local_files.append((file, size))
                logger.info(f"   📄 {file}: {size:,} bytes")
        
        if not local_files:
            logger.error("❌ No TensorBoard files were created!")
        else:
            logger.info(f"✅ TensorBoard files created: {len(local_files)} files")
            logger.info("ℹ️  Vertex AI Experiments will automatically sync these logs")
            
            # Give Vertex AI time to sync files before job ends
            logger.info("⏳ Waiting 15 seconds for Vertex AI to sync files...")
            time.sleep(15)
            logger.info("✅ Sync delay complete")
        
        # Save model
        save_model(pipeline, feature_cols, metrics, config)
        
        # Output final metrics
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        logger.info("=" * 70)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.1f} minutes")
        logger.info(f"Test accuracy: {metrics['test_accuracy']:.2%}")
        
        # Metrics for orchestrator
        output_metrics = {
            'status': 'success',
            'accuracy': metrics['test_accuracy'],
            'duration_minutes': round(duration, 2),
            'training_samples': len(df_enhanced),
            'metrics': metrics
        }
        
        print("\n" + "=" * 70)
        print("TRAINING METRICS (JSON):")
        print("=" * 70)
        print(json.dumps(output_metrics, indent=2, default=str))
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        
        print(json.dumps({
            'status': 'error',
            'error': str(e)
        }, indent=2))
        
        exit(1)

