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
from importlib import metadata
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
from tensorboard_logging import (
    run_comprehensive_evaluation,
    log_training_metrics_to_tensorboard,
    log_data_skew_to_tensorboard
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_package_versions():
    """Log key package versions for debugging environment mismatches."""
    logger.info("📦 Package Versions:")
    logger.info(f"   python: {sys.version.split()[0]}")
    packages = [
        'torch',
        'tensorboard',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'google-cloud-aiplatform',
        'google-cloud-storage',
        'google-cloud-bigquery',
        'protobuf'
    ]
    for package in packages:
        try:
            version = metadata.version(package)
            logger.info(f"   {package}: {version}")
        except Exception:
            logger.info(f"   {package}: NOT INSTALLED")
    logger.info("")


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
    region = config['project']['region']
    
    query = f"""
        SELECT 
            crop,
            ndvi_mean, ndvi_std, ndvi_min, ndvi_max,
            ndvi_p25, ndvi_p50, ndvi_p75,
            ndvi_early, ndvi_late
        FROM `{project_id}.{dataset}.{table}`
        WHERE crop IS NOT NULL
    """
    
    # Create BigQuery client - don't specify location to auto-detect (works for both US and us-central1)
    client = bigquery.Client(project=project_id)
    
    logger.info(f"   Project: {project_id}")
    logger.info(f"   Dataset: {dataset}")
    logger.info(f"   Table: {table}")
    
    try:
        # Don't specify location parameter - let BigQuery auto-detect dataset location
        df = client.query(query).to_dataframe()
        
        if len(df) == 0:
            raise ValueError(f"BigQuery table {project_id}.{dataset}.{table} exists but is empty!")
        
        logger.info(f"✅ Loaded {len(df)} samples")
        logger.info(f"   Crops: {df['crop'].value_counts().to_dict()}")
        
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load data from BigQuery: {e}")
        logger.error(f"   Table: {project_id}.{dataset}.{table}")
        logger.error(f"   Location: {region}")
        raise


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
    
    return pipeline, metrics, X_test, y_test, y_train, test_pred


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
    # CRITICAL: Clear Python cache to ensure we're using the latest code
    import sys
    import importlib
    import shutil
    
    cache_dirs = ['/app/__pycache__', '/tmp/__pycache__']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"🧹 Cleared Python cache: {cache_dir}")
            except Exception as e:
                print(f"⚠️  Could not clear cache {cache_dir}: {e}")
    
    # Force reload modules to ensure latest code (after logger is configured)
    # Note: We'll reload after logger setup
    
    logger.info("=" * 70)
    logger.info("🎯 VERTEX AI TRAINING (SIMPLIFIED)")
    logger.info("=" * 70)
    
    # CRITICAL: Log environment variables to debug TensorBoard path issues
    logger.info("🔍 Environment Variables:")
    logger.info(f"   AIP_MODEL_DIR: {os.environ.get('AIP_MODEL_DIR', 'NOT SET')}")
    logger.info(f"   AIP_TENSORBOARD_LOG_DIR: {os.environ.get('AIP_TENSORBOARD_LOG_DIR', 'NOT SET')}")
    logger.info(f"   AIP_TENSORBOARD_EXPERIMENT_NAME: {os.environ.get('AIP_TENSORBOARD_EXPERIMENT_NAME', 'NOT SET')}")
    logger.info("")
    logger.info("⚠️  CRITICAL: TensorBoard logs MUST be written to /tmp, NOT AIP_MODEL_DIR")
    logger.info("")

    # Log package versions to debug TensorBoard failures
    log_package_versions()
    
    # Force reload modules to ensure latest code (now that logger is ready)
    modules_to_reload = ['tensorboard_logging', 'feature_engineering', 'visualization_utils']
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            try:
                importlib.reload(sys.modules[mod_name])
                logger.info(f"🔄 Reloaded module: {mod_name}")
            except Exception as e:
                logger.warning(f"⚠️  Could not reload {mod_name}: {e}")
    
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
        pipeline, metrics, X_test, y_test, y_train, y_pred = train_model(df_enhanced, feature_cols, config)
        
        # === TENSORBOARD LOGGING ===
        # CRITICAL: SummaryWriter cannot write directly to GCS paths
        # Solution: Write to local directory, then upload to AIP_TENSORBOARD_LOG_DIR
        
        # Get the GCS destination from Vertex AI
        managed_tb_gcs_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
        
        # Create a unique run directory for this training run
        # SummaryWriter works best with a dedicated directory per run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f'run_{timestamp}'
        local_tb_base = '/tmp/tensorboard_logs'
        local_tb_dir = os.path.join(local_tb_base, run_name)
        os.makedirs(local_tb_dir, exist_ok=True)
        
        logger.info(f"📊 TensorBoard Configuration:")
        if managed_tb_gcs_dir:
            logger.info(f"   AIP_TENSORBOARD_LOG_DIR (GCS): {managed_tb_gcs_dir}")
            logger.info(f"   Local write directory: {local_tb_dir}")
            logger.info(f"   Run name: {run_name}")
            logger.info(f"   Will upload to GCS after logging completes")
        else:
            logger.warning(f"⚠️  AIP_TENSORBOARD_LOG_DIR not set")
            logger.info(f"   Writing to local: {local_tb_dir}")
        
        # Create writer pointing to LOCAL directory (SummaryWriter needs local filesystem)
        # Use the run-specific directory
        # CRITICAL: Ensure we're NOT using AIP_MODEL_DIR or any GCS path
        aip_model_dir = os.environ.get('AIP_MODEL_DIR', '')
        if aip_model_dir and aip_model_dir in local_tb_dir:
            raise ValueError(f"ERROR: TensorBoard log_dir cannot use AIP_MODEL_DIR! Got: {local_tb_dir}")
        
        assert not local_tb_dir.startswith('gs://'), f"ERROR: TensorBoard log_dir cannot be a GCS path: {local_tb_dir}"
        assert '/tmp' in local_tb_dir, f"ERROR: TensorBoard must write to /tmp, not: {local_tb_dir}"
        assert 'model' not in local_tb_dir.lower(), f"ERROR: TensorBoard log_dir cannot contain 'model': {local_tb_dir}"
        
        logger.info(f"🔒 Safety checks passed:")
        logger.info(f"   ✅ Not using AIP_MODEL_DIR")
        logger.info(f"   ✅ Not a GCS path")
        logger.info(f"   ✅ Writing to /tmp")
        
        writer = SummaryWriter(log_dir=local_tb_dir)
        logger.info(f"✅ SummaryWriter created for: {local_tb_dir}")
        logger.info(f"   Verified: Local filesystem path (not GCS, not AIP_MODEL_DIR)")
        
        # CRITICAL: Initialize Vertex AI Experiments to register in UI
        experiment_name = os.environ.get('AIP_TENSORBOARD_EXPERIMENT_NAME', 'crop_training')
        # Use the same run_name we created for TensorBoard directory
        
        # Get TensorBoard instance ID from config
        project_number = '303566498201'  # Numeric project ID
        tensorboard_id = config.get('tensorboard', {}).get('instance_id')
        
        if tensorboard_id:
            tensorboard_resource = f'projects/{project_number}/locations/{config["project"]["region"]}/tensorboards/{tensorboard_id}'
            logger.info(f"🔗 Linking to TensorBoard instance: {tensorboard_id}")
        else:
            tensorboard_resource = None
            logger.warning("⚠️  No TensorBoard instance ID in config")
        
        try:
            logger.info(f"🔗 Initializing Vertex AI Experiment: {experiment_name}")
            init_params = {
                'project': config['project']['id'],
                'location': config['project']['region'],
                'experiment': experiment_name
            }
            if tensorboard_resource:
                init_params['experiment_tensorboard'] = tensorboard_resource
            
            aiplatform.init(**init_params)
            
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
        
        # Log training metrics and data skew
        log_training_metrics_to_tensorboard(writer, config, metrics, y_test, y_pred)
        log_data_skew_to_tensorboard(writer, y_train, y_test)
        
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
        # SummaryWriter may create subdirectories, so we need to check recursively
        logger.info(f"🔍 Checking TensorBoard files in {local_tb_dir}")
        logger.info(f"   Directory exists: {os.path.exists(local_tb_dir)}")
        logger.info(f"   Is directory: {os.path.isdir(local_tb_dir) if os.path.exists(local_tb_dir) else False}")
        
        local_files = []
        if os.path.exists(local_tb_dir) and os.path.isdir(local_tb_dir):
            # List all contents first
            try:
                all_items = os.listdir(local_tb_dir)
                logger.info(f"   Directory contents: {all_items}")
            except Exception as e:
                logger.warning(f"   Could not list directory: {e}")
            
            # Walk recursively to find all files
            for root, dirs, files in os.walk(local_tb_dir):
                logger.info(f"   Walking: {root} (dirs: {dirs}, files: {files})")
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        try:
                            size = os.path.getsize(file_path)
                            rel_path = os.path.relpath(file_path, local_tb_dir)
                            local_files.append((rel_path, size))
                            logger.info(f"   📄 {rel_path}: {size:,} bytes")
                        except Exception as e:
                            logger.warning(f"   Could not get size for {file_path}: {e}")
        
        if not local_files:
            logger.error("❌ No TensorBoard files found in local directory!")
            logger.error(f"   Checked: {local_tb_dir}")
            logger.error("   This means SummaryWriter did not create any event files")
        else:
            logger.info(f"✅ TensorBoard files created locally: {len(local_files)} files")
            total_size = sum(size for _, size in local_files)
            logger.info(f"   Total size: {total_size:,} bytes")
        
        # CRITICAL: Upload local TensorBoard logs to AIP_TENSORBOARD_LOG_DIR
        # Vertex AI automatically syncs from this location to TensorBoard UI
        # Structure: gs://bucket/training_output/logs/run_TIMESTAMP/events.out.tfevents...
        if managed_tb_gcs_dir and local_files:
            logger.info(f"📤 Uploading TensorBoard logs to GCS")
            logger.info(f"   Source: {local_tb_dir}")
            logger.info(f"   Destination: {managed_tb_gcs_dir}/{run_name}/")
            try:
                from google.cloud import storage
                
                # Parse GCS path: gs://bucket/path/to/logs/
                if managed_tb_gcs_dir.startswith('gs://'):
                    gcs_path = managed_tb_gcs_dir.replace('gs://', '')
                    parts = gcs_path.split('/', 1)
                    bucket_name = parts[0]
                    gcs_prefix = parts[1].rstrip('/') if len(parts) > 1 else ''  # Remove trailing slash
                    
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(bucket_name)
                    
                    # Upload all files from local_tb_dir (the run-specific directory)
                    # Upload to: gs://bucket/prefix/run_name/...
                    # This preserves the directory structure TensorBoard expects
                    uploaded_count = 0
                    uploaded_files = []
                    
                    for root, dirs, files in os.walk(local_tb_dir):
                        for file in files:
                            local_path = os.path.join(root, file)
                            # Get relative path from local_tb_dir (preserves subdirectories created by SummaryWriter/hparams)
                            rel_path = os.path.relpath(local_path, local_tb_dir)
                            # Construct GCS path: prefix/run_name/relative_path
                            # Remove any leading slashes or dots
                            rel_path = rel_path.lstrip('./').replace('\\', '/')
                            if gcs_prefix:
                                blob_path = f"{gcs_prefix}/{run_name}/{rel_path}"
                            else:
                                blob_path = f"{run_name}/{rel_path}"
                            
                            blob = bucket.blob(blob_path)
                            blob.upload_from_filename(local_path)
                            uploaded_count += 1
                            uploaded_files.append(blob_path)
                            logger.info(f"   ✅ {rel_path} -> {blob_path}")
                    
                    logger.info("")
                    logger.info(f"✅ Uploaded {uploaded_count} TensorBoard files to GCS")
                    logger.info(f"   GCS location: gs://{bucket_name}/{gcs_prefix}/{run_name}/")
                    logger.info(f"   Files uploaded:")
                    for f in uploaded_files[:5]:  # Show first 5 files
                        logger.info(f"      - {f}")
                    if len(uploaded_files) > 5:
                        logger.info(f"      ... and {len(uploaded_files) - 5} more")
                    logger.info("")
                    logger.info("   ⏳ Vertex AI will automatically sync these logs to TensorBoard")
                    logger.info(f"   📊 View in TensorBoard: {managed_tb_gcs_dir}/{run_name}/")
                else:
                    logger.warning(f"⚠️  Invalid GCS path format: {managed_tb_gcs_dir}")
            except Exception as e:
                logger.error(f"❌ Failed to upload TensorBoard logs to GCS: {e}")
                import traceback
                logger.error(traceback.format_exc())
        elif not managed_tb_gcs_dir:
            logger.warning("⚠️  AIP_TENSORBOARD_LOG_DIR not set - logs will not be synced to TensorBoard")
        elif not local_files:
            logger.error("❌ No TensorBoard files to upload - SummaryWriter did not create any files!")
        
        # Give Vertex AI time to sync files before job ends
        logger.info("⏳ Waiting 10 seconds for Vertex AI to sync files...")
        time.sleep(10)
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

