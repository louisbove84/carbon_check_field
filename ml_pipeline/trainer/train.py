"""
Vertex AI Custom Training Script
=================================
Heavy ML workload that runs on Vertex AI managed infrastructure.
This script:
1. Loads training data from BigQuery/GCS
2. Trains RandomForest model
3. Saves model to GCS
4. Returns metrics

Environment variables (set by Vertex AI):
- AIP_MODEL_DIR: Where to save model artifacts
- AIP_TRAINING_DATA_URI: Location of training data (optional)
"""

import os
import yaml
import logging
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from google.cloud import bigquery, storage, aiplatform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import io
from PIL import Image
from feature_engineering import engineer_features_dataframe, compute_elevation_quantiles
from model_evaluation import run_comprehensive_evaluation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket('carboncheck-data')
        blob = bucket.blob('config/config.yaml')
        yaml_content = blob.download_as_text()
        logger.info("✅ Config loaded from Cloud Storage")
        return yaml.safe_load(yaml_content)
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        raise


def load_training_data(config):
    """Load training data from BigQuery."""
    logger.info("📥 Loading training data from BigQuery...")
    
    project_id = config['project']['id']
    dataset_id = config['bigquery']['dataset']
    training_table = config['bigquery']['tables']['training']
    holdout_table = config['bigquery']['tables']['holdout']
    
    client = bigquery.Client(project=project_id)
    
    # Load training data (excluding holdout if exists)
    query = f"""
    SELECT t.*
    FROM `{project_id}.{dataset_id}.{training_table}` t
    WHERE t.ndvi_mean IS NOT NULL
    """
    
    # Note: Holdout exclusion will be added once sample_id column is available
    
    df = client.query(query).to_dataframe()
    
    logger.info(f"✅ Loaded {len(df)} training samples")
    logger.info(f"   Crops: {df['crop'].value_counts().to_dict()}")
    
    return df


def engineer_features(df, config):
    """Create derived features using shared feature engineering module."""
    logger.info("🔧 Engineering features...")
    
    # Compute elevation quantiles from training data
    elevation_quantiles = compute_elevation_quantiles(df)
    logger.info(f"   Elevation quantiles: {elevation_quantiles}")
    
    # Save quantiles to config for use in prediction API
    try:
        client = storage.Client()
        bucket = client.bucket(config['storage']['bucket'])
        blob = bucket.blob('config/config.yaml')
        if blob.exists():
            import yaml
            config_dict = yaml.safe_load(blob.download_as_text())
            if 'features' not in config_dict:
                config_dict['features'] = {}
            config_dict['features']['elevation_quantiles'] = elevation_quantiles
            # Update base_columns to reflect new feature structure
            config_dict['features']['base_columns'] = [
                'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max',
                'ndvi_p25', 'ndvi_p50', 'ndvi_p75', 'ndvi_early', 'ndvi_late',
                'elevation_binned',  # Changed from elevation_m
                'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos'  # Changed from longitude, latitude
            ]
            # Save updated config
            blob.upload_from_string(yaml.dump(config_dict, default_flow_style=False))
            logger.info("✅ Updated config.yaml with elevation quantiles and new feature columns")
    except Exception as e:
        logger.warning(f"⚠️  Could not update config.yaml: {e}")
    
    # Use shared feature engineering
    df_enhanced, all_features = engineer_features_dataframe(df, elevation_quantiles)
    
    logger.info(f"✅ Created {len(all_features)} features")
    
    return df_enhanced, all_features


def train_model(df, feature_cols, config):
    """Train RandomForest pipeline."""
    logger.info("🤖 Training RandomForest model...")
    
    hyperparams = config['model']['hyperparameters']
    
    logger.info(f"   Hyperparameters:")
    logger.info(f"   - n_estimators: {hyperparams['n_estimators']}")
    logger.info(f"   - max_depth: {hyperparams['max_depth']}")
    
    X = df[feature_cols]
    y = df['crop']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=hyperparams['random_state'], stratify=y
    )
    
    logger.info(f"   Train: {len(X_train)} samples")
    logger.info(f"   Test: {len(X_test)} samples")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            min_samples_split=hyperparams['min_samples_split'],
            random_state=hyperparams['random_state'],
            n_jobs=-1
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    logger.info(f"✅ Training accuracy: {train_score:.2%}")
    logger.info(f"✅ Test accuracy: {test_score:.2%}")
    
    # Per-crop metrics
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info("📊 Per-crop F1 scores:")
    for crop in sorted(y_test.unique()):
        if crop in report:
            logger.info(f"   {crop}: {report[crop]['f1-score']:.2%}")
    
    metrics = {
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'classification_report': report,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    return pipeline, metrics, X_test, y_test, y_pred


# Removed generate_confusion_matrix - all visualizations now go to TensorBoard only


def log_training_metrics_to_tensorboard(writer, config, metrics, y_test, y_pred):
    """Log training-specific metrics and hyperparameters to TensorBoard."""
    try:
        # Log hyperparameters
        hparams = {
            'n_estimators': config['model']['hyperparameters']['n_estimators'],
            'max_depth': config['model']['hyperparameters']['max_depth'],
            'min_samples_split': config['model']['hyperparameters']['min_samples_split'],
            'n_train_samples': metrics['n_train_samples'],
            'n_test_samples': metrics['n_test_samples']
        }
        
        # Log training accuracy (separate from test accuracy in comprehensive eval)
        writer.add_scalar('training/train_accuracy', metrics['train_accuracy'], 0)
        
        # Log text summary
        report = metrics['classification_report']
        crops = sorted([c for c in set(y_test)])
        
        report_text = f"""
        Training Summary
        ================
        Train Accuracy: {metrics['train_accuracy']:.2%}
        Test Accuracy: {metrics['test_accuracy']:.2%}
        
        Per-Crop F1 Scores:
        {''.join([f'- {crop}: {report[crop]["f1-score"]:.2%}' + chr(10) for crop in crops if crop in report])}
        """
        writer.add_text('training_summary', report_text, 0)
        
        # Log hyperparameters with metrics
        metric_dict = {
            'hparam/train_accuracy': metrics['train_accuracy'],
            'hparam/test_accuracy': metrics['test_accuracy'],
        }
        writer.add_hparams(hparams, metric_dict)
        
        logger.info("✅ Logged training metrics to TensorBoard")
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to log training metrics to TensorBoard: {e}")


def upload_tensorboard_logs(local_log_dir, config):
    """Upload TensorBoard logs to GCS for Vertex AI TensorBoard."""
    try:
        project_id = config['project']['id']
        
        # Use Vertex AI's managed TensorBoard path if available
        tensorboard_gcs_path = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
        
        if not tensorboard_gcs_path:
            logger.info("📊 TensorBoard logs saved locally (no GCS upload configured)")
            return
        
        if not tensorboard_gcs_path.startswith('gs://'):
            logger.warning(f"⚠️  Invalid TensorBoard GCS path: {tensorboard_gcs_path}")
            return
        
        logger.info(f"📤 Uploading TensorBoard logs to GCS...")
        logger.info(f"   Target: {tensorboard_gcs_path}")
        
        # Parse GCS path
        gcs_path_parts = tensorboard_gcs_path.replace('gs://', '').split('/', 1)
        bucket_name = gcs_path_parts[0]
        gcs_prefix = gcs_path_parts[1] if len(gcs_path_parts) > 1 else ''
        
        # Upload all files from local_log_dir to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Find all files to upload
        files_to_upload = []
        for root, dirs, files in os.walk(local_log_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                files_to_upload.append(local_file_path)
        
        if not files_to_upload:
            logger.warning("   No TensorBoard log files found to upload")
            return
        
        logger.info(f"   Found {len(files_to_upload)} files to upload")
        
        # Upload files
        uploaded_count = 0
        for local_file_path in files_to_upload:
            # Get relative path from local_log_dir
            relative_path = os.path.relpath(local_file_path, local_log_dir)
            # Construct GCS blob path
            if gcs_prefix:
                gcs_blob_path = f"{gcs_prefix}/{relative_path}".replace('\\', '/')
            else:
                gcs_blob_path = relative_path.replace('\\', '/')
            
            blob = bucket.blob(gcs_blob_path)
            blob.upload_from_filename(local_file_path)
            uploaded_count += 1
            if uploaded_count <= 5:  # Log first 5 files
                logger.debug(f"   Uploaded: {gcs_blob_path}")
        
        logger.info(f"✅ Uploaded {uploaded_count} TensorBoard log files to {tensorboard_gcs_path}")
        logger.info(f"   View: https://console.cloud.google.com/vertex-ai/tensorboard?project={project_id}")
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to upload TensorBoard logs to GCS: {e}")
        logger.warning("   Logs are available locally but may not appear in TensorBoard UI")


def save_model(pipeline, feature_cols, metrics, config):
    """Save model and metrics to Cloud Storage."""
    logger.info("💾 Saving model to Cloud Storage...")
    
    bucket_name = config['storage']['bucket']
    
    # Get output directory from Vertex AI or use default
    output_dir = os.environ.get('AIP_MODEL_DIR', '/tmp/model')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model files
    model_path = os.path.join(output_dir, 'model.joblib')
    features_path = os.path.join(output_dir, 'feature_cols.json')
    metrics_path = os.path.join(output_dir, 'metrics.json')
    
    joblib.dump(pipeline, model_path, protocol=4)
    
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, default=str)
    
    logger.info(f"✅ Model saved to {model_path}")
    
    # Also save to standard location for deployment
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
    
    logger.info(f"✅ Model also saved to gs://{bucket_name}/{latest_prefix}")


if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("🎯 VERTEX AI CUSTOM TRAINING JOB")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # Load config
        config = load_config()
        
        # Load data
        df = load_training_data(config)
        
        # Engineer features
        df_enhanced, feature_cols = engineer_features(df, config)
        
        # Train model
        pipeline, metrics, X_test, y_test, y_pred = train_model(df_enhanced, feature_cols, config)
        
        # Create output directory for artifacts
        output_dir = os.environ.get('AIP_MODEL_DIR', '/tmp/model')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create TensorBoard writer
        # Note: SummaryWriter cannot write directly to GCS, so we write locally first
        local_tensorboard_dir = os.path.join(output_dir, 'tensorboard_logs')
        os.makedirs(local_tensorboard_dir, exist_ok=True)
        logger.info(f"📊 Writing TensorBoard logs to local directory: {local_tensorboard_dir}")
        
        # Check if Vertex AI TensorBoard is configured
        tensorboard_gcs_path = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
        if tensorboard_gcs_path:
            logger.info(f"📤 Will upload to Vertex AI TensorBoard: {tensorboard_gcs_path}")
        
        writer = SummaryWriter(log_dir=local_tensorboard_dir)
        
        # Run comprehensive evaluation (logs everything to TensorBoard)
        # Use multiple runs to show progression in scalars
        logger.info("\n🔍 Running comprehensive model evaluation...")
        eval_results = run_comprehensive_evaluation(
            model=pipeline,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            writer=writer,
            step=0,
            num_runs=5  # Log 5 runs for progression visualization
        )
        
        # Also log basic training metrics and hyperparameters to TensorBoard
        log_training_metrics_to_tensorboard(writer, config, metrics, y_test, y_pred)
        
        writer.close()
        logger.info("✅ SummaryWriter closed")
        
        # List files created locally
        logger.info(f"\n📂 Local TensorBoard files created:")
        for root, dirs, files in os.walk(local_tensorboard_dir):
            for file in files:
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                logger.info(f"   {filepath} ({size} bytes)")
        
        # Upload TensorBoard logs to GCS if Vertex AI TensorBoard is configured
        if tensorboard_gcs_path:
            logger.info(f"\n📤 Uploading TensorBoard logs to GCS...")
            logger.info(f"   Source: {local_tensorboard_dir}")
            logger.info(f"   Destination: {tensorboard_gcs_path}")
            
            try:
                # Parse GCS path
                if not tensorboard_gcs_path.startswith('gs://'):
                    logger.error(f"❌ Invalid GCS path: {tensorboard_gcs_path}")
                else:
                    gcs_path_parts = tensorboard_gcs_path.replace('gs://', '').split('/', 1)
                    bucket_name = gcs_path_parts[0]
                    gcs_prefix = gcs_path_parts[1] if len(gcs_path_parts) > 1 else ''
                    
                    # Use storage client to upload files
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(bucket_name)
                    
                    # Find all files to upload
                    uploaded_files = []
                    for root, dirs, files in os.walk(local_tensorboard_dir):
                        for file in files:
                            local_file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(local_file_path, local_tensorboard_dir)
                            
                            # Construct GCS blob path
                            if gcs_prefix:
                                gcs_blob_path = f"{gcs_prefix}/{relative_path}".replace('\\', '/')
                            else:
                                gcs_blob_path = relative_path.replace('\\', '/')
                            
                            blob = bucket.blob(gcs_blob_path)
                            blob.upload_from_filename(local_file_path)
                            uploaded_files.append(gcs_blob_path)
                            
                            if len(uploaded_files) <= 5:  # Log first 5 files
                                logger.info(f"   ✅ Uploaded: {gcs_blob_path}")
                    
                    logger.info(f"✅ Uploaded {len(uploaded_files)} TensorBoard log files to {tensorboard_gcs_path}")
                    
                    # List what was uploaded
                    if uploaded_files:
                        logger.info(f"\n📂 Files in GCS:")
                        for f in uploaded_files[:10]:  # Show first 10
                            logger.info(f"   - {f}")
                        if len(uploaded_files) > 10:
                            logger.info(f"   ... and {len(uploaded_files) - 10} more files")
                    
            except Exception as e:
                logger.error(f"❌ Failed to upload TensorBoard logs: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.info(f"📊 TensorBoard logs saved locally: {local_tensorboard_dir}")
            logger.warning("⚠️  No AIP_TENSORBOARD_LOG_DIR set - logs will not appear in Vertex AI TensorBoard")
        
        # Save model and metrics
        save_model(pipeline, feature_cols, metrics, config)
        
        # Output metrics (Vertex AI will capture this)
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        logger.info("=" * 70)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.1f} minutes")
        logger.info(f"Test accuracy: {metrics['test_accuracy']:.2%}")
        
        # Write metrics for orchestrator to read
        metrics_output = {
            'status': 'success',
            'accuracy': metrics['test_accuracy'],
            'duration_minutes': round(duration, 2),
            'training_samples': len(df),
            'metrics': metrics
        }
        
        print("\n" + "=" * 70)
        print("TRAINING METRICS (JSON):")
        print("=" * 70)
        print(json.dumps(metrics_output, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        
        metrics_output = {
            'status': 'error',
            'error': str(e)
        }
        
        print(json.dumps(metrics_output, indent=2))
        exit(1)

