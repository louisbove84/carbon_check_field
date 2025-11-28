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
    """Create derived features."""
    logger.info("🔧 Engineering features...")
    
    df = df.copy()
    
    # Derived features
    df['ndvi_range'] = df['ndvi_max'] - df['ndvi_min']
    df['ndvi_iqr'] = df['ndvi_p75'] - df['ndvi_p25']
    df['ndvi_change'] = df['ndvi_late'] - df['ndvi_early']
    df['ndvi_early_ratio'] = df['ndvi_early'] / (df['ndvi_mean'] + 0.001)
    df['ndvi_late_ratio'] = df['ndvi_late'] / (df['ndvi_mean'] + 0.001)
    
    # All features
    base_features = config['features']['base_columns']
    all_features = base_features + [
        'ndvi_range', 'ndvi_iqr', 'ndvi_change',
        'ndvi_early_ratio', 'ndvi_late_ratio'
    ]
    
    logger.info(f"✅ Created {len(all_features)} features")
    
    return df, all_features


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
    
    return pipeline, metrics, y_test, y_pred


def generate_confusion_matrix(y_true, y_pred, labels, output_path):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Confusion matrix saved to {output_path}")


def log_to_tensorboard(config, metrics, y_test, y_pred, output_dir):
    """Log metrics and visualizations to TensorBoard."""
    try:
        project_id = config['project']['id']
        
        # Use Vertex AI's managed TensorBoard path if available
        # This environment variable is set automatically when the training job
        # is associated with a TensorBoard instance
        tensorboard_gcs_path = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
        
        # Debug: Log all TensorBoard-related environment variables
        logger.info("🔍 Checking TensorBoard environment variables...")
        tb_env_vars = {k: v for k, v in os.environ.items() if 'TENSORBOARD' in k or 'AIP' in k}
        if tb_env_vars:
            logger.info(f"   Found TensorBoard env vars: {list(tb_env_vars.keys())}")
        else:
            logger.warning("   No TensorBoard environment variables found")
        
        # SummaryWriter cannot write directly to GCS, so we write locally first
        # then upload to GCS
        local_log_dir = os.path.join(output_dir, 'tensorboard_logs')
        os.makedirs(local_log_dir, exist_ok=True)
        
        if tensorboard_gcs_path:
            logger.info(f"📊 Using Vertex AI managed TensorBoard: {tensorboard_gcs_path}")
            logger.info(f"   Writing logs locally to: {local_log_dir}")
        else:
            # Fallback to custom GCS path
            bucket_name = config['storage']['bucket']
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tensorboard_gcs_path = f"gs://{bucket_name}/tensorboard_logs/{run_name}"
            logger.warning(f"⚠️  AIP_TENSORBOARD_LOG_DIR not set, using fallback path")
            logger.info(f"📊 Using custom TensorBoard path: {tensorboard_gcs_path}")
            logger.info(f"   Writing logs locally to: {local_log_dir}")
        
        # Create TensorBoard writer (must use local path)
        writer = SummaryWriter(log_dir=local_log_dir)
        
        # Log hyperparameters
        hparams = {
            'n_estimators': config['model']['hyperparameters']['n_estimators'],
            'max_depth': config['model']['hyperparameters']['max_depth'],
            'min_samples_split': config['model']['hyperparameters']['min_samples_split'],
            'n_train_samples': metrics['n_train_samples'],
            'n_test_samples': metrics['n_test_samples']
        }
        
        # Log scalar metrics
        writer.add_scalar('accuracy/train', metrics['train_accuracy'], 0)
        writer.add_scalar('accuracy/test', metrics['test_accuracy'], 0)
        
        # Log per-crop metrics
        report = metrics['classification_report']
        crops = sorted([c for c in set(y_test)])
        
        for crop in crops:
            if crop in report:
                writer.add_scalar(f'f1_score/{crop}', report[crop]['f1-score'], 0)
                writer.add_scalar(f'precision/{crop}', report[crop]['precision'], 0)
                writer.add_scalar(f'recall/{crop}', report[crop]['recall'], 0)
        
        # Generate and log confusion matrix as image
        cm = confusion_matrix(y_test, y_pred, labels=crops)
        
        # Create confusion matrix figure
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=crops, yticklabels=crops, ax=ax)
        ax.set_title('Confusion Matrix', fontsize=16, pad=20)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        # Convert matplotlib figure to image tensor for TensorBoard
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        
        # TensorBoard expects (C, H, W) format
        if len(image_array.shape) == 3:
            image_array = np.transpose(image_array, (2, 0, 1))
        
        writer.add_image('confusion_matrix', image_array, 0)
        plt.close(fig)
        
        # Also save confusion matrix to output directory for GCS upload
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        generate_confusion_matrix(y_test, y_pred, crops, cm_path)
        
        # Log text summary
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
        
        # Save classification report
        report_path = os.path.join(output_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        writer.close()
        
        # Upload TensorBoard logs to GCS if TensorBoard path is provided
        if tensorboard_gcs_path and tensorboard_gcs_path.startswith('gs://'):
            logger.info(f"📤 Uploading TensorBoard logs to GCS...")
            try:
                # storage is already imported at top of file
                
                # Parse GCS path
                # Format: gs://bucket-name/path/to/logs
                gcs_path_parts = tensorboard_gcs_path.replace('gs://', '').split('/', 1)
                bucket_name = gcs_path_parts[0]
                gcs_prefix = gcs_path_parts[1] if len(gcs_path_parts) > 1 else ''
                
                # Upload all files from local_log_dir to GCS
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                
                # Check if local directory has files
                files_to_upload = []
                for root, dirs, files in os.walk(local_log_dir):
                    for file in files:
                        local_file_path = os.path.join(root, file)
                        files_to_upload.append(local_file_path)
                
                if not files_to_upload:
                    logger.warning("   No TensorBoard log files found to upload")
                else:
                    logger.info(f"   Found {len(files_to_upload)} files to upload")
                    
                    # Walk through local directory and upload files
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
            except Exception as upload_error:
                logger.warning(f"⚠️  Failed to upload TensorBoard logs to GCS: {upload_error}")
                logger.warning("   Logs are available locally but may not appear in TensorBoard UI")
        
        logger.info("✅ Logged to TensorBoard")
        logger.info(f"   Local logs: {local_log_dir}")
        if tensorboard_gcs_path:
            logger.info(f"   GCS logs: {tensorboard_gcs_path}")
        logger.info(f"   View: https://console.cloud.google.com/vertex-ai/tensorboard?project={project_id}")
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to log to TensorBoard: {e}")
        logger.warning("   Training will continue, but metrics won't be in TensorBoard")
        import traceback
        traceback.print_exc()


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
        pipeline, metrics, y_test, y_pred = train_model(df_enhanced, feature_cols, config)
        
        # Create output directory for artifacts
        output_dir = os.environ.get('AIP_MODEL_DIR', '/tmp/model')
        os.makedirs(output_dir, exist_ok=True)
        
        # Log to TensorBoard
        log_to_tensorboard(config, metrics, y_test, y_pred, output_dir)
        
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

