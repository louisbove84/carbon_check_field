"""
Multi-Model Vertex AI Training Script
======================================
Trains both Random Forest and DNN models on the same data.
Logs each model's metrics to TensorBoard with model-type prefixes.

Features:
- Model registry pattern for easy addition of new models
- TensorBoard logging with rf/ and dnn/ prefixes
- Both models saved to GCS with appropriate naming
- Bayesian hyperparameter tuning for DNN via Keras Tuner
"""

import os
import sys
import json
import logging
import re
from importlib import metadata
import joblib
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import bigquery, storage, aiplatform
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter

# Import local modules
from feature_engineering import engineer_features_dataframe
from tensorboard_logging import (
    run_comprehensive_evaluation,
    log_training_metrics_to_tensorboard,
    log_dnn_training_history_to_tensorboard
)
from models import get_model, get_all_model_types, RandomForestModel, DNNModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_package_versions():
    """Log key package versions for debugging environment mismatches."""
    logger.info("Package Versions:")
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
        'protobuf',
        'tensorflow',
        'keras-tuner'
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
        logger.info("Loaded config from GCS")
        return yaml.safe_load(config_text)
    except Exception as e:
        logger.warning(f"Could not load from GCS: {e}, using local config")
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)


def load_training_data(config):
    """Load training data from BigQuery."""
    logger.info("Loading training data from BigQuery...")
    
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
    
    client = bigquery.Client(project=project_id)
    
    logger.info(f"   Project: {project_id}")
    logger.info(f"   Dataset: {dataset}")
    logger.info(f"   Table: {table}")
    
    try:
        df = client.query(query).to_dataframe()
        
        if len(df) == 0:
            raise ValueError(f"BigQuery table {project_id}.{dataset}.{table} exists but is empty!")
        
        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"   Crops: {df['crop'].value_counts().to_dict()}")
        
        return df
    except Exception as e:
        logger.error(f"Failed to load data from BigQuery: {e}")
        logger.error(f"   Table: {project_id}.{dataset}.{table}")
        logger.error(f"   Location: {region}")
        raise


def engineer_features(df, config):
    """Engineer features using shared module."""
    logger.info("Engineering features...")
    
    df_enhanced, all_features = engineer_features_dataframe(df)
    
    logger.info(f"Created {len(all_features)} features")
    
    return df_enhanced, all_features


def train_models(df, feature_cols, config):
    """
    Train all registered models on the same data.
    
    Returns:
        dict: Model type -> (model, metrics, test_pred)
    """
    logger.info("=" * 70)
    logger.info("MULTI-MODEL TRAINING")
    logger.info("=" * 70)
    
    # Prepare data
    X = df[feature_cols].values
    y = df['crop'].values
    
    logger.info(f"   Training samples: {len(X)}")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Classes: {list(np.unique(y))}")
    
    # Split data (same split for all models)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get model configurations from config
    model_configs = config.get('model', {})
    rf_config = model_configs.get('rf', {})
    dnn_config = model_configs.get('dnn', {})
    
    results = {}
    
    # Train Random Forest
    logger.info("")
    logger.info("-" * 50)
    logger.info("Training Random Forest Model")
    logger.info("-" * 50)
    
    rf_model = RandomForestModel(
        n_estimators=rf_config.get('n_estimators', 100),
        max_depth=rf_config.get('max_depth', 15),
        min_samples_split=rf_config.get('min_samples_split', 5),
        feature_names=list(feature_cols)
    )
    
    rf_train_metrics = rf_model.train(X_train, y_train, X_test, y_test)
    rf_pred = rf_model.predict(X_test)
    rf_test_acc = accuracy_score(y_test, rf_pred)
    
    rf_metrics = {
        'train_accuracy': rf_train_metrics['train_accuracy'],
        'test_accuracy': float(rf_test_acc),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'classification_report': classification_report(y_test, rf_pred, output_dict=True, zero_division=0)
    }
    
    logger.info(f"   RF Test accuracy: {rf_test_acc:.2%}")
    results['rf'] = (rf_model, rf_metrics, rf_pred)
    
    # Train DNN
    logger.info("")
    logger.info("-" * 50)
    logger.info("Training Deep Neural Network Model")
    logger.info("-" * 50)
    
    try:
        dnn_model = DNNModel(
            hidden_layers=dnn_config.get('hidden_layers', [64, 32, 16]),
            dropout_rate=dnn_config.get('dropout_rate', 0.3),
            learning_rate=dnn_config.get('learning_rate', 0.001),
            epochs=dnn_config.get('epochs', 100),
            batch_size=dnn_config.get('batch_size', 32),
            tuner_trials=dnn_config.get('tuner_trials', 15),
            use_tuner=dnn_config.get('use_tuner', True),
            feature_names=list(feature_cols)
        )
        
        dnn_train_metrics = dnn_model.train(X_train, y_train, X_test, y_test)
        dnn_pred = dnn_model.predict(X_test)
        dnn_test_acc = accuracy_score(y_test, dnn_pred)
        
        dnn_metrics = {
            'train_accuracy': dnn_train_metrics['train_accuracy'],
            'test_accuracy': float(dnn_test_acc),
            'val_accuracy': dnn_train_metrics.get('val_accuracy', dnn_test_acc),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'classification_report': classification_report(y_test, dnn_pred, output_dict=True, zero_division=0),
            'best_params': dnn_train_metrics.get('best_params', {})
        }
        
        logger.info(f"   DNN Test accuracy: {dnn_test_acc:.2%}")
        results['dnn'] = (dnn_model, dnn_metrics, dnn_pred)
        
    except ImportError as e:
        logger.warning(f"Could not train DNN model: {e}")
        logger.warning("TensorFlow/Keras not available - skipping DNN training")
    except Exception as e:
        logger.error(f"DNN training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Summary comparison
    logger.info("")
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)
    for model_type, (model, metrics, _) in results.items():
        logger.info(f"   {model.model_name}: {metrics['test_accuracy']:.2%}")
    
    best_model_type = max(results.keys(), key=lambda k: results[k][1]['test_accuracy'])
    best_acc = results[best_model_type][1]['test_accuracy']
    logger.info(f"   Best model: {results[best_model_type][0].model_name} ({best_acc:.2%})")
    
    return results, X_test, y_test, X_train, y_train


def save_models(models_dict, feature_cols, config):
    """
    Save all trained models to GCS.
    
    Creates:
        gs://bucket/models/crop_classifier_latest/model_rf.pkl
        gs://bucket/models/crop_classifier_latest/model_dnn.keras
        gs://bucket/models/crop_classifier_latest/feature_cols.json
        gs://bucket/models/crop_classifier_latest/metrics.json
    """
    logger.info("Saving models...")
    
    bucket_name = config['storage']['bucket']
    
    # Always use local temp directory for saving (AIP_MODEL_DIR might be GCS path)
    # This ensures os.walk works correctly when uploading to our bucket
    output_dir = '/tmp/model_output'
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"   Using local output directory: {output_dir}")
    
    # Save feature columns (shared)
    features_path = os.path.join(output_dir, 'feature_cols.json')
    with open(features_path, 'w') as f:
        json.dump(list(feature_cols), f)
    
    # Combined metrics
    combined_metrics = {}
    
    # Save each model
    for model_type, (model, metrics, _) in models_dict.items():
        logger.info(f"   Saving {model.model_name}...")
        model.save(output_dir)
        combined_metrics[model_type] = metrics
    
    # Save combined metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(combined_metrics, f, default=str)
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Log what files exist in output_dir before upload
    logger.info(f"   Files in output directory ({output_dir}):")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        logger.info(f"      {indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            logger.info(f"      {subindent}{file} ({file_size:,} bytes)")
    
    # Recursively upload all files from output_dir
    def upload_directory(local_dir, gcs_prefix):
        """Upload all files from local directory to GCS."""
        uploaded_count = 0
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir)
                gcs_path = f"{gcs_prefix}/{rel_path}"
                try:
                    bucket.blob(gcs_path).upload_from_filename(local_path)
                    logger.info(f"      Uploaded: {gcs_path}")
                    uploaded_count += 1
                except Exception as e:
                    logger.error(f"      Failed to upload {gcs_path}: {e}")
        logger.info(f"      Total files uploaded: {uploaded_count}")
    
    # Archive
    archive_prefix = f'models/crop_classifier_archive/crop_classifier_{timestamp}'
    logger.info(f"   Archiving to: gs://{bucket_name}/{archive_prefix}/")
    upload_directory(output_dir, archive_prefix)
    
    # Latest
    latest_prefix = 'models/crop_classifier_latest'
    logger.info(f"   Saving latest to: gs://{bucket_name}/{latest_prefix}/")
    upload_directory(output_dir, latest_prefix)
    
    logger.info(f"Models saved to gs://{bucket_name}/{latest_prefix}/")


def log_gcs_tensorboard_structure(bucket_name, gcs_prefix, experiment_id=None, max_runs=10, max_files=5, stage=""):
    """Log TensorBoard folder structure in GCS for debugging UI visibility."""
    try:
        storage_client = storage.Client()
        base_prefix = f"{gcs_prefix}/" if gcs_prefix else ""
        tb_prefix = f"{base_prefix}{experiment_id}/" if experiment_id else base_prefix
        stage_label = f" ({stage})" if stage else ""

        logger.info(f"TensorBoard GCS structure{stage_label}:")
        logger.info(f"   Bucket: {bucket_name}")
        logger.info(f"   Prefix: {tb_prefix or '[root]'}")

        run_prefixes = []
        iterator = storage_client.list_blobs(
            bucket_name,
            prefix=tb_prefix,
            delimiter='/'
        )
        for prefix in iterator.prefixes:
            run_prefixes.append(prefix)

        if not run_prefixes:
            logger.info("   No run folders found under this prefix.")
            return

        logger.info(f"   Run folders found: {len(run_prefixes)}")
        for prefix in run_prefixes[:max_runs]:
            logger.info(f"   - {prefix}")

        recent_prefix = run_prefixes[-1]
        logger.info(f"   Inspecting latest run: {recent_prefix}")
        files_listed = 0
        for blob in storage_client.list_blobs(bucket_name, prefix=recent_prefix):
            logger.info(f"      {blob.name} ({blob.size:,} bytes)")
            files_listed += 1
            if files_listed >= max_files:
                break
        if files_listed == 0:
            logger.info("      (No files found under latest run prefix)")
    except Exception as e:
        logger.warning(f"Could not inspect TensorBoard GCS structure: {e}")


if __name__ == '__main__':
    # Clear Python cache
    import shutil
    
    cache_dirs = ['/app/__pycache__', '/tmp/__pycache__']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"Cleared Python cache: {cache_dir}")
            except Exception as e:
                print(f"Could not clear cache {cache_dir}: {e}")
    
    logger.info("=" * 70)
    logger.info("VERTEX AI MULTI-MODEL TRAINING")
    logger.info("=" * 70)
    
    logger.info("Environment Variables:")
    logger.info(f"   AIP_MODEL_DIR: {os.environ.get('AIP_MODEL_DIR', 'NOT SET')}")
    logger.info(f"   AIP_TENSORBOARD_LOG_DIR: {os.environ.get('AIP_TENSORBOARD_LOG_DIR', 'NOT SET')}")
    logger.info(f"   AIP_TENSORBOARD_EXPERIMENT_NAME: {os.environ.get('AIP_TENSORBOARD_EXPERIMENT_NAME', 'NOT SET')}")
    logger.info("")

    log_package_versions()
    
    # Force reload modules
    import importlib
    modules_to_reload = ['tensorboard_logging', 'feature_engineering', 'visualization_utils']
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            try:
                importlib.reload(sys.modules[mod_name])
                logger.info(f"Reloaded module: {mod_name}")
            except Exception as e:
                logger.warning(f"Could not reload {mod_name}: {e}")
    
    start_time = datetime.now()
    
    try:
        # Load config
        config = load_config()
        
        # Load data
        df = load_training_data(config)
        
        # Normalize crop labels
        df['crop'] = df['crop'].apply(normalize_crop_labels)
        logger.info(f"Normalized crop labels: {df['crop'].unique().tolist()}")
        
        # Engineer features
        df_enhanced, feature_cols = engineer_features(df, config)
        
        # Train all models
        models_dict, X_test, y_test, X_train, y_train = train_models(df_enhanced, feature_cols, config)
        
        # === TENSORBOARD LOGGING ===
        managed_tb_gcs_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
        managed_tb_gcs_dir_clean = managed_tb_gcs_dir.rstrip('/') if managed_tb_gcs_dir else None
        experiment_name = os.environ.get('AIP_TENSORBOARD_EXPERIMENT_NAME', 'crop-training')
        experiment_name = experiment_name.lower()
        
        match = re.match(r'^crop-training-(\d{14})$', experiment_name)
        if match:
            date_only = match.group(1)[:8]
            experiment_name = f'crop-training-{date_only}'
        experiment_id = re.sub(r'[^a-z0-9-]', '-', experiment_name)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        run_name = f'run-{timestamp}'
        run_id = run_name
        local_tb_base = '/tmp/tensorboard_logs'
        local_tb_dir = os.path.join(local_tb_base, experiment_id, run_id)
        os.makedirs(local_tb_dir, exist_ok=True)
        
        logger.info(f"TensorBoard Configuration:")
        if managed_tb_gcs_dir:
            logger.info(f"   AIP_TENSORBOARD_LOG_DIR (GCS): {managed_tb_gcs_dir_clean}")
            logger.info(f"   Local write directory: {local_tb_dir}")
            logger.info(f"   Experiment: {experiment_name} (id: {experiment_id})")
            logger.info(f"   Run name: {run_name} (id: {run_id})")
        else:
            logger.warning("AIP_TENSORBOARD_LOG_DIR not set")
            logger.info(f"   Local write directory: {local_tb_dir}")
        
        # Safety checks
        aip_model_dir = os.environ.get('AIP_MODEL_DIR', '')
        assert not local_tb_dir.startswith('gs://'), f"TensorBoard log_dir cannot be a GCS path: {local_tb_dir}"
        assert '/tmp' in local_tb_dir, f"TensorBoard must write to /tmp, not: {local_tb_dir}"
        
        writer = SummaryWriter(log_dir=local_tb_dir)
        logger.info(f"SummaryWriter created for: {local_tb_dir}")
        
        # Initialize Vertex AI Experiments
        project_number = '303566498201'
        tensorboard_id = config.get('tensorboard', {}).get('instance_id')
        
        if tensorboard_id:
            tensorboard_resource = f'projects/{project_number}/locations/{config["project"]["region"]}/tensorboards/{tensorboard_id}'
            logger.info(f"Linking to TensorBoard instance: {tensorboard_id}")
        else:
            tensorboard_resource = None
            logger.warning("No TensorBoard instance ID in config")
        
        try:
            logger.info(f"Initializing Vertex AI Experiment: {experiment_name}")
            init_params = {
                'project': config['project']['id'],
                'location': config['project']['region'],
                'experiment': experiment_id
            }
            if tensorboard_resource:
                init_params['experiment_tensorboard'] = tensorboard_resource
            
            aiplatform.init(**init_params)
            aiplatform.start_run(run=run_id)
            logger.info(f"Started run: {run_id}")
        except Exception as e:
            logger.warning(f"Could not start Vertex AI Experiment: {e}")
        
        # Log evaluation for each model
        logger.info("")
        logger.info("=" * 70)
        logger.info("TENSORBOARD LOGGING")
        logger.info("=" * 70)
        
        all_metrics = {}
        
        for model_type, (model, metrics, y_pred) in models_dict.items():
            logger.info("")
            logger.info(f"Logging {model.model_name} to TensorBoard (prefix: {model_type}/)...")
            
            # Run comprehensive evaluation with model prefix
            eval_results = run_comprehensive_evaluation(
                model=model,
                X_test=X_test,
                y_test=y_test,
                feature_names=list(feature_cols),
                writer=writer,
                model_prefix=model_type,
                step=0,
                num_runs=1
            )
            
            # Log training metrics
            log_training_metrics_to_tensorboard(
                writer, config, metrics, y_test, y_pred,
                model_prefix=model_type,
                model_params=model.get_params()
            )
            
            # Log DNN training history if available
            if model_type == 'dnn' and hasattr(model, 'get_training_history'):
                history = model.get_training_history()
                if history:
                    log_dnn_training_history_to_tensorboard(writer, history, model_prefix=model_type)
            
            all_metrics[model_type] = {
                'test_accuracy': metrics['test_accuracy'],
                'train_accuracy': metrics['train_accuracy']
            }
        
        # Log comparison scalars
        writer.add_scalar('comparison/rf_accuracy', all_metrics.get('rf', {}).get('test_accuracy', 0), 0)
        if 'dnn' in all_metrics:
            writer.add_scalar('comparison/dnn_accuracy', all_metrics['dnn']['test_accuracy'], 0)
        
        # Log metrics to Vertex AI Experiments
        try:
            vertex_metrics = {
                'rf_accuracy': all_metrics.get('rf', {}).get('test_accuracy', 0),
                'num_features': len(feature_cols),
                'num_samples': len(df_enhanced)
            }
            if 'dnn' in all_metrics:
                vertex_metrics['dnn_accuracy'] = all_metrics['dnn']['test_accuracy']
            
            aiplatform.log_metrics(vertex_metrics)
            logger.info("Logged metrics to Vertex AI Experiments")
        except Exception as e:
            logger.warning(f"Could not log metrics to Vertex AI: {e}")
        
        # Flush and close writer
        logger.info("Flushing TensorBoard writer...")
        writer.flush()
        
        import time
        logger.info("Waiting 5 seconds for files to write...")
        time.sleep(5)
        
        try:
            aiplatform.end_run()
            logger.info("Ended Vertex AI Experiment run")
        except Exception as e:
            logger.warning(f"Could not end Vertex AI run: {e}")
        
        writer.close()
        logger.info("TensorBoard writer closed")
        
        # Check local files
        logger.info(f"Checking TensorBoard files in {local_tb_dir}")
        local_files = []
        if os.path.exists(local_tb_dir) and os.path.isdir(local_tb_dir):
            for root, dirs, files in os.walk(local_tb_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        try:
                            size = os.path.getsize(file_path)
                            rel_path = os.path.relpath(file_path, local_tb_dir)
                            local_files.append((rel_path, size))
                            logger.info(f"   {rel_path}: {size:,} bytes")
                        except Exception as e:
                            logger.warning(f"Could not get size for {file_path}: {e}")
        
        if not local_files:
            logger.error("No TensorBoard files found in local directory!")
        else:
            logger.info(f"TensorBoard files created locally: {len(local_files)} files")
        
        # Upload to GCS
        if managed_tb_gcs_dir_clean and local_files:
            logger.info(f"Uploading TensorBoard logs to GCS")
            try:
                if managed_tb_gcs_dir_clean.startswith('gs://'):
                    gcs_path = managed_tb_gcs_dir_clean.replace('gs://', '')
                    parts = gcs_path.split('/', 1)
                    bucket_name = parts[0]
                    gcs_prefix = parts[1].rstrip('/') if len(parts) > 1 else ''
                    
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(bucket_name)
                    
                    uploaded_count = 0
                    for root, dirs, files in os.walk(local_tb_dir):
                        for file in files:
                            local_path = os.path.join(root, file)
                            rel_path = os.path.relpath(local_path, local_tb_dir)
                            rel_path = rel_path.lstrip('./').replace('\\', '/')
                            if gcs_prefix:
                                blob_path = f"{gcs_prefix}/{experiment_id}/{run_id}/{rel_path}"
                            else:
                                blob_path = f"{experiment_id}/{run_id}/{rel_path}"
                            
                            blob = bucket.blob(blob_path)
                            blob.upload_from_filename(local_path)
                            uploaded_count += 1
                    
                    logger.info(f"Uploaded {uploaded_count} TensorBoard files to GCS")
                    logger.info(f"   GCS location: gs://{bucket_name}/{gcs_prefix}/{experiment_id}/{run_id}/")
            except Exception as e:
                logger.error(f"Failed to upload TensorBoard logs to GCS: {e}")
        
        # Wait for sync
        logger.info("Waiting 10 seconds for Vertex AI to sync files...")
        time.sleep(10)

        # Save models
        save_models(models_dict, feature_cols, config)
        
        # Output final metrics
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.1f} minutes")
        for model_type, (model, metrics, _) in models_dict.items():
            logger.info(f"   {model.model_name}: {metrics['test_accuracy']:.2%}")
        
        # Best model
        best_type = max(models_dict.keys(), key=lambda k: models_dict[k][1]['test_accuracy'])
        best_model_name = models_dict[best_type][0].model_name
        best_acc = models_dict[best_type][1]['test_accuracy']
        logger.info(f"Best model: {best_model_name} ({best_acc:.2%})")
        
        # Metrics for orchestrator
        output_metrics = {
            'status': 'success',
            'best_model': best_type,
            'best_accuracy': best_acc,
            'duration_minutes': round(duration, 2),
            'training_samples': len(df_enhanced),
            'models': {
                model_type: {
                    'accuracy': metrics['test_accuracy'],
                    'train_accuracy': metrics['train_accuracy']
                }
                for model_type, (_, metrics, _) in models_dict.items()
            }
        }
        
        print("\n" + "=" * 70)
        print("TRAINING METRICS (JSON):")
        print("=" * 70)
        print(json.dumps(output_metrics, indent=2, default=str))
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        
        print(json.dumps({
            'status': 'error',
            'error': str(e)
        }, indent=2))
        
        exit(1)
