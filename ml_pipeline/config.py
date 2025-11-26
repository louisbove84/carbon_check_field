"""
Centralized Configuration for ML Pipeline
==========================================
Configuration is loaded from Cloud Storage (config.yaml) if available,
otherwise falls back to environment variables, then hardcoded defaults.

Priority:
  1. Cloud Storage: gs://carboncheck-data/config/config.yaml
  2. Environment variables
  3. Hardcoded defaults

Usage:
    from config import PROJECT_ID, REGION, BUCKET_NAME, etc.

To update config without redeployment:
    1. Edit config.yaml
    2. Run: ./upload_config.sh
    3. Next function execution uses new config
"""

import os
import yaml
import logging
from google.cloud import storage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# LOAD CONFIGURATION FROM GCS
# ============================================================

def load_config_from_gcs():
    """
    Load configuration from Cloud Storage YAML file.
    Returns dict with config or None if not available.
    """
    try:
        # Default bucket and path (can be overridden by env var)
        bucket_name = os.environ.get('CONFIG_BUCKET', 'carboncheck-data')
        config_path = os.environ.get('CONFIG_PATH', 'config/config.yaml')
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(config_path)
        
        # Download and parse YAML
        yaml_content = blob.download_as_text()
        config = yaml.safe_load(yaml_content)
        
        logger.info(f"✅ Configuration loaded from gs://{bucket_name}/{config_path}")
        return config
    
    except Exception as e:
        logger.warning(f"⚠️  Could not load config from GCS: {e}")
        logger.info("   Falling back to environment variables and defaults")
        return None


# Try to load from GCS
_gcs_config = load_config_from_gcs()

# ============================================================
# HELPER TO GET CONFIG VALUES (GCS -> ENV -> DEFAULT)
# ============================================================

def get_config(gcs_path, env_var, default):
    """
    Get configuration value with priority:
    1. GCS YAML config
    2. Environment variable
    3. Default value
    """
    # Try GCS config first
    if _gcs_config:
        try:
            value = _gcs_config
            for key in gcs_path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            pass
    
    # Fall back to environment variable
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return env_value
    
    # Fall back to default
    return default


# ============================================================
# GOOGLE CLOUD PROJECT CONFIGURATION
# ============================================================

PROJECT_ID = get_config('project.id', 'PROJECT_ID', 'ml-pipeline-477612')
REGION = get_config('project.region', 'REGION', 'us-central1')

# ============================================================
# STORAGE & DATA CONFIGURATION
# ============================================================

BUCKET_NAME = get_config('storage.bucket', 'BUCKET_NAME', 'carboncheck-data')
DATASET_ID = get_config('bigquery.dataset', 'DATASET_ID', 'crop_ml')

# BigQuery Tables
TRAINING_TABLE_ID = get_config('bigquery.tables.training', 'TRAINING_TABLE_ID', 'training_features')
HOLDOUT_TABLE_ID = get_config('bigquery.tables.holdout', 'HOLDOUT_TABLE_ID', 'holdout_test_set')
METRICS_TABLE_ID = get_config('bigquery.tables.metrics', 'METRICS_TABLE_ID', 'model_performance')
DEPLOYMENT_TABLE_ID = get_config('bigquery.tables.deployment', 'DEPLOYMENT_TABLE_ID', 'deployment_history')

# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Vertex AI Model Names (consistent across deployments)
MODEL_NAME = get_config('model.name', 'MODEL_NAME', 'crop-classifier-latest')
ENDPOINT_NAME = get_config('model.endpoint_name', 'ENDPOINT_NAME', 'crop-endpoint')

# Model Training Parameters
N_ESTIMATORS = int(get_config('model.hyperparameters.n_estimators', 'N_ESTIMATORS', 100))
MAX_DEPTH = int(get_config('model.hyperparameters.max_depth', 'MAX_DEPTH', 10))
MIN_SAMPLES_SPLIT = int(get_config('model.hyperparameters.min_samples_split', 'MIN_SAMPLES_SPLIT', 5))
RANDOM_STATE = int(get_config('model.hyperparameters.random_state', 'RANDOM_STATE', 42))

# ============================================================
# DATA COLLECTION CONFIGURATION
# ============================================================

# Number of samples to collect per crop each month
SAMPLES_PER_CROP = int(get_config('data_collection.samples_per_crop', 'SAMPLES_PER_CROP', 100))

# ============================================================
# EVALUATION & DEPLOYMENT GATES
# ============================================================

# Minimum accuracy required for deployment
ABSOLUTE_MIN_ACCURACY = float(get_config('quality_gates.absolute_min_accuracy', 'ABSOLUTE_MIN_ACCURACY', 0.75))

# Minimum F1 score required per crop
MIN_PER_CROP_F1 = float(get_config('quality_gates.min_per_crop_f1', 'MIN_PER_CROP_F1', 0.70))

# How much better challenger must be than champion (e.g., 0.02 = 2%)
IMPROVEMENT_MARGIN = float(get_config('quality_gates.improvement_margin', 'IMPROVEMENT_MARGIN', 0.02))

# Minimum samples required for training
MIN_TRAINING_SAMPLES = int(get_config('quality_gates.min_training_samples', 'MIN_TRAINING_SAMPLES', 100))

# Minimum holdout test samples required
MIN_TEST_SAMPLES = int(get_config('quality_gates.min_test_samples', 'MIN_TEST_SAMPLES', 50))

# ============================================================
# FEATURE CONFIGURATION
# ============================================================

# Base features from BigQuery (NDVI + location)
_default_features = [
    'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max',
    'ndvi_p25', 'ndvi_p50', 'ndvi_p75',
    'ndvi_early', 'ndvi_late',
    'elevation_m', 'longitude', 'latitude'
]
BASE_FEATURE_COLUMNS = get_config('features.base_columns', 'BASE_FEATURE_COLUMNS', _default_features)

# ============================================================
# CROP CONFIGURATION
# ============================================================

# Crops to train on (with CDL codes and counties)
_default_crops = [
    {
        'name': 'Corn',
        'code': 1,
        'counties': ['17113', '17019', '17105', '18141', '18003', '19169', '19153', '19013', '27079', '27165']
    },
    {
        'name': 'Soybeans',
        'code': 5,
        'counties': ['17113', '17019', '18141', '18003', '18033', '19169', '19153', '19013', '27079', '27165']
    },
    {
        'name': 'Alfalfa',
        'code': 36,
        'counties': ['06025', '06107', '06099', '06111', '53077', '53003', '16083', '08123', '08069']
    },
    {
        'name': 'Winter Wheat',
        'code': 24,
        'counties': ['20095', '20199', '20155', '20051', '20165', '40047', '40011', '48179', '31081', '31089']
    }
]
CROPS = get_config('data_collection.crops', 'CROPS', _default_crops)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_model_config() -> dict:
    """Get RandomForest model configuration."""
    return {
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'min_samples_split': MIN_SAMPLES_SPLIT,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }


def get_full_table_name(table_id: str) -> str:
    """Get fully qualified BigQuery table name."""
    return f"{PROJECT_ID}.{DATASET_ID}.{table_id}"


def print_config():
    """Print current configuration (for debugging)."""
    print("=" * 60)
    print("CURRENT CONFIGURATION")
    print("=" * 60)
    print(f"Project ID:          {PROJECT_ID}")
    print(f"Region:              {REGION}")
    print(f"Bucket:              {BUCKET_NAME}")
    print(f"Dataset:             {DATASET_ID}")
    print(f"Model Name:          {MODEL_NAME}")
    print(f"Endpoint Name:       {ENDPOINT_NAME}")
    print(f"")
    print(f"Quality Gates:")
    print(f"  Min Accuracy:      {ABSOLUTE_MIN_ACCURACY:.0%}")
    print(f"  Min Crop F1:       {MIN_PER_CROP_F1:.0%}")
    print(f"  Improvement:       {IMPROVEMENT_MARGIN:.0%}")
    print(f"")
    print(f"Data Collection:")
    print(f"  Samples/Crop:      {SAMPLES_PER_CROP}")
    print("=" * 60)


if __name__ == '__main__':
    # For testing: print current config
    print_config()

