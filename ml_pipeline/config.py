"""
Centralized Configuration for ML Pipeline
==========================================
All configuration values are read from environment variables.
This allows changing config without modifying code.

Usage:
    from config import PROJECT_ID, REGION, BUCKET_NAME, etc.

Environment variables are set during Cloud Function deployment:
    gcloud functions deploy FUNCTION_NAME \
      --set-env-vars PROJECT_ID=ml-pipeline-477612,REGION=us-central1,...
"""

import os

# ============================================================
# GOOGLE CLOUD PROJECT CONFIGURATION
# ============================================================

PROJECT_ID = os.environ.get('PROJECT_ID', 'ml-pipeline-477612')
REGION = os.environ.get('REGION', 'us-central1')

# ============================================================
# STORAGE & DATA CONFIGURATION
# ============================================================

BUCKET_NAME = os.environ.get('BUCKET_NAME', 'carboncheck-data')
DATASET_ID = os.environ.get('DATASET_ID', 'crop_ml')

# BigQuery Tables
TRAINING_TABLE_ID = os.environ.get('TRAINING_TABLE_ID', 'training_features')
HOLDOUT_TABLE_ID = os.environ.get('HOLDOUT_TABLE_ID', 'holdout_test_set')
METRICS_TABLE_ID = os.environ.get('METRICS_TABLE_ID', 'model_performance')
DEPLOYMENT_TABLE_ID = os.environ.get('DEPLOYMENT_TABLE_ID', 'deployment_history')

# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Vertex AI Model Names (consistent across deployments)
MODEL_NAME = os.environ.get('MODEL_NAME', 'crop-classifier-latest')
ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 'crop-endpoint')

# Model Training Parameters
N_ESTIMATORS = int(os.environ.get('N_ESTIMATORS', '100'))
MAX_DEPTH = int(os.environ.get('MAX_DEPTH', '10'))
MIN_SAMPLES_SPLIT = int(os.environ.get('MIN_SAMPLES_SPLIT', '5'))
RANDOM_STATE = int(os.environ.get('RANDOM_STATE', '42'))

# ============================================================
# DATA COLLECTION CONFIGURATION
# ============================================================

# Number of samples to collect per crop each month
SAMPLES_PER_CROP = int(os.environ.get('SAMPLES_PER_CROP', '100'))

# ============================================================
# EVALUATION & DEPLOYMENT GATES
# ============================================================

# Minimum accuracy required for deployment
ABSOLUTE_MIN_ACCURACY = float(os.environ.get('ABSOLUTE_MIN_ACCURACY', '0.75'))

# Minimum F1 score required per crop
MIN_PER_CROP_F1 = float(os.environ.get('MIN_PER_CROP_F1', '0.70'))

# How much better challenger must be than champion (e.g., 0.02 = 2%)
IMPROVEMENT_MARGIN = float(os.environ.get('IMPROVEMENT_MARGIN', '0.02'))

# Minimum samples required for training
MIN_TRAINING_SAMPLES = int(os.environ.get('MIN_TRAINING_SAMPLES', '100'))

# Minimum holdout test samples required
MIN_TEST_SAMPLES = int(os.environ.get('MIN_TEST_SAMPLES', '50'))

# ============================================================
# FEATURE CONFIGURATION
# ============================================================

# Base features from BigQuery (NDVI + location)
BASE_FEATURE_COLUMNS = [
    'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max',
    'ndvi_p25', 'ndvi_p50', 'ndvi_p75',
    'ndvi_early', 'ndvi_late',
    'elevation_m', 'longitude', 'latitude'
]

# ============================================================
# CROP CONFIGURATION
# ============================================================

# Crops to train on (with CDL codes and counties)
CROPS = [
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

