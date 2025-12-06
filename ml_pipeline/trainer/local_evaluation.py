"""
Standalone script to evaluate an existing trained model.
This allows running comprehensive evaluation without retraining or data collection.

Usage:
    python local_evaluation.py [--model-path MODEL_PATH] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import logging
import joblib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from tensorboard_utils import run_comprehensive_evaluation
from feature_engineering import engineer_features_dataframe, compute_elevation_quantiles
from torch.utils.tensorboard import SummaryWriter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')



def load_config(config_path=None, bucket_name=None):
    """Load configuration from config.yaml, GCS, or environment."""
    try:
        # Try local file first
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Try default local path
        local_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        if os.path.exists(local_config_path):
            import yaml
            with open(local_config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Try loading from GCS (same as vertex_ai_training.py)
        try:
            from google.cloud import storage
            import yaml
            client = storage.Client()
            bucket = client.bucket('carboncheck-data')
            blob = bucket.blob('config/config.yaml')
            yaml_content = blob.download_as_text()
            logger.info("✅ Config loaded from Cloud Storage")
            return yaml.safe_load(yaml_content)
        except Exception as gcs_error:
            logger.warning(f"Could not load config from GCS: {gcs_error}")
        
        # Use defaults
        logger.warning("config.yaml not found, using defaults")
        return {
            'project': {'id': os.environ.get('GCP_PROJECT_ID', 'ml-pipeline-477612')},
            'bigquery': {
                'dataset': 'crop_ml',
                'tables': {'training': 'training_features'}
            },
            'storage': {'bucket': 'carboncheck-data'},
            'features': {
                'elevation_quantiles': {
                    'q25': 0.0,
                    'q50': 0.0,
                    'q75': 0.0
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def load_model_from_gcs(model_path, bucket_name):
    """Load a trained model from GCS."""
    from google.cloud import storage
    
    logger.info(f"📦 Loading model from gs://{bucket_name}/{model_path}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Download model file
    blob = bucket.blob(f"{model_path}/model.joblib")
    local_model_path = "/tmp/model.joblib"
    blob.download_to_filename(local_model_path)
    logger.info(f"   Downloaded model to {local_model_path}")
    
    # Download feature columns
    try:
        blob = bucket.blob(f"{model_path}/feature_cols.json")
        local_features_path = "/tmp/feature_cols.json"
        blob.download_to_filename(local_features_path)
        with open(local_features_path, 'r') as f:
            feature_cols = json.load(f)
        logger.info(f"   Loaded {len(feature_cols)} feature columns")
    except Exception as e:
        logger.warning(f"   Could not load feature_cols.json: {e}")
        feature_cols = None
    
    # Load model
    model = joblib.load(local_model_path)
    logger.info(f"✅ Model loaded successfully")
    
    return model, feature_cols


def load_model_from_local(model_path):
    """Load a trained model from local filesystem."""
    logger.info(f"📦 Loading model from {model_path}")
    
    # Load model
    model_file = os.path.join(model_path, 'model.joblib')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model = joblib.load(model_file)
    logger.info(f"   Loaded model from {model_file}")
    
    # Load feature columns
    feature_cols = None
    features_file = os.path.join(model_path, 'feature_cols.json')
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            feature_cols = json.load(f)
        logger.info(f"   Loaded {len(feature_cols)} feature columns")
    
    return model, feature_cols


def load_training_data(config):
    """Load training data from BigQuery."""
    from google.cloud import bigquery
    
    project_id = config['project']['id']
    dataset_id = config['bigquery']['dataset']
    table_id = config['bigquery']['tables']['training']
    
    logger.info(f"📊 Loading training data from BigQuery...")
    logger.info(f"   Project: {project_id}")
    logger.info(f"   Table: {dataset_id}.{table_id}")
    
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    query = f"SELECT * FROM `{table_ref}`"
    df = client.query(query).to_dataframe()
    
    logger.info(f"✅ Loaded {len(df)} samples")
    logger.info(f"   Columns: {list(df.columns)}")
    logger.info(f"   Crops: {df['crop'].value_counts().to_dict()}")
    
    return df


def load_config_from_gcs(config_path, bucket_name):
    """Load config.yaml from GCS."""
    from google.cloud import storage
    import yaml
    
    logger.info(f"📋 Loading config from gs://{bucket_name}/{config_path}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(config_path)
    
    local_config_path = "/tmp/config.yaml"
    blob.download_to_filename(local_config_path)
    
    with open(local_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✅ Config loaded")
    return config




def main():
    parser = argparse.ArgumentParser(
        description='Evaluate an existing trained model without retraining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate latest model from GCS
  python local_evaluation.py --model-path models/crop_classifier_latest --output-dir ./evaluation_results

  # Evaluate specific archived model
  python local_evaluation.py --model-path models/crop_classifier_archive/crop_classifier_20251205_2255

  # Evaluate local model
  python local_evaluation.py --model-path ./local_model --output-dir ./results

  # Use custom test data
  python local_evaluation.py --model-path models/crop_classifier_latest --test-data ./test_data.csv
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/crop_classifier_latest',
        help='Path to model (GCS path like "models/crop_classifier_latest" or local path)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_output',
        help='Directory to save evaluation results (default: ./evaluation_output)'
    )
    
    parser.add_argument(
        '--bucket',
        type=str,
        default=None,
        help='GCS bucket name (default: from config.yaml)'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to CSV file with test data (optional, uses BigQuery if not provided)'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Test split ratio if using training data (default: 0.2)'
    )
    
    parser.add_argument(
        '--local',
        action='store_true',
        help='Treat model-path as local filesystem path (not GCS)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yaml (local or GCS path like "config.yaml")'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=5,
        help='Number of evaluation runs to log (for multiple scalar data points, default: 5)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("🔍 MODEL EVALUATION (STANDALONE)")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Load config
        bucket_name = args.bucket
        config = load_config(config_path=args.config, bucket_name=bucket_name)
        if not bucket_name:
            bucket_name = config['storage']['bucket']
        
        # Load model
        if args.local:
            model, feature_cols = load_model_from_local(args.model_path)
        else:
            model, feature_cols = load_model_from_gcs(args.model_path, bucket_name)
        
        logger.info("")
        
        # Load data
        if args.test_data:
            logger.info(f"📊 Loading test data from {args.test_data}")
            df = pd.read_csv(args.test_data)
        else:
            df = load_training_data(config)
        
        logger.info("")
        
        # Engineer features (same as training)
        logger.info("🔧 Engineering features...")
        
        # Get elevation quantiles from config or compute from data
        elevation_quantiles = None
        if 'features' in config and 'elevation_quantiles' in config['features']:
            elevation_quantiles = config['features']['elevation_quantiles']
            logger.info(f"   Using elevation quantiles from config: {elevation_quantiles}")
        
        if not elevation_quantiles or not elevation_quantiles.get('q25'):
            logger.info("   Computing elevation quantiles from data...")
            elevation_quantiles = compute_elevation_quantiles(df)
            logger.info(f"   Elevation quantiles: {elevation_quantiles}")
        
        df_enhanced, feature_cols_engineered = engineer_features_dataframe(df, elevation_quantiles)
        
        # Use feature names that match what the model was trained with
        # Check what the model expects
        model_expected_features = None
        if hasattr(model, 'named_steps'):
            scaler = model.named_steps.get('scaler')
            if scaler and hasattr(scaler, 'feature_names_in_'):
                model_expected_features = list(scaler.feature_names_in_)
                logger.info(f"   Model expects {len(model_expected_features)} features from training")
        
        # Use model's expected features if available, otherwise use feature_cols from JSON, otherwise use engineered
        if model_expected_features:
            # Map old feature names to new ones if needed
            feature_mapping = {
                'elevation_m': 'elevation_m',  # Keep raw elevation if model expects it
                'longitude': 'longitude',
                'latitude': 'latitude'
            }
            
            # Check if model expects old features (elevation_m, lat/lon) or new ones (elevation_binned, sin/cos)
            if 'elevation_m' in model_expected_features:
                # Model was trained with old features - need to use those
                # But we have engineered features, so we need to map back
                logger.info("   Model was trained with old features (elevation_m, lat/lon)")
                logger.info("   Using raw features to match model expectations")
                # Use raw features from dataframe
                feature_cols = [f for f in model_expected_features if f in df_enhanced.columns]
                # Add raw lat/lon/elevation if they exist in the original df
                if 'elevation_m' in df.columns and 'elevation_m' in model_expected_features:
                    df_enhanced['elevation_m'] = df['elevation_m']
                if 'longitude' in df.columns and 'longitude' in model_expected_features:
                    df_enhanced['longitude'] = df['longitude']
                if 'latitude' in df.columns and 'latitude' in model_expected_features:
                    df_enhanced['latitude'] = df['latitude']
            else:
                # Model expects new engineered features
                feature_cols = [f for f in model_expected_features if f in df_enhanced.columns]
        elif feature_cols:
            # Use feature names from saved model
            feature_cols = [f for f in feature_cols if f in df_enhanced.columns]
            logger.info(f"   Using {len(feature_cols)} features from saved model")
        else:
            # Fallback to engineered features
            feature_cols = feature_cols_engineered
            logger.info(f"   Using {len(feature_cols)} engineered features")
        
        logger.info(f"   Final feature count: {len(feature_cols)}")
        logger.info(f"   Features: {feature_cols[:5]}...")
        
        # Verify feature count matches
        if hasattr(model, 'named_steps'):
            rf_model = model.named_steps['classifier']
        else:
            rf_model = model
        
        if hasattr(rf_model, 'n_features_in_'):
            expected_features = rf_model.n_features_in_
            if len(feature_cols) != expected_features:
                logger.error(f"❌ Feature count mismatch: model expects {expected_features}, we have {len(feature_cols)}")
                logger.error(f"   Missing: {set(model_expected_features or []) - set(feature_cols)}")
                logger.error(f"   Extra: {set(feature_cols) - set(model_expected_features or [])}")
            else:
                logger.info(f"   ✅ Feature count matches model: {expected_features}")
        
        logger.info("")
        
        # Split data (if using full dataset)
        if not args.test_data:
            logger.info(f"📊 Splitting data (test split: {args.test_split})...")
            X = df_enhanced[feature_cols]
            y = df_enhanced['crop']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_split, random_state=42, stratify=y
            )
            logger.info(f"   Test samples: {len(X_test)}")
            logger.info(f"   Train samples: {len(X_train)}")
        else:
            # Use all data as test set
            X_test = df_enhanced[feature_cols]
            y_test = df_enhanced['crop']
            logger.info(f"   Using all {len(X_test)} samples as test set")
        
        logger.info("")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create TensorBoard writer
        tensorboard_log_dir = os.path.join(args.output_dir, 'tensorboard_logs')
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        
        # Run comprehensive evaluation (logs everything to TensorBoard)
        logger.info("🔍 Running comprehensive evaluation...")
        logger.info("")
        
        eval_results = run_comprehensive_evaluation(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            writer=writer,
            step=0,
            num_runs=args.num_runs
        )
        
        writer.close()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ EVALUATION COMPLETE")
        logger.info("=" * 70)
        logger.info("")
        logger.info(f"📊 TensorBoard logs saved to: {tensorboard_log_dir}")
        logger.info("")
        logger.info("📊 Summary Metrics:")
        if 'advanced_metrics' in eval_results:
            metrics = eval_results['advanced_metrics']['overall_metrics']
            logger.info(f"   Accuracy: {metrics['accuracy']:.2%}")
            logger.info(f"   Cohen's Kappa: {metrics['cohen_kappa']:.3f}")
            logger.info(f"   Matthews Corr: {metrics['matthews_corrcoef']:.3f}")
            logger.info(f"   Macro F1: {metrics['macro_avg']['f1_score']:.2%}")
            logger.info(f"   Weighted F1: {metrics['weighted_avg']['f1_score']:.2%}")
        
        logger.info("")
        logger.info(f"💡 View results in TensorBoard:")
        logger.info(f"   tensorboard --logdir {tensorboard_log_dir}")
        logger.info(f"   Then open: http://localhost:6006")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

