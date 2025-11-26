"""
Model Retraining
================
Trains a new RandomForest model with all available data.
"""

import yaml
import logging
import pandas as pd
import joblib
import tempfile
import os
from datetime import datetime
from google.cloud import bigquery, storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    """Load configuration from Cloud Storage or local file."""
    try:
        client = storage.Client()
        bucket = client.bucket('carboncheck-data')
        blob = bucket.blob('config/config.yaml')
        yaml_content = blob.download_as_text()
        return yaml.safe_load(yaml_content)
    except:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)

config = load_config()


def retrain_model():
    """
    Retrain RandomForest model with all available data.
    
    Returns:
        dict with training results
    """
    logger.info("📥 Loading configuration...")
    project_id = config['project']['id']
    dataset_id = config['bigquery']['dataset']
    training_table = config['bigquery']['tables']['training']
    holdout_table = config['bigquery']['tables']['holdout']
    bucket_name = config['storage']['bucket']
    
    hyperparams = config['model']['hyperparameters']
    logger.info(f"   Hyperparameters: n_estimators={hyperparams['n_estimators']}, max_depth={hyperparams['max_depth']}")
    
    try:
        # Load training data (excluding holdout)
        logger.info("📥 Loading training data from BigQuery...")
        df = load_training_data(project_id, dataset_id, training_table, holdout_table)
        logger.info(f"   ✅ Loaded {len(df)} samples")
        logger.info(f"   Crops: {df['crop'].value_counts().to_dict()}")
        
        # Engineer features
        logger.info("🔧 Engineering features...")
        df_enhanced, feature_cols = engineer_features(df)
        logger.info(f"   ✅ Created {len(feature_cols)} features")
        
        # Train model
        logger.info("🤖 Training RandomForest model...")
        pipeline, metrics = train_model(df_enhanced, feature_cols, hyperparams)
        logger.info(f"   ✅ Training accuracy: {metrics['train_accuracy']:.2%}")
        logger.info(f"   ✅ Test accuracy: {metrics['test_accuracy']:.2%}")
        
        # Save model to GCS
        logger.info("💾 Saving model to Cloud Storage...")
        model_path = save_model_to_gcs(pipeline, feature_cols, bucket_name)
        logger.info(f"   ✅ Model saved to {model_path}")
        
        return {
            'status': 'success',
            'training_samples': len(df),
            'metrics': metrics,
            'model_path': model_path,
            'feature_count': len(feature_cols),
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Model retraining failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def load_training_data(project_id, dataset_id, training_table, holdout_table):
    """Load training data excluding holdout samples."""
    client = bigquery.Client(project=project_id)
    
    query = f"""
    SELECT t.*
    FROM `{project_id}.{dataset_id}.{training_table}` t
    LEFT JOIN `{project_id}.{dataset_id}.{holdout_table}` h
        ON t.sample_id = h.sample_id
    WHERE h.sample_id IS NULL
        AND t.ndvi_mean IS NOT NULL
    """
    
    df = client.query(query).to_dataframe()
    return df


def engineer_features(df):
    """Create derived features from raw data."""
    df = df.copy()
    
    # Derived NDVI features
    df['ndvi_range'] = df['ndvi_max'] - df['ndvi_min']
    df['ndvi_iqr'] = df['ndvi_p75'] - df['ndvi_p25']
    df['ndvi_change'] = df['ndvi_late'] - df['ndvi_early']
    df['ndvi_early_ratio'] = df['ndvi_early'] / (df['ndvi_mean'] + 0.001)
    df['ndvi_late_ratio'] = df['ndvi_late'] / (df['ndvi_mean'] + 0.001)
    
    # Base features from config
    base_features = config['features']['base_columns']
    
    # All features (base + derived)
    all_features = base_features + [
        'ndvi_range', 'ndvi_iqr', 'ndvi_change',
        'ndvi_early_ratio', 'ndvi_late_ratio'
    ]
    
    return df, all_features


def train_model(df, feature_cols, hyperparams):
    """Train RandomForest pipeline."""
    X = df[feature_cols]
    y = df['crop']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=hyperparams['random_state'], stratify=y
    )
    
    logger.info(f"   Train samples: {len(X_train)}")
    logger.info(f"   Test samples: {len(X_test)}")
    
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
    
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info("\n📊 Per-crop performance:")
    for crop in sorted(y_test.unique()):
        if crop in report:
            logger.info(f"   {crop}: F1={report[crop]['f1-score']:.2%}")
    
    metrics = {
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'classification_report': report
    }
    
    return pipeline, metrics


def save_model_to_gcs(pipeline, feature_cols, bucket_name):
    """Save model pipeline to Cloud Storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save pipeline
        model_path = os.path.join(tmpdir, 'model.joblib')
        features_path = os.path.join(tmpdir, 'feature_cols.json')
        
        joblib.dump(pipeline, model_path, protocol=4)
        
        import json
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f)
        
        # Upload to GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # Archive version
        archive_prefix = f'models/crop_classifier_archive/crop_classifier_{timestamp}'
        bucket.blob(f'{archive_prefix}/model.joblib').upload_from_filename(model_path)
        bucket.blob(f'{archive_prefix}/feature_cols.json').upload_from_filename(features_path)
        
        # Latest version (used by evaluate.py)
        latest_prefix = 'models/crop_classifier_latest'
        bucket.blob(f'{latest_prefix}/model.joblib').upload_from_filename(model_path)
        bucket.blob(f'{latest_prefix}/feature_cols.json').upload_from_filename(features_path)
    
    return f'gs://{bucket_name}/{latest_prefix}'


if __name__ == '__main__':
    # For testing
    result = retrain_model()
    import json
    print(json.dumps(result, indent=2, default=str))

