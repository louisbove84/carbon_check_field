"""
Automated Model Retraining Pipeline
====================================
Cloud Function that retrains the crop classification model when:
1. New training data is collected (triggered by monthly_data_collection)
2. Manual trigger via HTTP request
3. Scheduled monthly retraining

Uses custom Random Forest with feature engineering for better accuracy.
"""

import logging
import pandas as pd
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, Tuple, List
from google.cloud import aiplatform, bigquery, storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "ml-pipeline-477612"
REGION = "us-central1"
BUCKET_NAME = "carboncheck-data"
DATASET_ID = "crop_ml"
TABLE_ID = "training_features"
# ENDPOINT_ID = None  # Creates new endpoint for sklearn models (old AutoML endpoint deleted)

# Model training configuration
MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42,
    'n_jobs': -1
}

# Base features from BigQuery
BASE_FEATURE_COLUMNS = [
    'ndvi_mean',
    'ndvi_std',
    'ndvi_min',
    'ndvi_max',
    'ndvi_p25',
    'ndvi_p50',
    'ndvi_p75',
    'ndvi_early',
    'ndvi_late',
    'elevation_m',
    'longitude',
    'latitude'
]

# Minimum samples required for training
MIN_TRAINING_SAMPLES = 100  # Production threshold
MIN_ACCURACY_THRESHOLD = 0.70  # Production threshold

# ============================================================
# BIGQUERY DATA LOADING
# ============================================================

def load_data_from_bigquery() -> pd.DataFrame:
    """
    Load training data from BigQuery.
    
    Returns:
        DataFrame with training data
    """
    logger.info("📥 Loading data from BigQuery...")
    
    client = bigquery.Client(project=PROJECT_ID)
    
    query = f"""
    SELECT
        field_id,
        crop,
        crop_code,
        ndvi_mean,
        ndvi_std,
        ndvi_min,
        ndvi_max,
        ndvi_p25,
        ndvi_p50,
        ndvi_p75,
        ndvi_early,
        ndvi_late,
        elevation_m,
        longitude,
        latitude
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    WHERE ndvi_mean IS NOT NULL
        AND ndvi_std IS NOT NULL
        AND ndvi_p50 IS NOT NULL
    """
    
    df = client.query(query).to_dataframe()
    logger.info(f"✅ Loaded {len(df)} fields")
    logger.info(f"   Crops: {df['crop'].value_counts().to_dict()}")
    
    return df


def check_training_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify training data quality before retraining.
    
    Args:
        df: Training dataframe
    
    Returns:
        Dictionary with data quality metrics
    """
    logger.info("🔍 Checking training data quality...")
    
    quality_report = {
        'total_samples': len(df),
        'crops': {},
        'is_balanced': True,
        'min_samples': float('inf'),
        'max_samples': 0
    }
    
    # Check sample counts per crop
    crop_counts = df['crop'].value_counts()
    
    for crop, count in crop_counts.items():
        quality_report['crops'][crop] = {
            'sample_count': int(count)
        }
        quality_report['min_samples'] = min(quality_report['min_samples'], count)
        quality_report['max_samples'] = max(quality_report['max_samples'], count)
    
    # Check if balanced (max samples < 2x min samples)
    if quality_report['max_samples'] > 2 * quality_report['min_samples']:
        quality_report['is_balanced'] = False
        logger.warning(f"⚠️  Imbalanced dataset: {quality_report['min_samples']} - {quality_report['max_samples']} samples")
    
    # Check for NULL values in key columns
    null_counts = df[BASE_FEATURE_COLUMNS].isnull().sum()
    quality_report['null_counts'] = null_counts.to_dict()
    
    # Log quality report
    logger.info(f"📊 Total samples: {quality_report['total_samples']}")
    for crop, stats in quality_report['crops'].items():
        logger.info(f"   • {crop}: {stats['sample_count']} samples")
    
    if null_counts.sum() > 0:
        logger.warning(f"⚠️  Found NULL values: {null_counts[null_counts > 0].to_dict()}")
    else:
        logger.info("✅ No NULL values found")
    
    return quality_report


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create additional features from raw data.
    
    Args:
        df: DataFrame with base features
    
    Returns:
        Tuple of (enhanced dataframe, list of all feature columns)
    """
    logger.info("🔧 Engineering features...")
    
    # Create derived features
    df['ndvi_range'] = df['ndvi_max'] - df['ndvi_min']
    df['ndvi_iqr'] = df['ndvi_p75'] - df['ndvi_p25']
    df['ndvi_change'] = df['ndvi_late'] - df['ndvi_early']
    df['ndvi_early_ratio'] = df['ndvi_early'] / (df['ndvi_mean'] + 0.001)
    df['ndvi_late_ratio'] = df['ndvi_late'] / (df['ndvi_mean'] + 0.001)
    
    # Full feature list (base + engineered)
    all_features = BASE_FEATURE_COLUMNS + [
        'ndvi_range', 'ndvi_iqr', 'ndvi_change',
        'ndvi_early_ratio', 'ndvi_late_ratio'
    ]
    
    logger.info(f"✅ Created {len(all_features)} total features")
    
    return df, all_features


# ============================================================
# MODEL TRAINING
# ============================================================

def train_local_model(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[
    Pipeline, Dict[str, Any]
]:
    """
    Train Random Forest model locally using a Pipeline (scaler + model).
    Pipeline format is required for Vertex AI sklearn container.
    
    Args:
        df: Training dataframe with features
        feature_cols: List of feature column names
    
    Returns:
        Tuple of (trained pipeline, metrics dict)
    """
    logger.info("🤖 Training Random Forest model...")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Config: {MODEL_CONFIG}")
    
    # Split features and target
    X = df[feature_cols]
    y = df['crop']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"   Train samples: {len(X_train)}")
    logger.info(f"   Test samples: {len(X_test)}")
    
    # Create pipeline: scaler -> model
    # This ensures the scaler is included in the saved model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(**MODEL_CONFIG))
    ])
    
    # Train pipeline (scaler.fit_transform + model.fit)
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    logger.info(f"✅ Training accuracy: {train_score:.2%}")
    logger.info(f"✅ Test accuracy: {test_score:.2%}")
    
    # Detailed metrics
    y_pred = pipeline.predict(X_test)
    
    logger.info("\n📊 Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(classification_report(y_test, y_pred))
    
    logger.info("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(str(cm))
    
    # Feature importance (extract from the classifier step)
    classifier = pipeline.named_steps['classifier']
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n🔝 Top 5 Features:")
    logger.info(str(feature_importance.head()))
    
    # Compile metrics
    metrics = {
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict('records')[:10],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    return pipeline, metrics


# ============================================================
# MODEL SAVING & DEPLOYMENT
# ============================================================

def save_model_to_gcs(
    pipeline: Pipeline,
    feature_cols: List[str],
    bucket_name: str
) -> str:
    """
    Save trained model pipeline to GCS.
    Pipeline includes scaler + model, so only one file is needed.
    
    Args:
        pipeline: Trained sklearn Pipeline (scaler + model)
        feature_cols: List of feature names (for reference)
        bucket_name: GCS bucket name
    
    Returns:
        GCS path to model artifacts
    """
    logger.info(f"💾 Saving model to gs://{bucket_name}/models/...")
    
    # Create temporary directory for artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save pipeline (includes scaler + model)
        model_path = os.path.join(tmpdir, 'model.joblib')
        features_path = os.path.join(tmpdir, 'feature_cols.json')
        
        # Save pipeline with pickle protocol 4 (compatible with Python 3.7+)
        joblib.dump(pipeline, model_path, protocol=4)
        
        # Save feature columns for reference (not required by container, but useful)
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f)
        
        # Upload to GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # Save to archive (versioned, preserved for history)
        archive_prefix = f'models/crop_classifier_archive/crop_classifier_{timestamp}'
        bucket.blob(f'{archive_prefix}/model.joblib').upload_from_filename(model_path)
        bucket.blob(f'{archive_prefix}/feature_cols.json').upload_from_filename(features_path)
        
        # Save to latest (overwrites previous, used by Vertex AI)
        # Vertex AI sklearn container expects 'model.joblib' in the root
        latest_prefix = 'models/crop_classifier_latest'
        bucket.blob(f'{latest_prefix}/model.joblib').upload_from_filename(model_path)
        bucket.blob(f'{latest_prefix}/feature_cols.json').upload_from_filename(features_path)
    
    logger.info(f"✅ Model archived to gs://{bucket_name}/{archive_prefix}")
    logger.info(f"✅ Latest model updated at gs://{bucket_name}/{latest_prefix}")
    logger.info(f"   Pipeline includes: scaler + RandomForest classifier")
    
    # Return latest path (used by Vertex AI)
    return f'gs://{bucket_name}/{latest_prefix}'


def deploy_model_to_vertex_ai(
    model_gcs_path: str,
    endpoint_id: str = None
) -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
    """
    Deploy model to Vertex AI endpoint.
    Uses consistent names: 'crop-classifier-latest' and 'crop-endpoint'.
    
    Args:
        model_gcs_path: GCS path to model artifacts (should be latest folder)
        endpoint_id: Existing endpoint ID (finds by name if None)
    
    Returns:
        Vertex AI Endpoint
    """
    logger.info("🚀 Deploying to Vertex AI endpoint...")
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Delete old "latest" model if it exists (can't update, must recreate)
    MODEL_NAME = 'crop-classifier-latest'
    ENDPOINT_NAME = 'crop-endpoint'
    
    try:
        existing_models = aiplatform.Model.list(
            filter=f'display_name="{MODEL_NAME}"',
            order_by='create_time desc'
        )
        if existing_models:
            old_model = existing_models[0]
            logger.info(f"   Deleting old model: {old_model.display_name} ({old_model.name})")
            old_model.delete()
    except Exception as e:
        logger.info(f"   No existing model to delete: {e}")
    
    # Upload new model with consistent name (always points to latest folder)
    model = aiplatform.Model.upload(
        display_name=MODEL_NAME,
        artifact_uri=f'gs://{BUCKET_NAME}/models/crop_classifier_latest',
        serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest'
    )
    
    logger.info(f"✅ Model registered: {model.display_name} (ID: {model.name.split('/')[-1]})")
    
    # Find or create endpoint with consistent name
    if endpoint_id:
        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{endpoint_id}"
        )
        logger.info(f"   Using provided endpoint: {endpoint_id}")
    else:
        # Try to find existing endpoint by name
        try:
            existing_endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{ENDPOINT_NAME}"'
            )
            if existing_endpoints:
                endpoint = existing_endpoints[0]
                logger.info(f"   Using existing endpoint: {endpoint.display_name} (ID: {endpoint.name.split('/')[-1]})")
            else:
                # Create new endpoint with consistent name
                endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)
                logger.info(f"✅ Endpoint created: {endpoint.display_name} (ID: {endpoint.name.split('/')[-1]})")
        except Exception as e:
            # Fallback: create new endpoint
            endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)
            logger.info(f"✅ Endpoint created: {endpoint.display_name} (ID: {endpoint.name.split('/')[-1]})")
    
    # Undeploy old models from endpoint
    for deployed_model in endpoint.list_models():
        logger.info(f"   Undeploying old model: {deployed_model.id}")
        endpoint.undeploy(deployed_model_id=deployed_model.id)
    
    # Deploy new model
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name='crop-model-latest',
        machine_type='n1-standard-2',
        min_replica_count=1,
        max_replica_count=3,
        traffic_percentage=100
    )
    
    logger.info("✅ Model deployed successfully!")
    
    return model, endpoint


# ============================================================
# MAIN RETRAINING FUNCTION
# ============================================================

def retrain_model(request=None):
    """
    Main Cloud Function entry point.
    Retrains and deploys the crop classification model.
    
    Args:
        request: Flask request object (for Cloud Function compatibility)
    
    Returns:
        JSON response with training summary
    """
    logger.info("=" * 60)
    logger.info("🌾 AUTOMATED MODEL RETRAINING PIPELINE")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load data from BigQuery
        df = load_data_from_bigquery()
        
        # Step 2: Check data quality
        quality_report = check_training_data_quality(df)
        
        if quality_report['total_samples'] < MIN_TRAINING_SAMPLES:
            raise Exception(
                f"Insufficient training data: {quality_report['total_samples']} samples "
                f"(need {MIN_TRAINING_SAMPLES}+)"
            )
        
        # Step 3: Engineer features
        df_enhanced, feature_cols = engineer_features(df)
        
        # Step 4: Train model (returns Pipeline with scaler + model)
        pipeline, metrics = train_local_model(df_enhanced, feature_cols)
        
        # Step 5: Check accuracy threshold
        test_accuracy = metrics['test_accuracy']
        if test_accuracy < MIN_ACCURACY_THRESHOLD:
            logger.warning(
                f"⚠️  Model accuracy ({test_accuracy:.2%}) below threshold "
                f"({MIN_ACCURACY_THRESHOLD:.0%}). Proceeding with caution..."
            )
        
        # Step 6: Save model to GCS (pipeline includes scaler)
        model_gcs_path = save_model_to_gcs(pipeline, feature_cols, BUCKET_NAME)
        
        # Step 7: Deploy to Vertex AI endpoint (reuses existing 'crop-endpoint' if available)
        model, endpoint = deploy_model_to_vertex_ai(model_gcs_path, endpoint_id=None)
        
        # Extract IDs from resource names
        endpoint_id = endpoint.name.split('/')[-1]
        model_id = model.name.split('/')[-1]
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        logger.info("=" * 60)
        logger.info("✅ RETRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"⏱️  Duration: {duration:.1f} minutes")
        logger.info(f"📊 Training samples: {quality_report['total_samples']}")
        logger.info(f"🎯 Test accuracy: {test_accuracy:.2%}")
        logger.info(f"🤖 Model: {model.display_name} (ID: {model_id})")
        logger.info(f"🚀 Endpoint: {endpoint.display_name} (ID: {endpoint_id})")
        
        return {
            'status': 'success',
            'model_gcs_path': model_gcs_path,
            'model_name': 'crop-classifier-latest',
            'model_id': model_id,
            'endpoint_id': endpoint_id,
            'endpoint_name': endpoint.name,
            'endpoint_display_name': endpoint.display_name,
            'training_samples': quality_report['total_samples'],
            'crops': quality_report['crops'],
            'metrics': {
                'train_accuracy': metrics['train_accuracy'],
                'test_accuracy': metrics['test_accuracy'],
                'n_train_samples': metrics['n_train_samples'],
                'n_test_samples': metrics['n_test_samples']
            },
            'feature_count': len(feature_cols),
            'duration_minutes': round(duration, 1),
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds() / 60
        logger.error(f"❌ Retraining failed after {duration:.1f} minutes: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'duration_minutes': round(duration, 1),
            'timestamp': datetime.now().isoformat()
        }


# ============================================================
# LOCAL TESTING & UTILITIES
# ============================================================

def test_prediction(endpoint: aiplatform.Endpoint, test_features: List[float]) -> Dict:
    """
    Test prediction on deployed endpoint.
    
    Args:
        endpoint: Vertex AI endpoint
        test_features: List of feature values
    
    Returns:
        Prediction result
    """
    logger.info("🧪 Testing prediction...")
    
    prediction = endpoint.predict(instances=[test_features])
    
    logger.info(f"✅ Prediction: {prediction.predictions[0]}")
    
    return {
        'prediction': prediction.predictions[0],
        'confidence': prediction.predictions[0] if hasattr(prediction, 'predictions') else None
    }


if __name__ == '__main__':
    # For local testing
    result = retrain_model()
    print(json.dumps(result, indent=2, default=str))

