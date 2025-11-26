"""
Model Evaluation & Deployment
==============================
Evaluates new model against quality gates and deploys if it passes.
"""

import yaml
import logging
import joblib
import pandas as pd
from datetime import datetime
from google.cloud import aiplatform, bigquery, storage
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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


def evaluate_and_deploy():
    """
    Evaluate challenger model and deploy if quality gates pass.
    
    Returns:
        dict with evaluation results and deployment decision
    """
    logger.info("📥 Loading configuration...")
    project_id = config['project']['id']
    region = config['project']['region']
    bucket_name = config['storage']['bucket']
    dataset_id = config['bigquery']['dataset']
    holdout_table = config['bigquery']['tables']['holdout']
    
    gates = config['quality_gates']
    logger.info(f"   Quality Gates:")
    logger.info(f"   - Min Accuracy: {gates['absolute_min_accuracy']:.0%}")
    logger.info(f"   - Min Crop F1: {gates['min_per_crop_f1']:.0%}")
    logger.info(f"   - Improvement: {gates['improvement_margin']:.0%}")
    
    try:
        # Load holdout test set
        logger.info("📊 Loading holdout test set...")
        df_holdout = load_holdout_set(project_id, dataset_id, holdout_table)
        logger.info(f"   ✅ Loaded {len(df_holdout)} holdout samples")
        
        # Prepare test data
        df_holdout, feature_cols = engineer_features(df_holdout)
        X_test = df_holdout[feature_cols]
        y_test = df_holdout['crop']
        
        # Load and evaluate challenger
        logger.info("🆕 Evaluating challenger model...")
        challenger_path = f'gs://{bucket_name}/models/crop_classifier_latest/model.joblib'
        challenger_model = load_model_from_gcs(challenger_path)
        challenger_metrics = evaluate_model(challenger_model, X_test, y_test)
        logger.info(f"   Challenger accuracy: {challenger_metrics['accuracy']:.2%}")
        
        # Check quality gates
        logger.info("⚖️  Checking quality gates...")
        decision = check_quality_gates(challenger_metrics, gates)
        
        # Deploy if passed
        if decision['should_deploy']:
            logger.info("🚀 Quality gates passed - deploying challenger...")
            deploy_to_vertex_ai(project_id, region, bucket_name)
            logger.info("   ✅ Deployment complete")
        else:
            logger.warning("⛔ Quality gates failed - keeping current model")
            for reason in decision['reasons']:
                logger.warning(f"   {reason}")
        
        return {
            'status': 'success',
            'deployed': decision['should_deploy'],
            'challenger_accuracy': challenger_metrics['accuracy'],
            'per_crop_f1': challenger_metrics['per_crop_f1'],
            'reasons': decision['reasons'],
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'deployed': False,
            'timestamp': datetime.now().isoformat()
        }


def load_holdout_set(project_id, dataset_id, holdout_table):
    """Load holdout test set from BigQuery."""
    client = bigquery.Client(project=project_id)
    
    query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{holdout_table}`
    WHERE ndvi_mean IS NOT NULL
    """
    
    df = client.query(query).to_dataframe()
    return df


def engineer_features(df):
    """Create derived features (must match training)."""
    df = df.copy()
    df['ndvi_range'] = df['ndvi_max'] - df['ndvi_min']
    df['ndvi_iqr'] = df['ndvi_p75'] - df['ndvi_p25']
    df['ndvi_change'] = df['ndvi_late'] - df['ndvi_early']
    df['ndvi_early_ratio'] = df['ndvi_early'] / (df['ndvi_mean'] + 0.001)
    df['ndvi_late_ratio'] = df['ndvi_late'] / (df['ndvi_mean'] + 0.001)
    
    base_features = config['features']['base_columns']
    all_features = base_features + [
        'ndvi_range', 'ndvi_iqr', 'ndvi_change',
        'ndvi_early_ratio', 'ndvi_late_ratio'
    ]
    
    return df, all_features


def load_model_from_gcs(model_path):
    """Load model from Cloud Storage."""
    parts = model_path.replace('gs://', '').split('/')
    bucket_name = parts[0]
    blob_path = '/'.join(parts[1:])
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    model_bytes = blob.download_as_bytes()
    model = joblib.loads(model_bytes)
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-crop F1 scores
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=sorted(y_test.unique())
    )
    
    per_crop_f1 = {}
    for i, crop in enumerate(sorted(y_test.unique())):
        per_crop_f1[crop] = float(f1[i])
        logger.info(f"   {crop}: F1={f1[i]:.2%}")
    
    return {
        'accuracy': float(accuracy),
        'per_crop_f1': per_crop_f1
    }


def check_quality_gates(metrics, gates):
    """Check if model passes quality gates."""
    reasons = []
    gates_passed = []
    gates_failed = []
    
    accuracy = metrics['accuracy']
    per_crop_f1 = metrics['per_crop_f1']
    
    # Gate 1: Minimum accuracy
    if accuracy >= gates['absolute_min_accuracy']:
        gates_passed.append(f"✅ Accuracy {accuracy:.2%} >= {gates['absolute_min_accuracy']:.0%}")
    else:
        gates_failed.append(f"❌ Accuracy {accuracy:.2%} < {gates['absolute_min_accuracy']:.0%}")
    
    # Gate 2: Per-crop F1 scores
    for crop, f1 in per_crop_f1.items():
        if f1 >= gates['min_per_crop_f1']:
            gates_passed.append(f"✅ {crop} F1={f1:.2%} >= {gates['min_per_crop_f1']:.0%}")
        else:
            gates_failed.append(f"❌ {crop} F1={f1:.2%} < {gates['min_per_crop_f1']:.0%}")
    
    # Decision
    should_deploy = len(gates_failed) == 0
    
    if should_deploy:
        reasons.append("🎉 All quality gates passed")
    else:
        reasons.append(f"⛔ {len(gates_failed)} gate(s) failed")
    
    reasons.extend(gates_passed)
    reasons.extend(gates_failed)
    
    return {
        'should_deploy': should_deploy,
        'reasons': reasons
    }


def deploy_to_vertex_ai(project_id, region, bucket_name):
    """Deploy model to Vertex AI endpoint."""
    aiplatform.init(project=project_id, location=region)
    
    model_name = config['model']['name']
    endpoint_name = config['model']['endpoint_name']
    
    # Delete old model if exists
    try:
        existing_models = aiplatform.Model.list(filter=f'display_name="{model_name}"')
        if existing_models:
            existing_models[0].delete()
            logger.info(f"   Deleted old model: {model_name}")
    except:
        pass
    
    # Upload new model
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=f'gs://{bucket_name}/models/crop_classifier_latest',
        serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest'
    )
    logger.info(f"   Model registered: {model.display_name}")
    
    # Find or create endpoint
    try:
        existing_endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
        if existing_endpoints:
            endpoint = existing_endpoints[0]
        else:
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        logger.info(f"   Using endpoint: {endpoint.display_name}")
    except:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    
    # Undeploy old models
    for deployed_model in endpoint.list_models():
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


if __name__ == '__main__':
    # For testing
    result = evaluate_and_deploy()
    import json
    print(json.dumps(result, indent=2, default=str))

