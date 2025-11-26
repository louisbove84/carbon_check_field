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
from datetime import datetime
from google.cloud import bigquery, storage, aiplatform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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


def log_to_vertex_experiments(config, metrics, y_test, y_pred, output_dir):
    """Log metrics and artifacts to Vertex AI Experiments."""
    try:
        project_id = config['project']['id']
        region = config['project']['region']
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        # Create or get experiment
        experiment_name = 'crop-classifier-training'
        
        try:
            experiment = aiplatform.Experiment.create(
                experiment_name=experiment_name,
                description='Crop classification model training experiments'
            )
            logger.info(f"✅ Created new experiment: {experiment_name}")
        except:
            experiment = aiplatform.Experiment(experiment_name=experiment_name)
            logger.info(f"✅ Using existing experiment: {experiment_name}")
        
        # Start a new run with timestamp
        run_name = f"training-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        aiplatform.start_run(run_name)
        
        logger.info(f"📊 Logging to Vertex AI Experiments: {run_name}")
        
        # Log parameters
        aiplatform.log_params({
            'n_estimators': config['model']['hyperparameters']['n_estimators'],
            'max_depth': config['model']['hyperparameters']['max_depth'],
            'min_samples_split': config['model']['hyperparameters']['min_samples_split'],
            'n_train_samples': metrics['n_train_samples'],
            'n_test_samples': metrics['n_test_samples']
        })
        
        # Log metrics
        aiplatform.log_metrics({
            'train_accuracy': metrics['train_accuracy'],
            'test_accuracy': metrics['test_accuracy'],
        })
        
        # Log per-crop F1 scores
        report = metrics['classification_report']
        for crop in sorted(set(y_test)):
            if crop in report:
                aiplatform.log_metrics({
                    f'{crop}_f1': report[crop]['f1-score'],
                    f'{crop}_precision': report[crop]['precision'],
                    f'{crop}_recall': report[crop]['recall']
                })
        
        # Generate and log confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        generate_confusion_matrix(y_test, y_pred, sorted(set(y_test)), cm_path)
        
        # Log confusion matrix as artifact
        aiplatform.log_artifact(cm_path)
        
        # Log classification report as artifact
        report_path = os.path.join(output_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        aiplatform.log_artifact(report_path)
        
        # End the run
        aiplatform.end_run()
        
        logger.info("✅ Logged to Vertex AI Experiments")
        logger.info(f"   View: https://console.cloud.google.com/vertex-ai/experiments/experiments/{experiment_name}?project={project_id}")
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to log to Vertex AI Experiments: {e}")
        logger.warning("   Training will continue, but metrics won't be in Experiments")


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
        
        # Log to Vertex AI Experiments
        log_to_vertex_experiments(config, metrics, y_test, y_pred, output_dir)
        
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

