"""
Vertex AI Skew Audit Script
============================
Main entrypoint for running skew audits as a Vertex AI Custom Job.
This enables native TensorBoard integration via AIP_TENSORBOARD_LOG_DIR.

When run as a Vertex AI Custom Job with a linked TensorBoard instance,
Vertex AI automatically sets AIP_TENSORBOARD_LOG_DIR and syncs logs to TensorBoard.
"""

import os
import sys
import re
import json
import logging
from datetime import datetime
from importlib import metadata
from google.cloud import storage

# Import local module (downloaded at runtime by entrypoint.sh)
import skew_detector

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
        'scipy',
        'earthengine-api'
    ]
    for package in packages:
        try:
            version = metadata.version(package)
            logger.info(f"   {package}: {version}")
        except Exception:
            logger.info(f"   {package}: NOT INSTALLED")
    logger.info("")


def upload_tensorboard_logs_to_gcs(local_tb_dir: str, gcs_tb_dir: str) -> int:
    """
    Upload local TensorBoard logs to GCS for Vertex AI sync.
    
    Args:
        local_tb_dir: Local directory containing TensorBoard logs
        gcs_tb_dir: GCS path (gs://bucket/path/) to upload to
    
    Returns:
        Number of files uploaded
    """
    if not gcs_tb_dir.startswith('gs://'):
        logger.warning(f"Invalid GCS path: {gcs_tb_dir}")
        return 0
    
    # Parse GCS path
    gcs_path = gcs_tb_dir.replace('gs://', '').rstrip('/')
    parts = gcs_path.split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    uploaded_count = 0
    for root, dirs, files in os.walk(local_tb_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Get relative path from local_tb_dir
            rel_path = os.path.relpath(local_path, local_tb_dir)
            blob_path = f"{prefix}/{rel_path}" if prefix else rel_path
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            uploaded_count += 1
            logger.info(f"   Uploaded: {rel_path}")
    
    return uploaded_count


def run_vertex_ai_skew_audit():
    """
    Run skew audit with Vertex AI TensorBoard integration.
    
    This function:
    1. Gets AIP_TENSORBOARD_LOG_DIR from environment (set by Vertex AI)
    2. Runs the skew audit with TensorBoard logging to local directory
    3. Uploads logs to GCS for Vertex AI to sync to TensorBoard
    """
    logger.info("=" * 70)
    logger.info("VERTEX AI SKEW AUDIT")
    logger.info("=" * 70)
    
    # Log environment for debugging
    logger.info("Environment Variables:")
    logger.info(f"   AIP_TENSORBOARD_LOG_DIR: {os.environ.get('AIP_TENSORBOARD_LOG_DIR', 'NOT SET')}")
    logger.info(f"   AIP_TENSORBOARD_EXPERIMENT_NAME: {os.environ.get('AIP_TENSORBOARD_EXPERIMENT_NAME', 'NOT SET')}")
    logger.info(f"   AIP_MODEL_DIR: {os.environ.get('AIP_MODEL_DIR', 'NOT SET')}")
    logger.info(f"   AIP_TRAINING_DATA_URI: {os.environ.get('AIP_TRAINING_DATA_URI', 'NOT SET')}")
    logger.info("")
    
    log_package_versions()
    
    # Get TensorBoard configuration from Vertex AI
    managed_tb_gcs_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
    experiment_name = os.environ.get('AIP_TENSORBOARD_EXPERIMENT_NAME', 'skew-audit')
    
    # Create local TensorBoard directory structure
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = re.sub(r'[^a-z0-9-]', '-', experiment_name.lower())
    run_id = re.sub(r'[^a-z0-9-]', '-', f'run-{timestamp}'.lower())
    
    local_tb_base = '/tmp/tensorboard_logs'
    local_tb_dir = f'{local_tb_base}/{experiment_id}/{run_id}'
    os.makedirs(local_tb_dir, exist_ok=True)
    
    logger.info("TensorBoard Configuration:")
    if managed_tb_gcs_dir:
        logger.info(f"   AIP_TENSORBOARD_LOG_DIR (GCS): {managed_tb_gcs_dir}")
    else:
        logger.warning("   AIP_TENSORBOARD_LOG_DIR not set - images may not sync to TensorBoard")
    logger.info(f"   Local TensorBoard dir: {local_tb_dir}")
    logger.info(f"   Experiment ID: {experiment_id}")
    logger.info(f"   Run ID: {run_id}")
    logger.info("")
    
    # Load config
    config = skew_detector.load_config()
    
    # Run the skew audit pipeline steps manually so we can control TensorBoard logging
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("DATA SKEW AUDIT (Label-Free)")
    logger.info("=" * 70)
    
    try:
        # Step 1: Collect recent EE samples
        logger.info("STEP 1: Collect recent Earth Engine samples (label-free)")
        logger.info("-" * 70)
        recent_df = skew_detector.collect_recent_ee_samples(config, sample_fraction=0.1)
        logger.info("")
        
        # Step 2: Load training data
        logger.info("STEP 2: Load training data from BigQuery (features only)")
        logger.info("-" * 70)
        training_df = skew_detector.load_training_data(config)
        logger.info("")
        
        # Step 3: Get endpoint predictions
        logger.info("STEP 3: Get endpoint predictions")
        logger.info("-" * 70)
        training_pred_metrics = {}
        recent_pred_metrics = {}
        try:
            training_df, training_pred_metrics = skew_detector.get_endpoint_predictions_with_confidence(config, training_df)
            if len(recent_df) > 0:
                recent_df, recent_pred_metrics = skew_detector.get_endpoint_predictions_with_confidence(config, recent_df)
        except Exception as e:
            logger.warning(f"   Could not get predictions: {e}")
        logger.info("")
        
        # Step 4: Compute skew metrics
        logger.info("STEP 4: Compute label-free skew metrics")
        logger.info("-" * 70)
        metrics = skew_detector.compute_label_free_skew_metrics(
            training_df, recent_df,
            training_pred_metrics, recent_pred_metrics,
            config
        )
        logger.info("")
        
        # Step 5: Log to TensorBoard using Vertex AI's directory
        logger.info("STEP 5: Log to TensorBoard (Vertex AI Native)")
        logger.info("-" * 70)
        
        # Use the refactored function from skew_detector with vertex_ai_mode=True
        # This skips the manual API upload and lets Vertex AI handle GCS sync
        tb_path = skew_detector.log_skew_to_tensorboard(
            metrics, training_df, recent_df,
            tensorboard_resource=None,  # Not needed in Vertex AI mode
            config=config,
            local_tb_dir=local_tb_dir,
            vertex_ai_mode=True
        )
        metrics['tensorboard_path'] = tb_path
        logger.info("")
        
        # Step 6: Upload TensorBoard logs to GCS
        if managed_tb_gcs_dir:
            logger.info("STEP 6: Upload TensorBoard logs to GCS")
            logger.info("-" * 70)
            
            # Clean the GCS path
            managed_tb_gcs_dir_clean = managed_tb_gcs_dir.rstrip('/')
            gcs_upload_path = f"{managed_tb_gcs_dir_clean}/{experiment_id}/{run_id}"
            
            logger.info(f"   Uploading to: {gcs_upload_path}")
            uploaded_count = upload_tensorboard_logs_to_gcs(local_tb_dir, gcs_upload_path)
            logger.info(f"   Uploaded {uploaded_count} TensorBoard files to GCS")
            logger.info("   Vertex AI will automatically sync these logs to TensorBoard")
            logger.info("")
        else:
            logger.info("STEP 6: Skip GCS upload (AIP_TENSORBOARD_LOG_DIR not set)")
            logger.info("-" * 70)
            logger.warning("   TensorBoard logs saved locally only")
            logger.info("")
        
        # Step 7: Store in BigQuery
        logger.info("STEP 7: Store results in BigQuery")
        logger.info("-" * 70)
        skew_detector.store_results_in_bigquery(metrics, config)
        logger.info("")
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds() / 60
        summary = metrics.get('summary', {})
        drift_score_info = metrics.get('drift_score_info', {})
        
        logger.info("=" * 70)
        logger.info("SKEW AUDIT COMPLETE (Vertex AI)")
        logger.info("=" * 70)
        logger.info("")
        logger.info("+" + "=" * 44 + "+")
        logger.info(f"|  DRIFT SCORE: {drift_score_info.get('drift_score', 0):>5.1f}/100                    |")
        logger.info(f"|  RETRAINING: {drift_score_info.get('retraining_needed', 'N/A'):<21}       |")
        logger.info("+" + "=" * 44 + "+")
        logger.info("")
        logger.info(f"Duration: {duration:.1f} minutes")
        logger.info(f"Training samples: {metrics['training_samples']}")
        logger.info(f"Recent samples: {metrics['recent_samples']}")
        
        if managed_tb_gcs_dir:
            logger.info("")
            logger.info("TensorBoard Run Summary:")
            logger.info(f"   Experiment: {experiment_id}")
            logger.info(f"   Run: {run_id}")
            logger.info(f"   GCS logs: {managed_tb_gcs_dir}/{experiment_id}/{run_id}/")
        
        return {
            'status': 'success',
            'duration_minutes': round(duration, 2),
            'drift_score': drift_score_info.get('drift_score', 0),
            'retraining_needed': drift_score_info.get('retraining_needed', 'UNKNOWN'),
            'tensorboard': {
                'experiment_id': experiment_id,
                'run_id': run_id,
                'gcs_path': f"{managed_tb_gcs_dir}/{experiment_id}/{run_id}/" if managed_tb_gcs_dir else None
            },
            'metrics': metrics
        }
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds() / 60
        logger.error(f"Skew audit failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'duration_minutes': round(duration, 2)
        }


if __name__ == '__main__':
    result = run_vertex_ai_skew_audit()
    print(json.dumps(result, indent=2, default=str))
