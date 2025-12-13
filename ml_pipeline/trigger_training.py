#!/usr/bin/env python3
"""
Trigger a Vertex AI training job with TensorBoard integration.
Code is uploaded dynamically - no Docker rebuild needed for Python changes!
"""
import os
import yaml
import logging
from datetime import datetime
from google.cloud import aiplatform, storage

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
def load_config():
    """Load configuration from orchestrator config."""
    with open('orchestrator/config.yaml', 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# GCP Project Configuration
PROJECT_ID = CONFIG['project']['id']  # ml-pipeline-477612
PROJECT_NUMBER = '303566498201'  # Numeric ID for TensorBoard resource names
REGION = CONFIG['project']['region']  # us-central1

# Compute and Storage
VERTEX_BUCKET = f'{PROJECT_ID}-training'
SERVICE_ACCOUNT = f'ml-pipeline-sa@{PROJECT_ID}.iam.gserviceaccount.com'
CONTAINER_URI = f'{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-containers/crop-trainer:latest'
MACHINE_TYPE = CONFIG.get('training', {}).get('machine_type', 'n1-standard-4')

# TensorBoard Configuration
TENSORBOARD_INSTANCE_ID = CONFIG.get('tensorboard', {}).get('instance_id', '3503556418512879616')  # New aligned instance
# Old instance: '1461173987500359680'
TENSORBOARD_RESOURCE_NAME = f'projects/{PROJECT_NUMBER}/locations/{REGION}/tensorboards/{TENSORBOARD_INSTANCE_ID}'

# --- Helper Functions ---
def upload_training_code(timestamp):
    """
    Upload training code to GCS staging bucket.
    This allows code changes without Docker rebuilds!
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(VERTEX_BUCKET)
    
    code_files = [
        'trainer/vertex_ai_training.py',
        'trainer/feature_engineering.py',
        'trainer/tensorboard_utils.py',
        'trainer/visualization_utils.py'
    ]
    
    code_uri = f"gs://{VERTEX_BUCKET}/code/{timestamp}"
    logger.info(f"📤 Uploading training code to {code_uri}")
    
    for file_path in code_files:
        if not os.path.exists(file_path):
            logger.warning(f"⚠️  {file_path} not found, skipping")
            continue
        
        blob_path = f"code/{timestamp}/{os.path.basename(file_path)}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(file_path)
        logger.info(f"   ✅ {os.path.basename(file_path)}")
    
    logger.info(f"✅ Code uploaded successfully")
    return code_uri

def trigger_training():
    """Submit a Vertex AI Custom Container Training Job with TensorBoard integration."""
    
    # Generate unique identifiers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'crop_training_{datetime.now().strftime("%Y%m%d")}'
    
    logger.info("=" * 70)
    logger.info("🚀 VERTEX AI TRAINING JOB SUBMISSION")
    logger.info("=" * 70)
    logger.info(f"Project: {PROJECT_ID}")
    logger.info(f"Region: {REGION}")
    logger.info(f"Machine Type: {MACHINE_TYPE}")
    logger.info("")
    
    # Upload training code dynamically
    code_uri = upload_training_code(timestamp)
    logger.info("")
    
    # 1. Initialize Vertex AI SDK
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f'gs://{VERTEX_BUCKET}'
    )
    logger.info("✅ Vertex AI SDK initialized")
    
    # 2. Define Custom Container Training Job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=f'crop-training-{timestamp}',
        container_uri=CONTAINER_URI,
        staging_bucket=f'gs://{VERTEX_BUCKET}/staging'
    )
    logger.info("✅ Training job defined")
    
    # 3. Configure Output Directories
    # Structure: gs://bucket/training_output/
    #   ├── models/
    #   └── logs/
    #       └── run_TIMESTAMP/  (TensorBoard event files)
    base_output_dir = f'gs://{VERTEX_BUCKET}/training_output'
    tensorboard_log_dir = f'{base_output_dir}/logs'
    
    # 4. Environment Variables for Container
    env_vars = {
        'AIP_TENSORBOARD_EXPERIMENT_NAME': experiment_name,
        'AIP_TENSORBOARD_LOG_DIR': tensorboard_log_dir,  # Where training uploads logs
        'AIP_TRAINING_DATA_URI': code_uri  # Code location for entrypoint to download
    }
    
    # 5. Display Configuration
    logger.info("")
    logger.info("📊 TENSORBOARD CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"TensorBoard Instance: {TENSORBOARD_RESOURCE_NAME}")
    logger.info(f"Experiment Name: {experiment_name}")
    logger.info(f"Log Directory: {tensorboard_log_dir}")
    logger.info("")
    logger.info("📁 OUTPUT DIRECTORIES")
    logger.info("=" * 70)
    logger.info(f"Base Output: {base_output_dir}")
    logger.info(f"TensorBoard Logs: {tensorboard_log_dir}")
    logger.info(f"Code Staging: {code_uri}")
    logger.info("")
    
    # 6. Submit Job with TensorBoard Integration
    logger.info("🚀 Submitting training job (async)...")
    logger.info("")
    
    job.run(
        replica_count=1,
        machine_type=MACHINE_TYPE,
        accelerator_count=0,
        service_account=SERVICE_ACCOUNT,
        environment_variables=env_vars,
        # CRITICAL: These two parameters enable TensorBoard integration
        base_output_dir=base_output_dir,  # Where Vertex AI writes artifacts
        tensorboard=TENSORBOARD_RESOURCE_NAME,  # Links to TensorBoard instance
        sync=False  # Don't wait for completion
    )
    
    logger.info("=" * 70)
    logger.info("✅ JOB SUBMITTED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Job Name: {job.display_name}")
    logger.info(f"Resource Name: {job.resource_name}")
    logger.info("")
    logger.info("📋 NEXT STEPS")
    logger.info("=" * 70)
    
    job_id = job.resource_name.split('/')[-1]
    logger.info("1. View logs:")
    logger.info(f"   gcloud ai custom-jobs stream-logs {job.resource_name} --project {PROJECT_ID}")
    logger.info("")
    logger.info("2. View in Console:")
    logger.info(f"   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    logger.info("")
    logger.info("3. View TensorBoard (after job completes):")
    logger.info(f"   https://console.cloud.google.com/vertex-ai/tensorboard/locations/{REGION}/tensorboards/{TENSORBOARD_INSTANCE_ID}?project={PROJECT_ID}")
    logger.info("")
    logger.info("💡 TIP: TensorBoard logs will appear at:")
    logger.info(f"   {tensorboard_log_dir}/run_{timestamp}/")
    logger.info("")
    
    return job

if __name__ == '__main__':
    try:
        trigger_training()
    except Exception as e:
        logger.error(f"❌ Failed to submit training job: {e}")
        raise
