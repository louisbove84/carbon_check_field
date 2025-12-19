#!/usr/bin/env python3
"""
Setup GCP Resources and Run Full Pipeline
==========================================
This script:
1. Creates required GCS buckets
2. Creates TensorBoard instance
3. Uploads config to GCS
4. Runs the complete pipeline (Earth Engine → Training)
"""

import os
import sys
import yaml
import logging
from google.cloud import storage, bigquery, aiplatform
from google.cloud.aiplatform import tensorboard
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load config
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'orchestrator', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()
PROJECT_ID = CONFIG['project']['id']
PROJECT_NUMBER = '303566498201'
REGION = CONFIG['project']['region']
DATA_BUCKET = CONFIG['storage']['bucket']
TRAINING_BUCKET = f'{PROJECT_ID}-training'

def create_gcs_buckets():
    """Create required GCS buckets"""
    logger.info("=" * 70)
    logger.info("📦 CREATING GCS BUCKETS")
    logger.info("=" * 70)
    
    storage_client = storage.Client(project=PROJECT_ID)
    
    buckets_to_create = [
        (DATA_BUCKET, 'us-central1'),
        (TRAINING_BUCKET, REGION)
    ]
    
    for bucket_name, location in buckets_to_create:
        try:
            bucket = storage_client.bucket(bucket_name)
            if bucket.exists():
                logger.info(f"  ✅ Bucket {bucket_name} already exists")
            else:
                bucket.location = location
                bucket.create()
                logger.info(f"  ✅ Created bucket: {bucket_name} in {location}")
        except Exception as e:
            logger.error(f"  ❌ Failed to create {bucket_name}: {e}")
            raise

def upload_config_to_gcs():
    """Upload config.yaml to GCS"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📤 UPLOADING CONFIG TO GCS")
    logger.info("=" * 70)
    
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(DATA_BUCKET)
    
    # Create config directory structure
    config_path = os.path.join(os.path.dirname(__file__), 'orchestrator', 'config.yaml')
    blob = bucket.blob('config/config.yaml')
    blob.upload_from_filename(config_path)
    logger.info(f"  ✅ Uploaded config to gs://{DATA_BUCKET}/config/config.yaml")

def create_tensorboard_instance():
    """Create TensorBoard instance"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 CREATING TENSORBOARD INSTANCE")
    logger.info("=" * 70)
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    try:
        # Check if TensorBoard instance already exists
        instances = tensorboard.Tensorboard.list()
        if instances:
            instance = list(instances)[0]
            logger.info(f"  ✅ TensorBoard instance already exists: {instance.display_name}")
            logger.info(f"     Resource: {instance.resource_name}")
            return instance.resource_name
        
        # Create new TensorBoard instance
        logger.info("  Creating new TensorBoard instance...")
        tb_instance = tensorboard.Tensorboard.create(
            display_name="Crop Classifier Training",
            description="TensorBoard for crop classification model training metrics and visualizations"
        )
        logger.info(f"  ✅ Created TensorBoard instance: {tb_instance.display_name}")
        logger.info(f"     Resource: {tb_instance.resource_name}")
        
        # Update config with TensorBoard instance ID
        instance_id = tb_instance.name.split('/')[-1]
        CONFIG['tensorboard'] = {'instance_id': instance_id}
        logger.info(f"     Instance ID: {instance_id}")
        
        return tb_instance.resource_name
        
    except Exception as e:
        logger.error(f"  ❌ Failed to create TensorBoard: {e}")
        raise

def create_bigquery_resources():
    """Create BigQuery dataset and tables if they don't exist"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 CREATING BIGQUERY RESOURCES")
    logger.info("=" * 70)
    
    bq_client = bigquery.Client(project=PROJECT_ID)
    dataset_id = CONFIG['bigquery']['dataset']
    
    # Create dataset
    dataset_ref = bq_client.dataset(dataset_id)
    try:
        dataset = bq_client.get_dataset(dataset_ref)
        logger.info(f"  ✅ Dataset {dataset_id} already exists")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = REGION
        dataset = bq_client.create_dataset(dataset, exists_ok=True)
        logger.info(f"  ✅ Created dataset: {dataset_id}")
    
    # Tables will be created automatically by the pipeline when data is exported
    logger.info("  ℹ️  Tables will be created automatically during data export")

def run_full_pipeline():
    """Run the complete ML pipeline"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("🚀 RUNNING FULL PIPELINE")
    logger.info("=" * 70)
    
    # Import orchestrator
    orchestrator_path = os.path.join(os.path.dirname(__file__), 'orchestrator')
    sys.path.insert(0, orchestrator_path)
    import orchestrator
    
    result = orchestrator.run_pipeline()
    
    if result['status'] == 'success':
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {result.get('duration_minutes', 0):.1f} minutes")
        if 'training' in result:
            logger.info(f"Training Accuracy: {result['training'].get('accuracy', 0):.2%}")
    else:
        logger.error("")
        logger.error("=" * 70)
        logger.error("❌ PIPELINE FAILED")
        logger.error("=" * 70)
        logger.error(f"Error: {result.get('error', 'Unknown error')}")
    
    return result

def main():
    """Main setup and execution"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("🌾 ML PIPELINE SETUP AND EXECUTION")
    logger.info("=" * 70)
    logger.info(f"Project: {PROJECT_ID}")
    logger.info(f"Region: {REGION}")
    logger.info("")
    
    try:
        # Step 1: Create GCS buckets
        create_gcs_buckets()
        
        # Step 2: Upload config
        upload_config_to_gcs()
        
        # Step 3: Create TensorBoard instance
        tensorboard_resource = create_tensorboard_instance()
        
        # Step 4: Create BigQuery resources
        create_bigquery_resources()
        
        # Step 5: Run full pipeline
        result = run_full_pipeline()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ SETUP AND PIPELINE COMPLETE")
        logger.info("=" * 70)
        
        return result
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 70)
        logger.error("❌ SETUP FAILED")
        logger.error("=" * 70)
        logger.error(f"Error: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()

