#!/usr/bin/env python3
"""
Trigger a Vertex AI training job manually.
"""
import yaml
from datetime import datetime
from google.cloud import aiplatform

def load_config():
    """Load configuration from orchestrator config."""
    with open('orchestrator/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def trigger_training():
    """Trigger a Vertex AI training job."""
    # Load config
    config = load_config()
    project_id = config['project']['id']
    region = config['project']['region']
    
    # Training configuration
    machine_type = config.get('training', {}).get('machine_type', 'n1-standard-4')
    
    print(f"🚀 Triggering training job...")
    print(f"   Project: {project_id}")
    print(f"   Region: {region}")
    print(f"   Machine type: {machine_type}")
    
    # Use regional bucket for Vertex AI
    vertex_bucket = f'{project_id}-training'
    
    # Initialize Vertex AI
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=f'gs://{vertex_bucket}'
    )
    
    # Define custom training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=f'crop-training-{datetime.now().strftime("%Y%m%d_%H%M")}',
        container_uri=f'{region}-docker.pkg.dev/{project_id}/ml-containers/crop-trainer:latest',
        staging_bucket=f'gs://{vertex_bucket}/staging'
    )
    
    # Get TensorBoard instance resource name from config
    tensorboard_id = config.get('tensorboard', {}).get('instance_id', '1461173987500359680')
    # Use numeric project ID for TensorBoard resource name
    tensorboard_name = f'projects/303566498201/locations/{region}/tensorboards/{tensorboard_id}'
    
    # Service account for training job
    service_account = f'ml-pipeline-sa@{project_id}.iam.gserviceaccount.com'
    
    # Run training job with TensorBoard
    print(f"   TensorBoard ID: {tensorboard_id}")
    print(f"   Service account: {service_account}")
    print("")
    print("   Starting training job (this will run async)...")
    
    model = job.run(
        replica_count=1,
        machine_type=machine_type,
        accelerator_count=0,
        base_output_dir=f'gs://{vertex_bucket}/training_output',
        tensorboard=tensorboard_name,
        service_account=service_account,
        sync=False  # Don't wait for completion
    )
    
    print("")
    print("✅ Training job submitted!")
    print(f"   Job name: {job.display_name}")
    print(f"   Resource name: {job.resource_name}")
    print("")
    print("To view logs, run:")
    job_id = job.resource_name.split('/')[-1]
    print(f"   gcloud ai custom-jobs stream-logs {job.resource_name} --project {project_id}")
    print("")
    print(f"Or view in console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")
    
    return job

if __name__ == '__main__':
    trigger_training()

