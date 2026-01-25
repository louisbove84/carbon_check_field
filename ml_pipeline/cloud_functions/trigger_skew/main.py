"""
Cloud Function to trigger Skew Audit Vertex AI Job
===================================================
This function is triggered by Cloud Scheduler monthly to run the skew audit.
"""

import os
import functions_framework
from google.cloud import aiplatform
from datetime import datetime


# Configuration
PROJECT_ID = os.environ.get('PROJECT_ID', 'ml-pipeline-477612')
REGION = os.environ.get('REGION', 'us-central1')
PROJECT_NUMBER = os.environ.get('PROJECT_NUMBER', '303566498201')
TENSORBOARD_ID = os.environ.get('TENSORBOARD_ID', '1778818498718334976')


def upload_skew_code(storage_client, vertex_bucket: str, timestamp: str) -> str:
    """Upload skew code to GCS for dynamic loading."""
    from google.cloud import storage
    
    bucket = storage_client.bucket(vertex_bucket)
    code_prefix = f'skew_code/{timestamp}'
    
    # Files to upload (from the skew_job directory in the repo)
    # Note: These are bundled with the function deployment
    code_files = ['vertex_ai_skew.py', 'skew_detector.py']
    
    for filename in code_files:
        local_path = f'/workspace/skew_job/{filename}'
        if os.path.exists(local_path):
            blob = bucket.blob(f'{code_prefix}/{filename}')
            blob.upload_from_filename(local_path)
            print(f"Uploaded {filename}")
    
    return f'gs://{vertex_bucket}/{code_prefix}'


@functions_framework.http
def trigger_skew_audit(request):
    """
    HTTP Cloud Function to trigger Vertex AI Skew Audit job.
    
    This is called by Cloud Scheduler on a monthly basis.
    """
    print("=" * 60)
    print("TRIGGERING SKEW AUDIT JOB")
    print("=" * 60)
    
    try:
        # Initialize Vertex AI
        vertex_bucket = f'{PROJECT_ID}-training'
        
        aiplatform.init(
            project=PROJECT_ID,
            location=REGION,
            staging_bucket=f'gs://{vertex_bucket}'
        )
        
        # Generate timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'skew-audit-{datetime.now().strftime("%Y%m%d")}'
        
        # Code URI - the skew job container downloads code from here
        code_uri = f'gs://{vertex_bucket}/skew_code/{timestamp}'
        
        # Upload current code to GCS
        from google.cloud import storage
        storage_client = storage.Client(project=PROJECT_ID)
        
        # Upload skew_detector.py and vertex_ai_skew.py
        bucket = storage_client.bucket(vertex_bucket)
        
        # Read code from bundled files or fetch from repo
        import urllib.request
        base_url = 'https://raw.githubusercontent.com/louisbove84/carbon_check_field/main/ml_pipeline/skew_job'
        
        for filename in ['vertex_ai_skew.py', 'skew_detector.py']:
            url = f'{base_url}/{filename}'
            print(f"Fetching {filename} from GitHub...")
            
            try:
                with urllib.request.urlopen(url) as response:
                    content = response.read()
                    blob = bucket.blob(f'skew_code/{timestamp}/{filename}')
                    blob.upload_from_string(content)
                    print(f"  Uploaded {filename} ({len(content)} bytes)")
            except Exception as e:
                print(f"  Error fetching {filename}: {e}")
                return {'status': 'error', 'error': f'Failed to fetch {filename}: {e}'}, 500
        
        print(f"Code uploaded to: {code_uri}")
        
        # Create the custom job
        job = aiplatform.CustomContainerTrainingJob(
            display_name=f'skew-audit-{datetime.now().strftime("%Y%m%d_%H%M")}',
            container_uri=f'{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-containers/skew-audit:latest',
            staging_bucket=f'gs://{vertex_bucket}/staging'
        )
        
        # TensorBoard resource name
        tensorboard_name = f'projects/{PROJECT_NUMBER}/locations/{REGION}/tensorboards/{TENSORBOARD_ID}'
        
        # Service account
        service_account = f'ml-pipeline-sa@{PROJECT_ID}.iam.gserviceaccount.com'
        
        # Environment variables
        env_vars = {
            'AIP_TENSORBOARD_EXPERIMENT_NAME': experiment_name,
            'AIP_TRAINING_DATA_URI': code_uri
        }
        
        print(f"Starting Vertex AI job...")
        print(f"  TensorBoard: {tensorboard_name}")
        print(f"  Experiment: {experiment_name}")
        print(f"  Code URI: {code_uri}")
        
        # Run the job (non-blocking for Cloud Function)
        # We use sync=False to avoid timeout - job runs in background
        job.run(
            replica_count=1,
            machine_type='n1-standard-4',
            accelerator_count=0,
            base_output_dir=f'gs://{vertex_bucket}/skew_output',
            service_account=service_account,
            environment_variables=env_vars,
            tensorboard=tensorboard_name,
            sync=False  # Don't wait for completion
        )
        
        print("=" * 60)
        print("SKEW AUDIT JOB SUBMITTED")
        print("=" * 60)
        
        # Note: With sync=False, job.display_name is not available
        # The job is submitted and will run in the background
        return {
            'status': 'success',
            'message': 'Skew audit job submitted successfully',
            'experiment': experiment_name,
            'tensorboard_id': TENSORBOARD_ID,
            'code_uri': code_uri
        }, 200
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}, 500
