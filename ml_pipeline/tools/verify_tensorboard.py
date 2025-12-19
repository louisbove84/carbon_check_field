#!/usr/bin/env python3
"""
TensorBoard Verification and Diagnostic Tool
===========================================
This script helps diagnose TensorBoard issues by:
- Inspecting event files for images, scalars, and other data
- Checking TensorBoard instance configuration
- Verifying GCS paths and file structure

Usage:
    # Verify images in latest training run
    python tools/verify_tensorboard.py --bucket ml-pipeline-477612-training --prefix training_output/logs/

    # Check TensorBoard instance
    python tools/verify_tensorboard.py --check-instance

    # Verify specific run
    python tools/verify_tensorboard.py --run run_20251215_202351
"""

import os
import sys
import argparse
import tempfile
import shutil
from google.cloud import storage
from tensorboard.backend.event_processing import event_accumulator
import yaml

# Load config
def load_config():
    """Load configuration from orchestrator config."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'orchestrator', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()
PROJECT_ID = CONFIG['project']['id']
REGION = CONFIG['project']['region']
DEFAULT_BUCKET = f'{PROJECT_ID}-training'
DEFAULT_PREFIX = 'training_output/logs/'


def download_and_inspect_event_files(bucket_name, gcs_prefix, project_id=None):
    """
    Download TensorBoard event files from GCS and inspect them for images.
    
    Args:
        bucket_name: GCS bucket name
        gcs_prefix: GCS path prefix (e.g., 'training_output/logs/')
        project_id: GCP project ID (optional)
    """
    print("=" * 70)
    print("🔍 TENSORBOARD IMAGE VERIFICATION")
    print("=" * 70)
    print(f"Bucket: {bucket_name}")
    print(f"Prefix: {gcs_prefix}")
    print()
    
    # Initialize storage client
    if project_id:
        storage_client = storage.Client(project=project_id)
    else:
        storage_client = storage.Client()
    
    bucket = storage_client.bucket(bucket_name)
    
    # List all event files
    print("📂 Finding TensorBoard event files...")
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    event_files = [b for b in blobs if b.name.endswith('.tfevents.') or 'events.out.tfevents' in b.name]
    
    if not event_files:
        print(f"❌ No event files found in gs://{bucket_name}/{gcs_prefix}")
        print()
        print("Available paths in bucket:")
        prefixes = set()
        for blob in blobs[:50]:  # Limit to avoid too much output
            parts = blob.name.split('/')
            if len(parts) > 1:
                prefixes.add(parts[0] + '/')
        for prefix in sorted(prefixes):
            print(f"  - {prefix}")
        return
    
    print(f"✅ Found {len(event_files)} event file(s)")
    print()
    
    # Download and inspect each event file
    temp_dir = tempfile.mkdtemp()
    try:
        total_images = 0
        total_scalars = 0
        
        for i, blob in enumerate(event_files, 1):
            print(f"📥 Downloading file {i}/{len(event_files)}: {blob.name}")
            local_path = os.path.join(temp_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            
            # Inspect event file
            print(f"   Inspecting: {blob.name}")
            try:
                ea = event_accumulator.EventAccumulator(
                    os.path.dirname(local_path),
                    size_guidance={
                        event_accumulator.IMAGES: 0,  # Load all images
                        event_accumulator.SCALARS: 0,
                        event_accumulator.FIGURES: 0,
                    }
                )
                ea.Reload()
                
                # Check for scalars
                scalar_tags = ea.Tags().get('scalars', [])
                print(f"   📊 Scalars found: {len(scalar_tags)}")
                if scalar_tags:
                    print(f"      Examples: {scalar_tags[:5]}")
                    total_scalars += len(scalar_tags)
                
                # Check for images
                image_tags = ea.Tags().get('images', [])
                print(f"   🖼️  Images found: {len(image_tags)}")
                if image_tags:
                    print(f"      Image tags:")
                    for tag in image_tags:
                        print(f"         - {tag}")
                        # Get image data
                        image_events = ea.Images(tag)
                        if image_events:
                            print(f"           Count: {len(image_events)}")
                            img = image_events[0]
                            print(f"           Shape: {img.width}x{img.height}")
                            print(f"           Step: {img.step}")
                            total_images += len(image_events)
                else:
                    print(f"      ⚠️  No images found in this event file!")
                
                # Check for figures (matplotlib plots)
                figure_tags = ea.Tags().get('figures', [])
                print(f"   📈 Figures found: {len(figure_tags)}")
                if figure_tags:
                    print(f"      Figure tags:")
                    for tag in figure_tags[:5]:  # Limit output
                        print(f"         - {tag}")
                    if len(figure_tags) > 5:
                        print(f"         ... and {len(figure_tags) - 5} more")
                
                print()
                
            except Exception as e:
                print(f"   ❌ Error inspecting file: {e}")
                import traceback
                print(traceback.format_exc())
                print()
        
        print("=" * 70)
        print("✅ VERIFICATION SUMMARY")
        print("=" * 70)
        print(f"Total event files: {len(event_files)}")
        print(f"Total scalar tags: {total_scalars}")
        print(f"Total image tags: {len(image_tags) if 'image_tags' in locals() else 0}")
        print(f"Total images logged: {total_images}")
        print()
        
        if total_images == 0:
            print("⚠️  WARNING: No images found in event files!")
            print("   This could indicate:")
            print("   - Matplotlib dependencies missing in Docker image")
            print("   - Image logging code not executed")
            print("   - Images failed to render")
        else:
            print("✅ Images successfully logged to TensorBoard")
    
    finally:
        shutil.rmtree(temp_dir)


def check_tensorboard_instance(project_id, region, tensorboard_id):
    """
    Check TensorBoard instance configuration.
    """
    print("=" * 70)
    print("🔍 TENSORBOARD INSTANCE CONFIGURATION")
    print("=" * 70)
    
    try:
        from google.cloud import aiplatform
        
        aiplatform.init(project=project_id, location=region)
        
        tensorboard_name = f'projects/{project_id}/locations/{region}/tensorboards/{tensorboard_id}'
        
        print(f"Project: {project_id}")
        print(f"Region: {region}")
        print(f"Instance ID: {tensorboard_id}")
        print(f"Resource Name: {tensorboard_name}")
        print()
        
        # Try to get TensorBoard instance
        try:
            tb = aiplatform.Tensorboard(tensorboard_name=tensorboard_name)
            print(f"✅ TensorBoard instance found")
            print(f"   Display Name: {tb.display_name}")
            print(f"   Create Time: {tb.create_time}")
            print()
        except Exception as e:
            print(f"⚠️  Could not retrieve TensorBoard instance: {e}")
            print()
        
        # List experiments
        try:
            experiments = aiplatform.TensorboardExperiment.list(
                tensorboard_name=tensorboard_name,
                order_by="create_time desc",
                limit=10
            )
            
            print(f"📊 Recent Experiments ({len(experiments)} shown):")
            for exp in experiments:
                exp_name = exp.display_name or exp.name.split('/')[-1]
                print(f"   - {exp_name}")
                print(f"     Created: {exp.create_time}")
                print(f"     Updated: {exp.update_time}")
                print()
        except Exception as e:
            print(f"⚠️  Could not list experiments: {e}")
            print()
        
        print("To check TensorBoard instance details, run:")
        print(f"  gcloud ai tensorboards describe {tensorboard_id} --region={region} --project={project_id}")
        print()
        print("To list TensorBoard experiments, run:")
        print(f"  gcloud ai tensorboard-experiments list --tensorboard={tensorboard_id} --region={region} --project={project_id}")
        
    except Exception as e:
        print(f"❌ Error checking TensorBoard instance: {e}")
        import traceback
        print(traceback.format_exc())


def list_gcs_runs(bucket_name, prefix, project_id=None):
    """List available training runs in GCS."""
    print("=" * 70)
    print("📂 LISTING TRAINING RUNS")
    print("=" * 70)
    print(f"Bucket: {bucket_name}")
    print(f"Prefix: {prefix}")
    print()
    
    if project_id:
        storage_client = storage.Client(project=project_id)
    else:
        storage_client = storage.Client()
    
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # Extract unique run directories
    runs = set()
    for blob in blobs:
        parts = blob.name.replace(prefix, '').split('/')
        if len(parts) > 0 and parts[0]:
            runs.add(parts[0])
    
    if not runs:
        print("❌ No runs found")
        return
    
    print(f"Found {len(runs)} run(s):")
    for run in sorted(runs, reverse=True):  # Most recent first
        print(f"  - {run}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Verify TensorBoard image logging and configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--bucket',
        type=str,
        default=DEFAULT_BUCKET,
        help=f'GCS bucket name (default: {DEFAULT_BUCKET})'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default=DEFAULT_PREFIX,
        help=f'GCS path prefix (default: {DEFAULT_PREFIX})'
    )
    parser.add_argument(
        '--run',
        type=str,
        help='Specific run directory to verify (e.g., run_20251215_202351)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default=PROJECT_ID,
        help=f'GCP project ID (default: {PROJECT_ID})'
    )
    parser.add_argument(
        '--region',
        type=str,
        default=REGION,
        help=f'GCP region (default: {REGION})'
    )
    parser.add_argument(
        '--tensorboard-id',
        type=str,
        help='TensorBoard instance ID (from config if not provided)'
    )
    parser.add_argument(
        '--check-instance',
        action='store_true',
        help='Check TensorBoard instance configuration'
    )
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='List available training runs'
    )
    
    args = parser.parse_args()
    
    # Get TensorBoard ID from config if not provided
    if args.check_instance and not args.tensorboard_id:
        args.tensorboard_id = CONFIG.get('tensorboard', {}).get('instance_id')
        if not args.tensorboard_id:
            print("❌ ERROR: TensorBoard instance ID not found in config")
            print("   Provide --tensorboard-id or check config.yaml")
            return
    
    # List runs
    if args.list_runs:
        list_gcs_runs(args.bucket, args.prefix, args.project)
        return
    
    # Check instance
    if args.check_instance:
        check_tensorboard_instance(args.project, args.region, args.tensorboard_id)
        print()
    
    # Verify event files
    prefix = args.prefix
    if args.run:
        prefix = f"{prefix.rstrip('/')}/{args.run}/"
    
    download_and_inspect_event_files(args.bucket, prefix, args.project)


if __name__ == '__main__':
    main()

