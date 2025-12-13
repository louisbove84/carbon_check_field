#!/usr/bin/env python3
"""
Verify TensorBoard image logging by inspecting event files.
This helps diagnose why images aren't showing up in TensorBoard UI.
"""

import os
import sys
from google.cloud import storage
from tensorboard.backend.event_processing import event_accumulator
import tempfile
import shutil

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
        return
    
    print(f"✅ Found {len(event_files)} event file(s)")
    print()
    
    # Download and inspect each event file
    temp_dir = tempfile.mkdtemp()
    try:
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
                    }
                )
                ea.Reload()
                
                # Check for images
                image_tags = ea.Tags().get('images', [])
                scalar_tags = ea.Tags().get('scalars', [])
                
                print(f"   📊 Scalars found: {len(scalar_tags)}")
                if scalar_tags:
                    print(f"      Examples: {scalar_tags[:5]}")
                
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
                else:
                    print(f"      ⚠️  No images found in this event file!")
                
                print()
                
            except Exception as e:
                print(f"   ❌ Error inspecting file: {e}")
                print()
    
    finally:
        shutil.rmtree(temp_dir)
    
    print("=" * 70)
    print("✅ Verification complete")
    print("=" * 70)


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
        
        print(f"Instance: {tensorboard_name}")
        print()
        print("To check TensorBoard instance details, run:")
        print(f"  gcloud ai tensorboards describe {tensorboard_id} --region={region} --project={project_id}")
        print()
        print("To list TensorBoard experiments, run:")
        print(f"  gcloud ai tensorboard-experiments list --tensorboard={tensorboard_id} --region={region} --project={project_id}")
        
    except Exception as e:
        print(f"❌ Error checking TensorBoard instance: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify TensorBoard image logging')
    parser.add_argument('--bucket', type=str, default='ml-pipeline-477612-training',
                       help='GCS bucket name')
    parser.add_argument('--prefix', type=str, default='training_output/logs/',
                       help='GCS path prefix')
    parser.add_argument('--project', type=str, default='ml-pipeline-477612',
                       help='GCP project ID')
    parser.add_argument('--region', type=str, default='us-central1',
                       help='GCP region')
    parser.add_argument('--tensorboard-id', type=str, default='1461173987500359680',
                       help='TensorBoard instance ID')
    parser.add_argument('--check-instance', action='store_true',
                       help='Check TensorBoard instance configuration')
    
    args = parser.parse_args()
    
    if args.check_instance:
        check_tensorboard_instance(args.project, args.region, args.tensorboard_id)
        print()
    
    download_and_inspect_event_files(args.bucket, args.prefix, args.project)

