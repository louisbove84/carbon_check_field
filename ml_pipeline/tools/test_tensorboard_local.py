#!/usr/bin/env python3
"""
Simple TensorBoard test script to verify image logging works.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from google.cloud import storage
from datetime import datetime

def create_test_image():
    """Create a simple test image using matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Test Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def fig_to_array(fig):
    """Convert matplotlib figure to numpy array for TensorBoard."""
    import io
    from PIL import Image
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array and transpose to CHW format
    image_array = np.array(image, dtype=np.uint8)
    image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
    
    # Normalize to [0, 1]
    image_array = image_array.astype(np.float32) / 255.0
    
    buf.close()
    return image_array

def upload_to_gcs(local_dir, gcs_path):
    """Upload directory contents to GCS."""
    print(f"📤 Uploading to {gcs_path}")
    
    # Parse GCS path
    gcs_path_parts = gcs_path.replace('gs://', '').split('/', 1)
    bucket_name = gcs_path_parts[0]
    gcs_prefix = gcs_path_parts[1] if len(gcs_path_parts) > 1 else ''
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    uploaded = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            gcs_blob_path = os.path.join(gcs_prefix, relative_path) if gcs_prefix else relative_path
            
            blob = bucket.blob(gcs_blob_path)
            blob.upload_from_filename(local_path)
            uploaded.append(gcs_blob_path)
            print(f"   ✅ {gcs_blob_path} ({os.path.getsize(local_path)} bytes)")
    
    return uploaded

def main():
    print("=" * 70)
    print("🧪 TENSORBOARD TEST")
    print("=" * 70)
    
    # Create local directory
    local_dir = '/tmp/tb_test'
    os.makedirs(local_dir, exist_ok=True)
    print(f"📁 Writing to: {local_dir}")
    
    # Create writer
    writer = SummaryWriter(log_dir=local_dir)
    
    # Test 1: Log scalars
    print("\n1️⃣ Logging scalars...")
    for i in range(10):
        writer.add_scalar('test/accuracy', 0.8 + i * 0.02, i)
        writer.add_scalar('test/loss', 1.0 - i * 0.08, i)
    print("   ✅ Logged 20 scalar values")
    
    # Test 2: Log images
    print("\n2️⃣ Logging images...")
    for i in range(3):
        fig = create_test_image()
        image_array = fig_to_array(fig)
        writer.add_image(f'test/plot_{i}', image_array, i, dataformats='CHW')
        plt.close(fig)
    print("   ✅ Logged 3 images")
    
    # Test 3: Log text
    print("\n3️⃣ Logging text...")
    writer.add_text('test/summary', 'This is a test summary with **markdown**!', 0)
    print("   ✅ Logged text")
    
    # Close writer
    print("\n4️⃣ Closing writer...")
    writer.close()
    print("   ✅ Writer closed")
    
    # Check local files
    print("\n5️⃣ Checking local files...")
    local_files = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            local_files.append((file, size))
            print(f"   📄 {file}: {size:,} bytes")
    
    if not local_files:
        print("   ❌ ERROR: No files created!")
        return
    
    # Upload to GCS
    print("\n6️⃣ Uploading to GCS...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_path = f"gs://ml-pipeline-477612-training/training_output/logs/test_{timestamp}"
    
    try:
        uploaded_files = upload_to_gcs(local_dir, gcs_path)
        print(f"   ✅ Uploaded {len(uploaded_files)} files")
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        return
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    print(f"📊 TensorBoard URL: https://console.cloud.google.com/vertex-ai/tensorboard/")
    print(f"📁 GCS Path: {gcs_path}")
    print(f"📄 Files created: {len(local_files)}")
    print(f"📤 Files uploaded: {len(uploaded_files)}")
    print("\nTo view locally:")
    print(f"  tensorboard --logdir {local_dir}")
    print("\nOr from GCS:")
    print(f"  tensorboard --logdir {gcs_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()

