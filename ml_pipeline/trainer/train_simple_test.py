"""
Simple TensorBoard test script - just logs a few things to verify it works
"""
import os
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import subprocess

print("=" * 70)
print("🧪 TENSORBOARD SIMPLE TEST")
print("=" * 70)

# Check environment variables
tensorboard_gcs_path = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
print(f"\n📊 Environment:")
print(f"   AIP_TENSORBOARD_LOG_DIR: {tensorboard_gcs_path}")
print(f"   AIP_MODEL_DIR: {os.environ.get('AIP_MODEL_DIR')}")

# Create local directory
local_dir = '/tmp/tensorboard_test'
os.makedirs(local_dir, exist_ok=True)
print(f"\n📁 Local directory: {local_dir}")

# Create writer
print(f"\n✍️  Creating SummaryWriter...")
writer = SummaryWriter(log_dir=local_dir)
print(f"   ✅ Writer created")

# Log some simple scalars
print(f"\n📈 Logging scalars...")
for i in range(5):
    writer.add_scalar('test/accuracy', 0.9 + i*0.01, i)
    writer.add_scalar('test/loss', 0.5 - i*0.05, i)
    print(f"   Step {i}: accuracy={0.9 + i*0.01:.2f}, loss={0.5 - i*0.05:.2f}")

# Log text
print(f"\n📝 Logging text...")
writer.add_text('test/summary', """
Test Summary
============
This is a simple test to verify TensorBoard logging works.
If you can see this, the basic logging is working!
""", 0)
print(f"   ✅ Text logged")

# Log a simple image
print(f"\n🖼️  Logging image...")
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
ax.plot(x, np.cos(x), label='cos(x)', linewidth=2)
ax.set_title('Simple Test Plot', fontsize=14, fontweight='bold')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Convert to image
buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=100)
buf.seek(0)
image = Image.open(buf)
image_array = np.array(image)
if len(image_array.shape) == 3:
    image_array = np.transpose(image_array, (2, 0, 1))
writer.add_image('test/plot', image_array, 0)
plt.close(fig)
print(f"   ✅ Image logged")

# Close writer
print(f"\n💾 Closing writer...")
writer.close()
print(f"   ✅ Writer closed")

# List files created
print(f"\n📂 Files created:")
for root, dirs, files in os.walk(local_dir):
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        print(f"   {filepath} ({size} bytes)")

# Upload to GCS if path provided
if tensorboard_gcs_path:
    print(f"\n📤 Uploading to GCS...")
    print(f"   Source: {local_dir}")
    print(f"   Destination: {tensorboard_gcs_path}")
    
    try:
        result = subprocess.run(
            ['gsutil', '-m', 'rsync', '-r', '-v', local_dir, tensorboard_gcs_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"\n📋 Upload output:")
        if result.stdout:
            print(f"   STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"   STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            print(f"\n✅ Upload successful!")
            
            # List what's in GCS
            print(f"\n📂 GCS contents:")
            list_result = subprocess.run(
                ['gsutil', 'ls', '-lh', f'{tensorboard_gcs_path}/'],
                capture_output=True,
                text=True,
                timeout=60
            )
            print(list_result.stdout)
        else:
            print(f"\n❌ Upload failed with code {result.returncode}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print(f"\n⚠️  No AIP_TENSORBOARD_LOG_DIR set - skipping upload")

print(f"\n" + "=" * 70)
print(f"✅ TEST COMPLETE")
print(f"=" * 70)
print(f"\nIf this worked, you should see in TensorBoard:")
print(f"  - Scalars: test/accuracy, test/loss (5 data points each)")
print(f"  - Images: test/plot (sine/cosine graph)")
print(f"  - Text: test/summary")

