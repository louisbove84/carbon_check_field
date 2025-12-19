#!/bin/bash
set -e

echo "🚀 Starting Vertex AI Training Container"
echo "========================================"

# AIP_TRAINING_DATA_URI is set by Vertex AI to the staging bucket location
# where we upload our code
if [ -n "$AIP_TRAINING_DATA_URI" ]; then
    echo "📥 Downloading training code from: $AIP_TRAINING_DATA_URI"
    
    # Use Python google-cloud-storage instead of gsutil (already installed)
    python3 << 'PYEOF'
import os
from google.cloud import storage

code_uri = os.environ.get('AIP_TRAINING_DATA_URI', '')
if code_uri.startswith('gs://'):
    # Parse GCS path: gs://bucket/path/to/code
    parts = code_uri.replace('gs://', '').split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Download all files with the prefix
    blobs = list(bucket.list_blobs(prefix=prefix))
    print(f"   Found {len(blobs)} files to download")
    
    for blob in blobs:
        if blob.name.endswith('.py'):  # Only download Python files
            filename = os.path.basename(blob.name)
            dest_path = f'/app/{filename}'
            blob.download_to_filename(dest_path)
            print(f"   ✅ {filename}")
PYEOF
    
    echo "✅ Code downloaded successfully"
else
    echo "⚠️  AIP_TRAINING_DATA_URI not set, looking for code in /app/"
fi

# List downloaded files for debugging
echo ""
echo "📂 Files in /app/:"
ls -lh /app/*.py 2>/dev/null || echo "   No .py files found!"

# Clear Python cache to ensure new code is used
echo ""
echo "🧹 Clearing Python cache..."
find /app -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find /app -name "*.pyc" -delete 2>/dev/null || true
echo "   ✅ Cache cleared"

echo ""
echo "🎯 Starting training script..."
echo "========================================"

# Run the training script
python /app/vertex_ai_training.py

