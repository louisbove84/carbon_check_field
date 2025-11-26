#!/bin/bash
# Upload Configuration to Cloud Storage
# ======================================
# Uploads config.yaml to GCS where all Cloud Functions can access it.
# Changes take effect on next function execution (no redeployment needed!)

set -e

BUCKET="carboncheck-data"
CONFIG_PATH="config/config.yaml"
LOCAL_FILE="config.yaml"

echo "============================================================"
echo "📤 UPLOADING CONFIGURATION TO CLOUD STORAGE"
echo "============================================================"
echo ""

# Check if config.yaml exists
if [ ! -f "$LOCAL_FILE" ]; then
    echo "❌ Error: $LOCAL_FILE not found"
    exit 1
fi

# Validate YAML syntax
echo "🔍 Validating YAML syntax..."
python3 << EOF
import yaml
import sys

try:
    with open('$LOCAL_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ YAML syntax is valid")
    
    # Print summary
    print("")
    print("Configuration Summary:")
    print(f"  Project ID:        {config['project']['id']}")
    print(f"  Region:            {config['project']['region']}")
    print(f"  Quality Gates:")
    print(f"    Min Accuracy:    {config['quality_gates']['absolute_min_accuracy']:.0%}")
    print(f"    Min Crop F1:     {config['quality_gates']['min_per_crop_f1']:.0%}")
    print(f"    Improvement:     {config['quality_gates']['improvement_margin']:.0%}")
    print(f"  Data Collection:")
    print(f"    Samples/Crop:    {config['data_collection']['samples_per_crop']}")
    print("")
    
except yaml.YAMLError as e:
    print(f"❌ YAML syntax error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Configuration validation failed"
    exit 1
fi

# Create backup of current config (if exists)
echo "💾 Creating backup of current config..."
gsutil -m cp "gs://$BUCKET/$CONFIG_PATH" "gs://$BUCKET/config/backups/config_$(date +%Y%m%d_%H%M%S).yaml" 2>/dev/null || echo "   No previous config to backup"

# Upload new config
echo "📤 Uploading to gs://$BUCKET/$CONFIG_PATH..."
gsutil cp "$LOCAL_FILE" "gs://$BUCKET/$CONFIG_PATH"

# Verify upload
echo "🔍 Verifying upload..."
if gsutil ls "gs://$BUCKET/$CONFIG_PATH" > /dev/null 2>&1; then
    echo "✅ Upload successful!"
else
    echo "❌ Upload verification failed"
    exit 1
fi

# Show file info
echo ""
echo "============================================================"
echo "✅ CONFIGURATION UPDATED"
echo "============================================================"
echo ""
echo "📍 Location: gs://$BUCKET/$CONFIG_PATH"
echo ""

# Display uploaded config
echo "📄 Uploaded Configuration:"
echo "------------------------------------------------------------"
gsutil cat "gs://$BUCKET/$CONFIG_PATH" | head -50
echo "..."
echo ""

# Instructions
echo "============================================================"
echo "📋 NEXT STEPS"
echo "============================================================"
echo ""
echo "The new configuration will be used by Cloud Functions on their"
echo "next execution. No redeployment needed!"
echo ""
echo "To test immediately:"
echo "  ./test_pipeline.sh retrain"
echo ""
echo "To view current config:"
echo "  gsutil cat gs://$BUCKET/$CONFIG_PATH"
echo ""
echo "To view backups:"
echo "  gsutil ls gs://$BUCKET/config/backups/"
echo ""
echo "To rollback to previous version:"
echo "  gsutil cp gs://$BUCKET/config/backups/config_YYYYMMDD_HHMMSS.yaml \\"
echo "    gs://$BUCKET/$CONFIG_PATH"
echo ""

