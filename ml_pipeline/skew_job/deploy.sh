#!/bin/bash
# Trigger Skew Audit as Vertex AI Custom Job
# ===========================================
#
# This script triggers a skew audit job on Vertex AI with native TensorBoard
# integration, enabling proper image visibility in the TensorBoard UI.
#
# The job:
# 1. Uploads skew code to GCS
# 2. Runs as a Vertex AI Custom Job
# 3. Writes TensorBoard logs to AIP_TENSORBOARD_LOG_DIR (handled automatically)
# 4. Results are stored in BigQuery
#
# Prerequisites:
# - Docker image built: ./build_docker.sh
# - TensorBoard instance configured in config.yaml
#
# Usage:
#   cd ml_pipeline/skew_job
#   ./deploy.sh

set -e

PROJECT_ID="ml-pipeline-477612"
REGION="us-central1"

echo "=============================================="
echo "🔍 TRIGGERING SKEW AUDIT (Vertex AI)"
echo "=============================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo ""

# Get script directory and navigate to ml_pipeline root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ML_PIPELINE_DIR"

echo "Working directory: $(pwd)"
echo ""

# Check if the Docker image exists
echo "📋 Checking Docker image..."
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-containers/skew-audit:latest"
if gcloud artifacts docker images describe "${IMAGE_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "   ✅ Image exists: ${IMAGE_NAME}"
else
    echo "   ⚠️  Image not found. Building..."
    ./skew_job/build_docker.sh
fi
echo ""

# Trigger the Vertex AI job via orchestrator
echo "🚀 Triggering Vertex AI Skew Audit Job..."
echo ""

python3 << 'PYEOF'
import sys
import json

# Add orchestrator to path
sys.path.insert(0, '.')

from orchestrator.orchestrator import trigger_skew_job

print("=" * 60)
print("Starting Vertex AI Skew Audit Job")
print("=" * 60)
print("")

result = trigger_skew_job()

print("")
print("=" * 60)
if result['status'] == 'success':
    print("✅ SKEW AUDIT COMPLETE")
    print("=" * 60)
    print(f"Job Name: {result['job_name']}")
    if result.get('tensorboard', {}).get('resource'):
        print(f"TensorBoard: {result['tensorboard']['experiment_name']}")
else:
    print("❌ SKEW AUDIT FAILED")
    print("=" * 60)
    print(f"Error: {result.get('error', 'Unknown error')}")
    sys.exit(1)
PYEOF

echo ""
echo "=============================================="
echo "✅ SKEW AUDIT JOB TRIGGERED"
echo "=============================================="
echo ""
echo "View results:"
echo "  - TensorBoard: https://console.cloud.google.com/vertex-ai/experiments/tensorboards?project=${PROJECT_ID}"
echo "  - BigQuery: SELECT * FROM \`${PROJECT_ID}.crop_ml.skew_audit_history\` ORDER BY timestamp DESC LIMIT 10"
echo ""
