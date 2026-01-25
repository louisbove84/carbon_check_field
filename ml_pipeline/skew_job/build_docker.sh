#!/bin/bash
# Build and Push Skew Audit Container to Artifact Registry
# =========================================================
#
# This script builds the Docker container used by Vertex AI Custom Jobs
# for monthly skew/drift detection.
#
# When to rebuild:
#   ✅ Dockerfile changes (system libraries, base image, etc.)
#   ✅ requirements.txt changes (new Python packages)
#   ✅ entrypoint.sh changes
#   ✅ shared/ module changes
#   ❌ Python code changes (code is downloaded at runtime from GCS)
#
# Usage:
#   cd ml_pipeline/skew_job
#   ./build_docker.sh

set -e

PROJECT_ID="ml-pipeline-477612"
REGION="us-central1"
REPOSITORY="ml-containers"
IMAGE_NAME="skew-audit"
IMAGE_TAG="latest"

FULL_IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================================"
echo "🔨 BUILDING VERTEX AI SKEW AUDIT CONTAINER"
echo "============================================================"
echo ""
echo "Image: ${FULL_IMAGE_NAME}"
echo ""

# Get script directory and navigate to ml_pipeline root (needed for Dockerfile context)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ML_PIPELINE_DIR"

echo "Building from: $(pwd)"
echo ""

# Create Artifact Registry repository if it doesn't exist
echo "📦 Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories create ${REPOSITORY} \
  --repository-format=docker \
  --location=${REGION} \
  --project=${PROJECT_ID} \
  --description="ML training containers" 2>/dev/null || echo "   Repository already exists"

echo "✅ Repository ready"
echo ""

# Build and push image using cloudbuild.skew.yaml if it exists, otherwise use --tag
if [ -f "cloudbuild.skew.yaml" ]; then
    echo "🔨 Building with cloudbuild.skew.yaml..."
    gcloud builds submit --config cloudbuild.skew.yaml --project=${PROJECT_ID}
else
    echo "🔨 Building and pushing image..."
    gcloud builds submit \
        --tag ${FULL_IMAGE_NAME} \
        --project=${PROJECT_ID} \
        -f skew_job/Dockerfile \
        .
fi

echo ""
echo "============================================================"
echo "✅ SKEW AUDIT CONTAINER BUILT"
echo "============================================================"
echo ""
echo "Image: ${FULL_IMAGE_NAME}"
echo ""
echo "Next steps:"
echo "  1. Trigger a skew audit job via the orchestrator"
echo "  2. Or run manually: python -c 'from orchestrator.orchestrator import trigger_skew_job; trigger_skew_job()'"
echo ""
