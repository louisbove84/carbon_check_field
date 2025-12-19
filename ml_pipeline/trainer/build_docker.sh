#!/bin/bash
# Build and Push Trainer Container to Artifact Registry
# ======================================================
#
# This script builds the Docker container used by Vertex AI Custom Training Jobs.
#
# When to rebuild:
#   ✅ Dockerfile changes (system libraries, base image, etc.)
#   ✅ requirements.txt changes (new Python packages)
#   ✅ entrypoint.sh changes
#   ❌ Python code changes (code is downloaded at runtime from GCS)
#
# Usage:
#   cd ml_pipeline/trainer
#   ./build_docker.sh

set -e

PROJECT_ID="ml-pipeline-477612"
REGION="us-central1"
REPOSITORY="ml-containers"
IMAGE_NAME="crop-trainer"
IMAGE_TAG="latest"

FULL_IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================================"
echo "🔨 BUILDING VERTEX AI TRAINER CONTAINER"
echo "============================================================"
echo ""
echo "Image: ${FULL_IMAGE_NAME}"
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

# Build and push image
echo "🔨 Building and pushing image..."
gcloud builds submit --tag ${FULL_IMAGE_NAME} --project=${PROJECT_ID}

echo ""
echo "============================================================"
echo "✅ TRAINER CONTAINER BUILT"
echo "============================================================"
echo ""
echo "Image: ${FULL_IMAGE_NAME}"
echo ""
echo "Next: Deploy orchestrator with ./orchestrator/deploy.sh"
echo ""

