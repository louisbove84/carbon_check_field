#!/bin/bash
# Deploy Skew Audit Service to Cloud Run
# =======================================

set -e

PROJECT_ID="ml-pipeline-477612"
REGION="us-central1"
SERVICE_NAME="ml-skew-audit"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-containers/skew-audit:latest"

echo "=============================================="
echo "🔍 DEPLOYING SKEW AUDIT SERVICE"
echo "=============================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo "Image: ${IMAGE_NAME}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ML_PIPELINE_DIR"

echo "📦 Building Docker image..."
gcloud builds submit \
    --tag "${IMAGE_NAME}" \
    --config cloudbuild.skew.yaml \
    .

echo ""
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --region "${REGION}" \
    --platform managed \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 1 \
    --service-account "ml-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --set-env-vars "SKEW_TENSORBOARD_ID=${SKEW_TENSORBOARD_ID:-}" \
    --no-allow-unauthenticated

echo ""
echo "=============================================="
echo "✅ DEPLOYMENT COMPLETE"
echo "=============================================="

# Get service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --format='value(status.url)')
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "To trigger manually:"
echo "  curl -X POST ${SERVICE_URL} -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\""
echo ""
echo "Next steps:"
echo "  1. Create TensorBoard instance for skew audits"
echo "  2. Set SKEW_TENSORBOARD_ID environment variable"
echo "  3. Create Cloud Scheduler job for monthly execution"
