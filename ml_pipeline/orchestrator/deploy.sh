#!/bin/bash
# Deploy Orchestrator to Cloud Run
# =================================

set -e

PROJECT_ID="ml-pipeline-477612"
REGION="us-central1"
SERVICE_NAME="ml-pipeline"
IMAGE_NAME="gcr.io/${PROJECT_ID}/ml-orchestrator"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================"
echo "🚀 DEPLOYING ML ORCHESTRATOR TO CLOUD RUN"
echo "============================================================"
echo ""

# Upload config to Cloud Storage
echo "📤 Uploading config..."
gsutil cp "${SCRIPT_DIR}/config.yaml" gs://carboncheck-data/config/config.yaml
echo "✅ Config uploaded"
echo ""

# Build Docker image (use repo root as context)
echo "🔨 Building orchestrator image..."
gcloud builds submit \
  --config "${ROOT_DIR}/cloudbuild.orchestrator.yaml" \
  "${ROOT_DIR}" \
  --project="${PROJECT_ID}" \
  --quiet
echo "✅ Image built"
echo ""

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image=${IMAGE_NAME} \
  --platform=managed \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --allow-unauthenticated \
  --memory=2Gi \
  --timeout=3600s \
  --max-instances=1 \
  --service-account=ml-pipeline-sa@ml-pipeline-477612.iam.gserviceaccount.com \
  --quiet

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
  --platform=managed \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --format='value(status.url)')

echo ""
echo "============================================================"
echo "✅ ORCHESTRATOR DEPLOYED"
echo "============================================================"
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test: curl -X POST ${SERVICE_URL}"
echo ""

