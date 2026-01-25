#!/bin/bash
# Deploy Cloud Function to trigger Skew Audit
# ============================================

set -e

PROJECT_ID="ml-pipeline-477612"
REGION="us-central1"
FUNCTION_NAME="trigger-skew-audit"

echo "=============================================="
echo "DEPLOYING CLOUD FUNCTION: ${FUNCTION_NAME}"
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Deploy Cloud Function (Gen 2)
echo "Deploying Cloud Function..."
gcloud functions deploy ${FUNCTION_NAME} \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source=. \
    --entry-point=trigger_skew_audit \
    --trigger-http \
    --allow-unauthenticated \
    --timeout=540s \
    --memory=512MB \
    --set-env-vars="PROJECT_ID=${PROJECT_ID},REGION=${REGION},PROJECT_NUMBER=303566498201,TENSORBOARD_ID=1778818498718334976" \
    --service-account="ml-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --project=${PROJECT_ID}

echo ""
echo "=============================================="
echo "✅ CLOUD FUNCTION DEPLOYED"
echo "=============================================="

# Get the function URL
FUNCTION_URL=$(gcloud functions describe ${FUNCTION_NAME} --region=${REGION} --project=${PROJECT_ID} --gen2 --format='value(serviceConfig.uri)')
echo ""
echo "Function URL: ${FUNCTION_URL}"
echo ""
echo "To test manually:"
echo "  curl ${FUNCTION_URL}"
echo ""
