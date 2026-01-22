#!/bin/bash
# Setup Skew Audit Infrastructure
# ================================
# Creates TensorBoard instance and Cloud Scheduler job for monthly skew audits.

set -e

PROJECT_ID="ml-pipeline-477612"
PROJECT_NUMBER="303566498201"
REGION="us-central1"
SERVICE_NAME="ml-skew-audit"
TENSORBOARD_DISPLAY_NAME="skew-audit-tensorboard"
SCHEDULER_JOB_NAME="monthly-skew-audit"
SERVICE_ACCOUNT="ml-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=============================================="
echo "🔧 SETTING UP SKEW AUDIT INFRASTRUCTURE"
echo "=============================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo ""

# Step 1: Create TensorBoard instance for skew audits
echo "📊 Step 1: Creating TensorBoard instance..."
echo "-------------------------------------------"

# Check if TensorBoard already exists
EXISTING_TB=$(gcloud ai tensorboards list \
    --region="${REGION}" \
    --filter="displayName=${TENSORBOARD_DISPLAY_NAME}" \
    --format="value(name)" 2>/dev/null || echo "")

if [ -n "$EXISTING_TB" ]; then
    TENSORBOARD_ID=$(echo "$EXISTING_TB" | grep -oE '[0-9]+$')
    echo "   ✅ TensorBoard already exists: ${TENSORBOARD_ID}"
else
    echo "   Creating new TensorBoard instance..."
    gcloud ai tensorboards create \
        --display-name="${TENSORBOARD_DISPLAY_NAME}" \
        --region="${REGION}" \
        --project="${PROJECT_ID}"
    
    # Get the TensorBoard ID
    TENSORBOARD_ID=$(gcloud ai tensorboards list \
        --region="${REGION}" \
        --filter="displayName=${TENSORBOARD_DISPLAY_NAME}" \
        --format="value(name)" | grep -oE '[0-9]+$')
    
    echo "   ✅ Created TensorBoard: ${TENSORBOARD_ID}"
fi

TENSORBOARD_RESOURCE="projects/${PROJECT_NUMBER}/locations/${REGION}/tensorboards/${TENSORBOARD_ID}"
echo "   TensorBoard Resource: ${TENSORBOARD_RESOURCE}"
echo ""

# Step 2: Update Cloud Run service with TensorBoard ID
echo "🚀 Step 2: Updating Cloud Run service..."
echo "-------------------------------------------"

# Check if service exists
SERVICE_EXISTS=$(gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" \
    --format="value(status.url)" 2>/dev/null || echo "")

if [ -z "$SERVICE_EXISTS" ]; then
    echo "   ⚠️  Service ${SERVICE_NAME} not deployed yet."
    echo "   Run deploy.sh first, then re-run this script."
else
    echo "   Updating environment variable..."
    gcloud run services update "${SERVICE_NAME}" \
        --region="${REGION}" \
        --set-env-vars "SKEW_TENSORBOARD_ID=${TENSORBOARD_ID}"
    echo "   ✅ Updated service with SKEW_TENSORBOARD_ID=${TENSORBOARD_ID}"
fi
echo ""

# Step 3: Create Cloud Scheduler job
echo "⏰ Step 3: Creating Cloud Scheduler job..."
echo "-------------------------------------------"

# Get service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" \
    --format='value(status.url)' 2>/dev/null || echo "")

if [ -z "$SERVICE_URL" ]; then
    echo "   ⚠️  Service URL not available. Deploy service first."
    echo "   Skipping scheduler creation."
else
    # Check if scheduler job exists
    EXISTING_JOB=$(gcloud scheduler jobs describe "${SCHEDULER_JOB_NAME}" \
        --location="${REGION}" 2>/dev/null && echo "exists" || echo "")
    
    if [ -n "$EXISTING_JOB" ]; then
        echo "   Updating existing scheduler job..."
        gcloud scheduler jobs update http "${SCHEDULER_JOB_NAME}" \
            --location="${REGION}" \
            --schedule="0 3 1 * *" \
            --time-zone="America/Chicago" \
            --uri="${SERVICE_URL}" \
            --http-method=POST \
            --oidc-service-account-email="${SERVICE_ACCOUNT}" \
            --oidc-token-audience="${SERVICE_URL}" \
            --description="Monthly data skew audit - runs on 1st of each month at 3 AM CT"
        echo "   ✅ Updated scheduler job"
    else
        echo "   Creating new scheduler job..."
        gcloud scheduler jobs create http "${SCHEDULER_JOB_NAME}" \
            --location="${REGION}" \
            --schedule="0 3 1 * *" \
            --time-zone="America/Chicago" \
            --uri="${SERVICE_URL}" \
            --http-method=POST \
            --oidc-service-account-email="${SERVICE_ACCOUNT}" \
            --oidc-token-audience="${SERVICE_URL}" \
            --description="Monthly data skew audit - runs on 1st of each month at 3 AM CT"
        echo "   ✅ Created scheduler job"
    fi
    
    echo ""
    echo "   Schedule: 0 3 1 * * (1st of month at 3 AM CT)"
    echo "   Target: ${SERVICE_URL}"
fi
echo ""

# Summary
echo "=============================================="
echo "✅ INFRASTRUCTURE SETUP COMPLETE"
echo "=============================================="
echo ""
echo "TensorBoard Instance:"
echo "  ID: ${TENSORBOARD_ID}"
echo "  Resource: ${TENSORBOARD_RESOURCE}"
echo "  Console: https://console.cloud.google.com/vertex-ai/experiments/tensorboards/${TENSORBOARD_ID}?project=${PROJECT_ID}"
echo ""
echo "Cloud Scheduler Job:"
echo "  Name: ${SCHEDULER_JOB_NAME}"
echo "  Schedule: Monthly (1st at 3 AM CT)"
echo "  Console: https://console.cloud.google.com/cloudscheduler?project=${PROJECT_ID}"
echo ""
echo "To trigger manually:"
echo "  gcloud scheduler jobs run ${SCHEDULER_JOB_NAME} --location=${REGION}"
echo ""
echo "Or via curl:"
echo "  curl -X POST ${SERVICE_URL} -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\""
