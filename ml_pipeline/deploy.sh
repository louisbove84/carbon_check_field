#!/bin/bash
# Deploy ML Pipeline to Cloud Run
# ================================
# Single script to deploy entire pipeline

set -e

PROJECT_ID="ml-pipeline-477612"
REGION="us-central1"
SERVICE_NAME="ml-pipeline"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "============================================================"
echo "🚀 DEPLOYING ML PIPELINE TO CLOUD RUN"
echo "============================================================"
echo ""
echo "Project ID:     $PROJECT_ID"
echo "Region:         $REGION"
echo "Service:        $SERVICE_NAME"
echo "Image:          $IMAGE_NAME"
echo ""

# Step 1: Upload config to Cloud Storage
echo "============================================================"
echo "📤 Step 1: Uploading configuration"
echo "============================================================"
if [ -f "config.yaml" ]; then
    ./upload_config.sh
else
    echo "⚠️  config.yaml not found, skipping upload"
fi
echo ""

# Step 2: Build Docker image
echo "============================================================"
echo "🔨 Step 2: Building Docker image"
echo "============================================================"
gcloud builds submit --tag ${IMAGE_NAME} --project=${PROJECT_ID}
echo "✅ Image built: ${IMAGE_NAME}"
echo ""

# Step 3: Deploy to Cloud Run
echo "============================================================"
echo "🚀 Step 3: Deploying to Cloud Run"
echo "============================================================"
gcloud run deploy ${SERVICE_NAME} \
  --image=${IMAGE_NAME} \
  --platform=managed \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --allow-unauthenticated \
  --memory=4Gi \
  --timeout=3600s \
  --max-instances=1 \
  --service-account=ml-pipeline-sa@ml-pipeline-477612.iam.gserviceaccount.com

echo ""
echo "✅ Deployment complete!"
echo ""

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
  --platform=managed \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --format='value(status.url)')

echo "============================================================"
echo "✅ DEPLOYMENT SUCCESSFUL"
echo "============================================================"
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Test the pipeline:"
echo "   curl -X POST ${SERVICE_URL}"
echo ""
echo "2. Run specific step:"
echo "   curl -X POST ${SERVICE_URL}?step=1  # Data collection"
echo "   curl -X POST ${SERVICE_URL}?step=2  # Retraining"
echo "   curl -X POST ${SERVICE_URL}?step=3  # Evaluation"
echo ""
echo "3. Setup Cloud Scheduler for monthly runs:"
echo "   gcloud scheduler jobs create http ${SERVICE_NAME}-monthly \\"
echo "     --location=${REGION} \\"
echo "     --schedule='0 0 5 * *' \\"
echo "     --uri=${SERVICE_URL} \\"
echo "     --http-method=POST \\"
echo "     --time-zone='America/Chicago'"
echo ""
echo "4. View logs:"
echo "   gcloud run logs read ${SERVICE_NAME} --region=${REGION}"
echo ""

