#!/bin/bash

# Deploy Automated Model Retraining Cloud Function
# =================================================

set -e

PROJECT_ID="ml-pipeline-477612"
REGION="us-central1"
FUNCTION_NAME="retrain-crop-model"
ENTRY_POINT="retrain_model"

echo "🚀 Deploying retraining Cloud Function..."
echo "   Project: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Function: $FUNCTION_NAME"

gcloud functions deploy $FUNCTION_NAME \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=$ENTRY_POINT \
  --trigger-http \
  --allow-unauthenticated \
  --timeout=3600s \
  --memory=2Gi \
  --max-instances=1 \
  --set-env-vars PROJECT_ID=$PROJECT_ID \
  --service-account=ml-pipeline-sa@ml-pipeline-477612.iam.gserviceaccount.com

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Test the function:"
echo "   curl https://$REGION-$PROJECT_ID.cloudfunctions.net/$FUNCTION_NAME"
echo ""
echo "2. Schedule monthly retraining:"
echo "   gcloud scheduler jobs create http $FUNCTION_NAME-monthly \\"
echo "     --schedule='0 0 1 * *' \\"
echo "     --uri=https://$REGION-$PROJECT_ID.cloudfunctions.net/$FUNCTION_NAME \\"
echo "     --http-method=POST \\"
echo "     --location=$REGION"
echo ""
echo "3. Monitor logs:"
echo "   gcloud functions logs read $FUNCTION_NAME --region=$REGION --limit=50"

