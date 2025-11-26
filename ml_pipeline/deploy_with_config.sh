#!/bin/bash
# Deploy Cloud Functions with Centralized Configuration
# ======================================================
# This script deploys Cloud Functions and sets environment variables
# from a centralized configuration.

set -e

# ============================================================
# CONFIGURATION (Change these values as needed)
# ============================================================

PROJECT_ID="ml-pipeline-477612"
REGION="us-central1"
BUCKET_NAME="carboncheck-data"
DATASET_ID="crop_ml"

# Model Configuration
MODEL_NAME="crop-classifier-latest"
ENDPOINT_NAME="crop-endpoint"

# Quality Gates (can be adjusted without code changes!)
ABSOLUTE_MIN_ACCURACY="0.75"      # 75% minimum
MIN_PER_CROP_F1="0.70"            # 70% F1 per crop
IMPROVEMENT_MARGIN="0.02"          # 2% improvement required

# Data Collection
SAMPLES_PER_CROP="100"             # Samples per crop per month

# Service Account
SERVICE_ACCOUNT="ml-pipeline-sa@ml-pipeline-477612.iam.gserviceaccount.com"

# ============================================================
# BUILD ENVIRONMENT VARIABLES STRING
# ============================================================

ENV_VARS="PROJECT_ID=${PROJECT_ID},"
ENV_VARS+="REGION=${REGION},"
ENV_VARS+="BUCKET_NAME=${BUCKET_NAME},"
ENV_VARS+="DATASET_ID=${DATASET_ID},"
ENV_VARS+="MODEL_NAME=${MODEL_NAME},"
ENV_VARS+="ENDPOINT_NAME=${ENDPOINT_NAME},"
ENV_VARS+="ABSOLUTE_MIN_ACCURACY=${ABSOLUTE_MIN_ACCURACY},"
ENV_VARS+="MIN_PER_CROP_F1=${MIN_PER_CROP_F1},"
ENV_VARS+="IMPROVEMENT_MARGIN=${IMPROVEMENT_MARGIN},"
ENV_VARS+="SAMPLES_PER_CROP=${SAMPLES_PER_CROP}"

echo "============================================================"
echo "DEPLOYING CLOUD FUNCTIONS WITH CONFIGURATION"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Project ID:        $PROJECT_ID"
echo "  Region:            $REGION"
echo "  Bucket:            $BUCKET_NAME"
echo "  Dataset:           $DATASET_ID"
echo ""
echo "Quality Gates:"
echo "  Min Accuracy:      ${ABSOLUTE_MIN_ACCURACY} (${ABSOLUTE_MIN_ACCURACY}%)"
echo "  Min Crop F1:       ${MIN_PER_CROP_F1} (${MIN_PER_CROP_F1}%)"
echo "  Improvement:       ${IMPROVEMENT_MARGIN} (${IMPROVEMENT_MARGIN}%)"
echo ""
echo "Data Collection:"
echo "  Samples/Crop:      $SAMPLES_PER_CROP"
echo ""

# ============================================================
# DEPLOY DATA COLLECTION FUNCTION
# ============================================================

echo "============================================================"
echo "📥 Deploying: monthly-data-collection"
echo "============================================================"

gcloud functions deploy monthly-data-collection \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=collect_training_data \
  --trigger-http \
  --allow-unauthenticated \
  --timeout=3600s \
  --memory=4Gi \
  --max-instances=1 \
  --set-env-vars "$ENV_VARS" \
  --service-account=$SERVICE_ACCOUNT

echo ""
echo "✅ monthly-data-collection deployed"
echo ""

# ============================================================
# DEPLOY MODEL RETRAINING FUNCTION
# ============================================================

echo "============================================================"
echo "🤖 Deploying: auto-retrain-model"
echo "============================================================"

gcloud functions deploy auto-retrain-model \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=retrain_model \
  --trigger-http \
  --allow-unauthenticated \
  --timeout=3600s \
  --memory=4Gi \
  --max-instances=1 \
  --set-env-vars "$ENV_VARS" \
  --service-account=$SERVICE_ACCOUNT

echo ""
echo "✅ auto-retrain-model deployed"
echo ""

# ============================================================
# DEPLOYMENT COMPLETE
# ============================================================

echo "============================================================"
echo "✅ DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Test manually:"
echo "   ./test_pipeline.sh retrain"
echo ""
echo "2. View function URLs:"
echo "   gcloud functions list --region=$REGION"
echo ""
echo "3. View environment variables:"
echo "   gcloud functions describe auto-retrain-model --region=$REGION --gen2 --format='value(serviceConfig.environmentVariables)'"
echo ""
echo "4. Update config without redeploying code:"
echo "   gcloud functions deploy auto-retrain-model \\"
echo "     --region=$REGION \\"
echo "     --update-env-vars ABSOLUTE_MIN_ACCURACY=0.80 \\"
echo "     --gen2"
echo ""
echo "5. Setup Cloud Scheduler:"
echo "   See QUICK_START.md Step 4"
echo ""

