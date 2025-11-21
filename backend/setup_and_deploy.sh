#!/bin/bash

# CarbonCheck Field - Complete Setup and Deployment
# This script will guide you through authentication and deployment

set -e

echo "🚀 CarbonCheck Field - Complete Setup"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ID="ml-pipeline-477612"

echo -e "${BLUE}Step 1: Authenticate with Google Cloud${NC}"
echo "This will open your browser to sign in..."
echo ""

# Check if already authenticated
if gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q '@'; then
    ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    echo -e "${GREEN}✅ Already authenticated as: $ACCOUNT${NC}"
else
    gcloud auth login
    echo -e "${GREEN}✅ Authentication complete!${NC}"
fi

echo ""
echo -e "${BLUE}Step 2: Set project${NC}"
gcloud config set project $PROJECT_ID
echo -e "${GREEN}✅ Project set to: $PROJECT_ID${NC}"

echo ""
echo -e "${BLUE}Step 3: Set application default credentials${NC}"
echo "This allows the backend to access Google Cloud services..."

# Check if already set
if [ -f ~/.config/gcloud/application_default_credentials.json ]; then
    echo -e "${GREEN}✅ Application default credentials already set${NC}"
else
    gcloud auth application-default login
    echo -e "${GREEN}✅ Application default credentials configured!${NC}"
fi

echo ""
echo -e "${BLUE}Step 4: Enable required APIs${NC}"
echo "Enabling Cloud Build, Cloud Run, Earth Engine, and Vertex AI..."

gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    earthengine.googleapis.com \
    aiplatform.googleapis.com \
    --quiet

echo -e "${GREEN}✅ APIs enabled!${NC}"

echo ""
echo -e "${BLUE}Step 5: Create service account (if needed)${NC}"

# Check if service account exists
if gcloud iam service-accounts describe carbon-check-field@${PROJECT_ID}.iam.gserviceaccount.com &>/dev/null; then
    echo -e "${GREEN}✅ Service account already exists${NC}"
else
    echo "Creating service account..."
    gcloud iam service-accounts create carbon-check-field \
        --display-name="CarbonCheck Field Backend" \
        --project=$PROJECT_ID
    
    # Grant permissions
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:carbon-check-field@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/earthengine.viewer" \
        --quiet
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:carbon-check-field@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/aiplatform.user" \
        --quiet
    
    echo -e "${GREEN}✅ Service account created and configured!${NC}"
fi

echo ""
echo -e "${BLUE}Step 6: Build and deploy to Cloud Run${NC}"
echo "This will take 5-10 minutes (building Docker image)..."
echo ""

SERVICE_NAME="carboncheck-field-api"
REGION="us-central1"

# Build Docker image
echo "Building Docker image..."
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --timeout=15m

echo ""
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 10 \
    --service-account "carbon-check-field@${PROJECT_ID}.iam.gserviceaccount.com"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region $REGION \
    --format 'value(status.url)')

echo ""
echo -e "${GREEN}======================================"
echo "✅ DEPLOYMENT SUCCESSFUL!"
echo "======================================${NC}"
echo ""
echo -e "${YELLOW}🌐 Your Backend URL:${NC}"
echo -e "${GREEN}${SERVICE_URL}${NC}"
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo ""
echo "1. Test the API:"
echo -e "   ${BLUE}curl ${SERVICE_URL}/health${NC}"
echo ""
echo "2. Update Flutter app:"
echo "   File: lib/services/backend_service.dart"
echo "   Change this line:"
echo -e "   ${BLUE}static const String backendUrl = '${SERVICE_URL}';${NC}"
echo ""
echo "3. Setup Firebase (see MIGRATION_COMPLETE.md Step 2)"
echo ""
echo -e "${GREEN}🎉 Your secure backend is now live!${NC}"
echo ""

