#!/bin/bash

# CarbonCheck Field - Cloud Run Deployment Script
# This script builds and deploys the secure backend to Google Cloud Run

set -e

# Configuration
PROJECT_ID="ml-pipeline-477612"
REGION="us-central1"
SERVICE_NAME="carboncheck-field-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 CarbonCheck Field - Cloud Run Deployment"
echo "==========================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found. Please install it first."
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Authenticate (if needed)
echo "📋 Checking authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "🔐 Please authenticate with Google Cloud:"
    gcloud auth login
fi

# Set project
echo "📁 Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    earthengine.googleapis.com \
    aiplatform.googleapis.com \
    --quiet

# Build Docker image
echo "🏗️  Building Docker image..."
gcloud builds submit \
    --tag ${IMAGE_NAME} \
    --timeout=15m \
    .

# Deploy to Cloud Run
echo "🚢 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars "GCP_PROJECT=${PROJECT_ID}" \
    --service-account "carbon-check-field@${PROJECT_ID}.iam.gserviceaccount.com"

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region ${REGION} \
    --format 'value(status.url)')

echo ""
echo "✅ Deployment successful!"
echo "==========================================="
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "📝 Next steps:"
echo "   1. Test the API: curl ${SERVICE_URL}/health"
echo "   2. Update Flutter app with this URL"
echo "   3. Configure Firebase Auth in Flutter"
echo ""
echo "🔒 Security notes:"
echo "   - Service uses Application Default Credentials"
echo "   - No service account keys in the app!"
echo "   - Firebase Auth required for all requests"
echo ""

