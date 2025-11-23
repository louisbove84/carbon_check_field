# Manual Backend Deployment Steps

If the automated `deploy.sh` script doesn't work, follow these manual steps:

## 1. Set Project and Region

```bash
export PROJECT_ID="ml-pipeline-477612"
export REGION="us-central1"
export SERVICE_NAME="carboncheck-field-api"

gcloud config set project $PROJECT_ID
```

## 2. Enable Required APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  earthengine.googleapis.com \
  aiplatform.googleapis.com
```

This may take 1-2 minutes.

## 3. Create Service Account (if not exists)

```bash
# Create service account
gcloud iam service-accounts create carbon-check-field \
    --display-name="CarbonCheck Field Backend" \
    --project=$PROJECT_ID

# Grant Earth Engine permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:carbon-check-field@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/earthengine.viewer"

# Grant Vertex AI permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:carbon-check-field@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

## 4. Build Docker Image

```bash
cd /Users/beuxb/Desktop/Projects/carbon_check_field/backend

# Build and push to Google Container Registry
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --timeout=15m
```

This will take 5-10 minutes (building Python dependencies).

## 5. Deploy to Cloud Run

```bash
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
```

## 6. Get Service URL

```bash
gcloud run services describe $SERVICE_NAME \
    --region $REGION \
    --format 'value(status.url)'
```

Copy this URL! You'll need it for Flutter app.

## 7. Test the Deployment

```bash
# Get the URL from previous step
export SERVICE_URL="https://carboncheck-field-api-XXXXXXXX-uc.a.run.app"

# Test health endpoint
curl $SERVICE_URL/health

# Should return:
# {
#   "status": "healthy",
#   "earth_engine": "initialized",
#   "timestamp": "2024-11-21T..."
# }
```

## Troubleshooting

### "Service account already exists"
That's fine! Skip the service account creation step.

### "Permission denied"
Make sure you're authenticated:
```bash
gcloud auth login
gcloud auth application-default login
```

### "Billing not enabled"
Enable billing in Google Cloud Console:
https://console.cloud.google.com/billing

### "Build failed"
Check the error message. Common issues:
- Missing dependencies in requirements.txt
- Syntax errors in app.py
- Dockerfile issues

View build logs:
```bash
gcloud builds list --limit=1
gcloud builds log <BUILD_ID>
```

### "Deployment failed"
Check Cloud Run logs:
```bash
gcloud run logs read $SERVICE_NAME --region $REGION --limit=50
```

## Updating the Deployment

After making changes to app.py:

```bash
# Rebuild and redeploy (one command)
gcloud run deploy $SERVICE_NAME \
    --source . \
    --region $REGION \
    --allow-unauthenticated
```

## Viewing Logs

```bash
# Real-time logs
gcloud run logs tail $SERVICE_NAME --region $REGION

# Recent logs
gcloud run logs read $SERVICE_NAME --region $REGION --limit=100
```

## Deleting the Service (if needed)

```bash
gcloud run services delete $SERVICE_NAME --region $REGION
```

---

## Next Steps After Deployment

1. ✅ Copy the service URL
2. ✅ Update Flutter app: `lib/services/backend_service.dart`
3. ✅ Test with curl
4. ✅ Move to Step 2 (Firebase setup)

