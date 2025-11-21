# CarbonCheck Field - Secure Backend

FastAPI backend for crop classification and carbon credit estimation.

## Features

✅ **100% Secure** - No service account keys stored anywhere  
✅ **Application Default Credentials** - Managed by Google Cloud  
✅ **Firebase Authentication** - Token verification on all requests  
✅ **Earth Engine Integration** - Sentinel-2 NDVI computation  
✅ **Vertex AI Integration** - Crop type prediction  
✅ **Auto-scaling** - Scales to zero when not in use  

---

## Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud SDK (`gcloud`)
- Service account with permissions:
  - Earth Engine API access
  - Vertex AI User
  - Cloud Run Admin

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Cloud
gcloud auth application-default login

# Run server
python app.py
```

Server runs on http://localhost:8080

### Test Endpoints

```bash
# Health check
curl http://localhost:8080/health

# Analyze field (requires Firebase token)
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_FIREBASE_TOKEN" \
  -d '{
    "polygon": [
      {"lat": 41.0, "lng": -93.0},
      {"lat": 41.01, "lng": -93.0},
      {"lat": 41.01, "lng": -93.01}
    ]
  }'
```

---

## Deploy to Cloud Run

```bash
# Make deploy script executable
chmod +x deploy.sh

# Deploy (builds Docker image and deploys to Cloud Run)
./deploy.sh
```

After deployment, copy the service URL and update Flutter app:
- File: `lib/services/backend_service.dart`
- Variable: `backendUrl`

---

## API Documentation

### POST /analyze

Analyze a farm field and return crop prediction + CO₂ income estimates.

**Headers:**
- `Authorization: Bearer <firebase-id-token>` (required)
- `Content-Type: application/json`

**Request Body:**
```json
{
  "polygon": [
    {"lat": 41.0, "lng": -93.0},
    {"lat": 41.01, "lng": -93.0},
    {"lat": 41.01, "lng": -93.01},
    {"lat": 41.0, "lng": -93.01}
  ],
  "year": 2024  // optional, defaults to current year
}
```

**Response:**
```json
{
  "crop": "Corn",
  "confidence": 0.98,
  "area_acres": 47.2,
  "co2_income_min": 566.4,
  "co2_income_max": 849.6,
  "co2_income_avg": 708.0,
  "features": [0.75, 0.12, ...],  // 17 NDVI features
  "timestamp": "2024-11-21T10:30:00"
}
```

**Error Responses:**
- `401 Unauthorized` - Invalid or missing Firebase token
- `400 Bad Request` - Invalid polygon (too few/many points, too small/large)
- `500 Internal Server Error` - Earth Engine or Vertex AI error

---

## Architecture

```
Client (Flutter App)
  └─> Firebase Auth (anonymous)
  └─> POST /analyze with token
      │
      ▼
Cloud Run Backend
  ├─> Verify Firebase token
  ├─> Calculate polygon area
  ├─> Call Earth Engine
  │   └─> Compute 17 NDVI features
  ├─> Call Vertex AI
  │   └─> Predict crop type
  ├─> Calculate CO₂ income
  └─> Return JSON response
```

---

## Environment Variables

Set in Cloud Run deployment:

- `PORT` - Server port (default: 8080)
- `GCP_PROJECT` - Google Cloud project ID

Application Default Credentials are automatically configured by Cloud Run.

---

## Monitoring

### View Logs

```bash
# Real-time logs
gcloud run logs tail carboncheck-field-api --region us-central1

# Recent logs
gcloud run logs read carboncheck-field-api --region us-central1 --limit 50
```

### Metrics

- Cloud Run Console → carboncheck-field-api → Metrics
- Monitor: Request count, latency, errors, memory usage

---

## Troubleshooting

### "Earth Engine not initialized"

```bash
# Ensure service account has Earth Engine access
gcloud projects add-iam-policy-binding ml-pipeline-477612 \
  --member="serviceAccount:carbon-check-field@ml-pipeline-477612.iam.gserviceaccount.com" \
  --role="roles/earthengine.viewer"
```

### "Vertex AI prediction failed"

```bash
# Verify endpoint exists and is deployed
gcloud ai endpoints list --region=us-central1 --project=ml-pipeline-477612
```

### "Out of memory"

Increase Cloud Run memory:
```bash
gcloud run services update carboncheck-field-api \
  --region us-central1 \
  --memory 4Gi
```

---

## Cost Estimation

### Per 1,000 field analyses:

- Cloud Run: $0.40 (after free tier)
- Earth Engine: $0 (free tier)
- Vertex AI: $1-2 (depends on model)

**Total: ~$1.40-$2.40 per 1,000 analyses**

Free tier covers ~2 million requests/month!

---

## Security

✅ **Application Default Credentials** - No keys in code  
✅ **Firebase token verification** - All requests authenticated  
✅ **HTTPS only** - Enforced by Cloud Run  
✅ **CORS configured** - Restrict to your app domain  
✅ **Auto-scaling** - DDoS protection  

---

## Development

### Project Structure

```
backend/
├── app.py              # FastAPI application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── deploy.sh           # Deployment script
└── README.md           # This file
```

### Adding Features

Edit `app.py` and redeploy:

```bash
# Test locally first
python app.py

# Deploy
./deploy.sh
```

---

## License

MIT License - See LICENSE file for details

---

**Questions?** Check SECURITY_ARCHITECTURE.md or open an issue!

