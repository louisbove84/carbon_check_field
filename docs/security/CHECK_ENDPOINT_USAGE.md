# How to Check Which Endpoint is Being Used

## Quick Answer

The Flutter app uses the endpoint configured in:
**`backend/app.py`** → `VERTEX_AI_ENDPOINT` variable

## Current Configuration

**File:** `carbon_check_field/backend/app.py` (line 56-59)

```python
VERTEX_AI_ENDPOINT = (
    "projects/ml-pipeline-477612/locations/us-central1/"
    "endpoints/2450616804754587648"  # crop-endpoint (active)
)
```

**Current Active Endpoint:**
- **Endpoint ID:** `2450616804754587648`
- **Display Name:** `crop-endpoint`
- **Status:** ✅ Active and working

## How to Verify

### 1. Check Backend Code

```bash
grep -n "VERTEX_AI_ENDPOINT" backend/app.py
```

### 2. Check Active Endpoint in GCP

```bash
gcloud ai endpoints list --region=us-central1 \
  --filter="displayName=crop-endpoint" \
  --format="table(displayName,name)"
```

### 3. Test the Endpoint

```bash
cd ml_pipeline
python3 test_endpoint.py
```

## Endpoint Format

The endpoint is specified as a full resource name:
```
projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}
```

**Example:**
```
projects/ml-pipeline-477612/locations/us-central1/endpoints/2450616804754587648
```

## How the App Uses It

1. **Backend API** (`backend/app.py`):
   - Reads `VERTEX_AI_ENDPOINT` constant
   - Uses it in `predict_crop_type()` function
   - Calls Vertex AI endpoint for predictions

2. **Flow:**
   ```
   Flutter App → Backend API → Vertex AI Endpoint → Model Prediction
   ```

## If Endpoint Changes

When you retrain and deploy a new model:

1. **Check new endpoint ID:**
   ```bash
   gcloud ai endpoints list --region=us-central1 \
     --filter="displayName=crop-endpoint"
   ```

2. **Update backend/app.py:**
   ```python
   VERTEX_AI_ENDPOINT = (
       "projects/ml-pipeline-477612/locations/us-central1/"
       "endpoints/{NEW_ENDPOINT_ID}"
   )
   ```

3. **Redeploy backend** (if using Cloud Run/App Engine)

## Verification Checklist

- [ ] Backend `app.py` has correct endpoint ID
- [ ] Endpoint exists in GCP (check with `gcloud ai endpoints describe`)
- [ ] Endpoint has deployed model (check `deployedModels`)
- [ ] Model is receiving traffic (check `trafficSplit`)
- [ ] Test endpoint works (`python3 ml_pipeline/test_endpoint.py`)

## Troubleshooting

**Error: "Endpoint not found"**
- Check endpoint ID in `backend/app.py`
- Verify endpoint exists: `gcloud ai endpoints describe <ID>`

**Error: "No predictions returned"**
- Check if model is deployed: `gcloud ai endpoints describe <ID> --format="get(deployedModels)"`
- Check traffic split: `gcloud ai endpoints describe <ID> --format="get(trafficSplit)"`

---

**Last Updated:** Endpoint ID `2450616804754587648` (crop-endpoint) - Active ✅

