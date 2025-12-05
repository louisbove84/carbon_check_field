# Automated Earth Engine Data Collection

## ✅ Implementation Complete!

The Earth Engine data collection is now **fully automated** and integrated into the GCP pipeline. No more manual steps!

## How It Works

1. **Pipeline triggers data collection** via `export_earth_engine_data()`
2. **Earth Engine Python API** collects samples from CDL-verified crop fields
3. **Direct export to BigQuery** - no intermediate steps
4. **Automatic retraining** after data collection completes

## Configuration

Edit `ml_pipeline/config.yaml`:

```yaml
data_collection:
  num_fields_per_crop: 30      # Number of fields per crop
  num_samples_per_field: 3     # Samples per field
  cdl_year: 2024                # CDL year for verification
  crops:
    - name: Corn
      code: 1
      counties: ["17113", "17019", ...]
    # ... more crops
```

**Total samples:** `num_fields_per_crop × num_samples_per_field × number_of_crops`
- Current: 30 × 3 × 4 = **360 samples**

## Usage

### Manual Trigger

```bash
curl -X POST https://ml-pipeline-303566498201.us-central1.run.app
```

This will:
1. ✅ Collect fresh data from Earth Engine
2. ✅ Export directly to BigQuery
3. ✅ Wait for export to complete
4. ✅ Train new model
5. ✅ Deploy if quality gates pass

### Automated Schedule

The pipeline can be scheduled via Cloud Scheduler to run monthly:

```bash
gcloud scheduler jobs create http ml-pipeline-monthly \
  --location=us-central1 \
  --schedule="0 0 1 * *" \
  --uri="https://ml-pipeline-303566498201.us-central1.run.app" \
  --http-method=POST \
  --project=ml-pipeline-477612
```

## What Changed

### Before (Manual)
1. Open Earth Engine Code Editor
2. Copy/paste JavaScript script
3. Run script
4. Go to Tasks tab
5. Click "RUN" on export task
6. Wait for completion
7. Manually trigger retraining

### After (Automated)
1. **Just trigger the pipeline** - everything else is automatic!

## Implementation Details

- **New module:** `ml_pipeline/orchestrator/earth_engine_collector.py`
  - Converts JavaScript Earth Engine script to Python
  - Handles CDL verification
  - Exports directly to BigQuery
  
- **Updated:** `orchestrator.py`
  - `export_earth_engine_data()` now fully implemented
  - Waits for export completion before proceeding

## Monitoring

Check export status:
```bash
# View Earth Engine tasks
earthengine task list

# Check BigQuery data
bq query --use_legacy_sql=false \
  "SELECT crop, COUNT(*) FROM \`ml-pipeline-477612.crop_ml.training_features\` GROUP BY crop"
```

## Troubleshooting

**If export fails:**
- Check Earth Engine authentication: `earthengine authenticate`
- Verify BigQuery table exists and has write permissions
- Check Cloud Run logs: `gcloud run services logs read ml-pipeline --region=us-central1`

**If timeout:**
- Increase `timeout_minutes` in `wait_for_export()` (default: 30 min)
- Reduce `num_fields_per_crop` if collection takes too long

## Next Steps

1. **Test the automated collection:**
   ```bash
   curl -X POST https://ml-pipeline-303566498201.us-central1.run.app
   ```

2. **Monitor the pipeline:**
   - Check Cloud Run logs
   - Verify BigQuery data
   - Check training metrics

3. **Set up monthly schedule** (optional):
   - Use Cloud Scheduler as shown above

