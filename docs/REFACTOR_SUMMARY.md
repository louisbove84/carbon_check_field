# Earth Engine Feature Extraction Refactor

## Problem
Duplicate Earth Engine NDVI computation logic existed in two places:
1. `ml_pipeline/orchestrator/earth_engine_collector.py` (training data collection)
2. `backend/app.py` (real-time inference/prediction)

## Solution
Created a **single source of truth** for NDVI feature extraction:

### New Shared Module
`ml_pipeline/shared/earth_engine_features.py`

Contains three functions:

1. **`compute_ndvi_features_ee(geometry, year)`**
   - Core logic for computing 12 raw NDVI features from Earth Engine
   - Returns Earth Engine dictionary (not evaluated)
   - Used internally by other functions

2. **`compute_ndvi_features_ee_as_feature(point, year)`**
   - Wraps core logic, returns `ee.Feature` for batch processing
   - Used by: `earth_engine_collector.py` (training data collection)

3. **`compute_ndvi_features_sync(geometry, year)`**
   - Wraps core logic, calls `.getInfo()` to get Python values
   - Used by: `backend/app.py` (real-time inference)

### Features Extracted (12 raw features)
- `ndvi_mean`, `ndvi_std`, `ndvi_min`, `ndvi_max`
- `ndvi_p25`, `ndvi_p50`, `ndvi_p75`
- `ndvi_early`, `ndvi_late`
- `elevation_m`
- `longitude`, `latitude`

### Architecture Decision
**Split computation between collection and training:**

**At Collection Time** (Earth Engine):
- Compute raw features (NDVI stats, elevation, coordinates)
- Store in BigQuery
- ✅ Stable, unlikely to change
- ✅ Reduces storage costs (features vs. full imagery)

**At Training/Inference Time** (Python):
- Compute derived features (ranges, ratios, encodings, binnings)
- Apply using shared `feature_engineering.py`
- ✅ Fast experimentation (no need to re-collect from Earth Engine)
- ✅ Allows data-dependent features (e.g., quantile-based elevation binning)

## Benefits
1. **No Duplication**: Single source of truth for Earth Engine NDVI logic
2. **Consistency**: Training and inference use identical Earth Engine features
3. **Maintainability**: Bug fixes/improvements in one place
4. **Flexibility**: Can still experiment with derived features without re-collection

## Files Changed
- Created: `ml_pipeline/shared/earth_engine_features.py`
- Created: `ml_pipeline/shared/__init__.py`
- Modified: `ml_pipeline/orchestrator/earth_engine_collector.py` (now uses shared module)
- Modified: `backend/app.py` (now uses shared module)

## Next Steps
1. Deploy updated orchestrator to Cloud Run
2. Deploy updated backend API to Cloud Run
3. Test end-to-end:
   - Data collection → Training → Inference
   - Verify identical feature extraction at all stages

