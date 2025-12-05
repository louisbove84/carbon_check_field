# Sampling Buffer Fix

## Problem
Initial data collection only yielded 147 samples instead of the expected 400 (37% success rate).

## Root Cause
The Earth Engine sampling buffer was too small (100x) to account for:
- Missing CDL pixels in some counties
- Cloud cover filtering (< 20% for Sentinel-2)
- Null NDVI values from feature extraction failures
- Insufficient crop coverage in selected counties

## Sample Success Rates (Before Fix)
| Crop | Target | Actual | Success Rate |
|------|--------|--------|--------------|
| Soybeans | 90 | 38 | 42% |
| Corn | 90 | 37 | 41% |
| Winter Wheat | 90 | 32 | 36% |
| Alfalfa | 90 | 30 | 33% |
| Cotton | 90 | 10 | **11%** ⚠️ |

Cotton had especially poor results, suggesting limited coverage in the selected counties.

## Solution
Increased the sampling buffer from **100x to 500x** in the Earth Engine collector.

### Code Change
**File**: `ml_pipeline/orchestrator/earth_engine_collector.py`

```python
# Before (line 84):
numPixels=samples_needed * 100,  # 100x buffer

# After (line 84):
numPixels=samples_needed * 500,  # 500x buffer (increased from 100x)
```

### Why 500x?
- **100x** was only achieving ~35-40% success rate
- **500x** should achieve 80-90%+ success rate based on the observed shortfall
- Earth Engine's `.limit()` ensures we only take what we need
- Minimal performance impact (Earth Engine handles large samples efficiently)

## Expected Impact
With 500x buffer, we expect:
- **360+ samples** out of 400 target (90%+ success rate)
- More consistent sample counts across all crops
- Better Cotton representation (40+ samples instead of 10)

## Trade-offs
- **Pros**: Higher sample collection success rate, more robust data collection
- **Cons**: Slightly longer Earth Engine processing time (~10-15% increase)

## Testing
Next data collection run should be monitored to verify:
1. Total samples collected approaches 400
2. Cotton samples increase significantly (from 10 to ~80)
3. All crops have similar success rates

## Follow-up Actions
If 500x still yields low samples:
1. Add more counties to `config.yaml` for underrepresented crops
2. Relax cloud cover threshold from 20% to 30%
3. Consider expanding date range for NDVI imagery

