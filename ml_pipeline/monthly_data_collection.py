"""
Monthly Crop Data Collection Pipeline
======================================
Cloud Function that runs monthly to:
1. Sample latest Sentinel-2 imagery from Earth Engine
2. Extract NDVI features for crop training data
3. Verify samples against CDL (Cropland Data Layer)
4. Load results directly to BigQuery

Triggered by Cloud Scheduler on the 1st of each month.
"""

import ee
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from google.cloud import bigquery
from google.cloud import storage

# Import centralized configuration
from config import (
    PROJECT_ID, REGION, BUCKET_NAME, DATASET_ID,
    TRAINING_TABLE_ID, SAMPLES_PER_CROP, CROPS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION (from centralized config.py)
# ============================================================

TABLE_ID = TRAINING_TABLE_ID

# ============================================================
# EARTH ENGINE FUNCTIONS
# ============================================================

def initialize_earth_engine():
    """Initialize Earth Engine with Application Default Credentials."""
    try:
        ee.Initialize()
        logger.info("✅ Earth Engine initialized")
    except Exception as e:
        logger.error(f"❌ Earth Engine initialization failed: {e}")
        raise


def get_sentinel2_composite(start_date: str, end_date: str) -> ee.Image:
    """
    Get cloud-free Sentinel-2 composite for date range.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Sentinel-2 composite image with NDVI band
    """
    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(add_ndvi)
    
    return collection.select('NDVI').median()


def get_county_geometry(geoid: str) -> ee.Geometry:
    """Get county geometry from TIGER dataset."""
    counties = ee.FeatureCollection('TIGER/2018/Counties')
    county = counties.filter(ee.Filter.eq('GEOID', geoid))
    return county.geometry()


def extract_ndvi_features(point: ee.Geometry, ndvi_image: ee.Image, 
                          early_ndvi: ee.Image, late_ndvi: ee.Image) -> Dict:
    """
    Extract NDVI statistics at a point.
    
    Args:
        point: Sample location
        ndvi_image: Full season NDVI composite
        early_ndvi: Early season NDVI
        late_ndvi: Late season NDVI
    
    Returns:
        Dictionary of NDVI features
    """
    # Get elevation
    elevation = ee.Image('USGS/SRTMGL1_003')
    
    # Create 100m buffer for regional stats
    buffer = point.buffer(100)
    
    # Compute statistics
    stats = ndvi_image.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), '', True)
            .combine(ee.Reducer.min(), '', True)
            .combine(ee.Reducer.max(), '', True)
            .combine(ee.Reducer.percentile([25, 50, 75]), '', True),
        geometry=buffer,
        scale=30,
        maxPixels=1e9
    )
    
    early_mean = early_ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=30
    )
    
    late_mean = late_ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=30
    )
    
    elev = elevation.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=30
    )
    
    coords = point.coordinates()
    
    return {
        'ndvi_mean': stats.get('NDVI_mean'),
        'ndvi_std': stats.get('NDVI_stdDev'),
        'ndvi_min': stats.get('NDVI_min'),
        'ndvi_max': stats.get('NDVI_max'),
        'ndvi_p25': stats.get('NDVI_p25'),
        'ndvi_p50': stats.get('NDVI_p50'),
        'ndvi_p75': stats.get('NDVI_p75'),
        'ndvi_early': early_mean.get('NDVI_mean'),
        'ndvi_late': late_mean.get('NDVI_mean'),
        'elevation_m': elev.get('elevation'),
        'longitude': coords.get(0),
        'latitude': coords.get(1)
    }


def sample_crop_from_counties(crop: Dict, cdl: ee.Image, 
                               ndvi_full: ee.Image, ndvi_early: ee.Image, 
                               ndvi_late: ee.Image, samples_needed: int) -> List[Dict]:
    """
    Sample points for a specific crop from its top counties.
    
    Args:
        crop: Crop configuration dict
        cdl: CDL image for verification
        ndvi_full: Full season NDVI
        ndvi_early: Early season NDVI
        ndvi_late: Late season NDVI
        samples_needed: Number of samples to collect
    
    Returns:
        List of sample dictionaries
    """
    logger.info(f"🌱 Sampling {samples_needed} points for {crop['name']} (CDL code {crop['code']})")
    
    # Create mask for this crop
    crop_mask = cdl.select('cropland').eq(crop['code']).selfMask()
    
    samples_collected = []
    county_idx = 0
    
    # Try counties until we have enough samples
    while len(samples_collected) < samples_needed and county_idx < len(crop['counties']):
        county_geoid = crop['counties'][county_idx]
        county_geom = get_county_geometry(county_geoid)
        
        remaining = samples_needed - len(samples_collected)
        
        # Sample from this county (request 100x for buffer)
        county_samples = crop_mask.sample(
            region=county_geom,
            scale=30,
            numPixels=remaining * 100,
            seed=int(datetime.now().timestamp()) + crop['code'] + county_idx,
            geometries=True,
            tileScale=16
        )
        
        # Convert to list and extract features
        sample_list = county_samples.toList(remaining)
        sample_count = min(remaining, sample_list.size().getInfo())
        
        logger.info(f"  📦 County {county_idx + 1} (GEOID: {county_geoid}): {sample_count} samples")
        
        for i in range(sample_count):
            feature = ee.Feature(sample_list.get(i))
            point = feature.geometry()
            
            # Extract all NDVI features
            features_dict = extract_ndvi_features(point, ndvi_full, ndvi_early, ndvi_late)
            
            # Get all values
            values = features_dict.getInfo()
            
            # Add crop metadata
            values['crop'] = crop['name']
            values['crop_code'] = crop['code']
            values['cdl_code'] = crop['code']  # Verified by mask
            values['collection_date'] = datetime.now().isoformat()
            values['sample_id'] = f"{crop['code']}_{county_geoid}_{i}_{int(datetime.now().timestamp())}"
            
            samples_collected.append(values)
            
            if len(samples_collected) >= samples_needed:
                break
        
        county_idx += 1
    
    logger.info(f"  ✅ Collected {len(samples_collected)}/{samples_needed} samples for {crop['name']}")
    
    return samples_collected


# ============================================================
# BIGQUERY FUNCTIONS
# ============================================================

def load_to_bigquery(samples: List[Dict]) -> None:
    """
    Load samples directly to BigQuery.
    
    Args:
        samples: List of sample dictionaries
    """
    if not samples:
        logger.warning("⚠️  No samples to load")
        return
    
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    logger.info(f"📤 Loading {len(samples)} samples to BigQuery: {table_id}")
    
    # Insert rows
    errors = client.insert_rows_json(table_id, samples)
    
    if errors:
        logger.error(f"❌ BigQuery insert failed: {errors}")
        raise Exception(f"BigQuery insert failed: {errors}")
    
    logger.info(f"✅ Successfully loaded {len(samples)} samples to BigQuery")


def get_sample_counts() -> Dict[str, int]:
    """
    Get current sample counts per crop from BigQuery.
    
    Returns:
        Dictionary mapping crop name to sample count
    """
    client = bigquery.Client(project=PROJECT_ID)
    
    query = f"""
    SELECT crop, COUNT(*) as count
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    GROUP BY crop
    ORDER BY crop
    """
    
    results = client.query(query).result()
    
    counts = {}
    for row in results:
        counts[row.crop] = row.count
    
    return counts


# ============================================================
# MAIN PIPELINE FUNCTION
# ============================================================

def collect_training_data(request=None):
    """
    Main Cloud Function entry point.
    Collects monthly training data from Earth Engine.
    
    Args:
        request: Flask request object (unused, required for Cloud Functions)
    
    Returns:
        JSON response with collection summary
    """
    logger.info("=" * 60)
    logger.info("🌾 MONTHLY CROP DATA COLLECTION PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Initialize Earth Engine
        initialize_earth_engine()
        
        # Define date ranges
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        # Early season: April-May
        early_start = f"{end_date.year}-04-01"
        early_end = f"{end_date.year}-05-31"
        
        # Late season: August-September
        late_start = f"{end_date.year}-08-01"
        late_end = f"{end_date.year}-09-30"
        
        logger.info(f"📅 Collection period: {start_date.date()} to {end_date.date()}")
        
        # Get Sentinel-2 composites
        logger.info("🛰️  Creating Sentinel-2 composites...")
        ndvi_full = get_sentinel2_composite(start_date.strftime('%Y-%m-%d'), 
                                            end_date.strftime('%Y-%m-%d'))
        ndvi_early = get_sentinel2_composite(early_start, early_end)
        ndvi_late = get_sentinel2_composite(late_start, late_end)
        
        # Get current year's CDL
        cdl_year = end_date.year
        cdl = ee.Image(f'USDA/NASS/CDL/{cdl_year}')
        logger.info(f"📊 Using CDL year: {cdl_year}")
        
        # Collect samples for each crop
        all_samples = []
        
        for crop in CROPS:
            crop_samples = sample_crop_from_counties(
                crop, cdl, ndvi_full, ndvi_early, ndvi_late, SAMPLES_PER_CROP
            )
            all_samples.extend(crop_samples)
        
        # Load to BigQuery
        load_to_bigquery(all_samples)
        
        # Get updated counts
        final_counts = get_sample_counts()
        
        logger.info("=" * 60)
        logger.info("✅ COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"📊 Samples collected this run: {len(all_samples)}")
        logger.info("📊 Total samples in BigQuery:")
        for crop, count in final_counts.items():
            logger.info(f"   • {crop}: {count}")
        
        return {
            'status': 'success',
            'samples_collected': len(all_samples),
            'collection_date': datetime.now().isoformat(),
            'total_counts': final_counts
        }
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'collection_date': datetime.now().isoformat()
        }


# ============================================================
# LOCAL TESTING
# ============================================================

if __name__ == '__main__':
    # For local testing
    result = collect_training_data()
    print(json.dumps(result, indent=2))

