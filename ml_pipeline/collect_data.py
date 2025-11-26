"""
Data Collection from Earth Engine
==================================
Collects training samples from Earth Engine and saves to BigQuery.
"""

import ee
import yaml
import logging
from datetime import datetime, timedelta
from google.cloud import bigquery, storage

logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    """Load configuration from Cloud Storage or local file."""
    try:
        # Try Cloud Storage first
        client = storage.Client()
        bucket = client.bucket('carboncheck-data')
        blob = bucket.blob('config/config.yaml')
        yaml_content = blob.download_as_text()
        return yaml.safe_load(yaml_content)
    except:
        # Fall back to local file
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)

config = load_config()


def collect_training_data():
    """
    Collect training samples from Earth Engine.
    
    Returns:
        dict with collection results
    """
    logger.info("📥 Loading configuration...")
    samples_per_crop = config['data_collection']['samples_per_crop']
    crops = config['data_collection']['crops']
    project_id = config['project']['id']
    dataset_id = config['bigquery']['dataset']
    table_id = config['bigquery']['tables']['training']
    
    logger.info(f"   Samples per crop: {samples_per_crop}")
    logger.info(f"   Crops: {[c['name'] for c in crops]}")
    
    try:
        # Initialize Earth Engine with project ID
        logger.info("🌍 Initializing Earth Engine...")
        ee.Initialize(project=project_id)
        
        # Define date ranges
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        early_start = f"{end_date.year}-04-01"
        early_end = f"{end_date.year}-05-31"
        late_start = f"{end_date.year}-08-01"
        late_end = f"{end_date.year}-09-30"
        
        logger.info(f"📅 Collection period: {start_date.date()} to {end_date.date()}")
        
        # Get Sentinel-2 composites
        logger.info("🛰️  Creating Sentinel-2 composites...")
        ndvi_full = get_sentinel2_composite(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        ndvi_early = get_sentinel2_composite(early_start, early_end)
        ndvi_late = get_sentinel2_composite(late_start, late_end)
        
        # Get CDL (use previous year - CDL data typically has 1-year delay)
        cdl_year = end_date.year - 1
        cdl = ee.Image(f'USDA/NASS/CDL/{cdl_year}')
        logger.info(f"📊 Using CDL year: {cdl_year}")
        
        # Collect samples for each crop
        all_samples = []
        for crop in crops:
            logger.info(f"🌱 Collecting {samples_per_crop} samples for {crop['name']}...")
            crop_samples = sample_crop(crop, cdl, ndvi_full, ndvi_early, ndvi_late, samples_per_crop)
            all_samples.extend(crop_samples)
            logger.info(f"   ✅ Collected {len(crop_samples)} samples")
        
        # Load to BigQuery
        logger.info(f"📤 Loading {len(all_samples)} samples to BigQuery...")
        load_to_bigquery(all_samples, project_id, dataset_id, table_id)
        
        logger.info("✅ Data collection complete")
        
        return {
            'status': 'success',
            'samples_collected': len(all_samples),
            'collection_date': datetime.now().isoformat(),
            'crops': {crop['name']: samples_per_crop for crop in crops}
        }
    
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'collection_date': datetime.now().isoformat()
        }


def get_sentinel2_composite(start_date, end_date):
    """Get cloud-free Sentinel-2 composite with NDVI."""
    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(add_ndvi)
    
    return collection.select('NDVI').median()


def sample_crop(crop, cdl, ndvi_full, ndvi_early, ndvi_late, samples_needed):
    """Sample points for a specific crop."""
    crop_mask = cdl.select('cropland').eq(crop['code']).selfMask()
    elevation = ee.Image('USGS/SRTMGL1_003')
    
    samples_collected = []
    county_idx = 0
    
    while len(samples_collected) < samples_needed and county_idx < len(crop['counties']):
        county_geoid = crop['counties'][county_idx]
        counties = ee.FeatureCollection('TIGER/2018/Counties')
        county_geom = counties.filter(ee.Filter.eq('GEOID', county_geoid)).geometry()
        
        remaining = samples_needed - len(samples_collected)
        
        # Sample from this county
        county_samples = crop_mask.sample(
            region=county_geom,
            scale=30,
            numPixels=remaining * 100,
            seed=int(datetime.now().timestamp()) + crop['code'] + county_idx,
            geometries=True,
            tileScale=16
        )
        
        sample_list = county_samples.toList(remaining)
        sample_count = min(remaining, sample_list.size().getInfo())
        
        for i in range(sample_count):
            feature = ee.Feature(sample_list.get(i))
            point = feature.geometry()
            buffer = point.buffer(100)
            
            # Compute NDVI stats
            stats = ndvi_full.reduceRegion(
                reducer=ee.Reducer.mean()
                    .combine(ee.Reducer.stdDev(), '', True)
                    .combine(ee.Reducer.min(), '', True)
                    .combine(ee.Reducer.max(), '', True)
                    .combine(ee.Reducer.percentile([25, 50, 75]), '', True),
                geometry=buffer,
                scale=30,
                maxPixels=1e9
            )
            
            early_mean = ndvi_early.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=30)
            late_mean = ndvi_late.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=30)
            elev = elevation.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30)
            coords = point.coordinates()
            
            values = stats.getInfo()
            values['ndvi_early'] = early_mean.get('NDVI_mean').getInfo()
            values['ndvi_late'] = late_mean.get('NDVI_mean').getInfo()
            values['elevation_m'] = elev.get('elevation').getInfo()
            values['longitude'] = coords.get(0).getInfo()
            values['latitude'] = coords.get(1).getInfo()
            values['crop'] = crop['name']
            values['crop_code'] = crop['code']
            values['cdl_code'] = crop['code']
            values['collection_date'] = datetime.now().isoformat()
            values['sample_id'] = f"{crop['code']}_{county_geoid}_{i}_{int(datetime.now().timestamp())}"
            
            # Rename keys to match BigQuery schema
            values['ndvi_mean'] = values.pop('NDVI_mean')
            values['ndvi_std'] = values.pop('NDVI_stdDev')
            values['ndvi_min'] = values.pop('NDVI_min')
            values['ndvi_max'] = values.pop('NDVI_max')
            values['ndvi_p25'] = values.pop('NDVI_p25')
            values['ndvi_p50'] = values.pop('NDVI_p50')
            values['ndvi_p75'] = values.pop('NDVI_p75')
            
            samples_collected.append(values)
            
            if len(samples_collected) >= samples_needed:
                break
        
        county_idx += 1
    
    return samples_collected


def load_to_bigquery(samples, project_id, dataset_id, table_id):
    """Load samples to BigQuery."""
    if not samples:
        logger.warning("⚠️  No samples to load")
        return
    
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    errors = client.insert_rows_json(table_ref, samples)
    
    if errors:
        raise Exception(f"BigQuery insert failed: {errors}")
    
    logger.info(f"✅ Loaded {len(samples)} samples to {table_ref}")


if __name__ == '__main__':
    # For testing
    result = collect_training_data()
    import json
    print(json.dumps(result, indent=2, default=str))

