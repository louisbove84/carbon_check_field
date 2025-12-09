"""
Earth Engine Data Collector
===========================
Automated collection of training data from Google Earth Engine.
Converts the JavaScript NDVI_info script to Python for pipeline automation.
"""

import ee
import logging
import sys
import os
from typing import List, Dict, Any
from datetime import datetime
from google.cloud import bigquery

# Add shared module to Python path
# Works both locally (../shared) and in Docker (/app/shared)
shared_paths = [
    os.path.join(os.path.dirname(__file__), '..', 'shared'),  # Local: ml_pipeline/shared
    '/app/shared',  # Docker: /app/shared
    os.path.join(os.path.dirname(__file__), '..', '..', 'shared')  # Alternative local path
]
for path in shared_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from earth_engine_features import compute_ndvi_features_ee_as_feature

logger = logging.getLogger(__name__)


def initialize_earth_engine(project_id: str):
    """Initialize Earth Engine with the project."""
    try:
        ee.Initialize(project=project_id)
        logger.info("✅ Earth Engine initialized")
    except Exception as e:
        logger.error(f"❌ Earth Engine initialization failed: {e}")
        raise


def extract_features(point: ee.Geometry, cdl_year: int) -> ee.Feature:
    """
    Extract NDVI features from a point location.
    Uses shared feature extraction module.
    """
    # Use shared function - single source of truth!
    return compute_ndvi_features_ee_as_feature(point, cdl_year)


def sample_crop_fields(crop_info: Dict[str, Any], cdl_year: int, 
                       num_fields_per_crop: int, num_samples_per_field: int) -> ee.FeatureCollection:
    """
    Sample crop fields with CDL verification.
    Matches the JavaScript sampleCropFields function.
    """
    crop_name = crop_info['name']
    crop_code = crop_info['code']
    counties = crop_info['counties']
    
    logger.info(f"🌱 Collecting samples for {crop_name} (CDL code: {crop_code})")
    
    target_samples = num_fields_per_crop * num_samples_per_field
    logger.info(f"   🎯 Target: {target_samples} samples")
    
    # Load CDL
    cdl = ee.Image(f'USDA/NASS/CDL/{cdl_year}').select('cropland')
    
    # Mask CDL to only show pixels of this crop
    crop_mask = cdl.updateMask(cdl.eq(crop_code))
    
    # Load counties
    counties_fc = ee.FeatureCollection('TIGER/2018/Counties')
    
    # Collect samples from all counties
    all_samples = ee.FeatureCollection([])
    samples_collected = 0
    
    for i, county_geoid in enumerate(counties):
        if samples_collected >= target_samples:
            break
        
        # Get county geometry
        county_region = counties_fc.filter(ee.Filter.eq('GEOID', county_geoid)).geometry()
        
        # How many more samples do we need?
        samples_needed = target_samples - samples_collected
        
        # Sample from this county (request 500x what we need to ensure we get matches)
        # Higher buffer accounts for: missing pixels, cloud cover, null NDVI values
        county_samples = crop_mask.sample(
            region=county_region,
            scale=30,
            numPixels=samples_needed * 500,  # 500x buffer (increased from 100x)
            seed=42 + crop_code + i,
            geometries=True,
            tileScale=16
        )
        
        # Limit to what we need
        samples_to_add = county_samples.limit(samples_needed)
        all_samples = all_samples.merge(samples_to_add)
        
        # Get actual count (this is a server-side operation, so we approximate)
        samples_collected += samples_needed
        logger.info(f"   📦 County {i+1} (GEOID: {county_geoid}): ~{samples_needed} samples")
    
    logger.info(f"   ✅ Collected samples from {min(len(counties), len(counties))} counties")
    
    # Add crop info to each sample
    all_samples = all_samples.map(lambda feature: feature.set({
        'crop': crop_name,
        'crop_code': crop_code
    }))
    
    # Extract NDVI features for each sample
    all_samples = all_samples.map(lambda feature: extract_features(
        feature.geometry(), cdl_year
    ).copyProperties(feature))
    
    return all_samples


def sample_non_crop_areas(config: Dict[str, Any], cdl_year: int, 
                          num_fields: int, num_samples_per_field: int) -> ee.FeatureCollection:
    """
    Sample non-crop areas (buildings, roads, lakes, trees, etc.) for 'Other' category.
    Samples from areas that are NOT in the crop codes list.
    
    Args:
        config: Pipeline configuration dict
        cdl_year: CDL year to use
        num_fields: Number of fields to sample
        num_samples_per_field: Number of samples per field
    
    Returns:
        FeatureCollection with 'Other' category samples
    """
    project_id = config['project']['id']
    crops = config['data_collection']['crops']
    
    # Get all crop codes to exclude
    crop_codes = [crop['code'] for crop in crops]
    
    logger.info(f"🌍 Collecting samples for 'Other' category (non-crop areas)")
    logger.info(f"   Excluding crop codes: {crop_codes}")
    
    target_samples = num_fields * num_samples_per_field
    logger.info(f"   🎯 Target: {target_samples} samples")
    
    # Load CDL
    cdl = ee.Image(f'USDA/NASS/CDL/{cdl_year}').select('cropland')
    
    # Create mask for non-crop areas (exclude all crop codes)
    # Start with all pixels, then mask out crop codes
    non_crop_mask = cdl
    for crop_code in crop_codes:
        non_crop_mask = non_crop_mask.updateMask(non_crop_mask.neq(crop_code))
    
    # Use a mix of counties from different regions for diversity
    # Sample from the same counties used for crops, but get non-crop areas
    counties_fc = ee.FeatureCollection('TIGER/2018/Counties')
    
    # Collect all unique counties from crop definitions
    all_counties = []
    for crop in crops:
        all_counties.extend(crop.get('counties', []))
    unique_counties = list(set(all_counties))[:10]  # Use up to 10 counties for diversity
    
    all_samples = ee.FeatureCollection([])
    samples_collected = 0
    
    for i, county_geoid in enumerate(unique_counties):
        if samples_collected >= target_samples:
            break
        
        # Get county geometry
        county_region = counties_fc.filter(ee.Filter.eq('GEOID', county_geoid)).geometry()
        
        # How many more samples do we need?
        samples_needed = target_samples - samples_collected
        
        # Sample from non-crop areas in this county
        county_samples = non_crop_mask.sample(
            region=county_region,
            scale=30,
            numPixels=samples_needed * 500,  # 500x buffer
            seed=999 + i,  # Different seed for 'Other' category
            geometries=True,
            tileScale=16
        )
        
        # Limit to what we need
        samples_to_add = county_samples.limit(samples_needed)
        all_samples = all_samples.merge(samples_to_add)
        
        samples_collected += samples_needed
        logger.info(f"   📦 County {i+1} (GEOID: {county_geoid}): ~{samples_needed} samples")
    
    logger.info(f"   ✅ Collected samples from {min(len(unique_counties), len(unique_counties))} counties")
    
    # Add 'Other' category info to each sample
    all_samples = all_samples.map(lambda feature: feature.set({
        'crop': 'Other',
        'crop_code': 0  # Use 0 for 'Other' category
    }))
    
    # Extract NDVI features for each sample
    all_samples = all_samples.map(lambda feature: extract_features(
        feature.geometry(), cdl_year
    ).copyProperties(feature))
    
    return all_samples


def collect_training_data(config: Dict[str, Any]) -> ee.FeatureCollection:
    """
    Collect training data for all crops plus 'Other' category.
    
    Args:
        config: Pipeline configuration dict
    
    Returns:
        FeatureCollection with all training samples
    """
    project_id = config['project']['id']
    cdl_year = config.get('data_collection', {}).get('cdl_year', 2024)
    num_fields_per_crop = config.get('data_collection', {}).get('num_fields_per_crop', 30)
    num_samples_per_field = config.get('data_collection', {}).get('num_samples_per_field', 3)
    crops = config['data_collection']['crops']
    
    # Check if 'Other' category should be collected
    collect_other = config.get('data_collection', {}).get('collect_other', True)
    num_other_fields = config.get('data_collection', {}).get('num_other_fields', num_fields_per_crop)
    
    # Initialize Earth Engine
    initialize_earth_engine(project_id)
    
    # Collect samples for each crop
    all_training_data = ee.FeatureCollection([])
    
    for crop in crops:
        crop_samples = sample_crop_fields(
            crop, cdl_year, num_fields_per_crop, num_samples_per_field
        )
        all_training_data = all_training_data.merge(crop_samples)
    
    # Collect 'Other' category samples (non-crop areas)
    if collect_other:
        logger.info("")
        logger.info("🌍 Collecting 'Other' category (non-crop areas)...")
        other_samples = sample_non_crop_areas(
            config, cdl_year, num_other_fields, num_samples_per_field
        )
        all_training_data = all_training_data.merge(other_samples)
        logger.info("✅ 'Other' category samples collected")
    
    return all_training_data


def export_to_bigquery(feature_collection: ee.FeatureCollection, 
                       project_id: str, dataset_id: str, table_id: str) -> str:
    """
    Export FeatureCollection directly to BigQuery.
    
    Args:
        feature_collection: Earth Engine FeatureCollection
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
    
    Returns:
        Task ID for monitoring
    """
    # Create field_id for each sample
    def add_field_id(feature):
        coords = feature.geometry().coordinates()
        lon = ee.Number(coords.get(0)).format('%.5f')
        lat = ee.Number(coords.get(1)).format('%.5f')
        field_id = ee.String('field_').cat(lon).cat('_').cat(lat)
        return feature.set('field_id', field_id)
    
    feature_collection = feature_collection.map(add_field_id)
    
    # Export to BigQuery
    table_ref = f'{project_id}.{dataset_id}.{table_id}'
    
    task = ee.batch.Export.table.toBigQuery(
        collection=feature_collection,
        description='crop_features_to_bigquery_automated',
        table=table_ref,
        append=True,  # Add to existing table
        selectors=[
            'field_id', 'crop', 'crop_code',
            'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max',
            'ndvi_p25', 'ndvi_p50', 'ndvi_p75',
            'ndvi_early', 'ndvi_late',
            # REMOVED: 'elevation_m' — elevation removed from feature set
            # REMOVED: 'longitude', 'latitude' — model no longer uses geographic cheating
        ]
    )
    
    task.start()
    
    logger.info(f"✅ Export task started: {task.id}")
    logger.info(f"   Destination: {table_ref}")
    logger.info(f"   Mode: WRITE_APPEND (adds to existing data)")
    
    return task.id


def wait_for_export(task_id: str, timeout_minutes: int = 30):
    """
    Wait for BigQuery export to complete.
    
    Args:
        task_id: Earth Engine task ID
        timeout_minutes: Maximum time to wait
    
    Returns:
        True if completed, False if failed/timed out
    """
    import time
    
    logger.info(f"⏳ Waiting for export task {task_id} to complete...")
    logger.info(f"   (This may take 5-15 minutes for large datasets)")
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    check_interval = 30  # Check every 30 seconds
    
    while time.time() - start_time < timeout_seconds:
        try:
            # Get task status
            tasks = ee.batch.Task.list()
            task = None
            for t in tasks:
                if t.id == task_id:
                    task = t
                    break
            
            if task is None:
                logger.warning(f"   Task {task_id} not found in task list")
                time.sleep(check_interval)
                continue
            
            state = task.state
            
            if state == 'COMPLETED':
                logger.info(f"✅ Export completed successfully!")
                return True
            elif state == 'FAILED':
                error_msg = getattr(task, 'error_message', 'Unknown error')
                logger.error(f"❌ Export failed: {error_msg}")
                return False
            elif state == 'CANCELLED':
                logger.warning(f"⚠️  Export was cancelled")
                return False
            elif state in ['READY', 'RUNNING']:
                elapsed = int((time.time() - start_time) / 60)
                logger.info(f"   Status: {state}... ({elapsed} min elapsed)")
            
            # Wait before checking again
            time.sleep(check_interval)
            
        except Exception as e:
            logger.warning(f"   Error checking task status: {e}")
            time.sleep(check_interval)
    
    elapsed_min = int((time.time() - start_time) / 60)
    logger.warning(f"⚠️  Export timeout after {elapsed_min} minutes")
    logger.warning(f"   Task may still be running - check Earth Engine Tasks tab")
    logger.warning(f"   Pipeline will continue, but data may not be available yet")
    return False

