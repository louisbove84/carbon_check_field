"""
Shared Earth Engine Feature Extraction
======================================
Single source of truth for computing NDVI features from Earth Engine.
Used by both training data collection AND inference/prediction.
"""

import ee
from typing import Dict, List, Tuple


def compute_ndvi_features_ee(
    geometry: ee.Geometry,
    year: int,
    buffer_size: int = 50
) -> Dict[str, any]:
    """
    Compute NDVI and related features from Earth Engine for a given geometry.
    
    This is the SINGLE SOURCE OF TRUTH for NDVI feature extraction.
    Used by:
    - Training data collection (earth_engine_collector.py)
    - Inference API (backend/app.py)
    
    Args:
        geometry: Earth Engine geometry (point or polygon)
        year: Year for analysis (e.g., 2024)
        buffer_size: Buffer in meters for sampling stability
    
    Returns:
        Dict with 12 raw features:
        - ndvi_mean, ndvi_std, ndvi_min, ndvi_max
        - ndvi_p25, ndvi_p50, ndvi_p75
        - ndvi_early, ndvi_late
        - elevation_m
        - longitude, latitude
    """
    # Define growing season
    start_date = f'{year}-04-15'
    end_date = f'{year}-09-01'
    early_end = f'{year}-06-01'
    late_start = f'{year}-07-01'
    
    # Get Sentinel-2 data
    s2_collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )
    
    # Calculate NDVI
    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return ndvi.copyProperties(image, ['system:time_start'])
    
    ndvi_collection = s2_collection.map(add_ndvi).select('NDVI')
    
    # Overall statistics
    ndvi_composite = ndvi_collection.median()
    
    ndvi_stats = ndvi_composite.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), '', True)
            .combine(ee.Reducer.min(), '', True)
            .combine(ee.Reducer.max(), '', True)
            .combine(ee.Reducer.percentile([25, 50, 75]), '', True),
        geometry=geometry.buffer(buffer_size),
        scale=10,
        maxPixels=1e9
    )
    
    # Early season (Apr-May)
    early_ndvi = (
        ndvi_collection
        .filterDate(start_date, early_end)
        .median()
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry.buffer(buffer_size),
            scale=10,
            maxPixels=1e9
        )
    )
    
    # Late season (Jul-Aug)
    late_ndvi = (
        ndvi_collection
        .filterDate(late_start, end_date)
        .median()
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry.buffer(buffer_size),
            scale=10,
            maxPixels=1e9
        )
    )
    
    # Elevation from SRTM
    elevation = (
        ee.Image('USGS/SRTMGL1_003')
        .select('elevation')
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=30,
            maxPixels=1e9
        )
    )
    
    # Get centroid coordinates
    centroid = geometry.centroid().coordinates()
    
    return {
        'ndvi_mean': ndvi_stats.get('NDVI_mean'),
        'ndvi_std': ndvi_stats.get('NDVI_stdDev'),
        'ndvi_min': ndvi_stats.get('NDVI_min'),
        'ndvi_max': ndvi_stats.get('NDVI_max'),
        'ndvi_p25': ndvi_stats.get('NDVI_p25'),
        'ndvi_p50': ndvi_stats.get('NDVI_p50'),
        'ndvi_p75': ndvi_stats.get('NDVI_p75'),
        'ndvi_early': early_ndvi.get('NDVI'),
        'ndvi_late': late_ndvi.get('NDVI'),
        'elevation_m': elevation.get('elevation'),
        'longitude': centroid.get(0),
        'latitude': centroid.get(1)
    }


def compute_ndvi_features_ee_as_feature(
    point: ee.Geometry,
    year: int
) -> ee.Feature:
    """
    Compute NDVI features and return as Earth Engine Feature.
    Used by earth_engine_collector.py for batch processing.
    
    Args:
        point: Earth Engine point geometry
        year: Year for analysis
    
    Returns:
        ee.Feature with NDVI properties
    """
    features_dict = compute_ndvi_features_ee(point, year)
    return ee.Feature(point, features_dict)


def compute_ndvi_features_sync(
    geometry: ee.Geometry,
    year: int
) -> Dict[str, float]:
    """
    Compute NDVI features and return as Python dict (synchronous).
    Used by backend/app.py for real-time inference.
    
    Args:
        geometry: Earth Engine geometry
        year: Year for analysis
    
    Returns:
        Dict with computed feature values
    """
    features_dict = compute_ndvi_features_ee(geometry, year)
    
    # Convert to Python values with defaults
    return {
        'ndvi_mean': features_dict['ndvi_mean'].getInfo() if features_dict['ndvi_mean'] else 0.5,
        'ndvi_std': features_dict['ndvi_std'].getInfo() if features_dict['ndvi_std'] else 0.1,
        'ndvi_min': features_dict['ndvi_min'].getInfo() if features_dict['ndvi_min'] else 0.0,
        'ndvi_max': features_dict['ndvi_max'].getInfo() if features_dict['ndvi_max'] else 1.0,
        'ndvi_p25': features_dict['ndvi_p25'].getInfo() if features_dict['ndvi_p25'] else 0.4,
        'ndvi_p50': features_dict['ndvi_p50'].getInfo() if features_dict['ndvi_p50'] else 0.5,
        'ndvi_p75': features_dict['ndvi_p75'].getInfo() if features_dict['ndvi_p75'] else 0.6,
        'ndvi_early': features_dict['ndvi_early'].getInfo() if features_dict['ndvi_early'] else 0.5,
        'ndvi_late': features_dict['ndvi_late'].getInfo() if features_dict['ndvi_late'] else 0.5,
        'elevation_m': features_dict['elevation_m'].getInfo() if features_dict['elevation_m'] else 0.0,
        'longitude': features_dict['longitude'].getInfo() if features_dict['longitude'] else 0.0,
        'latitude': features_dict['latitude'].getInfo() if features_dict['latitude'] else 0.0,
    }

