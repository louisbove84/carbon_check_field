"""
Shared Feature Engineering Module
=================================
This module contains feature engineering functions used by both:
1. Training pipeline (vertex_ai_training.py)
2. Prediction API (backend/app.py)

This ensures feature engineering is consistent between training and inference.
"""

import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


def encode_location(latitude: float, longitude: float) -> List[float]:
    """
    Encode latitude/longitude using sin/cos transformation.
    This preserves geographic relationships better than raw values.
    
    Args:
        latitude: Latitude in degrees (-90 to 90)
        longitude: Longitude in degrees (-180 to 180)
    
    Returns:
        [lat_sin, lat_cos, lon_sin, lon_cos]
    """
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    
    return [
        math.sin(lat_rad),
        math.cos(lat_rad),
        math.sin(lon_rad),
        math.cos(lon_rad)
    ]


def bin_elevation_quantile(elevation_m: float, quantiles: Dict[str, float]) -> int:
    """
    Bin elevation into quantile-based zones.
    
    Args:
        elevation_m: Elevation in meters
        quantiles: Dict with 'q25', 'q50', 'q75' keys from training data
    
    Returns:
        0: Low (< q25)
        1: Medium (q25 to q50)
        2: High (q50 to q75)
        3: Very High (> q75)
    """
    if elevation_m < quantiles['q25']:
        return 0  # Low
    elif elevation_m < quantiles['q50']:
        return 1  # Medium
    elif elevation_m < quantiles['q75']:
        return 2  # High
    else:
        return 3  # Very High


def compute_elevation_quantiles(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute elevation quantiles from training data.
    Call this during training and save quantiles to config.
    
    Args:
        df: Training dataframe with 'elevation_m' column
    
    Returns:
        Dict with 'q25', 'q50', 'q75' quantile values
    """
    if 'elevation_m' not in df.columns:
        # Default quantiles if no elevation data
        return {'q25': 200.0, 'q50': 500.0, 'q75': 1000.0}
    
    quantiles = df['elevation_m'].quantile([0.25, 0.50, 0.75]).to_dict()
    
    return {
        'q25': float(quantiles[0.25]),
        'q50': float(quantiles[0.50]),
        'q75': float(quantiles[0.75])
    }


def engineer_features_from_raw(
    ndvi_mean: float,
    ndvi_std: float,
    ndvi_min: float,
    ndvi_max: float,
    ndvi_p25: float,
    ndvi_p50: float,
    ndvi_p75: float,
    ndvi_early: float,
    ndvi_late: float,
    elevation_m: float,
    # REMOVED: latitude, longitude — model no longer uses geographic cheating
    elevation_quantiles: Dict[str, float]
) -> List[float]:
    """
    Engineer all features from raw values.
    This is the single source of truth for feature engineering.
    
    Args:
        All raw feature values
        elevation_quantiles: Quantiles dict from training data
    
    Returns:
        List of 15 engineered features in exact order (removed 4 location features)
    """
    # Derived NDVI features
    ndvi_range = ndvi_max - ndvi_min
    ndvi_iqr = ndvi_p75 - ndvi_p25
    ndvi_change = ndvi_late - ndvi_early
    ndvi_early_ratio = ndvi_early / (ndvi_mean + 0.001)
    ndvi_late_ratio = ndvi_late / (ndvi_mean + 0.001)
    
    # REMOVED: Encode location (sin/cos) — model no longer uses geographic cheating
    # location_features = encode_location(latitude, longitude)
    
    # Bin elevation (quantile-based)
    elevation_binned = bin_elevation_quantile(elevation_m, elevation_quantiles)
    
    # Return features in exact order (must match training!)
    # REMOVED: lat_sin, lat_cos, lon_sin, lon_cos (4 features)
    features = [
        ndvi_mean,
        ndvi_std,
        ndvi_min,
        ndvi_max,
        ndvi_p25,
        ndvi_p50,
        ndvi_p75,
        ndvi_early,
        ndvi_late,
        float(elevation_binned),  # Binned elevation
        # REMOVED: location_features[0],  # lat_sin
        # REMOVED: location_features[1],  # lat_cos
        # REMOVED: location_features[2],  # lon_sin
        # REMOVED: location_features[3],  # lon_cos
        ndvi_range,
        ndvi_iqr,
        ndvi_change,
        ndvi_early_ratio,
        ndvi_late_ratio,
    ]
    
    return features


def engineer_features_dataframe(df: pd.DataFrame, elevation_quantiles: Dict[str, float]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Engineer features for training dataframe.
    
    Args:
        df: DataFrame with raw features
        elevation_quantiles: Quantiles from training data
    
    Returns:
        (df_enhanced, feature_columns)
    """
    df = df.copy()
    
    # REMOVED: Encode location features — model no longer uses geographic cheating
    # df['lat_sin'] = df['latitude'].apply(lambda x: math.sin(math.radians(x)))
    # df['lat_cos'] = df['latitude'].apply(lambda x: math.cos(math.radians(x)))
    # df['lon_sin'] = df['longitude'].apply(lambda x: math.sin(math.radians(x)))
    # df['lon_cos'] = df['longitude'].apply(lambda x: math.cos(math.radians(x)))
    
    # Bin elevation
    df['elevation_binned'] = df['elevation_m'].apply(
        lambda x: bin_elevation_quantile(x, elevation_quantiles)
    )
    
    # Derived NDVI features
    df['ndvi_range'] = df['ndvi_max'] - df['ndvi_min']
    df['ndvi_iqr'] = df['ndvi_p75'] - df['ndvi_p25']
    df['ndvi_change'] = df['ndvi_late'] - df['ndvi_early']
    df['ndvi_early_ratio'] = df['ndvi_early'] / (df['ndvi_mean'] + 0.001)
    df['ndvi_late_ratio'] = df['ndvi_late'] / (df['ndvi_mean'] + 0.001)
    
    # Feature column order (must match engineer_features_from_raw!)
    # REMOVED: lat_sin, lat_cos, lon_sin, lon_cos — model no longer uses geographic cheating
    feature_cols = [
        'ndvi_mean',
        'ndvi_std',
        'ndvi_min',
        'ndvi_max',
        'ndvi_p25',
        'ndvi_p50',
        'ndvi_p75',
        'ndvi_early',
        'ndvi_late',
        'elevation_binned',
        # REMOVED: 'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos' — model no longer uses geographic cheating
        'ndvi_range',
        'ndvi_iqr',
        'ndvi_change',
        'ndvi_early_ratio',
        'ndvi_late_ratio',
    ]
    
    return df, feature_cols

