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


# REMOVED: Elevation binning functions — elevation removed from feature set


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
    # REMOVED: elevation_m — elevation removed from feature set
    # REMOVED: latitude, longitude — model no longer uses geographic cheating
) -> List[float]:
    """
    Engineer all features from raw values.
    This is the single source of truth for feature engineering.
    
    Args:
        All raw NDVI feature values
        # REMOVED: elevation_m, elevation_quantiles — elevation removed from feature set
        # REMOVED: latitude, longitude — model no longer uses geographic cheating
    
    Returns:
        List of 14 engineered features in exact order (removed 4 location + 1 elevation features)
    """
    # Derived NDVI features
    ndvi_range = ndvi_max - ndvi_min
    ndvi_iqr = ndvi_p75 - ndvi_p25
    ndvi_change = ndvi_late - ndvi_early
    ndvi_early_ratio = ndvi_early / (ndvi_mean + 0.001)
    ndvi_late_ratio = ndvi_late / (ndvi_mean + 0.001)
    
    # REMOVED: Encode location (sin/cos) — model no longer uses geographic cheating
    # REMOVED: Bin elevation — elevation removed from feature set
    
    # Return features in exact order (must match training!)
    # REMOVED: lat_sin, lat_cos, lon_sin, lon_cos (4 features)
    # REMOVED: elevation_binned (1 feature)
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
        # REMOVED: float(elevation_binned),  # Binned elevation
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


def engineer_features_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Engineer features for training dataframe.
    
    Args:
        df: DataFrame with raw features
        # REMOVED: elevation_quantiles — elevation removed from feature set
    
    Returns:
        (df_enhanced, feature_columns)
    """
    df = df.copy()
    
    # REMOVED: Encode location features — model no longer uses geographic cheating
    # REMOVED: Bin elevation — elevation removed from feature set
    
    # Derived NDVI features
    df['ndvi_range'] = df['ndvi_max'] - df['ndvi_min']
    df['ndvi_iqr'] = df['ndvi_p75'] - df['ndvi_p25']
    df['ndvi_change'] = df['ndvi_late'] - df['ndvi_early']
    df['ndvi_early_ratio'] = df['ndvi_early'] / (df['ndvi_mean'] + 0.001)
    df['ndvi_late_ratio'] = df['ndvi_late'] / (df['ndvi_mean'] + 0.001)
    
    # Feature column order (must match engineer_features_from_raw!)
    # REMOVED: lat_sin, lat_cos, lon_sin, lon_cos — model no longer uses geographic cheating
    # REMOVED: elevation_binned — elevation removed from feature set
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
        # REMOVED: 'elevation_binned',  # Elevation removed from feature set
        # REMOVED: 'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos' — model no longer uses geographic cheating
        'ndvi_range',
        'ndvi_iqr',
        'ndvi_change',
        'ndvi_early_ratio',
        'ndvi_late_ratio',
    ]
    
    return df, feature_cols

