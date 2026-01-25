"""
Data Skew Detector (Label-Free)
===============================
Detects distribution shifts between training data and recent Earth Engine samples
WITHOUT requiring labeled data.

Metrics computed (all label-free):
1. Feature Distribution Skew:
   - KS (Kolmogorov-Smirnov) test - detects any distributional shift
   - PSI (Population Stability Index) - industry standard with clear thresholds
   - JS (Jensen-Shannon) divergence - symmetric, bounded divergence measure
2. Prediction Entropy - Model uncertainty on new data
3. Prediction Distribution Drift - How model output distribution changes

PSI Interpretation:
- < 0.1: No significant shift
- 0.1 - 0.25: Moderate shift, monitor closely
- >= 0.25: Significant shift, action required

JS Divergence Interpretation:
- < 0.1: Very similar distributions
- 0.1 - 0.2: Moderate difference
- >= 0.2: Significant difference

This job:
1. Pulls random samples from Earth Engine (same regions as training)
2. Loads existing training data from BigQuery
3. Extracts features and gets endpoint predictions
4. Compares feature distributions using KS, PSI, and JS (no labels needed)
5. Monitors prediction entropy
6. Logs visualizations to TensorBoard
7. Stores results in BigQuery
"""

import os
import sys
import re
import logging
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import jensenshannon
from google.cloud import aiplatform, bigquery, storage
import ee

# Add shared module to Python path
shared_paths = [
    os.path.join(os.path.dirname(__file__), '..', 'shared'),
    '/app/shared',
]
for path in shared_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

logger = logging.getLogger(__name__)


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions.
    
    PSI measures the shift in distribution between expected (training) and actual (recent) data.
    
    Interpretation:
    - PSI < 0.1: No significant shift
    - 0.1 <= PSI < 0.25: Moderate shift, monitor closely
    - PSI >= 0.25: Significant shift, action required
    
    Args:
        expected: Training/reference distribution values
        actual: Recent/production distribution values
        bins: Number of bins for discretization
    
    Returns:
        PSI value (0 = identical distributions, higher = more shift)
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    # Create bins based on expected distribution
    _, bin_edges = np.histogram(expected, bins=bins)
    
    # Extend edges to handle values outside training range
    bin_edges[0] = min(bin_edges[0], np.min(actual)) - 1e-6
    bin_edges[-1] = max(bin_edges[-1], np.max(actual)) + 1e-6
    
    # Calculate proportions in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Convert to proportions with small epsilon to avoid division by zero
    epsilon = 1e-6
    expected_pct = (expected_counts + epsilon) / (len(expected) + epsilon * bins)
    actual_pct = (actual_counts + epsilon) / (len(actual) + epsilon * bins)
    
    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return float(psi)


def compute_js_divergence(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    JS divergence is a symmetric, bounded measure of distribution similarity.
    
    Interpretation (JS divergence, not distance):
    - JS < 0.1: Very similar distributions
    - 0.1 <= JS < 0.2: Moderate difference
    - JS >= 0.2: Significant difference
    
    Args:
        expected: Training/reference distribution values
        actual: Recent/production distribution values
        bins: Number of bins for discretization
    
    Returns:
        JS divergence value (0 = identical, max = ln(2) ≈ 0.693)
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    # Create common bins spanning both distributions
    combined = np.concatenate([expected, actual])
    _, bin_edges = np.histogram(combined, bins=bins)
    
    # Calculate probability distributions
    expected_hist, _ = np.histogram(expected, bins=bin_edges, density=True)
    actual_hist, _ = np.histogram(actual, bins=bin_edges, density=True)
    
    # Normalize to proper probability distributions
    expected_prob = expected_hist / (expected_hist.sum() + 1e-10)
    actual_prob = actual_hist / (actual_hist.sum() + 1e-10)
    
    # scipy's jensenshannon returns the JS distance (sqrt of divergence)
    # We square it to get the divergence
    js_distance = jensenshannon(expected_prob, actual_prob)
    js_divergence = js_distance ** 2
    
    return float(js_divergence)


def load_config() -> Dict[str, Any]:
    """Load configuration from Cloud Storage or local file."""
    try:
        client = storage.Client()
        bucket = client.bucket('carboncheck-data')
        blob = bucket.blob('config/config.yaml')
        yaml_content = blob.download_as_text()
        logger.info("✅ Config loaded from Cloud Storage")
        return yaml.safe_load(yaml_content)
    except Exception as e:
        logger.warning(f"⚠️  Could not load from GCS: {e}")
        config_path = os.path.join(os.path.dirname(__file__), '..', 'orchestrator', 'config.yaml')
        with open(config_path, 'r') as f:
            logger.info("✅ Config loaded from local file")
            return yaml.safe_load(f)


def initialize_earth_engine(project_id: str):
    """Initialize Earth Engine with the project."""
    try:
        try:
            ee.Initialize(project=project_id)
            logger.info("✅ Earth Engine initialized with project")
        except Exception:
            ee.Initialize()
            logger.info("✅ Earth Engine initialized with default credentials")
    except Exception as e:
        logger.error(f"❌ Earth Engine initialization failed: {e}")
        raise


def collect_recent_ee_samples(config: Dict[str, Any], sample_fraction: float = 0.1) -> pd.DataFrame:
    """
    Collect recent UNLABELED samples from Earth Engine.
    Samples from the same regions used in training but without crop labels.
    
    Args:
        config: Pipeline configuration
        sample_fraction: Fraction of training data to collect (default 10%)
    
    Returns:
        DataFrame with NDVI features only (no labels)
    """
    from earth_engine_features import compute_ndvi_features_ee_as_feature
    
    project_id = config['project']['id']
    initialize_earth_engine(project_id)
    
    # Calculate total sample count (10% of training config)
    num_fields_per_crop = config.get('data_collection', {}).get('num_fields_per_crop', 300)
    num_other_fields = config.get('data_collection', {}).get('num_other_fields', 300)
    crops = config['data_collection']['crops']
    total_training_samples = (num_fields_per_crop * len(crops) + num_other_fields) * 3
    target_samples = int(total_training_samples * sample_fraction)
    
    cdl_year = config.get('data_collection', {}).get('cdl_year', 2024)
    
    logger.info(f"🌍 Collecting recent EE samples (label-free)")
    logger.info(f"   Target samples: {target_samples}")
    logger.info(f"   CDL year (for NDVI date range): {cdl_year}")
    
    all_samples = []
    
    # Get all unique counties from crop config (sample from same regions)
    all_counties = []
    for crop in crops:
        all_counties.extend(crop.get('counties', []))
    unique_counties = list(set(all_counties))
    
    counties_fc = ee.FeatureCollection('TIGER/2018/Counties')
    samples_per_county = max(10, target_samples // len(unique_counties))
    
    logger.info(f"   Sampling from {len(unique_counties)} counties")
    
    for county_geoid in unique_counties:
        try:
            county_region = counties_fc.filter(ee.Filter.eq('GEOID', county_geoid)).geometry()
            
            # Sample random points from the county (no crop filtering)
            # Use a simple land mask (any non-water area)
            land_mask = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2019').select('landcover')
            land_mask = land_mask.updateMask(land_mask.neq(11))  # Exclude open water
            
            county_samples = land_mask.sample(
                region=county_region,
                scale=30,
                numPixels=samples_per_county * 20,  # Oversample
                seed=42,
                geometries=True,
                tileScale=16
            ).limit(samples_per_county)
            
            # Extract NDVI features (no crop label)
            def extract_features(feature):
                return compute_ndvi_features_ee_as_feature(
                    feature.geometry(), cdl_year
                )
            
            county_samples = county_samples.map(extract_features)
            
            # Convert to pandas
            features_list = county_samples.getInfo()['features']
            for f in features_list:
                props = f['properties']
                # Only keep NDVI features, no labels
                all_samples.append(props)
                
        except Exception as e:
            logger.warning(f"   ⚠️  Error sampling county {county_geoid}: {e}")
            continue
    
    df = pd.DataFrame(all_samples)
    logger.info(f"   ✅ Collected {len(df)} samples from Earth Engine (no labels)")
    
    if len(df) == 0:
        logger.warning("   ⚠️  No samples collected! Creating empty DataFrame")
        df = pd.DataFrame(columns=['ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max',
                                   'ndvi_p25', 'ndvi_p50', 'ndvi_p75', 
                                   'ndvi_early', 'ndvi_late'])
    
    return df


def load_training_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load training data from BigQuery (features only, labels ignored for skew detection)."""
    project_id = config['project']['id']
    dataset_id = config['bigquery']['dataset']
    table_id = config['bigquery']['tables']['training']
    
    logger.info(f"📊 Loading training data from BigQuery...")
    logger.info(f"   Table: {project_id}.{dataset_id}.{table_id}")
    
    client = bigquery.Client(project=project_id)
    
    # Select only feature columns (no labels needed for drift detection)
    query = f"""
    SELECT 
        ndvi_mean, ndvi_std, ndvi_min, ndvi_max,
        ndvi_p25, ndvi_p50, ndvi_p75,
        ndvi_early, ndvi_late
    FROM `{project_id}.{dataset_id}.{table_id}`
    """
    
    df = client.query(query).to_dataframe()
    logger.info(f"   ✅ Loaded {len(df)} training samples (features only)")
    
    return df


def get_endpoint_predictions_with_confidence(
    config: Dict[str, Any], 
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Get predictions from Vertex AI endpoint with confidence scores.
    
    Returns:
        Tuple of (DataFrame with predictions, confidence metrics dict)
    """
    project_id = config['project']['id']
    region = config['project']['region']
    endpoint_name = config['model']['endpoint_name']
    
    logger.info(f"🔮 Getting predictions with confidence from endpoint...")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Find endpoint
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    if not endpoints:
        raise ValueError(f"Endpoint '{endpoint_name}' not found")
    
    endpoint = endpoints[0]
    logger.info(f"   Endpoint: {endpoint.display_name}")
    
    # Feature columns
    feature_cols = [
        'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max',
        'ndvi_p25', 'ndvi_p50', 'ndvi_p75',
        'ndvi_early', 'ndvi_late'
    ]
    
    # Engineer additional features
    df = df.copy()
    df['ndvi_range'] = df['ndvi_max'] - df['ndvi_min']
    df['ndvi_iqr'] = df['ndvi_p75'] - df['ndvi_p25']
    df['ndvi_change'] = df['ndvi_late'] - df['ndvi_early']
    df['ndvi_early_ratio'] = df['ndvi_early'] / (df['ndvi_mean'] + 1e-6)
    df['ndvi_late_ratio'] = df['ndvi_late'] / (df['ndvi_mean'] + 1e-6)
    
    engineered_cols = ['ndvi_range', 'ndvi_iqr', 'ndvi_change', 'ndvi_early_ratio', 'ndvi_late_ratio']
    all_feature_cols = feature_cols + engineered_cols
    
    # Filter to available columns
    available_cols = [c for c in all_feature_cols if c in df.columns]
    
    # Prepare features
    X = df[available_cols].fillna(0).values.tolist()
    
    # Batch predictions
    predictions = []
    batch_size = 100
    
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        try:
            response = endpoint.predict(instances=batch)
            predictions.extend(response.predictions)
        except Exception as e:
            logger.warning(f"   ⚠️  Batch {i//batch_size} failed: {e}")
            predictions.extend(['Unknown'] * len(batch))
    
    df['predicted_class'] = predictions
    
    # Calculate prediction distribution (for drift detection)
    pred_counts = df['predicted_class'].value_counts(normalize=True)
    
    # Note: sklearn models don't return probabilities by default
    # We'll use prediction distribution as a proxy for confidence
    confidence_metrics = {
        'prediction_entropy': float(stats.entropy(pred_counts.values)) if len(pred_counts) > 1 else 0.0,
        'num_unique_predictions': int(len(pred_counts)),
        'most_common_prediction': str(pred_counts.index[0]) if len(pred_counts) > 0 else 'Unknown',
        'most_common_pct': float(pred_counts.values[0]) if len(pred_counts) > 0 else 0.0
    }
    
    logger.info(f"   ✅ Got predictions for {len(df)} samples")
    logger.info(f"   Prediction entropy: {confidence_metrics['prediction_entropy']:.4f}")
    
    return df, confidence_metrics


def compute_label_free_skew_metrics(
    training_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    training_pred_metrics: Dict[str, float],
    recent_pred_metrics: Dict[str, float],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute label-free skew metrics between training and recent data.
    
    Returns dict with:
    - feature_skew: KS test statistics for each NDVI feature
    - prediction_entropy_drift: Change in prediction entropy
    - prediction_distribution_drift: How model output distribution changed
    - alerts: List of detected issues
    """
    logger.info("📈 Computing label-free skew metrics...")
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'training_samples': len(training_df),
        'recent_samples': len(recent_df),
        'alerts': []
    }
    
    # Handle empty recent DataFrame
    if len(recent_df) == 0:
        logger.warning("   ⚠️  No recent samples to compare - skipping skew analysis")
        metrics['alerts'].append("No recent samples collected from Earth Engine")
        metrics['feature_skew'] = {}
        metrics['prediction_metrics'] = {}
        metrics['needs_attention'] = True
        return metrics
    
    # 1. Feature Distribution Skew (KS test) - NO LABELS NEEDED
    logger.info("   Analyzing feature distributions (label-free)...")
    feature_cols = ['ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max',
                    'ndvi_p25', 'ndvi_p50', 'ndvi_p75', 'ndvi_early', 'ndvi_late']
    
    feature_skew = {}
    significant_drifts = 0
    
    for col in feature_cols:
        if col in training_df.columns and col in recent_df.columns:
            train_vals = training_df[col].dropna().values
            recent_vals = recent_df[col].dropna().values
            
            if len(train_vals) > 0 and len(recent_vals) > 0:
                ks_stat, ks_pval = stats.ks_2samp(train_vals, recent_vals)
                
                # Calculate PSI and JS divergence
                psi = compute_psi(train_vals, recent_vals)
                js_div = compute_js_divergence(train_vals, recent_vals)
                
                # Calculate distribution statistics
                train_mean = float(np.mean(train_vals))
                train_std = float(np.std(train_vals))
                recent_mean = float(np.mean(recent_vals))
                recent_std = float(np.std(recent_vals))
                
                # Normalized mean shift
                mean_shift = abs(recent_mean - train_mean) / (train_std + 1e-6)
                
                # === NEW: Fisher-Pearson Skewness & Kurtosis ===
                # Skewness: 0 = symmetric, >0 = right tail, <0 = left tail
                train_skewness = float(stats.skew(train_vals))
                recent_skewness = float(stats.skew(recent_vals))
                skewness_drift = abs(recent_skewness - train_skewness)
                
                # Kurtosis: 0 = normal, >0 = heavy tails, <0 = light tails
                train_kurtosis = float(stats.kurtosis(train_vals))
                recent_kurtosis = float(stats.kurtosis(recent_vals))
                kurtosis_drift = abs(recent_kurtosis - train_kurtosis)
                
                # === NEW: Tail Monitoring (L-infinity / Chebyshev) ===
                # Monitor 1st and 99th percentiles for tail drift
                train_p1 = float(np.percentile(train_vals, 1))
                train_p99 = float(np.percentile(train_vals, 99))
                recent_p1 = float(np.percentile(recent_vals, 1))
                recent_p99 = float(np.percentile(recent_vals, 99))
                
                # Normalize tail drift by training IQR for comparability
                train_iqr = np.percentile(train_vals, 75) - np.percentile(train_vals, 25)
                train_iqr = train_iqr if train_iqr > 1e-6 else 1e-6
                
                left_tail_drift = abs(recent_p1 - train_p1) / train_iqr
                right_tail_drift = abs(recent_p99 - train_p99) / train_iqr
                max_tail_drift = max(left_tail_drift, right_tail_drift)
                
                # Determine significance using multiple metrics
                # KS: > 0.15 with p < 0.05
                # PSI: >= 0.25 indicates significant shift
                # JS: >= 0.2 indicates significant difference
                # Skewness drift: > 0.5 indicates asymmetry change
                # Tail drift: > 0.5 IQR indicates tail movement
                ks_significant = ks_stat > 0.15 and ks_pval < 0.05
                psi_significant = psi >= 0.25
                js_significant = js_div >= 0.2
                skewness_significant = skewness_drift > 0.5
                tail_significant = max_tail_drift > 0.5
                
                feature_skew[col] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(ks_pval),
                    'psi': float(psi),
                    'js_divergence': float(js_div),
                    'train_mean': train_mean,
                    'train_std': train_std,
                    'recent_mean': recent_mean,
                    'recent_std': recent_std,
                    'normalized_mean_shift': float(mean_shift),
                    # NEW: Skewness metrics
                    'train_skewness': train_skewness,
                    'recent_skewness': recent_skewness,
                    'skewness_drift': skewness_drift,
                    # NEW: Kurtosis metrics
                    'train_kurtosis': train_kurtosis,
                    'recent_kurtosis': recent_kurtosis,
                    'kurtosis_drift': kurtosis_drift,
                    # NEW: Tail metrics
                    'train_p1': train_p1,
                    'train_p99': train_p99,
                    'recent_p1': recent_p1,
                    'recent_p99': recent_p99,
                    'left_tail_drift': float(left_tail_drift),
                    'right_tail_drift': float(right_tail_drift),
                    'max_tail_drift': float(max_tail_drift),
                    # Significance flags
                    'ks_significant': ks_significant,
                    'psi_significant': psi_significant,
                    'js_significant': js_significant,
                    'skewness_significant': skewness_significant,
                    'tail_significant': tail_significant,
                    'significant': ks_significant or psi_significant or js_significant or skewness_significant or tail_significant
                }
                
                if feature_skew[col]['significant']:
                    significant_drifts += 1
                    alert_parts = [f"Feature drift: {col}"]
                    if ks_significant:
                        alert_parts.append(f"KS={ks_stat:.3f}")
                    if psi_significant:
                        alert_parts.append(f"PSI={psi:.3f}")
                    if js_significant:
                        alert_parts.append(f"JS={js_div:.3f}")
                    if skewness_significant:
                        alert_parts.append(f"SKEW={skewness_drift:.2f}")
                    if tail_significant:
                        alert_parts.append(f"TAIL={max_tail_drift:.2f}")
                    alert_parts.append(f"shift={mean_shift:.2f}σ")
                    metrics['alerts'].append(" ".join(alert_parts))
    
    metrics['feature_skew'] = feature_skew
    metrics['significant_feature_drifts'] = significant_drifts
    
    # 2. Prediction Entropy Drift - NO LABELS NEEDED
    logger.info("   Analyzing prediction entropy drift...")
    train_entropy = training_pred_metrics.get('prediction_entropy', 0)
    recent_entropy = recent_pred_metrics.get('prediction_entropy', 0)
    entropy_drift = recent_entropy - train_entropy
    
    metrics['prediction_metrics'] = {
        'training_entropy': float(train_entropy),
        'recent_entropy': float(recent_entropy),
        'entropy_drift': float(entropy_drift),
        'training_unique_predictions': training_pred_metrics.get('num_unique_predictions', 0),
        'recent_unique_predictions': recent_pred_metrics.get('num_unique_predictions', 0)
    }
    
    # High entropy increase = model is less certain on new data
    if entropy_drift > 0.3:
        metrics['alerts'].append(
            f"Prediction entropy increased: {train_entropy:.3f} → {recent_entropy:.3f} (model less certain)"
        )
    
    # 3. Summary metrics
    avg_ks = np.mean([f['ks_statistic'] for f in feature_skew.values()]) if feature_skew else 0
    max_ks = max([f['ks_statistic'] for f in feature_skew.values()]) if feature_skew else 0
    avg_psi = np.mean([f['psi'] for f in feature_skew.values()]) if feature_skew else 0
    max_psi = max([f['psi'] for f in feature_skew.values()]) if feature_skew else 0
    avg_js = np.mean([f['js_divergence'] for f in feature_skew.values()]) if feature_skew else 0
    max_js = max([f['js_divergence'] for f in feature_skew.values()]) if feature_skew else 0
    
    # NEW: Skewness and Tail summary metrics
    avg_skewness_drift = np.mean([f['skewness_drift'] for f in feature_skew.values()]) if feature_skew else 0
    max_skewness_drift = max([f['skewness_drift'] for f in feature_skew.values()]) if feature_skew else 0
    avg_kurtosis_drift = np.mean([f['kurtosis_drift'] for f in feature_skew.values()]) if feature_skew else 0
    max_kurtosis_drift = max([f['kurtosis_drift'] for f in feature_skew.values()]) if feature_skew else 0
    avg_tail_drift = np.mean([f['max_tail_drift'] for f in feature_skew.values()]) if feature_skew else 0
    max_tail_drift = max([f['max_tail_drift'] for f in feature_skew.values()]) if feature_skew else 0
    
    # Count significant by type
    skewness_significant_count = sum(1 for f in feature_skew.values() if f.get('skewness_significant', False))
    tail_significant_count = sum(1 for f in feature_skew.values() if f.get('tail_significant', False))
    
    metrics['summary'] = {
        'avg_feature_ks': float(avg_ks),
        'max_feature_ks': float(max_ks),
        'avg_feature_psi': float(avg_psi),
        'max_feature_psi': float(max_psi),
        'avg_feature_js': float(avg_js),
        'max_feature_js': float(max_js),
        # NEW: Skewness metrics
        'avg_skewness_drift': float(avg_skewness_drift),
        'max_skewness_drift': float(max_skewness_drift),
        'avg_kurtosis_drift': float(avg_kurtosis_drift),
        'max_kurtosis_drift': float(max_kurtosis_drift),
        # NEW: Tail metrics
        'avg_tail_drift': float(avg_tail_drift),
        'max_tail_drift': float(max_tail_drift),
        # Counts
        'significant_drift_count': significant_drifts,
        'skewness_significant_count': skewness_significant_count,
        'tail_significant_count': tail_significant_count,
        'total_features_tested': len(feature_skew)
    }
    
    # Determine if attention is needed (based on label-free metrics only)
    metrics['needs_attention'] = (
        significant_drifts >= 3 or  # Multiple features drifting
        max_ks > 0.3 or  # Severe drift in any feature
        entropy_drift > 0.5  # Major uncertainty increase
    )
    
    logger.info(f"   ✅ Computed label-free skew metrics")
    logger.info(f"   KS  - Avg: {avg_ks:.4f}, Max: {max_ks:.4f}")
    logger.info(f"   PSI - Avg: {avg_psi:.4f}, Max: {max_psi:.4f}")
    logger.info(f"   JS  - Avg: {avg_js:.4f}, Max: {max_js:.4f}")
    logger.info(f"   SKEW - Avg: {avg_skewness_drift:.4f}, Max: {max_skewness_drift:.4f}")
    logger.info(f"   TAIL - Avg: {avg_tail_drift:.4f}, Max: {max_tail_drift:.4f}")
    logger.info(f"   Significant drifts: {significant_drifts}/{len(feature_skew)}")
    logger.info(f"   Skewness alerts: {skewness_significant_count}, Tail alerts: {tail_significant_count}")
    logger.info(f"   Entropy drift: {entropy_drift:+.4f}")
    logger.info(f"   Needs attention: {metrics['needs_attention']}")
    
    return metrics


def compute_drift_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a composite drift score (0-100) from multiple metrics.
    
    SKEW-FOCUSED WEIGHTING (for outlier-sensitive models like crop yield):
    - Skewness & Kurtosis (asymmetry detection) - weighted 40%
    - PSI & Distribution drift - weighted 40%  
    - Tail & Other metrics - weighted 20%
    
    This weighting prioritizes detecting when data becomes asymmetric
    or develops long tails, which can break models assuming normality.
    
    Returns:
        Dict with drift_score, component scores, and retraining recommendation
    """
    summary = metrics.get('summary', {})
    feature_skew = metrics.get('feature_skew', {})
    pred_metrics = metrics.get('prediction_metrics', {})
    
    # === 1. SKEWNESS & KURTOSIS SCORE (40% weight) ===
    # This is the PRIMARY indicator of asymmetry problems
    max_skewness_drift = summary.get('max_skewness_drift', 0)
    max_kurtosis_drift = summary.get('max_kurtosis_drift', 0)
    
    # Skewness drift: 0.0 = 0, 1.0+ = 100 (higher = more asymmetric change)
    skewness_score = min(100, max_skewness_drift * 100)
    # Kurtosis drift: 0.0 = 0, 2.0+ = 100 (higher = tail weight changed)
    kurtosis_score = min(100, (max_kurtosis_drift / 2.0) * 100)
    
    # Combined skew/kurtosis score (take max - worst case)
    skew_kurtosis_score = max(skewness_score, kurtosis_score)
    
    # === 2. PSI & DISTRIBUTION DRIFT SCORE (40% weight) ===
    max_ks = summary.get('max_feature_ks', 0)
    max_psi = summary.get('max_feature_psi', 0)
    max_js = summary.get('max_feature_js', 0)
    
    # Normalize each to 0-100 scale
    ks_score = min(100, (max_ks / 0.3) * 100)
    psi_score = min(100, (max_psi / 0.5) * 100)
    js_score = min(100, (max_js / 0.4) * 100)
    
    # Take max of the three (most severe)
    distribution_drift_score = max(ks_score, psi_score, js_score)
    
    # === 3. TAIL & OTHER METRICS (20% weight) ===
    # Tail drift - critical for detecting outlier accumulation
    max_tail_drift = summary.get('max_tail_drift', 0)
    # Tail drift: 0.0 = 0, 1.0 IQR+ = 100
    tail_score = min(100, max_tail_drift * 100)
    
    # Entropy drift
    entropy_drift = abs(pred_metrics.get('entropy_drift', 0))
    entropy_score = min(100, entropy_drift * 100)
    
    # Mean shift
    if feature_skew:
        max_shift = max([f.get('normalized_mean_shift', 0) for f in feature_skew.values()])
    else:
        max_shift = 0
    shift_score = min(100, (max_shift / 2.0) * 100)
    
    # Combined tail/other score
    tail_other_score = max(tail_score, entropy_score, shift_score)
    
    # === 4. COMPOSITE SCORE (Skew-Focused Weighting) ===
    drift_score = (
        skew_kurtosis_score * 0.40 +      # 40% - Asymmetry detection
        distribution_drift_score * 0.40 +  # 40% - Population stability
        tail_other_score * 0.20            # 20% - Tail/entropy/shift
    )
    
    # 5. Determine retraining recommendation
    if drift_score >= 70:
        recommendation = "YES - Immediate"
        severity = "critical"
    elif drift_score >= 50:
        recommendation = "YES - Soon"
        severity = "warning"
    elif drift_score >= 30:
        recommendation = "MONITOR"
        severity = "caution"
    else:
        recommendation = "NO"
        severity = "ok"
    
    return {
        'drift_score': round(drift_score, 1),
        # NEW: Skew-focused component scores
        'skew_kurtosis_score': round(skew_kurtosis_score, 1),
        'distribution_drift_score': round(distribution_drift_score, 1),
        'tail_other_score': round(tail_other_score, 1),
        # Detailed sub-scores
        'skewness_score': round(skewness_score, 1),
        'kurtosis_score': round(kurtosis_score, 1),
        'tail_score': round(tail_score, 1),
        'entropy_score': round(entropy_score, 1),
        'shift_score': round(shift_score, 1),
        'psi_score': round(psi_score, 1),
        # Recommendation
        'retraining_needed': recommendation,
        'severity': severity,
        # NEW: Skew-focused weighting
        'component_weights': {
            'skew_kurtosis': 0.40,
            'distribution': 0.40,
            'tail_other': 0.20
        }
    }


def create_drift_summary_table(
    metrics: Dict[str, Any],
    drift_score_info: Dict[str, Any]
):
    """
    Create a visual summary table of drift metrics with color-coded cells,
    thresholds, and brief method descriptions.
    
    Returns:
        matplotlib.figure.Figure
    
    Args:
        metrics: Full metrics dict from compute_label_free_skew_metrics
        drift_score_info: Output from compute_drift_score
    
    Returns:
        Matplotlib figure with formatted table
    """
    import matplotlib.pyplot as plt
    
    summary = metrics.get('summary', {})
    pred_metrics = metrics.get('prediction_metrics', {})
    feature_skew = metrics.get('feature_skew', {})
    
    # Calculate additional derived metrics
    max_shift = 0
    if feature_skew:
        max_shift = max([f.get('normalized_mean_shift', 0) for f in feature_skew.values()])
    
    std_change_pct = 0
    if feature_skew:
        std_changes = []
        for f in feature_skew.values():
            train_std = f.get('train_std', 1)
            recent_std = f.get('recent_std', 1)
            if train_std > 0:
                std_changes.append(abs(recent_std - train_std) / train_std * 100)
        if std_changes:
            std_change_pct = max(std_changes)
    
    def get_status(value, thresholds):
        """Get status color based on thresholds (ok, monitor)"""
        if value < thresholds[0]:
            return ('OK', 'lightgreen')
        elif value < thresholds[1]:
            return ('Monitor', 'khaki')
        else:
            return ('Alert', 'lightcoral')
    
    # Table rows: (Metric, Value, Threshold, Status, Color, Description)
    table_data = []
    
    # === SKEWNESS METRICS (Primary - 40% weight) ===
    # Fisher-Pearson Skewness Drift
    max_skewness = summary.get('max_skewness_drift', 0)
    status, color = get_status(max_skewness, (0.3, 0.5))
    table_data.append({
        'metric': '⚡ Skewness Drift',
        'value': f'{max_skewness:.3f}',
        'threshold': '<0.30 | <0.50',
        'status': status,
        'color': color,
        'desc': 'Asymmetry change (Fisher-Pearson)'
    })
    
    # Kurtosis Drift
    max_kurtosis = summary.get('max_kurtosis_drift', 0)
    status, color = get_status(max_kurtosis, (0.5, 1.0))
    table_data.append({
        'metric': '⚡ Kurtosis Drift',
        'value': f'{max_kurtosis:.3f}',
        'threshold': '<0.50 | <1.00',
        'status': status,
        'color': color,
        'desc': 'Tail heaviness change'
    })
    
    # Tail Drift (L-infinity)
    max_tail = summary.get('max_tail_drift', 0)
    status, color = get_status(max_tail, (0.3, 0.5))
    table_data.append({
        'metric': '⚡ Tail Drift',
        'value': f'{max_tail:.3f}',
        'threshold': '<0.30 | <0.50',
        'status': status,
        'color': color,
        'desc': 'P1/P99 percentile shift (IQR)'
    })
    
    # === DISTRIBUTION METRICS (40% weight) ===
    # PSI
    max_psi = summary.get('max_feature_psi', 0)
    status, color = get_status(max_psi, (0.1, 0.25))
    table_data.append({
        'metric': 'PSI',
        'value': f'{max_psi:.3f}',
        'threshold': '<0.10 | <0.25',
        'status': status,
        'color': color,
        'desc': 'Population stability via binning'
    })
    
    # KS Statistic
    max_ks = summary.get('max_feature_ks', 0)
    status, color = get_status(max_ks, (0.15, 0.25))
    table_data.append({
        'metric': 'KS Statistic',
        'value': f'{max_ks:.3f}',
        'threshold': '<0.15 | <0.25',
        'status': status,
        'color': color,
        'desc': 'Max distance between CDFs'
    })
    
    # JS Divergence
    max_js = summary.get('max_feature_js', 0)
    status, color = get_status(max_js, (0.1, 0.2))
    table_data.append({
        'metric': 'JS Divergence',
        'value': f'{max_js:.3f}',
        'threshold': '<0.10 | <0.20',
        'status': status,
        'color': color,
        'desc': 'Symmetric KL divergence'
    })
    
    # === OTHER METRICS (20% weight) ===
    # Mean Shift
    status, color = get_status(max_shift, (0.5, 1.0))
    table_data.append({
        'metric': 'Mean Shift',
        'value': f'{max_shift:.2f}σ',
        'threshold': '<0.5σ | <1.0σ',
        'status': status,
        'color': color,
        'desc': 'Shift in std deviations'
    })
    
    # Std Change
    status, color = get_status(std_change_pct, (25, 50))
    table_data.append({
        'metric': 'Std Change',
        'value': f'{std_change_pct:.1f}%',
        'threshold': '<25% | <50%',
        'status': status,
        'color': color,
        'desc': 'Variance change %'
    })
    
    # Entropy Drift
    entropy_drift = pred_metrics.get('entropy_drift', 0)
    status, color = get_status(abs(entropy_drift), (0.2, 0.5))
    table_data.append({
        'metric': 'Entropy Drift',
        'value': f'{entropy_drift:+.3f}',
        'threshold': '<0.20 | <0.50',
        'status': status,
        'color': color,
        'desc': 'Model uncertainty change'
    })
    
    # Create figure with two sections
    fig, (ax_table, ax_score) = plt.subplots(2, 1, figsize=(12, 8), 
                                              gridspec_kw={'height_ratios': [3, 1]})
    
    # Main metrics table
    ax_table.axis('off')
    ax_table.set_title('Drift Detection Summary', fontsize=16, fontweight='bold', pad=10)
    
    # Build table data
    cell_text = []
    cell_colors = []
    for row in table_data:
        cell_text.append([row['metric'], row['value'], row['threshold'], row['status'], row['desc']])
        cell_colors.append(['white', 'white', 'white', row['color'], 'whitesmoke'])
    
    table = ax_table.table(
        cellText=cell_text,
        colLabels=['Metric', 'Value', 'Thresholds (OK|Monitor)', 'Status', 'Description'],
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
        colColours=['lightsteelblue'] * 5
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    
    # Set custom column widths - make description column wider
    col_widths = [0.12, 0.08, 0.18, 0.08, 0.35]  # Metric, Value, Threshold, Status, Description
    for i, width in enumerate(col_widths):
        for row in range(len(cell_text) + 1):  # +1 for header
            table[(row, i)].set_width(width)
    
    # Bold header and style cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
        # Left-align description column
        if col == 4 and row > 0:
            cell.set_text_props(ha='left')
    
    # Drift Score section
    ax_score.axis('off')
    
    drift_score = drift_score_info['drift_score']
    recommendation = drift_score_info['retraining_needed']
    
    # Score color
    if drift_score < 30:
        score_color = 'green'
    elif drift_score < 50:
        score_color = 'goldenrod'
    elif drift_score < 70:
        score_color = 'orange'
    else:
        score_color = 'red'
    
    # Draw score box
    score_text = f"DRIFT SCORE: {drift_score:.0f}/100"
    ax_score.text(0.25, 0.6, score_text, fontsize=18, fontweight='bold', 
                  ha='center', va='center', color=score_color,
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=score_color, linewidth=2))
    
    # Recommendation color
    if 'NO' in recommendation:
        rec_color = 'green'
    elif 'MONITOR' in recommendation:
        rec_color = 'goldenrod'
    else:
        rec_color = 'red'
    
    rec_text = f"RETRAINING: {recommendation}"
    ax_score.text(0.75, 0.6, rec_text, fontsize=18, fontweight='bold',
                  ha='center', va='center', color=rec_color,
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=rec_color, linewidth=2))
    
    # Score breakdown (using skew-focused weights)
    breakdown = (f"Score = Skew/Kurtosis ({drift_score_info.get('skew_kurtosis_score', 0):.0f}) x 40% + "
                f"Distribution ({drift_score_info.get('distribution_drift_score', 0):.0f}) x 40% + "
                f"Tail/Other ({drift_score_info.get('tail_other_score', 0):.0f}) x 20%")
    ax_score.text(0.5, 0.15, breakdown, fontsize=9, ha='center', va='center', style='italic')
    
    # Timestamp
    timestamp = metrics.get('timestamp', 'N/A')
    fig.text(0.5, 0.01, f'Generated: {timestamp}', ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    return fig


def log_skew_to_tensorboard(
    metrics: Dict[str, Any],
    training_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    tensorboard_resource: Optional[str],
    config: Dict[str, Any],
    local_tb_dir: Optional[str] = None,
    vertex_ai_mode: bool = False
) -> str:
    """
    Log label-free skew metrics and visualizations to TensorBoard.
    Uses PyTorch's SummaryWriter with add_figure() for proper Vertex AI TensorBoard compatibility.
    (Matches trainer/tensorboard_logging.py approach)
    
    Args:
        metrics: Skew audit metrics dict
        training_df: Training data DataFrame
        recent_df: Recent samples DataFrame
        tensorboard_resource: TensorBoard resource name (can be None in Vertex AI mode)
        config: Pipeline configuration
        local_tb_dir: Optional local directory for TensorBoard logs. If None, creates default.
        vertex_ai_mode: If True, skip manual API upload (Vertex AI handles GCS sync)
    
    Returns:
        Path to TensorBoard logs (GCS path or local path)
    """
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter
    
    project_id = config['project']['id']
    region = config['project']['region']
    
    # Create experiment name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f'skew-audit-{datetime.now().strftime("%Y%m%d")}'
    experiment_id = re.sub(r'[^a-z0-9-]', '-', experiment_name.lower())
    run_id = re.sub(r'[^a-z0-9-]', '-', f'run-{timestamp}'.lower())
    
    # Local TensorBoard directory - use provided or create default
    if local_tb_dir is None:
        local_tb_dir = f'/tmp/tensorboard_logs/{experiment_id}/{run_id}'
    os.makedirs(local_tb_dir, exist_ok=True)
    
    logger.info(f"📊 Logging skew metrics to TensorBoard...")
    logger.info(f"   Experiment: {experiment_id}")
    logger.info(f"   Run: {run_id}")
    logger.info(f"   Local dir: {local_tb_dir}")
    if vertex_ai_mode:
        logger.info(f"   Mode: Vertex AI (GCS sync handled automatically)")
    
    # Use PyTorch's SummaryWriter (same as trainer)
    writer = SummaryWriter(log_dir=local_tb_dir)
    logger.info(f"   ✅ SummaryWriter created")
    
    try:
        feature_skew = metrics.get('feature_skew', {})
        
        # 1. Feature Distribution Comparisons (Histograms)
        if feature_skew and len(recent_df) > 0:
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.flatten()
            
            for idx, (feature, stats_dict) in enumerate(list(feature_skew.items())[:9]):
                ax = axes[idx]
                
                train_vals = training_df[feature].dropna()
                recent_vals = recent_df[feature].dropna()
                
                ax.hist(train_vals, bins=30, alpha=0.5, label='Training', color='steelblue', density=True)
                ax.hist(recent_vals, bins=30, alpha=0.5, label='Recent', color='coral', density=True)
                
                drift_status = "DRIFT" if stats_dict['significant'] else "OK"
                ax.set_title(f'{feature}\nKS={stats_dict["ks_statistic"]:.3f} {drift_status}')
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            
            plt.suptitle('Feature Distribution Comparison (Label-Free)', fontsize=14, y=1.02)
            plt.tight_layout()
            writer.add_figure('skew/feature_distributions', fig, 0, close=True)
            logger.info("   ✅ Logged feature_distributions")
        
        # 2. Multi-Metric Drift Summary (KS, PSI, JS)
        if feature_skew:
            features = list(feature_skew.keys())
            ks_stats = [feature_skew[f]['ks_statistic'] for f in features]
            psi_stats = [feature_skew[f]['psi'] for f in features]
            js_stats = [feature_skew[f]['js_divergence'] for f in features]
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # KS Statistic
            colors = ['red' if feature_skew[f]['ks_significant'] else 'green' for f in features]
            axes[0].bar(features, ks_stats, color=colors, alpha=0.7)
            axes[0].axhline(y=0.15, color='orange', linestyle='--', label='Threshold (0.15)')
            axes[0].set_ylabel('KS Statistic')
            axes[0].set_title('Kolmogorov-Smirnov Test')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)
            
            # PSI
            colors = ['red' if feature_skew[f]['psi_significant'] else 'orange' if feature_skew[f]['psi'] >= 0.1 else 'green' for f in features]
            axes[1].bar(features, psi_stats, color=colors, alpha=0.7)
            axes[1].axhline(y=0.25, color='red', linestyle='--', label='Significant (0.25)')
            axes[1].axhline(y=0.1, color='orange', linestyle='--', label='Moderate (0.1)')
            axes[1].set_ylabel('PSI')
            axes[1].set_title('Population Stability Index')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
            
            # JS Divergence
            colors = ['red' if feature_skew[f]['js_significant'] else 'orange' if feature_skew[f]['js_divergence'] >= 0.1 else 'green' for f in features]
            axes[2].bar(features, js_stats, color=colors, alpha=0.7)
            axes[2].axhline(y=0.2, color='red', linestyle='--', label='Significant (0.2)')
            axes[2].axhline(y=0.1, color='orange', linestyle='--', label='Moderate (0.1)')
            axes[2].set_ylabel('JS Divergence')
            axes[2].set_title('Jensen-Shannon Divergence')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].legend()
            axes[2].grid(axis='y', alpha=0.3)
            
            plt.suptitle('Feature Drift Detection (Label-Free)\nGreen=OK, Orange=Monitor, Red=Significant', fontsize=12, y=1.02)
            plt.tight_layout()
            writer.add_figure('skew/feature_drift_summary', fig, 0, close=True)
            logger.info("   ✅ Logged feature_drift_summary")
        
        # 3. Normalized Mean Shift Plot
        if feature_skew:
            features = list(feature_skew.keys())
            shifts = [feature_skew[f]['normalized_mean_shift'] for f in features]
            
            fig, ax = plt.subplots(figsize=(12, 4))
            colors = ['red' if s > 1.0 else 'orange' if s > 0.5 else 'green' for s in shifts]
            ax.bar(features, shifts, color=colors, alpha=0.7)
            ax.axhline(y=1.0, color='red', linestyle='--', label='1 sigma shift')
            ax.axhline(y=0.5, color='orange', linestyle='--', label='0.5 sigma shift')
            ax.set_ylabel('Normalized Mean Shift (sigma)')
            ax.set_title('Feature Mean Shift (in Standard Deviations)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            writer.add_figure('skew/mean_shift', fig, 0, close=True)
            logger.info("   ✅ Logged mean_shift")
        
        # 4. NEW: Skewness, Kurtosis, and Tail Drift (Key Asymmetry Indicators)
        if feature_skew:
            features = list(feature_skew.keys())
            skewness_drifts = [feature_skew[f].get('skewness_drift', 0) for f in features]
            kurtosis_drifts = [feature_skew[f].get('kurtosis_drift', 0) for f in features]
            tail_drifts = [feature_skew[f].get('max_tail_drift', 0) for f in features]
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            x = np.arange(len(features))
            
            # Skewness Drift
            colors = ['red' if feature_skew[f].get('skewness_significant', False) else 'orange' if d > 0.3 else 'green' 
                     for f, d in zip(features, skewness_drifts)]
            axes[0].bar(x, skewness_drifts, color=colors, alpha=0.7)
            axes[0].axhline(y=0.5, color='red', linestyle='--', label='Significant (0.5)')
            axes[0].axhline(y=0.3, color='orange', linestyle='--', label='Monitor (0.3)')
            axes[0].set_ylabel('Skewness Drift')
            axes[0].set_title('Fisher-Pearson Skewness Change\n(Detects asymmetry shift - "taffy pull" effect)')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(features, rotation=45, ha='right')
            axes[0].legend(loc='upper right')
            axes[0].grid(axis='y', alpha=0.3)
            
            # Kurtosis Drift
            colors = ['red' if d > 1.0 else 'orange' if d > 0.5 else 'green' for d in kurtosis_drifts]
            axes[1].bar(x, kurtosis_drifts, color=colors, alpha=0.7)
            axes[1].axhline(y=1.0, color='red', linestyle='--', label='Significant (1.0)')
            axes[1].axhline(y=0.5, color='orange', linestyle='--', label='Monitor (0.5)')
            axes[1].set_ylabel('Kurtosis Drift')
            axes[1].set_title('Kurtosis Change (Tail Heaviness)\n(Detects outlier accumulation)')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(features, rotation=45, ha='right')
            axes[1].legend(loc='upper right')
            axes[1].grid(axis='y', alpha=0.3)
            
            # Tail Drift (L-infinity)
            colors = ['red' if feature_skew[f].get('tail_significant', False) else 'orange' if d > 0.3 else 'green'
                     for f, d in zip(features, tail_drifts)]
            axes[2].bar(x, tail_drifts, color=colors, alpha=0.7)
            axes[2].axhline(y=0.5, color='red', linestyle='--', label='Significant (0.5 IQR)')
            axes[2].axhline(y=0.3, color='orange', linestyle='--', label='Monitor (0.3 IQR)')
            axes[2].set_ylabel('Tail Drift (IQR)')
            axes[2].set_title('P1/P99 Percentile Shift\n(Detects edge-case drift that PSI may miss)')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(features, rotation=45, ha='right')
            axes[2].legend(loc='upper right')
            axes[2].grid(axis='y', alpha=0.3)
            
            plt.suptitle('SKEW-AWARE MONITORING (40% of Drift Score)\nGreen=OK, Orange=Monitor, Red=Alert', fontsize=12, y=1.02)
            plt.tight_layout()
            writer.add_figure('skew/skewness_kurtosis_tail', fig, 0, close=True)
            logger.info("   ✅ Logged skewness_kurtosis_tail")
        
        # 5. NEW: Q-Q Plots (Visual Skew Detection)
        if feature_skew and len(recent_df) > 0:
            # Select top 4 features with highest skewness drift for Q-Q plots
            sorted_features = sorted(feature_skew.items(), 
                                    key=lambda x: x[1].get('skewness_drift', 0), 
                                    reverse=True)[:4]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for idx, (feature, stats_dict) in enumerate(sorted_features):
                ax = axes[idx]
                recent_vals = recent_df[feature].dropna().values
                
                if len(recent_vals) > 10:
                    # Q-Q plot against normal distribution
                    stats.probplot(recent_vals, dist="norm", plot=ax)
                    
                    # Add annotations
                    skew_val = stats_dict.get('recent_skewness', 0)
                    kurt_val = stats_dict.get('recent_kurtosis', 0)
                    skew_drift = stats_dict.get('skewness_drift', 0)
                    
                    status = "⚠️ SKEW" if stats_dict.get('skewness_significant', False) else "OK"
                    ax.set_title(f'{feature}\nSkew={skew_val:.2f} Kurt={kurt_val:.2f} Δ={skew_drift:.2f} {status}')
                    
                    # Color the regression line based on skewness
                    if abs(skew_val) > 0.5:
                        ax.get_lines()[1].set_color('red')
                    elif abs(skew_val) > 0.3:
                        ax.get_lines()[1].set_color('orange')
                    else:
                        ax.get_lines()[1].set_color('green')
                
                ax.grid(alpha=0.3)
            
            plt.suptitle('Q-Q Plots (Deviation from Normal Distribution)\nPoints off diagonal = non-normality (long tails, skew)', 
                        fontsize=12, y=1.02)
            plt.tight_layout()
            writer.add_figure('skew/qq_plots', fig, 0, close=True)
            logger.info("   ✅ Logged qq_plots")
        
        # 6. Prediction Entropy Comparison
        pred_metrics = metrics.get('prediction_metrics', {})
        if pred_metrics:
            fig, ax = plt.subplots(figsize=(8, 5))
            categories = ['Training', 'Recent']
            entropies = [pred_metrics.get('training_entropy', 0), 
                        pred_metrics.get('recent_entropy', 0)]
            colors = ['steelblue', 'coral']
            ax.bar(categories, entropies, color=colors, alpha=0.7)
            ax.set_ylabel('Prediction Entropy')
            ax.set_title('Model Uncertainty Comparison\n(Higher entropy = less certain predictions)')
            ax.grid(axis='y', alpha=0.3)
            
            drift = pred_metrics.get('entropy_drift', 0)
            drift_text = f"Entropy drift: {drift:+.4f}"
            if drift > 0.3:
                drift_text += " (model less certain on new data)"
            ax.text(0.5, 0.95, drift_text, transform=ax.transAxes, ha='center', fontsize=10)
            
            plt.tight_layout()
            writer.add_figure('skew/prediction_entropy', fig, 0, close=True)
            logger.info("   ✅ Logged prediction_entropy")
        
        # 5. DRIFT SUMMARY TABLE
        logger.info("   Creating drift summary table...")
        drift_score_info = compute_drift_score(metrics)
        fig = create_drift_summary_table(metrics, drift_score_info)
        writer.add_figure('skew/drift_summary', fig, 0, close=True)
        logger.info("   ✅ Logged drift_summary")
        
        # Store drift score in metrics for later use
        metrics['drift_score_info'] = drift_score_info
        logger.info(f"   Drift Score: {drift_score_info['drift_score']}/100")
        logger.info(f"   Retraining: {drift_score_info['retraining_needed']}")
        
        # 6. Log scalars using PyTorch SummaryWriter
        summary = metrics.get('summary', {})
        writer.add_scalar('skew/avg_ks_statistic', summary.get('avg_feature_ks', 0), 0)
        writer.add_scalar('skew/max_ks_statistic', summary.get('max_feature_ks', 0), 0)
        writer.add_scalar('skew/avg_psi', summary.get('avg_feature_psi', 0), 0)
        writer.add_scalar('skew/max_psi', summary.get('max_feature_psi', 0), 0)
        writer.add_scalar('skew/avg_js_divergence', summary.get('avg_feature_js', 0), 0)
        writer.add_scalar('skew/max_js_divergence', summary.get('max_feature_js', 0), 0)
        writer.add_scalar('skew/significant_drift_count', summary.get('significant_drift_count', 0), 0)
        writer.add_scalar('skew/training_samples', metrics['training_samples'], 0)
        writer.add_scalar('skew/recent_samples', metrics['recent_samples'], 0)
        writer.add_scalar('skew/needs_attention', int(metrics.get('needs_attention', False)), 0)
        
        # Log drift score (with new skew-focused components)
        writer.add_scalar('skew/drift_score', drift_score_info['drift_score'], 0)
        writer.add_scalar('skew/skew_kurtosis_score', drift_score_info.get('skew_kurtosis_score', 0), 0)
        writer.add_scalar('skew/distribution_drift_score', drift_score_info.get('distribution_drift_score', 0), 0)
        writer.add_scalar('skew/tail_other_score', drift_score_info.get('tail_other_score', 0), 0)
        writer.add_scalar('skew/skewness_score', drift_score_info.get('skewness_score', 0), 0)
        writer.add_scalar('skew/kurtosis_score', drift_score_info.get('kurtosis_score', 0), 0)
        writer.add_scalar('skew/tail_score', drift_score_info.get('tail_score', 0), 0)
        writer.add_scalar('skew/entropy_score', drift_score_info.get('entropy_score', 0), 0)
        writer.add_scalar('skew/shift_score', drift_score_info.get('shift_score', 0), 0)
        writer.add_scalar('skew/retraining_needed', 1 if 'YES' in drift_score_info['retraining_needed'] else 0, 0)
        
        # Log skewness and tail metrics
        writer.add_scalar('skew/max_skewness_drift', summary.get('max_skewness_drift', 0), 0)
        writer.add_scalar('skew/max_kurtosis_drift', summary.get('max_kurtosis_drift', 0), 0)
        writer.add_scalar('skew/max_tail_drift', summary.get('max_tail_drift', 0), 0)
        
        if pred_metrics:
            writer.add_scalar('skew/training_entropy', pred_metrics.get('training_entropy', 0), 0)
            writer.add_scalar('skew/recent_entropy', pred_metrics.get('recent_entropy', 0), 0)
            writer.add_scalar('skew/entropy_drift', pred_metrics.get('entropy_drift', 0), 0)
        
        for feature, stats_dict in feature_skew.items():
            writer.add_scalar(f'feature_ks/{feature}', stats_dict['ks_statistic'], 0)
            writer.add_scalar(f'feature_psi/{feature}', stats_dict['psi'], 0)
            writer.add_scalar(f'feature_js/{feature}', stats_dict['js_divergence'], 0)
            writer.add_scalar(f'feature_shift/{feature}', stats_dict['normalized_mean_shift'], 0)
            # NEW: Log skewness and tail metrics per feature
            writer.add_scalar(f'feature_skewness/{feature}', stats_dict.get('skewness_drift', 0), 0)
            writer.add_scalar(f'feature_kurtosis/{feature}', stats_dict.get('kurtosis_drift', 0), 0)
            writer.add_scalar(f'feature_tail/{feature}', stats_dict.get('max_tail_drift', 0), 0)
        
        # 7. Log text summary
        alerts_text = '\n'.join(metrics['alerts']) if metrics['alerts'] else 'No alerts'
        summary_text = f"""
SKEW-AWARE Drift Audit Summary
==============================
Timestamp: {metrics['timestamp']}
Training Samples: {metrics['training_samples']}
Recent Samples: {metrics['recent_samples']}

DRIFT SCORE: {drift_score_info['drift_score']:.1f}/100
RETRAINING: {drift_score_info['retraining_needed']}

=== SKEW-FOCUSED SCORING (for outlier-sensitive models) ===

Score Breakdown:
- Skewness/Kurtosis Score: {drift_score_info.get('skew_kurtosis_score', 0):.1f}/100 (weight: 40%)
- Distribution Drift Score: {drift_score_info.get('distribution_drift_score', 0):.1f}/100 (weight: 40%)
- Tail/Other Score: {drift_score_info.get('tail_other_score', 0):.1f}/100 (weight: 20%)

=== SKEWNESS & ASYMMETRY (PRIMARY - 40%) ===
- Max Skewness Drift: {summary.get('max_skewness_drift', 0):.4f} (threshold: 0.5)
- Max Kurtosis Drift: {summary.get('max_kurtosis_drift', 0):.4f} (threshold: 1.0)
- Max Tail Drift (IQR): {summary.get('max_tail_drift', 0):.4f} (threshold: 0.5)
- Skewness Alerts: {summary.get('skewness_significant_count', 0)}
- Tail Alerts: {summary.get('tail_significant_count', 0)}

=== DISTRIBUTION STABILITY (40%) ===
- Max PSI: {summary.get('max_feature_psi', 0):.4f} (threshold: 0.25)
- Max KS: {summary.get('max_feature_ks', 0):.4f} (threshold: 0.15)
- Max JS: {summary.get('max_feature_js', 0):.4f} (threshold: 0.2)

=== OTHER METRICS (20%) ===
- Entropy Drift: {pred_metrics.get('entropy_drift', 0):+.4f}
- Significant Drifts: {summary.get('significant_drift_count', 0)}/{summary.get('total_features_tested', 0)}

Alerts:
{alerts_text}

Needs Attention: {'YES' if metrics.get('needs_attention') else 'NO'}

NOTE: Skew-focused weighting prioritizes detecting asymmetry changes
that can break models assuming Gaussian distributions.
"""
        writer.add_text('skew/summary', summary_text, 0)
        
        # Flush and close the writer (critical for ensuring all data is written)
        logger.info("   💾 Flushing TensorBoard writer...")
        writer.flush()
        writer.close()
        logger.info("   ✅ TensorBoard writer closed")
        
        # In Vertex AI mode, skip manual upload - Vertex AI handles GCS sync via AIP_TENSORBOARD_LOG_DIR
        if vertex_ai_mode:
            logger.info("   ℹ️  Vertex AI mode: Skipping manual upload (Vertex AI syncs automatically)")
            return local_tb_dir
        
        # Legacy mode: Manual upload to GCS and Vertex AI TensorBoard API
        vertex_bucket = f'{project_id}-training'
        gcs_path = f'gs://{vertex_bucket}/skew_audit/logs/{experiment_id}/{run_id}/'
        
        logger.info(f"   📤 Uploading logs to GCS: {gcs_path}")
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(vertex_bucket)
        
        for root, dirs, files in os.walk(local_tb_dir):
            for file in files:
                local_path = os.path.join(root, file)
                blob_path = f'skew_audit/logs/{experiment_id}/{run_id}/{file}'
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
        
        logger.info(f"   ✅ TensorBoard logs uploaded")
        
        # Initialize Vertex AI Experiment (without TensorBoard to avoid lookup failures)
        try:
            aiplatform.init(
                project=project_id,
                location=region,
                experiment=experiment_id
            )
            aiplatform.start_run(run=run_id)
            
            # Log metrics to Vertex AI Experiments
            aiplatform.log_metrics({
                'avg_ks_statistic': summary.get('avg_feature_ks', 0),
                'max_ks_statistic': summary.get('max_feature_ks', 0),
                'avg_psi': summary.get('avg_feature_psi', 0),
                'max_psi': summary.get('max_feature_psi', 0),
                'avg_js_divergence': summary.get('avg_feature_js', 0),
                'max_js_divergence': summary.get('max_feature_js', 0),
                'significant_drifts': summary.get('significant_drift_count', 0),
                'entropy_drift': pred_metrics.get('entropy_drift', 0),
                'drift_score': drift_score_info['drift_score'],
                'skew_kurtosis_score': drift_score_info.get('skew_kurtosis_score', 0),
                'distribution_drift_score': drift_score_info.get('distribution_drift_score', 0),
                'tail_other_score': drift_score_info.get('tail_other_score', 0),
                'retraining_needed': 1 if 'YES' in drift_score_info['retraining_needed'] else 0,
                'needs_attention': int(metrics.get('needs_attention', False))
            })
            
            aiplatform.end_run()
            logger.info(f"   ✅ Logged to Vertex AI Experiments")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not log to Vertex AI Experiments: {e}")
        
        # Upload to Vertex AI TensorBoard using direct API (legacy - fragile for images)
        if tensorboard_resource:
            logger.info("   📤 Uploading to Vertex AI TensorBoard (legacy API)...")
            try:
                from google.cloud.aiplatform_v1 import TensorboardServiceClient
                from google.cloud.aiplatform_v1.types import (
                    tensorboard_time_series as tts_types,
                    tensorboard_data as td_types,
                    WriteTensorboardRunDataRequest,
                )
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                
                # Parse the tensorboard resource name
                tb_parts = tensorboard_resource.split('/')
                project_num = tb_parts[1]
                tb_region = tb_parts[3]
                tb_id = tb_parts[5]
                
                # Initialize client
                client = TensorboardServiceClient(
                    client_options={"api_endpoint": f"{tb_region}-aiplatform.googleapis.com"}
                )
                
                # Create/get run resource
                run_resource = f"{tensorboard_resource}/experiments/{experiment_id}/runs/{run_id}"
                
                # Load tfevents file
                ea = EventAccumulator(local_tb_dir)
                ea.Reload()
                
                time_series_data = []
                
                # Process scalars
                for tag in ea.Tags().get('scalars', []):
                    events = ea.Scalars(tag)
                    for event in events:
                        ts_name = f"{run_resource}/timeSeries/{tag.replace('/', '_')}"
                        time_series_data.append(
                            td_types.TimeSeriesData(
                                tensorboard_time_series_id=tag.replace('/', '_'),
                                value_type=tts_types.TensorboardTimeSeries.ValueType.SCALAR,
                                values=[
                                    td_types.TimeSeriesDataPoint(
                                        step=event.step,
                                        wall_time=event.wall_time,
                                        scalar=td_types.Scalar(value=event.value)
                                    )
                                ]
                            )
                        )
                
                # Process images (note: this is fragile and may not work reliably)
                for tag in ea.Tags().get('images', []):
                    events = ea.Images(tag)
                    for event in events:
                        time_series_data.append(
                            td_types.TimeSeriesData(
                                tensorboard_time_series_id=tag.replace('/', '_'),
                                value_type=tts_types.TensorboardTimeSeries.ValueType.BLOB_SEQUENCE,
                                values=[
                                    td_types.TimeSeriesDataPoint(
                                        step=event.step,
                                        wall_time=event.wall_time,
                                        blobs=td_types.TensorboardBlobSequence(
                                            values=[td_types.TensorboardBlob(data=event.encoded_image_string)]
                                        )
                                    )
                                ]
                            )
                        )
                
                # Write data in batches
                batch_size = 10
                for i in range(0, len(time_series_data), batch_size):
                    batch = time_series_data[i:i+batch_size]
                    try:
                        request = WriteTensorboardRunDataRequest(
                            tensorboard_run=run_resource,
                            time_series_data=batch
                        )
                        client.write_tensorboard_run_data(request=request)
                    except Exception as batch_err:
                        logger.warning(f"   ⚠️  Batch {i//batch_size + 1} failed: {batch_err}")
                
                logger.info(f"   ✅ Uploaded {len(time_series_data)} time series to Vertex AI TensorBoard")
            except Exception as upload_err:
                logger.warning(f"   ⚠️  Direct upload failed: {upload_err}")
                logger.info("   TensorBoard data is available in GCS and can be viewed locally")
        
        return gcs_path
        
    except Exception as e:
        logger.error(f"❌ TensorBoard logging failed: {e}")
        raise


def store_results_in_bigquery(metrics: Dict[str, Any], config: Dict[str, Any]):
    """Store skew audit results in BigQuery."""
    project_id = config['project']['id']
    dataset_id = config['bigquery']['dataset']
    table_id = 'skew_audit_history'
    
    logger.info(f"💾 Storing results in BigQuery...")
    
    client = bigquery.Client(project=project_id)
    
    # Ensure dataset exists
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = 'US'
        client.create_dataset(dataset, exists_ok=True)
    
    # Create table if needed
    table_ref = dataset_ref.table(table_id)
    schema = [
        bigquery.SchemaField('timestamp', 'TIMESTAMP'),
        bigquery.SchemaField('training_samples', 'INTEGER'),
        bigquery.SchemaField('recent_samples', 'INTEGER'),
        bigquery.SchemaField('avg_ks_statistic', 'FLOAT'),
        bigquery.SchemaField('max_ks_statistic', 'FLOAT'),
        bigquery.SchemaField('avg_psi', 'FLOAT'),
        bigquery.SchemaField('max_psi', 'FLOAT'),
        bigquery.SchemaField('avg_js_divergence', 'FLOAT'),
        bigquery.SchemaField('max_js_divergence', 'FLOAT'),
        bigquery.SchemaField('significant_drift_count', 'INTEGER'),
        bigquery.SchemaField('entropy_drift', 'FLOAT'),
        bigquery.SchemaField('drift_score', 'FLOAT'),
        bigquery.SchemaField('skew_kurtosis_score', 'FLOAT'),
        bigquery.SchemaField('distribution_drift_score', 'FLOAT'),
        bigquery.SchemaField('tail_other_score', 'FLOAT'),
        bigquery.SchemaField('retraining_needed', 'STRING'),
        bigquery.SchemaField('alert_count', 'INTEGER'),
        bigquery.SchemaField('needs_attention', 'BOOLEAN'),
        bigquery.SchemaField('alerts', 'STRING'),
        bigquery.SchemaField('metrics_json', 'STRING'),
    ]
    
    try:
        client.get_table(table_ref)
    except Exception:
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        logger.info(f"   Created table {table_id}")
    
    # Prepare row data
    summary = metrics.get('summary', {})
    pred_metrics = metrics.get('prediction_metrics', {})
    drift_score_info = metrics.get('drift_score_info', {})
    
    rows = [{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_samples': int(metrics['training_samples']),
        'recent_samples': int(metrics['recent_samples']),
        'avg_ks_statistic': float(summary.get('avg_feature_ks', 0)),
        'max_ks_statistic': float(summary.get('max_feature_ks', 0)),
        'avg_psi': float(summary.get('avg_feature_psi', 0)),
        'max_psi': float(summary.get('max_feature_psi', 0)),
        'avg_js_divergence': float(summary.get('avg_feature_js', 0)),
        'max_js_divergence': float(summary.get('max_feature_js', 0)),
        'significant_drift_count': int(summary.get('significant_drift_count', 0)),
        'entropy_drift': float(pred_metrics.get('entropy_drift', 0)),
        'drift_score': float(drift_score_info.get('drift_score', 0)),
        'skew_kurtosis_score': float(drift_score_info.get('skew_kurtosis_score', 0)),
        'distribution_drift_score': float(drift_score_info.get('distribution_drift_score', 0)),
        'tail_other_score': float(drift_score_info.get('tail_other_score', 0)),
        'retraining_needed': str(drift_score_info.get('retraining_needed', 'UNKNOWN')),
        'alert_count': int(len(metrics.get('alerts', []))),
        'needs_attention': bool(metrics.get('needs_attention', False)),
        'alerts': json.dumps(metrics.get('alerts', [])),
        'metrics_json': json.dumps(metrics, default=str)
    }]
    
    errors = client.insert_rows_json(table_ref, rows)
    if errors:
        logger.warning(f"   ⚠️  Insert errors: {errors}")
    else:
        logger.info(f"   ✅ Results stored in BigQuery")


def run_skew_audit() -> Dict[str, Any]:
    """
    Run the complete label-free skew audit pipeline.
    
    Returns:
        Dict with audit results
    """
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("🔍 DATA SKEW AUDIT (Label-Free)")
    logger.info("=" * 70)
    
    try:
        # Load config
        config = load_config()
        project_id = config['project']['id']
        region = config['project']['region']
        
        # Get TensorBoard instance
        tensorboard_id = os.environ.get('SKEW_TENSORBOARD_ID')
        if tensorboard_id:
            project_number = '303566498201'
            tensorboard_resource = f'projects/{project_number}/locations/{region}/tensorboards/{tensorboard_id}'
        else:
            tensorboard_resource = None
            logger.warning("   ⚠️  No SKEW_TENSORBOARD_ID set")
        
        # Step 1: Collect recent EE samples (no labels)
        logger.info("STEP 1: Collect recent Earth Engine samples (label-free)")
        logger.info("-" * 70)
        recent_df = collect_recent_ee_samples(config, sample_fraction=0.1)
        logger.info("")
        
        # Step 2: Load training data (features only)
        logger.info("STEP 2: Load training data from BigQuery (features only)")
        logger.info("-" * 70)
        training_df = load_training_data(config)
        logger.info("")
        
        # Step 3: Get endpoint predictions with confidence
        logger.info("STEP 3: Get endpoint predictions")
        logger.info("-" * 70)
        training_pred_metrics = {}
        recent_pred_metrics = {}
        try:
            training_df, training_pred_metrics = get_endpoint_predictions_with_confidence(config, training_df)
            if len(recent_df) > 0:
                recent_df, recent_pred_metrics = get_endpoint_predictions_with_confidence(config, recent_df)
        except Exception as e:
            logger.warning(f"   ⚠️  Could not get predictions: {e}")
        logger.info("")
        
        # Step 4: Compute label-free skew metrics
        logger.info("STEP 4: Compute label-free skew metrics")
        logger.info("-" * 70)
        metrics = compute_label_free_skew_metrics(
            training_df, recent_df, 
            training_pred_metrics, recent_pred_metrics,
            config
        )
        logger.info("")
        
        # Step 5: Log to TensorBoard
        logger.info("STEP 5: Log to TensorBoard")
        logger.info("-" * 70)
        if tensorboard_resource:
            tb_path = log_skew_to_tensorboard(
                metrics, training_df, recent_df, tensorboard_resource, config
            )
            metrics['tensorboard_path'] = tb_path
        else:
            logger.warning("   Skipping TensorBoard logging (no instance configured)")
        logger.info("")
        
        # Step 6: Store in BigQuery
        logger.info("STEP 6: Store results in BigQuery")
        logger.info("-" * 70)
        store_results_in_bigquery(metrics, config)
        logger.info("")
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds() / 60
        summary = metrics.get('summary', {})
        drift_score_info = metrics.get('drift_score_info', {})
        
        logger.info("=" * 70)
        logger.info("✅ SKEW AUDIT COMPLETE (Label-Free)")
        logger.info("=" * 70)
        logger.info("")
        logger.info("╔══════════════════════════════════════════╗")
        logger.info(f"║  DRIFT SCORE: {drift_score_info.get('drift_score', 0):>5.1f}/100                   ║")
        logger.info(f"║  RETRAINING: {drift_score_info.get('retraining_needed', 'N/A'):<20}        ║")
        logger.info("╚══════════════════════════════════════════╝")
        logger.info("")
        logger.info(f"Duration: {duration:.1f} minutes")
        logger.info(f"Training samples: {metrics['training_samples']}")
        logger.info(f"Recent samples: {metrics['recent_samples']}")
        logger.info(f"Avg KS: {summary.get('avg_feature_ks', 0):.4f} | Avg PSI: {summary.get('avg_feature_psi', 0):.4f} | Avg JS: {summary.get('avg_feature_js', 0):.4f}")
        logger.info(f"Max KS: {summary.get('max_feature_ks', 0):.4f} | Max PSI: {summary.get('max_feature_psi', 0):.4f} | Max JS: {summary.get('max_feature_js', 0):.4f}")
        logger.info(f"Score Breakdown: Skew/Kurt={drift_score_info.get('skew_kurtosis_score', 0):.1f} | Dist={drift_score_info.get('distribution_drift_score', 0):.1f} | Tail={drift_score_info.get('tail_other_score', 0):.1f}")
        logger.info(f"Significant drifts: {summary.get('significant_drift_count', 0)}")
        logger.info(f"Alerts: {len(metrics.get('alerts', []))}")
        logger.info(f"Needs attention: {metrics.get('needs_attention', False)}")
        
        return {
            'status': 'success',
            'duration_minutes': round(duration, 2),
            'drift_score': drift_score_info.get('drift_score', 0),
            'retraining_needed': drift_score_info.get('retraining_needed', 'UNKNOWN'),
            'metrics': metrics
        }
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds() / 60
        logger.error(f"❌ Skew audit failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'duration_minutes': round(duration, 2)
        }


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    result = run_skew_audit()
    print(json.dumps(result, indent=2, default=str))
