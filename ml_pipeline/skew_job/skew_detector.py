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
                
                # Determine significance using multiple metrics
                # KS: > 0.15 with p < 0.05
                # PSI: >= 0.25 indicates significant shift
                # JS: >= 0.2 indicates significant difference
                ks_significant = ks_stat > 0.15 and ks_pval < 0.05
                psi_significant = psi >= 0.25
                js_significant = js_div >= 0.2
                
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
                    'ks_significant': ks_significant,
                    'psi_significant': psi_significant,
                    'js_significant': js_significant,
                    'significant': ks_significant or psi_significant or js_significant
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
    
    metrics['summary'] = {
        'avg_feature_ks': float(avg_ks),
        'max_feature_ks': float(max_ks),
        'avg_feature_psi': float(avg_psi),
        'max_feature_psi': float(max_psi),
        'avg_feature_js': float(avg_js),
        'max_feature_js': float(max_js),
        'significant_drift_count': significant_drifts,
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
    logger.info(f"   Significant drifts: {significant_drifts}/{len(feature_skew)}")
    logger.info(f"   Entropy drift: {entropy_drift:+.4f}")
    logger.info(f"   Needs attention: {metrics['needs_attention']}")
    
    return metrics


def log_skew_to_tensorboard(
    metrics: Dict[str, Any],
    training_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    tensorboard_resource: str,
    config: Dict[str, Any]
) -> str:
    """
    Log label-free skew metrics and visualizations to TensorBoard.
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
    
    # Local TensorBoard directory
    local_tb_dir = f'/tmp/tensorboard_logs/{experiment_id}/{run_id}'
    os.makedirs(local_tb_dir, exist_ok=True)
    
    logger.info(f"📊 Logging skew metrics to TensorBoard...")
    logger.info(f"   Experiment: {experiment_id}")
    logger.info(f"   Run: {run_id}")
    
    writer = SummaryWriter(log_dir=local_tb_dir)
    
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
                
                drift_status = "⚠️ DRIFT" if stats_dict['significant'] else "✓ OK"
                ax.set_title(f'{feature}\nKS={stats_dict["ks_statistic"]:.3f} {drift_status}')
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            
            plt.suptitle('Feature Distribution Comparison (Label-Free)', fontsize=14, y=1.02)
            plt.tight_layout()
            writer.add_figure('skew/feature_distributions', fig, 0, close=True)
        
        # 2. Multi-Metric Drift Summary (KS, PSI, JS)
        if feature_skew:
            features = list(feature_skew.keys())
            ks_stats = [feature_skew[f]['ks_statistic'] for f in features]
            psi_stats = [feature_skew[f]['psi'] for f in features]
            js_stats = [feature_skew[f]['js_divergence'] for f in features]
            significants = [feature_skew[f]['significant'] for f in features]
            
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
        
        # 3. Normalized Mean Shift Plot
        if feature_skew:
            features = list(feature_skew.keys())
            shifts = [feature_skew[f]['normalized_mean_shift'] for f in features]
            
            fig, ax = plt.subplots(figsize=(12, 4))
            colors = ['red' if s > 1.0 else 'orange' if s > 0.5 else 'green' for s in shifts]
            ax.bar(features, shifts, color=colors, alpha=0.7)
            ax.axhline(y=1.0, color='red', linestyle='--', label='1σ shift')
            ax.axhline(y=0.5, color='orange', linestyle='--', label='0.5σ shift')
            ax.set_ylabel('Normalized Mean Shift (σ)')
            ax.set_title('Feature Mean Shift (in Standard Deviations)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            writer.add_figure('skew/mean_shift', fig, 0, close=True)
        
        # 4. Prediction Entropy Comparison
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
                drift_text += " ⚠️ (model less certain on new data)"
            ax.text(0.5, 0.95, drift_text, transform=ax.transAxes, ha='center', fontsize=10)
            
            plt.tight_layout()
            writer.add_figure('skew/prediction_entropy', fig, 0, close=True)
        
        # 5. Log scalars
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
        
        if pred_metrics:
            writer.add_scalar('skew/training_entropy', pred_metrics.get('training_entropy', 0), 0)
            writer.add_scalar('skew/recent_entropy', pred_metrics.get('recent_entropy', 0), 0)
            writer.add_scalar('skew/entropy_drift', pred_metrics.get('entropy_drift', 0), 0)
        
        for feature, stats_dict in feature_skew.items():
            writer.add_scalar(f'feature_ks/{feature}', stats_dict['ks_statistic'], 0)
            writer.add_scalar(f'feature_psi/{feature}', stats_dict['psi'], 0)
            writer.add_scalar(f'feature_js/{feature}', stats_dict['js_divergence'], 0)
            writer.add_scalar(f'feature_shift/{feature}', stats_dict['normalized_mean_shift'], 0)
        
        # 6. Log text summary
        alerts_text = '\n'.join(metrics['alerts']) if metrics['alerts'] else 'No alerts'
        summary_text = f"""
Label-Free Skew Audit Summary
=============================
Timestamp: {metrics['timestamp']}
Training Samples: {metrics['training_samples']}
Recent Samples: {metrics['recent_samples']}

Feature Drift Analysis (KS Test):
- Average KS Statistic: {summary.get('avg_feature_ks', 0):.4f}
- Max KS Statistic: {summary.get('max_feature_ks', 0):.4f}
- Threshold: 0.15 (with p < 0.05)

Feature Drift Analysis (PSI):
- Average PSI: {summary.get('avg_feature_psi', 0):.4f}
- Max PSI: {summary.get('max_feature_psi', 0):.4f}
- Thresholds: <0.1 OK, 0.1-0.25 Monitor, >=0.25 Action Required

Feature Drift Analysis (JS Divergence):
- Average JS: {summary.get('avg_feature_js', 0):.4f}
- Max JS: {summary.get('max_feature_js', 0):.4f}
- Thresholds: <0.1 OK, 0.1-0.2 Monitor, >=0.2 Significant

Significant Drifts: {summary.get('significant_drift_count', 0)}/{summary.get('total_features_tested', 0)}

Prediction Metrics:
- Training Entropy: {pred_metrics.get('training_entropy', 0):.4f}
- Recent Entropy: {pred_metrics.get('recent_entropy', 0):.4f}
- Entropy Drift: {pred_metrics.get('entropy_drift', 0):+.4f}

Alerts:
{alerts_text}

Needs Attention: {'YES' if metrics.get('needs_attention') else 'NO'}
"""
        writer.add_text('skew/summary', summary_text, 0)
        
        writer.flush()
        writer.close()
        
        # Upload to GCS
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
        
        # Initialize Vertex AI Experiment
        try:
            aiplatform.init(
                project=project_id,
                location=region,
                experiment=experiment_id,
                experiment_tensorboard=tensorboard_resource
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
                'needs_attention': int(metrics.get('needs_attention', False))
            })
            
            aiplatform.end_run()
            logger.info(f"   ✅ Logged to Vertex AI Experiments")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not log to Vertex AI Experiments: {e}")
        
        return gcs_path
        
    except Exception as e:
        logger.error(f"❌ TensorBoard logging failed: {e}")
        writer.close()
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
        
        logger.info("=" * 70)
        logger.info("✅ SKEW AUDIT COMPLETE (Label-Free)")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.1f} minutes")
        logger.info(f"Training samples: {metrics['training_samples']}")
        logger.info(f"Recent samples: {metrics['recent_samples']}")
        logger.info(f"Avg KS: {summary.get('avg_feature_ks', 0):.4f} | Avg PSI: {summary.get('avg_feature_psi', 0):.4f} | Avg JS: {summary.get('avg_feature_js', 0):.4f}")
        logger.info(f"Max KS: {summary.get('max_feature_ks', 0):.4f} | Max PSI: {summary.get('max_feature_psi', 0):.4f} | Max JS: {summary.get('max_feature_js', 0):.4f}")
        logger.info(f"Significant drifts: {summary.get('significant_drift_count', 0)}")
        logger.info(f"Alerts: {len(metrics.get('alerts', []))}")
        logger.info(f"Needs attention: {metrics.get('needs_attention', False)}")
        
        return {
            'status': 'success',
            'duration_minutes': round(duration, 2),
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
