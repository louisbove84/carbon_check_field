"""
MLflow utilities for experiment tracking and artifact logging.
Provides an alternative to TensorBoard for better image/artifact handling.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier
from visualization_utils import (
    create_confusion_matrix_figures,
    create_per_crop_metrics_figures,
    create_feature_importance_figure,
    create_advanced_metrics_figure
)

logger = logging.getLogger(__name__)


def setup_mlflow_tracking(tracking_uri=None, experiment_name="crop-classification"):
    """
    Set up MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking URI (if None, uses local file store or GCS)
        experiment_name: Name of the MLflow experiment
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"✅ Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            mlflow.set_experiment(experiment_name)
            logger.info(f"✅ Using existing MLflow experiment: {experiment_name}")
    except Exception as e:
        logger.warning(f"⚠️  Could not set MLflow experiment: {e}")
        logger.info("   Using default experiment")


def log_confusion_matrix_mlflow(y_true, y_pred, labels, run_id=None):
    """
    Log confusion matrix visualizations to MLflow.
    Uses shared visualization functions to ensure consistency with TensorBoard.
    """
    # Use shared visualization function
    figures = create_confusion_matrix_figures(y_true, y_pred, labels)
    
    # Log each figure to MLflow
    mlflow.log_figure(figures['counts_and_percent'], "confusion_matrix/counts_and_percent.png")
    plt.close(figures['counts_and_percent'])
    
    mlflow.log_figure(figures['percentage'], "confusion_matrix/percentage.png")
    plt.close(figures['percentage'])
    
    mlflow.log_figure(figures['normalized'], "confusion_matrix/normalized.png")
    plt.close(figures['normalized'])
    
    logger.info("✅ Logged confusion matrices to MLflow")


def log_per_crop_metrics_mlflow(y_true, y_pred, labels):
    """
    Log per-crop metrics to MLflow.
    Uses shared visualization functions to ensure consistency with TensorBoard.
    """
    # Use shared visualization function
    figures, metrics_df = create_per_crop_metrics_figures(y_true, y_pred, labels)
    
    # Log metrics as parameters
    for _, row in metrics_df.iterrows():
        mlflow.log_metric(f"per_crop/precision/{row['Crop']}", row['Precision'])
        mlflow.log_metric(f"per_crop/recall/{row['Crop']}", row['Recall'])
        mlflow.log_metric(f"per_crop/f1/{row['Crop']}", row['F1-Score'])
        mlflow.log_metric(f"per_crop/support/{row['Crop']}", row['Support'])
    
    # Log visualizations
    mlflow.log_figure(figures['comparison'], "per_crop_metrics/comparison.png")
    plt.close(figures['comparison'])
    
    mlflow.log_figure(figures['f1_sorted'], "per_crop_metrics/f1_sorted.png")
    plt.close(figures['f1_sorted'])
    
    logger.info("✅ Logged per-crop metrics to MLflow")
    return metrics_df


def log_feature_importance_mlflow(model, feature_names):
    """
    Log feature importance to MLflow.
    Uses shared visualization functions to ensure consistency with TensorBoard.
    """
    # Use shared visualization function
    result = create_feature_importance_figure(model, feature_names)
    if result is None:
        logger.warning("⚠️  Model does not have feature_importances_ attribute")
        return
    
    fig, filtered_names, filtered_importances = result
    
    # Log top features as metrics
    indices = np.argsort(filtered_importances)[::-1][:10]
    for idx in indices:
        mlflow.log_metric(f"feature_importance/{filtered_names[idx]}", filtered_importances[idx])
    
    # Log visualization
    mlflow.log_figure(fig, "feature_importance/top_features.png")
    plt.close(fig)
    
    logger.info("✅ Logged feature importance to MLflow")


def log_advanced_metrics_mlflow(y_true, y_pred, labels):
    """
    Log advanced metrics to MLflow.
    Uses shared visualization functions to ensure consistency with TensorBoard.
    """
    # Use shared visualization function
    fig, metrics_dict = create_advanced_metrics_figure(y_true, y_pred, labels)
    
    # Log all metrics
    mlflow.log_metric("metrics/accuracy", metrics_dict['accuracy'])
    mlflow.log_metric("metrics/cohen_kappa", metrics_dict['cohen_kappa'])
    mlflow.log_metric("metrics/matthews_corrcoef", metrics_dict['matthews_corrcoef'])
    mlflow.log_metric("metrics/precision_macro", metrics_dict['precision_macro'])
    mlflow.log_metric("metrics/recall_macro", metrics_dict['recall_macro'])
    mlflow.log_metric("metrics/f1_macro", metrics_dict['f1_macro'])
    
    # Log visualization
    mlflow.log_figure(fig, "metrics/overall.png")
    plt.close(fig)
    
    logger.info("✅ Logged advanced metrics to MLflow")
    
    return metrics_dict


def run_comprehensive_evaluation_mlflow(model, X_test, y_test, feature_names, labels):
    """
    Run comprehensive evaluation and log everything to MLflow.
    
    Args:
        model: Trained scikit-learn model
        X_test: Test features
        y_test: True labels
        feature_names: List of feature names
        labels: List of class labels
    """
    y_pred = model.predict(X_test)
    
    # Log all visualizations
    log_confusion_matrix_mlflow(y_test, y_pred, labels)
    log_per_crop_metrics_mlflow(y_test, y_pred, labels)
    log_feature_importance_mlflow(model, feature_names)
    metrics = log_advanced_metrics_mlflow(y_test, y_pred, labels)
    
    return metrics

