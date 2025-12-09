"""
Shared visualization utilities for model evaluation.
These functions generate matplotlib figures that can be logged to either TensorBoard or MLflow.
This ensures consistency between both logging systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support
)


def create_confusion_matrix_figures(y_true, y_pred, labels):
    """
    Create confusion matrix visualizations.
    Returns a dict with three matplotlib figures: counts_and_percent, percentage, normalized.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
    
    Returns:
        dict with keys: 'counts_and_percent', 'percentage', 'normalized'
        Each value is a matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    figures = {}
    
    # 1. Counts + Percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix (Count + Percentage)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    figures['counts_and_percent'] = fig
    
    # 2. Percentages only
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=0, vmax=100, cbar_kws={'label': 'Percentage'})
    ax.set_title('Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    figures['percentage'] = fig
    
    # 3. Normalized (0-1)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=0, vmax=1, cbar_kws={'label': 'Normalized'})
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    figures['normalized'] = fig
    
    return figures


def create_per_crop_metrics_figures(y_true, y_pred, labels):
    """
    Create per-crop metrics visualizations.
    Returns a dict with matplotlib figures and metrics DataFrame.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
    
    Returns:
        dict with keys:
            - 'comparison': Figure comparing precision/recall/f1 across crops
            - 'f1_sorted': Figure showing F1 scores sorted
            - 'metrics_df': DataFrame with all metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    metrics_df = pd.DataFrame({
        'Crop': labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    figures = {}
    
    # 1. Comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.25
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Crop', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Crop Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    figures['comparison'] = fig
    
    # 2. F1 scores sorted
    sorted_df = metrics_df.sort_values('F1-Score', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(sorted_df)), sorted_df['F1-Score'], color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['Crop'])
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title('Per-Crop F1 Scores (Sorted)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.1])
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_df['F1-Score'])):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', ha='left', va='center', fontweight='bold')
    
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Target: 0.9')
    ax.axvline(x=0.8, color='orange', linestyle='--', alpha=0.5, label='Warning: 0.8')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    figures['f1_sorted'] = fig
    
    return figures, metrics_df


def create_feature_importance_figure(model, feature_names):
    """
    Create feature importance visualization.
    
    Args:
        model: Trained scikit-learn model with feature_importances_
        feature_names: List of feature names
    
    Returns:
        matplotlib figure or None if model doesn't have feature_importances_
    """
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        importances = model.named_steps['classifier'].feature_importances_
    else:
        return None
    
    # Filter out excluded features
    excluded_features = {'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 'latitude', 'longitude',
                        'elevation_binned', 'elevation_m'}
    filtered_data = [(name, imp) for name, imp in zip(feature_names, importances)
                     if name not in excluded_features]
    
    if not filtered_data:
        return None
    
    filtered_names, filtered_importances = zip(*filtered_data)
    
    # Get top 15 features
    indices = np.argsort(filtered_importances)[::-1][:15]
    top_names = [filtered_names[i] for i in indices]
    top_importances = [filtered_importances[i] for i in indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances, align='center', color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    return fig, filtered_names, filtered_importances


def create_advanced_metrics_figure(y_true, y_pred, labels):
    """
    Create advanced metrics visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
    
    Returns:
        matplotlib figure and metrics dict
    """
    from sklearn.metrics import (
        cohen_kappa_score,
        matthews_corrcoef
    )
    
    accuracy = (y_true == y_pred).mean()
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0, average=None
    )
    
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()
    
    # Create visualization
    metrics_names = ['Accuracy', 'Cohen Kappa', 'MCC', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']
    metrics_values = [accuracy, cohen_kappa, mcc, precision_macro, recall_macro, f1_macro]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Model Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    metrics_dict = {
        'accuracy': float(accuracy),
        'cohen_kappa': float(cohen_kappa),
        'matthews_corrcoef': float(mcc),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro)
    }
    
    return fig, metrics_dict

