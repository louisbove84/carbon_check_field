"""
Comprehensive model evaluation utilities for crop classification.
All results are logged to TensorBoard - no image files are generated.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _convert_figure_to_tensorboard_image(fig):
    """
    Convert matplotlib figure to TensorBoard-compatible image array.
    Returns numpy array in CHW format (channels, height, width) with values in [0, 1].
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    # Convert RGBA to RGB if needed (TensorBoard expects RGB)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image_array = np.array(image)
    # Ensure CHW format (channels, height, width) - TensorBoard requirement
    if len(image_array.shape) == 3:
        image_array = np.transpose(image_array, (2, 0, 1))
    # Normalize to [0, 1] range as float32 (required for TensorBoard display)
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32) / 255.0
    return image_array


def log_confusion_matrix_to_tensorboard(writer, y_true, y_pred, labels, step=0):
    """
    Log confusion matrix visualizations to TensorBoard.
    Creates 3 views: counts+%, percentages, and normalized.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
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
    
    # Convert to image tensor
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('confusion_matrix/counts_and_percent', image_array, step)
    plt.close(fig)
    
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
    
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('confusion_matrix/percentage', image_array, step)
    plt.close(fig)
    
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
    
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('confusion_matrix/normalized', image_array, step)
    plt.close(fig)
    
    logger.info("✅ Logged confusion matrices to TensorBoard")


def log_per_crop_metrics_to_tensorboard(writer, y_true, y_pred, labels, step=0):
    """
    Log per-crop metrics charts to TensorBoard.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None
    )
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Crop': labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Chart 1: Grouped bar chart (Precision, Recall, F1)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.25
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#2ecc71')
    ax.bar(x, recall, width, label='Recall', alpha=0.8, color='#3498db')
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#e74c3c')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Crop Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('per_crop_metrics/comparison', image_array, step)
    plt.close(fig)
    
    # Chart 2: F1-Score sorted
    sorted_df = metrics_df.sort_values('F1-Score', ascending=True)
    colors = ['#e74c3c' if x < 0.8 else '#f39c12' if x < 0.9 else '#2ecc71' 
              for x in sorted_df['F1-Score']]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_df['Crop'], sorted_df['F1-Score'], color=colors, alpha=0.8)
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score by Crop (Sorted)', fontsize=14, fontweight='bold')
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Target: 0.9')
    ax.axvline(x=0.8, color='orange', linestyle='--', alpha=0.5, label='Warning: 0.8')
    ax.legend()
    ax.set_xlim([0, 1.1])
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('per_crop_metrics/f1_sorted', image_array, step)
    plt.close(fig)
    
    # Log individual metrics as scalars
    for crop in labels:
        idx = labels.index(crop)
        writer.add_scalar(f'per_crop/precision/{crop}', precision[idx], step)
        writer.add_scalar(f'per_crop/recall/{crop}', recall[idx], step)
        writer.add_scalar(f'per_crop/f1/{crop}', f1[idx], step)
        writer.add_scalar(f'per_crop/support/{crop}', support[idx], step)
    
    logger.info("✅ Logged per-crop metrics to TensorBoard")
    return metrics_df


def log_feature_importance_to_tensorboard(writer, model, feature_names, step=0):
    """
    Log feature importance analysis to TensorBoard.
    REMOVED: lat/lon features — model no longer uses geographic cheating
    """
    # Extract RandomForest classifier
    if hasattr(model, 'named_steps'):
        rf_model = model.named_steps['classifier']
    else:
        rf_model = model
    
    if not isinstance(rf_model, RandomForestClassifier):
        logger.warning("⚠️  Model is not RandomForest, skipping feature importance")
        return None
    
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Filter out location features (shouldn't exist, but filter just in case)
    # REMOVED: lat/lon features — model no longer uses geographic cheating
    location_features = {'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 'latitude', 'longitude'}
    filtered_indices = [i for i in indices if feature_names[i] not in location_features]
    filtered_importances = [importances[i] for i in filtered_indices]
    filtered_feature_names = [feature_names[i] for i in filtered_indices]
    
    importance_df = pd.DataFrame({
        'Feature': filtered_feature_names,
        'Importance': filtered_importances,
        'Rank': range(1, len(filtered_indices) + 1)
    })
    
    # Top 20 features bar chart
    top_n = min(20, len(feature_names))
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    ax.barh(range(top_n), top_features['Importance'], color=colors, alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('feature_importance/top_features', image_array, step)
    plt.close(fig)
    
    # Cumulative importance
    cumsum = np.cumsum(importance_df['Importance'])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumsum) + 1), cumsum, 'o-', linewidth=2, markersize=4)
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% threshold')
    ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Cumulative Importance', fontsize=12)
    ax.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    n_80 = np.argmax(cumsum >= 0.8) + 1
    n_90 = np.argmax(cumsum >= 0.9) + 1
    ax.axvline(x=n_80, color='r', linestyle=':', alpha=0.5)
    ax.axvline(x=n_90, color='orange', linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('feature_importance/cumulative', image_array, step)
    plt.close(fig)
    
    # Log top features as scalars
    for idx, row in top_features.iterrows():
        writer.add_scalar(f'feature_importance/{row["Feature"]}', row['Importance'], step)
    
    logger.info("✅ Logged feature importance to TensorBoard")
    return importance_df


def log_misclassification_analysis_to_tensorboard(writer, y_true, y_pred, labels, step=0):
    """
    Log misclassification patterns to TensorBoard.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    misclass_pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                misclass_pairs.append({
                    'True': labels[i],
                    'Predicted': labels[j],
                    'Count': int(cm[i, j]),
                    'Percentage': float(cm[i, j] / cm[i].sum() * 100)
                })
    
    misclass_df = pd.DataFrame(misclass_pairs).sort_values('Count', ascending=False)
    
    if len(misclass_df) == 0:
        logger.info("🎉 No misclassifications found!")
        return misclass_df
    
    # Top 10 misclassification pairs
    top_n = min(10, len(misclass_df))
    top_misclass = misclass_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    labels_text = [f"{row['True']} → {row['Predicted']}" for _, row in top_misclass.iterrows()]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, top_n))
    bars = ax.barh(range(top_n), top_misclass['Count'], color=colors, alpha=0.8)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels_text)
    ax.set_xlabel('Number of Misclassifications', fontsize=12)
    ax.set_title(f'Top {top_n} Misclassification Patterns', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    for i, (_, row) in enumerate(top_misclass.iterrows()):
        ax.text(row['Count'] + max(top_misclass['Count'])*0.02, i, 
               f"{row['Percentage']:.1f}%", 
               va='center', fontweight='bold')
    
    plt.tight_layout()
    
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('misclassification/top_patterns', image_array, step)
    plt.close(fig)
    
    logger.info("✅ Logged misclassification analysis to TensorBoard")
    return misclass_df


def log_advanced_metrics_to_tensorboard(writer, y_true, y_pred, step=0):
    """
    Log advanced metrics to TensorBoard and return report dict.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=sorted(set(y_true)), average=None
    )
    
    labels = sorted(set(y_true))
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    precision_weighted = np.average(precision, weights=support)
    recall_weighted = np.average(recall, weights=support)
    f1_weighted = np.average(f1, weights=support)
    
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    
    # Log scalar metrics
    writer.add_scalar('metrics/accuracy', accuracy, step)
    writer.add_scalar('metrics/cohen_kappa', cohen_kappa, step)
    writer.add_scalar('metrics/matthews_corrcoef', mcc, step)
    writer.add_scalar('metrics/macro_precision', precision_macro, step)
    writer.add_scalar('metrics/macro_recall', recall_macro, step)
    writer.add_scalar('metrics/macro_f1', f1_macro, step)
    writer.add_scalar('metrics/weighted_precision', precision_weighted, step)
    writer.add_scalar('metrics/weighted_recall', recall_weighted, step)
    writer.add_scalar('metrics/weighted_f1', f1_weighted, step)
    
    # Bar chart of key metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ['Accuracy', 'Cohen Kappa', 'Matthews Corr.', 'Macro F1', 'Weighted F1']
    metrics_values = [accuracy, cohen_kappa, mcc, f1_macro, f1_weighted]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('metrics/overall', image_array, step)
    plt.close(fig)
    
    # Create report dict
    report = {
        'overall_metrics': {
            'accuracy': float(accuracy),
            'cohen_kappa': float(cohen_kappa),
            'matthews_corrcoef': float(mcc),
            'macro_avg': {
                'precision': float(precision_macro),
                'recall': float(recall_macro),
                'f1_score': float(f1_macro)
            },
            'weighted_avg': {
                'precision': float(precision_weighted),
                'recall': float(recall_weighted),
                'f1_score': float(f1_weighted)
            }
        },
        'per_crop_metrics': {}
    }
    
    for i, crop in enumerate(labels):
        report['per_crop_metrics'][crop] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    logger.info("✅ Logged advanced metrics to TensorBoard")
    return report


def run_comprehensive_evaluation(model, X_test, y_test, feature_names, writer, step=0, num_runs=1):
    """
    Run all evaluation metrics and log everything to TensorBoard.
    
    Args:
        model: Trained sklearn pipeline
        X_test: Test features
        y_test: True test labels
        feature_names: List of feature names
        writer: TensorBoard SummaryWriter
        step: Starting step number for TensorBoard (default: 0)
        num_runs: Number of evaluation runs to log (for multiple data points in scalars)
        
    Returns:
        dict: Dictionary containing all evaluation results (no file paths)
    """
    logger.info("=" * 70)
    logger.info("🔍 COMPREHENSIVE MODEL EVALUATION (TensorBoard Only)")
    logger.info("=" * 70)
    
    # Verify feature names match model
    if hasattr(model, 'named_steps'):
        rf_model = model.named_steps['classifier']
    else:
        rf_model = model
    
    if hasattr(rf_model, 'n_features_in_'):
        if len(feature_names) != rf_model.n_features_in_:
            logger.warning(f"⚠️  Feature name count ({len(feature_names)}) doesn't match model ({rf_model.n_features_in_})")
            logger.warning(f"   Feature names: {feature_names}")
        else:
            logger.info(f"✅ Feature names verified: {len(feature_names)} features")
            # REMOVED: lat/lon sin/cos features — model no longer uses geographic cheating
    
    # Get predictions
    y_pred = model.predict(X_test)
    labels = sorted(set(y_test))
    
    results = {}
    
    # Log multiple runs for scalars (so we see multiple data points)
    for run in range(num_runs):
        current_step = step + run
        
        # 1. Confusion Matrix (only log once, on first run)
        if run == 0:
            logger.info("📊 Logging confusion matrices to TensorBoard...")
            log_confusion_matrix_to_tensorboard(writer, y_test, y_pred, labels, current_step)
        
        # 2. Per-Crop Metrics (log each run for scalars)
        if run == 0:
            logger.info("📈 Logging per-crop metrics to TensorBoard...")
        metrics_df = log_per_crop_metrics_to_tensorboard(writer, y_test, y_pred, labels, current_step)
        if run == 0:
            results['metrics_dataframe'] = metrics_df
        
        # 3. Feature Importance (only log once, on first run)
        if run == 0:
            logger.info("🔑 Logging feature importance to TensorBoard...")
            fi_df = log_feature_importance_to_tensorboard(writer, model, feature_names, current_step)
            if fi_df is not None:
                results['feature_importance_dataframe'] = fi_df
                # Log feature names for debugging
                logger.info(f"   Feature names in importance: {list(fi_df['Feature'].head(10))}")
        
        # 4. Misclassification Analysis (only log once, on first run)
        if run == 0:
            logger.info("🔍 Logging misclassification patterns to TensorBoard...")
            misclass_df = log_misclassification_analysis_to_tensorboard(writer, y_test, y_pred, labels, current_step)
            results['misclassification_dataframe'] = misclass_df
        
        # 5. Advanced Metrics (log each run for scalars)
        advanced_metrics = log_advanced_metrics_to_tensorboard(writer, y_test, y_pred, current_step)
        if run == 0:
            results['advanced_metrics'] = advanced_metrics
    
    # Log text summary (only once)
    summary_text = f"""
    Model Evaluation Summary
    ========================
    Accuracy: {advanced_metrics['overall_metrics']['accuracy']:.2%}
    Cohen's Kappa: {advanced_metrics['overall_metrics']['cohen_kappa']:.3f}
    Matthews Corr: {advanced_metrics['overall_metrics']['matthews_corrcoef']:.3f}
    Macro F1: {advanced_metrics['overall_metrics']['macro_avg']['f1_score']:.2%}
    Weighted F1: {advanced_metrics['overall_metrics']['weighted_avg']['f1_score']:.2%}
    
    Test Samples: {len(y_test)}
    Feature Count: {len(feature_names)}
    Features: {', '.join(feature_names)}
    """
    writer.add_text('evaluation_summary', summary_text, step)
    
    logger.info("=" * 70)
    logger.info("✅ COMPREHENSIVE EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info("All results logged to TensorBoard")
    
    # Print summary to console
    logger.info("\n📊 EVALUATION SUMMARY:")
    logger.info(f"   Accuracy: {advanced_metrics['overall_metrics']['accuracy']:.2%}")
    logger.info(f"   Cohen's Kappa: {advanced_metrics['overall_metrics']['cohen_kappa']:.3f}")
    logger.info(f"   Matthews Corr: {advanced_metrics['overall_metrics']['matthews_corrcoef']:.3f}")
    logger.info(f"   Macro F1: {advanced_metrics['overall_metrics']['macro_avg']['f1_score']:.2%}")
    logger.info(f"   Weighted F1: {advanced_metrics['overall_metrics']['weighted_avg']['f1_score']:.2%}")
    logger.info(f"   Features: {len(feature_names)} (including sin/cos: {sum(1 for f in feature_names if 'sin' in f or 'cos' in f)})")
    
    return results
