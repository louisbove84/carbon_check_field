"""
TensorBoard Logging Module
==========================
Functions for logging model evaluation metrics and visualizations to TensorBoard.

All functions in this module are focused on logging to TensorBoard:
- Confusion matrices
- Per-crop metrics
- Feature importance
- Misclassification analysis
- Advanced metrics
- Training metrics

Multi-Model Support:
- All functions accept a `model_prefix` parameter (e.g., "rf/", "dnn/")
- This allows logging metrics for multiple models to the same TensorBoard
- Example: rf/confusion_matrix/normalized, dnn/confusion_matrix/normalized
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from torch.utils.tensorboard import SummaryWriter
from visualization_utils import (
    create_confusion_matrix_figures,
    create_per_crop_metrics_figures,
    create_feature_importance_figure,
    create_misclassification_analysis_figure,
    create_advanced_metrics_figure
)

logger = logging.getLogger(__name__)


def _prefix_tag(tag: str, model_prefix: str = "") -> str:
    """Add model prefix to a tag if provided."""
    if model_prefix:
        # Ensure prefix ends with /
        if not model_prefix.endswith('/'):
            model_prefix = model_prefix + '/'
        return f"{model_prefix}{tag}"
    return tag


def log_confusion_matrix_to_tensorboard(writer, y_true, y_pred, labels, model_prefix="", step=0):
    """
    Log confusion matrices to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
        model_prefix: Prefix for TensorBoard tags (e.g., "rf", "dnn")
        step: TensorBoard step
    """
    figures = create_confusion_matrix_figures(y_true, y_pred, labels)
    
    for name, fig in figures.items():
        tag = _prefix_tag(f'confusion_matrix/{name}', model_prefix)
        writer.add_figure(tag, fig, step, close=True)
    
    logger.info(f"   Logged confusion matrices ({model_prefix or 'default'})")


def log_per_crop_metrics_to_tensorboard(writer, y_true, y_pred, labels, model_prefix="", step=0):
    """
    Log per-crop metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
        model_prefix: Prefix for TensorBoard tags (e.g., "rf", "dnn")
        step: TensorBoard step
    
    Returns:
        DataFrame with per-crop metrics
    """
    figures, metrics_df = create_per_crop_metrics_figures(y_true, y_pred, labels)
    
    for name, fig in figures.items():
        tag = _prefix_tag(f'per_crop_metrics/{name}', model_prefix)
        writer.add_figure(tag, fig, step, close=True)
    
    # Log scalars for each crop
    for _, row in metrics_df.iterrows():
        crop = row['Crop']
        writer.add_scalar(_prefix_tag(f'crop/{crop}/precision', model_prefix), row['Precision'], step)
        writer.add_scalar(_prefix_tag(f'crop/{crop}/recall', model_prefix), row['Recall'], step)
        writer.add_scalar(_prefix_tag(f'crop/{crop}/f1_score', model_prefix), row['F1-Score'], step)
    
    logger.info(f"   Logged per-crop metrics ({model_prefix or 'default'})")
    return metrics_df


def log_feature_importance_to_tensorboard(writer, model, feature_names, model_prefix="", step=0):
    """
    Log feature importance to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        model: Trained model (must have feature_importances_ or be a Pipeline)
        feature_names: List of feature names
        model_prefix: Prefix for TensorBoard tags (e.g., "rf", "dnn")
        step: TensorBoard step
    
    Returns:
        DataFrame with feature importance or None if not available
    """
    result = create_feature_importance_figure(model, feature_names)
    
    if result is None:
        logger.info(f"   No feature importance available for {model_prefix or 'model'}")
        return None
    
    fig, filtered_names, filtered_importances = result
    
    # Log the figure
    tag = _prefix_tag('feature_importance/top_15', model_prefix)
    writer.add_figure(tag, fig, step, close=True)
    
    # Log top features as scalars
    if filtered_names and filtered_importances:
        for name, importance in zip(list(filtered_names)[:20], list(filtered_importances)[:20]):
            scalar_tag = _prefix_tag(f'feature_importance/{name}', model_prefix)
            writer.add_scalar(scalar_tag, float(importance), step)
    
    logger.info(f"   Logged feature importance ({model_prefix or 'default'})")
    
    # Create DataFrame for return
    importance_df = pd.DataFrame({
        'Feature': filtered_names,
        'Importance': filtered_importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df


def log_misclassification_analysis_to_tensorboard(writer, y_true, y_pred, labels, model_prefix="", step=0):
    """
    Log misclassification patterns to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
        model_prefix: Prefix for TensorBoard tags (e.g., "rf", "dnn")
        step: TensorBoard step
    
    Returns:
        DataFrame with misclassification analysis
    """
    figures, misclass_df = create_misclassification_analysis_figure(y_true, y_pred, labels)
    
    if not figures:
        logger.info(f"   No misclassifications to log ({model_prefix or 'default'})")
        return misclass_df
    
    for name, fig in figures.items():
        tag = _prefix_tag(f'misclassification/{name}', model_prefix)
        writer.add_figure(tag, fig, step, close=True)
    
    logger.info(f"   Logged misclassification analysis ({model_prefix or 'default'})")
    return misclass_df


def log_advanced_metrics_to_tensorboard(writer, y_test, y_pred, model_prefix="", step=0):
    """
    Log advanced metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        y_test: True labels
        y_pred: Predicted labels
        model_prefix: Prefix for TensorBoard tags (e.g., "rf", "dnn")
        step: TensorBoard step
    
    Returns:
        Dict with overall_metrics and per_crop_metrics
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    # Get unique labels
    labels = sorted(list(set(y_test) | set(y_pred)))
    
    # Create figure
    fig, metrics_dict = create_advanced_metrics_figure(y_test, y_pred, labels)
    
    # Log figure
    tag = _prefix_tag('metrics/overall', model_prefix)
    writer.add_figure(tag, fig, step, close=True)
    
    # Log scalars
    writer.add_scalar(_prefix_tag('metrics/accuracy', model_prefix), metrics_dict['accuracy'], step)
    writer.add_scalar(_prefix_tag('metrics/cohen_kappa', model_prefix), metrics_dict['cohen_kappa'], step)
    writer.add_scalar(_prefix_tag('metrics/matthews_corrcoef', model_prefix), metrics_dict['matthews_corrcoef'], step)
    writer.add_scalar(_prefix_tag('metrics/macro_precision', model_prefix), metrics_dict['precision_macro'], step)
    writer.add_scalar(_prefix_tag('metrics/macro_recall', model_prefix), metrics_dict['recall_macro'], step)
    writer.add_scalar(_prefix_tag('metrics/macro_f1', model_prefix), metrics_dict['f1_macro'], step)
    
    # Compute per-crop metrics for return
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0, average=None
    )
    
    per_crop_metrics = {}
    for i, label in enumerate(labels):
        per_crop_metrics[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    # Compute weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0, average='weighted'
    )
    
    metrics_dict['weighted_avg'] = {
        'precision': float(precision_w),
        'recall': float(recall_w),
        'f1_score': float(f1_w)
    }
    
    logger.info(f"   Logged advanced metrics ({model_prefix or 'default'})")
    
    return {
        'overall_metrics': metrics_dict,
        'per_crop_metrics': per_crop_metrics
    }


def log_training_metrics_to_tensorboard(writer, config, metrics, y_test, y_pred, model_prefix="", model_params=None):
    """
    Log basic training metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        config: Pipeline configuration
        metrics: Dict with train_accuracy, test_accuracy, etc.
        y_test: True labels
        y_pred: Predicted labels
        model_prefix: Prefix for TensorBoard tags (e.g., "rf", "dnn")
        model_params: Optional dict of model hyperparameters (overrides config)
    """
    # Log accuracy as scalars
    writer.add_scalar(_prefix_tag('training/train_accuracy', model_prefix), metrics['train_accuracy'], 0)
    writer.add_scalar(_prefix_tag('training/test_accuracy', model_prefix), metrics['test_accuracy'], 0)
    
    # Log hyperparameters
    if model_params:
        for param_name, param_value in model_params.items():
            if isinstance(param_value, (int, float)):
                writer.add_scalar(_prefix_tag(f'hyperparameters/{param_name}', model_prefix), param_value, 0)
    else:
        # Fallback to config (legacy behavior)
        hparams = config.get('model', {}).get('hyperparameters', {})
        writer.add_scalar(_prefix_tag('hyperparameters/n_estimators', model_prefix), hparams.get('n_estimators', 100), 0)
        writer.add_scalar(_prefix_tag('hyperparameters/max_depth', model_prefix), hparams.get('max_depth', 10), 0)
        writer.add_scalar(_prefix_tag('hyperparameters/min_samples_split', model_prefix), hparams.get('min_samples_split', 5), 0)
    
    writer.add_scalar(_prefix_tag('data/n_train_samples', model_prefix), metrics['n_train_samples'], 0)
    writer.add_scalar(_prefix_tag('data/n_test_samples', model_prefix), metrics['n_test_samples'], 0)
    
    logger.info(f"   Logged training metrics ({model_prefix or 'default'})")


def log_dnn_training_history_to_tensorboard(writer, history, model_prefix="dnn"):
    """
    Log DNN training history (loss/accuracy curves) to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        history: Dict with 'loss', 'val_loss', 'accuracy', 'val_accuracy' keys
        model_prefix: Prefix for TensorBoard tags
    """
    if history is None:
        return
    
    # Log training curves
    if 'loss' in history:
        for epoch, loss in enumerate(history['loss']):
            writer.add_scalar(_prefix_tag('training/loss', model_prefix), loss, epoch)
    
    if 'val_loss' in history:
        for epoch, val_loss in enumerate(history['val_loss']):
            writer.add_scalar(_prefix_tag('training/val_loss', model_prefix), val_loss, epoch)
    
    if 'accuracy' in history:
        for epoch, acc in enumerate(history['accuracy']):
            writer.add_scalar(_prefix_tag('training/epoch_accuracy', model_prefix), acc, epoch)
    
    if 'val_accuracy' in history:
        for epoch, val_acc in enumerate(history['val_accuracy']):
            writer.add_scalar(_prefix_tag('training/epoch_val_accuracy', model_prefix), val_acc, epoch)
    
    logger.info(f"   Logged DNN training history ({model_prefix})")


def run_comprehensive_evaluation(model, X_test, y_test, feature_names, writer, model_prefix="", step=0, num_runs=1):
    """
    Run comprehensive evaluation and log to TensorBoard.
    
    Args:
        model: Trained model (sklearn Pipeline or BaseModel)
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        writer: TensorBoard SummaryWriter
        model_prefix: Prefix for TensorBoard tags (e.g., "rf", "dnn")
        step: TensorBoard step
        num_runs: Unused (kept for backward compatibility)
    
    Returns:
        Dict with evaluation results
    """
    prefix_str = f" [{model_prefix}]" if model_prefix else ""
    logger.info("=" * 70)
    logger.info(f"COMPREHENSIVE EVALUATION{prefix_str}")
    logger.info("=" * 70)
    
    # Make predictions - handle both sklearn and BaseModel
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        raise ValueError("Model must have a predict() method")
    
    labels = sorted(list(set(y_test) | set(y_pred)))
    
    results = {}
    
    # Log all visualizations with model prefix
    logger.info("   Logging confusion matrices...")
    log_confusion_matrix_to_tensorboard(writer, y_test, y_pred, labels, model_prefix, step)
    
    logger.info("   Logging per-crop metrics...")
    metrics_df = log_per_crop_metrics_to_tensorboard(writer, y_test, y_pred, labels, model_prefix, step)
    results['metrics_dataframe'] = metrics_df
    
    logger.info("   Logging feature importance...")
    # Handle both Pipeline and BaseModel
    if hasattr(model, 'pipeline') and model.pipeline is not None:
        # BaseModel with sklearn pipeline
        fi_df = log_feature_importance_to_tensorboard(writer, model.pipeline, feature_names, model_prefix, step)
    elif hasattr(model, 'get_feature_importance'):
        # BaseModel with get_feature_importance method
        importance_dict = model.get_feature_importance()
        if importance_dict:
            # Create figure manually
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
            names, values = zip(*sorted_items) if sorted_items else ([], [])
            ax.barh(range(len(names)), values)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance (Top 15)')
            plt.tight_layout()
            
            tag = _prefix_tag('feature_importance/top_15', model_prefix)
            writer.add_figure(tag, fig, step, close=True)
            
            fi_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
            fi_df = fi_df.sort_values('Importance', ascending=False)
            logger.info(f"   Logged feature importance ({model_prefix or 'default'})")
        else:
            fi_df = None
            logger.info(f"   No feature importance available for {model_prefix or 'model'}")
    else:
        # Standard sklearn model
        fi_df = log_feature_importance_to_tensorboard(writer, model, feature_names, model_prefix, step)
    
    if fi_df is not None:
        results['feature_importance_dataframe'] = fi_df
    
    logger.info("   Logging misclassification patterns...")
    misclass_df = log_misclassification_analysis_to_tensorboard(writer, y_test, y_pred, labels, model_prefix, step)
    results['misclassification_dataframe'] = misclass_df
    
    logger.info("   Logging advanced metrics...")
    advanced_metrics = log_advanced_metrics_to_tensorboard(writer, y_test, y_pred, model_prefix, step)
    results['advanced_metrics'] = advanced_metrics
    
    # Log text summary
    summary_text = f"""
Model Evaluation Summary ({model_prefix or 'default'})
{'=' * 40}
Accuracy: {advanced_metrics['overall_metrics']['accuracy']:.2%}
Cohen's Kappa: {advanced_metrics['overall_metrics']['cohen_kappa']:.3f}
Matthews Corr: {advanced_metrics['overall_metrics']['matthews_corrcoef']:.3f}
Macro F1: {advanced_metrics['overall_metrics']['f1_macro']:.2%}
Weighted F1: {advanced_metrics['overall_metrics']['weighted_avg']['f1_score']:.2%}

Test Samples: {len(y_test)}
Feature Count: {len(feature_names)}
"""
    tag = _prefix_tag('evaluation_summary', model_prefix)
    writer.add_text(tag, summary_text, step)
    
    logger.info("=" * 70)
    logger.info(f"EVALUATION COMPLETE{prefix_str}")
    logger.info("=" * 70)
    logger.info(f"   Accuracy: {advanced_metrics['overall_metrics']['accuracy']:.2%}")
    logger.info(f"   Features: {len(feature_names)}")
    
    return results
