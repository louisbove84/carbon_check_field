"""
SIMPLIFIED TensorBoard utilities
=================================
Fixes:
1. Single, simple image conversion method
2. No excessive flushes
3. Cleaner logging flow
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
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


def _convert_figure_to_tensorboard_image(fig):
    """
    SIMPLIFIED: Convert matplotlib figure to TensorBoard-compatible image.
    Returns numpy array in CHW format, float32, normalized to [0, 1].
    """
    # Save figure to BytesIO buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    
    # Load as PIL Image
    image = Image.open(buf)
    
    # Convert to RGB (TensorBoard requires RGB)
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            # White background for transparency
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        else:
            image = image.convert('RGB')
    
    # Convert to numpy array (HWC format: height, width, channels)
    image_array = np.array(image, dtype=np.uint8)
    
    # Convert to CHW format (TensorBoard requirement)
    if len(image_array.shape) == 3:
        image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
    
    # Normalize to [0, 1] as float32
    image_array = image_array.astype(np.float32) / 255.0
    
    buf.close()
    
    return image_array


def log_confusion_matrix_to_tensorboard(writer, y_true, y_pred, labels, step=0):
    """Log confusion matrices to TensorBoard."""
    figures = create_confusion_matrix_figures(y_true, y_pred, labels)
    
    for name, fig in figures.items():
        image_array = _convert_figure_to_tensorboard_image(fig)
        writer.add_image(f'confusion_matrix/{name}', image_array, step, dataformats='CHW')
        plt.close(fig)
    
    logger.info("✅ Logged confusion matrices")


def log_per_crop_metrics_to_tensorboard(writer, y_true, y_pred, labels, step=0):
    """Log per-crop metrics to TensorBoard."""
    figures, metrics_df = create_per_crop_metrics_figures(y_true, y_pred, labels)
    
    for name, fig in figures.items():
        image_array = _convert_figure_to_tensorboard_image(fig)
        writer.add_image(f'per_crop_metrics/{name}', image_array, step, dataformats='CHW')
        plt.close(fig)
    
    # Log scalars for each crop
    for _, row in metrics_df.iterrows():
        crop = row['Crop']
        writer.add_scalar(f'crop/{crop}/precision', row['Precision'], step)
        writer.add_scalar(f'crop/{crop}/recall', row['Recall'], step)
        writer.add_scalar(f'crop/{crop}/f1_score', row['F1-Score'], step)
    
    logger.info("✅ Logged per-crop metrics")
    return metrics_df


def log_feature_importance_to_tensorboard(writer, model, feature_names, step=0):
    """Log feature importance to TensorBoard."""
    result = create_feature_importance_figure(model, feature_names)
    
    if result is None:
        logger.warning("⚠️  No feature importance to log")
        return None
    
    fig, filtered_names, filtered_importances = result
    
    # Log the figure
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('feature_importance/top_15', image_array, step, dataformats='CHW')
    plt.close(fig)
    
    # Log top features as scalars
    if filtered_names and filtered_importances:
        for name, importance in zip(list(filtered_names)[:20], list(filtered_importances)[:20]):
            writer.add_scalar(f'feature_importance/{name}', float(importance), step)
    
    logger.info("✅ Logged feature importance")
    
    # Create DataFrame for return (optional, for backward compatibility)
    import pandas as pd
    importance_df = pd.DataFrame({
        'Feature': filtered_names,
        'Importance': filtered_importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df


def log_misclassification_analysis_to_tensorboard(writer, y_true, y_pred, labels, step=0):
    """Log misclassification patterns to TensorBoard."""
    figures, misclass_df = create_misclassification_analysis_figure(y_true, y_pred, labels)
    
    if not figures:
        logger.info("🎉 No misclassifications to log")
        return misclass_df
    
    for name, fig in figures.items():
        image_array = _convert_figure_to_tensorboard_image(fig)
        writer.add_image(f'misclassification/{name}', image_array, step, dataformats='CHW')
        plt.close(fig)
    
    logger.info("✅ Logged misclassification analysis")
    return misclass_df


def log_advanced_metrics_to_tensorboard(writer, y_test, y_pred, step=0):
    """Log advanced metrics to TensorBoard."""
    from sklearn.metrics import precision_recall_fscore_support
    
    # Get unique labels
    labels = sorted(list(set(y_test) | set(y_pred)))
    
    # Create figure
    fig, metrics_dict = create_advanced_metrics_figure(y_test, y_pred, labels)
    
    # Log image
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('metrics/overall', image_array, step, dataformats='CHW')
    plt.close(fig)
    
    # Log scalars
    writer.add_scalar('metrics/accuracy', metrics_dict['accuracy'], step)
    writer.add_scalar('metrics/cohen_kappa', metrics_dict['cohen_kappa'], step)
    writer.add_scalar('metrics/matthews_corrcoef', metrics_dict['matthews_corrcoef'], step)
    writer.add_scalar('metrics/macro_precision', metrics_dict['precision_macro'], step)
    writer.add_scalar('metrics/macro_recall', metrics_dict['recall_macro'], step)
    writer.add_scalar('metrics/macro_f1', metrics_dict['f1_macro'], step)
    
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
    
    logger.info("✅ Logged advanced metrics")
    
    return {
        'overall_metrics': metrics_dict,
        'per_crop_metrics': per_crop_metrics
    }


def log_training_metrics_to_tensorboard(writer, config, metrics, y_test, y_pred):
    """Log basic training metrics to TensorBoard."""
    # Log accuracy
    writer.add_scalar('training/train_accuracy', metrics['train_accuracy'], 0)
    writer.add_scalar('training/test_accuracy', metrics['test_accuracy'], 0)
    
    # Log hyperparameters
    hparams = {
        'n_estimators': config.get('training', {}).get('n_estimators', 100),
        'max_depth': config.get('training', {}).get('max_depth', 15),
        'n_train_samples': metrics['n_train_samples'],
        'n_test_samples': metrics['n_test_samples']
    }
    
    writer.add_hparams(hparams, {'accuracy': metrics['test_accuracy']})
    
    logger.info("✅ Logged training metrics")


def run_comprehensive_evaluation(model, X_test, y_test, feature_names, writer, step=0, num_runs=1):
    """
    SIMPLIFIED: Run comprehensive evaluation and log to TensorBoard.
    Removed complex progression logic - just log once.
    """
    logger.info("=" * 70)
    logger.info("🔍 COMPREHENSIVE EVALUATION")
    logger.info("=" * 70)
    
    # Make predictions
    y_pred = model.predict(X_test)
    labels = sorted(list(set(y_test) | set(y_pred)))
    
    results = {}
    
    # Log all visualizations
    logger.info("📊 Logging confusion matrices...")
    log_confusion_matrix_to_tensorboard(writer, y_test, y_pred, labels, step)
    
    logger.info("📈 Logging per-crop metrics...")
    metrics_df = log_per_crop_metrics_to_tensorboard(writer, y_test, y_pred, labels, step)
    results['metrics_dataframe'] = metrics_df
    
    logger.info("🔑 Logging feature importance...")
    fi_df = log_feature_importance_to_tensorboard(writer, model, feature_names, step)
    if fi_df is not None:
        results['feature_importance_dataframe'] = fi_df
    
    logger.info("🔍 Logging misclassification patterns...")
    misclass_df = log_misclassification_analysis_to_tensorboard(writer, y_test, y_pred, labels, step)
    results['misclassification_dataframe'] = misclass_df
    
    logger.info("📊 Logging advanced metrics...")
    advanced_metrics = log_advanced_metrics_to_tensorboard(writer, y_test, y_pred, step)
    results['advanced_metrics'] = advanced_metrics
    
    # Log text summary
    summary_text = f"""
Model Evaluation Summary
========================
Accuracy: {advanced_metrics['overall_metrics']['accuracy']:.2%}
Cohen's Kappa: {advanced_metrics['overall_metrics']['cohen_kappa']:.3f}
Matthews Corr: {advanced_metrics['overall_metrics']['matthews_corrcoef']:.3f}
Macro F1: {advanced_metrics['overall_metrics']['f1_macro']:.2%}
Weighted F1: {advanced_metrics['overall_metrics']['weighted_avg']['f1_score']:.2%}

Test Samples: {len(y_test)}
Feature Count: {len(feature_names)}
"""
    writer.add_text('evaluation_summary', summary_text, step)
    
    logger.info("=" * 70)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Accuracy: {advanced_metrics['overall_metrics']['accuracy']:.2%}")
    logger.info(f"Features: {len(feature_names)}")
    
    return results

