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
from visualization_utils import (
    create_confusion_matrix_figures,
    create_per_crop_metrics_figures,
    create_feature_importance_figure,
    create_advanced_metrics_figure
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _convert_figure_to_tensorboard_image(fig):
    """
    Convert matplotlib figure to TensorBoard-compatible image array.
    Returns numpy array in CHW format (channels, height, width) with values in [0, 1].
    
    Tries multiple approaches to ensure compatibility with TensorBoard on GCP.
    """
    import tempfile
    
    # Method 1: Try using temporary file (more reliable for GCP TensorBoard)
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Save figure to temporary file
        fig.savefig(tmp_path, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1, facecolor='white')
        
        # Load image from file
        image = Image.open(tmp_path)
        
        # Convert RGBA to RGB if needed (TensorBoard expects RGB)
        if image.mode == 'RGBA':
            # Create white background for RGBA images
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array (HWC format)
        image_array = np.array(image, dtype=np.uint8)
        
        # Ensure CHW format (channels, height, width) - TensorBoard requirement
        if len(image_array.shape) == 3:
            # HWC -> CHW
            image_array = np.transpose(image_array, (2, 0, 1))
        elif len(image_array.shape) == 2:
            # Grayscale: add channel dimension
            image_array = np.expand_dims(image_array, axis=0)
        
        # Normalize to [0, 1] range as float32 (required for TensorBoard display)
        image_array = image_array.astype(np.float32) / 255.0
        
        # Ensure values are in valid range [0, 1]
        image_array = np.clip(image_array, 0.0, 1.0)
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        return image_array
    
    except Exception as e:
        logger.warning(f"⚠️  Temp file method failed, trying BytesIO: {e}")
        
        # Fallback: Use BytesIO method
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1, facecolor='white')
        buf.seek(0)
        
        image = Image.open(buf)
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image, dtype=np.uint8)
        
        if len(image_array.shape) == 3:
            image_array = np.transpose(image_array, (2, 0, 1))
        elif len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=0)
        
        image_array = image_array.astype(np.float32) / 255.0
        image_array = np.clip(image_array, 0.0, 1.0)
        
        buf.close()
        
        return image_array


def log_confusion_matrix_to_tensorboard(writer, y_true, y_pred, labels, step=0):
    """
    Log confusion matrix visualizations to TensorBoard.
    Uses shared visualization functions to ensure consistency with MLflow.
    """
    # Use shared visualization function
    figures = create_confusion_matrix_figures(y_true, y_pred, labels)
    
    # Log each figure to TensorBoard
    image_array = _convert_figure_to_tensorboard_image(figures['counts_and_percent'])
    writer.add_image('confusion_matrix/counts_and_percent', image_array, step, dataformats='CHW')
    plt.close(figures['counts_and_percent'])
    
    image_array = _convert_figure_to_tensorboard_image(figures['percentage'])
    writer.add_image('confusion_matrix/percentage', image_array, step, dataformats='CHW')
    plt.close(figures['percentage'])
    
    image_array = _convert_figure_to_tensorboard_image(figures['normalized'])
    writer.add_image('confusion_matrix/normalized', image_array, step, dataformats='CHW')
    plt.close(figures['normalized'])
    
    logger.info("✅ Logged confusion matrices to TensorBoard")


def log_per_crop_metrics_to_tensorboard(writer, y_true, y_pred, labels, step=0):
    """
    Log per-crop metrics charts to TensorBoard.
    Uses shared visualization functions to ensure consistency with MLflow.
    """
    # Use shared visualization function
    figures, metrics_df = create_per_crop_metrics_figures(y_true, y_pred, labels)
    
    # Log comparison chart
    image_array = _convert_figure_to_tensorboard_image(figures['comparison'])
    writer.add_image('per_crop_metrics/comparison', image_array, step, dataformats='CHW')
    plt.close(figures['comparison'])
    
    # Log F1 sorted chart
    image_array = _convert_figure_to_tensorboard_image(figures['f1_sorted'])
    writer.add_image('per_crop_metrics/f1_sorted', image_array, step, dataformats='CHW')
    plt.close(figures['f1_sorted'])
    
    # Log individual metrics as scalars
    for _, row in metrics_df.iterrows():
        writer.add_scalar(f'per_crop/precision/{row["Crop"]}', row['Precision'], step)
        writer.add_scalar(f'per_crop/recall/{row["Crop"]}', row['Recall'], step)
        writer.add_scalar(f'per_crop/f1/{row["Crop"]}', row['F1-Score'], step)
        writer.add_scalar(f'per_crop/support/{row["Crop"]}', row['Support'], step)
    
    logger.info("✅ Logged per-crop metrics to TensorBoard")
    return metrics_df


def log_feature_importance_to_tensorboard(writer, model, feature_names, step=0):
    """
    Log feature importance to TensorBoard.
    Uses shared visualization functions to ensure consistency with MLflow.
    """
    # Use shared visualization function
    result = create_feature_importance_figure(model, feature_names)
    if result is None:
        logger.warning("⚠️  Model does not have feature_importances_ attribute")
        return None
    
    fig, filtered_names, filtered_importances = result
    
    # Log the figure
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('feature_importance/top_features', image_array, step, dataformats='CHW')
    plt.close(fig)
    
    # Log top features as scalars
    indices = np.argsort(filtered_importances)[::-1][:10]
    for idx in indices:
        writer.add_scalar(f'feature_importance/{filtered_names[idx]}', filtered_importances[idx], step)
    
    logger.info("✅ Logged feature importance to TensorBoard")
    logger.info(f"   Feature names in importance: {list(filtered_names[:10])}")
    
    # Create DataFrame for return
    importance_df = pd.DataFrame({
        'Feature': filtered_names,
        'Importance': filtered_importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df
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
    
    # Filter out location and elevation features (shouldn't exist, but filter just in case)
    # REMOVED: lat/lon features — model no longer uses geographic cheating
    # REMOVED: elevation features — elevation removed from feature set
    excluded_features = {'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 'latitude', 'longitude',
                        'elevation_binned', 'elevation_m', 'elevation'}
    filtered_indices = [i for i in indices if feature_names[i] not in excluded_features]
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
    writer.add_image('feature_importance/top_features', image_array, step, dataformats='CHW')
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
    writer.add_image('feature_importance/cumulative', image_array, step, dataformats='CHW')
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
    writer.add_image('misclassification/top_patterns', image_array, step, dataformats='CHW')
    plt.close(fig)
    
    logger.info("✅ Logged misclassification analysis to TensorBoard")
    return misclass_df


def log_advanced_metrics_to_tensorboard(writer, y_true, y_pred, step=0):
    """
    Log advanced metrics to TensorBoard and return report dict.
    Uses shared visualization functions to ensure consistency with MLflow.
    """
    labels = sorted(set(y_true))
    
    # Use shared visualization function
    fig, metrics_dict = create_advanced_metrics_figure(y_true, y_pred, labels)
    
    # Log scalar metrics
    writer.add_scalar('metrics/accuracy', metrics_dict['accuracy'], step)
    writer.add_scalar('metrics/cohen_kappa', metrics_dict['cohen_kappa'], step)
    writer.add_scalar('metrics/matthews_corrcoef', metrics_dict['matthews_corrcoef'], step)
    writer.add_scalar('metrics/macro_precision', metrics_dict['precision_macro'], step)
    writer.add_scalar('metrics/macro_recall', metrics_dict['recall_macro'], step)
    writer.add_scalar('metrics/macro_f1', metrics_dict['f1_macro'], step)
    
    # Log the figure
    image_array = _convert_figure_to_tensorboard_image(fig)
    writer.add_image('metrics/overall', image_array, step, dataformats='CHW')
    plt.close(fig)
    
    logger.info("✅ Logged advanced metrics to TensorBoard")
    return metrics_dict


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
