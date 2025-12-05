"""
Comprehensive model evaluation utilities for crop classification.

This module provides detailed analysis beyond TensorBoard's basic metrics:
- Enhanced confusion matrices with percentages
- Per-crop precision/recall/F1 curves
- Feature importance analysis
- Misclassification analysis
- Class balance visualization
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_enhanced_confusion_matrix(y_true, y_pred, labels, output_dir):
    """
    Generate multiple confusion matrix visualizations:
    1. Raw counts
    2. Percentages by true label
    3. Normalized (0-1)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 1. Raw counts with percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Counts + Percentages
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=axes[0],
                cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix\n(Count + Percentage)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Plot 2: Percentage only (easier to read)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn', 
                xticklabels=labels, yticklabels=labels, ax=axes[1],
                vmin=0, vmax=100, cbar_kws={'label': 'Percentage'})
    axes[1].set_title('Confusion Matrix\n(Percentage)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    # Plot 3: Normalized (0-1)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='viridis', 
                xticklabels=labels, yticklabels=labels, ax=axes[2],
                vmin=0, vmax=1, cbar_kws={'label': 'Normalized'})
    axes[2].set_title('Confusion Matrix\n(Normalized)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('True Label', fontsize=12)
    axes[2].set_xlabel('Predicted Label', fontsize=12)
    
    # Rotate labels
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrix_enhanced.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Enhanced confusion matrices saved to {output_path}")
    return output_path


def generate_per_crop_metrics_chart(y_true, y_pred, labels, output_dir):
    """
    Generate bar charts showing precision, recall, and F1 for each crop.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None
    )
    
    # Create DataFrame for easier plotting
    metrics_df = pd.DataFrame({
        'Crop': labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Precision, Recall, F1 grouped bar chart
    x = np.arange(len(labels))
    width = 0.25
    
    axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8, color='#2ecc71')
    axes[0, 0].bar(x, recall, width, label='Recall', alpha=0.8, color='#3498db')
    axes[0, 0].bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#e74c3c')
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_title('Per-Crop Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: F1-Score by crop (sorted)
    sorted_df = metrics_df.sort_values('F1-Score', ascending=True)
    colors = ['#e74c3c' if x < 0.8 else '#f39c12' if x < 0.9 else '#2ecc71' 
              for x in sorted_df['F1-Score']]
    axes[0, 1].barh(sorted_df['Crop'], sorted_df['F1-Score'], color=colors, alpha=0.8)
    axes[0, 1].set_xlabel('F1-Score', fontsize=12)
    axes[0, 1].set_title('F1-Score by Crop (Sorted)', fontsize=14, fontweight='bold')
    axes[0, 1].axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Target: 0.9')
    axes[0, 1].axvline(x=0.8, color='orange', linestyle='--', alpha=0.5, label='Warning: 0.8')
    axes[0, 1].legend()
    axes[0, 1].set_xlim([0, 1.1])
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Sample support by crop
    axes[1, 0].bar(labels, support, alpha=0.8, color='#9b59b6')
    axes[1, 0].set_ylabel('Test Samples', fontsize=12)
    axes[1, 0].set_title('Test Set Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(support):
        axes[1, 0].text(i, v + max(support)*0.02, str(int(v)), 
                       ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Precision vs Recall scatter
    axes[1, 1].scatter(recall, precision, s=support*3, alpha=0.6, c=range(len(labels)), 
                      cmap='tab10', edgecolors='black', linewidth=2)
    axes[1, 1].set_xlabel('Recall', fontsize=12)
    axes[1, 1].set_ylabel('Precision', fontsize=12)
    axes[1, 1].set_title('Precision-Recall Trade-off\n(size = test samples)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlim([0, 1.1])
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1, 1].grid(alpha=0.3)
    
    # Add crop labels to scatter points
    for i, crop in enumerate(labels):
        axes[1, 1].annotate(crop, (recall[i], precision[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'per_crop_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Per-crop metrics chart saved to {output_path}")
    return output_path, metrics_df


def generate_feature_importance_chart(model, feature_names, output_dir):
    """
    Generate feature importance visualization for RandomForest.
    """
    # Extract the RandomForest classifier from the pipeline
    if hasattr(model, 'named_steps'):
        rf_model = model.named_steps['classifier']
    else:
        rf_model = model
    
    if not isinstance(rf_model, RandomForestClassifier):
        logger.warning("⚠️  Model is not RandomForest, skipping feature importance")
        return None
    
    # Get feature importances
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices],
        'Rank': range(1, len(indices) + 1)
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Top 20 features (bar chart)
    top_n = min(20, len(feature_names))
    top_features = importance_df.head(top_n)
    
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    axes[0].barh(range(top_n), top_features['Importance'], color=colors, alpha=0.8)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_features['Feature'])
    axes[0].set_xlabel('Importance', fontsize=12)
    axes[0].set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Cumulative importance
    cumsum = np.cumsum(importance_df['Importance'])
    axes[1].plot(range(1, len(cumsum) + 1), cumsum, 'o-', linewidth=2, markersize=4)
    axes[1].axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% threshold')
    axes[1].axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    axes[1].set_xlabel('Number of Features', fontsize=12)
    axes[1].set_ylabel('Cumulative Importance', fontsize=12)
    axes[1].set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    # Find number of features for 80% and 90%
    n_80 = np.argmax(cumsum >= 0.8) + 1
    n_90 = np.argmax(cumsum >= 0.9) + 1
    axes[1].axvline(x=n_80, color='r', linestyle=':', alpha=0.5)
    axes[1].axvline(x=n_90, color='orange', linestyle=':', alpha=0.5)
    axes[1].text(n_80, 0.05, f'{n_80} features\n(80%)', ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].text(n_90, 0.15, f'{n_90} features\n(90%)', ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Feature importance chart saved to {output_path}")
    
    # Save top features to JSON
    json_path = os.path.join(output_dir, 'feature_importance.json')
    importance_df.to_json(json_path, orient='records', indent=2)
    logger.info(f"✅ Feature importance data saved to {json_path}")
    
    return output_path, importance_df


def generate_misclassification_analysis(y_true, y_pred, labels, output_dir):
    """
    Analyze patterns in misclassifications.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Find most common misclassifications (off-diagonal)
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
        return None, misclass_df
    
    # Plot top 10 misclassification pairs
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_n = min(10, len(misclass_df))
    top_misclass = misclass_df.head(top_n)
    
    labels_text = [f"{row['True']} → {row['Predicted']}" 
                   for _, row in top_misclass.iterrows()]
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, top_n))
    bars = ax.barh(range(top_n), top_misclass['Count'], color=colors, alpha=0.8)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels_text)
    ax.set_xlabel('Number of Misclassifications', fontsize=12)
    ax.set_title(f'Top {top_n} Misclassification Patterns', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (_, row) in enumerate(top_misclass.iterrows()):
        ax.text(row['Count'] + max(top_misclass['Count'])*0.02, i, 
               f"{row['Percentage']:.1f}%", 
               va='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'misclassification_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Misclassification analysis saved to {output_path}")
    
    # Save to JSON
    json_path = os.path.join(output_dir, 'misclassifications.json')
    misclass_df.to_json(json_path, orient='records', indent=2)
    logger.info(f"✅ Misclassification data saved to {json_path}")
    
    return output_path, misclass_df


def generate_advanced_metrics_report(y_true, y_pred, y_pred_proba, labels, output_dir):
    """
    Generate comprehensive metrics report including Cohen's Kappa, MCC, etc.
    """
    # Basic metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None
    )
    
    # Advanced metrics
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Overall accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    # Weighted averages
    precision_weighted = np.average(precision, weights=support)
    recall_weighted = np.average(recall, weights=support)
    f1_weighted = np.average(f1, weights=support)
    
    # Macro averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    
    # Create report
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
    
    # Save to JSON
    json_path = os.path.join(output_dir, 'advanced_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ Advanced metrics report saved to {json_path}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['Accuracy', 'Cohen Kappa', 'Matthews Corr.', 
                     'Macro F1', 'Weighted F1']
    metrics_values = [accuracy, cohen_kappa, mcc, f1_macro, f1_weighted]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'advanced_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Advanced metrics visualization saved to {output_path}")
    
    return report


def run_comprehensive_evaluation(model, X_test, y_test, feature_names, output_dir):
    """
    Run all evaluation metrics and generate comprehensive report.
    
    Args:
        model: Trained sklearn pipeline
        X_test: Test features
        y_test: True test labels
        feature_names: List of feature names
        output_dir: Directory to save outputs
        
    Returns:
        dict: Dictionary containing all evaluation results
    """
    logger.info("=" * 70)
    logger.info("🔍 COMPREHENSIVE MODEL EVALUATION")
    logger.info("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    labels = sorted(set(y_test))
    
    results = {}
    
    # 1. Enhanced Confusion Matrix
    logger.info("📊 Generating enhanced confusion matrices...")
    cm_path = generate_enhanced_confusion_matrix(y_test, y_pred, labels, output_dir)
    results['confusion_matrix_path'] = cm_path
    
    # 2. Per-Crop Metrics
    logger.info("📈 Generating per-crop metrics charts...")
    metrics_path, metrics_df = generate_per_crop_metrics_chart(y_test, y_pred, labels, output_dir)
    results['per_crop_metrics_path'] = metrics_path
    results['metrics_dataframe'] = metrics_df
    
    # 3. Feature Importance
    logger.info("🔑 Generating feature importance analysis...")
    fi_result = generate_feature_importance_chart(model, feature_names, output_dir)
    if fi_result:
        fi_path, fi_df = fi_result
        results['feature_importance_path'] = fi_path
        results['feature_importance_dataframe'] = fi_df
    
    # 4. Misclassification Analysis
    logger.info("🔍 Analyzing misclassification patterns...")
    misclass_result = generate_misclassification_analysis(y_test, y_pred, labels, output_dir)
    if misclass_result:
        misclass_path, misclass_df = misclass_result
        if misclass_path:
            results['misclassification_path'] = misclass_path
        results['misclassification_dataframe'] = misclass_df
    
    # 5. Advanced Metrics Report
    logger.info("📋 Generating advanced metrics report...")
    advanced_metrics = generate_advanced_metrics_report(y_test, y_pred, y_pred_proba, labels, output_dir)
    results['advanced_metrics'] = advanced_metrics
    
    logger.info("=" * 70)
    logger.info("✅ COMPREHENSIVE EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"All evaluation artifacts saved to: {output_dir}")
    
    # Print summary to console
    logger.info("\n📊 EVALUATION SUMMARY:")
    logger.info(f"   Accuracy: {advanced_metrics['overall_metrics']['accuracy']:.2%}")
    logger.info(f"   Cohen's Kappa: {advanced_metrics['overall_metrics']['cohen_kappa']:.3f}")
    logger.info(f"   Matthews Corr: {advanced_metrics['overall_metrics']['matthews_corrcoef']:.3f}")
    logger.info(f"   Macro F1: {advanced_metrics['overall_metrics']['macro_avg']['f1_score']:.2%}")
    logger.info(f"   Weighted F1: {advanced_metrics['overall_metrics']['weighted_avg']['f1_score']:.2%}")
    
    return results

