#!/usr/bin/env python3
"""
General Prediction Distribution Plot Generator
Creates scatter plots and histograms of predictions vs true labels for any experiment.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

def load_experiment_data(experiment_dir):
    """Load model and data from experiment directory"""
    experiment_path = Path(experiment_dir)
    
    # Load model (try different possible names)
    possible_model_paths = [
        experiment_path / 'best_model.pkl',
        experiment_path / 'model.joblib',
        experiment_path / 'best_model.joblib'
    ]
    
    model_path = None
    for path in possible_model_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError(f"Model not found in: {experiment_path}")
    
    model = joblib.load(model_path)
    
    # Load results
    results_path = experiment_path / 'results.json'
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return model, results

def load_features_and_labels(experiment_dir):
    """Load the original features and labels based on experiment directory name"""
    # Extract backbone name from experiment directory
    experiment_name = Path(experiment_dir).name
    print(f"Experiment directory: {experiment_name}")
    
    # Detect backbone model from experiment directory name
    backbone_model = None
    if 'efficientnet' in experiment_name:
        backbone_model = 'efficientnet'
    elif 'dinov2' in experiment_name:
        backbone_model = 'dinov2'
    elif 'dinov3' in experiment_name:
        backbone_model = 'dinov3'
    elif 'resnet50' in experiment_name:
        backbone_model = 'resnet50'
    elif 'yolov8n' in experiment_name:
        backbone_model = 'yolov8n'
    elif 'layoutxlm' in experiment_name:
        backbone_model = 'layoutxlm'
    else:
        # Default fallback to efficientnet
        backbone_model = 'efficientnet'
        print(f"⚠️  Could not detect backbone from '{experiment_name}', defaulting to efficientnet")
    
    print(f"🔍 Detected backbone model: {backbone_model}")
    
    # Find the most recent embeddings for the detected backbone
    embeddings_base = Path('readability_training_pagewise/embeddings')
    embedding_dirs = list(embeddings_base.glob(f'{backbone_model}_pagewise_embeddings_*'))
    
    if not embedding_dirs:
        raise FileNotFoundError(f"No {backbone_model} embeddings found in {embeddings_base}")
    
    # Use the most recent embeddings
    embeddings_dir = max(embedding_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using embeddings from: {embeddings_dir}")
    
    # Load features
    train_features = np.load(embeddings_dir / 'train_features.npy')
    test_features = np.load(embeddings_dir / 'test_features.npy')
    train_labels = np.load(embeddings_dir / 'train_labels.npy')
    test_labels = np.load(embeddings_dir / 'test_labels.npy')
    
    print(f"✅ Loaded features: Train {train_features.shape}, Test {test_features.shape}")
    
    return train_features, test_features, train_labels, test_labels

def create_prediction_plots(experiment_dir):
    """Create prediction distribution plots"""
    print(f"Creating prediction plots for: {experiment_dir}")
    
    # Load experiment data
    model, results = load_experiment_data(experiment_dir)
    
    # Load features and labels
    train_features, test_features, train_labels, test_labels = load_features_and_labels(experiment_dir)
    
    # Generate predictions and probabilities
    train_pred = model.predict(train_features)
    test_pred = model.predict(test_features)
    train_pred_proba = model.predict_proba(train_features)[:, 0]  # Probability of non-readable
    test_pred_proba = model.predict_proba(test_features)[:, 0]   # Probability of non-readable
    
    # Create output directory
    output_dir = Path(experiment_dir) / 'prediction_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Scatter Plot: Predictions vs True Labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Train scatter plot
    train_jitter_x = train_labels + np.random.normal(0, 0.05, len(train_labels))
    train_jitter_y = train_pred_proba + np.random.normal(0, 0.02, len(train_pred_proba))
    
    ax1.scatter(train_jitter_x, train_jitter_y, alpha=0.6, s=30)
    ax1.set_xlabel('True Labels (0=Non-readable, 1=Readable)')
    ax1.set_ylabel('Predicted Probability of Non-readable')
    ax1.set_title('Train Set: Predicted Probabilities vs True Labels')
    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Add text annotations
    train_acc = results['train_metrics']['accuracy']
    train_f1 = results['train_metrics']['f1']
    ax1.text(0.02, 0.98, f'Accuracy: {train_acc:.3f}\nF1 (Non-readable): {train_f1:.3f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Test scatter plot
    test_jitter_x = test_labels + np.random.normal(0, 0.05, len(test_labels))
    test_jitter_y = test_pred_proba + np.random.normal(0, 0.02, len(test_pred_proba))
    
    ax2.scatter(test_jitter_x, test_jitter_y, alpha=0.6, s=30)
    ax2.set_xlabel('True Labels (0=Non-readable, 1=Readable)')
    ax2.set_ylabel('Predicted Probability of Non-readable')
    ax2.set_title('Test Set: Predicted Probabilities vs True Labels')
    ax2.set_xlim(-0.3, 1.3)
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Add text annotations
    test_acc = results['test_metrics']['accuracy']
    test_f1 = results['test_metrics']['f1']
    ax2.text(0.02, 0.98, f'Accuracy: {test_acc:.3f}\nF1 (Non-readable): {test_f1:.3f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distribution_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Histograms: Prediction Distribution
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Train set histograms
    ax1.hist(train_pred_proba[train_labels == 0], bins=20, alpha=0.7, label='True Non-readable', color='red')
    ax1.hist(train_pred_proba[train_labels == 1], bins=20, alpha=0.7, label='True Readable', color='blue')
    ax1.set_xlabel('Predicted Probability of Non-readable')
    ax1.set_ylabel('Count')
    ax1.set_title('Train Set: Prediction Distribution by True Label')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test set histograms
    ax2.hist(test_pred_proba[test_labels == 0], bins=20, alpha=0.7, label='True Non-readable', color='red')
    ax2.hist(test_pred_proba[test_labels == 1], bins=20, alpha=0.7, label='True Readable', color='blue')
    ax2.set_xlabel('Predicted Probability of Non-readable')
    ax2.set_ylabel('Count')
    ax2.set_title('Test Set: Prediction Distribution by True Label')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrix style plots
    from sklearn.metrics import confusion_matrix
    
    # Train confusion matrix
    train_cm = confusion_matrix(train_labels, train_pred)
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel('Predicted Label')
    ax3.set_ylabel('True Label')
    ax3.set_title('Train Set: Confusion Matrix')
    ax3.set_xticklabels(['Non-readable', 'Readable'])
    ax3.set_yticklabels(['Non-readable', 'Readable'])
    
    # Test confusion matrix
    test_cm = confusion_matrix(test_labels, test_pred)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
    ax4.set_xlabel('Predicted Label')
    ax4.set_ylabel('True Label')
    ax4.set_title('Test Set: Confusion Matrix')
    ax4.set_xticklabels(['Non-readable', 'Readable'])
    ax4.set_yticklabels(['Non-readable', 'Readable'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distribution_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve and Performance Summary
    from sklearn.metrics import roc_curve, auc
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve for test set
    fpr, tpr, _ = roc_curve(test_labels, test_pred_proba, pos_label=0)  # Non-readable as positive
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve for Non-readable Detection (Test Set)')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Performance metrics summary
    metrics_text = f"""
    TEST SET PERFORMANCE (Non-readable as Positive):
    
    Accuracy:  {test_acc:.3f}
    Precision: {results['test_metrics']['precision']:.3f}
    Recall:    {results['test_metrics']['recall']:.3f}
    F1 Score:  {test_f1:.3f}
    ROC AUC:   {results['test_metrics']['roc_auc']:.3f}
    
    INTERPRETATION:
    • Precision: {results['test_metrics']['precision']*100:.1f}% of predicted non-readable are correct
    • Recall: {results['test_metrics']['recall']*100:.1f}% of actual non-readable are detected
    • F1: {test_f1*100:.1f}% balanced performance for non-readable detection
    """
    
    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.axis('off')
    ax2.set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve_and_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Prediction analysis plots saved to: {output_dir}")
    print(f"📊 Generated files:")
    print(f"   - prediction_distribution_scatter.png")
    print(f"   - prediction_distribution_histograms.png")
    print(f"   - roc_curve_and_summary.png")

def main():
    parser = argparse.ArgumentParser(description='Generate prediction distribution plots for any experiment')
    parser.add_argument('--experiment', required=True, help='Path to experiment directory')
    
    args = parser.parse_args()
    
    if not Path(args.experiment).exists():
        print(f"❌ Experiment directory not found: {args.experiment}")
        return
    
    create_prediction_plots(args.experiment)

if __name__ == "__main__":
    main()
