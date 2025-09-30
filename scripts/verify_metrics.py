#!/usr/bin/env python3
"""
Verify that metrics calculated from confusion matrix match reported values
"""

import numpy as np
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from pathlib import Path

def verify_experiment_metrics(experiment_dir):
    """Verify metrics for a specific experiment"""
    print(f"Verifying metrics for: {experiment_dir}")
    
    # Load results
    results_path = Path(experiment_dir) / 'results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load test labels
    embeddings_base = Path('readability_training_pagewise/embeddings')
    embedding_dirs = list(embeddings_base.glob('efficientnet_pagewise_embeddings_*'))
    embeddings_dir = max(embedding_dirs, key=lambda x: x.stat().st_mtime)
    test_labels = np.load(embeddings_dir / 'test_labels.npy')
    
    # Get predictions
    test_pred = np.array(results['test_predictions'])
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, test_pred)
    print(f"Confusion Matrix:")
    print(cm)
    print(f"[[{cm[0,0]} {cm[0,1]}]  <- Non-readable: {cm[0,0]} correct, {cm[0,1]} missed")
    print(f" [{cm[1,0]} {cm[1,1]}]] <- Readable: {cm[1,0]} false positives, {cm[1,1]} correct")
    
    # Calculate metrics manually (non-readable as positive)
    manual_precision = precision_score(test_labels, test_pred, pos_label=0)
    manual_recall = recall_score(test_labels, test_pred, pos_label=0)
    manual_f1 = f1_score(test_labels, test_pred, pos_label=0)
    
    # Get reported metrics
    reported_precision = results['test_metrics']['precision']
    reported_recall = results['test_metrics']['recall'] 
    reported_f1 = results['test_metrics']['f1']
    
    print(f"\nMETRICS COMPARISON (Non-readable as positive):")
    print(f"{'Metric':<12} {'Manual':<10} {'Reported':<10} {'Match':<8}")
    print(f"{'='*50}")
    print(f"{'Precision':<12} {manual_precision:<10.4f} {reported_precision:<10.4f} {'✅' if abs(manual_precision - reported_precision) < 0.001 else '❌'}")
    print(f"{'Recall':<12} {manual_recall:<10.4f} {reported_recall:<10.4f} {'✅' if abs(manual_recall - reported_recall) < 0.001 else '❌'}")
    print(f"{'F1 Score':<12} {manual_f1:<10.4f} {reported_f1:<10.4f} {'✅' if abs(manual_f1 - reported_f1) < 0.001 else '❌'}")
    
    print(f"\nINTERPRETATION:")
    print(f"• Precision: {manual_precision*100:.1f}% of pages predicted as non-readable are actually non-readable")
    print(f"• Recall: {manual_recall*100:.1f}% of actual non-readable pages are correctly detected")
    print(f"• F1: {manual_f1*100:.1f}% balanced performance for non-readable detection")
    
    return manual_precision, manual_recall, manual_f1

if __name__ == "__main__":
    # Check the SMOTE experiment
    smote_experiment = "readability_training_pagewise/experiments/xgboost_efficientnet_smote_nonreadable_20250808_091638"
    if Path(smote_experiment).exists():
        verify_experiment_metrics(smote_experiment)
    else:
        print("SMOTE experiment directory not found")
