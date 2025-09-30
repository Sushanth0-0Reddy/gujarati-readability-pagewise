#!/usr/bin/env python3
"""
Page-wise SVM Training Script for Gujarati Readability Classification
Trains SVM models using extracted backbone features with limited hyperparameter search.
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class PagewiseSVMTrainer:
    def __init__(self, backbone_name):
        self.backbone_name = backbone_name
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_embeddings(self):
        """Load embeddings from the specified backbone"""
        self.logger.info(f"Loading {self.backbone_name} embeddings...")
        
        # Find the embeddings directory
        embeddings_base = Path('readability_training_pagewise/embeddings')
        embedding_dirs = list(embeddings_base.glob(f'{self.backbone_name}_pagewise_embeddings_*'))
        
        if not embedding_dirs:
            raise FileNotFoundError(f"No embeddings found for backbone: {self.backbone_name}")
        
        # Use the most recent embeddings
        embeddings_dir = max(embedding_dirs, key=lambda x: x.stat().st_mtime)
        self.logger.info(f"Using embeddings from: {embeddings_dir}")
        
        # Load features and labels
        train_features = np.load(embeddings_dir / 'train_features.npy')
        test_features = np.load(embeddings_dir / 'test_features.npy')
        train_labels = np.load(embeddings_dir / 'train_labels.npy')
        test_labels = np.load(embeddings_dir / 'test_labels.npy')
        
        # Load metadata
        with open(embeddings_dir / 'feature_info.json', 'r') as f:
            self.feature_info = json.load(f)
            
        self.logger.info(f"Loaded embeddings:")
        self.logger.info(f"  Train features: {train_features.shape}")
        self.logger.info(f"  Test features: {test_features.shape}")
        self.logger.info(f"  Feature dimension: {self.feature_info['feature_dimension']}")
        
        return train_features, test_features, train_labels, test_labels
    
    def create_output_directory(self):
        """Create output directory for results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f'readability_training_pagewise/experiments/pagewise_svm_{self.backbone_name}_nonreadable_{timestamp}')
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {output_dir}")
        return output_dir
    
    def train_svm(self, X_train, X_test, y_train, y_test):
        """Train SVM with hyperparameter tuning"""
        self.logger.info("🚀 Starting SVM training for PAGE-WISE readability classification...")
        
        # Standardize features (important for SVM)
        self.logger.info("📊 Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Parameter grid for SVM (16 combinations: 2×2×2×2)
        param_grid = {
            'C': [1.0, 10.0],                    # 2 values - regularization parameter
            'kernel': ['rbf', 'linear'],         # 2 values - kernel type
            'gamma': ['scale', 'auto'],          # 2 values - kernel coefficient
            'class_weight': ['balanced', None]   # 2 values - handle class imbalance
        }
        
        self.logger.info(f"🔍 Hyperparameter grid search with {np.prod([len(v) for v in param_grid.values()])} combinations")
        
        # GridSearchCV with 3-fold cross-validation
        svm = SVC(random_state=42, probability=True)  # probability=True for ROC AUC
        # Use F1 score with non-readable as positive class
        nonreadable_f1_scorer = make_scorer(f1_score, pos_label=0)
        grid_search = GridSearchCV(
            estimator=svm,
            param_grid=param_grid,
            cv=3,
            scoring=nonreadable_f1_scorer,
            n_jobs=-1,
            verbose=1
        )
        
        # Train
        self.logger.info("🏋️ Training SVM...")
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        best_svm = grid_search.best_estimator_
        self.logger.info(f"✅ Best SVM parameters: {grid_search.best_params_}")
        self.logger.info(f"✅ Best CV F1 score: {grid_search.best_score_:.4f}")
        
        # Predictions
        train_pred = best_svm.predict(X_train_scaled)
        test_pred = best_svm.predict(X_test_scaled)
        train_pred_proba = best_svm.predict_proba(X_train_scaled)[:, 0]  # Probability of non-readable
        test_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 0]  # Probability of non-readable
        
        # Calculate metrics
        results = self.calculate_metrics(
            y_train, y_test, train_pred, test_pred, 
            train_pred_proba, test_pred_proba
        )
        
        return best_svm, scaler, results, grid_search.best_params_
    
    def calculate_metrics(self, y_train, y_test, train_pred, test_pred, 
                         train_pred_proba, test_pred_proba):
        """Calculate comprehensive metrics"""
        
        results = {
            'train_metrics': {
                'accuracy': float(accuracy_score(y_train, train_pred)),
                'precision': float(precision_score(y_train, train_pred, pos_label=0)),
                'recall': float(recall_score(y_train, train_pred, pos_label=0)),
                'f1_score': float(f1_score(y_train, train_pred, pos_label=0)),
                'roc_auc': float(roc_auc_score(y_train, train_pred_proba))
            },
            'test_metrics': {
                'accuracy': float(accuracy_score(y_test, test_pred)),
                'precision': float(precision_score(y_test, test_pred, pos_label=0)),
                'recall': float(recall_score(y_test, test_pred, pos_label=0)),
                'f1_score': float(f1_score(y_test, test_pred, pos_label=0)),
                'roc_auc': float(roc_auc_score(y_test, test_pred_proba))
            },
            'confusion_matrix': {
                'train': confusion_matrix(y_train, train_pred).tolist(),
                'test': confusion_matrix(y_test, test_pred).tolist()
            }
        }
        
        # Log results
        self.logger.info("📊 Training Results (Non-readable as positive):")
        self.logger.info(f"  Train Accuracy: {results['train_metrics']['accuracy']:.4f}")
        self.logger.info(f"  Train F1: {results['train_metrics']['f1_score']:.4f}")
        self.logger.info(f"  Train ROC AUC: {results['train_metrics']['roc_auc']:.4f}")
        
        self.logger.info("📊 Test Results (Non-readable as positive):")
        self.logger.info(f"  Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        self.logger.info(f"  Test F1: {results['test_metrics']['f1_score']:.4f}")
        self.logger.info(f"  Test ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
        
        return results
    
    def save_results(self, output_dir, model, scaler, results, best_params):
        """Save model, scaler, and results"""
        self.logger.info("💾 Saving results...")
        
        # Save model and scaler
        joblib.dump(model, output_dir / 'best_svm_model.pkl')
        joblib.dump(scaler, output_dir / 'feature_scaler.pkl')
        
        # Save results
        final_results = {
            'model_type': 'SVM',
            'backbone_name': self.backbone_name,
            'feature_dimension': self.feature_info['feature_dimension'],
            'training_date': datetime.now().isoformat(),
            'train_samples': int(self.feature_info['train_samples']),
            'test_samples': int(self.feature_info['test_samples']),
            'best_hyperparameters': best_params,
            'cv_folds': 3,
            'hyperparameter_combinations': 16,
            'feature_scaling': 'StandardScaler',
            'results': results
        }
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        self.logger.info(f"✅ Results saved to: {output_dir}")
        return output_dir

def main():
    parser = argparse.ArgumentParser(description='Train SVM for page-wise readability classification')
    parser.add_argument('--backbone', 
                       choices=['efficientnet', 'resnet50', 'yolov8n', 'layoutxlm', 'dinov2', 'dinov3'], 
                       required=True,
                       help='Backbone model to use for training')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 PAGE-WISE SVM TRAINING PIPELINE")
    print("="*70)
    print(f"📋 Backbone: {args.backbone}")
    print(f"🎯 Task: Page-wise readability classification")
    print(f"🔧 Model: Support Vector Machine (SVM)")
    print(f"🔍 Hyperparameter combinations: 16")
    print()
    
    try:
        # Initialize trainer
        trainer = PagewiseSVMTrainer(args.backbone)
        
        # Load embeddings
        X_train, X_test, y_train, y_test = trainer.load_embeddings()
        
        # Create output directory
        output_dir = trainer.create_output_directory()
        
        # Train SVM
        model, scaler, results, best_params = trainer.train_svm(X_train, X_test, y_train, y_test)
        
        # Save results
        trainer.save_results(output_dir, model, scaler, results, best_params)
        
        print("🎉 SVM training completed successfully!")
        print(f"📁 Output: {output_dir}")
        
        trainer.logger.info(f"{args.backbone} page-wise SVM training completed successfully")
        
        return output_dir
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
