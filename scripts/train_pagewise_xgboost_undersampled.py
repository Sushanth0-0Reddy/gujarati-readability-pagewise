#!/usr/bin/env python3
"""
Page-wise XGBoost Training Script with Undersampling

Trains XGBoost classifier on pre-extracted features for page-wise readability classification
with dataset balancing using undersampling techniques (Random, CNN, Tomek, OSS, etc.).
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, make_scorer
import warnings
warnings.filterwarnings('ignore')
import argparse

# Try to import required libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.under_sampling import (
        RandomUnderSampler, 
        TomekLinks, 
        EditedNearestNeighbours, 
        RepeatedEditedNearestNeighbours,
        AllKNN,
        CondensedNearestNeighbour,
        OneSidedSelection,
        NeighbourhoodCleaningRule,
        InstanceHardnessThreshold
    )
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class PagewiseXGBoostTrainerUndersampled:
    def __init__(self, embeddings_dir, experiment_dir, backbone_name='efficientnet', sampling_strategy='random'):
        self.embeddings_dir = Path(embeddings_dir)
        self.experiment_dir = Path(experiment_dir)
        self.backbone_name = backbone_name
        self.sampling_strategy = sampling_strategy
        self.model = None
        self.grid_search = None
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Log configuration
        self.logger.info(f"🚀 Initializing PAGE-WISE XGBoost Trainer with Undersampling")
        self.logger.info(f"📂 Embeddings directory: {self.embeddings_dir}")
        self.logger.info(f"📂 Experiment directory: {self.experiment_dir}")
        self.logger.info(f"🧠 Backbone model: {self.backbone_name}")
        self.logger.info(f"⚖️ Sampling strategy: {self.sampling_strategy}")
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.experiment_dir / "training.log"
        
        # Create logger
        self.logger = logging.getLogger('XGBoostTrainer')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def get_undersampler(self):
        """Get the appropriate undersampling technique."""
        if not IMBALANCED_LEARN_AVAILABLE:
            self.logger.error("❌ imbalanced-learn not available. Install with: pip install imbalanced-learn")
            return None
            
        undersamplers = {
            'random': RandomUnderSampler(random_state=42),
            'tomek': TomekLinks(),
            'enn': EditedNearestNeighbours(n_neighbors=3),
            'renn': RepeatedEditedNearestNeighbours(n_neighbors=3),
            'allknn': AllKNN(n_neighbors=3),
            'cnn': CondensedNearestNeighbour(random_state=42, n_neighbors=1),
            'oss': OneSidedSelection(random_state=42, n_neighbors=1),
            'ncr': NeighbourhoodCleaningRule(n_neighbors=3),
            'iht': InstanceHardnessThreshold(random_state=42)
        }
        
        if self.sampling_strategy not in undersamplers:
            self.logger.warning(f"⚠️ Unknown sampling strategy: {self.sampling_strategy}. Using Random.")
            return undersamplers['random']
            
        return undersamplers[self.sampling_strategy]
        
    def load_pagewise_features(self):
        """Load page-wise features and labels with proper mapping."""
        self.logger.info("📚 Loading page-wise features and labels...")
        
        # Find embedding directories for the specified backbone
        embedding_dirs = list(self.embeddings_dir.glob(f"{self.backbone_name}_pagewise_embeddings_*"))
        if not embedding_dirs:
            raise FileNotFoundError(f"No {self.backbone_name} page-wise embeddings found in {self.embeddings_dir}")
        
        # Use the most recent one
        embedding_dir = max(embedding_dirs, key=lambda x: x.stat().st_mtime)
        self.logger.info(f"📂 Using embeddings from: {embedding_dir}")
        
        # Load features and labels
        train_features = np.load(embedding_dir / "train_features.npy")
        train_labels = np.load(embedding_dir / "train_labels.npy")
        test_features = np.load(embedding_dir / "test_features.npy")
        test_labels = np.load(embedding_dir / "test_labels.npy")
        
        # Load mapping files for verification
        with open(embedding_dir / "train_paths.json", 'r') as f:
            train_paths = json.load(f)
        with open(embedding_dir / "test_paths.json", 'r') as f:
            test_paths = json.load(f)
            
        # Log dataset information
        self.logger.info(f"📊 Train features: {train_features.shape}")
        self.logger.info(f"📊 Train labels: {train_labels.shape}")
        self.logger.info(f"📊 Test features: {test_features.shape}")
        self.logger.info(f"📊 Test labels: {test_labels.shape}")
        self.logger.info(f"📊 Feature dimension: {train_features.shape[1]}")
        self.logger.info(f"📊 Classification type: page_wise")
        
        # Verify mapping consistency
        if len(train_features) == len(train_paths) and len(test_features) == len(test_paths):
            self.logger.info("✅ Feature-to-image mapping verified")
        else:
            self.logger.error("❌ Feature-to-image mapping mismatch!")
            
        # Log class distribution before undersampling
        unique_train, counts_train = np.unique(train_labels, return_counts=True)
        unique_test, counts_test = np.unique(test_labels, return_counts=True)
        
        self.logger.info(f"📊 Original train class distribution: {dict(zip(unique_train, counts_train))}")
        self.logger.info(f"📊 Test class distribution: {dict(zip(unique_test, counts_test))}")
        
        return train_features, train_labels, test_features, test_labels, train_paths, test_paths
        
    def apply_undersampling(self, X_train, y_train):
        """Apply undersampling to balance the training dataset."""
        self.logger.info(f"⚖️ Applying undersampling technique: {self.sampling_strategy}")
        
        undersampler = self.get_undersampler()
        if undersampler is None:
            self.logger.error("❌ Could not initialize undersampler")
            return X_train, y_train
            
        try:
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
            
            # Log new class distribution
            unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
            self.logger.info(f"📊 Resampled train class distribution: {dict(zip(unique_resampled, counts_resampled))}")
            self.logger.info(f"📊 Original samples: {len(X_train)} → Resampled samples: {len(X_resampled)}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.error(f"❌ Undersampling failed: {str(e)}")
            self.logger.info("📊 Proceeding with original unbalanced dataset")
            return X_train, y_train
            
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost with hyperparameter tuning."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        self.logger.info("🚀 Starting XGBoost training for PAGE-WISE readability classification...")
        
        # Apply undersampling to training data
        X_train_resampled, y_train_resampled = self.apply_undersampling(X_train, y_train)
        
        # MEDIUM hyperparameter grid for faster training
        self.logger.info("🔧 MEDIUM hyperparameter tuning for PAGE-WISE classification...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8],
            'reg_alpha': [0],
            'reg_lambda': [1]
        }
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        cv_folds = 3
        
        self.logger.info(f"📊 Total parameter combinations: {total_combinations}")
        self.logger.info(f"🔄 Cross-validation folds: {cv_folds}")
        self.logger.info(f"⏱️  Estimated time: {total_combinations * cv_folds * 0.5 / 60:.1f} - {total_combinations * cv_folds * 1.5 / 60:.1f} minutes")
        
        # Create XGBoost classifier
        xgb_classifier = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        self.logger.info("🚀 Starting hyperparameter search...")
        start_time = datetime.now()
        
        # Use F1 score with non-readable as positive class
        nonreadable_f1_scorer = make_scorer(f1_score, pos_label=0)
        self.grid_search = GridSearchCV(
            estimator=xgb_classifier,
            param_grid=param_grid,
            scoring=nonreadable_f1_scorer,
            cv=cv_folds,
            n_jobs=-1,
            verbose=2
        )
        
        # Fit on resampled training data
        self.grid_search.fit(X_train_resampled, y_train_resampled)
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        self.logger.info("✅ Hyperparameter search completed!")
        self.logger.info(f"⏱️  Total training time: {training_time}")
        
        # Get best model
        self.model = self.grid_search.best_estimator_
        
        # Log best parameters
        self.logger.info("🏆 Best parameters found:")
        for param, value in self.grid_search.best_params_.items():
            self.logger.info(f"   {param}: {value}")
        self.logger.info(f"🏆 Best CV F1 score: {self.grid_search.best_score_:.4f}")
        
        # Show top parameter combinations
        results_df = pd.DataFrame(self.grid_search.cv_results_)
        top_results = results_df.nlargest(5, 'mean_test_score')[['mean_test_score', 'std_test_score', 'params']]
        
        self.logger.info("🔝 Top parameter combinations:")
        for i, (idx, row) in enumerate(top_results.iterrows(), 1):
            mean_score = row['mean_test_score']
            std_score = row['std_test_score']
            params = row['params']
            self.logger.info(f"   {i}. F1: {mean_score:.4f} (±{std_score:.4f}) - {params}")
            
        return training_time
        
    def evaluate_model(self, X_train_original, y_train_original, X_test, y_test):
        """Evaluate the trained model on both original training and test sets."""
        self.logger.info("📊 Evaluating best model on training and test sets...")
        
        results = {}
        
        # Evaluate on original training set (not resampled)
        self.logger.info(f"🔍 Evaluating on original train set ({len(X_train_original)} samples)...")
        train_pred = self.model.predict(X_train_original)
        train_pred_proba = self.model.predict_proba(X_train_original)[:, 0]  # Probability of non-readable
        
        train_metrics = {
            'accuracy': accuracy_score(y_train_original, train_pred),
            'precision': precision_score(y_train_original, train_pred, pos_label=0),
            'recall': recall_score(y_train_original, train_pred, pos_label=0),
            'f1': f1_score(y_train_original, train_pred, pos_label=0),
            'roc_auc': auc(*roc_curve(y_train_original, train_pred_proba, pos_label=0)[:2])
        }
        
        self.logger.info("📊 TRAIN RESULTS (Non-readable as positive):")
        self.logger.info(f"   📈 Accuracy:  {train_metrics['accuracy']:.4f}")
        self.logger.info(f"   📈 Precision: {train_metrics['precision']:.4f}")
        self.logger.info(f"   📈 Recall:    {train_metrics['recall']:.4f}")
        self.logger.info(f"   📈 F1 Score:  {train_metrics['f1']:.4f}")
        self.logger.info(f"   📈 ROC AUC:   {train_metrics['roc_auc']:.4f}")
        
        # Log distribution
        unique_true, counts_true = np.unique(y_train_original, return_counts=True)
        unique_pred, counts_pred = np.unique(train_pred, return_counts=True)
        self.logger.info(f"   📊 True distribution: {dict(zip(unique_true, counts_true))}")
        self.logger.info(f"   📊 Pred distribution: {dict(zip(unique_pred, counts_pred))}")
        
        # Evaluate on test set
        self.logger.info(f"🔍 Evaluating on test set ({len(X_test)} samples)...")
        test_pred = self.model.predict(X_test)
        test_pred_proba = self.model.predict_proba(X_test)[:, 0]  # Probability of non-readable
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred, pos_label=0),
            'recall': recall_score(y_test, test_pred, pos_label=0),
            'f1': f1_score(y_test, test_pred, pos_label=0),
            'roc_auc': auc(*roc_curve(y_test, test_pred_proba, pos_label=0)[:2])
        }
        
        self.logger.info("📊 TEST RESULTS (Non-readable as positive):")
        self.logger.info(f"   📈 Accuracy:  {test_metrics['accuracy']:.4f}")
        self.logger.info(f"   📈 Precision: {test_metrics['precision']:.4f}")
        self.logger.info(f"   📈 Recall:    {test_metrics['recall']:.4f}")
        self.logger.info(f"   📈 F1 Score:  {test_metrics['f1']:.4f}")
        self.logger.info(f"   📈 ROC AUC:   {test_metrics['roc_auc']:.4f}")
        
        # Log distribution
        unique_true, counts_true = np.unique(y_test, return_counts=True)
        unique_pred, counts_pred = np.unique(test_pred, return_counts=True)
        self.logger.info(f"   📊 True distribution: {dict(zip(unique_true, counts_true))}")
        self.logger.info(f"   📊 Pred distribution: {dict(zip(unique_pred, counts_pred))}")
        
        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_predictions': train_pred.tolist(),
            'train_probabilities': train_pred_proba.tolist(),
            'test_predictions': test_pred.tolist(),
            'test_probabilities': test_pred_proba.tolist()
        }
        
        return results
        
    def save_results(self, results, training_time):
        """Save all results to files."""
        
        # Save performance metrics
        metrics_file = self.experiment_dir / "performance_metrics.txt"
        self.logger.info(f"📄 Performance metrics saved to: {metrics_file}")
        
        with open(metrics_file, 'w') as f:
            f.write("PAGE-WISE READABILITY CLASSIFICATION - XGBoost with Undersampling\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Backbone Model: {self.backbone_name}\n")
            f.write(f"Sampling Strategy: {self.sampling_strategy}\n")
            f.write(f"Training Time: {training_time}\n")
            f.write(f"Best CV F1 Score: {self.grid_search.best_score_:.4f}\n\n")
            
            f.write("BEST PARAMETERS:\n")
            f.write("-" * 20 + "\n")
            for param, value in self.grid_search.best_params_.items():
                f.write(f"{param}: {value}\n")
            f.write("\n")
            
            f.write("TRAINING SET PERFORMANCE (Original Data):\n")
            f.write("-" * 40 + "\n")
            for metric, value in results['train_metrics'].items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
            f.write("\n")
            
            f.write("TEST SET PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            for metric, value in results['test_metrics'].items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
        
        # Save model
        model_file = self.experiment_dir / "best_model.pkl"
        import pickle
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        self.logger.info(f"💾 Model saved to: {model_file}")
        
        # Save detailed results
        results_file = self.experiment_dir / "results.json"
        detailed_results = {
            'backbone_model': self.backbone_name,
            'sampling_strategy': self.sampling_strategy,
            'model_type': f'xgboost_{self.backbone_name}_undersampled_{self.sampling_strategy}',
            'training_time': str(training_time),
            'best_params': self.grid_search.best_params_,
            'best_cv_score': float(self.grid_search.best_score_),
            'train_metrics': results['train_metrics'],
            'test_metrics': results['test_metrics'],
            'train_predictions': results['train_predictions'],
            'train_probabilities': results['train_probabilities'],
            'test_predictions': results['test_predictions'],
            'test_probabilities': results['test_probabilities']
        }
        
        # Convert to serializable format
        detailed_results = convert_to_serializable(detailed_results)
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        self.logger.info(f"💾 Results saved to: {results_file}")
        
        # Save CV results
        cv_results_file = self.experiment_dir / "cv_results.csv"
        cv_results_df = pd.DataFrame(self.grid_search.cv_results_)
        cv_results_df.to_csv(cv_results_file, index=False)
        self.logger.info(f"💾 CV results saved to: {cv_results_file}")
        
    def create_visualizations(self, X_train, y_train, X_test, y_test, results):
        """Create visualization plots."""
        self.logger.info("📊 Creating visualization plots...")
        
        # Create plots directory
        plots_dir = self.experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        plot_count = 0
        
        # 1. Feature importance plot
        try:
            self.logger.info("   📈 Creating feature importance plot...")
            feature_importance = self.model.feature_importances_
            
            plt.figure(figsize=(12, 8))
            indices = np.argsort(feature_importance)[::-1][:20]  # Top 20 features
            
            plt.title(f'Top 20 Feature Importance - XGBoost ({self.backbone_name} + {self.sampling_strategy})', fontsize=14, fontweight='bold')
            plt.bar(range(len(indices)), feature_importance[indices])
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.xticks(range(len(indices)), indices, rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            plot_count += 1
        except Exception as e:
            self.logger.error(f"❌ Failed to create feature importance plot: {e}")
        
        # 2. Confusion matrix
        try:
            self.logger.info("   📈 Creating confusion matrix...")
            test_pred = np.array(results['test_predictions'])
            cm = confusion_matrix(y_test, test_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-readable', 'Readable'],
                       yticklabels=['Non-readable', 'Readable'])
            plt.title(f'Confusion Matrix - XGBoost ({self.backbone_name} + {self.sampling_strategy})', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            plot_count += 1
        except Exception as e:
            self.logger.error(f"❌ Failed to create confusion matrix: {e}")
        
        # 3. ROC curve
        try:
            self.logger.info("   📈 Creating ROC curve...")
            test_proba = np.array(results['test_probabilities'])
            fpr, tpr, _ = roc_curve(y_test, test_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - XGBoost ({self.backbone_name} + {self.sampling_strategy})', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            plot_count += 1
        except Exception as e:
            self.logger.error(f"❌ Failed to create ROC curve: {e}")
        
        # 4. Hyperparameter search visualization
        try:
            self.logger.info("   📈 Creating hyperparameter search visualization...")
            cv_results_df = pd.DataFrame(self.grid_search.cv_results_)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Hyperparameter Search Results - XGBoost ({self.backbone_name} + {self.sampling_strategy})', fontsize=16, fontweight='bold')
            
            # Learning rate vs F1
            axes[0,0].scatter(cv_results_df['param_learning_rate'], cv_results_df['mean_test_score'], alpha=0.7)
            axes[0,0].set_xlabel('Learning Rate')
            axes[0,0].set_ylabel('CV F1 Score')
            axes[0,0].set_title('Learning Rate vs F1 Score')
            axes[0,0].grid(alpha=0.3)
            
            # Max depth vs F1
            axes[0,1].scatter(cv_results_df['param_max_depth'], cv_results_df['mean_test_score'], alpha=0.7)
            axes[0,1].set_xlabel('Max Depth')
            axes[0,1].set_ylabel('CV F1 Score')
            axes[0,1].set_title('Max Depth vs F1 Score')
            axes[0,1].grid(alpha=0.3)
            
            # N estimators vs F1
            axes[1,0].scatter(cv_results_df['param_n_estimators'], cv_results_df['mean_test_score'], alpha=0.7)
            axes[1,0].set_xlabel('N Estimators')
            axes[1,0].set_ylabel('CV F1 Score')
            axes[1,0].set_title('N Estimators vs F1 Score')
            axes[1,0].grid(alpha=0.3)
            
            # Subsample vs F1
            axes[1,1].scatter(cv_results_df['param_subsample'], cv_results_df['mean_test_score'], alpha=0.7)
            axes[1,1].set_xlabel('Subsample')
            axes[1,1].set_ylabel('CV F1 Score')
            axes[1,1].set_title('Subsample vs F1 Score')
            axes[1,1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "hyperparameter_search.png", dpi=300, bbox_inches='tight')
            plt.close()
            plot_count += 1
        except Exception as e:
            self.logger.error(f"❌ Failed to create hyperparameter search visualization: {e}")
        
        self.logger.info(f"✅ All {plot_count} plots created successfully")
        self.logger.info(f"📁 Plots saved to: {plots_dir}")
        
    def run_training(self):
        """Run the complete training pipeline."""
        try:
            # Load features
            X_train, y_train, X_test, y_test, train_paths, test_paths = self.load_pagewise_features()
            
            # Train model (with undersampling applied internally)
            training_time = self.train_xgboost(X_train, y_train, X_test, y_test)
            
            # Evaluate model (on original training data and test data)
            results = self.evaluate_model(X_train, y_train, X_test, y_test)
            
            # Save results
            self.save_results(results, training_time)
            
            # Create visualizations
            self.create_visualizations(X_train, y_train, X_test, y_test, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost on page-wise features with undersampling')
    parser.add_argument('--backbone', type=str, default='efficientnet',
                      choices=['efficientnet', 'resnet50', 'yolov8n', 'layoutxlm'],
                      help='Backbone model used for feature extraction')
    parser.add_argument('--sampling', type=str, default='random',
                      choices=['random', 'tomek', 'enn', 'renn', 'allknn', 'cnn', 'oss', 'ncr', 'iht'],
                      help='Undersampling strategy to use')
    
    args = parser.parse_args()
    backbone = args.backbone
    sampling_strategy = args.sampling
    
    # Check if required libraries are available
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost not available. Install with: pip install xgboost")
        sys.exit(1)
        
    if not IMBALANCED_LEARN_AVAILABLE:
        print("❌ imbalanced-learn not available. Install with: pip install imbalanced-learn")
        sys.exit(1)
    
    # Setup paths
    embeddings_dir = Path("readability_training_pagewise/embeddings")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f'readability_training_pagewise/experiments/xgboost_{backbone}_{sampling_strategy}_under_nonreadable_{timestamp}')
    
    print("🚀 Starting PAGE-WISE XGBoost training with undersampling...")
    print(f"🧠 Backbone: {backbone}")
    print(f"⚖️ Sampling: {sampling_strategy}")
    print(f"📂 Experiment directory: {experiment_dir}")
    
    try:
        # Initialize trainer
        trainer = PagewiseXGBoostTrainerUndersampled(
            embeddings_dir=embeddings_dir,
            experiment_dir=experiment_dir,
            backbone_name=backbone,
            sampling_strategy=sampling_strategy
        )
        
        # Run training
        results = trainer.run_training()
        
        print("✅ PAGE-WISE XGBoost training with undersampling completed!")
        print(f"📁 Results saved to: {experiment_dir}")
        print(f"📊 Best CV F1 Score: {trainer.grid_search.best_score_:.4f}")
        print(f"📊 Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"📊 Test F1 Score: {results['test_metrics']['f1']:.4f}")
        print(f"📊 Test ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
        print("🎉 Training completed successfully!")
        
        # Print final summary
        print("📈 Final Performance:")
        print(f"   Test Accuracy:  {results['test_metrics']['accuracy']*100:.2f}%")
        print(f"   Test Precision: {results['test_metrics']['precision']*100:.2f}%")
        print(f"   Test Recall:    {results['test_metrics']['recall']*100:.2f}%")
        print(f"   Test F1 Score:  {results['test_metrics']['f1']*100:.2f}%")
        print(f"   Test ROC AUC:   {results['test_metrics']['roc_auc']:.3f}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
