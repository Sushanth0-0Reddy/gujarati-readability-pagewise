#!/usr/bin/env python3
"""
Test DINOv3 Integration Script

This script tests the DINOv3 integration with the existing pipeline
by running feature extraction and training with a small subset of data.
"""

import os
import sys
import torch
from pathlib import Path
import argparse

def test_dinov3_extraction():
    """Test DINOv3 feature extraction"""
    print("🧪 Testing DINOv3 Feature Extraction...")
    
    # Import the extraction script
    sys.path.append('/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/readability_training_pagewise/scripts')
    from extract_pagewise_features import extract_pagewise_features
    
    try:
        # Run DINOv3 feature extraction
        print("📊 Starting DINOv3 feature extraction...")
        output_dir = extract_pagewise_features('dinov3')
        print(f"✅ DINOv3 feature extraction completed!")
        print(f"📁 Features saved to: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"❌ DINOv3 feature extraction failed: {e}")
        raise

def test_dinov3_training(embeddings_dir):
    """Test XGBoost training with DINOv3 features"""
    print("\n🧪 Testing DINOv3 + XGBoost Training...")
    
    # Import the training script
    from train_pagewise_xgboost import PagewiseXGBoostTrainer
    from datetime import datetime
    
    try:
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path(f'readability_training_pagewise/experiments/test_dinov3_xgboost_{timestamp}')
        
        # Initialize trainer
        trainer = PagewiseXGBoostTrainer('readability_training_pagewise/embeddings', experiment_dir, 'dinov3')
        
        # Load features
        X_train, X_test, y_train, y_test, train_names, test_names = trainer.load_pagewise_features()
        print(f"📊 Loaded DINOv3 features: Train={X_train.shape}, Test={X_test.shape}")
        
        # Train model
        model, results = trainer.train_xgboost(X_train, X_test, y_train, y_test)
        
        print(f"✅ DINOv3 + XGBoost training completed!")
        print(f"📁 Results saved to: {experiment_dir}")
        print(f"📊 Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"📊 Test F1 Score: {results['test_metrics']['f1']:.4f}")
        
        return experiment_dir, results
        
    except Exception as e:
        print(f"❌ DINOv3 + XGBoost training failed: {e}")
        raise

def test_model_comparison():
    """Compare DINOv3 with other models"""
    print("\n📊 Model Comparison Summary:")
    print("="*60)
    
    # Load results from different experiments
    experiments_dir = Path('readability_training_pagewise/experiments')
    
    models_results = {}
    
    # Look for recent experiments
    for exp_dir in experiments_dir.glob('*xgboost*'):
        if 'dinov3' in exp_dir.name:
            models_results['DINOv3'] = exp_dir
        elif 'efficientnet' in exp_dir.name:
            models_results['EfficientNet'] = exp_dir
        elif 'resnet50' in exp_dir.name:
            models_results['ResNet50'] = exp_dir
    
    print(f"Found {len(models_results)} model experiments for comparison")
    for model_name, exp_path in models_results.items():
        print(f"  • {model_name}: {exp_path.name}")

def main():
    parser = argparse.ArgumentParser(description='Test DINOv3 integration with the readability pipeline')
    parser.add_argument('--extract-only', action='store_true', 
                       help='Only run feature extraction, skip training')
    parser.add_argument('--train-only', action='store_true',
                       help='Only run training (assumes features already extracted)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare results across different models')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🧪 DINOV3 INTEGRATION TEST")
    print("="*70)
    print("📋 Testing DINOv3 integration with Gujarati readability pipeline")
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    try:
        if args.compare:
            test_model_comparison()
            return
        
        embeddings_dir = None
        
        if not args.train_only:
            # Test feature extraction
            embeddings_dir = test_dinov3_extraction()
        
        if not args.extract_only:
            # Test training
            if embeddings_dir is None:
                # Find existing DINOv3 embeddings
                embeddings_base = Path('readability_training_pagewise/embeddings')
                dinov3_dirs = list(embeddings_base.glob('dinov3_pagewise_embeddings_*'))
                if dinov3_dirs:
                    embeddings_dir = max(dinov3_dirs, key=lambda x: x.stat().st_mtime)
                    print(f"📂 Using existing DINOv3 embeddings: {embeddings_dir}")
                else:
                    print("❌ No DINOv3 embeddings found. Run extraction first.")
                    return
            
            experiment_dir, results = test_dinov3_training(embeddings_dir)
        
        print("\n🎉 DINOv3 integration test completed successfully!")
        print("🔬 You can now use DINOv3 as a backbone in your readability pipeline")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


