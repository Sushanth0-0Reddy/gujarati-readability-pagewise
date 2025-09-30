# Usage Guide - Gujarati Readability Classification (Page-wise)

This guide provides step-by-step instructions and practical examples for using the page-wise readability classification system.

## 🎯 **Table of Contents**

1. [Setup and Installation](#setup-and-installation)
2. [Data Preparation](#data-preparation)
3. [Feature Extraction Workflow](#feature-extraction-workflow)
4. [Model Training Workflow](#model-training-workflow)
5. [Prediction Workflow](#prediction-workflow)
6. [Analysis and Visualization](#analysis-and-visualization)
7. [Batch Processing](#batch-processing)
8. [Best Practices](#best-practices)

## 🛠️ **Setup and Installation**

### **1. Environment Setup**

```bash
# Navigate to project directory
cd /root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification

# Create virtual environment (recommended)
python -m venv venv_pagewise
source venv_pagewise/bin/activate  # Linux/Mac
# or
venv_pagewise\Scripts\activate     # Windows

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install xgboost scikit-learn
pip install pandas numpy matplotlib seaborn
pip install openpyxl
pip install tqdm
```

### **2. Optional Dependencies**

```bash
# For oversampling techniques
pip install imbalanced-learn

# For YOLOv8 support
pip install ultralytics

# For advanced visualization
pip install plotly
```

### **3. Verify Installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
```

## 📊 **Data Preparation**

### **1. Required Data Structure**

Ensure your data follows this structure:

```
data/
├── Quality.xlsx                    # Main data file
├── images/                         # Image directory
│   ├── book1_page1.jpg
│   ├── book1_page2.jpg
│   └── ...
└── splits/                         # Train/test splits
    ├── train_split_page_level_splitting_20250819.json
    └── test_split_page_level_splitting_20250819.json
```

### **2. Quality.xlsx Format**

Your Excel file should have these columns:

| Column Name | Description | Example |
|-------------|-------------|---------|
| Book Name | Name of the book | "Sample Gujarati Book" |
| Image Name | Unique identifier | "book1_page001.jpg" |
| Image Path | Full path to image | "data/images/book1_page001.jpg" |
| Readability | Binary label | 0 (non-readable) or 1 (readable) |

### **3. Verify Data Integrity**

```bash
python -c "
import pandas as pd
df = pd.read_excel('data/Quality.xlsx')
print(f'Total images: {len(df)}')
print(f'Labeled images: {df[\"Readability\"].notna().sum()}')
print(f'Books: {df[\"Book Name\"].nunique()}')
print(f'Class distribution:')
print(df['Readability'].value_counts())
"
```

## 🔧 **Feature Extraction Workflow**

### **1. Basic Feature Extraction**

Extract features using different backbone models:

```bash
# EfficientNet (recommended for beginners)
python readability_training_pagewise/scripts/extract_pagewise_features.py --backbone efficientnet

# DINOv2 (good balance of performance and speed)
python readability_training_pagewise/scripts/extract_pagewise_features.py --backbone dinov2

# DINOv3 (latest model, best performance)
python readability_training_pagewise/scripts/extract_pagewise_features.py --backbone dinov3

# ResNet50 (traditional CNN)
python readability_training_pagewise/scripts/extract_pagewise_features.py --backbone resnet50
```

### **2. Monitor Feature Extraction**

The script will output progress information:

```
🚀 EXTRACTING DINOV2 FEATURES FOR PAGE-WISE CLASSIFICATION
📊 Loading page-wise data...
Total images in Quality.xlsx: 629
Images with page-wise labels: 629
Train split images: 501
Test split images: 128
✅ DINOv2 Small loaded via transformers - Feature dim: 384
🔍 Extracting dinov2 features for train set...
  Processed 10/32 batches
  Processed 20/32 batches
  Processed 32/32 batches
✅ Extracted train features shape: (501, 384)
```

### **3. Verify Feature Extraction**

```bash
# Check the generated embeddings
ls -la readability_training_pagewise/embeddings/

# Examine feature info
python -c "
import json
import numpy as np
from pathlib import Path

# Find latest embeddings
embeddings_dir = max(Path('readability_training_pagewise/embeddings').glob('dinov2_pagewise_embeddings_*'))
print(f'Latest embeddings: {embeddings_dir}')

# Load feature info
with open(embeddings_dir / 'feature_info.json') as f:
    info = json.load(f)
    
print(f'Feature dimension: {info[\"feature_dimension\"]}')
print(f'Train samples: {info[\"train_samples\"]}')
print(f'Test samples: {info[\"test_samples\"]}')
print(f'Train class distribution: {info[\"train_class_distribution\"]}')
"
```

## 🚀 **Model Training Workflow**

### **1. Standard Training**

Train XGBoost with basic hyperparameter tuning:

```bash
# Train with DINOv2 features
python readability_training_pagewise/scripts/train_pagewise_xgboost.py --backbone dinov2

# Train with custom output directory
python readability_training_pagewise/scripts/train_pagewise_xgboost.py \
    --backbone dinov2 \
    --output_dir custom_experiments/my_experiment
```

### **2. Training with Oversampling** (Recommended)

Handle class imbalance using oversampling:

```bash
# SMOTE (Synthetic Minority Oversampling Technique)
python readability_training_pagewise/scripts/train_pagewise_xgboost_oversampled.py \
    --backbone dinov2 \
    --sampling_method smote

# ADASYN (Adaptive Synthetic Sampling)
python readability_training_pagewise/scripts/train_pagewise_xgboost_oversampled.py \
    --backbone dinov2 \
    --sampling_method adasyn

# Random Oversampling
python readability_training_pagewise/scripts/train_pagewise_xgboost_oversampled.py \
    --backbone dinov2 \
    --sampling_method random
```

### **3. Training with Undersampling**

Reduce majority class size:

```bash
python readability_training_pagewise/scripts/train_pagewise_xgboost_undersampled.py --backbone dinov2
```

### **4. Monitor Training Progress**

Training will show progress like this:

```
🚀 PAGE-WISE XGBOOST TRAINING
📊 Using pre-extracted DINOv2 features
🏷️  Classification: Page-wise readability labels
🔧 Hyperparameter search with cross-validation

Loading dinov2 page-wise features from embeddings/dinov2_pagewise_embeddings_20250819_112657
✅ Page-wise features loaded successfully
📊 Train features: (501, 384)
📊 Test features: (128, 384)
🚀 Starting XGBoost training for PAGE-WISE readability classification...
🔧 FAST hyperparameter tuning for PAGE-WISE classification...
📊 Total parameter combinations: 4
🔄 Cross-validation folds: 3
⏱️  Estimated time: 1.2 - 3.2 minutes
```

### **5. Examine Training Results**

```bash
# Find latest experiment
EXPERIMENT_DIR=$(ls -td readability_training_pagewise/experiments/xgboost_dinov2_* | head -1)
echo "Latest experiment: $EXPERIMENT_DIR"

# View performance metrics
cat "$EXPERIMENT_DIR/performance_metrics.txt"

# Check training log
tail -20 "$EXPERIMENT_DIR/training.log"
```

## 🔮 **Prediction Workflow**

### **1. Single Book Prediction**

Generate predictions for all pages in a specific book:

```bash
# Basic prediction
python readability_training_pagewise/scripts/predict_single_book.py \
    --book_name "Your Book Name" \
    --backbone dinov2

# With custom settings
python readability_training_pagewise/scripts/predict_single_book.py \
    --book_name "Your Book Name" \
    --backbone dinov2 \
    --quality_file data/Quality.xlsx \
    --output_dir results/ \
    --batch_size 8
```

### **2. Understanding Prediction Output**

The script generates an Excel file with two sheets:

#### **Page_Predictions Sheet:**
| Column | Description |
|--------|-------------|
| Book Name | Original book name |
| Image Name | Page identifier |
| Image Path | Path to image file |
| Readability | True label (if available) |
| predicted_readability | Model prediction (0/1) |
| prediction_probability | Confidence score (0-1) |
| predicted_label | Human-readable prediction |
| true_label | Human-readable true label |
| prediction_correct | Whether prediction matches truth |

#### **Summary Sheet:**
| Metric | Description |
|--------|-------------|
| Total Pages | Number of pages in book |
| Labeled Pages | Pages with ground truth |
| Page Accuracy | Accuracy on labeled pages |
| Book Average Probability | Mean confidence across all pages |
| Book Prediction | Overall book readability |

### **3. Batch Book Prediction**

Process multiple books:

```bash
# Create batch script
cat > predict_multiple_books.sh << 'EOF'
#!/bin/bash

BOOKS=("Book 1" "Book 2" "Book 3" "Book 4")
BACKBONE="dinov2"

for book in "${BOOKS[@]}"; do
    echo "Processing: $book"
    python readability_training_pagewise/scripts/predict_single_book.py \
        --book_name "$book" \
        --backbone "$BACKBONE" \
        --output_dir "batch_results/"
    echo "Completed: $book"
    echo "---"
done
EOF

chmod +x predict_multiple_books.sh
./predict_multiple_books.sh
```

## 📊 **Analysis and Visualization**

### **1. Generate Prediction Plots**

Create comprehensive visualizations:

```bash
# Find your experiment directory
EXPERIMENT_DIR="readability_training_pagewise/experiments/xgboost_dinov2_20250819_112742"

# Generate plots
python readability_training_pagewise/scripts/plot_prediction_distribution_general.py \
    --experiment_dir "$EXPERIMENT_DIR"
```

### **2. Verify Model Metrics**

Double-check reported performance:

```bash
python readability_training_pagewise/scripts/verify_metrics.py "$EXPERIMENT_DIR"
```

### **3. Custom Analysis**

Create custom analysis scripts:

```python
# analyze_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load prediction results
results_file = "results/Your_Book_Name_predictions_20250930_143022.xlsx"
df = pd.read_excel(results_file, sheet_name='Page_Predictions')

# Analyze prediction distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df['prediction_probability'], bins=20, alpha=0.7)
plt.title('Prediction Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
df['predicted_readability'].value_counts().plot(kind='bar')
plt.title('Prediction Distribution')
plt.xlabel('Prediction')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
if 'prediction_correct' in df.columns:
    accuracy_by_prob = df.groupby(pd.cut(df['prediction_probability'], bins=10))['prediction_correct'].mean()
    accuracy_by_prob.plot(kind='bar')
    plt.title('Accuracy by Confidence')
    plt.xlabel('Probability Range')
    plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('custom_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 🔄 **Batch Processing**

### **1. Complete Pipeline Script**

Create an end-to-end pipeline:

```bash
# create_pipeline.sh
#!/bin/bash

BACKBONE="dinov2"
SAMPLING="smote"

echo "=== GUJARATI READABILITY PIPELINE ==="
echo "Backbone: $BACKBONE"
echo "Sampling: $SAMPLING"

# Step 1: Feature Extraction
echo "Step 1: Extracting features..."
python readability_training_pagewise/scripts/extract_pagewise_features.py --backbone "$BACKBONE"

if [ $? -ne 0 ]; then
    echo "Feature extraction failed!"
    exit 1
fi

# Step 2: Model Training
echo "Step 2: Training model..."
python readability_training_pagewise/scripts/train_pagewise_xgboost_oversampled.py \
    --backbone "$BACKBONE" \
    --sampling_method "$SAMPLING"

if [ $? -ne 0 ]; then
    echo "Model training failed!"
    exit 1
fi

# Step 3: Generate visualizations
echo "Step 3: Creating visualizations..."
LATEST_EXPERIMENT=$(ls -td readability_training_pagewise/experiments/xgboost_${BACKBONE}_* | head -1)
python readability_training_pagewise/scripts/plot_prediction_distribution_general.py \
    --experiment_dir "$LATEST_EXPERIMENT"

echo "Pipeline completed successfully!"
echo "Results in: $LATEST_EXPERIMENT"
```

### **2. Model Comparison Script**

Compare different backbone models:

```bash
# compare_models.sh
#!/bin/bash

BACKBONES=("efficientnet" "dinov2" "dinov3" "resnet50")

for backbone in "${BACKBONES[@]}"; do
    echo "=== Processing $backbone ==="
    
    # Extract features
    python readability_training_pagewise/scripts/extract_pagewise_features.py --backbone "$backbone"
    
    # Train model
    python readability_training_pagewise/scripts/train_pagewise_xgboost.py --backbone "$backbone"
    
    echo "Completed $backbone"
    echo "---"
done

echo "All models trained. Check experiments/ directory for results."
```

## 🎯 **Best Practices**

### **1. Model Selection Guidelines**

| Use Case | Recommended Backbone | Reason |
|----------|---------------------|--------|
| Quick prototyping | EfficientNet | Fast, reliable, good baseline |
| Best performance | DINOv3 | State-of-the-art vision transformer |
| Balanced approach | DINOv2 | Good performance, reasonable speed |
| Traditional approach | ResNet50 | Well-established, interpretable |
| Document-specific | LayoutXLM | Designed for document understanding |

### **2. Hyperparameter Tuning**

For custom hyperparameter tuning, modify the parameter grid:

```python
# In train_pagewise_xgboost.py, modify param_grid:
param_grid = {
    'max_depth': [3, 6, 9],           # Increase options
    'learning_rate': [0.05, 0.1, 0.2], # Add more rates
    'n_estimators': [100, 200, 300],   # More trees
    'subsample': [0.8, 1.0],          # Add subsampling
    'colsample_bytree': [0.6, 0.8, 1.0], # Feature sampling
}
```

### **3. Memory Management**

For large datasets or limited memory:

```python
# Reduce batch size in scripts
batch_size = 8  # Instead of 16

# Use CPU if GPU memory is insufficient
device = 'cpu'  # In feature extraction

# Process books individually instead of all at once
```

### **4. Quality Assurance**

Always verify your results:

```bash
# 1. Check data integrity
python -c "
import pandas as pd
df = pd.read_excel('data/Quality.xlsx')
print('Missing values:', df.isnull().sum().sum())
print('Duplicate images:', df['Image Name'].duplicated().sum())
"

# 2. Verify model performance
python readability_training_pagewise/scripts/verify_metrics.py [EXPERIMENT_DIR]

# 3. Cross-check predictions
python -c "
import pandas as pd
df = pd.read_excel('[PREDICTION_FILE].xlsx', sheet_name='Page_Predictions')
print('Prediction range:', df['prediction_probability'].min(), 'to', df['prediction_probability'].max())
print('Prediction distribution:', df['predicted_readability'].value_counts())
"
```

### **5. Performance Optimization**

```bash
# Use GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Increase number of workers for data loading
# Modify num_workers in DataLoader (scripts)

# Use mixed precision for memory efficiency
# Add to model training (advanced)
```

## 🚨 **Common Issues and Solutions**

### **Issue 1: CUDA Out of Memory**
```bash
# Solution: Reduce batch size
# Edit batch_size in scripts from 16 to 8 or 4
```

### **Issue 2: Model Download Fails**
```bash
# Solution: Clear cache and retry
rm -rf ~/.cache/torch/hub/
rm -rf ~/.cache/huggingface/
```

### **Issue 3: Feature Extraction Crashes**
```bash
# Solution: Check image paths and formats
python -c "
import pandas as pd
import os
df = pd.read_excel('data/Quality.xlsx')
missing = df[~df['Image Path'].apply(os.path.exists)]
print(f'Missing images: {len(missing)}')
if len(missing) > 0:
    print(missing[['Image Name', 'Image Path']].head())
"
```

### **Issue 4: Poor Model Performance**
```bash
# Solutions:
# 1. Try different backbone models
# 2. Use oversampling for imbalanced data
# 3. Increase hyperparameter search space
# 4. Check data quality and labels
```

This usage guide should help you effectively use the Gujarati readability classification system. For additional support, refer to the main README.md and the troubleshooting section.
