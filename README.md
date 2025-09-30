# Gujarati Readability Classification - Page-wise Training System

A comprehensive machine learning pipeline for **page-wise readability classification** of Gujarati text documents using deep learning feature extraction and XGBoost classification.

## 🎯 **Overview**

This system determines whether individual pages of Gujarati documents are "readable" or "non-readable" by:
1. Extracting visual features using state-of-the-art computer vision models
2. Training XGBoost classifiers with various sampling strategies
3. Providing comprehensive evaluation and prediction capabilities

## 📊 **System Architecture**

```
Data Input (Quality.xlsx) → Feature Extraction → Model Training → Evaluation → Prediction
                                    ↓
                            Multiple Backbone Models
                         (EfficientNet, DINOv2, DINOv3, etc.)
                                    ↓
                              XGBoost Training
                         (Standard, Oversampled, Undersampled)
                                    ↓
                            Comprehensive Evaluation
                         (Metrics, Plots, Excel Reports)
```

## 🗂️ **Directory Structure**

```
readability_training_pagewise/
├── scripts/                          # Main execution scripts
│   ├── extract_pagewise_features.py  # Feature extraction pipeline
│   ├── train_pagewise_xgboost.py     # Standard XGBoost training
│   ├── train_pagewise_xgboost_oversampled.py   # With oversampling
│   ├── train_pagewise_xgboost_undersampled.py  # With undersampling
│   ├── predict_single_book.py        # Single book prediction
│   ├── plot_prediction_distribution_general.py # Visualization
│   └── verify_metrics.py             # Metrics validation
├── embeddings/                       # Extracted features storage
│   ├── dinov2_pagewise_embeddings_*/
│   ├── dinov3_pagewise_embeddings_*/
│   ├── efficientnet_pagewise_embeddings_*/
│   └── ...
├── experiments/                      # Training results
│   ├── xgboost_dinov2_*/
│   ├── xgboost_efficientnet_*/
│   └── ...
└── README.md                        # This documentation
```

## 🚀 **Quick Start**

### **Prerequisites**

```bash
# Required packages
pip install torch torchvision transformers
pip install xgboost scikit-learn pandas numpy
pip install matplotlib seaborn openpyxl
pip install imbalanced-learn  # For oversampling
pip install ultralytics       # For YOLOv8
```

### **Basic Workflow**

1. **Extract Features**:
```bash
cd /root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification
python readability_training_pagewise/scripts/extract_pagewise_features.py --backbone dinov2
```

2. **Train Model**:
```bash
python readability_training_pagewise/scripts/train_pagewise_xgboost.py --backbone dinov2
```

3. **Predict Single Book**:
```bash
python readability_training_pagewise/scripts/predict_single_book.py --book_name "Your Book Name" --backbone dinov2
```

## 🔧 **Detailed Usage**

### **1. Feature Extraction**

Extract visual features from page images using various backbone models:

```bash
# Available backbones: efficientnet, resnet50, dinov2, dinov3, yolov8n, layoutxlm
python scripts/extract_pagewise_features.py --backbone [BACKBONE_NAME]
```

**Supported Models:**
- **EfficientNet-B0**: 1,280 features, fast and efficient
- **ResNet50**: 2,048 features, robust CNN features
- **DINOv2 Small**: 384 features, self-supervised vision transformer
- **DINOv3 ViT-S/16**: 384 features, latest DINO model
- **YOLOv8n**: 1,280 features, object detection features
- **LayoutXLM**: 768 features, document layout understanding

**Output**: Creates `embeddings/[backbone]_pagewise_embeddings_[timestamp]/` with:
- `train_features.npy`, `test_features.npy`: Feature arrays
- `train_labels.npy`, `test_labels.npy`: Label arrays
- `train_paths.json`, `test_paths.json`: Image path mappings
- `feature_info.json`: Extraction metadata

### **2. Model Training**

Train XGBoost classifiers with different strategies:

#### **Standard Training**
```bash
python scripts/train_pagewise_xgboost.py --backbone dinov2
```

#### **With Oversampling** (Recommended for imbalanced data)
```bash
python scripts/train_pagewise_xgboost_oversampled.py --backbone dinov2 --sampling_method smote
```

Available sampling methods:
- `smote`: Synthetic Minority Oversampling Technique
- `adasyn`: Adaptive Synthetic Sampling
- `random`: Random oversampling
- `smoteenn`: SMOTE + Edited Nearest Neighbours
- `smotetomek`: SMOTE + Tomek links

#### **With Undersampling**
```bash
python scripts/train_pagewise_xgboost_undersampled.py --backbone dinov2
```

**Training Output**: Creates `experiments/xgboost_[backbone]_[timestamp]/` with:
- `best_model.pkl`: Trained XGBoost model
- `results.json`: Performance metrics
- `performance_metrics.txt`: Human-readable results
- `plots/`: Visualization plots
- `training.log`: Detailed training log

### **3. Single Book Prediction**

Generate predictions for all pages in a specific book:

```bash
python scripts/predict_single_book.py \
    --book_name "Sample Book Name" \
    --backbone dinov2 \
    --quality_file data/Quality.xlsx \
    --output_dir results/
```

**Output**: Excel file with:
- **Page_Predictions**: Individual page results with probabilities
- **Summary**: Book-level statistics and accuracy metrics

### **4. Visualization and Analysis**

#### **Plot Prediction Distributions**
```bash
python scripts/plot_prediction_distribution_general.py \
    --experiment experiments/xgboost_dinov2_20250819_112742
```

**Note**: The script automatically detects the backbone model from the experiment directory name and loads the corresponding embeddings. It supports all backbone models (EfficientNet, DINOv2, DINOv3, ResNet50, YOLOv8n, LayoutXLM).

#### **Verify Metrics**
```bash
python scripts/verify_metrics.py experiments/xgboost_dinov2_20250819_112742
```

## 📊 **Data Requirements**

### **Input Data Structure**

The system expects:
1. **Quality.xlsx**: Main data file with columns:
   - `Book Name`: Name of the book
   - `Image Name`: Unique image identifier
   - `Image Path`: Full path to image file
   - `Readability`: Binary label (0=non-readable, 1=readable)

2. **Page-level Splits**: JSON files in `splits/` directory:
   - `train_split_page_level_splitting_*.json`
   - `test_split_page_level_splitting_*.json`

### **Current Dataset Statistics**
- **Training Set**: 501 pages (336 readable, 165 non-readable)
- **Test Set**: 128 pages (83 readable, 45 non-readable)
- **Class Distribution**: ~67% readable, ~33% non-readable
- **Image Format**: 224×224 RGB images

## 🎯 **Performance Metrics**

The system focuses on **non-readable detection** (treating non-readable as positive class):

### **Primary Metrics**
- **Precision (Non-readable)**: Accuracy of non-readable predictions
- **Recall (Non-readable)**: Coverage of actual non-readable pages
- **F1 Score (Non-readable)**: Harmonic mean of precision and recall
- **ROC AUC**: Overall discriminative ability

### **Evaluation Strategy**
- **3-fold Cross-validation**: For hyperparameter selection
- **Holdout Test Set**: For final performance evaluation
- **Class-aware Metrics**: Focus on minority class performance

## 🔬 **Experimental Features**

### **Hyperparameter Tuning**
The system uses grid search with optimized parameter space:
```python
param_grid = {
    'max_depth': [3, 6],
    'learning_rate': [0.1],
    'n_estimators': [100, 200],
    'subsample': [1.0],
    'colsample_bytree': [0.8],
    'reg_alpha': [0],
    'reg_lambda': [1]
}
```

### **Class Imbalance Handling**
Multiple strategies available:
1. **Cost-sensitive Learning**: Weighted loss functions
2. **Oversampling**: SMOTE, ADASYN, Random
3. **Undersampling**: Random undersampling
4. **Combined Methods**: SMOTEENN, SMOTETomek

## 📈 **Output Interpretation**

### **Training Results**
Each experiment produces:
- **Performance Metrics**: Detailed accuracy, precision, recall, F1
- **Feature Importance**: Top contributing features
- **Confusion Matrix**: Classification breakdown
- **ROC Curve**: Threshold analysis
- **Hyperparameter Analysis**: Parameter impact visualization

### **Prediction Results**
Single book predictions include:
- **Page-level Scores**: Individual readability probabilities
- **Book-level Aggregation**: Average readability score
- **Confidence Intervals**: Statistical reliability measures
- **Accuracy Assessment**: For labeled data

## 🛠️ **Advanced Usage**

### **Custom Backbone Integration**
To add a new backbone model:

1. Modify `PagewiseFeatureExtractor` in `extract_pagewise_features.py`
2. Add model loading logic in `_load_model()` method
3. Implement feature extraction in `extract_features()` method
4. Update supported models list in argument parser

### **Custom Sampling Strategies**
To implement new sampling methods:

1. Add to `train_pagewise_xgboost_oversampled.py`
2. Import required libraries
3. Implement in `apply_sampling()` method
4. Add to argument choices

### **Batch Processing**
For processing multiple books:

```bash
# Create a batch script
for book in "Book1" "Book2" "Book3"; do
    python scripts/predict_single_book.py --book_name "$book" --backbone dinov2
done
```

## 📊 **Prediction Analysis Generation**

### **Why Some Experiments Don't Have Prediction Analysis**

Training scripts automatically generate basic plots in the `plots/` folder, but detailed prediction analysis in the `prediction_analysis/` folder must be generated manually using the `plot_prediction_distribution_general.py` script.

### **Generating Missing Prediction Analysis**

If an experiment directory is missing the `prediction_analysis/` folder:

```bash
# For any experiment (script auto-detects backbone model)
python scripts/plot_prediction_distribution_general.py \
    --experiment experiments/xgboost_dinov2_nonreadable_20250819_060534

# For EfficientNet experiments
python scripts/plot_prediction_distribution_general.py \
    --experiment experiments/xgboost_efficientnet_20250807_144338

# For DINOv3 experiments
python scripts/plot_prediction_distribution_general.py \
    --experiment experiments/xgboost_dinov3_nonreadable_20250819_060617
```

### **Generated Analysis Files**

The script creates three comprehensive visualization files:

1. **`prediction_distribution_scatter.png`**: Scatter plots showing predicted probabilities vs true labels
2. **`prediction_distribution_histograms.png`**: Histograms and confusion matrices for both train/test sets
3. **`roc_curve_and_summary.png`**: ROC curve and performance metrics summary

### **Backbone Model Auto-Detection**

The script automatically detects the backbone model from the experiment directory name:
- `xgboost_efficientnet_*` → Uses EfficientNet embeddings
- `xgboost_dinov2_*` → Uses DINOv2 embeddings  
- `xgboost_dinov3_*` → Uses DINOv3 embeddings
- `xgboost_resnet50_*` → Uses ResNet50 embeddings
- `xgboost_yolov8n_*` → Uses YOLOv8n embeddings
- `xgboost_layoutxlm_*` → Uses LayoutXLM embeddings

### **Batch Generation for All Experiments**

```bash
#!/bin/bash
# Generate prediction analysis for all experiments
for experiment_dir in experiments/xgboost_*; do
    if [ -d "$experiment_dir" ] && [ -f "$experiment_dir/best_model.pkl" ]; then
        echo "Generating prediction analysis for: $experiment_dir"
        python scripts/plot_prediction_distribution_general.py --experiment "$experiment_dir"
    fi
done
```

## 🐛 **Troubleshooting**

### **Common Issues**

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   # Edit batch_size in scripts (default: 16 → 8 or 4)
   ```

2. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt  # If available
   # Or install individually as needed
   ```

3. **Model Download Issues**:
   ```bash
   # For transformers models, ensure internet connection
   # For torch hub models, clear cache if needed
   rm -rf ~/.cache/torch/hub/
   ```

4. **Data Path Issues**:
   ```bash
   # Ensure Quality.xlsx exists and paths are correct
   # Check image file accessibility
   ```

### **Performance Optimization**

1. **GPU Usage**: Ensure CUDA is available and properly configured
2. **Memory Management**: Monitor RAM usage during feature extraction
3. **Parallel Processing**: Adjust `num_workers` in DataLoader
4. **Batch Size**: Balance between speed and memory usage

## 🤖 **VLM as Judge - Quality Analysis**

The system includes a comprehensive **Vision Language Model (VLM) as Judge** component for automated document quality assessment using Google's Gemini 2.5 Flash model.

### **Overview**

The VLM quality analysis system (`quality_analysis_gemini/`) provides:
- **Automated Quality Assessment**: AI-powered evaluation of document scan quality
- **Multi-Crop Analysis**: Generates 5 random 40% crops per image for comprehensive analysis
- **Structured Output**: Detailed JSON reports with quality metrics
- **Parallel Processing**: Handles multiple images concurrently for efficiency

### **Quality Metrics Evaluated**

1. **Clarity**: Text sharpness and readability
2. **Contrast**: Text-background distinction  
3. **Noise & Artifacts**: Dust, smudges, digital artifacts
4. **Geometric Distortion**: Skew, rotation, warping
5. **Illumination & Shadows**: Lighting uniformity
6. **Completeness**: Content cropping and visibility
7. **Text Line Thickness**: Font consistency

Each metric is rated on a **1-5 scale** with detailed explanations.

### **Usage**

#### **Setup**
```bash
# Ensure Google Cloud credentials are configured
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# Install dependencies
pip install google-genai pillow tqdm tenacity python-dotenv
```

#### **Run Quality Analysis**
```bash
cd quality_analysis_gemini/

# Basic usage
python document_quality_analyzer.py

# Or use the runner script
python run_quality_analysis.py
```

#### **Input Requirements**
- Place document images in: `DQA_data/sampled_images/`
- Supported formats: PNG, JPG, JPEG, TIF, TIFF, WEBP, BMP

### **Output Structure**

For each input image, generates a comprehensive JSON report:

```json
{
  "image_metadata": {
    "original_image": "document.jpg",
    "original_size": [2480, 3508],
    "crop_percentage": 40.0,
    "total_crops": 5,
    "analysis_timestamp": "2024-07-30 17:30:45"
  },
  "crops": [
    {
      "crop_index": 1,
      "quality_analysis": {
        "overallRating": "4/5 (Good)",
        "detailedAnalysis": {
          "clarity": "Text is sharp and clear...",
          "contrast": "Good contrast between text and background...",
          "noiseAndArtifacts": "Minor dust spots present...",
          "geometricDistortion": "Slight rotation detected...",
          "illuminationAndShadows": "Even lighting...",
          "completeness": "All content visible...",
          "textLineThickness": "Consistent line thickness..."
        },
        "summary": {
          "metricRatings": {
            "Clarity": "4/5",
            "Contrast": "4/5",
            "Noise & Artifacts": "3.5/5",
            "Geometric Distortion": "3/5",
            "Illumination & Shadows": "4/5",
            "Completeness": "5/5",
            "Text Line Thickness": "4/5"
          }
        }
      }
    }
    // ... 4 more crops
  ]
}
```

### **Visualization**

The system generates quality analysis plots:
- `quality_summary_by_book.png`: Overall quality distribution by book
- `quality_analysis_by_book_candlestick.png`: Detailed quality metrics visualization

### **Integration with Readability Classification**

The VLM quality scores can be used to:
1. **Filter Low-Quality Images**: Remove poor scans before readability classification
2. **Quality-Aware Training**: Weight training samples based on scan quality
3. **Performance Analysis**: Correlate readability performance with image quality
4. **Data Curation**: Identify images needing re-scanning or preprocessing

### **Configuration**

Key settings in `document_quality_analyzer.py`:
```python
MODEL_ID = "gemini-2.5-flash"           # AI model to use
MAX_CONCURRENT_REQUESTS = 32            # Parallel API calls
REQUEST_DELAY = 1                       # Seconds between requests
NUM_WORKERS = 32                        # Worker processes
```

### **Performance**
- **Processing Time**: ~10-15 seconds per crop
- **Expected Output**: 1 JSON file per input image (5 crop analyses)
- **Resource Usage**: Moderate CPU, network-dependent

## 📚 **References**

### **Models Used**
- **EfficientNet**: Tan & Le, 2019
- **ResNet**: He et al., 2016
- **DINOv2**: Oquab et al., 2023
- **DINOv3**: Latest self-supervised vision transformer
- **YOLOv8**: Ultralytics, 2023
- **LayoutXLM**: Xu et al., 2021

### **Techniques**
- **XGBoost**: Chen & Guestrin, 2016
- **SMOTE**: Chawla et al., 2002
- **Cross-validation**: Standard ML practice

## 📞 **Support**

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in experiment directories
3. Verify data format and file paths
4. Ensure all dependencies are installed

## 🔄 **Version History**

- **v1.0**: Initial implementation with EfficientNet
- **v1.1**: Added multiple backbone support
- **v1.2**: Implemented sampling strategies
- **v1.3**: Added single book prediction
- **v1.4**: Enhanced visualization and analysis tools

---

*Last updated: September 2025*
