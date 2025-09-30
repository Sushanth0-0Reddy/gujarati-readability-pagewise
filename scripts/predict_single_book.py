#!/usr/bin/env python3
"""
Generate Embeddings and Predictions for a Single Book

This script generates embeddings for a specific book and creates predictions
using trained models, then saves results to an Excel file.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SingleBookDataset(Dataset):
    """Dataset for loading images from a single book"""
    
    def __init__(self, image_paths, labels, image_names, transform=None, target_size=(224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.image_names = image_names
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image_name = self.image_names[idx]
        
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Convert to PIL for transforms
            image = Image.fromarray(image)
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                image = transform(image)
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a zero tensor as fallback
            if self.transform:
                dummy_image = Image.fromarray(np.zeros((*self.target_size, 3), dtype=np.uint8))
                image = self.transform(dummy_image)
            else:
                image = torch.zeros(3, *self.target_size)
        
        return image, label, image_path, image_name

class SingleBookFeatureExtractor:
    """Extract features for a single book using different backbones"""
    
    def __init__(self, backbone_name='efficientnet', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.backbone_name = backbone_name
        self.device = device
        self.model = None
        self.feature_dim = None
        
    def _load_model(self):
        """Load the specified backbone model"""
        if self.backbone_name == 'efficientnet':
            # Use torchvision EfficientNet to avoid download issues
            import torchvision.models as models
            self.model = models.efficientnet_b0(pretrained=True)
            self.model.classifier = nn.Identity()  # Remove classifier
            self.feature_dim = 1280
            
        elif self.backbone_name == 'resnet50':
            import torchvision.models as models
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Identity()  # Remove classifier
            self.feature_dim = 2048
            
        elif self.backbone_name == 'yolov8n':
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            self.feature_dim = 1000  # Approximate feature dimension
            
        elif self.backbone_name == 'layoutxlm':
            from transformers import LayoutLMv2Model, LayoutXLMTokenizer
            self.model = LayoutLMv2Model.from_pretrained('microsoft/layoutlmv2-base-uncased')
            self.tokenizer = LayoutXLMTokenizer.from_pretrained('microsoft/layoutlmv2-base-uncased')
            self.feature_dim = 768
            
        elif self.backbone_name == 'dinov2':
            from transformers import AutoModel, AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
            self.model = AutoModel.from_pretrained('facebook/dinov2-small')
            self.feature_dim = 384  # DINOv2 ViT-S feature dimension
            
        elif self.backbone_name == 'dinov3':
            from transformers import AutoModel, AutoImageProcessor
            try:
                self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
                self.model = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
                self.feature_dim = 384  # DINOv3 ViT-S/16 feature dimension
            except:
                # Fallback to torch hub
                import torch.hub
                self.model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16')
                self.feature_dim = 384  # DINOv3 ViT-S/16 feature dimension
        
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
    
    def extract_features(self, dataloader, book_name):
        """Extract features for all images in the book"""
        if self.model is None:
            self._load_model()
        
        all_features = []
        all_labels = []
        all_paths = []
        all_names = []
        
        print(f"🔧 Extracting {self.backbone_name} features for book: {book_name}")
        
        with torch.no_grad():
            for batch_idx, (images, labels, paths, names) in enumerate(dataloader):
                print(f"   Processing batch {batch_idx + 1}/{len(dataloader)}")
                
                if self.backbone_name == 'yolov8n':
                    # YOLOv8 requires special handling
                    batch_features = []
                    for i in range(len(images)):
                        img_tensor = images[i].numpy().transpose(1, 2, 0)  # CHW to HWC
                        img_tensor = (img_tensor * 255).astype(np.uint8)  # Denormalize
                        
                        # Run YOLO prediction
                        results = self.yolo_model(img_tensor, verbose=False)
                        
                        # Extract features from the last layer
                        if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                            # Use detection features if available
                            features = results[0].boxes.data.flatten()
                            # Pad or truncate to fixed size
                            if len(features) > self.feature_dim:
                                features = features[:self.feature_dim]
                            else:
                                features = np.pad(features, (0, self.feature_dim - len(features)))
                        else:
                            # Fallback to zero features
                            features = np.zeros(self.feature_dim)
                        
                        batch_features.append(features)
                    
                    features = np.array(batch_features)
                    
                elif self.backbone_name == 'layoutxlm':
                    # LayoutXLM requires special handling for document images
                    batch_features = []
                    for i in range(len(images)):
                        # Convert image tensor back to PIL
                        img_tensor = images[i]
                        # Denormalize
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_tensor * std + mean
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        
                        # Convert to PIL
                        img_pil = transforms.ToPILImage()(img_tensor)
                        
                        # Use dummy text tokens for image-only processing
                        encoding = self.tokenizer(
                            text="dummy text", 
                            return_tensors="pt",
                            max_length=512,
                            truncation=True,
                            padding='max_length'
                        )
                        
                        # Create dummy bounding boxes
                        bbox = [[0, 0, 224, 224] for _ in range(len(encoding['input_ids'][0]))]
                        encoding['bbox'] = torch.tensor([bbox])
                        
                        # Move to device
                        encoding = {k: v.to(self.device) for k, v in encoding.items()}
                        
                        # Get model output
                        outputs = self.model(**encoding)
                        
                        # Use pooled output or mean of last hidden states
                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            features = outputs.pooler_output.squeeze().cpu().numpy()
                        else:
                            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                        
                        batch_features.append(features)
                    
                    features = np.array(batch_features)
                    
                elif self.backbone_name in ['dinov2', 'dinov3']:
                    # DINOv2/DINOv3 feature extraction
                    images = images.to(self.device)
                    with torch.no_grad():
                        # Check if using transformers model
                        if hasattr(self, 'processor'):
                            # Transformers DINOv2/DINOv3 model
                            outputs = self.model(images)
                            features = outputs.last_hidden_state[:, 0, :]  # CLS token
                        else:
                            # Torch hub DINOv3 model
                            outputs = self.model(images)
                            if isinstance(outputs, dict):
                                # Use class token features if available
                                features = outputs.get('x_norm_clstoken', outputs.get('cls_token', outputs))
                            else:
                                # If outputs is a tensor, use it directly
                                features = outputs
                        
                        # Ensure we have 2D features [batch_size, feature_dim]
                        if features.dim() > 2:
                            features = features.view(features.size(0), -1)
                    
                    features = features.cpu().numpy()
                    
                else:
                    # Standard CNN models (EfficientNet, ResNet)
                    images = images.to(self.device)
                    features = self.model(images)
                    features = features.cpu().numpy()
                
                # Store results
                all_features.extend(features)
                all_labels.extend(labels.numpy())
                all_paths.extend(paths)
                all_names.extend(names)
        
        return np.array(all_features), np.array(all_labels), all_paths, all_names

def load_book_data(book_name, quality_file='data/Quality.xlsx'):
    """Load data for a specific book"""
    
    print(f"📚 Loading data for book: {book_name}")
    
    # Load Quality.xlsx
    df = pd.read_excel(quality_file)
    
    # Filter for the specific book
    book_df = df[df['Book Name'] == book_name].copy()
    
    if len(book_df) == 0:
        raise ValueError(f"No data found for book: {book_name}")
    
    print(f"   📄 Found {len(book_df)} pages")
    
    # Check if all pages are labeled
    labeled_pages = book_df['Readability'].notna().sum()
    print(f"   ✅ Labeled pages: {labeled_pages}/{len(book_df)}")
    
    if labeled_pages < len(book_df):
        print(f"   ⚠️  Warning: {len(book_df) - labeled_pages} pages are not labeled")
    
    # Prepare data
    image_paths = []
    labels = []
    image_names = []
    
    for _, row in book_df.iterrows():
        image_path = row['Image Path']
        image_name = row['Image Name']
        label = row['Readability']
        
        # Check if image file exists
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(int(label) if not pd.isna(label) else -1)  # Use -1 for unlabeled
            image_names.append(image_name)
        else:
            print(f"   ⚠️  Image not found: {image_path}")
    
    print(f"   📸 Found {len(image_paths)} valid images")
    
    return image_paths, labels, image_names, book_df

def load_trained_model(backbone_name, experiments_dir='readability_training_pagewise/experiments'):
    """Load the best trained model for the specified backbone"""
    
    experiments_path = Path(experiments_dir)
    
    # Find experiment directory for this backbone
    pattern = f"xgboost_{backbone_name}_*"
    matching_dirs = list(experiments_path.glob(pattern))
    
    if not matching_dirs:
        raise FileNotFoundError(f"No trained model found for backbone: {backbone_name}")
    
    # Use the most recent experiment
    experiment_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
    
    print(f"🤖 Loading trained model from: {experiment_dir}")
    
    # Load model
    model_path = experiment_dir / 'best_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model, experiment_dir

def create_predictions_excel(book_df, features, labels, image_names, model, book_name, output_dir):
    """Create Excel file with predictions for the book"""
    
    print(f"📊 Generating predictions for {len(features)} images...")
    
    # Get predictions and probabilities
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]  # Probability of readable
    
    # Create results dataframe
    results_df = book_df.copy()
    
    # Create mappings
    pred_mapping = {}
    proba_mapping = {}
    
    for i, img_name in enumerate(image_names):
        pred_mapping[img_name] = int(predictions[i])
        proba_mapping[img_name] = float(probabilities[i])
    
    # Add prediction columns
    results_df['predicted_readability'] = results_df['Image Name'].map(pred_mapping)
    results_df['prediction_probability'] = results_df['Image Name'].map(proba_mapping)
    
    # Add prediction labels
    results_df['predicted_label'] = results_df['predicted_readability'].map({
        0: 'Not Readable',
        1: 'Readable'
    })
    
    results_df['true_label'] = results_df['Readability'].map({
        0: 'Not Readable', 
        1: 'Readable'
    })
    
    # Add correctness column
    results_df['prediction_correct'] = (
        results_df['predicted_readability'] == results_df['Readability']
    )
    
    # Calculate statistics
    total_pages = len(results_df)
    labeled_pages = results_df['Readability'].notna().sum()
    
    if labeled_pages > 0:
        accuracy = results_df['prediction_correct'].sum() / labeled_pages
        true_readable = (results_df['Readability'] == 1).sum()
        pred_readable = (results_df['predicted_readability'] == 1).sum()
    else:
        accuracy = None
        true_readable = 0
        pred_readable = (results_df['predicted_readability'] == 1).sum()
    
    # Calculate book-level averages
    book_avg_probability = results_df['prediction_probability'].mean()
    book_prediction = 1 if book_avg_probability > 0.5 else 0
    
    # Create summary statistics
    summary_data = [
        ['Book Name', book_name],
        ['Total Pages', total_pages],
        ['Labeled Pages', labeled_pages],
        ['True Readable Pages', true_readable],
        ['Predicted Readable Pages', pred_readable],
        ['Book Average Probability', f'{book_avg_probability:.4f}'],
        ['Book Prediction', 'Readable' if book_prediction == 1 else 'Not Readable'],
    ]
    
    if accuracy is not None:
        summary_data.append(['Page Accuracy', f'{accuracy:.4f} ({accuracy:.2%})'])
    
    summary_data.extend([
        ['Mean Probability', f'{results_df["prediction_probability"].mean():.4f}'],
        ['Std Probability', f'{results_df["prediction_probability"].std():.4f}'],
        ['Min Probability', f'{results_df["prediction_probability"].min():.4f}'],
        ['Max Probability', f'{results_df["prediction_probability"].max():.4f}'],
    ])
    
    summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    
    # Save to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f'{book_name.replace(" ", "_")}_predictions_{timestamp}.xlsx'
    excel_path = Path(output_dir) / excel_filename
    
    print(f"💾 Saving results to: {excel_path}")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Main results
        results_df.to_excel(writer, sheet_name='Page_Predictions', index=False)
        
        # Summary statistics
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✅ Excel file created: {excel_path}")
    
    # Print summary
    print(f"\n📊 Book Summary:")
    print(f"   📚 Book: {book_name}")
    print(f"   📄 Total pages: {total_pages}")
    print(f"   ✅ Labeled pages: {labeled_pages}")
    if accuracy is not None:
        print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"   📈 Book avg probability: {book_avg_probability:.4f}")
    print(f"   🏷️  Book prediction: {'Readable' if book_prediction == 1 else 'Not Readable'}")
    
    return excel_path, results_df, summary_df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate embeddings and predictions for a single book")
    parser.add_argument('--book_name', type=str, required=True,
                        help='Name of the book to process')
    parser.add_argument('--backbone', type=str, default='efficientnet',
                        choices=['efficientnet', 'resnet50', 'yolov8n', 'layoutxlm', 'dinov2', 'dinov3'],
                        help='Backbone model to use for feature extraction')
    parser.add_argument('--quality_file', type=str, default='data/Quality.xlsx',
                        help='Path to Quality.xlsx file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for feature extraction')
    
    args = parser.parse_args()
    
    try:
        # Load book data
        image_paths, labels, image_names, book_df = load_book_data(args.book_name, args.quality_file)
        
        # Load trained model
        model, experiment_dir = load_trained_model(args.backbone)
        
        # Set output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = experiment_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset and dataloader
        dataset = SingleBookDataset(image_paths, labels, image_names)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        # Extract features
        extractor = SingleBookFeatureExtractor(args.backbone)
        features, extracted_labels, extracted_paths, extracted_names = extractor.extract_features(
            dataloader, args.book_name
        )
        
        print(f"🎯 Extracted features shape: {features.shape}")
        
        # Create predictions Excel
        excel_path, results_df, summary_df = create_predictions_excel(
            book_df, features, extracted_labels, extracted_names, 
            model, args.book_name, output_dir
        )
        
        print(f"\n🎉 Processing completed successfully!")
        print(f"📁 Results saved to: {excel_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
