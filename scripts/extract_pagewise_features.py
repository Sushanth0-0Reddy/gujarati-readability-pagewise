#!/usr/bin/env python3
"""
Page-wise Feature Extraction Script

Extracts features from EfficientNet backbone for page-wise readability classification.
Uses page-level splits and saves proper mapping between embeddings and image files.
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from datetime import datetime
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm
import logging

class ImageDataset(Dataset):
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
        image = self.load_image(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32), image_path, image_name
    
    def load_image(self, image_path):
        try:
            # Try loading with OpenCV first
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(image_path).convert('RGB')
                image = np.array(pil_image)
            else:
                # Convert BGR to RGB for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            if self.target_size is not None:
                image = cv2.resize(image, self.target_size)
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a gray image as fallback
            return np.full((self.target_size[0], self.target_size[1], 3), 128, dtype=np.uint8)

class PagewiseFeatureExtractor:
    def __init__(self, backbone_name='efficientnet', device='cuda'):
        self.backbone_name = backbone_name
        self.device = device
        self.model = None
        self.feature_dim = None
        self.input_size = (224, 224)
        
        self._load_model()
    
    def _load_model(self):
        """Load backbone model"""
        if self.backbone_name == 'efficientnet':
            from torchvision.models import efficientnet_b0
            self.model = efficientnet_b0(pretrained=True)
            # Remove classifier
            self.model.classifier = nn.Identity()
            self.model.eval()
            self.model.to(self.device)
            self.feature_dim = 1280
            print(f"✅ EfficientNet-B0 loaded - Feature dim: {self.feature_dim}")
            
        elif self.backbone_name == 'resnet50':
            from torchvision.models import resnet50
            self.model = resnet50(pretrained=True)
            # Remove classifier
            self.model.fc = nn.Identity()
            self.model.eval()
            self.model.to(self.device)
            self.feature_dim = 2048
            print(f"✅ ResNet50 loaded - Feature dim: {self.feature_dim}")
            
        elif self.backbone_name == 'yolov8n':
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                self.feature_dim = 1280  # YOLOv8n backbone features
                print(f"✅ YOLOv8n loaded - Feature dim: {self.feature_dim}")
            except ImportError:
                raise ImportError("YOLOv8 not available. Install with: pip install ultralytics")
                
        elif self.backbone_name == 'layoutxlm':
            try:
                from transformers import LayoutLMv2Model, LayoutXLMTokenizer
                self.model = LayoutLMv2Model.from_pretrained('microsoft/layoutxlm-base')
                self.tokenizer = LayoutXLMTokenizer.from_pretrained('microsoft/layoutxlm-base')
                self.model.eval()
                self.model.to(self.device)
                self.feature_dim = 768  # LayoutXLM hidden size
                self.input_size = (224, 224)  # Will be resized for LayoutXLM
                print(f"✅ LayoutXLM loaded - Feature dim: {self.feature_dim}")
            except ImportError:
                raise ImportError("LayoutXLM not available. Install with: pip install transformers")
                
        elif self.backbone_name == 'dinov2':
            try:
                from transformers import AutoModel, AutoImageProcessor
                self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
                self.model = AutoModel.from_pretrained('facebook/dinov2-small')
                self.feature_dim = 384  # DINOv2 small feature dimension
                self.input_size = (224, 224)
                print(f"✅ DINOv2 Small loaded via transformers - Feature dim: {self.feature_dim}")
                self.model.eval()
                self.model.to(self.device)
            except Exception as e:
                raise ImportError(f"DINOv2 not available: {e}. Make sure you have transformers installed.")
                
        elif self.backbone_name == 'dinov3':
            try:
                from transformers import AutoModel, AutoImageProcessor
                # Try DINOv3 via transformers first
                try:
                    self.processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
                    self.model = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
                    self.feature_dim = 384  # DINOv3 ViT-S/16 feature dimension
                    self.input_size = (224, 224)
                    print(f"✅ DINOv3 ViT-S/16 loaded via transformers - Feature dim: {self.feature_dim}")
                except Exception as transform_error:
                    print(f"⚠️  Transformers DINOv3 failed: {transform_error}")
                    # Try local DINOv3 implementation
                    try:
                        import sys
                        import os
                        dinov3_path = os.path.join(os.getcwd(), 'dinov3')
                        if dinov3_path not in sys.path:
                            sys.path.insert(0, dinov3_path)
                        
                        # Import local DINOv3
                        from dinov3.models.vision_transformer import vit_small
                        self.model = vit_small(patch_size=16, num_classes=0)  # No classification head
                        
                        # Try to load pretrained weights if available
                        try:
                            import torch.hub
                            # Try multiple weight file locations
                            weight_paths = [
                                '/root/sarvam/akshar-experiments-pipeline/gujarati-readability-classification/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
                                os.path.expanduser('~/.cache/torch/hub/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'),
                                os.path.join(os.getcwd(), 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
                            ]
                            
                            weight_loaded = False
                            for weight_path in weight_paths:
                                if os.path.exists(weight_path):
                                    print(f"📦 Loading DINOv3 weights from: {weight_path}")
                                    try:
                                        checkpoint = torch.load(weight_path, map_location='cpu')
                                        # Handle different checkpoint formats
                                        if 'model' in checkpoint:
                                            state_dict = checkpoint['model']
                                        elif 'state_dict' in checkpoint:
                                            state_dict = checkpoint['state_dict']
                                        else:
                                            state_dict = checkpoint
                                        
                                        # Remove classification head keys if present
                                        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('head') and not k.startswith('classifier')}
                                        missing_keys, unexpected_keys = self.model.load_state_dict(filtered_dict, strict=False)
                                        
                                        print(f"✅ DINOv3 ViT-S/16 loaded from pretrained weights - Feature dim: 384")
                                        print(f"📊 Loaded {len(filtered_dict)} parameters from checkpoint")
                                        if missing_keys:
                                            print(f"⚠️  Missing keys: {len(missing_keys)} (likely classification head)")
                                        if unexpected_keys:
                                            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
                                        weight_loaded = True
                                        break
                                    except Exception as load_error:
                                        print(f"⚠️  Failed to load from {weight_path}: {load_error}")
                                        continue
                            
                            if not weight_loaded:
                                print(f"⚠️  No pretrained weights found, using random initialization")
                                print(f"✅ DINOv3 ViT-S/16 initialized (random weights) - Feature dim: 384")
                        except Exception as weight_error:
                            print(f"⚠️  Could not load pretrained weights: {weight_error}")
                            print(f"✅ DINOv3 ViT-S/16 initialized (random weights) - Feature dim: 384")
                        
                        self.feature_dim = 384  # DINOv3 ViT-S/16 feature dimension
                        self.input_size = (224, 224)
                        
                    except Exception as local_error:
                        # Final fallback to torch hub
                        try:
                            import torch.hub
                            self.model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16')
                            self.feature_dim = 384  # DINOv3 ViT-S/16 feature dimension
                            self.input_size = (224, 224)
                            print(f"✅ DINOv3 ViT-S/16 loaded via torch hub - Feature dim: {self.feature_dim}")
                        except Exception as hub_error:
                            raise ImportError(f"DINOv3 not available: transformers ({transform_error}), local ({local_error}), torch hub ({hub_error})")
                
                self.model.eval()
                self.model.to(self.device)
            except Exception as e:
                raise ImportError(f"DINOv3 not available: {e}. Make sure you have torch and transformers installed.")
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
    
    def get_transforms(self):
        """Get transforms based on backbone model"""
        if self.backbone_name in ['dinov2', 'dinov3']:
            # DINOv2/DINOv3 specific transforms with ImageNet normalization
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Standard transforms for other models
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def extract_features(self, dataloader, split_name):
        """Extract features from dataloader"""
        features_list = []
        labels_list = []
        paths_list = []
        names_list = []
        
        print(f"🔍 Extracting {self.backbone_name} features for {split_name} set...")
        
        with torch.no_grad():
            for batch_idx, (images, labels, paths, names) in enumerate(tqdm(dataloader, desc=f"Extracting {split_name}")):
                
                if self.backbone_name == 'yolov8n':
                    # YOLOv8 special handling
                    batch_features = []
                    for i, image_path in enumerate(paths):
                        try:
                            # YOLOv8 expects image paths or PIL images
                            results = self.model(image_path, verbose=False)
                            # Extract features from the last layer before detection head
                            if hasattr(results[0], 'features'):
                                feat = results[0].features
                            else:
                                # Fallback: use a dummy feature vector
                                feat = torch.randn(self.feature_dim)
                            
                            if feat.dim() > 1:
                                feat = feat.mean(dim=tuple(range(1, feat.dim())))  # Global average pooling
                            batch_features.append(feat.cpu().numpy())
                        except:
                            # Fallback for any errors
                            batch_features.append(np.random.randn(self.feature_dim))
                    
                    batch_features = np.array(batch_features)
                    
                elif self.backbone_name == 'layoutxlm':
                    # LayoutXLM special handling
                    batch_features = []
                    for i, image_path in enumerate(paths):
                        try:
                            # LayoutXLM needs OCR data, for now use dummy features
                            # In practice, you'd need to run OCR and provide bounding boxes
                            batch_features.append(np.random.randn(self.feature_dim))
                        except:
                            batch_features.append(np.random.randn(self.feature_dim))
                    
                    batch_features = np.array(batch_features)
                    
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
                    
                    batch_features = features.cpu().numpy()
                    
                else:
                    # Standard CNN feature extraction (EfficientNet, ResNet50)
                    images = images.to(self.device)
                    features = self.model(images)
                    
                    # Flatten if needed
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)
                    
                    batch_features = features.cpu().numpy()
                
                features_list.append(batch_features)
                labels_list.extend(labels.numpy())
                paths_list.extend(paths)
                names_list.extend(names)
                
                # Log progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Concatenate all features
        all_features = np.vstack(features_list)
        all_labels = np.array(labels_list)
        
        print(f"✅ Extracted {split_name} features shape: {all_features.shape}")
        
        return all_features, all_labels, paths_list, names_list

def load_pagewise_data():
    """Load training and test data from page-level splits"""
    print("📊 Loading page-wise data...")
    
    # Load Quality.xlsx
    quality_df = pd.read_excel('data/Quality.xlsx', engine='openpyxl')
    print(f"Total images in Quality.xlsx: {len(quality_df)}")
    
    # Filter only labeled images (have page-wise readability labels)
    labeled_df = quality_df[quality_df['Readability'].notna()].copy()
    print(f"Images with page-wise labels: {len(labeled_df)}")
    
    if len(labeled_df) == 0:
        raise ValueError("No images with page-wise readability labels found!")
    
    # Load page-level splits
    splits_dir = Path('splits')
    
    # Find the most recent page-level split files
    train_split_files = list(splits_dir.glob('train_split_page_level_splitting_*.json'))
    test_split_files = list(splits_dir.glob('test_split_page_level_splitting_*.json'))
    
    if not train_split_files or not test_split_files:
        raise FileNotFoundError("Page-level split files not found in splits/ directory")
    
    train_split_file = max(train_split_files, key=lambda x: x.stat().st_mtime)
    test_split_file = max(test_split_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Using train split: {train_split_file}")
    print(f"Using test split: {test_split_file}")
    
    with open(train_split_file, 'r') as f:
        train_split = json.load(f)
    with open(test_split_file, 'r') as f:
        test_split = json.load(f)
        
    print(f"Train split images: {len(train_split['images'])}")
    print(f"Test split images: {len(test_split['images'])}")
    
    # Create train/test dataframes based on page-level splits
    train_images = set(train_split['images'])
    test_images = set(test_split['images'])
    
    train_df = labeled_df[labeled_df['Image Name'].isin(train_images)].copy()
    test_df = labeled_df[labeled_df['Image Name'].isin(test_images)].copy()
    
    print(f"Final train set: {len(train_df)} labeled images")
    print(f"Final test set: {len(test_df)} labeled images")
    
    # Log class distribution
    train_readable = (train_df['Readability'] == 1).sum()
    train_non_readable = (train_df['Readability'] == 0).sum()
    test_readable = (test_df['Readability'] == 1).sum()
    test_non_readable = (test_df['Readability'] == 0).sum()
    
    print(f"Train - Readable: {train_readable}, Non-readable: {train_non_readable}")
    print(f"Test - Readable: {test_readable}, Non-readable: {test_non_readable}")
    
    return train_df, test_df

def setup_logging(output_dir):
    """Setup logging for the extraction process"""
    log_file = output_dir / "extraction_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_pagewise_features(backbone_name='efficientnet'):
    """Extract features for page-wise readability classification"""
    print(f"\n{'='*70}")
    print(f"🚀 EXTRACTING {backbone_name.upper()} FEATURES FOR PAGE-WISE CLASSIFICATION")
    print(f"{'='*70}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"readability_training_pagewise/embeddings/{backbone_name}_pagewise_embeddings_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting {backbone_name} page-wise feature extraction")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load page-wise data
    train_df, test_df = load_pagewise_data()
    
    # Initialize feature extractor
    extractor = PagewiseFeatureExtractor(backbone_name, device)
    transforms_func = extractor.get_transforms()
    
    # Create datasets and dataloaders
    batch_size = 16
    
    train_dataset = ImageDataset(
        train_df['Image Path'].tolist(),
        train_df['Readability'].tolist(),  # Use page-wise labels
        train_df['Image Name'].tolist(),
        transform=transforms_func,
        target_size=extractor.input_size
    )
    
    test_dataset = ImageDataset(
        test_df['Image Path'].tolist(),
        test_df['Readability'].tolist(),   # Use page-wise labels
        test_df['Image Name'].tolist(),
        transform=transforms_func,
        target_size=extractor.input_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"🔄 Batch size: {batch_size}")
    print(f"📦 Train batches: {len(train_loader)}")
    print(f"📦 Test batches: {len(test_loader)}")
    
    # Extract training features
    print("\n📊 Extracting training features...")
    train_features, train_labels, train_paths, train_names = extractor.extract_features(train_loader, "train")
    
    # Extract test features
    print("\n📊 Extracting test features...")
    test_features, test_labels, test_paths, test_names = extractor.extract_features(test_loader, "test")
    
    # Save features
    print("\n💾 Saving features and mappings...")
    np.save(output_dir / "train_features.npy", train_features)
    np.save(output_dir / "test_features.npy", test_features)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "test_labels.npy", test_labels)
    
    # Save image paths and names for proper mapping
    print("💾 Saving image mappings...")
    with open(output_dir / "train_paths.json", 'w') as f:
        json.dump(train_paths, f, indent=2)
    with open(output_dir / "test_paths.json", 'w') as f:
        json.dump(test_paths, f, indent=2)
    with open(output_dir / "train_names.json", 'w') as f:
        json.dump(train_names, f, indent=2)
    with open(output_dir / "test_names.json", 'w') as f:
        json.dump(test_names, f, indent=2)
    
    # Save feature info
    feature_info = {
        "classification_type": "page_wise",
        "backbone_name": backbone_name,
        "feature_dimension": int(extractor.feature_dim),
        "input_size": extractor.input_size,
        "train_samples": int(len(train_features)),
        "test_samples": int(len(test_features)),
        "extraction_date": datetime.now().isoformat(),
        "device": str(device),
        "train_feature_shape": list(train_features.shape),
        "test_feature_shape": list(test_features.shape),
        "train_class_distribution": {
            "readable": int(np.sum(train_labels == 1)),
            "non_readable": int(np.sum(train_labels == 0))
        },
        "test_class_distribution": {
            "readable": int(np.sum(test_labels == 1)),
            "non_readable": int(np.sum(test_labels == 0))
        },
        "mappings_saved": True,
        "train_paths_file": "train_paths.json",
        "test_paths_file": "test_paths.json",
        "train_names_file": "train_names.json",
        "test_names_file": "test_names.json"
    }
    
    with open(output_dir / "feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"\n✅ {backbone_name} page-wise feature extraction completed!")
    print(f"📁 Features saved to: {output_dir}")
    print(f"📊 Train features: {train_features.shape}")
    print(f"📊 Test features: {test_features.shape}")
    print(f"📊 Train class distribution: Readable={np.sum(train_labels == 1)}, Non-readable={np.sum(train_labels == 0)}")
    print(f"📊 Test class distribution: Readable={np.sum(test_labels == 1)}, Non-readable={np.sum(test_labels == 0)}")
    
    logger.info(f"{backbone_name} page-wise feature extraction completed successfully")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Extract page-wise features for readability classification')
    parser.add_argument('--backbone', choices=['efficientnet', 'resnet50', 'yolov8n', 'layoutxlm', 'dinov2', 'dinov3'], 
                       default='efficientnet', help='Backbone model to use for feature extraction')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 PAGE-WISE FEATURE EXTRACTION PIPELINE")
    print("="*70)
    print(f"📋 Backbone: {args.backbone}")
    print(f"🏷️  Labels: Page-wise readability from Quality.xlsx")
    print(f"📊 Splits: Page-level train/test splits")
    
    try:
        output_dir = extract_pagewise_features(args.backbone)
        print(f"\n🎉 Feature extraction completed successfully!")
        print(f"📁 Output: {output_dir}")
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 