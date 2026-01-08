#!/usr/bin/env python3


import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
import math
from datetime import datetime
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord, Resized, RandRotated, RandGaussianNoised, RandAffined
from monai.data import DataLoader, Dataset


try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("Warning: mamba-ssm not installed. Mamba models will not be available.")
    print("Install with: pip install mamba-ssm")
    MAMBA_AVAILABLE = False





def set_all_seeds(seed=42):
    
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)

def get_patient_id(id_str):
    
    match = re.search(r'(\d{4})', id_str)
    return match.group(1) if match else None

def make_json_serializable(obj):
    
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            if isinstance(key, (np.integer, np.int64)):
                new_key = str(int(key))
            elif isinstance(key, (np.floating, np.float32, np.float64)):
                new_key = str(float(key))
            else:
                new_key = key
            new_dict[new_key] = make_json_serializable(value)
        return new_dict
    else:
        return obj

def safe_save_json(data, filepath):
    
    temp_file = filepath + '.tmp'
    try:
        serializable_data = make_json_serializable(data)
        with open(temp_file, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        os.replace(temp_file, filepath)
        print(f"Results saved to: {filepath}")
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e

def create_consistent_folds(file_list, labels, patient_ids, k_folds=5, seed=42):
    
    file_list = np.array(file_list)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    
    gkf = GroupKFold(n_splits=k_folds)
    fold_indices = list(gkf.split(file_list, labels, groups=patient_ids))
    
    consistent_splits = []
    
    for fold in range(k_folds):
        train_val_idx, holdout_idx = fold_indices[fold]
        

        holdout_patients = patient_ids[holdout_idx]
        

        unique_patients = np.unique(holdout_patients)
        np.random.seed(seed + fold)
        np.random.shuffle(unique_patients)
        

        split_idx = len(unique_patients) // 2
        val_patients = unique_patients[:split_idx]
        test_patients = unique_patients[split_idx:]
        

        val_mask = np.isin(holdout_patients, val_patients)
        test_mask = np.isin(holdout_patients, test_patients)
        
        fold_data = {
            'train_val_idx': train_val_idx,
            'holdout_idx': holdout_idx,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'val_patients': val_patients.tolist(),
            'test_patients': test_patients.tolist()
        }
        
        consistent_splits.append(fold_data)
    
    return consistent_splits

def get_transforms(img_size=(96, 96, 96), is_training=False, model_type='cnn'):
    
    base_transforms = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image", "mask"], spatial_size=img_size),
    ]
    

    if is_training:
        if model_type == 'cnn':

            augmentation_transforms = [
                RandRotated(keys=["image", "mask"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.3),
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
                RandAffined(keys=["image", "mask"], prob=0.2, 
                           translate_range=(5, 5, 5), scale_range=(0.1, 0.1, 0.1))
            ]
        else:
            augmentation_transforms = [
                RandRotated(keys=["image", "mask"], range_x=0.05, range_y=0.05, range_z=0.05, prob=0.2),
                RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.005),
                RandAffined(keys=["image", "mask"], prob=0.1, 
                           translate_range=(3, 3, 3), scale_range=(0.05, 0.05, 0.05))
            ]
        base_transforms.extend(augmentation_transforms)
    
    base_transforms.append(ToTensord(keys=["image", "mask"]))
    
    return Compose(base_transforms)

def load_image_and_mask_data(labels_file, nifti_dir, lobe_masks_dir, task):
    
    print(f"Reading labels file for task: {task}")
    df = pd.read_csv(labels_file)
    df = df[df[task].notna()]
    df = df[df[task] != -1]
    
    file_list = []
    mask_list = []
    labels = []
    patient_ids = []
    
    for patient_dir in os.listdir(nifti_dir):
        patient_path = os.path.join(nifti_dir, patient_dir)
        if os.path.isdir(patient_path):
            for filename in os.listdir(patient_path):
                if filename.endswith('.nii.gz'):
                    file_id = filename.replace('.nii.gz', '')
                    matching_row = df[df['ScanID'] == file_id]
                    

                    mask_path = os.path.join(lobe_masks_dir, patient_dir, filename)
                    if not matching_row.empty and not pd.isna(matching_row[task].values[0]) and os.path.exists(mask_path):
                        label = int(matching_row[task].values[0])
                        
                        file_path = os.path.join(patient_path, filename)
                        file_list.append(file_path)
                        mask_list.append(mask_path)
                        labels.append(label)
                        patient_ids.append(get_patient_id(file_id))
    
    print(f"Total matched files with masks for {task}: {len(file_list)}")
    print(f"Label distribution: {Counter(labels)}")
    return file_list, mask_list, labels, patient_ids

def create_expert_mask(mask_tensor, expert_idx):
    
    if expert_idx == 0:
        return (mask_tensor == 1).float()
    elif expert_idx == 1:
        return (mask_tensor == 2).float()
    elif expert_idx == 2:
        return (mask_tensor == 3).float()
    elif expert_idx == 3:
        return (mask_tensor == 4).float()
    elif expert_idx == 4:
        return (mask_tensor == 5).float()
    elif expert_idx == 5:
        return ((mask_tensor == 1) | (mask_tensor == 2)).float()
    elif expert_idx == 6:
        return ((mask_tensor == 3) | (mask_tensor == 4) | (mask_tensor == 5)).float()
    else:
        raise ValueError(f"Invalid expert_idx: {expert_idx}. Should be 0-6.")

def prepare_data_dicts(files, masks, labels):
    
    return [{"image": file_path, "mask": mask_path, "label": label} 
            for file_path, mask_path, label in zip(files, masks, labels)]

class EarlyStopping:
    
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True, 
                 mode='val_loss', overfitting_threshold=0.15):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.overfitting_threshold = overfitting_threshold
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, val_auc, train_auc, model):
        if self.mode == 'val_loss':
            score = -val_loss
        elif self.mode == 'val_auc':
            score = val_auc
        elif self.mode == 'combined':
            score = val_auc - 0.1 * val_loss
        elif self.mode == 'overfitting_check':
            overfitting_gap = train_auc - val_auc
            if overfitting_gap > self.overfitting_threshold:
                print(f"Overfitting detected: train_auc({train_auc:.3f}) - val_auc({val_auc:.3f}) = {overfitting_gap:.3f}")
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
            score = -val_loss
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return False
        
    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()





class LobeExpertCNN(nn.Module):
    
    def __init__(self, input_size=(96, 96, 96), dropout_rate=0.5):
        super(LobeExpertCNN, self).__init__()
        

        self.conv1 = nn.Conv3d(1, 24, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(24)
        
        self.conv2 = nn.Conv3d(24, 48, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(48)
        
        self.conv3 = nn.Conv3d(48, 96, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(96)
        
        self.pool = nn.MaxPool3d(2)
        self.dropout3d = nn.Dropout3d(dropout_rate)
        


        flattened_size = 96 * 12 * 12 * 12
        

        self.fc1 = nn.Linear(flattened_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        
        self.output = nn.Linear(64, 2)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout3d(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout3d(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout3d(x)
        

        x = x.view(x.size(0), -1)
        

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        

        output = self.output(x)
        
        return output





if MAMBA_AVAILABLE:
    class PatchEmbedding3D_Mamba(nn.Module):
        
        def __init__(self, img_size=(96, 96, 96), patch_size=(8, 8, 8), in_channels=1, embed_dim=256):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.n_patches_per_dim = tuple(img_size[i] // patch_size[i] for i in range(3))
            self.n_patches = self.n_patches_per_dim[0] * self.n_patches_per_dim[1] * self.n_patches_per_dim[2]
            

            self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            
        def forward(self, x):

            B, C, D, H, W = x.shape
            

            x = self.proj(x)
            

            x = x.flatten(2)
            x = x.transpose(1, 2)
            
            return x

    class MambaBlock(nn.Module):
        
        def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
            super().__init__()
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.norm = nn.LayerNorm(d_model)
            
        def forward(self, x):


            residual = x
            x = self.norm(x)
            x = self.mamba(x)
            return x + residual

    class Mamba3D(nn.Module):
        
        def __init__(self, img_size=(96, 96, 96), patch_size=(8, 8, 8), in_channels=1, 
                     d_model=256, n_layers=4, d_state=64, d_conv=4, expand=2, 
                     num_classes=2, dropout=0.2):
            super().__init__()
            

            self.patch_embed = PatchEmbedding3D_Mamba(img_size, patch_size, in_channels, d_model)
            n_patches = self.patch_embed.n_patches
            

            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
            self.dropout = nn.Dropout(dropout)
            

            self.blocks = nn.ModuleList([
                MambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(n_layers)
            ])
            

            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_classes)
            

            self._init_weights()
            
        def _init_weights(self):
            
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        def forward(self, x):
            B = x.shape[0]
            

            x = self.patch_embed(x)
            

            x = x + self.pos_embed
            x = self.dropout(x)
            

            for block in self.blocks:
                x = block(x)
            

            x = self.norm(x)
            

            x = x.mean(dim=1)
            

            logits = self.head(x)
            
            return logits

    class LobeExpertMamba(nn.Module):
        
        def __init__(self, input_size=(96, 96, 96), dropout_rate=0.1):
            super(LobeExpertMamba, self).__init__()
            

            self.mamba = Mamba3D(
                img_size=input_size,
                patch_size=(8, 8, 8),
                in_channels=1,
                d_model=128,
                n_layers=3,
                d_state=32,
                d_conv=4,
                expand=2,
                num_classes=2,
                dropout=dropout_rate
            )
            
        def forward(self, x):
            return self.mamba(x)

else:

    class LobeExpertMamba(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("Mamba models require mamba-ssm. Install with: pip install mamba-ssm")
        def forward(self, x):
            pass





class PatchEmbedding3D_ViT(nn.Module):
    
    def __init__(self, img_size=(96, 96, 96), patch_size=(8, 8, 8), in_channels=1, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches_per_dim = tuple(img_size[i] // patch_size[i] for i in range(3))
        self.n_patches = self.n_patches_per_dim[0] * self.n_patches_per_dim[1] * self.n_patches_per_dim[2]
        

        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):

        B, C, D, H, W = x.shape
        

        x = self.proj(x)
        

        x = x.flatten(2)
        x = x.transpose(1, 2)
        
        return x

class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, embed_dim=384, n_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        

        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class TransformerBlock(nn.Module):
    
    def __init__(self, embed_dim=384, n_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        

        x = x + self.mlp(self.norm2(x))
        
        return x

class VisionTransformer3D(nn.Module):
    
    def __init__(self, img_size=(96, 96, 96), patch_size=(8, 8, 8), in_channels=1, 
                 embed_dim=384, n_layers=6, n_heads=8, mlp_ratio=4.0, 
                 num_classes=2, dropout=0.1):
        super().__init__()
        

        self.patch_embed = PatchEmbedding3D_ViT(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        

        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        

        self._init_weights()
        
    def _init_weights(self):
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        

        x = self.patch_embed(x)
        

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        

        x = x + self.pos_embed
        x = self.dropout(x)
        

        for block in self.blocks:
            x = block(x)
        

        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits

class LobeExpertViT(nn.Module):
    
    def __init__(self, input_size=(96, 96, 96), dropout_rate=0.1):
        super(LobeExpertViT, self).__init__()
        

        self.vit = VisionTransformer3D(
            img_size=input_size,
            patch_size=(8, 8, 8),
            in_channels=1,
            embed_dim=128,
            n_layers=3,
            n_heads=4,
            mlp_ratio=1.5,
            num_classes=2,
            dropout=dropout_rate
        )
        
    def forward(self, x):
        return self.vit(x)





def create_model(model_type, input_size=(96, 96, 96), dropout_rate=None):
    
    model_type = model_type.lower()
    
    if model_type == 'cnn':
        dropout_rate = dropout_rate or 0.5
        return LobeExpertCNN(input_size=input_size, dropout_rate=dropout_rate)
    
    elif model_type == 'mamba':
        if not MAMBA_AVAILABLE:
            raise ImportError("Mamba models require mamba-ssm. Install with: pip install mamba-ssm")
        dropout_rate = dropout_rate or 0.2
        return LobeExpertMamba(input_size=input_size, dropout_rate=dropout_rate)
    
    elif model_type == 'vit':
        dropout_rate = dropout_rate or 0.2
        return LobeExpertViT(input_size=input_size, dropout_rate=dropout_rate)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: cnn, mamba, vit")

def get_model_config(model_type):
    
    model_type = model_type.lower()
    
    if model_type == 'cnn':
        return {
            'batch_size': 4,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'max_epochs': 100,
            'early_stopping_patience': 15,
            'scheduler_type': 'ReduceLROnPlateau',
            'label_smoothing': 0.05,
            'dropout_rate': 0.5,
            'gradient_clip': 1.0
        }
    
    elif model_type == 'mamba':
        return {
            'batch_size': 4,
            'learning_rate': 1e-4,
            'weight_decay': 1e-3,
            'max_epochs': 80,
            'early_stopping_patience': 20,
            'scheduler_type': 'CosineAnnealingLR',
            'label_smoothing': 0.2,
            'dropout_rate': 0.2,
            'gradient_clip': 1.0
        }
    
    elif model_type == 'vit':
        return {
            'batch_size': 2,
            'learning_rate': 5e-5,
            'weight_decay': 1e-3,
            'max_epochs': 80,
            'early_stopping_patience': 20,
            'scheduler_type': 'CosineAnnealingLR',
            'label_smoothing': 0.2,
            'dropout_rate': 0.2,
            'gradient_clip': 1.0
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")





def train_single_expert(task, expert_idx, fold_idx, nifti_dir, lobe_masks_dir, labels_file, 
                       output_dir, model_type='cnn'):
    
    expert_names = ["Lobe_1_LUL", "Lobe_2_LLL", "Lobe_3_RUL", "Lobe_4_RML", 
                   "Lobe_5_RLL", "Left_Lung", "Right_Lung"]
    expert_name = expert_names[expert_idx]
    model_type = model_type.lower()
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Expert: {expert_name} (Index: {expert_idx})")
    print(f"Task: {task}")
    print(f"Fold: {fold_idx}")
    print(f"{'='*60}")
    

    fold_output_dir = os.path.join(output_dir, task, f"fold_{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)
    

    file_list, mask_list, labels, patient_ids = load_image_and_mask_data(
        labels_file, nifti_dir, lobe_masks_dir, task
    )
    

    k_folds = 5
    consistent_folds = create_consistent_folds(file_list, labels, patient_ids, k_folds=k_folds, seed=42)
    
    if fold_idx >= len(consistent_folds):
        raise ValueError(f"Fold index {fold_idx} out of range. Available folds: 0-{len(consistent_folds)-1}")
    
    fold_data = consistent_folds[fold_idx]
    

    train_idx = fold_data['train_val_idx']
    holdout_idx = fold_data['holdout_idx']
    val_mask = fold_data['val_mask']
    test_mask = fold_data['test_mask']
    

    holdout_files = np.array(file_list)[holdout_idx]
    holdout_masks = np.array(mask_list)[holdout_idx]
    holdout_labels = np.array(labels)[holdout_idx]
    

    val_files = holdout_files[val_mask]
    val_masks = holdout_masks[val_mask]
    val_labels = holdout_labels[val_mask]
    
    test_files = holdout_files[test_mask]
    test_masks = holdout_masks[test_mask]
    test_labels = holdout_labels[test_mask]
    
    print(f"Data split - Train: {len(train_idx)}, Val: {len(val_files)}, Test: {len(test_files)}")
    print(f"Train labels: {Counter(np.array(labels)[train_idx])}")
    print(f"Val labels: {Counter(val_labels)}")
    print(f"Test labels: {Counter(test_labels)}")
    

    train_data = prepare_data_dicts(np.array(file_list)[train_idx], 
                                  np.array(mask_list)[train_idx], 
                                  np.array(labels)[train_idx])
    val_data = prepare_data_dicts(val_files, val_masks, val_labels)
    test_data = prepare_data_dicts(test_files, test_masks, test_labels)
    

    train_transforms = get_transforms(is_training=True, model_type=model_type)
    val_test_transforms = get_transforms(is_training=False, model_type=model_type)
    
    train_dataset = Dataset(data=train_data, transform=train_transforms)
    val_dataset = Dataset(data=val_data, transform=val_test_transforms)
    test_dataset = Dataset(data=test_data, transform=val_test_transforms)
    

    config = get_model_config(model_type)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    model = create_model(model_type, dropout_rate=config['dropout_rate']).to(device)
    

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'], 
        betas=(0.9, 0.999)
    )
    

    if config['scheduler_type'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    elif config['scheduler_type'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
    else:
        scheduler = None
    

    start_epoch = 0
    checkpoint_pattern = f"{expert_name}_{model_type}_checkpoint_epoch_*.pth"
    existing_checkpoints = glob.glob(os.path.join(fold_output_dir, checkpoint_pattern))
    
    if existing_checkpoints:
        checkpoint_epochs = []
        for cp in existing_checkpoints:
            try:
                epoch_num = int(os.path.basename(cp).split('_epoch_')[1].split('.pth')[0])
                checkpoint_epochs.append((epoch_num, cp))
            except:
                continue
        
        if checkpoint_epochs:
            checkpoint_epochs.sort(key=lambda x: x[0], reverse=True)
            latest_epoch, latest_checkpoint = checkpoint_epochs[0]
            print(f"\n🔄 Found existing checkpoint: {os.path.basename(latest_checkpoint)}")
            print(f"   Resuming from epoch {latest_epoch + 1}")
            
            try:
                model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
                start_epoch = latest_epoch
                print(f"✅ Successfully loaded checkpoint from epoch {latest_epoch}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load checkpoint: {e}")
                print(f"   Starting training from scratch")
                start_epoch = 0
    

    final_model_path = os.path.join(fold_output_dir, f"{expert_name}_{model_type}_final_model.pth")
    if os.path.exists(final_model_path):
        print(f"\n✅ Final model already exists: {os.path.basename(final_model_path)}")
        print(f"   Skipping training (model already complete)")
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        model.eval()
        test_outputs = []
        test_labels_list = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Final Test Evaluation"):
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                target = batch["label"].to(device)
                expert_mask = create_expert_mask(mask, expert_idx)
                masked_image = image * expert_mask
                output = model(masked_image)
                test_outputs.append(F.softmax(output, dim=1).cpu())
                test_labels_list.append(target.cpu())
        test_outputs = torch.cat(test_outputs, dim=0)
        test_labels = torch.cat(test_labels_list, dim=0)
        test_probs = test_outputs[:, 1].numpy()
        test_auc = roc_auc_score(test_labels.numpy(), test_probs)
        test_preds = test_outputs.argmax(dim=1).numpy()
        test_acc = accuracy_score(test_labels.numpy(), test_preds)
        return {
            'expert_name': expert_name,
            'expert_idx': expert_idx,
            'task': task,
            'fold': fold_idx,
            'model_type': model_type,
            'final_test_metrics': {
                'auc': float(test_auc),
                'accuracy': float(test_acc)
            },
            'model_path': final_model_path,
            'status': 'COMPLETED (already existed)'
        }
    

    train_labels_array = np.array(labels)[train_idx]
    class_counts = Counter(train_labels_array)
    total_samples = len(train_labels_array)
    class_weights = torch.tensor([total_samples / (2 * class_counts.get(i, 1)) for i in range(2)], 
                                dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config['label_smoothing'])
    
    print(f"Class weights: {class_weights}")
    print(f"Model config: {config}")
    

    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=0.001, 
        mode='overfitting_check',
        overfitting_threshold=0.15
    )
    

    history = {
        'train_loss': [], 'train_auc': [], 'train_acc': [],
        'val_loss': [], 'val_auc': [], 'val_acc': [],
        'learning_rates': []
    }
    

    for epoch in range(start_epoch, config['max_epochs']):

        model.train()
        total_train_loss = 0
        train_outputs = []
        train_labels_list = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} - Training"):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            target = batch["label"].to(device)
            

            expert_mask = create_expert_mask(mask, expert_idx)
            

            masked_image = image * expert_mask
            
            optimizer.zero_grad()
            output = model(masked_image)
            loss = criterion(output, target)
            loss.backward()
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
            
            optimizer.step()
            
            total_train_loss += loss.item()
            train_outputs.append(F.softmax(output, dim=1).detach().cpu())
            train_labels_list.append(target.cpu())
            

            if len(train_outputs) % (50 if model_type == 'cnn' else 20) == 0:
                torch.cuda.empty_cache()
        

        model.eval()
        total_val_loss = 0
        val_outputs = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                target = batch["label"].to(device)
                

                expert_mask = create_expert_mask(mask, expert_idx)
                

                masked_image = image * expert_mask
                
                output = model(masked_image)
                loss = criterion(output, target)
                
                total_val_loss += loss.item()
                val_outputs.append(F.softmax(output, dim=1).cpu())
                val_labels_list.append(target.cpu())
        

        train_outputs = torch.cat(train_outputs)
        train_labels = torch.cat(train_labels_list)
        train_probs = train_outputs[:, 1].numpy()
        train_preds = train_outputs.argmax(dim=1).numpy()
        
        val_outputs = torch.cat(val_outputs)
        val_labels = torch.cat(val_labels_list)
        val_probs = val_outputs[:, 1].numpy()
        val_preds = val_outputs.argmax(dim=1).numpy()
        

        try:
            train_auc = roc_auc_score(train_labels.numpy(), train_probs)
        except ValueError:
            train_auc = 0.5
            
        try:
            val_auc = roc_auc_score(val_labels.numpy(), val_probs)
        except ValueError:
            val_auc = 0.5
        
        train_acc = accuracy_score(train_labels.numpy(), train_preds)
        val_acc = accuracy_score(val_labels.numpy(), val_preds)
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        

        history['train_loss'].append(float(avg_train_loss))
        history['train_auc'].append(float(train_auc))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(avg_val_loss))
        history['val_auc'].append(float(val_auc))
        history['val_acc'].append(float(val_acc))
        history['learning_rates'].append(float(optimizer.param_groups[0]['lr']))
        
        print(f"Epoch {epoch+1}/{config['max_epochs']}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        

        if scheduler is not None:
            if config['scheduler_type'] == 'ReduceLROnPlateau':
                scheduler.step(val_auc)
            else:
                scheduler.step()
        

        if epoch >= 15:
            if early_stopping(avg_val_loss, val_auc, train_auc, model):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(fold_output_dir, f"{expert_name}_{model_type}_checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
    

    final_model_path = os.path.join(fold_output_dir, f"{expert_name}_{model_type}_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    

    model.eval()
    test_outputs = []
    test_labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Test Evaluation"):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            target = batch["label"].to(device)
            
            expert_mask = create_expert_mask(mask, expert_idx)
            masked_image = image * expert_mask
            
            output = model(masked_image)
            test_outputs.append(F.softmax(output, dim=1).cpu())
            test_labels_list.append(target.cpu())
    
    test_outputs = torch.cat(test_outputs)
    test_labels = torch.cat(test_labels_list)
    test_probs = test_outputs[:, 1].numpy()
    test_preds = test_outputs.argmax(dim=1).numpy()
    
    try:
        test_auc = roc_auc_score(test_labels.numpy(), test_probs)
    except ValueError:
        test_auc = 0.5
        
    test_acc = accuracy_score(test_labels.numpy(), test_preds)
    

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels.numpy(), test_preds, average='weighted', zero_division=0
    )
    

    final_val_auc = history['val_auc'][-1] if history['val_auc'] else 0.5
    

    if model_type == 'cnn':
        architecture_info = {
            'type': 'CNN 3D',
            'conv_channels': [24, 48, 96],
            'fc_layers': [256, 64],
            'dropout_rate': config['dropout_rate']
        }
    elif model_type == 'mamba':
        architecture_info = {
            'type': 'Mamba 3D',
            'd_model': 128,
            'n_layers': 3,
            'd_state': 32,
            'd_conv': 4,
            'expand': 2,
            'patch_size': [8, 8, 8],
            'dropout_rate': config['dropout_rate']
        }
    elif model_type == 'vit':
        architecture_info = {
            'type': 'Vision Transformer 3D',
            'embed_dim': 128,
            'n_layers': 3,
            'n_heads': 4,
            'patch_size': [8, 8, 8],
            'mlp_ratio': 1.5,
            'dropout_rate': config['dropout_rate']
        }
    

    results = {
        'expert_name': expert_name,
        'expert_idx': int(expert_idx),
        'task': task,
        'fold': int(fold_idx),
        'model_type': model_type.upper(),
        'training_completed': True,
        'final_val_auc': float(final_val_auc),
        'final_test_metrics': {
            'auc': float(test_auc),
            'accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'training_history': history,
        'data_split': {
            'train_size': int(len(train_idx)),
            'val_size': int(len(val_files)), 
            'test_size': int(len(test_files)),
            'train_labels': {str(k): int(v) for k, v in Counter(np.array(labels)[train_idx]).items()},
            'val_labels': {str(k): int(v) for k, v in Counter(val_labels).items()},
            'test_labels': {str(k): int(v) for k, v in Counter(test_labels).items()}
        },
        'model_path': final_model_path,
        'model_architecture': architecture_info,
        'training_config': config,
        'regularization_applied': {
            'dropout': True,
            'dropout_rate': config['dropout_rate'],
            'label_smoothing': config['label_smoothing'],
            'weight_decay': config['weight_decay'],
            'gradient_clipping': True,
            'data_augmentation': True,
            'early_stopping_patience': config['early_stopping_patience'],
            'scheduler_type': config['scheduler_type']
        },
        'timestamp': datetime.now().isoformat()
    }
    

    results_file = os.path.join(fold_output_dir, f"{expert_name}_{model_type}_results.json")
    safe_save_json(results, results_file)
    
    print(f"\n{'='*60}")
    print(f"FINAL {model_type.upper()} RESULTS - {expert_name}")
    print(f"{'='*60}")
    print(f"Final Val AUC: {final_val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Val-Test AUC Gap: {abs(final_val_auc - test_auc):.4f}")
    print(f"Results saved to: {results_file}")
    
    return results





if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Expert Training Module')
    parser.add_argument('--task', type=str, required=True,
                       help='Task to analyze (e.g., has_ILD, death_3, etc.)')
    parser.add_argument('--model_type', type=str, required=True, choices=['cnn', 'mamba', 'vit'],
                       help='Model type to use')
    parser.add_argument('--expert_idx', type=int, required=True, choices=range(7),
                       help='Expert index (0-6)')
    parser.add_argument('--fold_idx', type=int, required=True, choices=range(5),
                       help='Fold index (0-4)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--nifti_dir', type=str, default=None,
                       help='Directory containing NIFTI files')
    parser.add_argument('--lobe_masks_dir', type=str, default=None,
                       help='Directory containing lobe masks')
    parser.add_argument('--labels_path', type=str, default=None,
                       help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device number to use')
    
    args = parser.parse_args()
    
    import json
    from pathlib import Path
    
    if not Path(args.config).exists():
        parser.error(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    data_paths = config.get('data_paths', {})
    output_config = config.get('output', {})
    
    if args.nifti_dir is None:
        args.nifti_dir = data_paths.get('nifti_dir')
    if args.lobe_masks_dir is None:
        args.lobe_masks_dir = data_paths.get('lobe_masks_dir')
    if args.labels_path is None:
        args.labels_path = data_paths.get('labels_file')
    if args.output_dir is None:
        args.output_dir = output_config.get('unified_expert_results')
    
    if not args.nifti_dir or not args.lobe_masks_dir or not args.labels_path or not args.output_dir:
        parser.error("Missing required paths in config. Please check data_paths and output sections.")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    expert_names = ["Lobe_1_LUL", "Lobe_2_LLL", "Lobe_3_RUL", "Lobe_4_RML", 
                   "Lobe_5_RLL", "Left_Lung", "Right_Lung"]
    
    print(f"Starting unified training:")
    print(f"  Task: {args.task}")
    print(f"  Model: {args.model_type.upper()}")
    print(f"  Expert: {expert_names[args.expert_idx]} (Index: {args.expert_idx})")
    print(f"  Fold: {args.fold_idx}")
    print(f"  GPU: {args.gpu}")
    

    results = train_single_expert(
        task=args.task,
        expert_idx=args.expert_idx,
        fold_idx=args.fold_idx,
        nifti_dir=args.nifti_dir,
        lobe_masks_dir=args.lobe_masks_dir,
        labels_file=args.labels_path,
        output_dir=args.output_dir,
        model_type=args.model_type
    )
    
    print(f"\n{args.model_type.upper()} training completed successfully!")
    print(f"Expert: {results['expert_name']}")
    print(f"Final Test AUC: {results['final_test_metrics']['auc']:.4f}")
    print(f"Validation-Test AUC Gap: {abs(results['final_val_auc'] - results['final_test_metrics']['auc']):.4f}")