

import os
import sys
import json
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from monai.networks.nets import SwinUNETR
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord, Resized
from monai.data import Dataset

try:
    from experts import (
        load_image_and_mask_data, 
        create_consistent_folds,
        prepare_data_dicts,
        get_transforms,
        create_expert_mask,
        get_patient_id
    )
except ImportError:
    print("⚠️ Warning: Could not import data loading functions. Make sure experts.py is available.")
    sys.exit(1)

SUPPORTED_EXPERT_TYPES = ['vit', 'mamba', 'cnn']
WEIGHT_MODES = ['fixed', 'learned_moe', 'dynamic_moe']


class MoEWeightedSwinUNETR(nn.Module):
    
    def __init__(self, 
                 img_size=(96, 96, 96),
                 in_channels=1,
                 out_channels=2,
                 feature_size=48,
                 use_checkpoint=True,
                 spatial_dims=3,
                 expert_weights=None,
                 expert_type='cnn',
                 weight_mode='learned_moe',
                 num_experts=7,
                 expert_indices=None,
                 temperature=1.0,
                 dropout_rate=0.1):
        super().__init__()
        
        self.expert_type = expert_type
        self.weight_mode = weight_mode
        self.temperature = temperature
        self.num_experts = num_experts
        
        if expert_indices is None:
            if num_experts == 2:
                self.expert_indices = [5, 6]
                self.region_names = ['Left_Lung', 'Right_Lung']
            elif num_experts == 5:
                self.expert_indices = [0, 1, 2, 3, 4]
                self.region_names = ['Lobe_1_LUL', 'Lobe_2_LLL', 'Lobe_3_RUL', 'Lobe_4_RML', 'Lobe_5_RLL']
            elif num_experts == 7:
                self.expert_indices = [0, 1, 2, 3, 4, 5, 6]
                self.region_names = ['Lobe_1_LUL', 'Lobe_2_LLL', 'Lobe_3_RUL', 'Lobe_4_RML', 'Lobe_5_RLL', 'Left_Lung', 'Right_Lung']
            else:
                raise ValueError(f"num_experts must be 2, 5, or 7, got {num_experts}")
        else:
            self.expert_indices = expert_indices
            if num_experts == 2:
                self.region_names = ['Left_Lung', 'Right_Lung']
            elif num_experts == 5:
                self.region_names = ['Lobe_1_LUL', 'Lobe_2_LLL', 'Lobe_3_RUL', 'Lobe_4_RML', 'Lobe_5_RLL']
            elif num_experts == 7:
                self.region_names = ['Lobe_1_LUL', 'Lobe_2_LLL', 'Lobe_3_RUL', 'Lobe_4_RML', 'Lobe_5_RLL', 'Left_Lung', 'Right_Lung']
            else:
                self.region_names = [f'Expert_{i}' for i in range(num_experts)]
        
        self.swin_backbone = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size * 8,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.backbone_feature_dim = feature_size * 8
        self.expert_feature_dim = 64
        
        self.expert_extractors = nn.ModuleList([
            self._create_expert_extractor(in_channels) for _ in range(self.num_experts)
        ])
        
        self.region_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
                nn.InstanceNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 8, kernel_size=3, padding=1),
                nn.InstanceNorm3d(8),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(8),
                nn.Flatten(),
                nn.Linear(8 * 8 * 8 * 8, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(self.num_experts)
        ])
        
        if expert_weights is not None:
            expert_weights = np.array([float(w) for w in expert_weights])
            if len(expert_weights) != self.num_experts:
                if len(expert_weights) == 7 and self.num_experts == 2:
                    expert_weights = expert_weights[[5, 6]]
                elif len(expert_weights) == 7 and self.num_experts == 5:
                    expert_weights = expert_weights[:5]
                elif len(expert_weights) == 5 and self.num_experts == 2:
                    raise ValueError("Cannot map 5 expert weights to 2 experts")
                else:
                    raise ValueError(f"Expert weights length {len(expert_weights)} doesn't match num_experts {self.num_experts}")
            
            normalized_weights = expert_weights / np.sum(expert_weights)
            print(f"\nInitialized MoE weights from {expert_type.upper()} expert results ({self.num_experts} experts):")
            for name, auc, weight in zip(self.region_names, expert_weights, normalized_weights):
                print(f"  {name}: AUC={auc:.4f}, Weight={weight:.4f}")
            
            self.register_buffer('base_weights', torch.tensor(normalized_weights, dtype=torch.float32))
        else:
            self.register_buffer('base_weights', torch.ones(self.num_experts) / self.num_experts)
        
        if weight_mode == 'learned_moe':
            self.weight_logits = nn.Parameter(torch.zeros(self.num_experts))
        elif weight_mode == 'dynamic_moe':
            self.weight_predictor = nn.Sequential(
                nn.Linear(self.backbone_feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(64, self.num_experts)
            )
        
        total_feature_dim = self.backbone_feature_dim + self.expert_feature_dim
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, out_channels)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def _create_expert_extractor(self, in_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, self.expert_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(self.expert_feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
    
    def get_region_masks(self, lobe_mask):
        masks = []
        
        for expert_idx in self.expert_indices:
            if expert_idx < 5:
                mask = (lobe_mask == expert_idx + 1).float()
            elif expert_idx == 5:
                mask = ((lobe_mask == 1) | (lobe_mask == 2)).float()
            elif expert_idx == 6:
                mask = ((lobe_mask == 3) | (lobe_mask == 4) | (lobe_mask == 5)).float()
            else:
                raise ValueError(f"Invalid expert index: {expert_idx}")
            masks.append(mask)
        
        return masks
    
    def get_current_weights(self, backbone_features=None):
        if self.weight_mode == 'fixed':
            return self.base_weights
        elif self.weight_mode == 'learned_moe':
            adjusted_logits = torch.log(self.base_weights + 1e-8) + self.weight_logits
            return F.softmax(adjusted_logits / self.temperature, dim=0)
        elif self.weight_mode == 'dynamic_moe' and backbone_features is not None:
            if len(backbone_features.shape) > 2:
                backbone_features = backbone_features.view(backbone_features.shape[0], -1)
            weight_logits = self.weight_predictor(backbone_features)
            base_batch = self.base_weights.unsqueeze(0).expand(backbone_features.shape[0], -1)
            dynamic_weights = F.softmax(weight_logits / self.temperature, dim=1)
            combined = base_batch * dynamic_weights
            return combined / combined.sum(dim=1, keepdim=True)
        else:
            return self.base_weights
    
    def forward(self, x, lobe_mask=None, return_weights=False, return_expert_features=False):
        batch_size = x.shape[0]
        
        backbone_features = self.swin_backbone(x)
        pooled_backbone = self.global_pool(backbone_features)
        pooled_backbone = pooled_backbone.view(batch_size, -1)
        
        if lobe_mask is None:
            padded_features = torch.cat([
                pooled_backbone,
                torch.zeros(batch_size, self.expert_feature_dim, device=pooled_backbone.device)
            ], dim=1)
            features = self.feature_fusion(padded_features)
            output = self.classifier(features)
            
            if return_weights:
                return output, None
            return output
        
        region_masks = self.get_region_masks(lobe_mask)
        current_weights = self.get_current_weights(pooled_backbone)
        
        expert_features = []
        expert_attentions = []
        
        for i, (extractor, attention, mask) in enumerate(zip(self.expert_extractors, self.region_attention, region_masks)):
            masked_input = x * mask.unsqueeze(1)
            expert_feat = extractor(masked_input)
            expert_feat = expert_feat.view(batch_size, -1)
            expert_features.append(expert_feat)
            
            attn = attention(masked_input)
            expert_attentions.append(attn)
        
        expert_features = torch.stack(expert_features, dim=1)
        expert_attentions = torch.stack(expert_attentions, dim=1)
        expert_attentions = expert_attentions.squeeze(-1)
        
        if len(current_weights.shape) == 1:
            current_weights = current_weights.unsqueeze(0).expand(batch_size, -1)
        
        combined_weights = current_weights * expert_attentions
        combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        weighted_expert_features = expert_features * combined_weights.unsqueeze(-1)
        aggregated_expert_features = weighted_expert_features.sum(dim=1)
        
        combined_features = torch.cat([pooled_backbone, aggregated_expert_features], dim=1)
        features = self.feature_fusion(combined_features)
        output = self.classifier(features)
        
        if return_weights:
            return output, combined_weights
        if return_expert_features:
            return output, {
                'expert_features': expert_features,
                'expert_weights': combined_weights,
                'backbone_features': pooled_backbone
            }
        return output






