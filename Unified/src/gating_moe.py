

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from monai.networks.nets import SwinUNETR
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord, Resized
from monai.data import Dataset
import pandas as pd
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import argparse
import sys


from experts import (
    load_image_and_mask_data, 
    create_consistent_folds,
    prepare_data_dicts,
    get_transforms,
    create_expert_mask,
    get_patient_id
)


try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("⚠️ mamba_ssm not available. Please install with: pip install mamba-ssm")
    MAMBA_AVAILABLE = False
    
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.d_model = d_model
            self.linear = nn.Linear(d_model, d_model)
            print("⚠️ Using fallback Mamba implementation")
        
        def forward(self, x):
            return self.linear(x)






class SharedExpertExtractor(nn.Module):
    
    def __init__(self, in_channels, expert_feature_dim=128):
        super().__init__()
        self.expert_feature_dim = expert_feature_dim
        self.extractor = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, expert_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(expert_feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
    
    def forward(self, x):
        return self.extractor(x)


class SharedRegionAttention(nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
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
        )
    
    def forward(self, x):
        return self.attention(x)


class GatingMoEBase(nn.Module):
    
    def __init__(self, 
                 in_channels=1,
                 out_channels=2,
                 expert_feature_dim=128,
                 gating_weights=None,
                 gating_strategy_name="unknown",
                 weight_mode='gating_fixed',
                 temperature=1.0,
                 dropout_rate=0.3):
        super().__init__()
        

        self.num_experts = 5
        self.expert_feature_dim = expert_feature_dim
        self.weight_mode = weight_mode
        self.temperature = temperature
        self.gating_strategy_name = gating_strategy_name
        

        self.expert_extractors = nn.ModuleList([
            SharedExpertExtractor(in_channels, expert_feature_dim) 
            for _ in range(self.num_experts)
        ])
        
        self.region_attention = nn.ModuleList([
            SharedRegionAttention(in_channels) 
            for _ in range(self.num_experts)
        ])
        

        self._initialize_gating_weights(gating_weights, weight_mode, dropout_rate)
        

        self.dropout = nn.Dropout(dropout_rate)
        
    def _initialize_gating_weights(self, gating_weights, weight_mode, dropout_rate):
        
        if gating_weights is not None:

            gating_weights = np.array([float(w) for w in gating_weights[:5]])
            print(f"\n🎯 Initialized MoE weights from best gating strategy: {self.gating_strategy_name}")
            region_names = ['Lobe_1_LUL', 'Lobe_2_LLL', 'Lobe_3_RUL', 'Lobe_4_RML', 'Lobe_5_RLL']
            for name, weight in zip(region_names, gating_weights):
                print(f"  {name}: {weight:.4f}")
            
            self.register_buffer('gating_weights', torch.tensor(gating_weights, dtype=torch.float32))
        else:

            default_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            self.register_buffer('gating_weights', torch.tensor(default_weights, dtype=torch.float32))
            print(f"⚠️ No gating weights provided, using uniform weights for 5 lobe experts")
    
        

        if weight_mode == 'gating_learnable':
            self.weight_adjustments = nn.Parameter(torch.zeros(self.num_experts))
            print(f"🔧 Learnable gating adjustments enabled")
        elif weight_mode == 'gating_dynamic':

            self._dynamic_mode = True
            self._dropout_rate = dropout_rate
            print(f"🔄 Dynamic gating prediction enabled")
        else:
            self._dynamic_mode = False
    
    def get_region_masks(self, lobe_mask):
        
        masks = []
        

        for lobe_id in range(1, 6):
            mask = (lobe_mask == lobe_id).float()
            masks.append(mask)
        
        return masks
    
    def get_current_weights(self, backbone_features=None):
        
        if self.weight_mode == 'gating_fixed':
            return self.gating_weights
        elif self.weight_mode == 'gating_learnable':
            adjusted_logits = torch.log(self.gating_weights + 1e-8) + self.weight_adjustments
            return F.softmax(adjusted_logits / self.temperature, dim=0)
        elif self.weight_mode == 'gating_dynamic' and backbone_features is not None:
            if len(backbone_features.shape) > 2:
                backbone_features = backbone_features.view(backbone_features.shape[0], -1)
            weight_logits = self.weight_predictor(backbone_features)
            

            base_batch = self.gating_weights.unsqueeze(0).expand(backbone_features.shape[0], -1)
            dynamic_weights = F.softmax(weight_logits / self.temperature, dim=1)
            combined = base_batch * dynamic_weights
            return combined / combined.sum(dim=1, keepdim=True)
        else:
            return self.gating_weights
    
    def process_experts(self, x, lobe_mask, backbone_features):
        
        if lobe_mask is None:
            return None, None, None
        
        batch_size = x.shape[0]
        region_masks = self.get_region_masks(lobe_mask)
        current_weights = self.get_current_weights(backbone_features)
        

        expert_features = []
        expert_attentions = []
        
        for i, (mask, expert_extractor, attention_module) in enumerate(
            zip(region_masks, self.expert_extractors, self.region_attention)
        ):
            masked_input = x * mask
            expert_feat = expert_extractor(masked_input).view(batch_size, -1)
            attention_weight = attention_module(masked_input)
            
            expert_features.append(expert_feat)
            expert_attentions.append(attention_weight)
        
        expert_features = torch.stack(expert_features, dim=1)
        expert_attentions = torch.stack(expert_attentions, dim=1).squeeze(-1)
        

        if len(current_weights.shape) == 1:
            gating_weights_batch = current_weights.unsqueeze(0).expand(batch_size, -1)
        else:
            gating_weights_batch = current_weights
        

        combined_weights = gating_weights_batch * expert_attentions
        combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)
        

        weighted_expert_features = expert_features * combined_weights.unsqueeze(-1)
        aggregated_expert_features = weighted_expert_features.sum(dim=1)
        
        expert_info = {
            'expert_features': expert_features,
            'expert_attentions': expert_attentions,
            'gating_weights': gating_weights_batch,
            'combined_weights': combined_weights,
            'backbone_features': backbone_features
        }
        
        return aggregated_expert_features, combined_weights, expert_info






class GatingWeightedSwinUNETR(GatingMoEBase):
    
    def __init__(self, 
                 img_size=(96, 96, 96),
                 in_channels=1,
                 out_channels=2,
                 feature_size=48,
                 use_checkpoint=True,
                 spatial_dims=3,
                 dropout_rate=0.3,
                 **kwargs):
        

        self._local_dropout_rate = dropout_rate
        
        super().__init__(in_channels=in_channels, out_channels=out_channels, 
                        dropout_rate=dropout_rate, **kwargs)
        

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
        

        if self.weight_mode == 'gating_dynamic':
            self.weight_predictor = nn.Sequential(
                nn.Linear(self.backbone_feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(self._local_dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(self._local_dropout_rate),
                nn.Linear(64, self.num_experts)
            )
        

        total_feature_dim = self.backbone_feature_dim + self.expert_feature_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self._local_dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self._local_dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self._local_dropout_rate),
            nn.Linear(128, out_channels)
        )
    
    def forward(self, x, lobe_mask=None, return_weights=False, return_expert_features=False):
        batch_size = x.shape[0]
        

        backbone_features = self.swin_backbone(x)
        pooled_backbone = self.global_pool(backbone_features).view(batch_size, -1)
        
        if lobe_mask is None:
            padded_features = torch.cat([
                pooled_backbone,
                torch.zeros(batch_size, self.expert_feature_dim, device=pooled_backbone.device)
            ], dim=1)
            features = self.feature_fusion(padded_features)
            output = self.classifier(features)
            return self._return_outputs(output, None, None, return_weights, return_expert_features)
        

        aggregated_expert_features, combined_weights, expert_info = self.process_experts(
            x, lobe_mask, pooled_backbone
        )
        

        combined_features = torch.cat([pooled_backbone, aggregated_expert_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        output = self.classifier(fused_features)
        
        return self._return_outputs(output, combined_weights, expert_info, return_weights, return_expert_features)
    
    def _return_outputs(self, output, weights, expert_info, return_weights, return_expert_features):
        returns = [output]
        if return_weights:
            returns.append(weights)
        if return_expert_features:
            returns.append(expert_info)
        
        return returns[0] if len(returns) == 1 else tuple(returns)


class GatingWeightedMamba(GatingMoEBase):
    
    def __init__(self, 
                 img_size=(96, 96, 96),
                 in_channels=1,
                 out_channels=2,
                 d_model=256,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dropout_rate=0.3,
                 **kwargs):
        

        self._local_dropout_rate = dropout_rate
        
        super().__init__(in_channels=in_channels, out_channels=out_channels, 
                        dropout_rate=dropout_rate, **kwargs)
        
        self.d_model = d_model
        

        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        
        self.feature_size = 128 * 4 * 4 * 4
        self.feature_projection = nn.Linear(self.feature_size, d_model)
        

        self.mamba_backbone = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.backbone_feature_dim = d_model
        

        if self.weight_mode == 'gating_dynamic':
            self.weight_predictor = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(self._local_dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(self._local_dropout_rate),
                nn.Linear(64, self.num_experts)
            )
        

        total_feature_dim = d_model + self.expert_feature_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self._local_dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self._local_dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self._local_dropout_rate),
            nn.Linear(128, out_channels)
        )
    
    def forward(self, x, lobe_mask=None, return_weights=False, return_expert_features=False):
        batch_size = x.shape[0]
        

        cnn_features = self.feature_extractor(x)
        cnn_features_flat = cnn_features.view(batch_size, -1)
        projected_features = self.feature_projection(cnn_features_flat)
        

        mamba_input = projected_features.unsqueeze(1)
        mamba_output = self.mamba_backbone(mamba_input)
        backbone_features = mamba_output.squeeze(1)
        
        if lobe_mask is None:
            padded_features = torch.cat([
                backbone_features,
                torch.zeros(batch_size, self.expert_feature_dim, device=backbone_features.device)
            ], dim=1)
            features = self.feature_fusion(padded_features)
            output = self.classifier(features)
            return self._return_outputs(output, None, None, return_weights, return_expert_features)
        

        aggregated_expert_features, combined_weights, expert_info = self.process_experts(
            x, lobe_mask, backbone_features
        )
        

        combined_features = torch.cat([backbone_features, aggregated_expert_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        output = self.classifier(fused_features)
        
        return self._return_outputs(output, combined_weights, expert_info, return_weights, return_expert_features)
    
    def _return_outputs(self, output, weights, expert_info, return_weights, return_expert_features):
        returns = [output]
        if return_weights:
            returns.append(weights)
        if return_expert_features:
            returns.append(expert_info)
        
        return returns[0] if len(returns) == 1 else tuple(returns)



class PatchEmbedding3D(nn.Module):
    
    def __init__(self, img_size=(96, 96, 96), patch_size=(16, 16, 16), in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = tuple([img_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.projection = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        
        x = (attention @ v).transpose(1, 2).reshape(B, N, C)
        x = self.projection(x)
        x = self.projection_dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
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
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer3D(nn.Module):
    
    def __init__(self, img_size=(96, 96, 96), patch_size=(16, 16, 16), in_channels=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding3D(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedding.num_patches
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        return x[:, 0]


class GatingWeightedViT(GatingMoEBase):
    def __init__(self, 
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                in_channels=1,
                out_channels=2,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                dropout_rate=0.3,
                **kwargs):
        

        self._local_dropout_rate = dropout_rate
        
        super().__init__(in_channels=in_channels, out_channels=out_channels, 
                        dropout_rate=dropout_rate, **kwargs)
        

        self.vit_backbone = VisionTransformer3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=self._local_dropout_rate
        )
        
        self.backbone_feature_dim = embed_dim
        

        if self.weight_mode == 'gating_dynamic':
            self.weight_predictor = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(self._local_dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(self._local_dropout_rate),
                nn.Linear(64, self.num_experts)
            )
        

        total_feature_dim = embed_dim + self.expert_feature_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self._local_dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self._local_dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self._local_dropout_rate),
            nn.Linear(128, out_channels)
        )
    
    def forward(self, x, lobe_mask=None, return_weights=False, return_expert_features=False):
        batch_size = x.shape[0]

        backbone_features = self.vit_backbone(x)
        
        if lobe_mask is None:
            padded_features = torch.cat([
                backbone_features,
                torch.zeros(batch_size, self.expert_feature_dim, device=backbone_features.device)
            ], dim=1)
            features = self.feature_fusion(padded_features)
            output = self.classifier(features)
            return self._return_outputs(output, None, None, return_weights, return_expert_features)
        

        aggregated_expert_features, combined_weights, expert_info = self.process_experts(
            x, lobe_mask, backbone_features
        )
        

        combined_features = torch.cat([backbone_features, aggregated_expert_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        output = self.classifier(fused_features)
        
        return self._return_outputs(output, combined_weights, expert_info, return_weights, return_expert_features)
    
    def _return_outputs(self, output, weights, expert_info, return_weights, return_expert_features):
        returns = [output]
        if return_weights:
            returns.append(weights)
        if return_expert_features:
            returns.append(expert_info)
        
        return returns[0] if len(returns) == 1 else tuple(returns)






class UnifiedGatingMoELoss(nn.Module):
    
    def __init__(self, base_criterion=None, weight_regularization=0.01, 
                 diversity_regularization=0.01, gating_regularization=0.005):
        super().__init__()
        self.base_criterion = base_criterion or nn.CrossEntropyLoss()
        self.weight_regularization = weight_regularization
        self.diversity_regularization = diversity_regularization
        self.gating_regularization = gating_regularization
    
    def forward(self, pred, target, expert_weights=None, expert_features=None):
        

        base_loss = self.base_criterion(pred, target)
        total_loss = base_loss
        

        if expert_weights is not None and self.gating_regularization > 0:
            if len(expert_weights.shape) > 1:
                weight_entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=1).mean()
            else:
                weight_entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum()
            
            gating_reg_loss = -self.gating_regularization * weight_entropy
            total_loss += gating_reg_loss
        

        if expert_weights is not None and self.weight_regularization > 0:
            if len(expert_weights.shape) > 1:
                uniform_target = torch.ones_like(expert_weights) / expert_weights.shape[1]
                weight_balance_loss = F.mse_loss(expert_weights, uniform_target)
            else:
                uniform_target = torch.ones_like(expert_weights) / len(expert_weights)
                weight_balance_loss = F.mse_loss(expert_weights, uniform_target)
            
            total_loss += self.weight_regularization * weight_balance_loss
        

        if expert_features is not None and self.diversity_regularization > 0:
            expert_feats = expert_features['expert_features']
            expert_feats_norm = F.normalize(expert_feats, dim=2)
            similarities = torch.bmm(expert_feats_norm, expert_feats_norm.transpose(1, 2))
            
            mask = torch.eye(similarities.shape[1], device=similarities.device).unsqueeze(0)
            similarities = similarities * (1 - mask)
            diversity_loss = self.diversity_regularization * similarities.abs().mean()
            total_loss += diversity_loss
        
        return total_loss







def load_best_gating_strategy(gating_results_dir, task, fold, weight_source_type):
    
    

    if weight_source_type == 'cnn':
        strategy_file = Path(gating_results_dir) / "gating_ensemble_results" / task / f"fold_{fold}" / "best_cnn_gating_strategy.json"
    elif weight_source_type == 'vit':
        strategy_file = Path(gating_results_dir) / "gating_ensemble_results" / task / f"fold_{fold}" / "best_vit_gating_strategy.json"
    elif weight_source_type == 'mamba':
        strategy_file = Path(gating_results_dir) / "gating_ensemble_results" / task / f"fold_{fold}" / "best_mamba_gating_strategy.json"
    elif weight_source_type == 'radiomics':
        strategy_file = Path(gating_results_dir) / "gating_ensemble_results" / task / f"fold_{fold}" / "best_radiomics_strategy.json"
    else:
        raise ValueError(f"Unknown weight source type: {weight_source_type}")
    
    if not strategy_file.exists():
        print(f"❌ Best gating strategy file not found: {strategy_file}")
        print(f"Please run Stage 2 gating extraction for {weight_source_type.upper()} first!")
        return None
    
    print(f"✅ Loading {weight_source_type.upper()}-derived gating strategy from: {strategy_file}")
    
    with open(strategy_file, 'r') as f:
        best_strategy_data = json.load(f)
    
    original_strategy = best_strategy_data['best_strategy']
    

    if original_strategy['weights'] is None:
        print(f"⚠️ Strategy '{original_strategy['name']}' has null weights, need to load from ensemble results")
        


        ensemble_results_file = strategy_file.parent / "ensemble_results_weighted_average_val_auc.json"
        
        if ensemble_results_file.exists():
            print(f"📊 Loading ensemble weights from: {ensemble_results_file}")
            with open(ensemble_results_file, 'r') as f:
                ensemble_data = json.load(f)
            

            individual_metrics = ensemble_data.get('individual_expert_metrics', {})
            weights = []
            expert_names = []
            

            expected_order = ['Lobe_1', 'Lobe_2', 'Lobe_3', 'Lobe_4', 'Lobe_5']
            
            for expert_prefix in expected_order:
                found = False
                for expert_key, metrics in individual_metrics.items():
                    if expert_key.startswith(expert_prefix):
                        weights.append(metrics['auc'])
                        expert_names.append(expert_key)
                        found = True
                        break
                
                if not found:
                    print(f"Warning: Could not find expert for {expert_prefix}, using default AUC=0.6")
                    weights.append(0.6)
                    expert_names.append(f"{expert_prefix}_missing")
            
            print(f"📊 Extracted {len(weights)} lobe expert AUC weights:")
            for name, weight in zip(expert_names, weights):
                print(f"  {name}: {weight:.4f}")
            

            weights = np.array(weights)
            normalized_weights = weights / np.sum(weights)
            
        else:
            print(f"❌ Ensemble results file not found: {ensemble_results_file}")
            print(f"Using uniform weights as fallback")
            normalized_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        strategy_weights = normalized_weights.tolist()
        
    else:

        strategy_weights = original_strategy['weights'][:5]
        print(f"📊 Using explicit weights: {strategy_weights}")
    

    modified_strategy = {
        'name': original_strategy['name'] + '_five_lobes',
        'category': original_strategy['category'],
        'weights': strategy_weights,
        'performance': original_strategy['performance']
    }
    
    print(f"✅ Final strategy weights for 5 lobe experts: {modified_strategy['weights']}")
    
    return modified_strategy


def create_model(model_type, best_gating_strategy=None, model_config=None, weight_mode='gating_fixed'):
    
    

    default_configs = {
        'swinunetr': {
            'img_size': (96, 96, 96),
            'in_channels': 1,
            'out_channels': 2,
            'feature_size': 48,
            'use_checkpoint': True,
            'dropout_rate': 0.3
        },
        'mamba': {
            'img_size': (96, 96, 96),
            'in_channels': 1,
            'out_channels': 2,
            'd_model': 256,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dropout_rate': 0.3
        },
        'vit': {
            'img_size': (96, 96, 96),
            'patch_size': (12, 12, 12),
            'in_channels': 1,
            'out_channels': 2,
            'embed_dim': 256,
            'depth': 6,
            'num_heads': 8,
            'mlp_ratio': 4.0,
            'dropout_rate': 0.3
        }
    }
    
    if model_type not in default_configs:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(default_configs.keys())}")
    
    config = default_configs[model_type].copy()
    if model_config:
        config.update(model_config)
    

    if best_gating_strategy:
        gating_weights = best_gating_strategy['weights']
        strategy_name = best_gating_strategy['name']
        print(f"\n🎯 Creating {model_type.upper()} Gating-based MoE model:")
        print(f"  Best gating strategy: {strategy_name}")
        print(f"  Category: {best_gating_strategy['category']}")
        print(f"  Validation AUC: {best_gating_strategy['performance']['auc']:.4f}")
    else:
        gating_weights = None
        strategy_name = "uniform_fallback"
        print(f"\n⚠️ No gating strategy provided for {model_type.upper()}, using uniform weights")
    
    print(f"  Weight mode: {weight_mode}")
    print(f"  Model config: {config}")
    

    config.update({
        'gating_weights': gating_weights,
        'gating_strategy_name': strategy_name,
        'weight_mode': weight_mode
    })
    
    print(f"Creating SwinUNETR with {model_type.upper()}-derived gating weights")
    model = GatingWeightedSwinUNETR(**config)
    
    return model


class EarlyStopping:
    
    def __init__(self, patience=10, min_delta=0.001, mode='max', restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score, model, epoch):
        if self.mode == 'max':
            score_improved = (self.best_score is None or 
                            score > self.best_score + self.min_delta)
        else:
            score_improved = (self.best_score is None or 
                            score < self.best_score - self.min_delta)
        
        if score_improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            print(f"  ✅ Validation improved to {score:.4f}")
        else:
            self.counter += 1
            print(f"  ⚠️  No improvement for {self.counter} epochs (best: {self.best_score:.4f} at epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"  🛑 Early stopping triggered!")
                
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"  ↩️  Restored best weights from epoch {self.best_epoch}")
                
        return self.early_stop


def train_unified_model(model, model_type, train_loader, val_loader, device,
                       num_epochs=50, learning_rate=1e-4, weight_decay=1e-2,
                       save_dir='unified_checkpoints', early_stopping_patience=10):
    
    os.makedirs(save_dir, exist_ok=True)
    

    backbone_params = []
    moe_params = []
    
    backbone_names = {
        'swinunetr': ['swin_backbone'],
        'mamba': ['feature_extractor', 'mamba_backbone'],
        'vit': ['vit_backbone']
    }
    
    for name, param in model.named_parameters():
        if any(backbone_name in name for backbone_name in backbone_names[model_type]):
            backbone_params.append(param)
        else:
            moe_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.1},
        {'params': moe_params, 'lr': learning_rate}
    ], weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    criterion = UnifiedGatingMoELoss(
        base_criterion=nn.CrossEntropyLoss(),
        weight_regularization=0.005,
        diversity_regularization=0.01,
        gating_regularization=0.005
    )
    
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=0.001,
        mode='max',
        restore_best_weights=True
    )
    
    history = {
        'train_loss': [], 'train_auc': [], 'train_acc': [],
        'val_loss': [], 'val_auc': [], 'val_acc': [],
        'gating_weights_history': [],
        'early_stopped': False,
        'stopped_epoch': None
    }
    
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        epoch_gating_weights = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            output, weights, expert_info = model(image, mask, 
                                               return_weights=True, 
                                               return_expert_features=True)
            
            loss = criterion(output, label, weights, expert_info)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            probs = F.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
            preds = output.argmax(dim=1).detach().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())
            
            if weights is not None:
                if len(weights.shape) > 1:
                    epoch_gating_weights.append(weights.mean(dim=0).detach().cpu().numpy())
                else:
                    epoch_gating_weights.append(weights.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_probs = []
        val_gating_weights = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in pbar:
                image = batch['image'].to(device)
                mask = batch['mask'].to(device)
                label = batch['label'].to(device)
                
                output, weights, expert_info = model(image, mask, 
                                                   return_weights=True, 
                                                   return_expert_features=True)
                
                loss = criterion(output, label, weights, expert_info)
                val_loss += loss.item()
                
                probs = F.softmax(output, dim=1)[:, 1].cpu().numpy()
                preds = output.argmax(dim=1).cpu().numpy()
                
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_labels.extend(label.cpu().numpy())
                
                if weights is not None:
                    if len(weights.shape) > 1:
                        val_gating_weights.append(weights.mean(dim=0).cpu().numpy())
                    else:
                        val_gating_weights.append(weights.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
        
        train_val_gap = train_auc - val_auc
        if train_val_gap > 0.15:
            print(f"\n🛑 OVERFITTING DETECTED!")
            print(f"   Train AUC: {train_auc:.4f}")
            print(f"   Val AUC: {val_auc:.4f}")
            print(f"   Gap: {train_val_gap:.4f} > 0.15 threshold")
            print(f"   Stopping training early at epoch {epoch + 1}")
            
            history['early_stopped'] = True
            history['stopped_epoch'] = epoch + 1
            history['stop_reason'] = 'overfitting'
            break
        

        if early_stopping(val_auc, model, epoch + 1):
            history['early_stopped'] = True
            history['stopped_epoch'] = epoch + 1
            history['stop_reason'] = 'early_stopping'
            print(f"\n🛑 Training stopped early at epoch {epoch + 1}")
            break
        
        scheduler.step()
        

        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        
        if epoch_gating_weights:
            avg_epoch_weights = np.mean(epoch_gating_weights, axis=0)
            history['gating_weights_history'].append(avg_epoch_weights.tolist())
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        
        if val_gating_weights:
            avg_val_weights = np.mean(val_gating_weights, axis=0)
            region_names = ['L1', 'L2', 'L3', 'L4', 'L5']
            print(f"{model_type.upper()} gating weights:", {name: f"{w:.4f}" for name, w in zip(region_names, avg_val_weights)})
        

        if early_stopping.restore_best_weights and early_stopping.best_weights is not None:
            checkpoint = {
                'epoch': early_stopping.best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': early_stopping.best_score,
                'history': history,
                'model_type': model_type,
                'model_config': {
                    'weight_mode': model.weight_mode,
                    'gating_strategy_name': model.gating_strategy_name,
                    'gating_weights': model.gating_weights.cpu().numpy().tolist()
                }
            }
            torch.save(checkpoint, os.path.join(save_dir, f'best_{model_type}_gating_model.pth'))
    
    print(f"\n🎉 {model_type.upper()} training completed!")
    return history


def evaluate_unified_model(model, model_type, test_loader, device, save_results=True, output_dir='.'):
    
    print(f"🔬 Final {model_type.upper()} test evaluation (using test data for the first time!)")
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_gating_weights = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Final {model_type.upper()} Test Evaluation'):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            label = batch['label'].to(device)
            
            output, weights, expert_info = model(image, mask, 
                                               return_weights=True, 
                                               return_expert_features=True)
            
            probs = F.softmax(output, dim=1)[:, 1].cpu().numpy()
            preds = output.argmax(dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())
            
            if weights is not None:
                if len(weights.shape) > 1:
                    all_gating_weights.extend(weights.cpu().numpy())
                else:
                    all_gating_weights.extend([weights.cpu().numpy()] * len(label))
    

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_gating_weights = np.array(all_gating_weights)
    
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    cm = confusion_matrix(all_labels, all_preds)
    

    avg_gating_weights = np.mean(all_gating_weights, axis=0) if len(all_gating_weights) > 0 else None
    weight_std = np.std(all_gating_weights, axis=0) if len(all_gating_weights) > 0 else None
    
    pos_weights = np.mean(all_gating_weights[all_labels == 1], axis=0) if np.sum(all_labels == 1) > 0 else None
    neg_weights = np.mean(all_gating_weights[all_labels == 0], axis=0) if np.sum(all_labels == 0) > 0 else None
    
    results = {
        'model_type': model_type,
        'performance': {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'confusion_matrix': cm.tolist()
        },
        'gating_analysis': {
            'avg_gating_weights': avg_gating_weights.tolist() if avg_gating_weights is not None else None,
            'weight_std': weight_std.tolist() if weight_std is not None else None,
            'positive_class_weights': pos_weights.tolist() if pos_weights is not None else None,
            'negative_class_weights': neg_weights.tolist() if neg_weights is not None else None
        },
        'predictions': {
            'labels': all_labels.tolist(),
            'predictions': all_preds.tolist(),
            'probabilities': all_probs.tolist()
        }
    }
    
    print(f"\n📊 {model_type.upper()} Gating-based MoE Final Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    if avg_gating_weights is not None:
        region_names = ['Lobe_1', 'Lobe_2', 'Lobe_3', 'Lobe_4', 'Lobe_5']
        print(f"\n🎯 Final {model_type.upper()} Gating Weight Analysis:")
        for name, weight, std in zip(region_names, avg_gating_weights, weight_std):
            print(f"  {name}: {weight:.4f} ± {std:.4f}")
    
    if save_results:
        results_path = os.path.join(output_dir, f'{model_type}_gating_test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n💾 Results saved to: {results_path}")
    
    return results



def run_unified_training_pipeline(task, fold, weight_source_type,
                                 expert_results_dir=None,
                                 nifti_dir=None,
                                 lobe_masks_dir=None, 
                                 labels_file=None,
                                 weight_mode='gating_fixed',
                                 num_epochs=50,
                                 batch_size=4,
                                 learning_rate=1e-4,
                                 early_stopping_patience=10,
                                 output_dir=None,
                                 gpu='0'):
    
    

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting SwinUNETR training with {weight_source_type.upper()}-derived weights")
  

    task_output_dir = os.path.join(output_dir, task, f"fold_{fold}", "swinunetr", f"{weight_source_type}_weights")
    os.makedirs(task_output_dir, exist_ok=True)
 
    print(f"\n📁 Output directory: {task_output_dir}")
    

    print(f"\n🎯 Step 1: Loading best gating strategy from step 3...")
    best_gating_strategy = load_best_gating_strategy(expert_results_dir, task, fold, weight_source_type)
    
    if best_gating_strategy is None:
        return None
    
    print(f"✅ Using {weight_source_type.upper()}-derived gating strategy: {best_gating_strategy['name']}")
    

    print(f"\n📊 Step 2: Loading data for task '{task}', fold {fold}...")
    
    file_list, mask_list, labels, patient_ids = load_image_and_mask_data(
        labels_file, nifti_dir, lobe_masks_dir, task
    )
    
    k_folds = 5
    consistent_folds = create_consistent_folds(file_list, labels, patient_ids, k_folds=k_folds, seed=42)
    fold_data = consistent_folds[fold]
    

    holdout_idx = fold_data['holdout_idx']
    val_mask = fold_data['val_mask']
    test_mask = fold_data['test_mask']
    
    holdout_files = np.array(file_list)[holdout_idx]
    holdout_masks = np.array(mask_list)[holdout_idx]
    holdout_labels = np.array(labels)[holdout_idx]
    
    val_files = holdout_files[val_mask]
    val_masks_list = holdout_masks[val_mask]
    val_labels_list = holdout_labels[val_mask]
    
    test_files = holdout_files[test_mask]
    test_masks_list = holdout_masks[test_mask]
    test_labels_list = holdout_labels[test_mask]
    
    all_indices = np.arange(len(file_list))
    train_idx = np.setdiff1d(all_indices, holdout_idx)
    
    print(f"Data split sizes - Train: {len(train_idx)}, Val: {len(val_files)}, Test: {len(test_files)}")
    

    train_data = prepare_data_dicts(
        [file_list[i] for i in train_idx],
        [mask_list[i] for i in train_idx], 
        [labels[i] for i in train_idx]
    )
    
    val_data = prepare_data_dicts(val_files, val_masks_list, val_labels_list)
    test_data = prepare_data_dicts(test_files, test_masks_list, test_labels_list)
    
    transforms = get_transforms()
    
    train_dataset = Dataset(data=train_data, transform=transforms)
    val_dataset = Dataset(data=val_data, transform=transforms)
    test_dataset = Dataset(data=test_data, transform=transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    

    print(f"\n🏗️  Step 3: Creating {weight_mode.upper()} Gating-based MoE model...")
    model = create_model(
        model_type='swinunetr',
        best_gating_strategy=best_gating_strategy,
        weight_mode=weight_mode
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    

    print(f"\n🎯 Step 4: Starting SwinUNETR Gating-based MoE training...")
    
    history = train_unified_model(
        model=model,
        model_type='swinunetr',
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_dir=task_output_dir,
        early_stopping_patience=early_stopping_patience
    )
    

    with open(os.path.join(task_output_dir, 'swinunetr_gating_training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    

    print(f"\n📈 Step 5: Loading best SwinUNETR model and evaluating on TEST data...")
    best_model_path = os.path.join(task_output_dir, 'best_swinunetr_gating_model.pth')
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best SwinUNETR model with validation AUC: {checkpoint['val_auc']:.4f}")
    
    test_results = evaluate_unified_model(
        model=model,
        model_type='swinunetr',
        test_loader=test_loader,
        device=device,
        save_results=True,
        output_dir=task_output_dir
    )
    

    print(f"\n📋 Step 6: Creating SwinUNETR comprehensive summary...")
    
    summary = {
        'task': task,
        'fold': fold,
        'model_type': 'swinunetr',
        'weight_source_type': weight_source_type,
        'weight_mode': weight_mode,
        'best_gating_strategy_used': {
            'name': best_gating_strategy['name'],
            'category': best_gating_strategy['category'],
            'weights': best_gating_strategy['weights'],
            'validation_performance': best_gating_strategy['performance']
        },
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_mode': weight_mode,
            'early_stopping_patience': early_stopping_patience,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        },
        'data_split': {
            'train_samples': len(train_idx),
            'val_samples': len(val_files),
            'test_samples': len(test_files)
        },
        'final_performance': {
            'best_val_auc': checkpoint.get('val_auc', 0.0) if os.path.exists(best_model_path) else 0.0,
            'test_auc': test_results['performance']['auc'],
            'test_accuracy': test_results['performance']['accuracy']
        },
        'gating_analysis': test_results['gating_analysis']
    }
    
    summary_path = os.path.join(task_output_dir, 'swinunetr_gating_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✅ SwinUNETR Gating-based MoE training pipeline completed!")
    print(f"📊 Final Results:")
    print(f"  - Model: SwinUNETR Gating-based MoE with {weight_source_type.upper()}-derived weights")
    print(f"  - Best gating strategy: {best_gating_strategy['name']}")
    print(f"  - Test AUC: {summary['final_performance']['test_auc']:.4f}")
    print(f"  - Test Accuracy: {summary['final_performance']['test_accuracy']:.4f}")
    print(f"\n💾 All results saved to: {task_output_dir}")
    
    return summary


def compare_weight_modes(task, fold, weight_source_type, **kwargs):
    
    weight_modes = ['gating_fixed', 'gating_learnable', 'gating_dynamic']
    results = {}
    
    print(f"🔬 Comparing {weight_source_type.upper()} Gating Weight Modes for task '{task}', fold {fold}")
    
    for weight_mode in weight_modes:
        print(f"\n{'='*60}")
        print(f"Training SwinUNETR with {weight_source_type.upper()}-derived weights, mode: {weight_mode}")
        print(f"{'='*60}")
        
        try:
            summary = run_unified_training_pipeline(
                task=task,
                fold=fold,
                weight_source_type=weight_source_type,
                weight_mode=weight_mode,
                **kwargs
            )
            results[weight_mode] = summary
            
        except Exception as e:
            print(f"❌ Error training SwinUNETR with {weight_source_type.upper()}-derived weights {weight_mode}: {e}")
            import traceback
            traceback.print_exc()
            results[weight_mode] = {'error': str(e)}
    

    comparison = {
        'task': task,
        'fold': fold,
        'weight_source_type': weight_source_type,
        'comparison_results': results,
        'summary': {}
    }
    

    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_mode = max(valid_results.keys(), 
                       key=lambda k: valid_results[k]['final_performance']['test_auc'])
        
        comparison['summary'] = {
            'best_gating_weight_mode': best_mode,
            'best_test_auc': valid_results[best_mode]['final_performance']['test_auc'],
            'performance_ranking': sorted(
                valid_results.items(),
                key=lambda x: x[1]['final_performance']['test_auc'],
                reverse=True
            )
        }
        
        print(f"\n🏆 {weight_source_type.upper()} Gating Weight Mode Comparison Results:")
        print(f"Best mode: {best_mode}")
        print(f"Best test AUC: {valid_results[best_mode]['final_performance']['test_auc']:.4f}")
        
        print(f"\nPerformance ranking:")
        for i, (mode, result) in enumerate(comparison['summary']['performance_ranking'], 1):
            auc = result['final_performance']['test_auc']
            acc = result['final_performance']['test_accuracy']
            print(f"  {i}. {mode}: AUC={auc:.4f}, Acc={acc:.4f}")
    

    comparison_dir = kwargs.get('output_dir')
    if not comparison_dir:
        raise ValueError("output_dir must be provided in kwargs")
    comparison_path = os.path.join(comparison_dir, task, f"fold_{fold}", "swinunetr", f"{weight_source_type}_weights", "weight_mode_comparison.json")
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print(f"\n💾 {weight_source_type.upper()} comparison results saved to: {comparison_path}")
    
    return comparison


def compare_models(task, fold, expert_results_base_dir, **kwargs):
    
    weight_source_types = ['cnn', 'vit', 'mamba', 'radiomics']
    results = {}
    
    print(f"🔬 Comparing All Weight Source Types for task '{task}', fold {fold}")
    
    for weight_source_type in weight_source_types:
        print(f"\n{'='*80}")
        print(f"Training SwinUNETR with {weight_source_type.upper()}-derived weights")
        print(f"{'='*80}")
        
        try:

            formatted_expert_dir = expert_results_base_dir.format(weight_source_type=weight_source_type)
            
            summary = run_unified_training_pipeline(
                task=task,
                fold=fold,
                weight_source_type=weight_source_type,
                expert_results_dir=formatted_expert_dir,
                **kwargs
            )
            results[weight_source_type] = summary
            
        except Exception as e:
            print(f"❌ Error training SwinUNETR with {weight_source_type.upper()}-derived weights: {e}")
            import traceback
            traceback.print_exc()
            results[weight_source_type] = {'error': str(e)}
    

    comparison = {
        'task': task,
        'fold': fold,
        'weight_source_comparison': results,
        'summary': {}
    }
    

    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_weight_source = max(valid_results.keys(), 
                               key=lambda k: valid_results[k]['final_performance']['test_auc'])
        
        comparison['summary'] = {
            'best_weight_source_type': best_weight_source,
            'best_test_auc': valid_results[best_weight_source]['final_performance']['test_auc'],
            'performance_ranking': sorted(
                valid_results.items(),
                key=lambda x: x[1]['final_performance']['test_auc'],
                reverse=True
            )
        }
        
        print(f"\n🏆 Weight Source Comparison Results:")
        print(f"Best weight source: {best_weight_source.upper()}")
        print(f"Best test AUC: {valid_results[best_weight_source]['final_performance']['test_auc']:.4f}")
        
        print(f"\nPerformance ranking:")
        for i, (weight_source, result) in enumerate(comparison['summary']['performance_ranking'], 1):
            auc = result['final_performance']['test_auc']
            acc = result['final_performance']['test_accuracy']
            params = result['training_config']['total_parameters']
            print(f"  {i}. {weight_source.upper()}: AUC={auc:.4f}, Acc={acc:.4f}, Params={params:,}")
    

    comparison_dir = kwargs.get('output_dir')
    if not comparison_dir:
        raise ValueError("output_dir must be provided in kwargs")
    comparison_path = os.path.join(comparison_dir, task, f"fold_{fold}", "weight_source_comparison.json")
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print(f"\n💾 Weight source comparison results saved to: {comparison_path}")
    
    return comparison



def run_all_folds(task, weight_source_type, **kwargs):
    
    results = {}
    
    print(f"🔬 Running SwinUNETR with {weight_source_type.upper()}-derived weights for all folds of task '{task}'")
    
    for fold in range(5):
        print(f"\n{'='*80}")
        print(f"Training SwinUNETR with {weight_source_type.upper()}-derived weights for fold {fold}")
        print(f"{'='*80}")
        
        try:
            summary = run_unified_training_pipeline(
                task=task,
                fold=fold,
                weight_source_type=weight_source_type,
                **kwargs
            )
            results[f'fold_{fold}'] = summary
            
        except Exception as e:
            print(f"❌ Error training SwinUNETR with {weight_source_type.upper()}-derived weights fold {fold}: {e}")
            import traceback
            traceback.print_exc()
            results[f'fold_{fold}'] = {'error': str(e)}
    

    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        aucs = [result['final_performance']['test_auc'] for result in valid_results.values()]
        accs = [result['final_performance']['test_accuracy'] for result in valid_results.values()]
        
        fold_summary = {
            'task': task,
            'model_type': 'swinunetr',
            'weight_source_type': weight_source_type,
            'completed_folds': len(valid_results),
            'total_folds': 5,
            'cross_validation_results': {
                'mean_auc': np.mean(aucs),
                'std_auc': np.std(aucs),
                'mean_accuracy': np.mean(accs),
                'std_accuracy': np.std(accs),
                'individual_results': valid_results
            }
        }
        
        print(f"\n🏆 SwinUNETR with {weight_source_type.upper()}-derived weights Cross-Validation Results for {task}:")
        print(f"Mean AUC: {fold_summary['cross_validation_results']['mean_auc']:.4f} ± {fold_summary['cross_validation_results']['std_auc']:.4f}")
        print(f"Mean Accuracy: {fold_summary['cross_validation_results']['mean_accuracy']:.4f} ± {fold_summary['cross_validation_results']['std_accuracy']:.4f}")
        print(f"Completed folds: {len(valid_results)}/5")
        

        output_dir = kwargs.get('output_dir')
        if not output_dir:
            raise ValueError("output_dir must be provided in kwargs")
        summary_path = os.path.join(output_dir, task, 'swinunetr', f"{weight_source_type}_weights", "cross_validation_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(fold_summary, f, indent=4)
        
        print(f"💾 Cross-validation summary saved to: {summary_path}")
        
        return fold_summary
    else:
        print("❌ No valid results across folds")
        return None


def run_all_models_all_folds(task, expert_results_base_dir, **kwargs):
    
    weight_source_types = ['cnn', 'vit', 'mamba', 'radiomics']
    all_results = {}
    
    print(f"🔬 Running SwinUNETR with ALL WEIGHT SOURCES for ALL FOLDS of task '{task}' - Comprehensive Analysis")
    
    for weight_source in weight_source_types:
        print(f"\n{'='*100}")
        print(f"RUNNING SWINUNETR WITH {weight_source.upper()}-DERIVED WEIGHTS")
        print(f"{'='*100}")
        
        try:

            formatted_expert_dir = expert_results_base_dir.format(weight_source_type=weight_source)
            
            fold_results = run_all_folds(task, weight_source, expert_results_dir=formatted_expert_dir, **kwargs)
            all_results[weight_source] = fold_results
        except Exception as e:
            print(f"❌ Error training SwinUNETR with {weight_source} weights: {e}")
            continue
    
    

    valid_model_results = {k: v for k, v in all_results.items() if 'error' not in v and v is not None}
    
    if valid_model_results:
        comprehensive_summary = {
            'task': task,
            'analysis_type': 'comprehensive_all_models_all_folds',
            'model_results': all_results,
            'summary': {}
        }
        

        model_performances = {}
        for model_type, results in valid_model_results.items():
            if 'cross_validation_results' in results:
                model_performances[model_type] = {
                    'mean_auc': results['cross_validation_results']['mean_auc'],
                    'std_auc': results['cross_validation_results']['std_auc'],
                    'mean_accuracy': results['cross_validation_results']['mean_accuracy'],
                    'std_accuracy': results['cross_validation_results']['std_accuracy'],
                    'completed_folds': results['completed_folds']
                }
        
        if model_performances:
            best_model = max(model_performances.keys(), 
                           key=lambda k: model_performances[k]['mean_auc'])
            
            comprehensive_summary['summary'] = {
                'best_overall_model': best_model,
                'best_mean_auc': model_performances[best_model]['mean_auc'],
                'model_ranking': sorted(
                    model_performances.items(),
                    key=lambda x: x[1]['mean_auc'],
                    reverse=True
                )
            }
            
            print(f"\n🏆 COMPREHENSIVE RESULTS for {task}:")
            print(f"Best overall model: {best_model.upper()}")
            print(f"Best mean AUC: {model_performances[best_model]['mean_auc']:.4f}")
            
            print(f"\nModel ranking by mean AUC:")
            for i, (model, perf) in enumerate(comprehensive_summary['summary']['model_ranking'], 1):
                auc = perf['mean_auc']
                std = perf['std_auc']
                acc = perf['mean_accuracy']
                folds = perf['completed_folds']
                print(f"  {i}. {model.upper()}: AUC={auc:.4f}±{std:.4f}, Acc={acc:.4f}, Folds={folds}/5")
        

        output_dir = kwargs.get('output_dir')
        if not output_dir:
            raise ValueError("output_dir must be provided in kwargs")
        comprehensive_path = os.path.join(output_dir, task, "comprehensive_analysis.json")
        os.makedirs(os.path.dirname(comprehensive_path), exist_ok=True)
        
        with open(comprehensive_path, 'w') as f:
            json.dump(comprehensive_summary, f, indent=4)
        
        print(f"\n💾 Comprehensive analysis saved to: {comprehensive_path}")
        
        return comprehensive_summary
    else:
        print("❌ No valid comprehensive results")
        return None






def main():
    parser = argparse.ArgumentParser(
        description='Unified Gating-based MoE Training Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    

    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., has_ILD)')
    

    parser.add_argument('--mode', type=str, required=True,
                    choices=['all', 'weight_source', 'fold', 'single'],
                    help='Training mode: all (all weight sources, all folds), weight_source (one source, all folds), etc.')    

    parser.add_argument('--model_type', type=str, 
                    choices=['cnn', 'vit', 'mamba', 'radiomics'],
                    help='Weight source type (determines which expert weights to use for SwinUNETR initialization)')
    parser.add_argument('--fold_idx', type=int, choices=range(5),
                       help='Fold index (0-4)')
    

    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--expert_results_dir', type=str, default=None,
                       help='Directory containing expert training results and gating evaluation')
    parser.add_argument('--nifti_dir', type=str, default=None,
                       help='Directory containing NIFTI files')
    parser.add_argument('--lobe_masks_dir', type=str, default=None,
                       help='Directory containing lobe masks')
    parser.add_argument('--labels_path', type=str, default=None,
                       help='Path to labels CSV file')
    parser.add_argument('--weight_mode', type=str, default='gating_fixed',
                       choices=['gating_fixed', 'gating_learnable', 'gating_dynamic'],
                       help='Gating weight mode')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device to use')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Patience for early stopping (default: 10)')
    

    parser.add_argument('--compare_modes', action='store_true',
                       help='Compare all gating weight modes for specified model/fold')
    
    args = parser.parse_args()
    
    import json
    from pathlib import Path
    
    if not Path(args.config).exists():
        parser.error(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    data_paths = config.get('data_paths', {})
    output_config = config.get('output', {})
    
    weight_source_type = args.weight_source_type if hasattr(args, 'weight_source_type') else 'cnn'
    if args.expert_results_dir is None:
        expert_dir_key = f'unified_{weight_source_type}_results'
        args.expert_results_dir = output_config.get(expert_dir_key)
    if args.nifti_dir is None:
        args.nifti_dir = data_paths.get('nifti_dir')
    if args.lobe_masks_dir is None:
        args.lobe_masks_dir = data_paths.get('lobe_masks_dir')
    if args.labels_path is None:
        args.labels_path = data_paths.get('labels_file')
    if args.output_dir is None:
        args.output_dir = output_config.get('unified_gating_moe_output')
    
    if not args.expert_results_dir or not args.nifti_dir or not args.lobe_masks_dir or not args.labels_path or not args.output_dir:
        parser.error("Missing required paths in config. Please check data_paths and output sections.")

    if not args.mode:
        if args.model_type and args.fold_idx is not None:
            args.mode = 'single'
        elif args.model_type:
            args.mode = 'model'
        elif args.fold_idx is not None:
            args.mode = 'fold'
        else:
            args.mode = 'all'
    

    if args.mode in ['model', 'single'] and not args.model_type:
        parser.error("--model_type is required when mode is 'model' or 'single'")
    
    if args.mode in ['fold', 'single'] and args.fold_idx is None:
        parser.error("--fold_idx is required when mode is 'fold' or 'single'")
    
    if args.compare_modes and (not args.model_type or args.fold_idx is None):
        parser.error("--compare_modes requires both --model_type and --fold_idx")
    
    print(f"🚀 Unified Gating-based MoE Training Framework")
    print(f"Task: {args.task}")
    print(f"Mode: {args.mode}")
    if args.model_type:
        print(f"Model: {args.model_type.upper()}")
    if args.fold_idx is not None:
        print(f"Fold: {args.fold_idx}")
    print(f"GPU: {args.gpu}")
    if MAMBA_AVAILABLE:
        print(f"Mamba: Available")
    else:
        print(f"Mamba: Using fallback implementation")
    

    if args.mode != 'all':
        fold_to_check = args.fold_idx if args.fold_idx is not None else 0
        

        if args.model_type:
            formatted_expert_dir = args.expert_results_dir.format(weight_source_type=args.model_type)
            


            if args.model_type == 'cnn':
                gating_file = Path(formatted_expert_dir) / "gating_ensemble_results" / args.task / f"fold_{fold_to_check}" / "best_cnn_gating_strategy.json"
            elif args.model_type == 'vit':
                gating_file = Path(formatted_expert_dir) / "gating_ensemble_results" / args.task / f"fold_{fold_to_check}" / "best_vit_gating_strategy.json"
            elif args.model_type == 'mamba':
                gating_file = Path(formatted_expert_dir) / "gating_ensemble_results" / args.task / f"fold_{fold_to_check}" / "best_mamba_gating_strategy.json"
            elif args.model_type == 'radiomics':
                gating_file = Path(formatted_expert_dir) / "gating_ensemble_results" / args.task / f"fold_{fold_to_check}" / "best_radiomics_strategy.json"
            else:
                gating_file = Path(formatted_expert_dir) / "gating_ensemble_results" / args.task / f"fold_{fold_to_check}" / "best_gating_strategy.json"
            
            if not gating_file.exists():
                print(f"\n❌ ERROR: Best gating strategy not found: {gating_file}")
                print(f"Expected directory structure: {formatted_expert_dir}/gating_ensemble_results/")
                print(f"Please ensure you have run the complete pipeline for {args.model_type.upper()} weight extraction:")
                print(f"1. Extract {args.model_type.upper()} expert weights")
                print(f"2. Generate {args.model_type.upper()}-derived gating strategy")  
                print(f"3. Then run: python unified_gating_moe_training.py --task {args.task} --model_type {args.model_type} --fold_idx {fold_to_check} --mode single")
                sys.exit(1)
        else:

            print(f"⚠️ Skipping prerequisite check for mode '{args.mode}' without specific model_type")
    
    print(f"\n✅ Prerequisites validated")
    


    common_kwargs = {
        'nifti_dir': args.nifti_dir,
        'lobe_masks_dir': args.lobe_masks_dir,
        'labels_file': args.labels_path,
        'weight_mode': args.weight_mode,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'early_stopping_patience': args.early_stopping_patience,
        'output_dir': args.output_dir,
        'gpu': args.gpu

    }
    

    try:
        if args.compare_modes:

            print(f"\n🔬 COMPARING WEIGHT MODES for {args.model_type.upper()}")

            formatted_expert_dir = args.expert_results_dir.format(weight_source_type=args.model_type)
            comparison = compare_weight_modes(
                task=args.task,
                fold=args.fold_idx,
                weight_source_type=args.model_type,
                expert_results_dir=formatted_expert_dir,
                **common_kwargs
            )
            
        elif args.mode == 'all':

            print(f"\n🔬 COMPREHENSIVE ANALYSIS: ALL WEIGHT SOURCES, ALL FOLDS")
            results = run_all_models_all_folds(args.task, expert_results_base_dir=args.expert_results_dir, **common_kwargs)
            
        elif args.mode == 'weight_source':

            print(f"\n🔬 CROSS-VALIDATION: {args.model_type.upper()} WEIGHTS FOR ALL FOLDS")

            formatted_expert_dir = args.expert_results_dir.format(weight_source_type=args.model_type)
            results = run_all_folds(args.task, args.model_type, expert_results_dir=formatted_expert_dir, **common_kwargs)
            
        elif args.mode == 'fold':

            print(f"\n🔬 WEIGHT SOURCE COMPARISON: ALL WEIGHT SOURCES FOR FOLD {args.fold_idx}")
            results = compare_models(args.task, args.fold_idx, expert_results_base_dir=args.expert_results_dir, **common_kwargs)
            
        elif args.mode == 'single':

            print(f"\n🔬 SINGLE RUN: {args.model_type.upper()} WEIGHTS FOR FOLD {args.fold_idx}")

            formatted_expert_dir = args.expert_results_dir.format(weight_source_type=args.model_type)
            results = run_unified_training_pipeline(
                task=args.task,
                fold=args.fold_idx,
                weight_source_type=args.model_type,
                expert_results_dir=formatted_expert_dir,
                **common_kwargs
            )
        
        print(f"\n🎉 Unified Gating-based MoE training completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()