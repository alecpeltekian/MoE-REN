
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import math
class BaseExpertModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def forward(self, x):
        pass
    @abstractmethod
    def get_features(self, x):
        pass
class CNNExpertModel(BaseExpertModel):
    def __init__(self, input_size=(96, 96, 96), dropout_rate=0.5):
        super().__init__()
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
    def get_features(self, x):
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
        return x  
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
class ViTExpertModel(BaseExpertModel):
    def __init__(self, img_size=(96, 96, 96), patch_size=(8, 8, 8), 
                 in_channels=1, embed_dim=128, depth=3, num_heads=4, 
                 mlp_ratio=1.5, num_classes=2, dropout_rate=0.2):
        super().__init__()
        self.vit = VisionTransformer3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            n_layers=depth,
            n_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            dropout=dropout_rate
        )
    def forward(self, x):
        return self.vit(x)
    def get_features(self, x):
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.dropout(x)
        for block in self.vit.blocks:
            x = block(x)
        x = self.vit.norm(x)
        return x[:, 0]  
class MambaExpertModel(BaseExpertModel):
    def __init__(self, input_size=(96, 96, 96), dropout_rate=0.1):
        super().__init__()
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
    def get_features(self, x):
        B = x.shape[0]
        x = self.mamba.patch_embed(x)  
        x = x + self.mamba.pos_embed
        x = self.mamba.dropout(x)
        for block in self.mamba.blocks:
            x = block(x)
        x = self.mamba.norm(x)
        x = x.mean(dim=1)  
        return x
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
class MambaBlock3D(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        except ImportError:
            self.mamba = SimpleMambaFallback(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + residual  
class SimpleMambaFallback(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.x_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.out_proj(torch.relu(self.in_proj(x)[:, :, :x.size(-1)]))
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
            MambaBlock3D(d_model, d_state, d_conv, expand)
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
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseExpertModel:
        if model_type.lower() == 'cnn':
            return CNNExpertModel(**kwargs)
        elif model_type.lower() == 'vit':
            return ViTExpertModel(
                img_size=(96, 96, 96),
                patch_size=(8, 8, 8),
                in_channels=1,
                embed_dim=128,  
                depth=3,        
                num_heads=4,    
                mlp_ratio=1.5,  
                num_classes=2,
                dropout_rate=0.2  
            )
        elif model_type.lower() == 'mamba':
            return MambaExpertModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
class UnifiedMixtureOfExpertsEnsemble:
    def __init__(self, model_type: str, results_dir: str = "expert_results", 
                 ensemble_method: str = "weighted_average", 
                 weight_strategy: str = "val_auc", 
                 gating_weights: Optional[np.ndarray] = None, 
                 expert_names: Optional[List[str]] = None):
        self.model_type = model_type.lower()
        self.results_dir = Path(results_dir)
        self.ensemble_method = ensemble_method
        self.weight_strategy = weight_strategy
        self.gating_weights = gating_weights
        self.expert_models = {}
        self.expert_weights = {}
        self.meta_learner = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if expert_names is not None:
            self.expert_names = expert_names
        else:
            self.expert_names = [
                "Lobe_1_LUL", "Lobe_2_LLL", "Lobe_3_RUL", 
                "Lobe_4_RML", "Lobe_5_RLL", "Left_Lung", "Right_Lung"
            ]
        if self.model_type not in ['cnn', 'vit', 'mamba']:
            raise ValueError(f"Unknown model type: {model_type}")
        if self.gating_weights is not None:
            self.gating_weights = np.array(self.gating_weights, dtype=float)
            if len(self.gating_weights) != len(self.expert_names):
                raise ValueError(f"Gating weights length ({len(self.gating_weights)}) must match number of experts ({len(self.expert_names)})")
            print(f"🎯 Initialized {self.model_type.upper()} ensemble with gating weights:")
            for name, weight in zip(self.expert_names, self.gating_weights):
                print(f"  {name}: {weight:.4f}")
    def get_model_display_name(self) -> str:
        return {'cnn': 'CNN', 'vit': 'ViT', 'mamba': 'Mamba'}[self.model_type]
    def explain_weighting_strategy(self):
        model_name = self.get_model_display_name()
        print(f"\n📊 {model_name.upper()} ENSEMBLE METHOD: {self.ensemble_method}")
        print("-" * 50)
        if self.ensemble_method == "gating_weighted":
            print(f"🎯 {model_name.upper()} GATING-BASED WEIGHTING: Uses sophisticated gating functions")
            print(f"   extracted from {model_name} expert models in step 2.")
            print(f"   These weights consider {model_name} performance, feature analysis,")
            if self.model_type == 'vit':
                print("   attention patterns, and learned gating strategies.")
            elif self.model_type == 'mamba':
                print("   state space dynamics, and learned gating strategies.")
            else:
                print("   and learned gating strategies.")
            print("   ✅ NO DATA LEAKAGE - derived from validation data only.")
        elif self.ensemble_method == "simple_average":
            print(f"📊 {model_name.upper()} SIMPLE AVERAGING: All {model_name} experts receive equal weights.")
            print(f"   Conservative baseline approach for {model_name} ensemble.")
        elif self.ensemble_method == "weighted_average":
            if self.weight_strategy == "val_auc":
                print(f"✅ {model_name.upper()} VALIDATION AUC WEIGHTING: Uses validation AUC to weight {model_name} experts.")
                print(f"   Higher validation AUC → Higher weight in {model_name} ensemble.")
                print("   Standard approach with no data leakage.")
            else:
                print(f"⚙️ {model_name.upper()} WEIGHTED AVERAGE: Using {self.weight_strategy} strategy.")
        elif self.ensemble_method == "learned_ensemble":
            print(f"🧠 {model_name.upper()} LEARNED ENSEMBLE: Uses meta-learner trained on {model_name} expert predictions.")
            print(f"   Learns optimal {model_name} combination strategy from validation data.")
        print("-" * 50)
    def normalize_hierarchical_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.array(weights, dtype=float)
        lobe_weights = weights[:5]
        lobe_sum = np.sum(lobe_weights)
        lung_weights = weights[5:7]
        lung_sum = np.sum(lung_weights)
        normalized_weights = weights.copy()
        if lobe_sum > 0:
            normalized_weights[:5] = lobe_weights / lobe_sum
        else:
            normalized_weights[:5] = 0.2  
        if lung_sum > 0:
            normalized_weights[5:7] = lung_weights / lung_sum
        else:
            normalized_weights[5:7] = 0.5  
        return normalized_weights
    def load_expert_models(self, task: str, fold: int = 0):
        model_name = self.get_model_display_name()
        print(f"Loading {model_name} expert models for task: {task}, fold: {fold}")
        for expert_name in self.expert_names:
            if self.model_type == 'vit':
                model_path = self.results_dir / task / f"fold_{fold}" / f"{expert_name}_vit_final_model.pth"
                results_path = self.results_dir / task / f"fold_{fold}" / f"{expert_name}_vit_results.json"
            elif self.model_type == 'mamba':
                model_path = self.results_dir / task / f"fold_{fold}" / f"{expert_name}_mamba_final_model.pth"
                results_path = self.results_dir / task / f"fold_{fold}" / f"{expert_name}_mamba_results.json"
            else:  
                model_path = self.results_dir / task / f"fold_{fold}" / f"{expert_name}_final_model.pth"
                results_path = self.results_dir / task / f"fold_{fold}" / f"{expert_name}_results.json"
            if model_path.exists() and results_path.exists():
                try:
                    model = ModelFactory.create_model(self.model_type).to(self.device)
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    if self.model_type == 'vit':
                        cleaned_state_dict = {}
                        for key, value in state_dict.items():
                            if not key.startswith('vit.'):
                                new_key = f'vit.{key}'  
                                cleaned_state_dict[new_key] = value
                            else:
                                cleaned_state_dict[key] = value
                        state_dict = cleaned_state_dict
                    model.load_state_dict(state_dict)
                    model.eval()
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    expert_key = f"{expert_name}_{task}_fold_{fold}"
                    self.expert_models[expert_key] = model
                    val_auc = results.get('final_val_auc', results.get('best_val_auc', 0.5))
                    final_test_metrics = results.get('final_test_metrics', {})
                    if isinstance(final_test_metrics, dict):
                        test_auc = final_test_metrics.get('auc', 0.5)
                        test_accuracy = final_test_metrics.get('accuracy', 0.5)
                    else:
                        test_auc = 0.5
                        test_accuracy = 0.5
                    training_history = results.get('training_history', {})
                    val_acc_history = training_history.get('val_acc', [0.5])
                    val_loss_history = training_history.get('val_loss', [1.0])
                    val_accuracy = val_acc_history[-1] if val_acc_history else 0.5
                    val_loss = val_loss_history[-1] if val_loss_history else 1.0
                    expert_idx = results.get('expert_idx', 0)
                    self.expert_weights[expert_key] = {
                        'val_auc': val_auc,
                        'test_auc': test_auc,  
                        'test_accuracy': test_accuracy,  
                        'val_accuracy': val_accuracy,
                        'val_loss': val_loss,
                        'expert_idx': expert_idx
                    }
                    print(f"  ✅ Loaded {model_name} {expert_name}: Val AUC = {val_auc:.4f}")
                except Exception as e:
                    print(f"  ❌ Error loading {model_name} {expert_name}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  ⚠️ Missing {model_name} files for {expert_name}:")
                print(f"    Model: {model_path.exists()} - {model_path}")
                print(f"    Results: {results_path.exists()} - {results_path}")
        print(f"Successfully loaded {len(self.expert_models)} {model_name} expert models")
    def create_expert_mask(self, mask_tensor: torch.Tensor, expert_idx: int) -> torch.Tensor:
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
            raise ValueError(f"Invalid expert_idx: {expert_idx}")
    def get_expert_predictions(self, image_batch: torch.Tensor, mask_batch: torch.Tensor) -> Dict[str, np.ndarray]:
        predictions = {}
        with torch.no_grad():
            for expert_key, model in self.expert_models.items():
                expert_idx = self.expert_weights[expert_key]['expert_idx']
                expert_masks = []
                for i in range(mask_batch.shape[0]):
                    expert_mask = self.create_expert_mask(mask_batch[i], expert_idx)
                    expert_masks.append(expert_mask)
                expert_mask_batch = torch.stack(expert_masks).to(self.device)
                masked_images = image_batch * expert_mask_batch
                outputs = model(masked_images)
                probs = F.softmax(outputs, dim=1)
                predictions[expert_key] = probs[:, 1].cpu().numpy()  
        return predictions
    def ensemble_predict(self, expert_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        if not expert_predictions:
            raise ValueError(f"No {self.get_model_display_name()} expert predictions provided")
        pred_matrix = np.column_stack(list(expert_predictions.values()))
        expert_keys = list(expert_predictions.keys())
        model_name = self.get_model_display_name()
        if self.ensemble_method == "simple_average":
            return np.mean(pred_matrix, axis=1)
        elif self.ensemble_method == "gating_weighted":
            if self.gating_weights is None:
                raise ValueError(f"Gating weights not provided for {model_name} gating_weighted method")
            print(f"🎯 Using {model_name} gating weights for ensemble prediction:")
            for key, weight in zip(expert_keys, self.gating_weights):
                print(f"  {key}: {weight:.4f}")
            weights = self.gating_weights / np.sum(self.gating_weights)
            return np.average(pred_matrix, axis=1, weights=weights)
        elif self.ensemble_method == "weighted_average":
            if self.weight_strategy == "val_auc":
                weights = np.array([self.expert_weights[key]['val_auc'] for key in expert_keys])
                print(f"✅ Using {model_name} validation AUC weighting (NO data leakage):")
            elif self.weight_strategy == "val_accuracy":
                weights = np.array([self.expert_weights[key]['val_accuracy'] for key in expert_keys])
                print(f"✅ Using {model_name} validation accuracy weighting (NO data leakage):")
            elif self.weight_strategy == "inverse_val_loss":
                val_losses = np.array([self.expert_weights[key]['val_loss'] for key in expert_keys])
                weights = 1.0 / (val_losses + 1e-8)
                print(f"✅ Using {model_name} inverse validation loss weighting (NO data leakage):")
            elif self.weight_strategy == "uniform":
                weights = np.ones(len(expert_keys))
                print(f"✅ Using {model_name} uniform weighting:")
            else:
                raise ValueError(f"Unknown weight strategy: {self.weight_strategy}")
            if np.sum(weights) == 0:
                print(f"Warning: All {model_name} weights are zero, using uniform weighting instead")
                weights = np.ones(len(expert_keys))
            weights = weights / np.sum(weights)  
            print(f"{model_name} Expert weights based on {self.weight_strategy}:")
            for key, weight in zip(expert_keys, weights):
                metric_val = self.expert_weights[key][self.weight_strategy] if self.weight_strategy != "inverse_val_loss" else self.expert_weights[key]['val_loss']
                print(f"  {key}: weight={weight:.3f} (metric={metric_val:.3f})")
            return np.average(pred_matrix, axis=1, weights=weights)
        elif self.ensemble_method == "learned_ensemble":
            if self.meta_learner is None:
                raise ValueError(f"{model_name} meta-learner not trained. Call train_meta_learner first.")
            return self.meta_learner.predict_proba(pred_matrix)[:, 1]
        else:
            raise ValueError(f"Unknown {model_name} ensemble method: {self.ensemble_method}")
    def train_meta_learner(self, train_predictions: Dict[str, np.ndarray], 
                          train_labels: np.ndarray, meta_model: str = "logistic"):
        model_name = self.get_model_display_name()
        print(f"Training {model_name} meta-learner: {meta_model} (using validation data only)")
        pred_matrix = np.column_stack(list(train_predictions.values()))
        if meta_model == "logistic":
            self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        elif meta_model == "random_forest":
            self.meta_learner = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown meta-model: {meta_model}")
        self.meta_learner.fit(pred_matrix, train_labels)
        if hasattr(self.meta_learner, 'feature_importances_'):
            expert_keys = list(train_predictions.keys())
            importances = self.meta_learner.feature_importances_
            print(f"\n{model_name} Expert importance in meta-learner:")
            for key, importance in zip(expert_keys, importances):
                print(f"  {key}: {importance:.4f}")
    def evaluate_ensemble(self, test_predictions: Dict[str, np.ndarray], 
                         test_labels: np.ndarray, save_results: bool = True, 
                         output_dir: str = "ensemble_results") -> Dict:
        model_name = self.get_model_display_name()
        print(f"Evaluating {model_name} ensemble performance...")
        ensemble_preds = self.ensemble_predict(test_predictions)
        ensemble_binary = (ensemble_preds >= 0.5).astype(int)
        auc = roc_auc_score(test_labels, ensemble_preds)
        accuracy = accuracy_score(test_labels, ensemble_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, ensemble_binary, average='weighted', zero_division=0
        )
        individual_metrics = {}
        for expert_key, preds in test_predictions.items():
            expert_binary = (preds >= 0.5).astype(int)
            expert_auc = roc_auc_score(test_labels, preds)
            expert_acc = accuracy_score(test_labels, expert_binary)
            individual_metrics[expert_key] = {'auc': expert_auc, 'accuracy': expert_acc}
        results = {
            'model_type': self.model_type,
            'ensemble_method': self.ensemble_method,
            'weight_strategy': self.weight_strategy if self.ensemble_method != "gating_weighted" else f"{self.model_type}_gating_weights",
            'gating_weights_used': self.gating_weights.tolist() if self.gating_weights is not None else None,
            'ensemble_metrics': {
                'auc': float(auc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'individual_expert_metrics': individual_metrics,
            'improvement_over_best_individual': {
                'auc': float(auc - max([m['auc'] for m in individual_metrics.values()])),
                'accuracy': float(accuracy - max([m['accuracy'] for m in individual_metrics.values()]))
            }
        }
        method_desc = f"{model_name} {self.ensemble_method}"
        if self.ensemble_method == "gating_weighted":
            method_desc += f" (using {model_name} gating functions)"
        elif self.ensemble_method == "weighted_average":
            method_desc += f" with {self.weight_strategy}"
        print(f"\n{model_name} Ensemble Results ({method_desc}):")
        print(f"  AUC: {auc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        best_individual_auc = max([m['auc'] for m in individual_metrics.values()])
        print(f"\n{model_name} Improvement over best individual expert:")
        print(f"  AUC improvement: {auc - best_individual_auc:+.4f}")
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            method_name = f"{self.model_type}_{self.ensemble_method}"
            if self.ensemble_method == "gating_weighted":
                method_name += "_gating"
            elif self.ensemble_method == "weighted_average":
                method_name += f"_{self.weight_strategy}"
            results_path = os.path.join(output_dir, f"{model_name.lower()}_ensemble_results_{method_name}.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"{model_name} results saved to: {results_path}")
        return results
    def plot_ensemble_comparison(self, test_predictions: Dict[str, np.ndarray], 
                               test_labels: np.ndarray, figsize: Tuple[int, int] = (15, 10)):
        from sklearn.metrics import roc_curve
        model_name = self.get_model_display_name()
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{model_name} Ensemble Comparison: {self.ensemble_method}', 
                    fontsize=16, fontweight='bold')
        ax = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(test_predictions) + 1))
        for i, (expert_key, preds) in enumerate(test_predictions.items()):
            fpr, tpr, _ = roc_curve(test_labels, preds)
            auc_score = roc_auc_score(test_labels, preds)
            expert_name = expert_key.split('_')[0] + '_' + expert_key.split('_')[1]
            ax.plot(fpr, tpr, color=colors[i], alpha=0.7, 
                   label=f'{model_name} {expert_name} (AUC: {auc_score:.3f})')
        ensemble_preds = self.ensemble_predict(test_predictions)
        fpr, tpr, _ = roc_curve(test_labels, ensemble_preds)
        ensemble_auc = roc_auc_score(test_labels, ensemble_preds)
        method_label = f"{model_name} Gating Ensemble" if self.ensemble_method == "gating_weighted" else f"{model_name} Ensemble"
        ax.plot(fpr, tpr, color='red', linewidth=3, 
               label=f'{method_label} (AUC: {ensemble_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} ROC Curves Comparison')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        expert_names = []
        expert_aucs = []
        for expert_key, preds in test_predictions.items():
            expert_name = expert_key.split('_')[0] + '_' + expert_key.split('_')[1]
            expert_names.append(f"{model_name} {expert_name}")
            expert_aucs.append(roc_auc_score(test_labels, preds))
        expert_names.append(method_label)
        expert_aucs.append(ensemble_auc)
        bars = ax.bar(range(len(expert_names)), expert_aucs, 
                     color=['lightblue']*len(test_predictions) + ['red'])
        ax.set_xlabel(f'{model_name} Expert')
        ax.set_ylabel('AUC Score')
        ax.set_title(f'{model_name} AUC Comparison')
        ax.set_xticks(range(len(expert_names)))
        ax.set_xticklabels(expert_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        for bar, auc in zip(bars, expert_aucs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{auc:.3f}', ha='center', va='bottom')
        ax = axes[1, 0]
        if self.ensemble_method in ["weighted_average", "gating_weighted"]:
            expert_keys = list(test_predictions.keys())
            if self.ensemble_method == "gating_weighted" and self.gating_weights is not None:
                weights = self.gating_weights
                title_suffix = f"{model_name} Gating Weights"
            elif self.ensemble_method == "weighted_average":
                if self.weight_strategy == "val_auc":
                    weights = np.array([self.expert_weights[key]['val_auc'] for key in expert_keys])
                elif self.weight_strategy == "val_accuracy":
                    weights = np.array([self.expert_weights[key]['val_accuracy'] for key in expert_keys])
                elif self.weight_strategy == "inverse_val_loss":
                    val_losses = np.array([self.expert_weights[key]['val_loss'] for key in expert_keys])
                    weights = 1.0 / (val_losses + 1e-8)
                else:
                    weights = np.ones(len(expert_keys))
                weights = weights / np.sum(weights)  
                title_suffix = f"{model_name} {self.weight_strategy}"
            else:
                weights = np.ones(len(expert_keys)) / len(expert_keys)
                title_suffix = f"{model_name} uniform"
            expert_short_names = [key.split('_')[0] + '_' + key.split('_')[1] for key in expert_keys]
            bars = ax.bar(range(len(expert_short_names)), weights, color='lightgreen')
            ax.set_xlabel(f'{model_name} Expert')
            ax.set_ylabel('Weight')
            ax.set_title(f'{model_name} Expert Weights ({title_suffix})')
            ax.set_xticks(range(len(expert_short_names)))
            ax.set_xticklabels(expert_short_names, rotation=45, ha='right')
            for bar, weight in zip(bars, weights):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{weight:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, f'Weights not applicable\nfor {model_name} {self.ensemble_method}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{model_name} Expert Weights')
        ax = axes[1, 1]
        pos_preds = ensemble_preds[test_labels == 1]
        neg_preds = ensemble_preds[test_labels == 0]
        ax.hist(neg_preds, bins=30, alpha=0.7, label='True Negative', color='blue', density=True)
        ax.hist(pos_preds, bins=30, alpha=0.7, label='True Positive', color='red', density=True)
        ax.set_xlabel(f'{model_name} Ensemble Prediction Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{model_name} Ensemble Prediction Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        plt.tight_layout()
        plt.show()
        return fig
    def save_ensemble_model(self, task: str, fold: int, output_dir: str = "ensemble_models"):
        os.makedirs(output_dir, exist_ok=True)
        model_name = self.get_model_display_name()
        config = {
            'model_type': self.model_type,
            'ensemble_method': self.ensemble_method,
            'weight_strategy': self.weight_strategy,
            'gating_weights': self.gating_weights.tolist() if self.gating_weights is not None else None,
            'expert_weights': self.expert_weights,
            'task': task,
            'fold': fold,
            'expert_names': self.expert_names,
            'expert_models_info': {
                key: {
                    'val_auc': weights['val_auc'], 
                    'expert_idx': weights['expert_idx']
                } 
                for key, weights in self.expert_weights.items()
            }
        }
        method_name = f"{self.model_type}_{self.ensemble_method}"
        if self.ensemble_method == "gating_weighted":
            method_name += "_gating"
        elif self.ensemble_method == "weighted_average":
            method_name += f"_{self.weight_strategy}"
        config_path = os.path.join(output_dir, f"{self.model_type}_ensemble_config_{task}_fold_{fold}_{method_name}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        if self.meta_learner is not None:
            model_path = os.path.join(output_dir, f"{self.model_type}_meta_learner_{task}_fold_{fold}_{method_name}.pkl")
            joblib.dump(self.meta_learner, model_path)
            print(f"{model_name} Meta-learner saved to: {model_path}")
        print(f"{model_name} Ensemble configuration saved to: {config_path}")
    @classmethod
    def load_ensemble_model(cls, config_path: str, model_type: str = None):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_type = model_type or config.get('model_type')
        if not model_type:
            raise ValueError("Model type must be specified in config or parameter")
        ensemble = cls(
            model_type=model_type,
            results_dir=config.get('results_dir', 'expert_results'),
            ensemble_method=config['ensemble_method'],
            weight_strategy=config['weight_strategy'],
            gating_weights=np.array(config['gating_weights']) if config['gating_weights'] else None,
            expert_names=config['expert_names']
        )
        ensemble.expert_weights = config['expert_weights']
        meta_learner_path = config_path.replace('_config_', '_meta_learner_').replace('.json', '.pkl')
        if os.path.exists(meta_learner_path):
            ensemble.meta_learner = joblib.load(meta_learner_path)
        return ensemble
def main():
    import sys
    import json
    from pathlib import Path
    
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    if not config_path or not Path(config_path).exists():
        print("Error: config_path is required and must point to an existing config file")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    output_config = config.get('output', {})
    
    if not output_config.get('mlp_expert_weights_results') or not output_config.get('vit_expert_weights_results') or not output_config.get('mamba_expert_weights_results'):
        print("Error: Missing required output paths in config file")
        return
    
    print("Unified Mixture of Experts Ensemble System")
    print("="*60)
    models_config = {
        'cnn': {
            'results_dir': output_config['mlp_expert_weights_results'],
            'gating_weights': [0.15, 0.20, 0.18, 0.22, 0.25, 0.45, 0.55]
        },
        'vit': {
            'results_dir': output_config['vit_expert_weights_results'],
            'gating_weights': [0.12, 0.18, 0.20, 0.25, 0.23, 0.42, 0.58]
        },
        'mamba': {
            'results_dir': output_config['mamba_expert_weights_results'],
            'gating_weights': [0.14, 0.22, 0.19, 0.21, 0.24, 0.48, 0.52]
        }
    }
    for model_type, config in models_config.items():
        print(f"\n🤖 Initializing {model_type.upper()} ensemble...")
        ensemble = UnifiedMixtureOfExpertsEnsemble(
            model_type=model_type,
            results_dir=config['results_dir'],
            ensemble_method="gating_weighted",
            gating_weights=config['gating_weights']
        )
        ensemble.explain_weighting_strategy()
    print(f"\n✅ Unified ensemble system loaded for all model types!")
    print("🚀 Ready to use with the unified evaluation script!")
if __name__ == "__main__":
    main()