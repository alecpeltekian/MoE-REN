

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
SUPPORTED_MODES = ['all', 'expert', 'fold', 'single', 'compare']
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
                 temperature=1.0,
                 dropout_rate=0.1):
        super().__init__()
        

        self.expert_type = expert_type
        self.weight_mode = weight_mode
        self.temperature = temperature
        

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
        

        self.num_experts = 5
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

            expert_weights = np.array([float(w) for w in expert_weights[:5]])
            normalized_weights = expert_weights / np.sum(expert_weights)
            print(f"\nInitialized MoE weights from {expert_type.upper()} expert results (LOBE EXPERTS ONLY):")
            region_names = ['Lobe_1_LUL', 'Lobe_2_LLL', 'Lobe_3_RUL', 'Lobe_4_RML', 'Lobe_5_RLL']
            for name, auc, weight in zip(region_names, expert_weights, normalized_weights):
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
        

        for lobe_id in range(1, 6):
            mask = (lobe_mask == lobe_id).float()
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
        
        for i, (mask, expert_extractor, attention_module) in enumerate(
            zip(region_masks, self.expert_extractors, self.region_attention)
        ):

            masked_input = x * mask
            

            expert_feat = expert_extractor(masked_input)
            expert_feat = expert_feat.view(batch_size, -1)
            

            attention_weight = attention_module(masked_input)
            
            expert_features.append(expert_feat)
            expert_attentions.append(attention_weight)
        

        expert_features = torch.stack(expert_features, dim=1)
        expert_attentions = torch.stack(expert_attentions, dim=1)
        expert_attentions = expert_attentions.squeeze(-1)
        

        if len(current_weights.shape) == 1:

            weighted_expert_weights = current_weights.unsqueeze(0).expand(batch_size, -1)
        else:

            weighted_expert_weights = current_weights
        

        combined_weights = weighted_expert_weights * expert_attentions
        combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)
        

        weighted_expert_features = expert_features * combined_weights.unsqueeze(-1)
        

        aggregated_expert_features = weighted_expert_features.sum(dim=1)
        

        combined_features = torch.cat([pooled_backbone, aggregated_expert_features], dim=1)
        

        fused_features = self.feature_fusion(combined_features)
        output = self.classifier(fused_features)
        

        returns = [output]
        if return_weights:
            returns.append(combined_weights)
        if return_expert_features:
            returns.append({
                'expert_features': expert_features,
                'expert_attentions': expert_attentions,
                'combined_weights': combined_weights,
                'backbone_features': pooled_backbone
            })
        
        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)


class MoELoss(nn.Module):
    
    def __init__(self, base_criterion=None, weight_regularization=0.01, diversity_regularization=0.01):
        super().__init__()
        self.base_criterion = base_criterion or nn.CrossEntropyLoss()
        self.weight_regularization = weight_regularization
        self.diversity_regularization = diversity_regularization
    
    def forward(self, pred, target, expert_weights=None, expert_features=None):
        

        base_loss = self.base_criterion(pred, target)
        total_loss = base_loss
        

        if expert_weights is not None and self.weight_regularization > 0:

            if len(expert_weights.shape) > 1:

                weight_entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=1).mean()
            else:

                weight_entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum()
            

            weight_reg_loss = -self.weight_regularization * weight_entropy
            total_loss += weight_reg_loss
        

        if expert_features is not None and self.diversity_regularization > 0:

            expert_feats = expert_features['expert_features']
            

            expert_feats_norm = F.normalize(expert_feats, dim=2)
            

            similarities = torch.bmm(expert_feats_norm, expert_feats_norm.transpose(1, 2))
            

            mask = torch.eye(similarities.shape[1], device=similarities.device).unsqueeze(0)
            similarities = similarities * (1 - mask)
            

            diversity_loss = self.diversity_regularization * similarities.abs().mean()
            total_loss += diversity_loss
        
        return total_loss


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


class ExpertConfig:
    
    
    @staticmethod
    def get_expert_config(expert_type: str, output_config: Optional[Dict] = None) -> Dict[str, Any]:
        if output_config is None:
            output_config = {}
        configs = {
            'vit': {
                'expert_results_dir': output_config.get('unified_vit_results'),
                'output_dir': output_config.get('swin_moe_vit_experts_output'),
                'description': 'SwinUNETR with MoE weights from ViT expert training'
            },
            'mamba': {
                'expert_results_dir': output_config.get('unified_mamba_results'),
                'output_dir': output_config.get('swin_moe_mamba_experts_output'),
                'description': 'SwinUNETR with MoE weights from Mamba expert training'
            },
            'cnn': {
                'expert_results_dir': output_config.get('unified_cnn_results'),
                'output_dir': output_config.get('swin_moe_cnn_experts_output'),
                'description': 'SwinUNETR with MoE weights from CNN expert training'
            }
        }
        
        if expert_type not in configs:
            raise ValueError(f"Unsupported expert type: {expert_type}. Supported: {list(configs.keys())}")
        
        return configs[expert_type]


def load_ensemble_weights(results_dir: str, task: str, fold: int, model: str) -> List[float]:
    

    ensemble_path = Path(results_dir) / "ensemble_results" / task / f"fold_{fold}" / f"{model}_ensemble_results_weighted_average_val_auc.json"
    if ensemble_path.exists():
        print(f"Loading ensemble weights from: {ensemble_path}")
        with open(ensemble_path, 'r') as f:
            ensemble_data = json.load(f)
        

        individual_metrics = ensemble_data.get('individual_expert_metrics', {})
        expert_aucs = []
        expert_names = []
        

        expected_order = ['Lobe_1', 'Lobe_2', 'Lobe_3', 'Lobe_4', 'Lobe_5']
        
        for expert_prefix in expected_order:
            found = False
            for expert_key, metrics in individual_metrics.items():
                if expert_key.startswith(expert_prefix):
                    expert_aucs.append(metrics['auc'])
                    expert_names.append(expert_key)
                    found = True
                    break
            
            if not found:
                print(f"Warning: Could not find expert for {expert_prefix}, using default AUC=0.6")
                expert_aucs.append(0.6)
                expert_names.append(f"{expert_prefix}_missing")
        
        print(f"Loaded {len(expert_aucs)} LOBE expert AUC scores:")
        for name, auc in zip(expert_names, expert_aucs):
            print(f"  {name}: {auc:.4f}")
        
        return expert_aucs
    
    else:

        print(f"Ensemble results not found, loading from individual expert results...")
        expert_aucs = []

        expert_names = ['Lobe_1_LUL', 'Lobe_2_LLL', 'Lobe_3_RUL', 'Lobe_4_RML', 'Lobe_5_RLL']
        
        results_path = Path(results_dir) / task / f"fold_{fold}"
        
        for expert_name in expert_names:
            expert_result_path = results_path / f"{expert_name}_results.json"
            
            if expert_result_path.exists():
                with open(expert_result_path, 'r') as f:
                    expert_data = json.load(f)
                

                test_auc = None
                if 'final_test_metrics' in expert_data:
                    test_auc = expert_data['final_test_metrics'].get('auc')
                elif 'best_val_auc' in expert_data:
                    test_auc = expert_data['best_val_auc']
                elif 'test_auc' in expert_data:
                    test_auc = expert_data['test_auc']
                
                if test_auc is not None:
                    expert_aucs.append(float(test_auc))
                else:
                    print(f"Warning: Could not find AUC for {expert_name}, using default=0.6")
                    expert_aucs.append(0.6)
            else:
                print(f"Warning: Results file not found for {expert_name}, using default AUC=0.6")
                expert_aucs.append(0.6)
        
        print(f"Loaded AUC scores from individual LOBE expert results:")
        for name, auc in zip(expert_names, expert_aucs):
            print(f"  {name}: {auc:.4f}")
        
        return expert_aucs


def create_moe_swin_model(expert_weights: Optional[List[float]] = None, 
                         expert_type: str = 'cnn',
                         weight_mode: str = 'learned_moe',
                         model_config: Optional[Dict] = None) -> MoEWeightedSwinUNETR:
    
    default_config = {
        'img_size': (96, 96, 96),
        'in_channels': 1,
        'out_channels': 2,
        'feature_size': 48,
        'use_checkpoint': True,
        'dropout_rate': 0.1
    }
    
    if model_config:
        default_config.update(model_config)
    
    print(f"\nCreating MoE-weighted SwinUNETR model:")
    print(f"  Expert type: {expert_type.upper()}")
    print(f"  Weight mode: {weight_mode}")
    print(f"  Feature size: {default_config['feature_size']}")
    print(f"  Dropout rate: {default_config['dropout_rate']}")
    
    model = MoEWeightedSwinUNETR(
        expert_weights=expert_weights,
        expert_type=expert_type,
        weight_mode=weight_mode,
        **default_config
    )
    
    return model


def train_moe_swin_model(model: MoEWeightedSwinUNETR, 
                        train_loader: DataLoader, 
                        val_loader: DataLoader, 
                        device: torch.device,
                        num_epochs: int = 50, 
                        learning_rate: float = 1e-4,
                        weight_decay: float = 1e-5, 
                        save_dir: str = 'moe_checkpoints', 
                        early_stopping_patience: int = 10) -> Dict:
    
    os.makedirs(save_dir, exist_ok=True)
    

    backbone_params = []
    moe_params = []
    
    for name, param in model.named_parameters():
        if 'swin_backbone' in name:
            backbone_params.append(param)
        else:
            moe_params.append(param)
    

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.1},
        {'params': moe_params, 'lr': learning_rate}
    ], weight_decay=weight_decay)
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    

    criterion = MoELoss(
        base_criterion=nn.CrossEntropyLoss(),
        weight_regularization=0.01,
        diversity_regularization=0.01
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
        'expert_weights_history': [],
        'early_stopped': False,
        'stopped_epoch': None
    }
    
    best_val_auc = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        epoch_weights = []
        
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
                    epoch_weights.append(weights.mean(dim=0).detach().cpu().numpy())
                else:
                    epoch_weights.append(weights.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        

        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_probs = []
        val_weights = []
        
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
                        val_weights.append(weights.mean(dim=0).cpu().numpy())
                    else:
                        val_weights.append(weights.cpu().numpy())
        

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
        
        if early_stopping(val_auc, model, epoch + 1):
            history['early_stopped'] = True
            history['stopped_epoch'] = epoch + 1
            print(f"\n🛑 Training stopped early at epoch {epoch + 1}")
            break


        scheduler.step()
        

        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        

        if epoch_weights:
            avg_epoch_weights = np.mean(epoch_weights, axis=0)
            history['expert_weights_history'].append(avg_epoch_weights.tolist())
        

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        

        if val_weights:
            avg_val_weights = np.mean(val_weights, axis=0)
            region_names = ['Lobe_1', 'Lobe_2', 'Lobe_3', 'Lobe_4', 'Lobe_5', 'Left_Lung', 'Right_Lung']
            print("Current expert weights:", {name: f"{w:.4f}" for name, w in zip(region_names, avg_val_weights)})
        

        if early_stopping.restore_best_weights and early_stopping.best_weights is not None:
            best_val_auc = early_stopping.best_score
            best_epoch = early_stopping.best_epoch
            

            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'val_acc': history['val_acc'][best_epoch - 1] if best_epoch <= len(history['val_acc']) else val_acc,
                'history': history,
                'model_config': {
                    'expert_type': model.expert_type,
                    'weight_mode': model.weight_mode,
                    'num_experts': model.num_experts,
                    'expert_weights': model.base_weights.cpu().numpy().tolist()
                },
                'early_stopped': True,
                'stopped_epoch': epoch + 1
            }
            
            torch.save(checkpoint, os.path.join(save_dir, 'best_moe_swin_model.pth'))
            print(f"✅ Saved best model from epoch {best_epoch} (AUC: {best_val_auc:.4f})")

        

        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
        'final_performance': {'val_auc': val_auc, 'val_acc': val_acc}
    }, os.path.join(save_dir, 'final_moe_swin_model.pth'))
    
    return history


def evaluate_moe_swin_model(model: MoEWeightedSwinUNETR, 
                           test_loader: DataLoader, 
                           device: torch.device, 
                           save_results: bool = True, 
                           output_dir: str = '.') -> Dict:
    
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_weights = []
    all_expert_features = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating MoE SwinUNETR Model'):
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
                    all_weights.extend(weights.cpu().numpy())
                else:
                    all_weights.extend([weights.cpu().numpy()] * len(label))
            
            if expert_info is not None:
                all_expert_features.append({
                    'expert_features': expert_info['expert_features'].cpu().numpy(),
                    'expert_attentions': expert_info['expert_attentions'].cpu().numpy(),
                    'combined_weights': expert_info['combined_weights'].cpu().numpy()
                })
    

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_weights = np.array(all_weights)
    
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    

    cm = confusion_matrix(all_labels, all_preds)
    

    avg_weights = np.mean(all_weights, axis=0) if len(all_weights) > 0 else None
    weight_std = np.std(all_weights, axis=0) if len(all_weights) > 0 else None
    

    pos_weights = np.mean(all_weights[all_labels == 1], axis=0) if np.sum(all_labels == 1) > 0 else None
    neg_weights = np.mean(all_weights[all_labels == 0], axis=0) if np.sum(all_labels == 0) > 0 else None
    
    results = {
        'performance': {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'confusion_matrix': cm.tolist()
        },
        'expert_analysis': {
            'avg_weights': avg_weights.tolist() if avg_weights is not None else None,
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
    

    print(f"\n📊 MoE SwinUNETR Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    if avg_weights is not None:
        region_names = ['Lobe_1', 'Lobe_2', 'Lobe_3', 'Lobe_4', 'Lobe_5', 'Left_Lung', 'Right_Lung']
        print(f"\n🎯 Expert Weight Analysis:")
        print(f"Average weights across all samples:")
        for name, weight, std in zip(region_names, avg_weights, weight_std):
            print(f"  {name}: {weight:.4f} ± {std:.4f}")
        
        if pos_weights is not None and neg_weights is not None:
            print(f"\nClass-specific expert usage:")
            print(f"Positive class (ILD=1):")
            for name, weight in zip(region_names, pos_weights):
                print(f"  {name}: {weight:.4f}")
            print(f"Negative class (ILD=0):")
            for name, weight in zip(region_names, neg_weights):
                print(f"  {name}: {weight:.4f}")
    
    if save_results:
        results_path = os.path.join(output_dir, 'moe_swin_test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n💾 Results saved to: {results_path}")
    
    return results


def run_moe_swin_training_pipeline(task: str, 
                                  fold: int, 
                                  expert_type: str = 'cnn',
                                  expert_results_dir: Optional[str] = None,
                                  nifti_dir: Optional[str] = None,
                                  lobe_masks_dir: Optional[str] = None, 
                                  labels_file: Optional[str] = None,
                                  output_config: Optional[Dict] = None,
                                  weight_mode: str = 'learned_moe',
                                  num_epochs: int = 50,
                                  batch_size: int = 4,
                                  learning_rate: float = 1e-4,
                                  early_stopping_patience: int = 10,
                                  output_dir: Optional[str] = None,
                                  gpu: str = '0') -> Dict:
    
    

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting MoE SwinUNETR training pipeline on device: {device}")
    
    if output_config is None:
        raise ValueError("output_config is required")
    expert_config = ExpertConfig.get_expert_config(expert_type, output_config)
    if not expert_config.get('expert_results_dir') or not expert_config.get('output_dir'):
        raise ValueError(f"Missing required paths in output_config for expert_type {expert_type}")
    

    if expert_results_dir is None:
        expert_results_dir = expert_config['expert_results_dir']
    if output_dir is None:
        output_dir = expert_config['output_dir']
    

    task_output_dir = os.path.join(output_dir, task, f"fold_{fold}", weight_mode)
    os.makedirs(task_output_dir, exist_ok=True)
    
    print(f"\n📁 Output directory: {task_output_dir}")
    print(f"🧠 Expert type: {expert_type.upper()}")
    print(f"📝 Description: {expert_config['description']}")
    

    print(f"\n🔍 Step 1: Loading expert weights from {expert_type.upper()} ensemble results...")
    expert_weights = load_ensemble_weights(expert_results_dir, task, fold, model=expert_type)

    

    print(f"\n📊 Step 2: Loading data for task '{task}', fold {fold}...")
    

    file_list, mask_list, labels, patient_ids = load_image_and_mask_data(
        labels_file, nifti_dir, lobe_masks_dir, task
    )
    

    k_folds = 5
    consistent_folds = create_consistent_folds(file_list, labels, patient_ids, k_folds=k_folds, seed=42)
    fold_data = consistent_folds[fold]
    

    print(f"Available keys in fold_data: {list(fold_data.keys())}")
    

    if 'train_idx' in fold_data:
        train_idx = fold_data['train_idx']
        val_idx = fold_data['val_idx'] 
        test_idx = fold_data['test_idx']
    elif 'holdout_idx' in fold_data:

        holdout_idx = fold_data['holdout_idx']
        val_mask = fold_data['val_mask']
        test_mask = fold_data['test_mask']
        

        all_indices = np.arange(len(file_list))
        train_idx = np.setdiff1d(all_indices, holdout_idx)
        

        val_idx = holdout_idx[val_mask]
        test_idx = holdout_idx[test_mask]
    else:
        raise ValueError(f"Unexpected fold_data structure. Available keys: {list(fold_data.keys())}")
    
    print(f"Data split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    

    train_data = prepare_data_dicts(
        [file_list[i] for i in train_idx],
        [mask_list[i] for i in train_idx], 
        [labels[i] for i in train_idx]
    )
    
    val_data = prepare_data_dicts(
        [file_list[i] for i in val_idx],
        [mask_list[i] for i in val_idx],
        [labels[i] for i in val_idx]
    )
    
    test_data = prepare_data_dicts(
        [file_list[i] for i in test_idx],
        [mask_list[i] for i in test_idx],
        [labels[i] for i in test_idx]
    )
    

    transforms = get_transforms()
    

    train_dataset = Dataset(data=train_data, transform=transforms)
    val_dataset = Dataset(data=val_data, transform=transforms)
    test_dataset = Dataset(data=test_data, transform=transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    

    print(f"\n🏗️  Step 3: Creating MoE-weighted SwinUNETR model...")
    model = create_moe_swin_model(
        expert_weights=expert_weights,
        expert_type=expert_type,
        weight_mode=weight_mode,
        model_config={
            'feature_size': 48,
            'dropout_rate': 0.1
        }
    )
    
    model = model.to(device)
    

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    

    print(f"\n🎯 Step 4: Starting MoE SwinUNETR model training...")
    print(f"Training configuration:")
    print(f"  - Expert type: {expert_type.upper()}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Early stopping patience: {early_stopping_patience}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight mode: {weight_mode}")
    
    history = train_moe_swin_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_dir=task_output_dir,
        early_stopping_patience=early_stopping_patience
    )

    if history.get('early_stopped', False):
        print(f"\n📊 Training stopped early at epoch {history['stopped_epoch']}")


    with open(os.path.join(task_output_dir, 'swin_moe_training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    

    print(f"\n📈 Step 5: Loading best model and evaluating...")
    best_model_path = os.path.join(task_output_dir, 'best_moe_swin_model.pth')
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with validation AUC: {checkpoint['val_auc']:.4f}")
    else:
        print("Warning: Best model not found, using current model state")
    

    test_results = evaluate_moe_swin_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_results=True,
        output_dir=task_output_dir
    )
    

    print(f"\n📋 Step 6: Creating summary report...")
    
    summary = {
        'task': task,
        'fold': fold,
        'expert_type': expert_type,
        'weight_mode': weight_mode,
        'model_type': 'MoEWeightedSwinUNETR',
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        },
        'expert_weights_used': expert_weights,
        'data_split': {
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'test_samples': len(test_idx)
        },
        'final_performance': {
            'best_val_auc': checkpoint.get('val_auc', 0.0) if os.path.exists(best_model_path) else 0.0,
            'test_auc': test_results['performance']['auc'],
            'test_accuracy': test_results['performance']['accuracy']
        },
        'expert_analysis': test_results['expert_analysis']
    }
    

    summary_path = os.path.join(task_output_dir, 'swin_moe_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✅ MoE SwinUNETR training pipeline completed!")
    print(f"📊 Final Results:")
    print(f"  - Model: MoE-weighted SwinUNETR with {expert_type.upper()} expert weights")
    print(f"  - Best validation AUC: {summary['final_performance']['best_val_auc']:.4f}")
    print(f"  - Test AUC: {summary['final_performance']['test_auc']:.4f}")
    print(f"  - Test Accuracy: {summary['final_performance']['test_accuracy']:.4f}")
    print(f"\n💾 All results saved to: {task_output_dir}")
    
    return summary


def compare_weight_modes_swin(task: str, 
                             fold: int, 
                             expert_type: str = 'cnn', 
                             **kwargs) -> Dict:
    
    weight_modes = ['fixed', 'learned_moe', 'dynamic_moe']
    results = {}
    
    print(f"🔬 Comparing MoE SwinUNETR weight modes for task '{task}', fold {fold}, expert type {expert_type.upper()}")
    
    for weight_mode in weight_modes:
        print(f"\n{'='*60}")
        print(f"Training with weight mode: {weight_mode}")
        print(f"{'='*60}")
        
        try:
            summary = run_moe_swin_training_pipeline(
                task=task,
                fold=fold,
                expert_type=expert_type,
                weight_mode=weight_mode,
                **kwargs
            )
            results[weight_mode] = summary
            
        except Exception as e:
            print(f"❌ Error training {weight_mode}: {e}")
            traceback.print_exc()
            results[weight_mode] = {'error': str(e)}
    

    comparison = {
        'task': task,
        'fold': fold,
        'expert_type': expert_type,
        'model_type': 'MoEWeightedSwinUNETR',
        'comparison_results': results,
        'summary': {}
    }
    

    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_mode = max(valid_results.keys(), 
                       key=lambda k: valid_results[k]['final_performance']['test_auc'])
        
        comparison['summary'] = {
            'best_weight_mode': best_mode,
            'best_test_auc': valid_results[best_mode]['final_performance']['test_auc'],
            'performance_ranking': sorted(
                valid_results.items(),
                key=lambda x: x[1]['final_performance']['test_auc'],
                reverse=True
            )
        }
        
        print(f"\n🏆 SwinUNETR Comparison Results:")
        print(f"Best weight mode: {best_mode}")
        print(f"Best test AUC: {valid_results[best_mode]['final_performance']['test_auc']:.4f}")
        
        print(f"\nPerformance ranking:")
        for i, (mode, result) in enumerate(comparison['summary']['performance_ranking'], 1):
            auc = result['final_performance']['test_auc']
            acc = result['final_performance']['test_accuracy']
            print(f"  {i}. {mode}: AUC={auc:.4f}, Acc={acc:.4f}")
    

    output_config = kwargs.get('output_config', {})
    expert_config = ExpertConfig.get_expert_config(expert_type, output_config)
    comparison_dir = kwargs.get('output_dir', expert_config['output_dir'])
    comparison_path = os.path.join(comparison_dir, task, f"fold_{fold}", "swin_moe_weight_mode_comparison.json")
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print(f"\n💾 Comparison results saved to: {comparison_path}")
    
    return comparison


class UnifiedMoETrainer:
    
    
    def __init__(self, base_args: Dict):
        self.base_args = base_args
        self.results_summary = {}
        

        output_config = base_args.get('output_config', {})
        self.base_output_dir = base_args.get('base_output_dir') or output_config.get('unified_swin_moe_results')
        if not self.base_output_dir:
            raise ValueError("base_output_dir must be provided in base_args or output_config")
        os.makedirs(self.base_output_dir, exist_ok=True)
        

        self.setup_logging()
    
    def setup_logging(self):
        
        import logging
        
        log_file = os.path.join(self.base_output_dir, f"unified_swin_moe_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Unified SwinUNETR MoE Training Started - Log file: {log_file}")
    
    def train_single_model(self, expert_type: str, fold_idx: int, weight_mode: str = 'learned_moe') -> Optional[Dict]:
        
        self.logger.info(f"🚀 Training SwinUNETR with {expert_type.upper()} experts - Fold {fold_idx} - Weight mode: {weight_mode}")
        
        try:
            expert_config = ExpertConfig.get_expert_config(expert_type, self.base_args.get('output_config', {}))
            
            train_args = {
                'task': self.base_args['task'],
                'fold': fold_idx,
                'expert_type': expert_type,
                'weight_mode': weight_mode,
                'expert_results_dir': self.base_args.get('expert_results_dir', expert_config['expert_results_dir']),
                'nifti_dir': self.base_args.get('nifti_dir'),
                'lobe_masks_dir': self.base_args.get('lobe_masks_dir'),
                'labels_file': self.base_args.get('labels_file'),
                'output_config': self.base_args.get('output_config', {}),
                'num_epochs': self.base_args.get('num_epochs', 50),
                'batch_size': self.base_args.get('batch_size', 4),
                'learning_rate': self.base_args.get('learning_rate', 1e-4),
                'early_stopping_patience': self.base_args.get('early_stopping_patience', 10),
                'output_dir': self.base_args.get('output_dir', expert_config['output_dir']),
                'gpu': self.base_args.get('gpu', '0')
            }
            

            summary = run_moe_swin_training_pipeline(**train_args)
            

            result_key = f"swin_{expert_type}_fold_{fold_idx}_{weight_mode}"
            self.results_summary[result_key] = {
                'expert_type': expert_type,
                'fold': fold_idx,
                'weight_mode': weight_mode,
                'status': 'success',
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Successfully trained SwinUNETR with {expert_type.upper()} experts - Fold {fold_idx}")
            self.logger.info(f"   Test AUC: {summary['final_performance']['test_auc']:.4f}")
            
            return summary
            
        except Exception as e:
            error_msg = f"❌ Error training SwinUNETR with {expert_type} experts - Fold {fold_idx}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            

            result_key = f"swin_{expert_type}_fold_{fold_idx}_{weight_mode}"
            self.results_summary[result_key] = {
                'expert_type': expert_type,
                'fold': fold_idx,
                'weight_mode': weight_mode,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            return None
    
    def compare_weight_modes_single(self, expert_type: str, fold_idx: int) -> Optional[Dict]:
        
        self.logger.info(f"🔬 Comparing weight modes for SwinUNETR with {expert_type.upper()} experts - Fold {fold_idx}")
        
        try:

            compare_args = {
                'task': self.base_args['task'],
                'fold': fold_idx,
                'expert_type': expert_type,
                'num_epochs': self.base_args.get('num_epochs', 30),
                'batch_size': self.base_args.get('batch_size', 4),
                'learning_rate': self.base_args.get('learning_rate', 1e-4),
                'early_stopping_patience': self.base_args.get('early_stopping_patience', 10),
                'gpu': self.base_args.get('gpu', '0')
            }
            

            for key in ['expert_results_dir', 'nifti_dir', 'lobe_masks_dir', 'labels_file', 'output_dir']:
                if key in self.base_args:
                    compare_args[key] = self.base_args[key]
            

            comparison = compare_weight_modes_swin(**compare_args)
            

            result_key = f"swin_{expert_type}_fold_{fold_idx}_comparison"
            self.results_summary[result_key] = {
                'expert_type': expert_type,
                'fold': fold_idx,
                'mode': 'weight_comparison',
                'status': 'success',
                'comparison': comparison,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Successfully compared weight modes for SwinUNETR with {expert_type.upper()} experts - Fold {fold_idx}")
            
            return comparison
            
        except Exception as e:
            error_msg = f"❌ Error comparing weight modes for SwinUNETR with {expert_type} experts - Fold {fold_idx}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            

            result_key = f"swin_{expert_type}_fold_{fold_idx}_comparison"
            self.results_summary[result_key] = {
                'expert_type': expert_type,
                'fold': fold_idx,
                'mode': 'weight_comparison',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            return None
    
    def run_all_experts_all_folds(self):
        
        self.logger.info("🎯 Running SwinUNETR with ALL expert types for ALL folds")
        
        total_runs = len(SUPPORTED_EXPERT_TYPES) * 5
        completed_runs = 0
        
        for expert_type in SUPPORTED_EXPERT_TYPES:
            for fold_idx in range(5):
                self.logger.info(f"Progress: {completed_runs + 1}/{total_runs}")
                self.train_single_model(expert_type, fold_idx)
                completed_runs += 1
        
        self.logger.info("✅ Completed training SwinUNETR with all expert types for all folds")
    
    def run_expert_all_folds(self, expert_type: str):
        
        self.logger.info(f"🎯 Running SwinUNETR with {expert_type.upper()} experts for ALL folds")
        
        for fold_idx in range(5):
            self.logger.info(f"Progress: Fold {fold_idx + 1}/5")
            self.train_single_model(expert_type, fold_idx)
        
        self.logger.info(f"✅ Completed training SwinUNETR with {expert_type.upper()} experts for all folds")
    
    def run_all_experts_single_fold(self, fold_idx: int):
        
        self.logger.info(f"🎯 Running SwinUNETR with ALL expert types for Fold {fold_idx}")
        
        for i, expert_type in enumerate(SUPPORTED_EXPERT_TYPES):
            self.logger.info(f"Progress: Expert type {i + 1}/{len(SUPPORTED_EXPERT_TYPES)}")
            self.train_single_model(expert_type, fold_idx)
        
        self.logger.info(f"✅ Completed training SwinUNETR with all expert types for Fold {fold_idx}")
    
    def save_final_summary(self) -> str:
        
        summary_file = os.path.join(self.base_output_dir, f"swin_moe_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        

        final_summary = {
            'experiment_info': {
                'task': self.base_args['task'],
                'start_time': datetime.now().isoformat(),
                'total_runs': len(self.results_summary),
                'base_args': self.base_args,
                'model_type': 'MoEWeightedSwinUNETR'
            },
            'results': self.results_summary,
            'performance_analysis': self.analyze_performance()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=4)
        
        self.logger.info(f"💾 Final summary saved to: {summary_file}")
        

        self.print_summary_statistics()
        
        return summary_file
    
    def analyze_performance(self) -> Dict:
        
        analysis = {
            'by_expert_type': {},
            'by_fold': {},
            'by_weight_mode': {},
            'best_overall': None,
            'summary_stats': {}
        }
        
        successful_runs = {k: v for k, v in self.results_summary.items() 
                          if v['status'] == 'success' and 'summary' in v}
        
        if not successful_runs:
            return analysis
        

        for expert_type in SUPPORTED_EXPERT_TYPES:
            expert_runs = {k: v for k, v in successful_runs.items() 
                          if v['expert_type'] == expert_type}
            
            if expert_runs:
                aucs = [v['summary']['final_performance']['test_auc'] for v in expert_runs.values()]
                analysis['by_expert_type'][expert_type] = {
                    'mean_auc': float(np.mean(aucs)) if aucs else 0,
                    'std_auc': float(np.std(aucs)) if aucs else 0,
                    'max_auc': float(max(aucs)) if aucs else 0,
                    'num_runs': len(aucs)
                }
        

        for fold_idx in range(5):
            fold_runs = {k: v for k, v in successful_runs.items() 
                        if v['fold'] == fold_idx}
            
            if fold_runs:
                aucs = [v['summary']['final_performance']['test_auc'] for v in fold_runs.values()]
                analysis['by_fold'][f'fold_{fold_idx}'] = {
                    'mean_auc': float(np.mean(aucs)) if aucs else 0,
                    'std_auc': float(np.std(aucs)) if aucs else 0,
                    'max_auc': float(max(aucs)) if aucs else 0,
                    'num_runs': len(aucs)
                }
        

        for weight_mode in WEIGHT_MODES:
            weight_runs = {k: v for k, v in successful_runs.items() 
                          if v.get('weight_mode') == weight_mode}
            
            if weight_runs:
                aucs = [v['summary']['final_performance']['test_auc'] for v in weight_runs.values()]
                analysis['by_weight_mode'][weight_mode] = {
                    'mean_auc': float(np.mean(aucs)) if aucs else 0,
                    'std_auc': float(np.std(aucs)) if aucs else 0,
                    'max_auc': float(max(aucs)) if aucs else 0,
                    'num_runs': len(aucs)
                }
        

        if successful_runs:
            best_run_key = max(successful_runs.keys(), 
                              key=lambda k: successful_runs[k]['summary']['final_performance']['test_auc'])
            analysis['best_overall'] = {
                'run_key': best_run_key,
                'details': successful_runs[best_run_key]
            }
        
        return analysis
    
    def print_summary_statistics(self):
        
        print("\n" + "="*80)
        print("🏆 UNIFIED SWINUNETR MOE TRAINING SUMMARY")
        print("="*80)
        
        total_runs = len(self.results_summary)
        successful_runs = sum(1 for v in self.results_summary.values() if v['status'] == 'success')
        failed_runs = total_runs - successful_runs
        
        print(f"Total runs: {total_runs}")
        print(f"Successful: {successful_runs}")
        print(f"Failed: {failed_runs}")
        print(f"Success rate: {successful_runs/total_runs*100:.1f}%" if total_runs > 0 else "N/A")
        

        print(f"\n📊 Performance by Expert Type:")
        analysis = self.analyze_performance()
        
        for expert_type, stats in analysis.get('by_expert_type', {}).items():
            print(f"  {expert_type.upper()} experts: Mean AUC = {stats['mean_auc']:.4f} ± {stats['std_auc']:.4f} "
                  f"(Max: {stats['max_auc']:.4f}, Runs: {stats['num_runs']})")
        

        if analysis.get('by_weight_mode'):
            print(f"\n🎯 Performance by Weight Mode:")
            for weight_mode, stats in analysis['by_weight_mode'].items():
                print(f"  {weight_mode}: Mean AUC = {stats['mean_auc']:.4f} ± {stats['std_auc']:.4f} "
                      f"(Max: {stats['max_auc']:.4f}, Runs: {stats['num_runs']})")
        

        if analysis.get('best_overall'):
            best = analysis['best_overall']['details']
            print(f"\n🥇 Best Overall Performance:")
            print(f"  Expert Type: {best['expert_type'].upper()}")
            print(f"  Fold: {best['fold']}")
            print(f"  Weight Mode: {best['weight_mode']}")
            print(f"  Test AUC: {best['summary']['final_performance']['test_auc']:.4f}")
            print(f"  Test Accuracy: {best['summary']['final_performance']['test_accuracy']:.4f}")
        
        print("="*80)


def main():
    
    parser = argparse.ArgumentParser(
        description='Unified MoE SwinUNETR Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    

    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., has_ILD)')
    parser.add_argument('--mode', type=str, required=True, choices=SUPPORTED_MODES,
                       help='Training mode')
    

    parser.add_argument('--expert_type', type=str, choices=SUPPORTED_EXPERT_TYPES,
                       help='Expert type (required for expert, single, compare modes)')
    parser.add_argument('--fold_idx', type=int, choices=range(5),
                       help='Fold index 0-4 (required for fold, single, compare modes)')
    parser.add_argument('--weight_mode', type=str, default='learned_moe', choices=WEIGHT_MODES,
                       help='Weight mode for single training (default: learned_moe)')
    

    parser.add_argument('--expert_results_dir', type=str,
                       help='Directory containing expert training results (expert-specific defaults used if not provided)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--nifti_dir', type=str, default=None,
                       help='Directory containing NIFTI files')
    parser.add_argument('--lobe_masks_dir', type=str, default=None,
                       help='Directory containing lobe masks')
    parser.add_argument('--labels_file', type=str, default=None,
                       help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory (expert-specific defaults used if not provided)')
    parser.add_argument('--base_output_dir', type=str, default=None,
                       help='Base output directory for unified results')
    

    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device to use')
    

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
    if args.labels_file is None:
        args.labels_file = data_paths.get('labels_file')
    if args.base_output_dir is None:
        args.base_output_dir = output_config.get('unified_swin_moe_results')
    
    if not args.nifti_dir or not args.lobe_masks_dir or not args.labels_file or not args.base_output_dir:
        parser.error("Missing required paths in config. Please check data_paths and output sections.")

    if args.mode in ['expert', 'single', 'compare'] and not args.expert_type:
        parser.error(f"--expert_type is required for mode '{args.mode}'")
    
    if args.mode in ['fold', 'single', 'compare'] and args.fold_idx is None:
        parser.error(f"--fold_idx is required for mode '{args.mode}'")
    

    base_args = vars(args)
    base_args['output_config'] = output_config
    
    print(f"🚀 Unified MoE SwinUNETR Training")
    print(f"Task: {args.task}")
    print(f"Mode: {args.mode}")
    if args.expert_type:
        print(f"Expert Type: {args.expert_type}")
    if args.fold_idx is not None:
        print(f"Fold: {args.fold_idx}")
    print(f"GPU: {args.gpu}")
    

    trainer = UnifiedMoETrainer(base_args)
    
    try:

        if args.mode == 'all':
            trainer.run_all_experts_all_folds()
            
        elif args.mode == 'expert':
            trainer.run_expert_all_folds(args.expert_type)
            
        elif args.mode == 'fold':
            trainer.run_all_experts_single_fold(args.fold_idx)
            
        elif args.mode == 'single':
            trainer.train_single_model(args.expert_type, args.fold_idx, args.weight_mode)
            
        elif args.mode == 'compare':
            trainer.compare_weight_modes_single(args.expert_type, args.fold_idx)
        

        summary_file = trainer.save_final_summary()
        
        print(f"\n🎉 Unified MoE SwinUNETR training completed successfully!")
        print(f"📊 Summary saved to: {summary_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer.save_final_summary()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        traceback.print_exc()
        trainer.save_final_summary()
        sys.exit(1)


if __name__ == "__main__":
    main()