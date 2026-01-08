#!/usr/bin/env python3


import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

ablation_dir = os.path.dirname(os.path.abspath(__file__))
unified_dir = os.path.dirname(ablation_dir)
sys.path.insert(0, str(Path(unified_dir) / "src"))

try:
    from moe import (
        MoEWeightedSwinUNETR, MoELoss, 
        run_moe_swin_training_pipeline, ExpertConfig
    )
except ImportError:
    print("Warning: Could not import from moe")
    MoEWeightedSwinUNETR = None
    ExpertConfig = None

try:
    from gating_moe import (
        GatingWeightedSwinUNETR, UnifiedGatingMoELoss,
        run_unified_training_pipeline, load_best_gating_strategy
    )
except ImportError:
    print("Warning: Could not import from gating_moe")
    GatingWeightedSwinUNETR = None
    load_best_gating_strategy = None

from experts import (
    load_image_and_mask_data, 
    create_consistent_folds,
    prepare_data_dicts,
    get_transforms
)
from monai.data import DataLoader, Dataset


class AblationMetricsTracker:
    
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.expert_weights_history = []
        self.expert_features_history = []
        self.expert_activations_history = []
        self.entropy_history = []
        self.load_balance_history = []
        self.diversity_history = []
    
    def update(self, expert_weights, expert_features=None):
        
        if expert_weights is None:
            return
        
        weights = expert_weights.detach().cpu().numpy()
        if len(weights.shape) == 1:
            weights = weights.unsqueeze(0)
        
        self.expert_weights_history.append(weights)
        
        # Calculate expert activation (threshold at 0.1)
        activations = (weights > 0.1).astype(int)
        self.expert_activations_history.append(activations)
        
        # Calculate entropy
        entropy = -(weights * np.log(weights + 1e-8)).sum(axis=1)
        self.entropy_history.append(entropy)
        
        # Calculate load balance (variance of expert usage)
        expert_usage = weights.sum(axis=0)
        load_balance_variance = np.var(expert_usage)
        self.load_balance_history.append(load_balance_variance)
        
        # Calculate diversity (cosine similarity between expert features)
        if expert_features is not None:
            expert_feats = expert_features['expert_features'].detach().cpu().numpy()
            batch_size = expert_feats.shape[0]
            num_experts = expert_feats.shape[1]
            
            diversity_scores = []
            for b in range(batch_size):
                feats = expert_feats[b]
                feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
                similarities = np.dot(feats_norm, feats_norm.T)
                mask = np.eye(num_experts)
                off_diagonal = similarities * (1 - mask)
                diversity_scores.append(np.abs(off_diagonal).mean())
            
            self.diversity_history.append(diversity_scores)
            self.expert_features_history.append(expert_feats)
    
    def compute_summary(self) -> Dict:
        
        if len(self.expert_weights_history) == 0:
            return {}
        
        all_weights = np.concatenate(self.expert_weights_history, axis=0)
        all_activations = np.concatenate(self.expert_activations_history, axis=0)
        
        # Expert activation frequency
        num_active_experts_per_sample = all_activations.sum(axis=1)
        expert_activation_frequency = {
            'mean': float(np.mean(num_active_experts_per_sample)),
            'std': float(np.std(num_active_experts_per_sample)),
            'min': int(np.min(num_active_experts_per_sample)),
            'max': int(np.max(num_active_experts_per_sample)),
            'median': float(np.median(num_active_experts_per_sample))
        }
        
        # Expert usage variance (load balancing)
        expert_usage_per_expert = all_weights.sum(axis=0)
        expert_usage_variance = {
            'variance': float(np.var(expert_usage_per_expert)),
            'std': float(np.std(expert_usage_per_expert)),
            'mean_usage': float(np.mean(expert_usage_per_expert)),
            'usage_per_expert': expert_usage_per_expert.tolist()
        }
        
        # Entropy statistics
        all_entropy = np.concatenate(self.entropy_history)
        entropy_stats = {
            'mean': float(np.mean(all_entropy)),
            'std': float(np.std(all_entropy)),
            'min': float(np.min(all_entropy)),
            'max': float(np.max(all_entropy))
        }
        
        # Load balance statistics
        load_balance_stats = {
            'mean_variance': float(np.mean(self.load_balance_history)),
            'std_variance': float(np.std(self.load_balance_history))
        }
        
        # Diversity statistics
        diversity_stats = {}
        if len(self.diversity_history) > 0:
            all_diversity = np.concatenate(self.diversity_history)
            diversity_stats = {
                'mean_cosine_similarity': float(np.mean(all_diversity)),
                'std_cosine_similarity': float(np.std(all_diversity)),
                'min': float(np.min(all_diversity)),
                'max': float(np.max(all_diversity))
            }
        
        return {
            'expert_activation_frequency': expert_activation_frequency,
            'expert_usage_variance': expert_usage_variance,
            'entropy_stats': entropy_stats,
            'load_balance_stats': load_balance_stats,
            'diversity_stats': diversity_stats
        }


class AblationMoELoss(nn.Module):
    
    
    def __init__(self, base_criterion=None, 
                 entropy_weight=0.01, 
                 diversity_weight=0.01,
                 use_entropy=True,
                 use_diversity=True):
        super().__init__()
        self.base_criterion = base_criterion or nn.CrossEntropyLoss()
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.use_entropy = use_entropy
        self.use_diversity = use_diversity
    
    def forward(self, pred, target, expert_weights=None, expert_features=None):
        base_loss = self.base_criterion(pred, target)
        total_loss = base_loss
        
        # Entropy regularization
        if expert_weights is not None and self.use_entropy and self.entropy_weight > 0:
            if len(expert_weights.shape) > 1:
                weight_entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=1).mean()
            else:
                weight_entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum()
            total_loss += -self.entropy_weight * weight_entropy
        
        # Diversity regularization
        if expert_features is not None and self.use_diversity and self.diversity_weight > 0:
            expert_feats = expert_features['expert_features']
            expert_feats_norm = F.normalize(expert_feats, dim=2)
            similarities = torch.bmm(expert_feats_norm, expert_feats_norm.transpose(1, 2))
            mask = torch.eye(similarities.shape[1], device=similarities.device).unsqueeze(0)
            similarities = similarities * (1 - mask)
            total_loss += self.diversity_weight * similarities.abs().mean()
        
        return total_loss


class AblationGatingMoELoss(nn.Module):
    
    
    def __init__(self, base_criterion=None,
                 entropy_weight=0.005,
                 load_balance_weight=0.01,
                 diversity_weight=0.01,
                 use_entropy=True,
                 use_load_balance=True,
                 use_diversity=True):
        super().__init__()
        self.base_criterion = base_criterion or nn.CrossEntropyLoss()
        self.entropy_weight = entropy_weight
        self.load_balance_weight = load_balance_weight
        self.diversity_weight = diversity_weight
        self.use_entropy = use_entropy
        self.use_load_balance = use_load_balance
        self.use_diversity = use_diversity
    
    def forward(self, pred, target, expert_weights=None, expert_features=None):
        base_loss = self.base_criterion(pred, target)
        total_loss = base_loss
        
        # Entropy regularization (gating)
        if expert_weights is not None and self.use_entropy and self.entropy_weight > 0:
            if len(expert_weights.shape) > 1:
                weight_entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=1).mean()
            else:
                weight_entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum()
            total_loss += -self.entropy_weight * weight_entropy
        
        # Load balance regularization
        if expert_weights is not None and self.use_load_balance and self.load_balance_weight > 0:
            if len(expert_weights.shape) > 1:
                uniform_target = torch.ones_like(expert_weights) / expert_weights.shape[1]
                weight_balance_loss = F.mse_loss(expert_weights, uniform_target)
            else:
                uniform_target = torch.ones_like(expert_weights) / len(expert_weights)
                weight_balance_loss = F.mse_loss(expert_weights, uniform_target)
            total_loss += self.load_balance_weight * weight_balance_loss
        
        # Diversity regularization
        if expert_features is not None and self.use_diversity and self.diversity_weight > 0:
            expert_feats = expert_features['expert_features']
            expert_feats_norm = F.normalize(expert_feats, dim=2)
            similarities = torch.bmm(expert_feats_norm, expert_feats_norm.transpose(1, 2))
            mask = torch.eye(similarities.shape[1], device=similarities.device).unsqueeze(0)
            similarities = similarities * (1 - mask)
            total_loss += self.diversity_weight * similarities.abs().mean()
        
        return total_loss


def train_with_ablation_config(model, train_loader, val_loader, device,
                              loss_function, metrics_tracker,
                              num_epochs=30, learning_rate=1e-4,
                              weight_decay=1e-5, early_stopping_patience=10):
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': [],
        'ablation_metrics': []
    }
    
    for epoch in range(num_epochs):
        metrics_tracker.reset()
        
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            output, weights, expert_info = model(image, mask,
                                                return_weights=True,
                                                return_expert_features=True)
            
            loss = loss_function(output, label, weights, expert_info)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            probs = F.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
            preds = output.argmax(dim=1).detach().cpu().numpy()
            
            train_preds.extend(probs)
            train_labels.extend(label.cpu().numpy())
            
            metrics_tracker.update(weights, expert_info)
        
        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds) if len(np.unique(train_labels)) > 1 else 0.5
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False):
                image = batch['image'].to(device)
                mask = batch['mask'].to(device)
                label = batch['label'].to(device)
                
                output, weights, expert_info = model(image, mask,
                                                   return_weights=True,
                                                   return_expert_features=True)
                
                loss = loss_function(output, label, weights, expert_info)
                val_loss += loss.item()
                
                probs = F.softmax(output, dim=1)[:, 1].cpu().numpy()
                preds = output.argmax(dim=1).cpu().numpy()
                
                val_preds.extend(probs)
                val_labels.extend(label.cpu().numpy())
                
                metrics_tracker.update(weights, expert_info)
        
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds) if len(np.unique(val_labels)) > 1 else 0.5
        
        # Track metrics
        ablation_metrics = metrics_tracker.compute_summary()
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['ablation_metrics'].append(ablation_metrics)
        
        scheduler.step()
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
    
    return history, best_val_auc, best_epoch


def run_ablation_experiment(task: str, fold: int, model_type: str,
                           ablation_config: Dict,
                           nifti_dir: str, lobe_masks_dir: str, labels_file: str,
                           expert_results_dir: Optional[str] = None,
                           num_epochs: int = 30, batch_size: int = 4,
                           learning_rate: float = 1e-4, gpu: str = '0') -> Dict:
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config_name = ablation_config['name']
    print(f"\n{'='*80}")
    print(f"Running ablation: {config_name}")
    print(f"{'='*80}")
    
    # Load data
    file_list, mask_list, labels, patient_ids = load_image_and_mask_data(
        labels_file, nifti_dir, lobe_masks_dir, task
    )
    
    k_folds = 5
    consistent_folds = create_consistent_folds(file_list, labels, patient_ids, k_folds=k_folds, seed=42)
    fold_data = consistent_folds[fold]
    
    # Extract splits
    holdout_idx = fold_data['holdout_idx']
    val_mask = fold_data['val_mask']
    test_mask = fold_data['test_mask']
    
    all_indices = np.arange(len(file_list))
    train_idx = np.setdiff1d(all_indices, holdout_idx)
    
    holdout_files = np.array(file_list)[holdout_idx]
    holdout_masks = np.array(mask_list)[holdout_idx]
    holdout_labels = np.array(labels)[holdout_idx]
    
    val_files = holdout_files[val_mask]
    val_masks = holdout_masks[val_mask]
    val_labels = holdout_labels[val_mask]
    
    test_files = holdout_files[test_mask]
    test_masks = holdout_masks[test_mask]
    test_labels = holdout_labels[test_mask]
    
    # Prepare datasets
    train_data = prepare_data_dicts(
        [file_list[i] for i in train_idx],
        [mask_list[i] for i in train_idx],
        [labels[i] for i in train_idx]
    )
    
    val_data = prepare_data_dicts(val_files, val_masks, val_labels)
    test_data = prepare_data_dicts(test_files, test_masks, test_labels)
    
    transforms = get_transforms()
    train_dataset = Dataset(data=train_data, transform=transforms)
    val_dataset = Dataset(data=val_data, transform=transforms)
    test_dataset = Dataset(data=test_data, transform=transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    if model_type == 'moe':
        if MoEWeightedSwinUNETR is None or ExpertConfig is None:
            raise ImportError("MoE model classes not available. Check imports.")
        
        expert_config = ExpertConfig.get_expert_config('cnn')
        expert_weights = None
        
        model = MoEWeightedSwinUNETR(
            expert_weights=expert_weights,
            expert_type='cnn',
            weight_mode='learned_moe',
            feature_size=48,
            dropout_rate=0.1
        )
        
        loss_function = AblationMoELoss(
            entropy_weight=ablation_config.get('entropy_weight', 0.01),
            diversity_weight=ablation_config.get('diversity_weight', 0.01),
            use_entropy=ablation_config.get('use_entropy', True),
            use_diversity=ablation_config.get('use_diversity', True)
        )
    
    elif model_type == 'gating':
        if GatingWeightedSwinUNETR is None or load_best_gating_strategy is None:
            raise ImportError("Gating model classes not available. Check imports.")
        
        if expert_results_dir is None:
            expert_results_dir = '/data2/akp4895/unified_cnn_results'
        
        best_gating_strategy = load_best_gating_strategy(expert_results_dir, task, fold, 'cnn')
        if best_gating_strategy is None:
            print(f"Warning: Could not load gating strategy, using uniform weights")
            gating_weights = None
        else:
            gating_weights = best_gating_strategy['weights']
        
        model = GatingWeightedSwinUNETR(
            gating_weights=gating_weights,
            weight_mode='gating_learnable',
            feature_size=48,
            dropout_rate=0.3
        )
        
        loss_function = AblationGatingMoELoss(
            entropy_weight=ablation_config.get('entropy_weight', 0.005),
            load_balance_weight=ablation_config.get('load_balance_weight', 0.01),
            diversity_weight=ablation_config.get('diversity_weight', 0.01),
            use_entropy=ablation_config.get('use_entropy', True),
            use_load_balance=ablation_config.get('use_load_balance', True),
            use_diversity=ablation_config.get('use_diversity', True)
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model = model.to(device)
    
    # Train with metrics tracking
    metrics_tracker = AblationMetricsTracker()
    history, best_val_auc, best_epoch = train_with_ablation_config(
        model, train_loader, val_loader, device, loss_function, metrics_tracker,
        num_epochs=num_epochs, learning_rate=learning_rate,
        early_stopping_patience=10
    )
    
    # Evaluate on test set
    model.eval()
    test_preds = []
    test_labels_list = []
    final_metrics_tracker = AblationMetricsTracker()
    
    with torch.no_grad():
        for batch in test_loader:
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            label = batch['label'].to(device)
            
            output, weights, expert_info = model(image, mask,
                                                return_weights=True,
                                                return_expert_features=True)
            
            probs = F.softmax(output, dim=1)[:, 1].cpu().numpy()
            preds = output.argmax(dim=1).cpu().numpy()
            
            test_preds.extend(probs)
            test_labels_list.extend(label.cpu().numpy())
            
            final_metrics_tracker.update(weights, expert_info)
    
    test_auc = roc_auc_score(test_labels_list, test_preds) if len(np.unique(test_labels_list)) > 1 else 0.5
    test_acc = accuracy_score(test_labels_list, (np.array(test_preds) >= 0.5).astype(int))
    
    # Get final ablation metrics
    final_ablation_metrics = final_metrics_tracker.compute_summary()
    
    results = {
        'config': ablation_config,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'test_auc': test_auc,
        'test_accuracy': test_acc,
        'ablation_metrics': final_ablation_metrics,
        'history': {
            'train_auc': history['train_auc'],
            'val_auc': history['val_auc'],
            'ablation_metrics': history['ablation_metrics']
        }
    }
    
    return results


def create_ablation_configs(model_type: str) -> List[Dict]:
    
    configs = []
    
    if model_type == 'moe':
        # Baseline: all regularization
        configs.append({
            'name': 'baseline_all',
            'use_entropy': True,
            'use_diversity': True,
            'entropy_weight': 0.01,
            'diversity_weight': 0.01
        })
        
        # Remove entropy
        configs.append({
            'name': 'no_entropy',
            'use_entropy': False,
            'use_diversity': True,
            'entropy_weight': 0.0,
            'diversity_weight': 0.01
        })
        
        # Remove diversity
        configs.append({
            'name': 'no_diversity',
            'use_entropy': True,
            'use_diversity': False,
            'entropy_weight': 0.01,
            'diversity_weight': 0.0
        })
        
        # Remove both
        configs.append({
            'name': 'no_regularization',
            'use_entropy': False,
            'use_diversity': False,
            'entropy_weight': 0.0,
            'diversity_weight': 0.0
        })
        
        # Vary entropy weight
        for weight in [0.001, 0.005, 0.02, 0.05]:
            configs.append({
                'name': f'entropy_{weight}',
                'use_entropy': True,
                'use_diversity': True,
                'entropy_weight': weight,
                'diversity_weight': 0.01
            })
        
        # Vary diversity weight
        for weight in [0.001, 0.005, 0.02, 0.05]:
            configs.append({
                'name': f'diversity_{weight}',
                'use_entropy': True,
                'use_diversity': True,
                'entropy_weight': 0.01,
                'diversity_weight': weight
            })
    
    elif model_type == 'gating':
        # Baseline: all regularization
        configs.append({
            'name': 'baseline_all',
            'use_entropy': True,
            'use_load_balance': True,
            'use_diversity': True,
            'entropy_weight': 0.005,
            'load_balance_weight': 0.01,
            'diversity_weight': 0.01
        })
        
        # Remove entropy
        configs.append({
            'name': 'no_entropy',
            'use_entropy': False,
            'use_load_balance': True,
            'use_diversity': True,
            'entropy_weight': 0.0,
            'load_balance_weight': 0.01,
            'diversity_weight': 0.01
        })
        
        # Remove load balance
        configs.append({
            'name': 'no_load_balance',
            'use_entropy': True,
            'use_load_balance': False,
            'use_diversity': True,
            'entropy_weight': 0.005,
            'load_balance_weight': 0.0,
            'diversity_weight': 0.01
        })
        
        # Remove diversity
        configs.append({
            'name': 'no_diversity',
            'use_entropy': True,
            'use_load_balance': True,
            'use_diversity': False,
            'entropy_weight': 0.005,
            'load_balance_weight': 0.01,
            'diversity_weight': 0.0
        })
        
        # Remove all
        configs.append({
            'name': 'no_regularization',
            'use_entropy': False,
            'use_load_balance': False,
            'use_diversity': False,
            'entropy_weight': 0.0,
            'load_balance_weight': 0.0,
            'diversity_weight': 0.0
        })
        
        # Vary entropy weight
        for weight in [0.001, 0.002, 0.01, 0.02]:
            configs.append({
                'name': f'entropy_{weight}',
                'use_entropy': True,
                'use_load_balance': True,
                'use_diversity': True,
                'entropy_weight': weight,
                'load_balance_weight': 0.01,
                'diversity_weight': 0.01
            })
        
        # Vary load balance weight
        for weight in [0.001, 0.005, 0.02, 0.05]:
            configs.append({
                'name': f'load_balance_{weight}',
                'use_entropy': True,
                'use_load_balance': True,
                'use_diversity': True,
                'entropy_weight': 0.005,
                'load_balance_weight': weight,
                'diversity_weight': 0.01
            })
        
        # Vary diversity weight
        for weight in [0.001, 0.005, 0.02, 0.05]:
            configs.append({
                'name': f'diversity_{weight}',
                'use_entropy': True,
                'use_load_balance': True,
                'use_diversity': True,
                'entropy_weight': 0.005,
                'load_balance_weight': 0.01,
                'diversity_weight': weight
            })
    
    return configs


def run_full_ablation(task: str, model_type: str, fold: int,
                     nifti_dir: str, lobe_masks_dir: str, labels_file: str,
                     expert_results_dir: Optional[str] = None,
                     num_epochs: int = 30, batch_size: int = 4,
                     learning_rate: float = 1e-4, gpu: str = '0',
                     output_dir: str = None) -> pd.DataFrame:
    
    
    if output_dir is None:
        output_dir = f'/data2/akp4895/ablation_results/{model_type}/{task}/fold_{fold}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*100}")
    print(f"FULL ABLATION STUDY")
    print(f"Task: {task}, Model Type: {model_type}, Fold: {fold}")
    print(f"{'='*100}\n")
    
    ablation_configs = create_ablation_configs(model_type)
    all_results = []
    
    for i, config in enumerate(ablation_configs):
        print(f"\n[{i+1}/{len(ablation_configs)}] Running: {config['name']}")
        
        try:
            result = run_ablation_experiment(
                task=task,
                fold=fold,
                model_type=model_type,
                ablation_config=config,
                nifti_dir=nifti_dir,
                lobe_masks_dir=lobe_masks_dir,
                labels_file=labels_file,
                expert_results_dir=expert_results_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                gpu=gpu
            )
            
            # Extract metrics for table
            ablation_metrics = result['ablation_metrics']
            row = {
                'config': config['name'],
                'test_auc': result['test_auc'],
                'test_accuracy': result['test_accuracy'],
                'best_val_auc': result['best_val_auc'],
                'best_epoch': result['best_epoch']
            }
            
            # Add ablation-specific metrics
            if 'expert_activation_frequency' in ablation_metrics:
                row['mean_active_experts'] = ablation_metrics['expert_activation_frequency']['mean']
                row['std_active_experts'] = ablation_metrics['expert_activation_frequency']['std']
            
            if 'expert_usage_variance' in ablation_metrics:
                row['expert_usage_variance'] = ablation_metrics['expert_usage_variance']['variance']
            
            if 'entropy_stats' in ablation_metrics:
                row['mean_entropy'] = ablation_metrics['entropy_stats']['mean']
            
            if 'diversity_stats' in ablation_metrics:
                row['mean_cosine_similarity'] = ablation_metrics['diversity_stats']['mean_cosine_similarity']
            
            # Add config flags
            row.update({k: v for k, v in config.items() if k != 'name'})
            
            all_results.append(row)
            
            # Save individual result
            result_file = os.path.join(output_dir, f"{config['name']}_results.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=4, default=str)
            
        except Exception as e:
            print(f"Error running {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create ablation table
    ablation_df = pd.DataFrame(all_results)
    if len(ablation_df) > 0 and 'test_auc' in ablation_df.columns:
        ablation_df = ablation_df.sort_values('test_auc', ascending=False)
    elif len(ablation_df) == 0:
        print("Warning: No successful ablation experiments completed!")
        return None
    
    # Save table
    table_file = os.path.join(output_dir, 'ablation_table.csv')
    ablation_df.to_csv(table_file, index=False)
    
    # Save summary
    summary = {
        'task': task,
        'model_type': model_type,
        'fold': fold,
        'ablation_table': ablation_df.to_dict('records'),
        'best_config': ablation_df.iloc[0].to_dict() if len(ablation_df) > 0 else None
    }
    
    summary_file = os.path.join(output_dir, 'ablation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    
    print(f"\n{'='*100}")
    print(f"ABLATION STUDY COMPLETE")
    print(f"{'='*100}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nTop 5 Configurations:")
    print(ablation_df[['config', 'test_auc', 'mean_active_experts', 'expert_usage_variance', 'mean_cosine_similarity']].head().to_string(index=False))
    
    return ablation_df


def main():
    parser = argparse.ArgumentParser(description='Ablation Analysis for MoE Regularization Terms')
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., has_ILD)')
    parser.add_argument('--model_type', type=str, required=True, choices=['moe', 'gating'],
                       help='Model type: moe or gating')
    parser.add_argument('--fold_idx', type=int, required=True, choices=range(5),
                       help='Fold index (0-4)')
    parser.add_argument('--all_folds', action='store_true',
                       help='Run ablation for all folds')
    parser.add_argument('--nifti_dir', type=str, default='/data2/akp4895/MultipliedImagesClean',
                       help='Directory containing NIFTI files')
    parser.add_argument('--lobe_masks_dir', type=str, default='/data2/akp4895/1mm_OrientedImagesSegmented',
                       help='Directory containing lobe masks')
    parser.add_argument('--labels_file', type=str, default='/home/akp4895/MoE_MultiModal/Radiomics/data/labels.csv',
                       help='Path to labels CSV file')
    parser.add_argument('--expert_results_dir', type=str, default=None,
                       help='Directory containing expert results (for gating model)')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.all_folds:
        print(f"Running ablation for all folds...")
        for fold in range(5):
            print(f"\n{'='*100}")
            print(f"FOLD {fold}")
            print(f"{'='*100}")
            run_full_ablation(
                task=args.task,
                model_type=args.model_type,
                fold=fold,
                nifti_dir=args.nifti_dir,
                lobe_masks_dir=args.lobe_masks_dir,
                labels_file=args.labels_file,
                expert_results_dir=args.expert_results_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gpu=args.gpu,
                output_dir=args.output_dir
            )
    else:
        run_full_ablation(
            task=args.task,
            model_type=args.model_type,
            fold=args.fold_idx,
            nifti_dir=args.nifti_dir,
            lobe_masks_dir=args.lobe_masks_dir,
            labels_file=args.labels_file,
            expert_results_dir=args.expert_results_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gpu=args.gpu,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()

