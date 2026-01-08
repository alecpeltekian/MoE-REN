#!/usr/bin/env python3


import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from collections import defaultdict
import threading
import psutil

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from experts import (
    load_image_and_mask_data,
    create_consistent_folds,
    prepare_data_dicts,
    get_transforms
)
from monai.data import Dataset, DataLoader

class GPUMemoryMonitor:
    def __init__(self, device_id=0, interval=0.1):
        self.device_id = 0
        self.interval = interval
        self.monitoring = False
        self.peak_memory_mb = 0
        self.memory_samples = []
        self.monitor_thread = None
        
    def _monitor_loop(self):
        while self.monitoring:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated(0) / (1024**2)
                reserved = torch.cuda.memory_reserved(0) / (1024**2)
                self.memory_samples.append({
                    'allocated_mb': allocated,
                    'reserved_mb': reserved,
                    'timestamp': time.time()
                })
                self.peak_memory_mb = max(self.peak_memory_mb, allocated)
            time.sleep(self.interval)
    
    def start(self):
        self.monitoring = True
        self.peak_memory_mb = 0
        self.memory_samples = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        if torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
            return {
                'peak_allocated_mb': peak_allocated,
                'peak_reserved_mb': peak_reserved,
                'monitored_peak_mb': self.peak_memory_mb,
                'samples': self.memory_samples
            }
        return {
            'peak_allocated_mb': self.peak_memory_mb,
            'peak_reserved_mb': self.peak_memory_mb,
            'monitored_peak_mb': self.peak_memory_mb,
            'samples': self.memory_samples
        }

class TrainingComplexityProfiler:
    def __init__(self, device='cuda:0', gpu_id=0):
        self.device = torch.device(device)
        self.gpu_id = gpu_id
        self.monitor = GPUMemoryMonitor(device_id=0)
        
    def profile_training(self, model, train_loader, val_loader, num_epochs=5, 
                        loss_fn=None, optimizer=None, verbose=True):
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        model.to(self.device)
        model.train()
        
        wall_clock_start = time.time()
        self.monitor.start()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated(self.device) / (1024**2)
        else:
            initial_memory = 0
        
        epoch_times = []
        batch_times = []
        
        try:
            for epoch in range(num_epochs):
                epoch_start = time.time()
                
                for batch_idx, batch in enumerate(train_loader):
                    batch_start = time.time()
                    
                    inputs = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    try:
                        if 'lobe_mask' in batch:
                            mask = batch['lobe_mask'].to(self.device)
                            outputs = model(inputs, lobe_mask=mask)
                        elif 'mask' in batch:
                            mask = batch['mask'].to(self.device)
                            outputs = model(inputs, lobe_mask=mask)
                        else:
                            outputs = model(inputs)
                    except TypeError as e:
                        try:
                            outputs = model(inputs)
                        except Exception:
                            raise e
                    
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    
                    if verbose and batch_idx % 10 == 0:
                        current_memory = torch.cuda.memory_allocated(self.device) / (1024**2) if torch.cuda.is_available() else 0
                        print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                              f"Loss: {loss.item():.4f}, Memory: {current_memory:.1f} MB")
                
                if val_loader is not None:
                    model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            inputs = batch['image'].to(self.device)
                            labels = batch['label'].to(self.device)
                            try:
                                if 'lobe_mask' in batch:
                                    mask = batch['lobe_mask'].to(self.device)
                                    outputs = model(inputs, lobe_mask=mask)
                                elif 'mask' in batch:
                                    mask = batch['mask'].to(self.device)
                                    outputs = model(inputs, mask=mask)
                                else:
                                    outputs = model(inputs)
                            except TypeError:
                                outputs = model(inputs)
                    model.train()
                
                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        
        finally:
            memory_stats = self.monitor.stop()
            wall_clock_end = time.time()
        
        total_wall_clock = wall_clock_end - wall_clock_start
        gpu_hours = total_wall_clock / 3600.0
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(self.device) / (1024**2)
            peak_memory = memory_stats['peak_allocated_mb']
        else:
            final_memory = 0
            peak_memory = memory_stats['monitored_peak_mb']
        
        return {
            'wall_clock_seconds': total_wall_clock,
            'wall_clock_hours': total_wall_clock / 3600.0,
            'gpu_hours': gpu_hours,
            'num_epochs': num_epochs,
            'avg_epoch_time_seconds': np.mean(epoch_times) if epoch_times else 0,
            'avg_batch_time_seconds': np.mean(batch_times) if batch_times else 0,
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'memory_stats': memory_stats,
            'epoch_times': epoch_times,
            'batch_times': batch_times[:100]
        }

class BaselineSwinUNETR(nn.Module):
    def __init__(self, img_size=(96, 96, 96), in_channels=1, out_channels=2, feature_size=48):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,
            feature_size=feature_size,
            use_checkpoint=True
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(feature_size, out_channels)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def create_baseline_swinunetr(img_size=(96, 96, 96), in_channels=1, out_channels=2, feature_size=48):
    return BaselineSwinUNETR(img_size, in_channels, out_channels, feature_size)

def create_moe_model(expert_type='cnn', img_size=(96, 96, 96), in_channels=1, out_channels=2, feature_size=48):
    from moe import MoEWeightedSwinUNETR
    model = MoEWeightedSwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        expert_weights=None,
        expert_type=expert_type,
        weight_mode='learned_moe'
    )
    return model

def create_gating_model(img_size=(96, 96, 96), in_channels=1, out_channels=2, feature_size=48):
    from gating_moe import GatingWeightedSwinUNETR
    model = GatingWeightedSwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        gating_weights=None,
        weight_mode='gating_fixed'
    )
    return model

def create_expert_model(expert_type='cnn', img_size=(96, 96, 96), in_channels=1, out_channels=2, feature_size=48):
    from experts import create_model
    model = create_model(
        model_type=expert_type,
        input_size=img_size
    )
    return model

def run_complexity_analysis(model_type, task, fold, num_epochs, batch_size, 
                           nifti_dir, lobe_masks_dir, labels_file, 
                           gpu_id, expert_idx=None, expert_type='cnn'):
    print(f"\n{'='*80}")
    print(f"Training Complexity Analysis")
    print(f"{'='*80}")
    print(f"Model Type: {model_type}")
    print(f"Task: {task}")
    print(f"Fold: {fold}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"GPU: {gpu_id}")
    print(f"{'='*80}\n")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    profiler = TrainingComplexityProfiler(device=device, gpu_id=gpu_id)
    
    print("Loading data...")
    file_list, mask_list, labels, patient_ids = load_image_and_mask_data(
        labels_file, nifti_dir, lobe_masks_dir, task
    )
    
    k_folds = 5
    consistent_folds = create_consistent_folds(file_list, labels, patient_ids, k_folds=k_folds, seed=42)
    fold_data = consistent_folds[fold]
    
    train_val_idx = fold_data['train_val_idx']
    holdout_idx = fold_data['holdout_idx']
    val_mask = fold_data['val_mask']
    
    train_files = np.array(file_list)[train_val_idx]
    train_masks = np.array(mask_list)[train_val_idx]
    train_labels = np.array(labels)[train_val_idx]
    
    holdout_files = np.array(file_list)[holdout_idx]
    holdout_masks = np.array(mask_list)[holdout_idx]
    holdout_labels = np.array(labels)[holdout_idx]
    
    val_files = holdout_files[val_mask]
    val_masks = holdout_masks[val_mask]
    val_labels = holdout_labels[val_mask]
    
    train_data = prepare_data_dicts(train_files, train_masks, train_labels)
    val_data = prepare_data_dicts(val_files, val_masks, val_labels)
    
    transforms = get_transforms()
    train_dataset = Dataset(data=train_data, transform=transforms)
    val_dataset = Dataset(data=val_data, transform=transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    print(f"\nCreating {model_type} model...")
    if model_type == 'swinunetr':
        model = create_baseline_swinunetr()
    elif model_type == 'moe':
        model = create_moe_model(expert_type=expert_type)
    elif model_type == 'gating':
        model = create_gating_model()
    elif model_type.startswith('expert_'):
        expert_type_name = model_type.replace('expert_', '')
        model = create_expert_model(expert_type=expert_type_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024**2)
    
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Estimated model size: {model_size_mb:.2f} MB")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print(f"\nStarting training profiling...")
    print(f"This will train for {num_epochs} epochs to measure complexity metrics.\n")
    
    results = profiler.profile_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        verbose=True
    )
    
    results['model_type'] = model_type
    results['task'] = task
    results['fold'] = fold
    results['batch_size'] = batch_size
    results['total_params'] = int(total_params)
    results['trainable_params'] = int(trainable_params)
    results['model_size_mb'] = float(model_size_mb)
    results['gpu_id'] = gpu_id
    results['expert_type'] = expert_type if model_type in ['moe', 'expert_cnn', 'expert_vit', 'expert_mamba'] else None
    
    print(f"\n{'='*80}")
    print(f"Results Summary")
    print(f"{'='*80}")
    print(f"Wall-clock time: {results['wall_clock_hours']:.4f} hours ({results['wall_clock_seconds']:.2f} seconds)")
    print(f"GPU-hours: {results['gpu_hours']:.4f}")
    print(f"Peak GPU memory: {results['peak_memory_mb']:.2f} MB ({results['peak_memory_mb']/1024:.2f} GB)")
    print(f"Average epoch time: {results['avg_epoch_time_seconds']:.2f} seconds")
    print(f"Average batch time: {results['avg_batch_time_seconds']:.4f} seconds")
    print(f"{'='*80}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Measure training time and memory complexity')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['swinunetr', 'moe', 'gating', 'expert_cnn', 'expert_vit', 'expert_mamba'],
                       help='Type of model to profile')
    parser.add_argument('--task', type=str, default='has_ILD',
                       help='Task name')
    parser.add_argument('--fold', type=int, default=0, choices=range(5),
                       help='Fold index (0-4)')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of epochs to train (for profiling)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ID')
    parser.add_argument('--expert_type', type=str, default='cnn',
                       choices=['cnn', 'vit', 'mamba'],
                       help='Expert type for MoE models')
    parser.add_argument('--nifti_dir', type=str, default='/data2/akp4895/MultipliedImagesClean',
                       help='Directory containing NIFTI files')
    parser.add_argument('--lobe_masks_dir', type=str, default='/data2/akp4895/1mm_OrientedImagesSegmented',
                       help='Directory containing lobe masks')
    parser.add_argument('--labels_file', type=str, default='/home/akp4895/MoE_MultiModal/Radiomics/data/labels.csv',
                       help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default='./complexity_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        results = run_complexity_analysis(
            model_type=args.model_type,
            task=args.task,
            fold=args.fold,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            nifti_dir=args.nifti_dir,
            lobe_masks_dir=args.lobe_masks_dir,
            labels_file=args.labels_file,
            gpu_id=args.gpu,
            expert_type=args.expert_type
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            args.output_dir,
            f"complexity_{args.model_type}_{args.task}_fold{args.fold}_{timestamp}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

