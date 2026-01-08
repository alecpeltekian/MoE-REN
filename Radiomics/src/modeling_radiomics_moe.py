#!/usr/bin/env python3

import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from collections import Counter
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import re
import argparse
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord, Resized
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, Dataset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def set_all_seeds(seed=42):
    import random
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
        return {key: make_json_serializable(value) for key, value in obj.items()}
    else:
        return obj

def safe_save_json(data, filepath):
    temp_file = filepath + '.tmp'
    try:
        serializable_data = make_json_serializable(data)
        with open(temp_file, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        os.replace(temp_file, filepath)
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e

def load_importance_weights(json_file_path, task, current_fold, expert_config):
    try:
        with open(json_file_path, 'r') as f:
            fold_data = json.load(f)
        
        fold_key = f"fold_{current_fold}"
        
        if fold_key not in fold_data:
            print(f"Warning: No data found for fold {current_fold}")
            return {name: 1.0 for name in expert_config['expert_names']}
        
        importance_weights = {}
        
        if expert_config['num_experts'] == 2:
            region_aucs = fold_data[fold_key].get("region_aucs", {})
            for expert_name in expert_config['expert_names']:
                if expert_name == "Left_Lung":
                    key = "left_lung"
                elif expert_name == "Right_Lung":
                    key = "right_lung"
                else:
                    key = expert_name.lower()
                
                if key in region_aucs:
                    importance_weights[expert_name] = region_aucs[key]
                    print(f"Loaded {expert_name}: AUC = {region_aucs[key]:.4f}")
                else:
                    importance_weights[expert_name] = 1.0
                    print(f"Warning: {expert_name} not found, using default weight 1.0")
        
        elif expert_config['num_experts'] == 5:
            lobe_aucs = fold_data[fold_key].get("lobe_aucs", {})
            for expert_name in expert_config['expert_names']:
                lobe_num = expert_name.split('_')[1]
                if lobe_num in lobe_aucs:
                    importance_weights[expert_name] = lobe_aucs[lobe_num]
                    print(f"Loaded {expert_name}: AUC = {lobe_aucs[lobe_num]:.4f}")
                else:
                    importance_weights[expert_name] = 1.0
        
        elif expert_config['num_experts'] == 7:
            lobe_aucs = fold_data[fold_key].get("lobe_aucs", {})
            region_aucs = fold_data[fold_key].get("region_aucs", {})
            
            for expert_name in expert_config['expert_names']:
                if expert_name.startswith("Lobe_"):
                    lobe_num = expert_name.split('_')[1]
                    if lobe_num in lobe_aucs:
                        importance_weights[expert_name] = lobe_aucs[lobe_num]
                    else:
                        importance_weights[expert_name] = 1.0
                elif expert_name == "Left_Lung":
                    importance_weights[expert_name] = region_aucs.get("left_lung", 1.0)
                elif expert_name == "Right_Lung":
                    importance_weights[expert_name] = region_aucs.get("right_lung", 1.0)
                else:
                    importance_weights[expert_name] = 1.0
        
        return importance_weights
        
    except Exception as e:
        print(f"Error loading importance weights: {e}")
        return {name: 1.0 for name in expert_config['expert_names']}

def create_consistent_folds(file_list, labels, patient_ids, scan_ids=None, k_folds=5, seed=42):
    file_list = np.array(file_list)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    if scan_ids is not None:
        scan_ids = np.array(scan_ids)
    
    gkf = GroupKFold(n_splits=k_folds)
    fold_indices = list(gkf.split(file_list, labels, groups=patient_ids))
    
    consistent_splits = []
    
    for fold in range(k_folds):
        train_val_idx, holdout_idx = fold_indices[fold]
        holdout_patients = patient_ids[holdout_idx]
        unique_patients = np.unique(holdout_patients)
        np.random.seed(seed)
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
            'test_mask': test_mask
        }
        consistent_splits.append(fold_data)
    
    return consistent_splits

class RegionAttentionModule(nn.Module):
    def __init__(self, feature_size):
        super(RegionAttentionModule, self).__init__()
        self.attention_conv = nn.Conv3d(feature_size, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, region_mask, region_weight):
        batch_size = x.size(0)
        attention_base = self.sigmoid(self.attention_conv(x))
        region_weight = region_weight.view(batch_size, 1, 1, 1, 1)
        weighted_attention = attention_base * region_mask * region_weight
        attended_features = x * weighted_attention
        return attended_features

class RegionExpert(nn.Module):
    def __init__(self, img_size, in_channels, feature_size, expert_type='lobe', region_id=None, num_classes=2):
        super(RegionExpert, self).__init__()
        self.expert_type = expert_type
        self.region_id = region_id
        
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,
            feature_size=feature_size,
            use_checkpoint=True,
        )
        
        self.region_attention = RegionAttentionModule(feature_size)
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(feature_size, feature_size)
        )
    
    def forward(self, x, region_mask=None, region_weight=None):
        batch_size = x.size(0)
        device = x.device
        
        features = self.swin_unetr(x)
        
        if region_mask is not None and region_weight is not None:
            if region_mask.shape[2:] != features.shape[2:]:
                region_mask = F.interpolate(
                    region_mask.float(),
                    size=features.shape[2:],
                    mode='nearest'
                )
            
            attended_features = self.region_attention(features, region_mask, region_weight)
        else:
            attended_features = features
        
        return self.classification_head(attended_features)

class GeneralMoE(nn.Module):
    def __init__(self, experts, feature_size=48, num_classes=2):
        super(GeneralMoE, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
        
        self.gating_network = nn.Sequential(
            nn.Linear(feature_size * self.num_experts, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.num_experts),
            nn.Softmax(dim=1)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_size * self.num_experts, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.final_classifier = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self.gate_weights_history = []
    
    def forward(self, x, region_masks=None, region_weights=None, return_gates=False):
        expert_outputs = []
        
        for i, expert in enumerate(self.experts):
            if region_masks is not None and region_weights is not None:
                mask = region_masks[i] if isinstance(region_masks, list) else region_masks[:, i:i+1]
                weight = region_weights[:, i] if region_weights is not None else None
                expert_feat = expert(x, mask, weight)
            else:
                expert_feat = expert(x)
            expert_outputs.append(expert_feat)
        
        combined_features = torch.cat(expert_outputs, dim=1)
        gates = self.gating_network(combined_features)
        
        if return_gates:
            self.gate_weights_history.append(gates.detach().cpu().numpy())
        
        expert_outputs_tensor = torch.stack(expert_outputs, dim=1)
        gated_output = torch.sum(gates.unsqueeze(-1) * expert_outputs_tensor, dim=1)
        
        fused_features = self.fusion_layer(combined_features)
        final_features = gated_output + fused_features
        
        return self.final_classifier(final_features)

def get_transforms(img_size=(96, 96, 96)):
    return Compose([
        LoadImaged(keys=["image", "lobe_mask"]),
        EnsureChannelFirstd(keys=["image", "lobe_mask"]),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image", "lobe_mask"], spatial_size=img_size),
        ToTensord(keys=["image", "lobe_mask"]),
    ])

def load_image_and_mask_data(labels_file, nifti_dir, lobe_masks_dir, task):
    print(f"Reading labels file for task: {task}")
    df = pd.read_csv(labels_file)
    df = df[df[task].notna()]
    df = df[df[task] != -1]
    
    file_list = []
    mask_list = []
    labels = []
    patient_ids = []
    
    for patient_dir in sorted(os.listdir(nifti_dir)):
        patient_path = os.path.join(nifti_dir, patient_dir)
        if os.path.isdir(patient_path):
            for filename in sorted(os.listdir(patient_path)):
                if filename.endswith('.nii.gz'):
                    file_id = filename.replace('.nii.gz', '')
                    matching_row = df[df['ScanID'] == file_id]
                    
                    mask_path = os.path.join(lobe_masks_dir, patient_dir, filename)
                    if not matching_row.empty and not pd.isna(matching_row[task].values[0]) and os.path.exists(mask_path):
                        label = float(matching_row[task].values[0])
                        label = int(label)
                        
                        file_path = os.path.join(patient_path, filename)
                        file_list.append(file_path)
                        mask_list.append(mask_path)
                        labels.append(label)
                        patient_ids.append(get_patient_id(file_id))
    
    print(f"Total matched files with masks for {task}: {len(file_list)}")
    return file_list, mask_list, labels, patient_ids

def prepare_data_with_masks(files, masks, labels):
    data_dicts = []
    for file_path, mask_path, label in zip(files, masks, labels):
        file_id = os.path.basename(file_path).replace('.nii.gz', '')
        data_dicts.append({
            "image": file_path,
            "lobe_mask": mask_path,
            "label": label,
            "scan_id": file_id
        })
    return data_dicts

def create_region_masks(lobe_masks, expert_config, device):
    batch_size = lobe_masks.size(0)
    region_masks = []
    
    if lobe_masks.shape[1] == 1:
        lobe_masks_5_channel = torch.zeros(
            (batch_size, 5, *lobe_masks.shape[2:]),
            device=device
        )
        for i in range(5):
            lobe_masks_5_channel[:, i] = (lobe_masks[:, 0] == i+1)
        lobe_masks = lobe_masks_5_channel
    
    for expert_name, expert_type in zip(expert_config['expert_names'], expert_config['expert_types']):
        if expert_type == 'lobe':
            lobe_num = int(expert_name.split('_')[1])
            region_mask = lobe_masks_5_channel[:, lobe_num - 1:lobe_num].float()
        elif expert_type == 'lung':
            if expert_name == "Left_Lung":
                region_mask = (lobe_masks_5_channel[:, 0] + lobe_masks_5_channel[:, 1]).clamp(0, 1).unsqueeze(1)
            elif expert_name == "Right_Lung":
                region_mask = (lobe_masks_5_channel[:, 2] + lobe_masks_5_channel[:, 3] + lobe_masks_5_channel[:, 4]).clamp(0, 1).unsqueeze(1)
            else:
                region_mask = torch.ones((batch_size, 1, *lobe_masks.shape[2:]), device=device)
        else:
            region_mask = torch.ones((batch_size, 1, *lobe_masks.shape[2:]), device=device)
        
        region_masks.append(region_mask)
    
    return region_masks

def create_region_weights(importance_weights, expert_config, batch_size, device):
    weights = torch.ones(batch_size, expert_config['num_experts'], device=device)
    
    for i, expert_name in enumerate(expert_config['expert_names']):
        if expert_name in importance_weights:
            auc_value = importance_weights[expert_name]
            if auc_value >= 0.5:
                weight = 0.1 + (auc_value - 0.5) * 3.8
            else:
                weight = 0.1
            weights[:, i] = weight
    
    return weights

def calculate_metrics(outputs, labels):
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    labels_np = labels.cpu().numpy()
    
    try:
        accuracy = np.mean(predictions == labels_np)
        auroc = roc_auc_score(labels_np, probabilities[:, 1])
        conf_mat = confusion_matrix(labels_np, predictions)
        class_report = classification_report(labels_np, predictions, output_dict=True)
        precision, recall, _ = precision_recall_curve(labels_np, probabilities[:, 1])
        avg_precision = average_precision_score(labels_np, probabilities[:, 1])
        
        return {
            'accuracy': float(accuracy),
            'auroc': float(auroc),
            'conf_matrix': conf_mat,
            'precision': float(class_report['weighted avg']['precision']),
            'recall': float(class_report['weighted avg']['recall']),
            'f1': float(class_report['weighted avg']['f1-score']),
            'avg_precision': float(avg_precision),
            'loss': 0.0
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

def save_checkpoint(state, is_best, logdir, fold, epoch):
    os.makedirs(os.path.join(logdir, f"fold_{fold}"), exist_ok=True)
    
    checkpoint_state = {
        'epoch': epoch,
        'model_state_dict': state['model'].state_dict(),
        'optimizer_state_dict': state['optimizer'].state_dict(),
        'best_val_loss': state['best_val_loss'],
        'train_metrics_history': state['train_metrics_history'],
        'val_metrics_history': state['val_metrics_history'],
        'early_stop_counter': state['early_stop_counter']
    }
    
    def safe_save(path, state_dict):
        temp_path = path + '.tmp'
        try:
            torch.save(state_dict, temp_path)
            os.replace(temp_path, path)
            return True
        except Exception as e:
            print(f"Error saving checkpoint to {path}: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False
    
    if epoch % 5 == 0:
        checkpoint_path = os.path.join(logdir, f"fold_{fold}", f"checkpoint_epoch_{epoch}.pth")
        if not safe_save(checkpoint_path, checkpoint_state):
            print(f"Failed to save checkpoint for epoch {epoch}")
    
    if is_best:
        best_model_path = os.path.join(logdir, f"fold_{fold}_best_model.pth")
        if not safe_save(best_model_path, checkpoint_state):
            print(f"Failed to save best model checkpoint")

def load_checkpoint(logdir, fold):
    fold_dir = os.path.join(logdir, f"fold_{fold}")
    if not os.path.exists(fold_dir):
        return None
    
    checkpoints = [f for f in os.listdir(fold_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(fold_dir, latest_checkpoint)
    
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} for fold {fold}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def clean_old_checkpoints(logdir, fold, current_epoch, keep_last_n=3):
    fold_dir = os.path.join(logdir, f"fold_{fold}")
    if not os.path.exists(fold_dir):
        return
        
    checkpoints = [f for f in os.listdir(fold_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    for checkpoint in checkpoints[:-keep_last_n]:
        try:
            os.remove(os.path.join(fold_dir, checkpoint))
        except Exception as e:
            print(f"Error removing old checkpoint: {e}")

def evaluate_model(model, data_loader, loss_function, device, expert_config, importance_weights):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = batch["image"].to(device)
            lobe_masks = batch["lobe_mask"].to(device)
            labels = batch["label"].to(device)
            
            region_masks = create_region_masks(lobe_masks, expert_config, device)
            region_weights = create_region_weights(importance_weights, expert_config, inputs.size(0), device)
            
            outputs = model(inputs, region_masks, region_weights)
            
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(data_loader)
    
    return metrics

def train_model(config_path, resume_dir=None, single_fold=None):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    task = config['task']
    num_experts = config['num_experts']
    expert_key = f"{num_experts}_experts"
    expert_config = config['expert_config'][expert_key]
    expert_config['num_experts'] = num_experts
    
    data_paths = config['data_paths']
    output_paths = config['output_paths']
    training_config = config['training']
    
    BASE_LOGDIR = output_paths['model_results_dir']
    data_path = data_paths['nifti_dir']
    LABELS_FILE = data_paths['labels_file']
    LOBE_MASKS_PATH = data_paths['lobe_masks_dir']
    
    lr = training_config['learning_rate']
    max_epochs = training_config['max_epochs']
    batch_size = training_config['batch_size']
    num_classes = training_config['num_classes']
    patience = training_config['patience']
    feature_size = training_config['feature_size']
    img_size = tuple(training_config['img_size'])
    
    if resume_dir:
        print(f"Attempting to resume training from {resume_dir}")
        if not os.path.exists(resume_dir):
            raise ValueError(f"Resume directory {resume_dir} does not exist")
        
        try:
            with open(os.path.join(resume_dir, 'config.json'), 'r') as f:
                resume_config = json.load(f)
            logdir = resume_dir
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Failed to load config from resume directory: {e}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if single_fold is not None:
            logdir = os.path.join(BASE_LOGDIR, f"{task}_moe_{num_experts}expert_fold_{single_fold}_{timestamp}")
        else:
            logdir = os.path.join(BASE_LOGDIR, f"{task}_moe_{num_experts}expert_{timestamp}")
        os.makedirs(logdir, exist_ok=True)
        
        save_config = {
            'task': task,
            'num_experts': num_experts,
            'expert_config': expert_config,
            'learning_rate': lr,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'num_classes': num_classes,
            'patience': patience,
            'timestamp': timestamp,
            'single_fold': single_fold,
        }
        safe_save_json(save_config, os.path.join(logdir, 'config.json'))
    
    file_list, mask_list, labels, patient_ids = load_image_and_mask_data(
        LABELS_FILE, data_path, LOBE_MASKS_PATH, task
    )
    
    class_counts = Counter(labels)
    class_weights = torch.tensor(
        [len(labels) / (2 * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float32
    )
    
    k_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)
    loss_function = CrossEntropyLoss(weight=class_weights)
    
    file_list = np.array(file_list)
    mask_list = np.array(mask_list)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    
    consistent_folds = create_consistent_folds(file_list, labels, patient_ids,
                                               [os.path.basename(f).replace('.nii.gz', '') for f in file_list],
                                               k_folds=k_folds, seed=42)
    
    fold_splits_file = os.path.join(logdir, 'fold_splits.json')
    fold_splits_data = {}
    for fold_idx, fold_data in enumerate(consistent_folds):
        fold_splits_data[f"fold_{fold_idx}"] = {
            'train_val_count': int(len(fold_data['train_val_idx'])),
            'holdout_count': int(len(fold_data['holdout_idx'])),
            'val_count': int(np.sum(fold_data['val_mask'])),
            'test_count': int(np.sum(fold_data['test_mask'])),
        }
    safe_save_json(fold_splits_data, fold_splits_file)
    
    fold_results = []
    
    if single_fold is not None:
        if single_fold < 0 or single_fold >= k_folds:
            raise ValueError(f"single_fold must be between 0 and {k_folds-1}")
        fold_range = [single_fold]
        print(f"Training single fold: {single_fold}")
    else:
        starting_fold = 0
        if resume_dir:
            for potential_fold in range(k_folds):
                fold_summary_path = os.path.join(resume_dir, f"fold_{potential_fold}_summary.json")
                if not os.path.exists(fold_summary_path):
                    starting_fold = potential_fold
                    break
            
            if starting_fold == k_folds:
                print("All folds have been completed. Nothing to resume.")
                return
            
            print(f"Resuming training from fold {starting_fold}")
        
        fold_range = range(starting_fold, k_folds)
    
    importance_file = os.path.join(output_paths['importance_results_dir'], 'fold_specific', f"{task}_fold_auc_results.json")
    if num_experts == 2:
        importance_file = os.path.join(output_paths['importance_results_dir'], 'fold_specific', f"{task}_fold_auc_results.json")
    elif num_experts == 5:
        importance_file = os.path.join(output_paths['importance_results_dir'], 'fold_specific', f"{task}_fold_auc_results.json")
    elif num_experts == 7:
        lobe_file = os.path.join(output_paths['importance_results_dir'], 'fold_specific', f"{task}_fold_auc_results.json")
        lung_file = os.path.join(output_paths['importance_results_dir'], 'fold_specific', f"{task}_fold_auc_results.json")
        importance_file = lobe_file
    
    for fold in fold_range:
        print(f"\nProcessing Fold {fold + 1}/{k_folds}")
        
        fold_data = consistent_folds[fold]
        train_val_idx = fold_data['train_val_idx']
        holdout_idx = fold_data['holdout_idx']
        val_mask = fold_data['val_mask']
        test_mask = fold_data['test_mask']
        
        checkpoint = load_checkpoint(logdir, fold) if resume_dir else None
        
        holdout_files = file_list[holdout_idx]
        holdout_masks = mask_list[holdout_idx]
        holdout_labels = labels[holdout_idx]
        
        train_files_dict = prepare_data_with_masks(file_list[train_val_idx], mask_list[train_val_idx], labels[train_val_idx])
        val_files_dict = prepare_data_with_masks(holdout_files[val_mask], holdout_masks[val_mask], holdout_labels[val_mask])
        test_files_dict = prepare_data_with_masks(holdout_files[test_mask], holdout_masks[test_mask], holdout_labels[test_mask])
        
        transforms = get_transforms(img_size)
        train_ds = Dataset(data=train_files_dict, transform=transforms)
        val_ds = Dataset(data=val_files_dict, transform=transforms)
        test_ds = Dataset(data=test_files_dict, transform=transforms)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=2, prefetch_factor=2, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, prefetch_factor=2, persistent_workers=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                               num_workers=2, prefetch_factor=2, persistent_workers=True)
        
        importance_weights = load_importance_weights(importance_file, task, fold, expert_config)
        
        experts = []
        for expert_name, expert_type in zip(expert_config['expert_names'], expert_config['expert_types']):
            if expert_type == 'lobe':
                region_id = int(expert_name.split('_')[1])
            else:
                region_id = None
            expert = RegionExpert(img_size, 1, feature_size, expert_type, region_id, num_classes)
            experts.append(expert)
        
        model = GeneralMoE(experts, feature_size, num_classes)
        model.to(device)
        
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=training_config['weight_decay'])
        
        start_epoch = 1
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_metrics_history = []
        val_metrics_history = []
        
        if checkpoint is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            train_metrics_history = checkpoint['train_metrics_history']
            val_metrics_history = checkpoint['val_metrics_history']
            early_stop_counter = checkpoint['early_stop_counter']
            print(f"Resuming training from epoch {start_epoch}")
        
        try:
            for epoch in range(start_epoch, max_epochs + 1):
                print(f"\nEpoch {epoch}/{max_epochs}")
                
                model.train()
                train_loss = 0
                train_outputs = []
                train_labels_list = []
                
                for batch in tqdm(train_loader, desc="Training"):
                    inputs = batch["image"].to(device)
                    lobe_masks = batch["lobe_mask"].to(device)
                    batch_labels = batch["label"].to(device)
                    
                    region_masks = create_region_masks(lobe_masks, expert_config, device)
                    region_weights = create_region_weights(importance_weights, expert_config, inputs.size(0), device)
                    
                    optimizer.zero_grad()
                    
                    return_gates = (epoch % 5 == 0)
                    outputs = model(inputs, region_masks, region_weights, return_gates=return_gates)
                    
                    if return_gates and len(model.gate_weights_history) > 0:
                        gate_weights_epoch = model.gate_weights_history[-1]
                        avg_gates = gate_weights_epoch.mean(axis=0)
                        print(f"Gate weights: {avg_gates}")
                        model.gate_weights_history = []
                    
                    loss = loss_function(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_outputs.append(outputs.detach())
                    train_labels_list.append(batch_labels)
                
                train_outputs = torch.cat(train_outputs)
                train_labels = torch.cat(train_labels_list)
                train_metrics = calculate_metrics(train_outputs, train_labels)
                train_metrics['loss'] = train_loss / len(train_loader)
                train_metrics_history.append(train_metrics)
                
                val_metrics = evaluate_model(model, val_loader, loss_function, device, expert_config, importance_weights)
                val_metrics_history.append(val_metrics)
                
                print(f"Train Loss: {train_metrics['loss']:.4f}, Train AUC: {train_metrics['auroc']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['auroc']:.4f}")
                
                is_best = val_metrics['loss'] < best_val_loss
                if is_best or epoch % 5 == 0:
                    state = {
                        'model': model,
                        'optimizer': optimizer,
                        'best_val_loss': best_val_loss,
                        'train_metrics_history': train_metrics_history,
                        'val_metrics_history': val_metrics_history,
                        'early_stop_counter': early_stop_counter
                    }
                    save_checkpoint(state, is_best, logdir, fold, epoch)
                
                if epoch % 5 == 0:
                    clean_old_checkpoints(logdir, fold, epoch)
                
                if is_best:
                    best_val_loss = val_metrics['loss']
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
                    
        except Exception as e:
            print(f"Error during training: {str(e)}")
            state = {
                'model': model,
                'optimizer': optimizer,
                'best_val_loss': best_val_loss,
                'train_metrics_history': train_metrics_history,
                'val_metrics_history': val_metrics_history,
                'early_stop_counter': early_stop_counter
            }
            save_checkpoint(state, False, logdir, fold, epoch)
            raise e
        
        best_model_path = os.path.join(logdir, f"fold_{fold}_best_model.pth")
        best_checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_metrics = evaluate_model(model, test_loader, loss_function, device, expert_config, importance_weights)
        
        fold_summary = {
            'fold': fold,
            'best_val_loss': float(best_val_loss),
            'best_epoch': int(best_checkpoint['epoch']),
            'total_epochs': int(epoch),
            'early_stopped': bool(early_stop_counter >= patience),
            'test_metrics': {
                'loss': float(test_metrics['loss']),
                'accuracy': float(test_metrics['accuracy']),
                'auroc': float(test_metrics['auroc']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1': float(test_metrics['f1']),
                'avg_precision': float(test_metrics['avg_precision']),
                'conf_matrix': test_metrics['conf_matrix'].tolist()
            },
            'final_val_metrics': {
                'loss': float(best_val_loss),
                'auc': float(val_metrics['auroc'])
            }
        }
        
        safe_save_json(fold_summary, os.path.join(logdir, f"fold_{fold}_summary.json"))
        fold_results.append(fold_summary)
        
        print(f"\nFold {fold} completed!")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Test AUC: {test_metrics['auroc']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
    
    if fold_results:
        overall_results = {
            'task': str(task),
            'num_experts': num_experts,
            'number_of_folds': len(fold_results),
            'average_best_val_loss': float(np.mean([f['best_val_loss'] for f in fold_results])),
            'average_test_auc': float(np.mean([f['test_metrics']['auroc'] for f in fold_results])),
            'std_test_auc': float(np.std([f['test_metrics']['auroc'] for f in fold_results])),
            'average_test_accuracy': float(np.mean([f['test_metrics']['accuracy'] for f in fold_results])),
            'average_test_f1': float(np.mean([f['test_metrics']['f1'] for f in fold_results])),
            'fold_summaries': fold_results,
            'model_type': f'moe_{num_experts}expert',
            'single_fold_training': single_fold is not None,
        }
        
        safe_save_json(overall_results, os.path.join(logdir, 'overall_results.json'))
        
        print(f"\nTraining completed!")
        if len(fold_results) > 1:
            print(f"Average best validation loss: {overall_results['average_best_val_loss']:.4f}")
            print(f"Average test AUC: {overall_results['average_test_auc']:.4f} (±{overall_results['std_test_auc']:.4f})")
            print(f"Average test F1: {overall_results['average_test_f1']:.4f}")
    
    return overall_results if fold_results else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--fold', type=int, default=None)
    
    args = parser.parse_args()
    
    train_model(args.config, args.resume, args.fold)

