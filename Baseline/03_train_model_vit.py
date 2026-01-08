import os
import pandas as pd
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
import math
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord, Resized
from monai.data import DataLoader, CacheDataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from collections import Counter
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import re
import json
import argparse
from datetime import datetime


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

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

data_path = "/data2/akp4895/MultipliedImagesClean"
LABELS_FILE = '../Radiomics/data/labels.csv'
BASE_LOGDIR = "/data2/akp4895/MoE_MultiModal/radiomicsresults/baseline"

lr = 4e-4
max_epochs = 50
batch_size = 2
num_classes = 2
patience = 10

transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image"], spatial_size=(96, 96, 96)),
    ToTensord(keys=["image"]),
])

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
            if hasattr(os, 'statvfs'):
                st = os.statvfs(os.path.dirname(path))
                free_space = st.f_frsize * st.f_bavail
                if free_space < 1024 * 1024 * 1024:
                    print(f"Warning: Low disk space ({free_space / (1024*1024*1024):.2f}GB free)")
            
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

def load_data(labels_file, nifti_dir, task):
    print(f"Reading labels file for task: {task}")
    df = pd.read_csv(labels_file)
    df = df[df[task].notna()]
    df = df[df[task] != -1]
    file_list = []
    labels = []
    patient_ids = []
    scan_ids = []
    
    for patient_dir in os.listdir(nifti_dir):
        patient_path = os.path.join(nifti_dir, patient_dir)
        if os.path.isdir(patient_path):
            for filename in os.listdir(patient_path):
                if filename.endswith('.nii.gz'):
                    file_id = filename.replace('.nii.gz', '')
                    matching_row = df[df['ScanID'] == file_id]
                    
                    if not matching_row.empty and not pd.isna(matching_row[task].values[0]):
                        label = float(matching_row[task].values[0])
                        label = int(label)
                        file_path = os.path.join(patient_path, filename)
                        file_list.append(file_path)
                        labels.append(label)
                        patient_ids.append(get_patient_id(file_id))
                        scan_ids.append(file_id)
    
    print(f"Total matched files for {task}: {len(file_list)}")
    return file_list, labels, patient_ids, scan_ids

def prepare_data(files, labels, scan_ids=None):
    if scan_ids is None:
        return [{"image": file_path, "label": label} for file_path, label in zip(files, labels)]
    else:
        return [{"image": file_path, "label": label, "scan_id": scan_id} 
                for file_path, label, scan_id in zip(files, labels, scan_ids)]

def evaluate_model(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            total_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(data_loader)
    
    return metrics

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
                 embed_dim=128, n_layers=3, n_heads=4, mlp_ratio=1.5, 
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

class BaselineViT(nn.Module):
    def __init__(self, input_size=(96, 96, 96), dropout_rate=0.1):
        super(BaselineViT, self).__init__()
        
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

def train_model(task, resume_dir=None):
    if resume_dir:
        print(f"Attempting to resume training from {resume_dir}")
        if not os.path.exists(resume_dir):
            raise ValueError(f"Resume directory {resume_dir} does not exist")
        
        try:
            with open(os.path.join(resume_dir, 'config.json'), 'r') as f:
                config = json.load(f)
            logdir = resume_dir
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Failed to load config from resume directory: {e}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logdir = os.path.join(BASE_LOGDIR, f"{task}_vit_{timestamp}")
        os.makedirs(logdir, exist_ok=True)
        
        config = {
            'task': task,
            'learning_rate': lr,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'num_classes': num_classes,
            'patience': patience,
            'timestamp': timestamp,
            'model_type': 'vit'
        }
        safe_save_json(config, os.path.join(logdir, 'config.json'))
    
    file_list, labels, patient_ids, scan_ids = load_data(LABELS_FILE, data_path, task)
    
    class_counts = Counter(labels)
    class_weights = torch.tensor(
        [len(labels) / (2 * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float32
    )
    
    k_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)
    loss_function = CrossEntropyLoss(weight=class_weights)
    
    consistent_folds = create_consistent_folds(file_list, labels, patient_ids, scan_ids, 
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
    
    file_list = np.array(file_list)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    scan_ids = np.array(scan_ids)
    
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
    
    
    for fold_idx in range(starting_fold, k_folds):
        fold = fold_idx
        print(f"\nProcessing Fold {fold + 1}/{k_folds}")
        
        fold_data = consistent_folds[fold]
        train_val_idx = fold_data['train_val_idx']
        holdout_idx = fold_data['holdout_idx']
        val_mask = fold_data['val_mask']
        test_mask = fold_data['test_mask']
        
        checkpoint = load_checkpoint(logdir, fold) if resume_dir else None
        
        holdout_files = file_list[holdout_idx]
        holdout_labels = labels[holdout_idx]
        holdout_scan_ids = scan_ids[holdout_idx]
        
        train_files_dict = prepare_data(file_list[train_val_idx], labels[train_val_idx], scan_ids[train_val_idx])
        val_files_dict = prepare_data(holdout_files[val_mask], holdout_labels[val_mask], holdout_scan_ids[val_mask])
        test_files_dict = prepare_data(holdout_files[test_mask], holdout_labels[test_mask], holdout_scan_ids[test_mask])
        
        test_scan_id_list = holdout_scan_ids[test_mask].tolist()
        with open(os.path.join(logdir, f"fold_{fold}_test_scan_ids.json"), 'w') as f:
            json.dump(test_scan_id_list, f)
        
        train_ds = CacheDataset(data=train_files_dict, transform=transforms, cache_rate=0.0)
        val_ds = CacheDataset(data=val_files_dict, transform=transforms, cache_rate=0.0)
        test_ds = CacheDataset(data=test_files_dict, transform=transforms, cache_rate=0.0)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                num_workers=2, prefetch_factor=2, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                              num_workers=2, prefetch_factor=2, persistent_workers=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                               num_workers=2, prefetch_factor=2, persistent_workers=True)
        
        model = BaselineViT(input_size=(96, 96, 96), dropout_rate=0.1)
        model.to(device)
        
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
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
                    inputs, batch_labels = batch["image"].to(device), batch["label"].to(device)
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
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
                
                val_metrics = evaluate_model(model, val_loader, loss_function, device)
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

        test_metrics = evaluate_model(model, test_loader, loss_function, device)
        
        fold_summary = {
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
                'auc': float(best_checkpoint['val_metrics_history'][-1]['auroc'])
            }
        }
        
        safe_save_json(fold_summary, os.path.join(logdir, f"fold_{fold}_summary.json"))
        fold_results.append(fold_summary)
    
    overall_results = {
        'task': str(task),
        'number_of_folds': len(fold_results),
        'average_best_val_loss': float(np.mean([f['best_val_loss'] for f in fold_results])),
        'average_test_auc': float(np.mean([f['test_metrics']['auroc'] for f in fold_results])),
        'std_test_auc': float(np.std([f['test_metrics']['auroc'] for f in fold_results])),
        'average_test_accuracy': float(np.mean([f['test_metrics']['accuracy'] for f in fold_results])),
        'average_test_f1': float(np.mean([f['test_metrics']['f1'] for f in fold_results])),
        'fold_summaries': fold_results,
        'model_type': 'vit'
    }
    
    safe_save_json(overall_results, os.path.join(logdir, 'overall_results.json'))
    
    print("\nTraining completed for all folds!")
    print(f"Average best validation loss: {overall_results['average_best_val_loss']:.4f}")
    print(f"Average test AUC: {overall_results['average_test_auc']:.4f} (±{overall_results['std_test_auc']:.4f})")
    print(f"Average test F1: {overall_results['average_test_f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ViT baseline model for a specific task')
    parser.add_argument('--task', type=str, required=True, 
                      help='Task to train on (e.g., has_ILD, death_1, death_3, death_5, ILD_1, ILD_3, ILD_5)')
    parser.add_argument('--gpu', type=str, default='5',
                      help='GPU device number to use')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=4e-4,
                      help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                      help='Early stopping patience')
    parser.add_argument('--max_epochs', type=int, default=50,
                      help='Maximum number of epochs')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint directory to resume training from')
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size
    lr = args.lr
    patience = args.patience
    max_epochs = args.max_epochs
    
    print(f"\nStarting ViT baseline training for task: {args.task}")
    print(f"Using GPU: {args.gpu}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Early stopping patience: {patience}")
    print(f"Maximum epochs: {max_epochs}\n")
    
    try:
        train_model(args.task, args.resume)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

