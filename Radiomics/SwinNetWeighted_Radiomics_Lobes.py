import os
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import json
import time
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, ToTensord, Resized
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, Dataset
from monai.networks.blocks import MLPBlock
import numpy as np
from datetime import datetime
from collections import Counter
from monai.losses import DiceLoss
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import re
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
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


BASE_LOGDIR = "/data2/akp4895/radiomicsresults"
data_path = "/data2/akp4895/MultipliedImagesClean"
LABELS_FILE = '../../radiomics/filtered_radiomics_output/labels.csv'
LOBE_MASKS_PATH = "/data2/akp4895/1mm_OrientedImagesSegmented"


lr = 4e-4
max_epochs = 50
batch_size = 4
num_classes = 2
patience = 10

def get_patient_id(id_str):
    match = re.search(r'(\d{4})', id_str)
    return match.group(1) if match else None

def make_json_serializable(obj):
    """Convert numpy types to standard Python types for JSON serialization"""
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
    """Safely save JSON by writing to temporary file first"""
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

def load_radiomics_auc_weights(json_file_path, task='has_ILD', current_fold=0):
    """
    Load radiomics AUC values from JSON file and convert to lobe importance weights
    
    Args:
        json_file_path: Path to the JSON file with radiomics results
        task: Task name (e.g., 'has_ILD') - used for file path construction
        current_fold: Current fold number (0-4)
    
    Returns:
        Dictionary with lobe importance weights
    """
    try:

        with open(json_file_path, 'r') as f:
            fold_data = json.load(f)


        fold_key = f"fold_{current_fold}"

        if fold_key not in fold_data:
            print(f"Warning: No data found for fold {current_fold}")
            return {str(i): 1.0 for i in range(1, 8)}


        region_aucs = fold_data[fold_key]["region_aucs"]


        importance_weights = {}

        region_names = {
            '1': 'Left Upper Lobe (LUL)',
            '2': 'Left Lower Lobe (LLL)',
            '3': 'Right Upper Lobe (RUL)',
            '4': 'Right Middle Lobe (RML)',
            '5': 'Right Lower Lobe (RLL)',
            'left_lung': 'Left Lung',
            'right_lung': 'Right Lung'
        }

        for lobe_key, auc_value in region_aucs.items():
            importance_weights[lobe_key] = auc_value
            region_name = region_names.get(lobe_key, lobe_key)
            print(f"Loaded {region_name}: AUC = {auc_value:.4f}")

        return importance_weights

    except Exception as e:
        print(f"Error loading radiomics AUC weights: {e}")

        return {str(i): 1.0 for i in range(1, 8)}

def create_consistent_folds(file_list, labels, patient_ids, scan_ids=None, k_folds=5, seed=42):
    """Create consistent folds for cross-validation with deterministic splits"""
    import numpy as np
    from sklearn.model_selection import GroupKFold, train_test_split


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
    """Save checkpoint with all necessary state information with error handling"""
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
        """Safely save checkpoint with temporary file"""
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
    """Load the latest checkpoint for a given fold"""
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
    """Remove old checkpoints, keeping only the last n"""
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
    """Calculate comprehensive metrics for the predictions"""
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


class EnhancedLobeAttentionModule(nn.Module):
    """
    Enhanced attention module that handles both individual lobes and full lung regions
    """
    def __init__(self, feature_size):
        super(EnhancedLobeAttentionModule, self).__init__()

        self.attention_conv = nn.Conv3d(feature_size, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lobe_masks, lobe_weights):
        """
        x: Feature maps [B, C, H, W, D]
        lobe_masks: One-hot encoded masks [B, 7, H, W, D] (5 lobes + 2 full lungs)
        lobe_weights: Importance weights [B, 7]
        """
        batch_size, channels = x.size(0), x.size(1)


        attention_base = self.sigmoid(self.attention_conv(x))


        weighted_attention = torch.zeros_like(attention_base)

        for i in range(7):

            region_mask = lobe_masks[:, i:i+1, :, :, :]


            region_weight = lobe_weights[:, i].view(batch_size, 1, 1, 1, 1)


            weighted_attention += attention_base * region_mask * region_weight


        attended_features = x * weighted_attention

        return attended_features


class RadiomicsFusionModule(nn.Module):
    """
    Fuses image features with radiomics features using gating mechanism
    """
    def __init__(self, image_features, radiomics_features):
        super(RadiomicsFusionModule, self).__init__()
        self.image_proj = nn.Linear(image_features, 256)
        self.radiomics_proj = nn.Linear(radiomics_features, 256)
        self.gate = nn.Linear(image_features + radiomics_features, 2)
        self.fusion = nn.Linear(256, 256)

    def forward(self, image_feats, radiomics_feats):
        """
        Fuses image and radiomics features using a gating mechanism
        """
        batch_size = image_feats.size(0)


        img_proj = self.image_proj(image_feats)
        rad_proj = self.radiomics_proj(radiomics_feats)


        concat_feats = torch.cat([image_feats, radiomics_feats], dim=1)
        gates = F.softmax(self.gate(concat_feats), dim=1)


        fused = gates[:, 0:1] * img_proj + gates[:, 1:2] * rad_proj


        return self.fusion(fused)

class LobeImportanceGuidedTransformer(nn.Module):
    """
    Enhanced Swin-UNETR with corrected lobe importance guidance for 7 regions
    """
    def __init__(self, img_size, in_channels, feature_size, lobe_importance_file, task, use_auc=False,
                radiomics_features=0, use_relative=True, num_classes=2, current_fold=0):
        super(LobeImportanceGuidedTransformer, self).__init__()
        self.task = task
        self.num_classes = num_classes
        self.use_auc = use_auc
        self.use_relative = use_relative
        self.current_fold = current_fold


        self.lobe_importance = self._load_lobe_importance(lobe_importance_file, task)


        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,
            feature_size=feature_size,
            use_checkpoint=True,
        )


        self.lobe_attention = EnhancedLobeAttentionModule(feature_size)


        self.use_radiomics = radiomics_features > 0
        if self.use_radiomics:
            self.fusion_module = RadiomicsFusionModule(feature_size, radiomics_features)
            output_size = 256
        else:
            output_size = feature_size


        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(output_size, num_classes)
        )

    def _load_lobe_importance(self, lobe_importance_file, task):
        """Load lobe importance scores from JSON file with fold-specific data"""
        try:

            if lobe_importance_file.endswith('.json'):
                return load_radiomics_auc_weights(lobe_importance_file, task, self.current_fold)


            with open(lobe_importance_file, 'r') as f:
                lobe_data = json.load(f)


            if f"fold_{self.current_fold}" in lobe_data:

                fold_key = f"fold_{self.current_fold}"
                importance = {}
                if "lobe_aucs" in lobe_data[fold_key]:
                    importance = lobe_data[fold_key]["lobe_aucs"]


                    print(f"Fold {self.current_fold} lobe AUCs: {importance}")


                    low_auc_lobes = [lobe for lobe, auc in importance.items() if auc < 0.7]
                    if low_auc_lobes:
                        print(f"Warning: Fold {self.current_fold} has low AUC lobes: {low_auc_lobes}")

                    return importance
            else:

                importance = {}
                for lobe_num, data in lobe_data.items():
                    if task in data['outcomes']:
                        if self.use_auc:

                            importance[lobe_num] = data['outcomes'][task]['auc']
                        else:

                            importance[lobe_num] = data['outcomes'][task]['relative_importance']

                print(f"Loaded lobe importance for task {task}: {importance}")
                return importance
        except Exception as e:
            print(f"Error loading lobe importance data: {e}")

            return {str(i): 1.0 for i in range(1, 8)}

    def get_lobe_weights(self, batch_size, device, verbose=False):
        """Generate tensor of lobe weights for 7 regions with improved scaling"""

        weights = torch.ones(batch_size, 7).to(device)

        if self.lobe_importance:
            if verbose:
                print(f"Applying lobe weights for fold {self.current_fold}:")


            region_keys = ['1', '2', '3', '4', '5', 'left_lung', 'right_lung']
            region_names = ['LUL', 'LLL', 'RUL', 'RML', 'RLL', 'Left Lung', 'Right Lung']

            for i, (region_key, region_name) in enumerate(zip(region_keys, region_names)):
                if region_key in self.lobe_importance:
                    auc_value = self.lobe_importance[region_key]



                    if auc_value >= 0.5:
                        weight = 0.1 + (auc_value - 0.5) * 3.8
                    else:
                        weight = 0.1

                    weights[:, i] = weight
                    if verbose:
                        print(f"  {region_name}: AUC={auc_value:.3f} -> Weight={weight:.3f}")


        return weights

    def forward(self, x, lobe_masks=None, radiomics_features=None):
        """Forward pass with corrected lobe importance guidance for 7 regions"""
        batch_size = x.size(0)
        device = x.device


        features = self.swin_unetr(x)


        if lobe_masks is not None:

            lobe_weights = self.get_lobe_weights(batch_size, device, verbose=False)


            if lobe_masks.shape[2:] != features.shape[2:]:
                lobe_masks = F.interpolate(
                    lobe_masks.float(),
                    size=features.shape[2:],
                    mode='nearest'
                ).long()


            if lobe_masks.shape[1] == 1:

                lobe_masks_7_channel = torch.zeros(
                    (lobe_masks.size(0), 7, *lobe_masks.shape[2:]),
                    device=device
                )


                for i in range(5):
                    lobe_masks_7_channel[:, i] = (lobe_masks[:, 0] == i+1)


                lobe_masks_7_channel[:, 5] = ((lobe_masks[:, 0] == 1) | (lobe_masks[:, 0] == 2))


                lobe_masks_7_channel[:, 6] = ((lobe_masks[:, 0] == 3) | (lobe_masks[:, 0] == 4) | (lobe_masks[:, 0] == 5))

                lobe_masks = lobe_masks_7_channel


            attended_features = self.lobe_attention(features, lobe_masks, lobe_weights)
        else:
            attended_features = features


        if self.use_radiomics and radiomics_features is not None:

            pooled_features = F.adaptive_avg_pool3d(attended_features, 1)
            flattened_features = pooled_features.view(batch_size, -1)


            final_features = self.fusion_module(flattened_features, radiomics_features)


            return self.classification_head[2](final_features)
        else:

            return self.classification_head(attended_features)


def debug_lobe_weights(model, sample_batch, device):
    """Debug function to check if weights are being applied correctly"""
    model.eval()
    with torch.no_grad():
        batch_size = sample_batch["image"].size(0)


        lobe_weights = model.get_lobe_weights(batch_size, device)

        print(f"Debug - Fold {model.current_fold} lobe weights:")
        region_names = ['LUL', 'LLL', 'RUL', 'RML', 'RLL', 'Left Lung', 'Right Lung']
        region_keys = ['1', '2', '3', '4', '5', 'left_lung', 'right_lung']

        for i, (region_name, region_key) in enumerate(zip(region_names, region_keys)):
            lobe_auc = model.lobe_importance.get(region_key, 'N/A')
            weight = lobe_weights[0, i].item()
            print(f"  {region_name}: AUC={lobe_auc}, Weight={weight:.3f}")


        lobe_masks = sample_batch["lobe_mask"].to(device)
        if lobe_masks.shape[1] == 1:

            lobe_masks_7_channel = torch.zeros(
                (lobe_masks.size(0), 7, *lobe_masks.shape[2:]),
                device=device
            )


            for i in range(5):
                lobe_masks_7_channel[:, i] = (lobe_masks[:, 0] == i+1)


            lobe_masks_7_channel[:, 5] = ((lobe_masks[:, 0] == 1) | (lobe_masks[:, 0] == 2))


            lobe_masks_7_channel[:, 6] = ((lobe_masks[:, 0] == 3) | (lobe_masks[:, 0] == 4) | (lobe_masks[:, 0] == 5))

            lobe_masks = lobe_masks_7_channel


        for i, region_name in enumerate(region_names):
            mask_sum = lobe_masks[:, i].sum().item()
            mask_total = lobe_masks[:, i].numel()
            print(f"  {region_name} mask coverage: {mask_sum/mask_total*100:.1f}%")


def validate_weight_setup(model, train_loader, device, fold):
    """Validate that the weight setup is working correctly"""
    print(f"\n=== Validating Weight Setup for Fold {fold} ===")


    sample_batch = next(iter(train_loader))


    batch_size = sample_batch["image"].size(0)
    lobe_weights = model.get_lobe_weights(batch_size, device, verbose=True)

    print("=== End Weight Setup Validation ===\n")


def get_transforms(img_size=(96, 96, 96)):
    return Compose([
        LoadImaged(keys=["image", "lobe_mask"]),
        EnsureChannelFirstd(keys=["image", "lobe_mask"]),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image", "lobe_mask"], spatial_size=img_size),
        ToTensord(keys=["image", "lobe_mask"]),
    ])

def load_image_and_mask_data(labels_file, nifti_dir, lobe_masks_dir, task):
    """Load data including corresponding lobe masks"""
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
    """Prepare data dictionary for MONAI Dataset with masks"""
    return [{"image": file_path, "lobe_mask": mask_path, "label": label}
            for file_path, mask_path, label in zip(files, masks, labels)]

def load_radiomics_features(radiomics_dir, file_ids):
    """Load radiomics features for each file ID"""
    print("Loading radiomics features...")
    try:

        feature_files = {}
        for lobe_num in range(1, 6):
            file_path = os.path.join(radiomics_dir, f"lobe_{lobe_num}_features.csv")
            if os.path.exists(file_path):
                feature_files[lobe_num] = pd.read_csv(file_path)
                print(f"Loaded radiomics features for lobe {lobe_num}")

        if not feature_files:
            print("No radiomics feature files found!")
            return None


        radiomics_dict = {}
        for file_id in file_ids:
            features_per_lobe = []

            for lobe_num in range(1, 6):
                if lobe_num in feature_files:
                    df = feature_files[lobe_num]

                    row = df[df['Filename'] == file_id]

                    if not row.empty:

                        feature_cols = [col for col in df.columns
                                       if col.startswith('original_') and not col.endswith('_x') and not col.endswith('_y')]

                        if feature_cols:
                            lobe_features = row[feature_cols].values.flatten()
                            features_per_lobe.append(lobe_features)
                        else:

                            features_per_lobe.append(np.zeros(100))
                    else:

                        features_per_lobe.append(np.zeros(100))


            if features_per_lobe:
                radiomics_dict[file_id] = np.concatenate(features_per_lobe)

        return radiomics_dict
    except Exception as e:
        print(f"Error loading radiomics features: {e}")
        return None

def train_model_with_lobe_importance(task, lobe_masks_dir, radiomics_dir=None,
                                    lobe_importance_file='/home/akp4895/DeepLearning/fusion_multimodal/Step1_AdaptiveFusionNetwork/Radiomics/seven_region_predictive_power_results_validation/fold_specific/has_ILD_fold_auc_results.json',
                                    use_auc=True, use_relative=False, resume_dir=None,
                                    single_fold=None):
    """
    Train model with lobe importance guidance and optional radiomics features
    
    Args:
        task: Task name (e.g., 'has_ILD')
        lobe_masks_dir: Directory containing lobe segmentation masks
        radiomics_dir: Directory containing radiomics features (optional)
        lobe_importance_file: Path to lobe importance CSV file
        use_auc: Whether to use AUC values instead of relative importance
        use_relative: Whether to use relative importance (ignored if use_auc=True)
        resume_dir: Directory to resume training from
        single_fold: If specified, only train this fold (0-4)
    """
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
        if single_fold is not None:
            logdir = os.path.join(BASE_LOGDIR, f"{task}_7region_guided_fold_{single_fold}_{timestamp}")
        else:
            logdir = os.path.join(BASE_LOGDIR, f"{task}_7region_guided_{timestamp}")
        os.makedirs(logdir, exist_ok=True)

        config = {
            'task': task,
            'learning_rate': lr,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'num_classes': num_classes,
            'patience': patience,
            'timestamp': timestamp,
            'use_radiomics': radiomics_dir is not None,
            'lobe_importance_file': lobe_importance_file,
            'use_auc': use_auc,
            'use_relative': use_relative,
            'single_fold': single_fold,
            'regions': 7
        }
        safe_save_json(config, os.path.join(logdir, 'config.json'))


    file_list, mask_list, labels, patient_ids = load_image_and_mask_data(
        LABELS_FILE, data_path, lobe_masks_dir, task
    )


    radiomics_features = None
    radiomics_feature_size = 0
    if radiomics_dir and os.path.exists(radiomics_dir):
        radiomics_features = load_radiomics_features(radiomics_dir, [os.path.basename(f).replace('.nii.gz', '') for f in file_list])
        if radiomics_features:

            first_key = next(iter(radiomics_features))
            radiomics_feature_size = len(radiomics_features[first_key])


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
            'train_val_sample': [int(x) for x in fold_data['train_val_idx'][:5]],
            'holdout_sample': [int(x) for x in fold_data['holdout_idx'][:5]]
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


    for fold in fold_range:
        fold_start_time = time.time()
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


        num_val_samples = np.sum(val_mask)
        num_test_samples = np.sum(test_mask)
        num_train_samples = len(train_val_idx)

        print(f"Fold {fold + 1} data splits:")
        print(f"  - Training samples: {num_train_samples}")
        print(f"  - Validation samples: {num_val_samples}")
        print(f"  - Test samples: {num_test_samples}")
        print(f"  - Total holdout samples: {len(holdout_idx)} (val + test)")


        train_files_dict = prepare_data_with_masks(file_list[train_val_idx], mask_list[train_val_idx], labels[train_val_idx])
        val_files_dict = prepare_data_with_masks(holdout_files[val_mask], holdout_masks[val_mask], holdout_labels[val_mask])
        test_files_dict = prepare_data_with_masks(holdout_files[test_mask], holdout_masks[test_mask], holdout_labels[test_mask])


        transforms = get_transforms()
        train_ds = Dataset(data=train_files_dict, transform=transforms)
        val_ds = Dataset(data=val_files_dict, transform=transforms)
        test_ds = Dataset(data=test_files_dict, transform=transforms)


        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=2, prefetch_factor=2, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, prefetch_factor=2, persistent_workers=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                               num_workers=2, prefetch_factor=2, persistent_workers=True)


        model = LobeImportanceGuidedTransformer(
            img_size=(96, 96, 96),
            in_channels=1,
            feature_size=48,
            lobe_importance_file=lobe_importance_file,
            task=task,
            use_auc=use_auc,
            use_relative=use_relative,
            radiomics_features=radiomics_feature_size,
            num_classes=num_classes,
            current_fold=fold
        )


        validate_weight_setup(model, train_loader, device, fold)
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
                    inputs = batch["image"].to(device)
                    lobe_masks = batch["lobe_mask"].to(device)
                    batch_labels = batch["label"].to(device)



                    if lobe_masks.shape[1] == 1:

                        lobe_masks_7_channel = torch.zeros(
                            (lobe_masks.size(0), 7, *lobe_masks.shape[2:]),
                            device=device
                        )


                        for i in range(5):
                            lobe_masks_7_channel[:, i] = (lobe_masks[:, 0] == i+1)


                        lobe_masks_7_channel[:, 5] = ((lobe_masks[:, 0] == 1) | (lobe_masks[:, 0] == 2))


                        lobe_masks_7_channel[:, 6] = ((lobe_masks[:, 0] == 3) | (lobe_masks[:, 0] == 4) | (lobe_masks[:, 0] == 5))

                        lobe_masks = lobe_masks_7_channel


                    rad_features = None
                    if radiomics_features:

                        file_ids = [os.path.basename(file_dict["image"]).replace('.nii.gz', '')
                                   for file_dict in batch["image_meta_dict"]]


                        rad_batch = []
                        for file_id in file_ids:
                            if file_id in radiomics_features:
                                rad_batch.append(radiomics_features[file_id])
                            else:

                                rad_batch.append(np.zeros(radiomics_feature_size))

                        rad_features = torch.tensor(np.array(rad_batch), dtype=torch.float32).to(device)

                    optimizer.zero_grad()


                    outputs = model(inputs, lobe_masks, rad_features)


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


                val_metrics = evaluate_model_with_lobe_masks(
                    model, val_loader, loss_function, device, radiomics_features, radiomics_feature_size
                )
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


        test_metrics = evaluate_model_with_lobe_masks(
            model, test_loader, loss_function, device, radiomics_features, radiomics_feature_size
        )


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


        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        fold_hours = int(fold_duration // 3600)
        fold_minutes = int((fold_duration % 3600) // 60)
        fold_seconds = int(fold_duration % 60)

        print(f"\nFold {fold} completed in {fold_hours:02d}h {fold_minutes:02d}m {fold_seconds:02d}s!")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Test AUC: {test_metrics['auroc']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")


    if fold_results:
        overall_results = {
            'task': str(task),
            'number_of_folds': len(fold_results),
            'average_best_val_loss': float(np.mean([f['best_val_loss'] for f in fold_results])),
            'average_test_auc': float(np.mean([f['test_metrics']['auroc'] for f in fold_results])),
            'std_test_auc': float(np.std([f['test_metrics']['auroc'] for f in fold_results])),
            'average_test_accuracy': float(np.mean([f['test_metrics']['accuracy'] for f in fold_results])),
            'average_test_f1': float(np.mean([f['test_metrics']['f1'] for f in fold_results])),
            'fold_summaries': fold_results,
            'lobe_importance_type': 'radiomics_7_region_auc',
            'used_radiomics': radiomics_dir is not None,
            'single_fold_training': single_fold is not None,
            'regions_used': 7
        }

        safe_save_json(overall_results, os.path.join(logdir, 'overall_results.json'))

        print(f"\nTraining completed!")
        if len(fold_results) > 1:
            print(f"Average best validation loss: {overall_results['average_best_val_loss']:.4f}")
            print(f"Average test AUC: {overall_results['average_test_auc']:.4f} (±{overall_results['std_test_auc']:.4f})")
            print(f"Average test F1: {overall_results['average_test_f1']:.4f}")
        else:
            print(f"Single fold results:")
            print(f"Best validation loss: {fold_results[0]['best_val_loss']:.4f}")
            print(f"Test AUC: {fold_results[0]['test_metrics']['auroc']:.4f}")
            print(f"Test F1: {fold_results[0]['test_metrics']['f1']:.4f}")

    return overall_results if fold_results else None

def evaluate_model_with_lobe_masks(model, data_loader, loss_function, device, radiomics_features=None, radiomics_feature_size=100):
    """
    Evaluate model on given data loader with 7-channel lobe masks
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader containing validation/test data
        loss_function: Loss function to use
        device: Device to run on
        radiomics_features: Optional dictionary of radiomics features
        radiomics_feature_size: Size of radiomics feature vector (default 100)
    """
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = batch["image"].to(device)
            lobe_masks = batch["lobe_mask"].to(device)
            labels = batch["label"].to(device)


            if lobe_masks.shape[1] == 1:

                lobe_masks_7_channel = torch.zeros(
                    (lobe_masks.size(0), 7, *lobe_masks.shape[2:]),
                    device=device
                )


                for i in range(5):
                    lobe_masks_7_channel[:, i] = (lobe_masks[:, 0] == i+1)


                lobe_masks_7_channel[:, 5] = ((lobe_masks[:, 0] == 1) | (lobe_masks[:, 0] == 2))


                lobe_masks_7_channel[:, 6] = ((lobe_masks[:, 0] == 3) | (lobe_masks[:, 0] == 4) | (lobe_masks[:, 0] == 5))

                lobe_masks = lobe_masks_7_channel


            rad_features = None
            if radiomics_features:

                file_ids = [os.path.basename(file_dict["image"]).replace('.nii.gz', '')
                          for file_dict in batch["image_meta_dict"]]


                rad_batch = []
                for file_id in file_ids:
                    if file_id in radiomics_features:
                        rad_batch.append(radiomics_features[file_id])
                    else:

                        rad_batch.append(np.zeros(radiomics_feature_size))

                rad_features = torch.tensor(np.array(rad_batch), dtype=torch.float32).to(device)


            outputs = model(inputs, lobe_masks, rad_features)


            loss = loss_function(outputs, labels)

            total_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(data_loader)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Swin-UNETR model with 7-region lobe importance guidance')
    parser.add_argument('--task', type=str, required=True,
                      help='Task to train on (e.g., has_ILD, death_1, death_3, death_5, ILD_1, ILD_3, ILD_5)')
    parser.add_argument('--gpu', type=str, default='0',
                      help='GPU device number to use')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=4e-4,
                      help='Learning rate')
    parser.add_argument('--patience', type=int, default=50,
                      help='Early stopping patience')
    parser.add_argument('--max_epochs', type=int, default=50,
                      help='Maximum number of epochs')
    parser.add_argument('--lobe_masks_dir', type=str, default='/data2/akp4895/1mm_OrientedImagesSegmented',
                      help='Path to lobe segmentation masks')
    parser.add_argument('--radiomics_dir', type=str, default=None,
                      help='Path to radiomics features (optional)')
    parser.add_argument('--lobe_importance_file', type=str, default='/home/akp4895/DeepLearning/fusion_multimodal/Step1_AdaptiveFusionNetwork/Radiomics/seven_region_predictive_power_results_validation/fold_specific/has_ILD_fold_auc_results.json',
                      help='Path to 7-region lobe importance JSON file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint directory to resume training from')
    parser.add_argument('--fold', type=int, default=None,
                      help='Train only a specific fold (0-4). If not specified, trains all folds.')

    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size
    lr = args.lr
    patience = args.patience
    max_epochs = args.max_epochs
    LOBE_MASKS_PATH = args.lobe_masks_dir

    print(f"\nStarting 7-region lobe-guided training for task: {args.task}")
    print(f"Using GPU: {args.gpu}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Early stopping patience: {patience}")
    print(f"Maximum epochs: {max_epochs}")
    print(f"Using 7-region radiomics AUC values from: {args.lobe_importance_file}")

    if args.fold is not None:
        print(f"Training single fold: {args.fold}")

    try:
        train_model_with_lobe_importance(
            task=args.task,
            lobe_masks_dir=args.lobe_masks_dir,
            radiomics_dir=args.radiomics_dir,
            lobe_importance_file=args.lobe_importance_file,
            use_auc=True,
            use_relative=False,
            resume_dir=args.resume,
            single_fold=args.fold
        )
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
