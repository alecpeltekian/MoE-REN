#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import re
import argparse
warnings.filterwarnings('ignore')

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

def get_patient_id(id_str):
    match = re.search(r'(\d{4})', id_str)
    return match.group(1) if match else None

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

class RegionPredictivePowerAnalyzer:
    def __init__(self, radiomics_dir, labels_path, nifti_dir, output_dir, expert_config):
        self.radiomics_dir = radiomics_dir
        self.labels_path = labels_path
        self.nifti_dir = nifti_dir
        self.output_dir = output_dir
        self.expert_config = expert_config
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fold_specific"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "validation_aucs"), exist_ok=True)
        self.labels_df = pd.read_csv(labels_path)
        print(f"Loaded labels for {len(self.labels_df)} patients")
        self.outcome_cols = ['has_ILD', 'death_1', 'death_3', 'death_5', 'ILD_1', 'ILD_3', 'ILD_5']
        self.radiomics_prefixes = [
            'original_shape_', 'original_firstorder_', 'original_glcm_', 
            'original_gldm_', 'original_glrlm_', 'original_glszm_', 'original_ngtdm_'
        ]
        self.region_features = self._load_region_features()
    
    def _load_region_features(self):
        region_features = {}
        for expert_name, expert_type in zip(self.expert_config['expert_names'], self.expert_config['expert_types']):
            if expert_type == 'lobe':
                lobe_num = int(expert_name.split('_')[1])
                possible_paths = [
                    os.path.join(self.radiomics_dir, f"lobe_{lobe_num}", "radiomics_features.csv"),
                    os.path.join(self.radiomics_dir, f"lobe_{lobe_num}", f"lobe_{lobe_num}_features.csv"),
                    os.path.join(self.radiomics_dir, "lobe_features", f"lobe_{lobe_num}", "radiomics_features.csv"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        print(f"Loaded {len(df)} entries for {expert_name} from {path}")
                        region_features[expert_name] = df
                        break
            elif expert_type == 'lung':
                lung_key = expert_name.lower().replace('_', '_')
                possible_paths = [
                    os.path.join(self.radiomics_dir, lung_key, "radiomics_features.csv"),
                    os.path.join(self.radiomics_dir, f"{lung_key}_features", f"{lung_key}_features.csv"),
                    os.path.join(self.radiomics_dir, "combined_lung_features", "left_right_lung_features.csv"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        if 'Region' in df.columns:
                            region_name = "Left Lung" if "left" in expert_name.lower() else "Right Lung"
                            df = df[df['Region'] == region_name].copy()
                        print(f"Loaded {len(df)} entries for {expert_name} from {path}")
                        region_features[expert_name] = df
                        break
        if not region_features:
            raise ValueError("No valid region feature files could be loaded")
        return region_features
    
    def load_image_and_mask_data_for_splits(self, task):
        print(f"Reading labels file for task: {task}")
        df = pd.read_csv(self.labels_path)
        df = df[df[task].notna()]
        df = df[df[task] != -1]
        file_list = []
        labels = []
        patient_ids = []
        for patient_dir in sorted(os.listdir(self.nifti_dir)):
            patient_path = os.path.join(self.nifti_dir, patient_dir)
            if os.path.isdir(patient_path):
                for filename in sorted(os.listdir(patient_path)):
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
        print(f"Total matched files for {task}: {len(file_list)}")
        return file_list, labels, patient_ids
    
    def _exclude_metadata_columns(self, df):
        radiomic_features = []
        for col in df.columns:
            if any(col.startswith(prefix) for prefix in self.radiomics_prefixes):
                radiomic_features.append(col)
        if radiomic_features:
            return radiomic_features
        metadata_cols = [
            'Patient', 'PatientID', 'Filename', 'Image', 'Mask', 'error', 
            'Region', 'RegionType', 'LungLabel', 'ScanID', 'case_number', 
            'procedure_end_dt', 'CT_desc', 'event_date', 'scan_id', 'index'
        ]
        outcome_cols = self.outcome_cols + ['death_date', 'ILD_date']
        merge_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
        exclude_cols = metadata_cols + outcome_cols + merge_cols
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def analyze_fold_specific_predictive_power(self, task, k_folds=5):
        print(f"\nAnalyzing fold-specific predictive power for: {task}")
        file_list, labels, patient_ids = self.load_image_and_mask_data_for_splits(task)
        scan_ids = [os.path.basename(f).replace('.nii.gz', '') for f in file_list]
        valid_filenames = set([os.path.basename(f) for f in file_list])
        print(f"Valid filenames in NIFTI dir: {len(valid_filenames)}")
        merged_features = {}
        for expert_name, features_df in self.region_features.items():
            if 'Filename' not in features_df.columns:
                print(f"{expert_name} features missing Filename column, cannot merge with labels")
                continue
            features_df = features_df.copy()
            features_df['Filename_normalized'] = features_df['Filename'].apply(
                lambda x: re.sub(r'^(\d+)_\1_', r'\1_', x) if isinstance(x, str) else x
            )
            merged = None
            if 'ScanID' in self.labels_df.columns:
                features_df['ScanID'] = features_df['Filename_normalized'].apply(
                    lambda x: x.replace('.nii.gz', '').replace('.nii', '') if isinstance(x, str) else x
                )
                merged = pd.merge(features_df, self.labels_df, on='ScanID', how='inner', suffixes=('', '_labels'))
            else:
                merged = pd.merge(features_df, self.labels_df, left_on='Filename_normalized', right_on='Filename', how='inner', suffixes=('', '_labels'))
            if merged is None or len(merged) == 0:
                print(f"Warning: No matches found for {expert_name}. Trying direct Filename match...")
                merged = pd.merge(features_df, self.labels_df, on='Filename', how='inner', suffixes=('', '_labels'))
            merged = merged[merged[task].notna()]
            merged = merged[merged[task] != -1]
            normalized_valid_filenames = set([re.sub(r'^(\d+)_\1_', r'\1_', f) for f in valid_filenames])
            merged = merged[merged['Filename_normalized'].isin(normalized_valid_filenames)]
            print(f"{expert_name}: Merged {len(merged)} records with labels for {task}")
            merged_features[expert_name] = merged
        consistent_folds = create_consistent_folds(file_list, labels, patient_ids, scan_ids, k_folds=k_folds, seed=42)
        fold_auc_results = {}
        for fold_idx, fold_data in enumerate(consistent_folds):
            print(f"\nAnalyzing Fold {fold_idx + 1}/{k_folds}")
            holdout_idx = fold_data['holdout_idx']
            val_mask = fold_data['val_mask']
            val_idx = holdout_idx[val_mask]
            region_aucs = {}
            for expert_name, features_df in merged_features.items():
                val_filenames = [os.path.basename(file_list[i]) for i in val_idx]
                if 'Filename_normalized' in features_df.columns:
                    val_data = features_df[features_df['Filename_normalized'].isin(val_filenames)]
                else:
                    val_data = features_df[features_df['Filename'].isin(val_filenames)]
                print(f"  Fold {fold_idx + 1}, {expert_name}: {len(val_data)} validation samples")
                if len(val_data) < 10:
                    print(f"  Insufficient validation samples for {expert_name}: {len(val_data)}")
                    region_aucs[expert_name] = 0.5
                    continue
                unique_labels = val_data[task].unique()
                if len(unique_labels) < 2:
                    print(f"  Only one class in validation set for {expert_name}")
                    region_aucs[expert_name] = 0.5
                    continue
                feature_cols = self._exclude_metadata_columns(val_data)
                if not feature_cols:
                    print(f"  No features found for {expert_name}")
                    region_aucs[expert_name] = 0.5
                    continue
                X = val_data[feature_cols].values
                y = val_data[task].astype(int).values
                if np.isnan(X).any() or np.isinf(X).any():
                    X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model = xgb.XGBClassifier(
                        n_estimators=50,
                        learning_rate=0.1,
                        max_depth=3,
                        use_label_encoder=False,
                        eval_metric='logloss',
                        random_state=42
                    )
                    model.fit(X_train_scaled, y_train)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                    region_aucs[expert_name] = float(auc)
                    print(f"    {expert_name} AUC: {auc:.3f}")
                except Exception as e:
                    print(f"    Error calculating AUC for {expert_name}: {e}")
                    region_aucs[expert_name] = 0.5
            if self.expert_config['num_experts'] == 2:
                fold_auc_results[f"fold_{fold_idx}"] = {
                    'region_aucs': {
                        'left_lung': region_aucs.get('Left_Lung', 0.5),
                        'right_lung': region_aucs.get('Right_Lung', 0.5)
                    },
                    'fold_info': {
                        'val_size': len(val_idx),
                        'holdout_size': len(holdout_idx),
                        'val_samples': len(val_mask[val_mask]),
                        'test_samples': len(fold_data['test_mask'][fold_data['test_mask']])
                    }
                }
            elif self.expert_config['num_experts'] == 5:
                lobe_names_map = {
                    'Lobe_1_LUL': '1',
                    'Lobe_2_LLL': '2',
                    'Lobe_3_RUL': '3',
                    'Lobe_4_RML': '4',
                    'Lobe_5_RLL': '5'
                }
                fold_auc_results[f"fold_{fold_idx}"] = {
                    'lobe_aucs': {
                        lobe_names_map.get(name, str(i+1)): region_aucs.get(name, 0.5)
                        for i, name in enumerate(self.expert_config['expert_names'])
                    },
                    'fold_info': {
                        'val_size': len(val_idx),
                        'holdout_size': len(holdout_idx),
                        'val_samples': len(val_mask[val_mask]),
                        'test_samples': len(fold_data['test_mask'][fold_data['test_mask']])
                    }
                }
            elif self.expert_config['num_experts'] == 7:
                lobe_names_map = {
                    'Lobe_1_LUL': '1',
                    'Lobe_2_LLL': '2',
                    'Lobe_3_RUL': '3',
                    'Lobe_4_RML': '4',
                    'Lobe_5_RLL': '5'
                }
                lobe_aucs = {}
                region_aucs_dict = {}
                for name in self.expert_config['expert_names']:
                    if name.startswith('Lobe_'):
                        lobe_key = lobe_names_map.get(name, name.split('_')[1])
                        lobe_aucs[lobe_key] = region_aucs.get(name, 0.5)
                    elif name == 'Left_Lung':
                        region_aucs_dict['left_lung'] = region_aucs.get(name, 0.5)
                    elif name == 'Right_Lung':
                        region_aucs_dict['right_lung'] = region_aucs.get(name, 0.5)
                fold_auc_results[f"fold_{fold_idx}"] = {
                    'lobe_aucs': lobe_aucs,
                    'region_aucs': region_aucs_dict,
                    'fold_info': {
                        'val_size': len(val_idx),
                        'holdout_size': len(holdout_idx),
                        'val_samples': len(val_mask[val_mask]),
                        'test_samples': len(fold_data['test_mask'][fold_data['test_mask']])
                    }
                }
        output_file = os.path.join(self.output_dir, "fold_specific", f"{task}_fold_auc_results.json")
        safe_save_json(fold_auc_results, output_file)
        print(f"\nFold-specific analysis completed for {task}")
        return fold_auc_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    task = config['task']
    num_experts = config['num_experts']
    expert_key = f"{num_experts}_experts"
    expert_config = config['expert_config'][expert_key]
    expert_config['num_experts'] = num_experts
    
    data_paths = config['data_paths']
    output_paths = config['output_paths']
    
    radiomics_dir = output_paths['radiomics_features_dir']
    labels_path = data_paths['labels_file']
    nifti_dir = data_paths['nifti_dir']
    output_dir = output_paths['importance_results_dir']
    
    analyzer = RegionPredictivePowerAnalyzer(
        radiomics_dir,
        labels_path,
        nifti_dir,
        output_dir,
        expert_config
    )
    
    fold_results = analyzer.analyze_fold_specific_predictive_power(task)
    print(f"\n✅ Importance weight extraction completed for {task}")

if __name__ == "__main__":
    main()

