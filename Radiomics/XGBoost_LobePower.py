
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import json
import re
warnings.filterwarnings('ignore')

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
    """Safely save JSON by writing to temporary file first and ensuring all types are serializable"""
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

def load_validation_aucs(results_dir, task):
    """
    Load validation AUCs from saved results for easy access
    Returns a dictionary with fold-specific and summary statistics
    """
    fold_results_path = os.path.join(results_dir, "fold_specific", f"{task}_fold_auc_results.json")
    avg_results_path = os.path.join(results_dir, "fold_specific", f"{task}_average_fold_aucs.json")

    if not os.path.exists(fold_results_path):
        raise FileNotFoundError(f"Fold results not found: {fold_results_path}")


    with open(fold_results_path, 'r') as f:
        fold_results = json.load(f)


    avg_results = None
    if os.path.exists(avg_results_path):
        with open(avg_results_path, 'r') as f:
            avg_results = json.load(f)


    val_aucs = {
        'by_fold': {},
        'by_lobe': {},
        'summary': avg_results
    }


    for fold_name, fold_data in fold_results.items():
        fold_idx = int(fold_name.split('_')[1])
        val_aucs['by_fold'][fold_idx] = fold_data['lobe_aucs']


    lobe_names = {
        '1': "Left Upper Lobe",
        '2': "Left Lower Lobe",
        '3': "Right Upper Lobe",
        '4': "Right Middle Lobe",
        '5': "Right Lower Lobe"
    }

    for lobe_num in lobe_names.keys():
        val_aucs['by_lobe'][lobe_num] = []
        for fold_data in fold_results.values():
            if lobe_num in fold_data['lobe_aucs']:
                val_aucs['by_lobe'][lobe_num].append(fold_data['lobe_aucs'][lobe_num])

    return val_aucs


def get_patient_id(id_str):
    """Extract patient ID from filename - EXACT COPY FROM MODELING"""
    match = re.search(r'(\d{4})', id_str)
    return match.group(1) if match else None

def create_consistent_folds(file_list, labels, patient_ids, scan_ids=None, k_folds=5, seed=42):
    """Create consistent folds for cross-validation with deterministic splits - EXACT COPY FROM MODELING"""
    from sklearn.model_selection import GroupKFold


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

class LobePredictivePowerAnalyzer:
    def __init__(self, radiomics_dir, labels_path, nifti_dir, output_dir="lobe_predictive_power_results"):
        """Initialize analyzer - now requires nifti_dir to match modeling exactly"""
        self.radiomics_dir = radiomics_dir
        self.labels_path = labels_path
        self.nifti_dir = nifti_dir
        self.output_dir = output_dir


        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fold_specific"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "validation_aucs"), exist_ok=True)


        self.labels_df = pd.read_csv(labels_path)
        print(f"Loaded labels for {len(self.labels_df)} patients")


        self.outcome_cols = ['has_ILD', 'death_1', 'death_3', 'death_5', 'ILD_1', 'ILD_3', 'ILD_5']


        self.lobe_names = {
            '1': "Left Upper Lobe",
            '2': "Left Lower Lobe",
            '3': "Right Upper Lobe",
            '4': "Right Middle Lobe",
            '5': "Right Lower Lobe"
        }


        self.radiomics_prefixes = [
            'original_shape_', 'original_firstorder_', 'original_glcm_',
            'original_gldm_', 'original_glrlm_', 'original_glszm_', 'original_ngtdm_'
        ]


        self.lobe_features = self._load_lobe_features()

    def _load_lobe_features(self):
        """Load feature data for each lobe"""
        lobe_features = {}

        for lobe_num in range(1, 6):
            file_path = os.path.join(self.radiomics_dir, f"lobe_{lobe_num}_features.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Loaded {len(df)} entries for {self.lobe_names[str(lobe_num)]} (Lobe {lobe_num})")
                lobe_features[str(lobe_num)] = df
            else:
                print(f"File not found: {file_path}")

        if not lobe_features:
            raise ValueError("No valid lobe feature files could be loaded")

        return lobe_features

    def load_image_and_mask_data_for_splits(self, task):
        """
        Load data in the EXACT SAME WAY as modeling_radiomics.py
        This ensures identical file lists and ordering
        """
        print(f"Reading labels file for task: {task} - MATCHING MODELING LOGIC")
        df = pd.read_csv(self.labels_path)
        df = df[df[task].notna()]
        df = df[df[task] != -1]

        file_list = []
        labels = []
        patient_ids = []

        for patient_dir in os.listdir(self.nifti_dir):
            patient_path = os.path.join(self.nifti_dir, patient_dir)
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

        print(f"Total matched files for {task}: {len(file_list)}")
        return file_list, labels, patient_ids

    def _exclude_metadata_columns(self, df):
        """Identify and exclude metadata columns from feature set."""

        radiomic_features = []
        for col in df.columns:
            if any(col.startswith(prefix) for prefix in self.radiomics_prefixes):
                radiomic_features.append(col)

        if radiomic_features:
            return radiomic_features


        metadata_cols = [
            'Patient', 'PatientID', 'Filename', 'Image', 'Mask', 'error',
            'Lobe', 'LobeNumber', 'ScanID', 'case_number', 'procedure_end_dt',
            'CT_desc', 'event_date', 'scan_id', 'index'
        ]

        outcome_cols = self.outcome_cols + ['death_date', 'ILD_date']
        merge_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
        exclude_cols = metadata_cols + outcome_cols + merge_cols
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        return feature_cols

    def save_validation_aucs_summary(self, fold_auc_results, task):
        """
        Save validation AUCs in multiple formats for easy access
        """

        val_auc_data = []

        for fold_name, fold_data in fold_auc_results.items():
            fold_idx = int(fold_name.split('_')[1])
            for lobe_num, auc in fold_data['lobe_aucs'].items():
                val_auc_data.append({
                    'task': task,
                    'fold': fold_idx,
                    'lobe_num': lobe_num,
                    'lobe_name': self.lobe_names[lobe_num],
                    'validation_auc': auc
                })


        val_auc_df = pd.DataFrame(val_auc_data)
        csv_path = os.path.join(self.output_dir, "validation_aucs", f"{task}_validation_aucs.csv")
        val_auc_df.to_csv(csv_path, index=False)
        print(f"Validation AUCs saved to: {csv_path}")


        pivot_df = val_auc_df.pivot_table(index='fold', columns='lobe_name', values='validation_auc')
        pivot_path = os.path.join(self.output_dir, "validation_aucs", f"{task}_validation_aucs_pivot.csv")
        pivot_df.to_csv(pivot_path)
        print(f"Validation AUCs pivot table saved to: {pivot_path}")


        summary_stats = val_auc_df.groupby('lobe_name')['validation_auc'].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).round(4)
        summary_path = os.path.join(self.output_dir, "validation_aucs", f"{task}_validation_aucs_summary.csv")
        summary_stats.to_csv(summary_path)
        print(f"Validation AUCs summary statistics saved to: {summary_path}")

        return val_auc_df, pivot_df, summary_stats

    def analyze_fold_specific_predictive_power(self, task, k_folds=5):
        """
        Analyze predictive power using EXACT same splits as modeling_radiomics.py
        """
        print(f"\nAnalyzing fold-specific predictive power for: {task}")
        print("Using EXACT same split logic as modeling_radiomics.py")


        file_list, labels, patient_ids = self.load_image_and_mask_data_for_splits(task)


        scan_ids = [os.path.basename(f).replace('.nii.gz', '') for f in file_list]


        valid_filenames = set([os.path.basename(f) for f in file_list])
        print(f"Valid filenames in NIFTI dir: {len(valid_filenames)}")


        merged_features = {}
        for lobe_num, features_df in self.lobe_features.items():
            if 'Filename' not in features_df.columns:
                print(f"Lobe {lobe_num} features missing Filename column, cannot merge with labels")
                continue


            merged = pd.merge(features_df, self.labels_df, on='Filename', how='inner')


            merged = merged[merged[task].notna()]
            merged = merged[merged[task] != -1]


            merged = merged[merged['Filename'].isin(valid_filenames)]

            print(f"Lobe {lobe_num}: Merged {len(merged)} records with labels for {task}")
            merged_features[lobe_num] = merged


        consistent_folds = create_consistent_folds(file_list, labels, patient_ids, scan_ids, k_folds=k_folds, seed=42)


        fold_auc_results = {}


        for fold_idx, fold_data in enumerate(consistent_folds):
            print(f"\nAnalyzing Fold {fold_idx + 1}/{k_folds}")


            holdout_idx = fold_data['holdout_idx']
            val_mask = fold_data['val_mask']


            val_idx = holdout_idx[val_mask]


            lobe_aucs = {}

            for lobe_num, features_df in merged_features.items():

                val_filenames = [os.path.basename(file_list[i]) for i in val_idx]
                val_data = features_df[features_df['Filename'].isin(val_filenames)]

                print(f"  Fold {fold_idx + 1}, Lobe {lobe_num}: {len(val_data)} validation samples")


                if len(val_data) < 10:
                    print(f"  Insufficient validation samples for lobe {lobe_num}: {len(val_data)}")
                    lobe_aucs[lobe_num] = 0.5
                    continue


                unique_labels = val_data[task].unique()
                if len(unique_labels) < 2:
                    print(f"  Only one class in validation set for lobe {lobe_num}")
                    lobe_aucs[lobe_num] = 0.5
                    continue


                feature_cols = self._exclude_metadata_columns(val_data)
                if not feature_cols:
                    print(f"  No features found for lobe {lobe_num}")
                    lobe_aucs[lobe_num] = 0.5
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
                    lobe_aucs[lobe_num] = float(auc)

                    print(f"    Lobe {lobe_num} AUC: {auc:.3f}")

                except Exception as e:
                    print(f"    Error calculating AUC for lobe {lobe_num}: {e}")
                    lobe_aucs[lobe_num] = 0.5


            fold_auc_results[f"fold_{fold_idx}"] = {
                'lobe_aucs': lobe_aucs,
                'fold_info': {
                    'val_size': len(val_idx),
                    'holdout_size': len(holdout_idx),
                    'val_samples': len(val_mask[val_mask]),
                    'test_samples': len(fold_data['test_mask'][fold_data['test_mask']])
                }
            }


        output_file = os.path.join(self.output_dir, "fold_specific", f"{task}_fold_auc_results.json")
        safe_save_json(fold_auc_results, output_file)


        val_auc_df, pivot_df, summary_stats = self.save_validation_aucs_summary(fold_auc_results, task)


        self._plot_fold_auc_distributions(fold_auc_results, task)


        average_aucs = {}
        for lobe_num in self.lobe_names.keys():
            aucs = []
            for fold_data in fold_auc_results.values():
                if lobe_num in fold_data['lobe_aucs']:
                    aucs.append(fold_data['lobe_aucs'][lobe_num])

            if aucs:
                average_aucs[lobe_num] = {
                    'mean': np.mean(aucs),
                    'std': np.std(aucs),
                    'min': np.min(aucs),
                    'max': np.max(aucs)
                }


        avg_output_file = os.path.join(self.output_dir, "fold_specific", f"{task}_average_fold_aucs.json")
        safe_save_json(average_aucs, avg_output_file)


        splits_info = []
        for fold_idx, fold_data in enumerate(consistent_folds):
            splits_info.append({
                'fold': int(fold_idx),
                'train_val_count': int(len(fold_data['train_val_idx'])),
                'holdout_count': int(len(fold_data['holdout_idx'])),
                'val_count': int(sum(fold_data['val_mask'])),
                'test_count': int(sum(fold_data['test_mask']))
            })

        splits_file = os.path.join(self.output_dir, "fold_specific", f"{task}_fold_splits_info.json")
        safe_save_json(splits_info, splits_file)

        print(f"\nFold splits info saved to {splits_file}")
        print("These should match exactly with modeling_radiomics.py fold_splits.json")


        print(f"\nValidation AUC Summary for {task}:")
        print(summary_stats)

        return fold_auc_results, average_aucs, val_auc_df

    def _plot_fold_auc_distributions(self, fold_auc_results, task):
        """Create visualization of AUC distributions across folds for each lobe"""

        data_for_plot = []

        for fold_name, fold_data in fold_auc_results.items():
            fold_idx = int(fold_name.split('_')[1])
            for lobe_num, auc in fold_data['lobe_aucs'].items():
                data_for_plot.append({
                    'Fold': fold_idx + 1,
                    'Lobe': self.lobe_names[lobe_num],
                    'AUC': auc
                })

        df_plot = pd.DataFrame(data_for_plot)


        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_plot, x='Lobe', y='AUC', hue='Fold', palette='Set3')
        plt.title(f'AUC Distribution Across Folds for {task}')
        plt.ylabel('ROC AUC')
        plt.xlabel('Lung Lobe')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Fold', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()


        plt.savefig(os.path.join(self.output_dir, "plots", f"{task}_fold_auc_distributions.png"), dpi=300)
        plt.close()


        plt.figure(figsize=(12, 8))

        for lobe_num, lobe_name in self.lobe_names.items():
            lobe_data = df_plot[df_plot['Lobe'] == lobe_name]
            folds = lobe_data['Fold'].values
            aucs = lobe_data['AUC'].values
            plt.plot(folds, aucs, marker='o', label=lobe_name, linewidth=2, markersize=8)

        plt.xlabel('Fold')
        plt.ylabel('ROC AUC')
        plt.title(f'AUC Trends Across Folds for {task}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.5, 5.5)
        plt.xticks(range(1, 6))
        plt.tight_layout()


        plt.savefig(os.path.join(self.output_dir, "plots", f"{task}_fold_auc_trends.png"), dpi=300)
        plt.close()


        pivot_data = df_plot.pivot_table(index='Fold', columns='Lobe', values='AUC')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'ROC AUC'})
        plt.title(f'AUC Heatmap Across Folds for {task}')
        plt.tight_layout()


        plt.savefig(os.path.join(self.output_dir, "plots", f"{task}_fold_auc_heatmap.png"), dpi=300)
        plt.close()



def load_validation_aucs_from_results(results_dir, task):
    """
    Convenience function to load validation AUCs from saved results
    """
    return load_validation_aucs(results_dir, task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze fold-specific predictive power using exact same splits as modeling')
    parser.add_argument('--radiomics_dir', type=str, default='../../radiomics/radiomics_output/lobe_features',
                       help='Directory containing lobe feature CSV files')
    parser.add_argument('--labels_path', type=str, default='../../radiomics/filtered_radiomics_output/labels.csv',
                       help='Path to labels CSV file')
    parser.add_argument('--nifti_dir', type=str, default='/data2/akp4895/MultipliedImagesClean',
                       help='Directory containing NIFTI files (must match modeling)')
    parser.add_argument('--output_dir', type=str, default='1_lobe_predictive_power_results_validation',
                       help='Directory to save analysis results')
    parser.add_argument('--task', type=str, required=True,
                       help='Specific task to analyze')

    args = parser.parse_args()


    analyzer = LobePredictivePowerAnalyzer(
        args.radiomics_dir,
        args.labels_path,
        args.nifti_dir,
        args.output_dir
    )


    fold_results, avg_results, val_auc_df = analyzer.analyze_fold_specific_predictive_power(args.task)
    print(f"\nFold-specific analysis completed for {args.task}")
    print("Results saved in format compatible with modeling_radiomics.py")


    print(f"\nValidation AUCs DataFrame shape: {val_auc_df.shape}")
    print("\nExample usage:")
    print("# Load validation AUCs from saved results:")
    print(f"val_aucs = load_validation_aucs_from_results('{args.output_dir}', '{args.task}')")
    print("# Access fold 0 results: val_aucs['by_fold'][0]")
    print("# Access lobe 1 results across all folds: val_aucs['by_lobe']['1']")
