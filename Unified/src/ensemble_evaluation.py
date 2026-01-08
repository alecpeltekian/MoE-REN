

import os
import sys
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from collections import Counter
import argparse
from typing import Dict, List, Optional, Union
import importlib.util


from ensemble_gating import UnifiedMixtureOfExpertsEnsemble


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
    print("❌ Could not import data processing functions")
    sys.exit(1)

from monai.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class UnifiedEnsembleEvaluator:
    
    def __init__(self, config_path: Optional[str] = None):
        import json
        from pathlib import Path
        
        if not config_path or not Path(config_path).exists():
            raise ValueError("config_path is required and must point to an existing config file")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        output_config = config.get('output', {})
        
        if not output_config.get('mlp_expert_weights_results') or not output_config.get('unified_vit_results') or not output_config.get('unified_mamba_results'):
            raise ValueError("Missing required output paths in config file")
        
        self.model_configs = {
            'cnn': {
                'results_dir': output_config['mlp_expert_weights_results']
            },
            'vit': {
                'results_dir': output_config['unified_vit_results']
            },
            'mamba': {
                'results_dir': output_config['unified_mamba_results']
            }
        }
    
    def create_ensemble(self, model_type: str, ensemble_method: str = "weighted_average", 
                       weight_strategy: str = "val_auc"):
        results_dir = self.model_configs[model_type]['results_dir']
        return UnifiedMixtureOfExpertsEnsemble(
            model_type=model_type,
            results_dir=results_dir,
            ensemble_method=ensemble_method,
            weight_strategy=weight_strategy
        )
    
    def run_single_evaluation(self, task: str, model_type: str, fold: int, 
                                nifti_dir: str, lobe_masks_dir: str, labels_file: str,
                                ensemble_methods: List[str] = None, 
                                weight_strategy: str = "val_auc",
                                compare_strategies: bool = False):
            
            print(f"\n{'='*80}")
            print(f"EVALUATING {model_type.upper()} MODEL - TASK: {task}, FOLD: {fold}")
            print(f"{'='*80}")
            

            if ensemble_methods is None:
                ensemble_methods = ["simple_average", "weighted_average"]
            

            file_list, mask_list, labels, patient_ids = load_image_and_mask_data(
                labels_file, nifti_dir, lobe_masks_dir, task
            )
            
            k_folds = 5
            consistent_folds = create_consistent_folds(file_list, labels, patient_ids, k_folds=k_folds, seed=42)
            
            if fold >= len(consistent_folds):
                raise ValueError(f"Fold index {fold} out of range. Available folds: 0-{len(consistent_folds)-1}")
            
            fold_data = consistent_folds[fold]
            

            holdout_idx = fold_data['holdout_idx']
            test_mask = fold_data['test_mask']
            
            holdout_files = np.array(file_list)[holdout_idx]
            holdout_masks = np.array(mask_list)[holdout_idx]
            holdout_labels = np.array(labels)[holdout_idx]
            
            test_files = holdout_files[test_mask]
            test_masks = holdout_masks[test_mask]
            test_labels = holdout_labels[test_mask]
            
            print(f"Test set size: {len(test_files)}")
            print(f"Test label distribution: {Counter(test_labels)}")
            

            test_data = prepare_data_dicts(test_files, test_masks, test_labels)
            transforms = get_transforms()
            test_dataset = Dataset(data=test_data, transform=transforms)
            test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
            

            results_dir = self.model_configs[model_type]['results_dir']
            

            model_dir = Path(results_dir) / task / f"fold_{fold}"
            if not model_dir.exists():
                print(f"❌ Model directory not found: {model_dir}")
                return None
            
            expert_files = list(model_dir.glob("*_final_model.pth"))
            if len(expert_files) == 0:
                print(f"❌ No expert model files found in {model_dir}")
                return None
            
            print(f"✅ Found {len(expert_files)} {model_type.upper()} expert model files")
            
            if compare_strategies:

                return self.run_weighting_strategy_comparison(
                    task, model_type, fold, nifti_dir, lobe_masks_dir, labels_file
                )
            else:

                return self.run_standard_evaluation(
                    task, model_type, fold, test_loader, holdout_files, holdout_masks, 
                    holdout_labels, fold_data, ensemble_methods, weight_strategy
                )
    
    def run_standard_evaluation(self, task: str, model_type: str, fold: int,
                               test_loader, holdout_files, holdout_masks, holdout_labels,
                               fold_data, ensemble_methods: List[str], weight_strategy: str):
        
        results_dir = self.model_configs[model_type]['results_dir']
        
        all_results = {}
        
        for ensemble_method in ensemble_methods:
            print(f"\n{'='*60}")
            print(f"Testing {model_type.upper()} Ensemble Method: {ensemble_method}")
            print(f"{'='*60}")
            

            ensemble = self.create_ensemble(
                model_type=model_type,
                ensemble_method=ensemble_method,
                weight_strategy=weight_strategy
            )
            
            if ensemble_method == "weighted_average":
                ensemble.explain_weighting_strategy()
            
            ensemble.load_expert_models(task, fold)
            
            if len(ensemble.expert_models) == 0:
                print(f"No {model_type.upper()} expert models found for {ensemble_method}")
                continue
            

            test_predictions = {}
            true_labels = []
            
            print(f"Getting {model_type.upper()} expert predictions on test data...")
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    image = batch["image"].to(ensemble.device)
                    mask = batch["mask"].to(ensemble.device)
                    target = batch["label"]
                    
                    expert_preds = ensemble.get_expert_predictions(image, mask)
                    
                    for key, preds in expert_preds.items():
                        if key not in test_predictions:
                            test_predictions[key] = []
                        test_predictions[key].extend(preds.tolist())
                    
                    true_labels.extend(target.tolist())
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
            

            for key in test_predictions:
                test_predictions[key] = np.array(test_predictions[key])
            true_labels = np.array(true_labels)
            
            print(f"Collected predictions from {len(test_predictions)} {model_type.upper()} experts")
            print(f"Total test samples: {len(true_labels)}")
            

            if ensemble_method == "learned_ensemble":
                print(f"Training {model_type.upper()} meta-learner on validation data...")
                
                val_mask = fold_data['val_mask']
                val_files = holdout_files[val_mask]
                val_masks = holdout_masks[val_mask]
                val_labels = holdout_labels[val_mask]
                
                val_data = prepare_data_dicts(val_files, val_masks, val_labels)
                transforms = get_transforms()
                val_dataset = Dataset(data=val_data, transform=transforms)
                val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
                
                val_predictions = {}
                val_true_labels = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        image = batch["image"].to(ensemble.device)
                        mask = batch["mask"].to(ensemble.device)
                        target = batch["label"]
                        
                        expert_preds = ensemble.get_expert_predictions(image, mask)
                        for key, preds in expert_preds.items():
                            if key not in val_predictions:
                                val_predictions[key] = []
                            val_predictions[key].extend(preds.tolist())
                        
                        val_true_labels.extend(target.tolist())
                
                for key in val_predictions:
                    val_predictions[key] = np.array(val_predictions[key])
                val_true_labels = np.array(val_true_labels)
                
                ensemble.train_meta_learner(val_predictions, val_true_labels, meta_model="logistic")
            

            results = ensemble.evaluate_ensemble(
                test_predictions, true_labels, 
                save_results=True, 
                output_dir=f"{results_dir}/ensemble_results/{task}/fold_{fold}"
            )
            

            fig = ensemble.plot_ensemble_comparison(test_predictions, true_labels)
            

            os.makedirs(f"{results_dir}/ensemble_results/{task}/fold_{fold}", exist_ok=True)
            plot_filename = f"{model_type}_ensemble_comparison_{ensemble_method}_{weight_strategy}.png"
            fig.savefig(f"{results_dir}/ensemble_results/{task}/fold_{fold}/{plot_filename}", 
                       dpi=300, bbox_inches='tight')
            

            ensemble.save_ensemble_model(task, fold, output_dir=f"{model_type}_ensemble_models/{task}")
            
            all_results[ensemble_method] = results
            

            print(f"\nDetailed {model_type.upper()} Results for {ensemble_method} with {weight_strategy}:")
            print("-" * 40)
            ensemble_preds = ensemble.ensemble_predict(test_predictions)
            ensemble_binary = (ensemble_preds >= 0.5).astype(int)
            
            print(f"\n{model_type.upper()} Classification Report:")
            print(classification_report(true_labels, ensemble_binary, 
                                      target_names=['No ILD', 'Has ILD'] if task == 'has_ILD' else ['Class 0', 'Class 1']))
        
        return all_results
    
    def run_weighting_strategy_comparison(self, task: str, model_type: str, fold: int,
                                        nifti_dir: str, lobe_masks_dir: str, labels_file: str):
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE {model_type.upper()} WEIGHTING STRATEGY COMPARISON")
        print(f"{'='*80}")
        print("This experiment compares validation vs test-based weighting to demonstrate data leakage effects")
        print("⚠️ Test-based results are optimistically biased and should NOT be used for model selection!")
        print(f"{'='*80}\n")
        
        strategies = [
            ("simple_average", "uniform"),
            ("weighted_average", "val_auc"),
            ("weighted_average", "val_accuracy"),
            ("weighted_average", "test_auc"),
            ("weighted_average", "test_accuracy"),
        ]
        
        all_results = {}
        comparison_data = []
        
        for ensemble_method, weight_strategy in strategies:
            strategy_name = f"{ensemble_method}_{weight_strategy}" if ensemble_method == "weighted_average" else "simple_average"
            
            print(f"\n{'='*60}")
            print(f"Testing {model_type.upper()}: {strategy_name}")
            print(f"{'='*60}")
            
            try:
                results = self.run_single_evaluation(
                    task=task,
                    model_type=model_type,
                    fold=fold,
                    nifti_dir=nifti_dir,
                    lobe_masks_dir=lobe_masks_dir,
                    labels_file=labels_file,
                    ensemble_methods=[ensemble_method],
                    weight_strategy=weight_strategy,
                    compare_strategies=False
                )
                
                if results and ensemble_method in results:
                    metrics = results[ensemble_method]['ensemble_metrics']
                    all_results[strategy_name] = results[ensemble_method]
                    
                    comparison_data.append({
                        'Model': model_type.upper(),
                        'Strategy': strategy_name,
                        'Method': ensemble_method,
                        'Weighting': weight_strategy,
                        'Data_Leakage': 'YES' if 'test_' in weight_strategy else 'NO',
                        'AUC': metrics['auc'],
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1_Score': metrics['f1_score']
                    })
                    
                    leak_status = "⚠️ DATA LEAKAGE" if 'test_' in weight_strategy else "✅ VALID"
                    print(f"{model_type.upper()} Results: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f} [{leak_status}]")
                    
            except Exception as e:
                print(f"Error with {model_type.upper()} {strategy_name}: {e}")
                import traceback
                traceback.print_exc()
        
        if not comparison_data:
            print(f"\n❌ ERROR: No successful {model_type.upper()} evaluations completed!")
            return None
        

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC', ascending=False)
        
        results_dir = self.model_configs[model_type]['results_dir']
        self.save_strategy_comparison_results(comparison_df, task, model_type, fold, results_dir)
        
        return {
            'comparison_df': comparison_df,
            'all_results': all_results,
            'model_type': model_type
        }
    
    def save_strategy_comparison_results(self, comparison_df, task, model_type, fold, results_dir):
        
        print(f"\n{'='*80}")
        print(f"{model_type.upper()} WEIGHTING STRATEGY COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"\n📊 {model_type.upper()} Performance Ranking (sorted by AUC):")
        print(comparison_df[['Strategy', 'Data_Leakage', 'AUC', 'Accuracy', 'F1_Score']].round(4).to_string(index=False))
        

        valid_methods = comparison_df[comparison_df['Data_Leakage'] == 'NO']
        leaky_methods = comparison_df[comparison_df['Data_Leakage'] == 'YES']
        
        print(f"\n🔍 {model_type.upper()} DATA LEAKAGE ANALYSIS:")
        print("-" * 40)
        
        if not valid_methods.empty:
            best_valid_auc = valid_methods['AUC'].max()
            best_valid_method = valid_methods.loc[valid_methods['AUC'].idxmax(), 'Strategy']
            print(f"Best valid {model_type.upper()} method: {best_valid_method} (AUC: {best_valid_auc:.4f})")
        
        if not leaky_methods.empty:
            best_leaky_auc = leaky_methods['AUC'].max()
            best_leaky_method = leaky_methods.loc[leaky_methods['AUC'].idxmax(), 'Strategy']
            print(f"Best leaky {model_type.upper()} method: {best_leaky_method} (AUC: {best_leaky_auc:.4f})")
            
            if not valid_methods.empty:
                auc_inflation = best_leaky_auc - best_valid_auc
                print(f"{model_type.upper()} AUC inflation due to data leakage: +{auc_inflation:.4f} ({100*auc_inflation/best_valid_auc:.1f}%)")
        

        os.makedirs(f"{results_dir}/ensemble_results/{task}/fold_{fold}", exist_ok=True)
        comparison_df.to_csv(f"{results_dir}/ensemble_results/{task}/fold_{fold}/{model_type}_weighting_strategy_comparison.csv", index=False)
        

        self.create_strategy_comparison_plot(comparison_df, task, model_type, fold, results_dir)
        
        print(f"\n💾 {model_type.upper()} Results saved to: {results_dir}/ensemble_results/{task}/fold_{fold}/{model_type}_weighting_strategy_comparison.csv")
    
    def create_strategy_comparison_plot(self, comparison_df, task, model_type, fold, results_dir):
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{model_type.upper()} Weighting Strategy Comparison: {task} (Fold {fold})', fontsize=16, fontweight='bold')
        

        ax = axes[0, 0]
        colors = ['green' if leak == 'NO' else 'red' for leak in comparison_df['Data_Leakage']]
        bars = ax.bar(range(len(comparison_df)), comparison_df['AUC'], color=colors, alpha=0.7)
        ax.set_xlabel('Strategy')
        ax.set_ylabel('AUC')
        ax.set_title(f'{model_type.upper()} AUC by Weighting Strategy')
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df['Strategy'], rotation=45, ha='right')
        
        for bar, auc in zip(bars, comparison_df['AUC']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{auc:.3f}', ha='center', va='bottom', fontsize=10)
        

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Valid (No Data Leakage)'),
                          Patch(facecolor='red', alpha=0.7, label='Invalid (Data Leakage)')]
        ax.legend(handles=legend_elements)
        

        ax = axes[0, 1]
        bars = ax.bar(range(len(comparison_df)), comparison_df['Accuracy'], color=colors, alpha=0.7)
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_type.upper()} Accuracy by Weighting Strategy')
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df['Strategy'], rotation=45, ha='right')
        

        ax = axes[1, 0]
        valid_methods = comparison_df[comparison_df['Data_Leakage'] == 'NO']
        leaky_methods = comparison_df[comparison_df['Data_Leakage'] == 'YES']
        
        if not valid_methods.empty and not leaky_methods.empty:
            valid_aucs = valid_methods['AUC']
            leaky_aucs = leaky_methods['AUC']
            
            box_data = [valid_aucs.values, leaky_aucs.values]
            labels = [f'Valid {model_type.upper()} Methods', f'Leaky {model_type.upper()} Methods']
            colors_box = ['green', 'red']
            
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('AUC')
            ax.set_title(f'{model_type.upper()} AUC Distribution: Valid vs Leaky Methods')
        else:
            ax.text(0.5, 0.5, f'Insufficient {model_type.upper()} data\nfor comparison', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{model_type.upper()} AUC Distribution Comparison')
        

        ax = axes[1, 1]
        metrics_data = comparison_df[['AUC', 'Accuracy', 'Precision', 'Recall', 'F1_Score']].T
        im = ax.imshow(metrics_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df['Strategy'], rotation=45, ha='right')
        ax.set_yticks(range(len(metrics_data)))
        ax.set_yticklabels(metrics_data.index)
        ax.set_title(f'{model_type.upper()} All Metrics Heatmap')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/ensemble_results/{task}/fold_{fold}/{model_type}_weighting_strategy_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_cross_validation(self, task: str, model_type: str, nifti_dir: str, 
                           lobe_masks_dir: str, labels_file: str, ensemble_method: str = "weighted_average"):
        
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION {model_type.upper()} ENSEMBLE EVALUATION")
        print(f"Task: {task}, Method: {ensemble_method}")
        print(f"{'='*80}")
        
        all_fold_results = []
        
        for fold in range(5):
            print(f"\n{'='*60}")
            print(f"{model_type.upper()} FOLD {fold}")
            print(f"{'='*60}")
            
            try:
                results = self.run_single_evaluation(
                    task=task,
                    model_type=model_type,
                    fold=fold,
                    nifti_dir=nifti_dir,
                    lobe_masks_dir=lobe_masks_dir,
                    labels_file=labels_file,
                    ensemble_methods=[ensemble_method]
                )
                
                if results and ensemble_method in results:
                    fold_metrics = results[ensemble_method]['ensemble_metrics']
                    all_fold_results.append({
                        'model_type': model_type,
                        'fold': fold,
                        'auc': fold_metrics['auc'],
                        'accuracy': fold_metrics['accuracy'],
                        'precision': fold_metrics['precision'],
                        'recall': fold_metrics['recall'],
                        'f1_score': fold_metrics['f1_score']
                    })
                    
                    print(f"{model_type.upper()} Fold {fold} results: AUC={fold_metrics['auc']:.4f}, Acc={fold_metrics['accuracy']:.4f}")
                
            except Exception as e:
                print(f"Error processing {model_type.upper()} fold {fold}: {e}")
                continue
        
        if not all_fold_results:
            print(f"No successful {model_type.upper()} fold results obtained")
            return None
        

        return self.analyze_cv_results(all_fold_results, task, model_type, ensemble_method)
    
    def analyze_cv_results(self, all_fold_results, task, model_type, ensemble_method):
        
        cv_df = pd.DataFrame(all_fold_results)
        
        cv_stats = {
            'model_type': model_type,
            'task': task,
            'ensemble_method': ensemble_method,
            'mean_auc': cv_df['auc'].mean(),
            'std_auc': cv_df['auc'].std(),
            'mean_accuracy': cv_df['accuracy'].mean(),
            'std_accuracy': cv_df['accuracy'].std(),
            'mean_precision': cv_df['precision'].mean(),
            'std_precision': cv_df['precision'].std(),
            'mean_recall': cv_df['recall'].mean(),
            'std_recall': cv_df['recall'].std(),
            'mean_f1': cv_df['f1_score'].mean(),
            'std_f1': cv_df['f1_score'].std(),
            'fold_results': all_fold_results
        }
        
        print(f"\n{'='*80}")
        print(f"{model_type.upper()} CROSS-VALIDATION RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Task: {task}")
        print(f"Model: {model_type.upper()}")
        print(f"Ensemble Method: {ensemble_method}")
        print(f"Number of folds: {len(all_fold_results)}")
        print(f"\n{model_type.upper()} Performance Metrics (Mean ± Std):")
        print(f"  AUC:       {cv_stats['mean_auc']:.4f} ± {cv_stats['std_auc']:.4f}")
        print(f"  Accuracy:  {cv_stats['mean_accuracy']:.4f} ± {cv_stats['std_accuracy']:.4f}")
        print(f"  Precision: {cv_stats['mean_precision']:.4f} ± {cv_stats['std_precision']:.4f}")
        print(f"  Recall:    {cv_stats['mean_recall']:.4f} ± {cv_stats['std_recall']:.4f}")
        print(f"  F1-Score:  {cv_stats['mean_f1']:.4f} ± {cv_stats['std_f1']:.4f}")
        

        results_dir = self.model_configs[model_type]['results_dir']
        os.makedirs(f"{results_dir}/ensemble_results/{task}", exist_ok=True)
        cv_results_file = f"{results_dir}/ensemble_results/{task}/{model_type}_cv_results_{ensemble_method}.json"
        

        serializable_stats = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                             for k, v in cv_stats.items() if k != 'fold_results'}
        serializable_stats['fold_results'] = cv_stats['fold_results']
        
        with open(cv_results_file, 'w') as f:
            json.dump(serializable_stats, f, indent=4)
        
        print(f"\n{model_type.upper()} Cross-validation results saved to: {cv_results_file}")
        

        self.create_cv_plot(cv_df, task, model_type, ensemble_method, results_dir)
        
        return cv_stats
    
    def create_cv_plot(self, cv_df, task, model_type, ensemble_method, results_dir):
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_type.upper()} Cross-Validation Results: {task} - {ensemble_method}', fontsize=16, fontweight='bold')
        
        metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            values = cv_df[metric].values
            folds = cv_df['fold'].values
            
            ax.bar(folds, values, alpha=0.7, color=f'C{idx}')
            ax.axhline(y=values.mean(), color='red', linestyle='--', 
                      label=f'Mean: {values.mean():.3f}')
            ax.fill_between([-0.5, 4.5], 
                           values.mean() - values.std(), 
                           values.mean() + values.std(),
                           alpha=0.2, color='red', label=f'±1 Std: {values.std():.3f}')
            
            ax.set_xlabel('Fold')
            ax.set_ylabel(f'{model_type.upper()} {name}')
            ax.set_title(f'{model_type.upper()} {name} Across Folds')
            ax.set_xticks(folds)
            ax.legend()
            ax.grid(True, alpha=0.3)
        

        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/ensemble_results/{task}/{model_type}_cv_results_{ensemble_method}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_multi_model_comparison(self, task: str, model_types: List[str], fold: int,
                                  nifti_dir: str, lobe_masks_dir: str, labels_file: str,
                                  ensemble_method: str = "weighted_average", weight_strategy: str = "val_auc"):
        
        print(f"\n{'='*80}")
        print(f"MULTI-MODEL COMPARISON")
        print(f"Task: {task}, Fold: {fold}, Method: {ensemble_method}")
        print(f"Models: {', '.join([m.upper() for m in model_types])}")
        print(f"{'='*80}")
        
        all_model_results = {}
        comparison_data = []
        
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Evaluating {model_type.upper()} Model")
            print(f"{'='*60}")
            
            try:
                results = self.run_single_evaluation(
                    task=task,
                    model_type=model_type,
                    fold=fold,
                    nifti_dir=nifti_dir,
                    lobe_masks_dir=lobe_masks_dir,
                    labels_file=labels_file,
                    ensemble_methods=[ensemble_method],
                    weight_strategy=weight_strategy
                )
                
                if results and ensemble_method in results:
                    metrics = results[ensemble_method]['ensemble_metrics']
                    all_model_results[model_type] = results[ensemble_method]
                    
                    comparison_data.append({
                        'Model': model_type.upper(),
                        'AUC': metrics['auc'],
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1_Score': metrics['f1_score']
                    })
                    
                    print(f"{model_type.upper()} Results: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {model_type.upper()}: {e}")
                continue
        
        if not comparison_data:
            print("❌ No successful model evaluations completed!")
            return None
        

        return self.analyze_multi_model_results(comparison_data, all_model_results, task, fold, ensemble_method)
    
    def analyze_multi_model_results(self, comparison_data, all_model_results, task, fold, ensemble_method):
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC', ascending=False)
        
        print(f"\n{'='*80}")
        print(f"MULTI-MODEL COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"Task: {task}, Fold: {fold}, Method: {ensemble_method}")
        print(f"\n📊 Model Performance Ranking (sorted by AUC):")
        print(comparison_df.round(4).to_string(index=False))
        

        best_model = comparison_df.iloc[0]['Model']
        best_auc = comparison_df.iloc[0]['AUC']
        print(f"\n🏆 Best performing model: {best_model} (AUC: {best_auc:.4f})")
        

        output_dir = f"multi_model_ensemble_results/{task}/fold_{fold}"
        os.makedirs(output_dir, exist_ok=True)
        comparison_df.to_csv(f"{output_dir}/model_comparison_{ensemble_method}.csv", index=False)
        

        self.create_multi_model_plot(comparison_df, task, fold, ensemble_method, output_dir)
        
        print(f"\n💾 Multi-model comparison saved to: {output_dir}/model_comparison_{ensemble_method}.csv")
        
        return {
            'comparison_df': comparison_df,
            'all_model_results': all_model_results,
            'best_model': best_model,
            'best_auc': best_auc
        }
    
    def create_multi_model_plot(self, comparison_df, task, fold, ensemble_method, output_dir):
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Multi-Model Ensemble Comparison: {task} (Fold {fold})', fontsize=16, fontweight='bold')
        

        ax = axes[0, 0]
        colors = ['gold', 'silver', 'chocolate'] + ['lightblue'] * (len(comparison_df) - 3)
        bars = ax.bar(comparison_df['Model'], comparison_df['AUC'], color=colors[:len(comparison_df)])
        ax.set_ylabel('AUC Score')
        ax.set_title('Model AUC Comparison')
        ax.set_ylim(0, 1)
        for bar, auc in zip(bars, comparison_df['AUC']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{auc:.3f}', ha='center', va='bottom')
        

        ax = axes[0, 1]
        bars = ax.bar(comparison_df['Model'], comparison_df['Accuracy'], color=colors[:len(comparison_df)])
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim(0, 1)
        for bar, acc in zip(bars, comparison_df['Accuracy']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        

        ax = axes[1, 0]
        metrics_to_plot = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
        x = np.arange(len(comparison_df))
        width = 0.15
        
        for i, metric in enumerate(metrics_to_plot):
            ax.bar(x + i*width, comparison_df[metric], width, 
                  label=metric, alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Multiple Metrics Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(comparison_df['Model'])
        ax.legend()
        ax.set_ylim(0, 1)
        

        ax = axes[1, 1]
        metrics_data = comparison_df[['AUC', 'Accuracy', 'Precision', 'Recall', 'F1_Score']].T
        im = ax.imshow(metrics_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df['Model'])
        ax.set_yticks(range(len(metrics_data)))
        ax.set_yticklabels(metrics_data.index)
        ax.set_title('All Metrics Heatmap')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comparison_{ensemble_method}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Unified Ensemble Evaluation for Multiple Model Types')
    parser.add_argument('--task', type=str, required=True,
                       help='Task to analyze (e.g., has_ILD)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'model', 'fold', 'all', 'cv', 'compare'],
                       help='Evaluation mode')
    parser.add_argument('--model_type', type=str, 
                       choices=['cnn', 'vit', 'mamba'],
                       help='Model type (required for single, model, cv modes)')
    parser.add_argument('--model_types', nargs='+',
                       choices=['cnn', 'vit', 'mamba'],
                       help='Multiple model types for comparison (for compare mode)')
    parser.add_argument('--fold_idx', type=int, choices=range(5),
                       help='Fold index (required for single, fold modes)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--nifti_dir', type=str, default=None,
                       help='Directory containing NIFTI files')
    parser.add_argument('--lobe_masks_dir', type=str, default=None,
                       help='Directory containing lobe masks')
    parser.add_argument('--labels_path', type=str, default=None,
                       help='Path to labels CSV file')
    parser.add_argument('--methods', nargs='+', 
                       choices=['simple_average', 'weighted_average', 'learned_ensemble'],
                       default=['simple_average', 'weighted_average'],
                       help='Ensemble methods to test')
    parser.add_argument('--weight_strategy', type=str, 
                       choices=['val_auc', 'val_accuracy', 'inverse_val_loss', 'uniform', 'test_auc', 'test_accuracy'],
                       default='val_auc',
                       help='Strategy for weighting experts in weighted_average method')
    parser.add_argument('--compare_strategies', action='store_true',
                       help='Run comprehensive comparison of all weighting strategies')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device number to use')
    
    args = parser.parse_args()
    
    import json
    from pathlib import Path
    
    if not Path(args.config).exists():
        parser.error(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    data_paths = config.get('data_paths', {})
    if args.nifti_dir is None:
        args.nifti_dir = data_paths.get('nifti_dir')
    if args.lobe_masks_dir is None:
        args.lobe_masks_dir = data_paths.get('lobe_masks_dir')
    if args.labels_path is None:
        args.labels_path = data_paths.get('labels_file')
    
    if not args.nifti_dir or not args.lobe_masks_dir or not args.labels_path:
        parser.error("Missing required paths in config. Please check data_paths section.")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    

    evaluator = UnifiedEnsembleEvaluator(config_path=args.config)
    
    print(f"Starting unified ensemble evaluation:")
    print(f"  Mode: {args.mode}")
    print(f"  Task: {args.task}")
    print(f"  GPU: {args.gpu}")
    
    try:
        if args.mode == 'single':

            if not args.model_type or args.fold_idx is None:
                print("❌ Single mode requires --model_type and --fold_idx")
                return
            
            print(f"  Model: {args.model_type}")
            print(f"  Fold: {args.fold_idx}")
            
            results = evaluator.run_single_evaluation(
                task=args.task,
                model_type=args.model_type,
                fold=args.fold_idx,
                nifti_dir=args.nifti_dir,
                lobe_masks_dir=args.lobe_masks_dir,
                labels_file=args.labels_path,
                ensemble_methods=args.methods,
                weight_strategy=args.weight_strategy,
                compare_strategies=args.compare_strategies
            )
            
            if results:
                print(f"\n✅ {args.model_type.upper()} evaluation completed successfully!")
        
        elif args.mode == 'model':

            if not args.model_type:
                print("❌ Model mode requires --model_type")
                return
            
            print(f"  Model: {args.model_type}")
            print("  Evaluating all folds...")
            
            cv_results = evaluator.run_cross_validation(
                task=args.task,
                model_type=args.model_type,
                nifti_dir=args.nifti_dir,
                lobe_masks_dir=args.lobe_masks_dir,
                labels_file=args.labels_path,
                ensemble_method="weighted_average"
            )
            
            if cv_results:
                print(f"\n✅ {args.model_type.upper()} cross-validation completed!")
        
        elif args.mode == 'fold':

            if args.fold_idx is None:
                print("❌ Fold mode requires --fold_idx")
                return
            
            print(f"  Fold: {args.fold_idx}")
            print("  Evaluating all model types...")
            
            model_types = ['cnn', 'vit', 'mamba']
            comparison_results = evaluator.run_multi_model_comparison(
                task=args.task,
                model_types=model_types,
                fold=args.fold_idx,
                nifti_dir=args.nifti_dir,
                lobe_masks_dir=args.lobe_masks_dir,
                labels_file=args.labels_path,
                ensemble_method="weighted_average",
                weight_strategy=args.weight_strategy
            )
            
            if comparison_results:
                print(f"\n✅ Multi-model comparison completed!")
                print(f"🏆 Best model: {comparison_results['best_model']}")
        
        elif args.mode == 'all':

            print("  Evaluating all models and all folds...")
            
            model_types = ['cnn', 'vit', 'mamba']
            all_results = {}
            
            for model_type in model_types:
                print(f"\n{'='*60}")
                print(f"Cross-validation for {model_type.upper()}")
                print(f"{'='*60}")
                
                cv_results = evaluator.run_cross_validation(
                    task=args.task,
                    model_type=model_type,
                    nifti_dir=args.nifti_dir,
                    lobe_masks_dir=args.lobe_masks_dir,
                    labels_file=args.labels_path
                )
                
                if cv_results:
                    all_results[model_type] = cv_results
            

            if all_results:
                print(f"\n{'='*80}")
                print("FINAL COMPARISON ACROSS ALL MODELS")
                print(f"{'='*80}")
                
                for model_type, results in all_results.items():
                    print(f"{model_type.upper():>6}: AUC={results['mean_auc']:.4f}±{results['std_auc']:.4f}")
                
                best_model = max(all_results.keys(), key=lambda k: all_results[k]['mean_auc'])
                print(f"\n🏆 Overall best model: {best_model.upper()}")
        
        elif args.mode == 'cv':

            if not args.model_type:
                print("❌ CV mode requires --model_type")
                return
            
            cv_results = evaluator.run_cross_validation(
                task=args.task,
                model_type=args.model_type,
                nifti_dir=args.nifti_dir,
                lobe_masks_dir=args.lobe_masks_dir,
                labels_file=args.labels_path
            )
            
            if cv_results:
                print(f"\n✅ {args.model_type.upper()} cross-validation completed!")
        
        elif args.mode == 'compare':

            if not args.model_types or args.fold_idx is None:
                print("❌ Compare mode requires --model_types and --fold_idx")
                return
            
            comparison_results = evaluator.run_multi_model_comparison(
                task=args.task,
                model_types=args.model_types,
                fold=args.fold_idx,
                nifti_dir=args.nifti_dir,
                lobe_masks_dir=args.lobe_masks_dir,
                labels_file=args.labels_path
            )
            
            if comparison_results:
                print(f"\n✅ Model comparison completed!")
                print(f"🏆 Best model: {comparison_results['best_model']}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()