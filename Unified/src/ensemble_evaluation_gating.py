

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


from monai.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

class UnifiedEnsembleEvaluator:
    
    
    def __init__(self, config_path: Optional[str] = None):
        if not config_path or not Path(config_path).exists():
            raise ValueError("config_path is required and must point to an existing config file")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        output_config = config.get('output', {})
        
        if not output_config.get('unified_cnn_results') or not output_config.get('unified_vit_results') or not output_config.get('unified_mamba_results') or not output_config.get('unified_extracted_gating_results'):
            raise ValueError("Missing required output paths in config file")
        
        self.model_configs = {
            'cnn': {
                'expert_results_dir': output_config['unified_cnn_results'],
                'gating_results_dir': output_config['unified_extracted_gating_results'],
                'ensemble_module': 'mixture_experts_ensemble_gating_cnn',
                'data_loader_module': 'unified_expert_models',
                'model_name': 'CNN'
            },
            'vit': {
                'expert_results_dir': output_config['unified_vit_results'],
                'gating_results_dir': output_config['unified_extracted_gating_results'],
                'ensemble_module': 'mixture_experts_ensemble_gating_vit',
                'data_loader_module': 'unified_expert_models',
                'model_name': 'ViT'
            },
            'mamba': {
                'expert_results_dir': output_config['unified_mamba_results'],
                'gating_results_dir': output_config['unified_extracted_gating_results'],
                'ensemble_module': 'mixture_experts_ensemble_gating_mamba',
                'data_loader_module': 'unified_expert_models',
                'model_name': 'Mamba'
            }
        }
        
        self.config_path = config_path
        self.output_config = output_config
        
        self.expert_names = [
            "Lobe_1_LUL", "Lobe_2_LLL", "Lobe_3_RUL", 
            "Lobe_4_RML", "Lobe_5_RLL", "Left_Lung", "Right_Lung"
        ]
    
    def load_data_functions(self, model_type: str):
        
        config = self.model_configs[model_type]
        module_name = config['data_loader_module']
        
        try:
            if module_name == 'unified_expert_models':
                from experts import (
                    load_image_and_mask_data, 
                    create_consistent_folds,
                    prepare_data_dicts,
                    get_transforms,
                    create_expert_mask,
                    get_patient_id
                )

            return {
                'load_image_and_mask_data': load_image_and_mask_data,
                'create_consistent_folds': create_consistent_folds,
                'prepare_data_dicts': prepare_data_dicts,
                'get_transforms': get_transforms,
                'create_expert_mask': create_expert_mask,
                'get_patient_id': get_patient_id
            }
        except ImportError as e:
            print(f"❌ Error loading data functions for {model_type}: {e}")
            return None
    
    def load_ensemble_class(self, model_type: str):
        
        try:
            from ensemble_gating import UnifiedMixtureOfExpertsEnsemble
            return UnifiedMixtureOfExpertsEnsemble
        except ImportError as e:
            print(f"❌ Error loading unified ensemble class: {e}")
            return None
    
    def load_gating_functions(self, model_type: str, task: str, fold: int) -> Optional[Dict]:
        base_gating_dir = self.output_config['unified_extracted_gating_results']
        

        if model_type == 'vit':
            gating_file = Path(base_gating_dir) / "vit" / task / f"fold_{fold}" / "extracted_vit_gating_functions.json"
        elif model_type == 'mamba':
            gating_file = Path(base_gating_dir) / "mamba" / task / f"fold_{fold}" / "extracted_mamba_gating_functions.json"
        else:
            gating_file = Path(base_gating_dir) / "cnn" / task / f"fold_{fold}" / "extracted_gating_functions.json"
        
        if not gating_file.exists():
            print(f"❌ {self.model_configs[model_type]['model_name']} gating functions file not found: {gating_file}")
            print(f"Please run step 2 (extract {model_type} gating functions) first!")
            return None
        
        print(f"✅ Loading {self.model_configs[model_type]['model_name']} gating functions from: {gating_file}")
        
        with open(gating_file, 'r') as f:
            gating_data = json.load(f)
        
        return gating_data['gating_functions']
    
    def normalize_hierarchical_weights(self, weights: np.ndarray, model_type: str) -> np.ndarray:
        
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
        
        model_name = self.model_configs[model_type]['model_name']
        print(f"{model_name} Hierarchical normalization:")
        print(f"  Lobes (sum={np.sum(normalized_weights[:5]):.3f}): {[f'{w:.3f}' for w in normalized_weights[:5]]}")
        print(f"  Lungs (sum={np.sum(normalized_weights[5:]):.3f}): {[f'{w:.3f}' for w in normalized_weights[5:]]}")
        
        return normalized_weights
    
    def check_prerequisites(self, model_type: str, task: str, fold: int) -> bool:
        
        config = self.model_configs[model_type]
        model_name = config['model_name']
        

        if model_type == 'vit':
            gating_file = Path(config['gating_results_dir']) /  "vit" / task / f"fold_{fold}" / "extracted_vit_gating_functions.json"
        elif model_type == 'mamba':
            gating_file = Path(config['gating_results_dir']) /  "mamba" / task / f"fold_{fold}" / "extracted_mamba_gating_functions.json"
        else:
            gating_file = Path(config['gating_results_dir']) /  "cnn" / task / f"fold_{fold}" / "extracted_gating_functions.json"
            
        if not gating_file.exists():
            print(f"❌ {model_name} gating functions not found: {gating_file}")
            return False
        

        expert_dir = Path(config['expert_results_dir']) / task / f"fold_{fold}"
        if not expert_dir.exists():
            print(f"❌ {model_name} expert results not found: {expert_dir}")
            return False
        
        return True
    
    def run_single_evaluation(self, model_type: str, task: str, fold: int, 
                            nifti_dir: str, lobe_masks_dir: str, labels_file: str,
                            normalize_weights: bool = True) -> Optional[Dict]:
        
        
        config = self.model_configs[model_type]
        model_name = config['model_name']
        
        print(f"\n{'='*80}")
        print(f"🚀 Running {model_name} ensemble evaluation for task: {task}, fold: {fold}")
        print(f"{'='*80}")
        

        if not self.check_prerequisites(model_type, task, fold):
            print(f"❌ Prerequisites not met for {model_name}")
            return None
        

        data_functions = self.load_data_functions(model_type)
        EnsembleClass = self.load_ensemble_class(model_type)
        
        if data_functions is None or EnsembleClass is None:
            print(f"❌ Failed to load required modules for {model_name}")
            return None
        

        gating_functions = self.load_gating_functions(model_type, task, fold)
        if gating_functions is None:
            return None
        

        all_strategies = []
        

        performance_strategies = gating_functions.get('performance_based', {})
        for strategy_name, weights in performance_strategies.items():
            if isinstance(weights, (list, np.ndarray)) and len(weights) == 7:
                if 'test' not in strategy_name.lower():
                    normalized_weights = self.normalize_hierarchical_weights(weights, model_type) if normalize_weights else weights
                    all_strategies.append({
                        'name': f"{model_type}_perf_{strategy_name}",
                        'category': 'performance_based',
                        'weights': normalized_weights,
                        'description': f"{model_name} Performance-based: {strategy_name}",
                        'model_type': model_type
                    })
        

        feature_strategies = gating_functions.get('feature_based', {})
        for strategy_name, weights in feature_strategies.items():
            if isinstance(weights, (list, np.ndarray)) and len(weights) == 7:
                normalized_weights = self.normalize_hierarchical_weights(weights, model_type) if normalize_weights else weights
                all_strategies.append({
                    'name': f"{model_type}_feat_{strategy_name}",
                    'category': 'feature_based', 
                    'weights': normalized_weights,
                    'description': f"{model_name} Feature-based: {strategy_name}",
                    'model_type': model_type
                })
        

        learned_strategies = gating_functions.get('learned_based', {})
        for strategy_name, weights in learned_strategies.items():
            if isinstance(weights, (list, np.ndarray)) and len(weights) == 7:
                normalized_weights = self.normalize_hierarchical_weights(weights, model_type) if normalize_weights else weights
                all_strategies.append({
                    'name': f"{model_type}_learned_{strategy_name}",
                    'category': 'learned_based',
                    'weights': normalized_weights, 
                    'description': f"{model_name} Learned: {strategy_name}",
                    'model_type': model_type
                })
        




        
















        
        print(f"📊 Found {len(all_strategies)} {model_name} gating strategies to evaluate:")
        for strategy in all_strategies:
            print(f"  - {strategy['name']}: {strategy['description']}")
        

        file_list, mask_list, labels, patient_ids = data_functions['load_image_and_mask_data'](
            labels_file, nifti_dir, lobe_masks_dir, task
        )
        

        k_folds = 5
        consistent_folds = data_functions['create_consistent_folds'](
            file_list, labels, patient_ids, k_folds=k_folds, seed=42
        )
        fold_data = consistent_folds[fold]
        

        holdout_idx = fold_data['holdout_idx']
        val_mask = fold_data['val_mask']
        
        holdout_files = np.array(file_list)[holdout_idx]
        holdout_masks = np.array(mask_list)[holdout_idx]
        holdout_labels = np.array(labels)[holdout_idx]
        
        val_files = holdout_files[val_mask]
        val_masks = holdout_masks[val_mask]
        val_labels = holdout_labels[val_mask]
        
        print(f"📈 {model_name} Validation set size: {len(val_files)} (NO test data used!)")
        print(f"{model_name} Validation label distribution: {Counter(val_labels)}")
        

        val_data = data_functions['prepare_data_dicts'](val_files, val_masks, val_labels)
        transforms = data_functions['get_transforms']()
        val_dataset = Dataset(data=val_data, transform=transforms)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
        

        all_results = {}
        
        for strategy in all_strategies:
            strategy_name = strategy['name']
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name} Strategy: {strategy_name}")
            print(f"Description: {strategy['description']}")
            print(f"{'='*60}")
            
            try:

                if strategy['weights'] is not None:
                    ensemble = EnsembleClass(
                        model_type=model_type,
                        results_dir=config['expert_results_dir'], 
                        ensemble_method="gating_weighted",
                        gating_weights=strategy['weights'],
                        expert_names=self.expert_names
                    )
                else:
                    ensemble = EnsembleClass(
                        model_type=model_type,
                        results_dir=config['expert_results_dir'],
                        ensemble_method="weighted_average", 
                        weight_strategy="val_auc"
                    )
                
                ensemble.load_expert_models(task, fold)
                
                if len(ensemble.expert_models) == 0:
                    print(f"❌ No {model_name} expert models found for {strategy_name}")
                    continue
                

                val_predictions = {}
                true_labels = []
                
                print(f"Getting {model_name} expert predictions on validation data...")
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        image = batch["image"].to(ensemble.device)
                        mask = batch["mask"].to(ensemble.device)
                        target = batch["label"]
                        
                        expert_preds = ensemble.get_expert_predictions(image, mask)
                        
                        for key, preds in expert_preds.items():
                            if key not in val_predictions:
                                val_predictions[key] = []
                            val_predictions[key].extend(preds.tolist())
                        
                        true_labels.extend(target.tolist())
                        
                        if (batch_idx + 1) % 10 == 0:
                            print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")
                

                for key in val_predictions:
                    val_predictions[key] = np.array(val_predictions[key])
                true_labels = np.array(true_labels)
                
                print(f"Collected predictions from {len(val_predictions)} {model_name} experts")
                print(f"Total validation samples: {len(true_labels)}")
                

                results = ensemble.evaluate_ensemble(
                    val_predictions, 
                    true_labels, 
                    save_results=True, 
                    output_dir=f"{config['expert_results_dir']}/gating_ensemble_results/{task}/fold_{fold}"
                )
                

                results['strategy_info'] = {
                    'name': strategy_name,
                    'category': strategy['category'],
                    'description': strategy['description'],
                    'weights_used': strategy['weights'].tolist() if strategy['weights'] is not None else None,
                    'hierarchically_normalized': normalize_weights,
                    'model_type': model_type
                }
                
                all_results[strategy_name] = results
                

                metrics = results['ensemble_metrics']
                print(f"\n📊 {model_name} Results for {strategy_name}:")
                print(f"  AUC: {metrics['auc']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                
                if strategy['weights'] is not None:
                    print(f"  Weights used: {[f'{w:.3f}' for w in strategy['weights']]}")
                    
            except Exception as e:
                print(f"❌ Error evaluating {model_name} {strategy_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[strategy_name] = {'error': str(e)}
        

        if len(all_results) > 1:
            print(f"\n{'='*80}")
            print(f"{model_name.upper()} GATING STRATEGY COMPARISON RESULTS")
            print(f"{'='*80}")
            
            comparison_data = []
            for strategy_name, result in all_results.items():
                if 'error' not in result:
                    metrics = result['ensemble_metrics']
                    strategy_info = result['strategy_info']
                    

                    weights_used = strategy_info['weights_used']
                    if weights_used is not None and len(weights_used) >= 7:
                        weight_dict = {
                            'Weight_Lobe1_LUL': weights_used[0],
                            'Weight_Lobe2_LLL': weights_used[1],
                            'Weight_Lobe3_RUL': weights_used[2],
                            'Weight_Lobe4_RML': weights_used[3],
                            'Weight_Lobe5_RLL': weights_used[4],
                            'Weight_Left_Lung': weights_used[5],
                            'Weight_Right_Lung': weights_used[6]
                        }
                    else:

                        weight_dict = {
                            'Weight_Lobe1_LUL': None,
                            'Weight_Lobe2_LLL': None,
                            'Weight_Lobe3_RUL': None,
                            'Weight_Lobe4_RML': None,
                            'Weight_Lobe5_RLL': None,
                            'Weight_Left_Lung': None,
                            'Weight_Right_Lung': None
                        }

                    comparison_data.append({
                        'Strategy': strategy_name,
                        'Model_Type': model_type.upper(),
                        'Category': strategy_info['category'],
                        'Description': strategy_info['description'],
                        'AUC': metrics['auc'],
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1_Score': metrics['f1_score'],
                        'Normalized': normalize_weights,
                        **weight_dict
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('AUC', ascending=False)
                
                print(f"\n🏆 {model_name} Performance Ranking (sorted by AUC):")
                print(comparison_df[['Strategy', 'Category', 'AUC', 'Accuracy', 'F1_Score']].round(4).to_string(index=False))
                

                best_strategy = comparison_df.iloc[0]
                best_strategy_name = best_strategy['Strategy']
                best_auc = best_strategy['AUC']
                
                print(f"\n🥇 BEST {model_name.upper()} GATING STRATEGY: {best_strategy_name}")
                print(f"   Category: {best_strategy['Category']}")
                print(f"   AUC: {best_auc:.4f}")
                print(f"   Description: {best_strategy['Description']}")
                

                best_strategy_info = {
                    'best_strategy': {
                        'name': best_strategy_name,
                        'weights': all_results[best_strategy_name]['strategy_info']['weights_used'],
                        'category': best_strategy['Category'],
                        'performance': {
                            'auc': best_auc,
                            'accuracy': best_strategy['Accuracy'],
                            'f1_score': best_strategy['F1_Score']
                        },
                        'hierarchically_normalized': normalize_weights,
                        'model_type': model_type
                    },
                    'all_strategies_comparison': comparison_df.to_dict('records'),
                    'task': task,
                    'fold': fold,
                    'validation_only': True,
                    'model_type': model_type
                }
                
                output_dir = Path(config['expert_results_dir']) / "gating_ensemble_results" / task / f"fold_{fold}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                comparison_df.to_csv(output_dir / f"{model_type}_gating_strategy_comparison.csv", index=False)
                
                with open(output_dir / f"best_{model_type}_gating_strategy.json", 'w') as f:
                    json.dump(best_strategy_info, f, indent=4)
                
                print(f"\n💾 {model_name} Results saved to: {output_dir}")
                print(f"📊 {model_name} Comparison saved: {model_type}_gating_strategy_comparison.csv")
                print(f"🏆 Best {model_name} strategy saved: best_{model_type}_gating_strategy.json")
                
                return best_strategy_info
        
        return all_results
    
    def run_mode_all(self, task: str, nifti_dir: str, lobe_masks_dir: str, 
                    labels_file: str, normalize_weights: bool = True) -> Dict:
        
        print(f"\n🌟 Running COMPREHENSIVE evaluation for task: {task}")
        print("📋 Mode: ALL models, ALL folds")
        print(f"🎯 Models: {list(self.model_configs.keys())}")
        print(f"📊 Folds: 0-4")
        
        all_results = {}
        
        for model_type in self.model_configs.keys():
            model_results = {}
            for fold in range(5):
                print(f"\n🔄 Processing {model_type.upper()} - Fold {fold}")
                result = self.run_single_evaluation(
                    model_type, task, fold, nifti_dir, lobe_masks_dir, 
                    labels_file, normalize_weights
                )
                if result:
                    model_results[f'fold_{fold}'] = result
            
            all_results[model_type] = model_results
        

        self.compare_across_models(all_results, task)
        
        return all_results
    
    def run_mode_model(self, task: str, model_type: str, nifti_dir: str, 
                      lobe_masks_dir: str, labels_file: str, 
                      normalize_weights: bool = True) -> Dict:
        
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_name = self.model_configs[model_type]['model_name']
        print(f"\n🎯 Running {model_name} evaluation for task: {task}")
        print(f"📋 Mode: {model_name} model, ALL folds")
        print(f"📊 Folds: 0-4")
        
        model_results = {}
        for fold in range(5):
            print(f"\n🔄 Processing {model_name} - Fold {fold}")
            result = self.run_single_evaluation(
                model_type, task, fold, nifti_dir, lobe_masks_dir, 
                labels_file, normalize_weights
            )
            if result:
                model_results[f'fold_{fold}'] = result
        
        return {model_type: model_results}
    
    def run_mode_fold(self, task: str, fold_idx: int, nifti_dir: str, 
                     lobe_masks_dir: str, labels_file: str, 
                     normalize_weights: bool = True) -> Dict:
        
        print(f"\n📊 Running evaluation for task: {task}, fold: {fold_idx}")
        print(f"📋 Mode: ALL models, SINGLE fold")
        print(f"🎯 Models: {list(self.model_configs.keys())}")
        print(f"📊 Fold: {fold_idx}")
        
        fold_results = {}
        for model_type in self.model_configs.keys():
            model_name = self.model_configs[model_type]['model_name']
            print(f"\n🔄 Processing {model_name} - Fold {fold_idx}")
            result = self.run_single_evaluation(
                model_type, task, fold_idx, nifti_dir, lobe_masks_dir, 
                labels_file, normalize_weights
            )
            if result:
                fold_results[model_type] = result
        

        self.compare_models_single_fold(fold_results, task, fold_idx)
        
        return fold_results
    
    def run_mode_single(self, task: str, model_type: str, fold_idx: int, 
                       nifti_dir: str, lobe_masks_dir: str, labels_file: str, 
                       normalize_weights: bool = True) -> Dict:
        
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_name = self.model_configs[model_type]['model_name']
        print(f"\n🎯 Running {model_name} evaluation for task: {task}, fold: {fold_idx}")
        print(f"📋 Mode: SINGLE model, SINGLE fold")
        
        result = self.run_single_evaluation(
            model_type, task, fold_idx, nifti_dir, lobe_masks_dir, 
            labels_file, normalize_weights
        )
        
        return {model_type: {f'fold_{fold_idx}': result}} if result else {}
    
    def compare_across_models(self, all_results: Dict, task: str):
        
        print(f"\n{'='*100}")
        print("🏆 COMPREHENSIVE CROSS-MODEL COMPARISON")
        print(f"{'='*100}")
        

        comparison_data = []
        
        for model_type, model_results in all_results.items():
            model_name = self.model_configs[model_type]['model_name']
            for fold_key, fold_result in model_results.items():
                if 'best_strategy' in fold_result:
                    best = fold_result['best_strategy']
                    fold_num = int(fold_key.split('_')[1])
                    
                    comparison_data.append({
                        'Model_Type': model_name,
                        'Fold': fold_num,
                        'Strategy': best['name'],
                        'Category': best['category'],
                        'AUC': best['performance']['auc'],
                        'Accuracy': best['performance']['accuracy'],
                        'F1_Score': best['performance']['f1_score']
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            

            best_overall = comparison_df.loc[comparison_df['AUC'].idxmax()]
            
            print(f"\n🥇 OVERALL BEST PERFORMANCE:")
            print(f"   Model: {best_overall['Model_Type']}")
            print(f"   Fold: {best_overall['Fold']}")
            print(f"   Strategy: {best_overall['Strategy']}")
            print(f"   AUC: {best_overall['AUC']:.4f}")
            print(f"   Accuracy: {best_overall['Accuracy']:.4f}")
            print(f"   F1-Score: {best_overall['F1_Score']:.4f}")
            

            print(f"\n📊 MODEL-WISE SUMMARY:")
            model_summary = comparison_df.groupby('Model_Type').agg({
                'AUC': ['mean', 'std', 'max'],
                'Accuracy': ['mean', 'std', 'max'],
                'F1_Score': ['mean', 'std', 'max']
            }).round(4)
            
            print(model_summary)
            

            print(f"\n📈 FOLD-WISE SUMMARY:")
            fold_summary = comparison_df.groupby('Fold').agg({
                'AUC': ['mean', 'std', 'max'],
                'Accuracy': ['mean', 'std', 'max'],
                'F1_Score': ['mean', 'std', 'max']
            }).round(4)
            
            print(fold_summary)
            

            output_dir = Path("comprehensive_comparison_results") / task
            output_dir.mkdir(parents=True, exist_ok=True)
            
            comparison_df.to_csv(output_dir / "all_models_all_folds_comparison.csv", index=False)
            model_summary.to_csv(output_dir / "model_wise_summary.csv")
            fold_summary.to_csv(output_dir / "fold_wise_summary.csv")
            

            overall_best_info = {
                'overall_best': best_overall.to_dict(),
                'model_summary': model_summary.to_dict(),
                'fold_summary': fold_summary.to_dict(),
                'task': task,
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(output_dir / "comprehensive_comparison_summary.json", 'w') as f:
                json.dump(overall_best_info, f, indent=4, default=str)
            
            print(f"\n💾 Comprehensive comparison saved to: {output_dir}")
    
    def compare_models_single_fold(self, fold_results: Dict, task: str, fold_idx: int):
        
        print(f"\n{'='*80}")
        print(f"🏆 CROSS-MODEL COMPARISON - FOLD {fold_idx}")
        print(f"{'='*80}")
        
        comparison_data = []
        
        for model_type, result in fold_results.items():
            model_name = self.model_configs[model_type]['model_name']
            if 'best_strategy' in result:
                best = result['best_strategy']
                
                comparison_data.append({
                    'Model_Type': model_name,
                    'Strategy': best['name'],
                    'Category': best['category'],
                    'AUC': best['performance']['auc'],
                    'Accuracy': best['performance']['accuracy'],
                    'F1_Score': best['performance']['f1_score']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('AUC', ascending=False)
            
            print(f"\n🏆 Model Performance Ranking - Fold {fold_idx}:")
            print(comparison_df.round(4).to_string(index=False))
            
            best_model = comparison_df.iloc[0]
            print(f"\n🥇 BEST MODEL FOR FOLD {fold_idx}:")
            print(f"   Model: {best_model['Model_Type']}")
            print(f"   Strategy: {best_model['Strategy']}")
            print(f"   AUC: {best_model['AUC']:.4f}")
            

            output_dir = Path("fold_comparison_results") / task / f"fold_{fold_idx}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            comparison_df.to_csv(output_dir / f"fold_{fold_idx}_model_comparison.csv", index=False)
            
            fold_best_info = {
                'fold': fold_idx,
                'best_model': best_model.to_dict(),
                'all_models_comparison': comparison_df.to_dict('records'),
                'task': task
            }
            
            with open(output_dir / f"fold_{fold_idx}_best_model.json", 'w') as f:
                json.dump(fold_best_info, f, indent=4)
            
            print(f"\n💾 Fold {fold_idx} comparison saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Ensemble Evaluation for Multiple Model Types',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    

    parser.add_argument('--task', type=str, required=True,
                       help='Task to analyze (e.g., has_ILD)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['all', 'model', 'fold', 'single'],
                       help='Evaluation mode')
    

    parser.add_argument('--model_type', type=str, choices=['cnn', 'vit', 'mamba'],
                       help='Model type (required for modes: model, single)')
    parser.add_argument('--fold_idx', type=int, choices=range(5),
                       help='Fold index 0-4 (required for modes: fold, single)')
    

    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--nifti_dir', type=str, default=None,
                       help='Directory containing NIFTI files')
    parser.add_argument('--lobe_masks_dir', type=str, default=None,
                       help='Directory containing lobe masks')
    parser.add_argument('--labels_path', type=str, default=None,
                       help='Path to labels CSV file')
    

    parser.add_argument('--normalize_weights', action='store_true', default=True,
                       help='Apply hierarchical weight normalization')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device number to use')
    
    args = parser.parse_args()
    
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

    if args.mode in ['model', 'single'] and args.model_type is None:
        parser.error(f"--model_type is required when mode is '{args.mode}'")
    
    if args.mode in ['fold', 'single'] and args.fold_idx is None:
        parser.error(f"--fold_idx is required when mode is '{args.mode}'")
    

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    print(f"🚀 Starting Unified Ensemble Evaluation")
    print(f"📋 Mode: {args.mode}")
    print(f"🎯 Task: {args.task}")
    if args.model_type:
        print(f"🤖 Model Type: {args.model_type}")
    if args.fold_idx is not None:
        print(f"📊 Fold: {args.fold_idx}")
    print(f"🔧 Normalize weights: {args.normalize_weights}")
    print(f"💻 GPU: {args.gpu}")
    print(f"⚠️  VALIDATION DATA ONLY - No test data leakage!")
    

    evaluator = UnifiedEnsembleEvaluator(config_path=args.config)
    
    try:

        if args.mode == 'all':
            results = evaluator.run_mode_all(
                args.task, args.nifti_dir, args.lobe_masks_dir, 
                args.labels_path, args.normalize_weights
            )
            print(f"\n🎉 Comprehensive evaluation completed!")
            print(f"📊 Evaluated all models across all folds")
            
        elif args.mode == 'model':
            results = evaluator.run_mode_model(
                args.task, args.model_type, args.nifti_dir, 
                args.lobe_masks_dir, args.labels_path, args.normalize_weights
            )
            model_name = evaluator.model_configs[args.model_type]['model_name']
            print(f"\n🎉 {model_name} evaluation completed!")
            print(f"📊 Evaluated {model_name} across all folds")
            
        elif args.mode == 'fold':
            results = evaluator.run_mode_fold(
                args.task, args.fold_idx, args.nifti_dir, 
                args.lobe_masks_dir, args.labels_path, args.normalize_weights
            )
            print(f"\n🎉 Fold {args.fold_idx} evaluation completed!")
            print(f"📊 Evaluated all models for fold {args.fold_idx}")
            
        elif args.mode == 'single':
            results = evaluator.run_mode_single(
                args.task, args.model_type, args.fold_idx, args.nifti_dir, 
                args.lobe_masks_dir, args.labels_path, args.normalize_weights
            )
            model_name = evaluator.model_configs[args.model_type]['model_name']
            print(f"\n🎉 {model_name} fold {args.fold_idx} evaluation completed!")
            

        if results:
            print(f"\n✅ Evaluation successful!")
            print(f"🔄 Ready for step 4: MoE SwinUNETR training")
        else:
            print(f"\n⚠️ Evaluation completed but no results generated")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())