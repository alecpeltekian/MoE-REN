#!/usr/bin/env python3
"""
Gating MoE Pipeline Script
Follows the complete pipeline:
1. Train individual experts
2. Extract gating functions
3. Train Gating MoE model
4. Evaluate ensemble performance

Usage:
    python run_gating_pipeline.py --config config_gating.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from experts_training import UnifiedExpertTrainer
from gating_extractor import AutomatedGatingExtractor
from gating_moe import run_unified_training_pipeline
from ensemble_evaluation_gating import UnifiedEnsembleEvaluator

class GatingMoEPipeline:
    """Complete Gating MoE pipeline orchestrator"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.task = self.config['task']
        self.expert_type = self.config['expert_type']
        self.num_experts = self.config['num_experts']
        self.gpu = self.config['gpu']
        self.folds = self.config['folds']
        
        expert_key = f"{self.num_experts}_experts"
        self.expert_config = self.config['expert_config'][expert_key]
        self.expert_indices = self.expert_config['expert_indices']
        self.expert_names = self.expert_config['expert_names']
        
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        
        self.data_paths = self.config['data_paths']
        self.output_config = self.config['output']
        
        print(f"\n{'='*80}")
        print(f"Gating MoE Pipeline Configuration")
        print(f"{'='*80}")
        print(f"Task: {self.task}")
        print(f"Expert Type: {self.expert_type.upper()}")
        print(f"Number of Experts: {self.num_experts}")
        print(f"Expert Names: {', '.join(self.expert_names)}")
        print(f"Expert Indices: {self.expert_indices}")
        print(f"Folds: {self.folds}")
        print(f"GPU: {self.gpu}")
        print(f"{'='*80}\n")
    
    def step1_train_experts(self):
        """Step 1: Train individual expert models"""
        print(f"\n{'='*80}")
        print(f"STEP 1: Training Individual {self.expert_type.upper()} Experts")
        print(f"{'='*80}\n")
        
        expert_results_dir = self.output_config['expert_results_dir']
        model_output_dirs = {
            'cnn': 'unified_cnn_results',
            'mamba': 'unified_mamba_results',
            'vit': 'unified_vit_results'
        }
        
        base_output_dir = Path(expert_results_dir) / model_output_dirs[self.expert_type]
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = UnifiedExpertTrainer(
            task=self.task,
            model_type=self.expert_type,
            nifti_dir=self.data_paths['nifti_dir'],
            lobe_masks_dir=self.data_paths['lobe_masks_dir'],
            labels_file=self.data_paths['labels_file'],
            output_dir=str(base_output_dir.parent),
            gpu=self.gpu
        )
        
        all_results = []
        for fold_idx in self.folds:
            for i, expert_idx in enumerate(self.expert_indices):
                expert_name = self.expert_names[i]
                print(f"\nTraining Expert {expert_idx} ({expert_name}) - Fold {fold_idx}")
                result = trainer.train_single(expert_idx, fold_idx)
                all_results.append(result)
        
        print(f"\n✅ Step 1 Complete: All {len(self.expert_indices)} experts trained for {len(self.folds)} folds")
        return all_results
    
    def step2_extract_gating(self):
        """Step 2: Extract gating functions from trained experts"""
        print(f"\n{'='*80}")
        print(f"STEP 2: Extracting Gating Functions")
        print(f"{'='*80}\n")
        
        expert_results_dir = self.output_config['expert_results_dir']
        gating_results_dir = self.output_config['gating_results_dir']
        
        model_output_dirs = {
            'cnn': 'unified_cnn_results',
            'mamba': 'unified_mamba_results',
            'vit': 'unified_vit_results'
        }
        
        base_results_dir = Path(expert_results_dir) / model_output_dirs[self.expert_type]
        gating_output_dir = Path(gating_results_dir) / model_output_dirs[self.expert_type]
        gating_output_dir.mkdir(parents=True, exist_ok=True)
        
        extractor = AutomatedGatingExtractor(
            task=self.task,
            base_results_dir=str(base_results_dir),
            nifti_dir=self.data_paths['nifti_dir'],
            lobe_masks_dir=self.data_paths['lobe_masks_dir'],
            labels_file=self.data_paths['labels_file'],
            output_dir=str(gating_output_dir),
            gpu=self.gpu,
            max_workers=1
        )
        
        all_results = []
        for fold_idx in self.folds:
            print(f"\nExtracting Gating Functions - Fold {fold_idx}")
            result = extractor.extract_single(self.expert_type, fold_idx)
            if result:
                all_results.append(result)
        
        print(f"\n✅ Step 2 Complete: Gating functions extracted for {len(self.folds)} folds")
        return all_results
    
    def step3_train_gating_moe(self):
        """Step 3: Train Gating MoE model"""
        print(f"\n{'='*80}")
        print(f"STEP 3: Training Gating MoE Model")
        print(f"{'='*80}\n")
        
        expert_results_dir = self.output_config['expert_results_dir']
        gating_results_dir = self.output_config['gating_results_dir']
        moe_output_dir = Path(self.output_config['moe_results_dir']) / f"{self.expert_type}_{self.num_experts}experts"
        moe_output_dir.mkdir(parents=True, exist_ok=True)
        
        model_output_dirs = {
            'cnn': 'unified_cnn_results',
            'mamba': 'unified_mamba_results',
            'vit': 'unified_vit_results'
        }
        
        expert_results_path = Path(expert_results_dir) / model_output_dirs[self.expert_type]
        gating_results_path = Path(gating_results_dir) / model_output_dirs[self.expert_type]
        
        all_results = []
        for fold_idx in self.folds:
            print(f"\nTraining Gating MoE Model - Fold {fold_idx}")
            
            gating_extracted_file = gating_results_path / self.expert_type.lower() / self.task / f"fold_{fold_idx}" / f"extracted_{self.expert_type.lower()}_gating_functions.json"
            
            if not gating_extracted_file.exists():
                print(f"⚠️ Warning: Gating functions file not found: {gating_extracted_file}")
                print(f"   Skipping fold {fold_idx}")
                continue
            
            gating_ensemble_dir = gating_results_path / "gating_ensemble_results" / self.task / f"fold_{fold_idx}"
            gating_ensemble_dir.mkdir(parents=True, exist_ok=True)
            
            with open(gating_extracted_file, 'r') as f:
                gating_data = json.load(f)
            
            best_strategy_name = None
            best_strategy_weights = None
            
            if 'gating_functions' in gating_data:
                gating_funcs = gating_data['gating_functions']
                
                if 'performance_based' in gating_funcs and len(gating_funcs['performance_based']) > 0:
                    best_strategy_name = 'performance_based'
                    perf_funcs = gating_funcs['performance_based']
                    
                    if 'val_auc_softmax' in perf_funcs:
                        weights = perf_funcs['val_auc_softmax']
                        if isinstance(weights, list):
                            best_strategy_weights = weights
                        elif isinstance(weights, np.ndarray):
                            best_strategy_weights = weights.tolist()
                    elif len(perf_funcs) > 0:
                        first_key = list(perf_funcs.keys())[0]
                        weights = perf_funcs[first_key]
                        if isinstance(weights, list):
                            best_strategy_weights = weights
                        elif isinstance(weights, np.ndarray):
                            best_strategy_weights = weights.tolist()
                
                if best_strategy_weights is None and 'feature_based' in gating_funcs:
                    best_strategy_name = 'feature_based'
                    feat_funcs = gating_funcs['feature_based']
                    
                    if 'magnitude_based' in feat_funcs:
                        weights = feat_funcs['magnitude_based']
                        if isinstance(weights, list):
                            best_strategy_weights = weights
                        elif isinstance(weights, np.ndarray):
                            best_strategy_weights = weights.tolist()
                    elif len(feat_funcs) > 0:
                        first_key = list(feat_funcs.keys())[0]
                        weights = feat_funcs[first_key]
                        if isinstance(weights, list):
                            best_strategy_weights = weights
                        elif isinstance(weights, np.ndarray):
                            best_strategy_weights = weights.tolist()
            
            if best_strategy_weights is None:
                print(f"⚠️ Warning: Could not extract gating weights from {gating_extracted_file}")
                print(f"   Available gating functions: {list(gating_data.get('gating_functions', {}).keys())}")
                print(f"   Skipping fold {fold_idx}")
                continue
            
            if len(best_strategy_weights) != len(self.expert_indices):
                print(f"⚠️ Warning: Gating weights length ({len(best_strategy_weights)}) doesn't match number of experts ({len(self.expert_indices)})")
                print(f"   Using first {len(self.expert_indices)} weights")
                best_strategy_weights = best_strategy_weights[:len(self.expert_indices)]
            
            best_strategy_file = gating_ensemble_dir / f"best_{self.expert_type}_gating_strategy.json"
            best_strategy_data = {
                'best_strategy': {
                    'name': best_strategy_name or 'performance_based',
                    'weights': best_strategy_weights,
                    'source': 'extracted_gating_functions'
                },
                'task': self.task,
                'fold': fold_idx,
                'model_type': self.expert_type
            }
            
            with open(best_strategy_file, 'w') as f:
                json.dump(best_strategy_data, f, indent=2)
            
            print(f"✅ Created best gating strategy file: {best_strategy_file}")
            
            result = run_unified_training_pipeline(
                task=self.task,
                fold=fold_idx,
                weight_source_type=self.expert_type,
                expert_results_dir=str(gating_results_path),
                nifti_dir=self.data_paths['nifti_dir'],
                lobe_masks_dir=self.data_paths['lobe_masks_dir'],
                labels_file=self.data_paths['labels_file'],
                weight_mode=self.config['moe_training']['weight_mode'],
                num_epochs=self.config['moe_training']['num_epochs'],
                batch_size=self.config['moe_training']['batch_size'],
                learning_rate=self.config['moe_training']['learning_rate'],
                early_stopping_patience=self.config['moe_training']['early_stopping_patience'],
                output_dir=str(moe_output_dir),
                gpu=self.gpu
            )
            
            if result:
                all_results.append(result)
        
        print(f"\n✅ Step 3 Complete: Gating MoE models trained for {len(self.folds)} folds")
        return all_results
    
    def step4_evaluate_ensemble(self):
        """Step 4: Evaluate ensemble performance"""
        print(f"\n{'='*80}")
        print(f"STEP 4: Evaluating Ensemble Performance")
        print(f"{'='*80}\n")
        
        expert_results_dir = self.output_config['expert_results_dir']
        model_output_dirs = {
            'cnn': 'unified_cnn_results',
            'mamba': 'unified_mamba_results',
            'vit': 'unified_vit_results'
        }
        
        expert_results_path = Path(expert_results_dir) / model_output_dirs[self.expert_type]
        
        evaluator = UnifiedEnsembleEvaluator()
        evaluator.model_configs[self.expert_type]['results_dir'] = str(expert_results_path)
        
        all_results = {}
        for fold_idx in self.folds:
            print(f"\nEvaluating Ensemble - Fold {fold_idx}")
            results = evaluator.run_single_evaluation(
                model_type=self.expert_type,
                task=self.task,
                fold=fold_idx,
                nifti_dir=self.data_paths['nifti_dir'],
                lobe_masks_dir=self.data_paths['lobe_masks_dir'],
                labels_file=self.data_paths['labels_file'],
                normalize_weights=True
            )
            if results:
                all_results[f'fold_{fold_idx}'] = results
        
        print(f"\n✅ Step 4 Complete: Ensemble evaluation done for {len(self.folds)} folds")
        return all_results
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print(f"\n{'='*80}")
        print(f"STARTING COMPLETE GATING MoE PIPELINE")
        print(f"{'='*80}")
        print(f"Task: {self.task}")
        print(f"Expert Type: {self.expert_type.upper()}")
        print(f"Number of Experts: {self.num_experts}")
        print(f"Folds: {self.folds}")
        print(f"{'='*80}\n")
        
        pipeline_start = datetime.now()
        
        expert_results = self.step1_train_experts()
        gating_results = self.step2_extract_gating()
        moe_results = self.step3_train_gating_moe()
        ensemble_results = self.step4_evaluate_ensemble()
        
        pipeline_end = datetime.now()
        pipeline_duration = (pipeline_end - pipeline_start).total_seconds()
        
        summary = {
            'pipeline_type': 'Gating MoE',
            'task': self.task,
            'expert_type': self.expert_type,
            'num_experts': self.num_experts,
            'expert_names': self.expert_names,
            'expert_indices': self.expert_indices,
            'folds': self.folds,
            'start_time': pipeline_start.isoformat(),
            'end_time': pipeline_end.isoformat(),
            'duration_seconds': pipeline_duration,
            'duration_hours': pipeline_duration / 3600,
            'expert_training_results': expert_results,
            'gating_extraction_results': gating_results,
            'gating_moe_training_results': moe_results,
            'ensemble_evaluation_results': ensemble_results
        }
        
        summary_file = Path(self.output_config['base_dir']) / f"gating_pipeline_summary_{self.task}_{self.expert_type}_{self.num_experts}experts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Total Duration: {pipeline_duration/3600:.2f} hours ({pipeline_duration:.0f} seconds)")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*80}\n")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Run Gating MoE Pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration JSON file')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4],
                       help='Run specific step only (1=experts, 2=gating, 3=moe, 4=evaluation)')
    
    args = parser.parse_args()
    
    pipeline = GatingMoEPipeline(args.config)
    
    if args.step:
        if args.step == 1:
            pipeline.step1_train_experts()
        elif args.step == 2:
            pipeline.step2_extract_gating()
        elif args.step == 3:
            pipeline.step3_train_gating_moe()
        elif args.step == 4:
            pipeline.step4_evaluate_ensemble()
    else:
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()

