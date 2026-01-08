#!/usr/bin/env python3
"""
MoE Pipeline Script (Normal MoE with val_auc weighting)
Follows the complete pipeline:
1. Train individual experts
2. Train MoE model with learned weights
3. Evaluate ensemble performance

Usage:
    python run_moe_pipeline.py --config config_moe.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from experts_training import UnifiedExpertTrainer
from moe import UnifiedMoETrainer
from ensemble_evaluation import UnifiedEnsembleEvaluator
from moe_flexible import MoEWeightedSwinUNETR

class MoEPipeline:
    """Complete MoE pipeline orchestrator"""
    
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
        print(f"MoE Pipeline Configuration")
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
    
    def step2_train_moe(self):
        """Step 2: Train MoE model with learned weights"""
        print(f"\n{'='*80}")
        print(f"STEP 2: Training MoE Model with {self.expert_type.upper()} Experts")
        print(f"{'='*80}\n")
        
        expert_results_dir = self.output_config['expert_results_dir']
        model_output_dirs = {
            'cnn': 'unified_cnn_results',
            'mamba': 'unified_mamba_results',
            'vit': 'unified_vit_results'
        }
        
        expert_results_path = Path(expert_results_dir) / model_output_dirs[self.expert_type]
        
        moe_output_dir = Path(self.output_config['moe_results_dir']) / f"{self.expert_type}_{self.num_experts}experts"
        moe_output_dir.mkdir(parents=True, exist_ok=True)
        
        base_args = {
            'task': self.task,
            'expert_type': self.expert_type,
            'expert_results_dir': str(expert_results_path),
            'nifti_dir': self.data_paths['nifti_dir'],
            'lobe_masks_dir': self.data_paths['lobe_masks_dir'],
            'labels_file': self.data_paths['labels_file'],
            'base_output_dir': str(moe_output_dir),
            'gpu': self.gpu,
            'num_epochs': self.config['moe_training']['num_epochs'],
            'batch_size': self.config['moe_training']['batch_size'],
            'learning_rate': self.config['moe_training']['learning_rate'],
            'early_stopping_patience': self.config['moe_training']['early_stopping_patience'],
            'weight_mode': self.config['moe_training']['weight_mode'],
            'num_experts': self.num_experts,
            'expert_indices': self.expert_indices,
            'expert_names': self.expert_names
        }
        
        moe_trainer = UnifiedMoETrainer(base_args)
        
        all_results = []
        for fold_idx in self.folds:
            print(f"\nTraining MoE Model - Fold {fold_idx}")
            result = moe_trainer.train_single_model(
                expert_type=self.expert_type,
                fold_idx=fold_idx,
                weight_mode=self.config['moe_training']['weight_mode']
            )
            if result:
                all_results.append(result)
        
        moe_trainer.save_final_summary()
        print(f"\n✅ Step 2 Complete: MoE models trained for {len(self.folds)} folds")
        return all_results
    
    def step3_evaluate_ensemble(self):
        """Step 3: Evaluate ensemble performance"""
        print(f"\n{'='*80}")
        print(f"STEP 3: Evaluating Ensemble Performance")
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
                task=self.task,
                model_type=self.expert_type,
                fold=fold_idx,
                nifti_dir=self.data_paths['nifti_dir'],
                lobe_masks_dir=self.data_paths['lobe_masks_dir'],
                labels_file=self.data_paths['labels_file'],
                ensemble_methods=[self.config['ensemble']['method']],
                weight_strategy=self.config['ensemble']['weight_strategy']
            )
            if results:
                all_results[f'fold_{fold_idx}'] = results
        
        print(f"\n✅ Step 3 Complete: Ensemble evaluation done for {len(self.folds)} folds")
        return all_results
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print(f"\n{'='*80}")
        print(f"STARTING COMPLETE MoE PIPELINE")
        print(f"{'='*80}")
        print(f"Task: {self.task}")
        print(f"Expert Type: {self.expert_type.upper()}")
        print(f"Number of Experts: {self.num_experts}")
        print(f"Folds: {self.folds}")
        print(f"{'='*80}\n")
        
        pipeline_start = datetime.now()
        
        expert_results = self.step1_train_experts()
        moe_results = self.step2_train_moe()
        ensemble_results = self.step3_evaluate_ensemble()
        
        pipeline_end = datetime.now()
        pipeline_duration = (pipeline_end - pipeline_start).total_seconds()
        
        summary = {
            'pipeline_type': 'MoE',
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
            'moe_training_results': moe_results,
            'ensemble_evaluation_results': ensemble_results
        }
        
        summary_file = Path(self.output_config['base_dir']) / f"moe_pipeline_summary_{self.task}_{self.expert_type}_{self.num_experts}experts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    parser = argparse.ArgumentParser(description='Run MoE Pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration JSON file')
    parser.add_argument('--step', type=int, choices=[1, 2, 3],
                       help='Run specific step only (1=experts, 2=moe, 3=evaluation)')
    
    args = parser.parse_args()
    
    pipeline = MoEPipeline(args.config)
    
    if args.step:
        if args.step == 1:
            pipeline.step1_train_experts()
        elif args.step == 2:
            pipeline.step2_train_moe()
        elif args.step == 3:
            pipeline.step3_evaluate_ensemble()
    else:
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()

