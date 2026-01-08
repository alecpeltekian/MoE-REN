

import os
import time
import argparse
from datetime import datetime
import json
import numpy as np
from pathlib import Path

from experts import train_single_expert

class UnifiedExpertTrainer:
    
    
    def __init__(self, task, model_type, nifti_dir, lobe_masks_dir, labels_file, output_dir, gpu="0"):
        self.task = task
        self.model_type = model_type.lower()
        self.nifti_dir = nifti_dir
        self.lobe_masks_dir = lobe_masks_dir
        self.labels_file = labels_file
        self.output_dir = output_dir
        self.gpu = gpu
        
        self.expert_names = ["Lobe_1_LUL", "Lobe_2_LLL", "Lobe_3_RUL", "Lobe_4_RML", 
                           "Lobe_5_RLL", "Left_Lung", "Right_Lung"]
        self.all_experts = list(range(7))  # 0 to 6
        self.all_folds = list(range(5))    # 0 to 4
        
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        
        if self.model_type not in ['cnn', 'mamba', 'vit']:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: cnn, mamba, vit")
        
        model_output_dirs = {
            'cnn': 'unified_cnn_results',
            'mamba': 'unified_mamba_results', 
            'vit': 'unified_vit_results'
        }
        
        base_dir = Path(self.output_dir)
        self.model_output_dir = base_dir / model_output_dirs[self.model_type]
        
        if self.model_type == 'mamba':
            try:
                from experts import MAMBA_AVAILABLE
                if not MAMBA_AVAILABLE:
                    raise ImportError("Mamba models require mamba-ssm. Install with: pip install mamba-ssm")
            except ImportError:
                raise ImportError("Mamba models require mamba-ssm. Install with: pip install mamba-ssm")
    
    def train_single(self, expert_idx, fold_idx):
        
        expert_name = self.expert_names[expert_idx]
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} Expert: {expert_name} (Index: {expert_idx})")
        print(f"Task: {self.task}, Fold: {fold_idx}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = train_single_expert(
                task=self.task,
                expert_idx=expert_idx,
                fold_idx=fold_idx,
                nifti_dir=self.nifti_dir,
                lobe_masks_dir=self.lobe_masks_dir,
                labels_file=self.labels_file,
                output_dir=str(self.model_output_dir),
                model_type=self.model_type
            )
            
            duration = time.time() - start_time
            result['job_duration_seconds'] = duration
            result['job_start_time'] = datetime.fromtimestamp(start_time).isoformat()
            result['status'] = 'SUCCESS'
            
            print(f"✅ COMPLETED: {expert_name} - Fold {fold_idx}")
            print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
            print(f"   Test AUC: {result['final_test_metrics']['auc']:.4f}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            print(f"❌ FAILED: {expert_name} - Fold {fold_idx}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Error: {str(e)}")
            
            error_result = {
                'expert_name': expert_name,
                'expert_idx': expert_idx,
                'task': self.task,
                'fold': fold_idx,
                'model_type': self.model_type,
                'status': 'FAILED',
                'error': str(e),
                'job_duration_seconds': duration,
                'job_start_time': datetime.fromtimestamp(start_time).isoformat()
            }
            
            return error_result
    
    def run_all_experts_and_folds(self):
        
        print(f"\n{'='*80}")
        print(f"STARTING FULL TRAINING - {self.model_type.upper()}")
        print(f"{'='*80}")
        print(f"Task: {self.task}")
        print(f"GPU: {self.gpu}")
        print(f"Total jobs: {len(self.all_experts)} experts × {len(self.all_folds)} folds = {len(self.all_experts) * len(self.all_folds)}")
        
        return self._run_job_list([(e, f) for f in self.all_folds for e in self.all_experts])
    
    def run_all_experts_for_fold(self, fold_idx):
        
        print(f"\n{'='*80}")
        print(f"TRAINING ALL EXPERTS FOR FOLD {fold_idx} - {self.model_type.upper()}")
        print(f"{'='*80}")
        print(f"Task: {self.task}")
        print(f"GPU: {self.gpu}")
        print(f"Total jobs: {len(self.all_experts)} experts")
        
        return self._run_job_list([(e, fold_idx) for e in self.all_experts])
    
    def run_all_folds_for_expert(self, expert_idx):
        
        expert_name = self.expert_names[expert_idx]
        print(f"\n{'='*80}")
        print(f"TRAINING ALL FOLDS FOR {expert_name} - {self.model_type.upper()}")
        print(f"{'='*80}")
        print(f"Task: {self.task}")
        print(f"GPU: {self.gpu}")
        print(f"Total jobs: {len(self.all_folds)} folds")
        
        return self._run_job_list([(expert_idx, f) for f in self.all_folds])
    
    def _run_job_list(self, job_list):
        
        overall_start_time = time.time()
        all_results = []
        completed_jobs = 0
        failed_jobs = 0
        total_jobs = len(job_list)
        
        for expert_idx, fold_idx in job_list:
            expert_name = self.expert_names[expert_idx]
            completed_jobs += 1
            
            print(f"\n[Job {completed_jobs}/{total_jobs}] Starting: {expert_name} (Expert {expert_idx}) - Fold {fold_idx}")
            
            result = self.train_single(expert_idx, fold_idx)
            all_results.append(result)
            
            if result['status'] == 'FAILED':
                failed_jobs += 1
            
            remaining_jobs = total_jobs - completed_jobs
            elapsed_time = time.time() - overall_start_time
            avg_time_per_job = elapsed_time / completed_jobs if completed_jobs > 0 else 0
            estimated_remaining_time = remaining_jobs * avg_time_per_job
            
            print(f"   Progress: {completed_jobs}/{total_jobs} completed ({failed_jobs} failed)")
            if remaining_jobs > 0:
                print(f"   Estimated remaining time: {estimated_remaining_time/3600:.1f} hours")
        
        overall_duration = time.time() - overall_start_time
        success_jobs = completed_jobs - failed_jobs
        success_rate = success_jobs / total_jobs if total_jobs > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED - {self.model_type.upper()}")
        print(f"{'='*80}")
        print(f"Task: {self.task}")
        print(f"Total jobs: {total_jobs}")
        print(f"Successful: {success_jobs}")
        print(f"Failed: {failed_jobs}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Total duration: {overall_duration:.1f}s ({overall_duration/3600:.1f} hours)")
        print(f"Average time per job: {overall_duration/total_jobs:.1f}s" if total_jobs > 0 else "N/A")
        
        summary = {
            'task': self.task,
            'model_type': self.model_type,
            'completion_time': datetime.now().isoformat(),
            'total_jobs': total_jobs,
            'successful_jobs': success_jobs,
            'failed_jobs': failed_jobs,
            'success_rate': success_rate,
            'total_duration_seconds': overall_duration,
            'total_duration_hours': overall_duration / 3600,
            'average_time_per_job_seconds': overall_duration / total_jobs if total_jobs > 0 else 0,
            'all_results': all_results,
            'job_list': job_list
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.model_output_dir / f"training_summary_{self.model_type}_{self.task}_{timestamp}.json"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary saved to: {summary_file}")
        
        self._print_detailed_summary(all_results)
        
        return summary
    
    def _print_detailed_summary(self, all_results):
        
        successful_results = [r for r in all_results if r.get('status') == 'SUCCESS']
        
        if successful_results:
            print(f"\nDETAILED RESULTS:")
            
            folds_present = set(r.get('fold') for r in successful_results)
            if len(folds_present) > 1:
                print(f"\nPER-FOLD SUMMARY:")
                for fold_idx in sorted(folds_present):
                    fold_results = [r for r in successful_results if r.get('fold') == fold_idx]
                    if fold_results:
                        avg_auc = sum(r.get('final_test_metrics', {}).get('auc', 0) for r in fold_results) / len(fold_results)
                        print(f"  Fold {fold_idx}: {len(fold_results)} experts, Avg AUC: {avg_auc:.4f}")
            
            experts_present = set(r.get('expert_idx') for r in successful_results)
            if len(experts_present) > 1:
                print(f"\nPER-EXPERT SUMMARY:")
                for expert_idx in sorted(experts_present):
                    expert_results = [r for r in successful_results if r.get('expert_idx') == expert_idx]
                    if expert_results:
                        expert_name = self.expert_names[expert_idx]
                        avg_auc = sum(r.get('final_test_metrics', {}).get('auc', 0) for r in expert_results) / len(expert_results)
                        print(f"  {expert_name}: {len(expert_results)} folds, Avg AUC: {avg_auc:.4f}")
            
            all_aucs = [r.get('final_test_metrics', {}).get('auc', 0) for r in successful_results]
            if all_aucs:
                print(f"\nOVERALL STATISTICS:")
                print(f"  Mean AUC: {np.mean(all_aucs):.4f}")
                print(f"  Std AUC: {np.std(all_aucs):.4f}")
                print(f"  Min AUC: {np.min(all_aucs):.4f}")
                print(f"  Max AUC: {np.max(all_aucs):.4f}")
        
        failed_results = [r for r in all_results if r.get('status') == 'FAILED']
        if failed_results:
            print(f"\nFAILED JOBS:")
            for result in failed_results:
                expert_name = result.get('expert_name', 'Unknown')
                fold = result.get('fold', 'Unknown')
                error = result.get('error', 'Unknown error')
                print(f"  {expert_name} - Fold {fold}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Expert Training Script using consolidated models module',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--task', type=str, required=True,
                       help='Task to analyze (e.g., has_ILD, death_3, etc.)')
    parser.add_argument('--model_type', type=str, required=True, 
                       choices=['cnn', 'mamba', 'vit'],
                       help='Model type to use')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['all', 'fold', 'expert', 'single'],
                       help='Execution mode')
    
    parser.add_argument('--expert_idx', type=int, choices=range(7),
                       help='Expert index (0-6) - required for "expert" and "single" modes')
    parser.add_argument('--fold_idx', type=int, choices=range(5),
                       help='Fold index (0-4) - required for "fold" and "single" modes')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--nifti_dir', type=str, default=None,
                       help='Directory containing NIFTI files')
    parser.add_argument('--lobe_masks_dir', type=str, default=None,
                       help='Directory containing lobe masks')
    parser.add_argument('--labels_path', type=str, default=None,
                       help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base directory to save results (model-specific subdirs will be created)')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device number to use')
    
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be executed without actually running training')
    
    args = parser.parse_args()
    
    import json
    from pathlib import Path
    
    if not Path(args.config).exists():
        parser.error(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    data_paths = config.get('data_paths', {})
    output_config = config.get('output', {})
    
    if args.nifti_dir is None:
        args.nifti_dir = data_paths.get('nifti_dir')
    if args.lobe_masks_dir is None:
        args.lobe_masks_dir = data_paths.get('lobe_masks_dir')
    if args.labels_path is None:
        args.labels_path = data_paths.get('labels_file')
    if args.output_dir is None:
        args.output_dir = output_config.get('base_dir')
    
    if not args.nifti_dir or not args.lobe_masks_dir or not args.labels_path or not args.output_dir:
        parser.error("Missing required paths in config. Please check data_paths and output sections.")
    
    if args.mode in ['expert', 'single'] and args.expert_idx is None:
        parser.error(f"--expert_idx is required for mode '{args.mode}'")
    
    if args.mode in ['fold', 'single'] and args.fold_idx is None:
        parser.error(f"--fold_idx is required for mode '{args.mode}'")
    
    print(f"="*80)
    print(f"UNIFIED EXPERT TRAINING")
    print(f"="*80)
    print(f"Task: {args.task}")
    print(f"Model: {args.model_type.upper()}")
    print(f"Mode: {args.mode}")
    print(f"GPU: {args.gpu}")
    
    expert_names = ["Lobe_1_LUL", "Lobe_2_LLL", "Lobe_3_RUL", "Lobe_4_RML", 
                   "Lobe_5_RLL", "Left_Lung", "Right_Lung"]
    
    if args.mode in ['expert', 'single']:
        print(f"Expert: {expert_names[args.expert_idx]} (Index: {args.expert_idx})")
    if args.mode in ['fold', 'single']:
        print(f"Fold: {args.fold_idx}")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - No training will be executed")
        
        job_list = []
        if args.mode == 'all':
            job_list = [(e, f) for f in range(5) for e in range(7)]
        elif args.mode == 'fold':
            job_list = [(e, args.fold_idx) for e in range(7)]
        elif args.mode == 'expert':
            job_list = [(args.expert_idx, f) for f in range(5)]
        elif args.mode == 'single':
            job_list = [(args.expert_idx, args.fold_idx)]
        
        print(f"\nJobs that would be executed ({len(job_list)} total):")
        for i, (expert_idx, fold_idx) in enumerate(job_list, 1):
            print(f"  {i:2d}. {expert_names[expert_idx]} (Expert {expert_idx}) - Fold {fold_idx}")
        
        print(f"\nModel configuration for {args.model_type.upper()}:")
        try:
            from experts import get_model_config
            config = get_model_config(args.model_type)
            for key, value in config.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Could not load config: {e}")
        
        return
    
    try:
        trainer = UnifiedExpertTrainer(
            task=args.task,
            model_type=args.model_type,
            nifti_dir=args.nifti_dir,
            lobe_masks_dir=args.lobe_masks_dir,
            labels_file=args.labels_path,
            output_dir=args.output_dir,
            gpu=args.gpu
        )
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        exit(1)
    
    try:
        if args.mode == 'all':
            summary = trainer.run_all_experts_and_folds()
            
        elif args.mode == 'fold':
            summary = trainer.run_all_experts_for_fold(args.fold_idx)
            
        elif args.mode == 'expert':
            summary = trainer.run_all_folds_for_expert(args.expert_idx)
            
        elif args.mode == 'single':
            result = trainer.train_single(args.expert_idx, args.fold_idx)
            summary = {
                'task': args.task,
                'model_type': args.model_type,
                'mode': 'single',
                'total_jobs': 1,
                'successful_jobs': 1 if result['status'] == 'SUCCESS' else 0,
                'failed_jobs': 0 if result['status'] == 'SUCCESS' else 1,
                'all_results': [result]
            }
        
        if summary['failed_jobs'] > 0:
            print(f"\n⚠️  {summary['failed_jobs']} job(s) failed - check logs for details")
            exit(1)
        else:
            print(f"\n✅ All {summary['total_jobs']} job(s) completed successfully!")
            exit(0)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        exit(1)


if __name__ == "__main__":
    main()





