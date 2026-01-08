

import os
import time
import argparse
from datetime import datetime
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import sys


class AutomatedGatingExtractor:
    
    
    def __init__(self, task, base_results_dir, nifti_dir, lobe_masks_dir, labels_file, 
                 output_dir, gpu="0", max_workers=1):
        self.task = task
        self.base_results_dir = Path(base_results_dir)
        self.nifti_dir = nifti_dir
        self.lobe_masks_dir = lobe_masks_dir
        self.labels_file = labels_file
        self.output_dir = output_dir
        self.gpu = gpu
        self.max_workers = max_workers
        
        self.model_configs = {
            'cnn': {
                'results_dir': self.base_results_dir / 'unified_cnn_results',
                'available': True
            },
            'mamba': {
                'results_dir': self.base_results_dir / 'unified_mamba_results',
                'available': self._check_mamba_availability()
            },
            'vit': {
                'results_dir': self.base_results_dir / 'unified_vit_results',
                'available': True
            }
        }
        
        self.all_folds = list(range(5))  # 0 to 4
        
    def _check_mamba_availability(self):
        
        try:
            from experts import MAMBA_AVAILABLE
            return MAMBA_AVAILABLE
        except ImportError:
            return False
    
    def check_model_availability(self, task, model_type, fold_idx):
        
        model_config = self.model_configs[model_type]
        if not model_config['available']:
            return False, f"{model_type.upper()} not available"
        
        results_dir = model_config['results_dir']
        fold_dir = results_dir / task / f"fold_{fold_idx}"
        
        if not fold_dir.exists():
            return False, f"Fold directory not found: {fold_dir}"
        
        expert_names = ["Lobe_1_LUL", "Lobe_2_LLL", "Lobe_3_RUL", "Lobe_4_RML", 
                       "Lobe_5_RLL", "Left_Lung", "Right_Lung"]
        
        found_models = 0
        for expert_name in expert_names:
            model_file = fold_dir / f"{expert_name}_{model_type}_final_model.pth"
            results_file = fold_dir / f"{expert_name}_{model_type}_results.json"
            
            if model_file.exists() and results_file.exists():
                found_models += 1
        
        if found_models == 0:
            return False, f"No trained {model_type.upper()} models found in {fold_dir}"
        elif found_models < len(expert_names):
            return True, f"Partial models available ({found_models}/{len(expert_names)} experts)"
        else:
            return True, f"All models available ({found_models}/{len(expert_names)} experts)"
    
    def extract_single(self, model_type, fold_idx):
        
        start_time = time.time()
        
        available, status_msg = self.check_model_availability(self.task, model_type, fold_idx)
        if not available:
            return {
                'task': self.task,
                'model_type': model_type,
                'fold': fold_idx,
                'status': 'SKIPPED',
                'reason': status_msg,
                'duration_seconds': 0
            }
        
        print(f"\n{'='*60}")
        print(f"Extracting {model_type.upper()} Gating - Fold {fold_idx}")
        print(f"Task: {self.task}")
        print(f"Status: {status_msg}")
        print(f"{'='*60}")
        
        try:
            expert_results_dir = self.model_configs[model_type]['results_dir']
            
            result = extract_gating_functions(
                task=self.task,
                fold=fold_idx,
                model_type=model_type,
                expert_results_dir=str(expert_results_dir),
                nifti_dir=self.nifti_dir,
                lobe_masks_dir=self.lobe_masks_dir,
                labels_file=self.labels_file,
                output_dir=self.output_dir,
                device=f'cuda:{self.gpu}'
            )
            
            duration = time.time() - start_time
            
            if result is not None:
                print(f"✅ COMPLETED: {model_type.upper()} - Fold {fold_idx}")
                print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
                
                return {
                    'task': self.task,
                    'model_type': model_type,
                    'fold': fold_idx,
                    'status': 'SUCCESS',
                    'duration_seconds': duration,
                    'validation_samples': result.get('validation_samples', 0),
                    'extracted_functions': list(result.get('gating_functions', {}).keys())
                }
            else:
                print(f"❌ FAILED: {model_type.upper()} - Fold {fold_idx} (No result returned)")
                return {
                    'task': self.task,
                    'model_type': model_type,
                    'fold': fold_idx,
                    'status': 'FAILED',
                    'reason': 'No result returned',
                    'duration_seconds': duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ FAILED: {model_type.upper()} - Fold {fold_idx}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Error: {str(e)}")
            
            return {
                'task': self.task,
                'model_type': model_type,
                'fold': fold_idx,
                'status': 'FAILED',
                'reason': str(e),
                'duration_seconds': duration
            }
    
    def run_all_models_and_folds(self):
        
        print(f"\n{'='*80}")
        print(f"AUTOMATED GATING EXTRACTION - ALL MODELS")
        print(f"{'='*80}")
        print(f"Task: {self.task}")
        print(f"GPU: {self.gpu}")
        
        job_list = []
        for model_type in ['cnn', 'mamba', 'vit']:
            if self.model_configs[model_type]['available']:
                for fold_idx in self.all_folds:
                    job_list.append((model_type, fold_idx))
            else:
                print(f"⚠️  Skipping {model_type.upper()}: Not available")
        
        print(f"Total jobs: {len(job_list)}")
        
        return self._run_job_list(job_list)
    
    def run_all_folds_for_model(self, model_type):
        
        print(f"\n{'='*80}")
        print(f"AUTOMATED GATING EXTRACTION - {model_type.upper()} ALL FOLDS")
        print(f"{'='*80}")
        print(f"Task: {self.task}")
        print(f"GPU: {self.gpu}")
        
        if not self.model_configs[model_type]['available']:
            print(f"❌ {model_type.upper()} is not available")
            return {
                'task': self.task,
                'model_type': model_type,
                'status': 'UNAVAILABLE',
                'total_jobs': 0,
                'successful_jobs': 0,
                'failed_jobs': 0,
                'all_results': []
            }
        
        job_list = [(model_type, fold_idx) for fold_idx in self.all_folds]
        print(f"Total jobs: {len(job_list)} folds")
        
        return self._run_job_list(job_list)
    
    def run_all_models_for_fold(self, fold_idx):
        
        print(f"\n{'='*80}")
        print(f"AUTOMATED GATING EXTRACTION - ALL MODELS FOLD {fold_idx}")
        print(f"{'='*80}")
        print(f"Task: {self.task}")
        print(f"GPU: {self.gpu}")
        
        job_list = []
        for model_type in ['cnn', 'mamba', 'vit']:
            if self.model_configs[model_type]['available']:
                job_list.append((model_type, fold_idx))
            else:
                print(f"⚠️  Skipping {model_type.upper()}: Not available")
        
        print(f"Total jobs: {len(job_list)} models")
        
        return self._run_job_list(job_list)
    
    def _run_job_list(self, job_list):
        
        overall_start_time = time.time()
        all_results = []
        completed_jobs = 0
        failed_jobs = 0
        skipped_jobs = 0
        total_jobs = len(job_list)
        
        for model_type, fold_idx in job_list:
            completed_jobs += 1
            
            print(f"\n[Job {completed_jobs}/{total_jobs}] Starting: {model_type.upper()} - Fold {fold_idx}")
            
            result = self.extract_single(model_type, fold_idx)
            all_results.append(result)
            
            if result['status'] == 'FAILED':
                failed_jobs += 1
            elif result['status'] == 'SKIPPED':
                skipped_jobs += 1
            
            remaining_jobs = total_jobs - completed_jobs
            elapsed_time = time.time() - overall_start_time
            avg_time_per_job = elapsed_time / completed_jobs if completed_jobs > 0 else 0
            estimated_remaining_time = remaining_jobs * avg_time_per_job
            
            success_jobs = completed_jobs - failed_jobs - skipped_jobs
            print(f"   Progress: {completed_jobs}/{total_jobs} completed ({success_jobs} success, {failed_jobs} failed, {skipped_jobs} skipped)")
            if remaining_jobs > 0:
                print(f"   Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
        
        overall_duration = time.time() - overall_start_time
        success_jobs = total_jobs - failed_jobs - skipped_jobs
        success_rate = success_jobs / total_jobs if total_jobs > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"GATING EXTRACTION COMPLETED")
        print(f"{'='*80}")
        print(f"Task: {self.task}")
        print(f"Total jobs: {total_jobs}")
        print(f"Successful: {success_jobs}")
        print(f"Failed: {failed_jobs}")
        print(f"Skipped: {skipped_jobs}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Total duration: {overall_duration:.1f}s ({overall_duration/60:.1f} minutes)")
        print(f"Average time per job: {overall_duration/total_jobs:.1f}s" if total_jobs > 0 else "N/A")
        
        summary = {
            'task': self.task,
            'completion_time': datetime.now().isoformat(),
            'total_jobs': total_jobs,
            'successful_jobs': success_jobs,
            'failed_jobs': failed_jobs,
            'skipped_jobs': skipped_jobs,
            'success_rate': success_rate,
            'total_duration_seconds': overall_duration,
            'total_duration_minutes': overall_duration / 60,
            'average_time_per_job_seconds': overall_duration / total_jobs if total_jobs > 0 else 0,
            'all_results': all_results,
            'job_list': job_list,
            'model_availability': {model: config['available'] for model, config in self.model_configs.items()}
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = Path(self.output_dir) / f"gating_extraction_summary_{self.task}_{timestamp}.json"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary saved to: {summary_file}")
        
        self._print_detailed_summary(all_results)
        
        return summary
    
    def _print_detailed_summary(self, all_results):
        
        successful_results = [r for r in all_results if r.get('status') == 'SUCCESS']
        failed_results = [r for r in all_results if r.get('status') == 'FAILED']
        skipped_results = [r for r in all_results if r.get('status') == 'SKIPPED']
        
        if successful_results:
            print(f"\nSUCCESSFUL EXTRACTIONS ({len(successful_results)}):")
            
            by_model = {}
            for result in successful_results:
                model_type = result['model_type']
                if model_type not in by_model:
                    by_model[model_type] = []
                by_model[model_type].append(result)
            
            for model_type, results in by_model.items():
                folds = [r['fold'] for r in results]
                avg_duration = np.mean([r['duration_seconds'] for r in results])
                total_samples = sum(r.get('validation_samples', 0) for r in results)
                
                print(f"  {model_type.upper()}: {len(results)} folds {folds}")
                print(f"    Avg duration: {avg_duration:.1f}s, Total samples: {total_samples}")
        
        if failed_results:
            print(f"\nFAILED EXTRACTIONS ({len(failed_results)}):")
            for result in failed_results:
                model_type = result['model_type']
                fold = result['fold']
                reason = result.get('reason', 'Unknown')
                print(f"  {model_type.upper()} - Fold {fold}: {reason}")
        
        if skipped_results:
            print(f"\nSKIPPED EXTRACTIONS ({len(skipped_results)}):")
            for result in skipped_results:
                model_type = result['model_type']
                fold = result['fold']
                reason = result.get('reason', 'Unknown')
                print(f"  {model_type.upper()} - Fold {fold}: {reason}")
    
    def create_comparison_report(self, summary):
        
        print(f"\nCreating comparison report...")
        
        successful_results = [r for r in summary['all_results'] if r['status'] == 'SUCCESS']
        
        if len(successful_results) == 0:
            print("No successful extractions to compare")
            return
        
        comparison_data = {
            'task': self.task,
            'creation_time': datetime.now().isoformat(),
            'total_extractions': len(successful_results),
            'models_compared': {},
            'cross_model_analysis': {}
        }
        
        for result in successful_results:
            model_type = result['model_type']
            fold = result['fold']
            
            results_path = Path(self.output_dir) / model_type.lower() / self.task / f"fold_{fold}" / f"extracted_{model_type.lower()}_gating_functions.json"
            
            if results_path.exists():
                try:
                    with open(results_path, 'r') as f:
                        gating_data = json.load(f)
                    
                    if model_type not in comparison_data['models_compared']:
                        comparison_data['models_compared'][model_type] = {}
                    
                    comparison_data['models_compared'][model_type][f'fold_{fold}'] = {
                        'gating_functions': gating_data['gating_functions'],
                        'expert_performance': gating_data['expert_performance'],
                        'validation_samples': gating_data['validation_samples']
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not load gating data for {model_type} fold {fold}: {e}")
        
        self._analyze_cross_model_patterns(comparison_data)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = Path(self.output_dir) / f"gating_comparison_report_{self.task}_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        print(f"Comparison report saved to: {report_file}")
        
        markdown_file = Path(self.output_dir) / f"gating_comparison_summary_{self.task}_{timestamp}.md"
        self._create_markdown_summary(comparison_data, markdown_file)
        print(f"Markdown summary saved to: {markdown_file}")
    
    def _analyze_cross_model_patterns(self, comparison_data):
        
        models_data = comparison_data['models_compared']
        
        if len(models_data) < 2:
            return
        
        common_folds = None
        for model_type, model_data in models_data.items():
            folds = set(model_data.keys())
            if common_folds is None:
                common_folds = folds
            else:
                common_folds = common_folds.intersection(folds)
        
        if not common_folds:
            return
        
        cross_analysis = {
            'common_folds': list(common_folds),
            'expert_ranking_consistency': {},
            'gating_method_comparison': {},
            'performance_correlation': {}
        }
        
        expert_names = ["Lobe_1_LUL", "Lobe_2_LLL", "Lobe_3_RUL", "Lobe_4_RML", 
                       "Lobe_5_RLL", "Left_Lung", "Right_Lung"]
        
        for fold in common_folds:
            fold_analysis = {}
            
            for gating_category in ['performance_based', 'feature_based', 'learned_based']:
                for model_type, model_data in models_data.items():
                    if fold in model_data:
                        gating_functions = model_data[fold]['gating_functions']
                        if gating_category in gating_functions:
                            for method_name, weights in gating_functions[gating_category].items():
                                if isinstance(weights, list) and len(weights) == len(expert_names):
                                    key = f"{gating_category}_{method_name}"
                                    if key not in fold_analysis:
                                        fold_analysis[key] = {}
                                    fold_analysis[key][model_type] = weights
            
            cross_analysis['gating_method_comparison'][fold] = fold_analysis
        
        comparison_data['cross_model_analysis'] = cross_analysis
    
    def _create_markdown_summary(self, comparison_data, output_file):
        
        with open(output_file, 'w') as f:
            f.write(f"# Gating Function Comparison Report\n\n")
            f.write(f"**Task:** {comparison_data['task']}\n")
            f.write(f"**Created:** {comparison_data['creation_time']}\n")
            f.write(f"**Total Extractions:** {comparison_data['total_extractions']}\n\n")
            
            f.write("## Model Overview\n\n")
            for model_type, model_data in comparison_data['models_compared'].items():
                folds = list(model_data.keys())
                f.write(f"### {model_type.upper()}\n")
                f.write(f"- **Folds processed:** {len(folds)} ({', '.join(folds)})\n")
                
                if folds:
                    sample_fold = list(model_data.values())[0]
                    total_samples = sample_fold.get('validation_samples', 0)
                    f.write(f"- **Validation samples per fold:** ~{total_samples}\n")
                f.write("\n")
            
            cross_analysis = comparison_data.get('cross_model_analysis', {})
            if cross_analysis and cross_analysis.get('common_folds'):
                f.write("## Cross-Model Analysis\n\n")
                f.write(f"**Common folds:** {', '.join(cross_analysis['common_folds'])}\n\n")
                
                gating_methods = set()
                for fold_data in cross_analysis['gating_method_comparison'].values():
                    gating_methods.update(fold_data.keys())
                
                f.write(f"**Available gating methods:** {len(gating_methods)}\n")
                for method in sorted(gating_methods):
                    f.write(f"- {method}\n")
                f.write("\n")
            
            f.write("## Usage Instructions\n\n")
            f.write("### Loading Gating Weights\n\n")
            f.write("```python\n")
            f.write("import json\n")
            f.write("import numpy as np\n\n")
            f.write("# Load gating results for a specific model and fold\n")
            f.write("with open('extracted_cnn_gating_functions.json', 'r') as f:\n")
            f.write("    cnn_gating = json.load(f)\n\n")
            f.write("# Extract specific gating weights\n")
            f.write("val_auc_weights = cnn_gating['gating_functions']['performance_based']['val_auc_softmax']\n")
            f.write("learned_weights = cnn_gating['gating_functions']['learned_based']['learned_softmax']\n\n")
            f.write("# Apply to expert predictions\n")
            f.write("expert_predictions = [pred1, pred2, pred3, pred4, pred5, pred6, pred7]\n")
            f.write("ensemble_prediction = sum(w * p for w, p in zip(val_auc_weights, expert_predictions))\n")
            f.write("```\n\n")
            
            f.write("## Generated Files Structure\n\n")
            f.write("```\n")
            f.write("extracted_gating_results/\n")
            f.write("├── cnn/\n")
            f.write("│   └── {task}/\n")
            f.write("│       ├── fold_0/\n")
            f.write("│       │   ├── extracted_cnn_gating_functions.json\n")
            f.write("│       │   └── cnn_gating_equations.md\n")
            f.write("│       └── fold_1/ ... fold_4/\n")
            f.write("├── mamba/ (if available)\n")
            f.write("│   └── {task}/fold_X/\n")
            f.write("└── vit/\n")
            f.write("    └── {task}/fold_X/\n")
            f.write("```\n")


def main():
    parser = argparse.ArgumentParser(
        description='Automated Gating Function Extraction from Unified Training Results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--task', type=str, required=True,
                       help='Task to analyze (e.g., has_ILD, death_3, etc.)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['all', 'model', 'fold', 'single'],
                       help='Extraction mode')
    
    parser.add_argument('--model_type', type=str, choices=['cnn', 'mamba', 'vit'],
                       help='Model type - required for "model" and "single" modes')
    parser.add_argument('--fold_idx', type=int, choices=range(5),
                       help='Fold index (0-4) - required for "fold" and "single" modes')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config JSON file')
    parser.add_argument('--base_results_dir', type=str, default=None,
                       help='Base directory containing unified training results')
    parser.add_argument('--nifti_dir', type=str, default=None,
                       help='Directory containing NIFTI files')
    parser.add_argument('--lobe_masks_dir', type=str, default=None,
                       help='Directory containing lobe masks')
    parser.add_argument('--labels_path', type=str, default=None,
                       help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save gating extraction results')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device number to use')
    
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be extracted without actually running')
    parser.add_argument('--create_report', action='store_true',
                       help='Create comparison report after extraction')
    
    args = parser.parse_args()
    
    import json
    from pathlib import Path
    
    if not Path(args.config).exists():
        parser.error(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    data_paths = config.get('data_paths', {})
    output_config = config.get('output', {})
    
    if args.base_results_dir is None:
        args.base_results_dir = output_config.get('base_dir')
    if args.nifti_dir is None:
        args.nifti_dir = data_paths.get('nifti_dir')
    if args.lobe_masks_dir is None:
        args.lobe_masks_dir = data_paths.get('lobe_masks_dir')
    if args.labels_path is None:
        args.labels_path = data_paths.get('labels_file')
    if args.output_dir is None:
        args.output_dir = output_config.get('extracted_gating_results')
    
    if not args.base_results_dir or not args.nifti_dir or not args.lobe_masks_dir or not args.labels_path or not args.output_dir:
        parser.error("Missing required paths in config. Please check data_paths and output sections.")
    
    if args.mode in ['model', 'single'] and args.model_type is None:
        parser.error(f"--model_type is required for mode '{args.mode}'")
    
    if args.mode in ['fold', 'single'] and args.fold_idx is None:
        parser.error(f"--fold_idx is required for mode '{args.mode}'")
    
    print(f"="*80)
    print(f"AUTOMATED GATING EXTRACTION")
    print(f"="*80)
    print(f"Task: {args.task}")
    print(f"Mode: {args.mode}")
    print(f"GPU: {args.gpu}")
    
    if args.mode in ['model', 'single']:
        print(f"Model: {args.model_type.upper()}")
    if args.mode in ['fold', 'single']:
        print(f"Fold: {args.fold_idx}")
    
    try:
        extractor = AutomatedGatingExtractor(
            task=args.task,
            base_results_dir=args.base_results_dir,
            nifti_dir=args.nifti_dir,
            lobe_masks_dir=args.lobe_masks_dir,
            labels_file=args.labels_path,
            output_dir=args.output_dir,
            gpu=args.gpu
        )
    except Exception as e:
        print(f"❌ Failed to create extractor: {e}")
        exit(1)
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - No extraction will be performed")
        
        if args.mode == 'all':
            job_list = [(m, f) for m in ['cnn', 'mamba', 'vit'] for f in range(5)]
        elif args.mode == 'model':
            job_list = [(args.model_type, f) for f in range(5)]
        elif args.mode == 'fold':
            job_list = [(m, args.fold_idx) for m in ['cnn', 'mamba', 'vit']]
        elif args.mode == 'single':
            job_list = [(args.model_type, args.fold_idx)]
        
        print(f"\nJobs that would be executed ({len(job_list)} total):")
        available_jobs = 0
        for i, (model_type, fold_idx) in enumerate(job_list, 1):
            available, status = extractor.check_model_availability(args.task, model_type, fold_idx)
            status_icon = "✅" if available else "❌"
            print(f"  {i:2d}. {status_icon} {model_type.upper()} - Fold {fold_idx}: {status}")
            if available:
                available_jobs += 1
        
        print(f"\nSummary: {available_jobs}/{len(job_list)} jobs available for extraction")
        return
    
    try:
        if args.mode == 'all':
            summary = extractor.run_all_models_and_folds()
            
        elif args.mode == 'model':
            summary = extractor.run_all_folds_for_model(args.model_type)
            
        elif args.mode == 'fold':
            summary = extractor.run_all_models_for_fold(args.fold_idx)
            
        elif args.mode == 'single':
            result = extractor.extract_single(args.model_type, args.fold_idx)
            summary = {
                'task': args.task,
                'mode': 'single',
                'total_jobs': 1,
                'successful_jobs': 1 if result['status'] == 'SUCCESS' else 0,
                'failed_jobs': 0 if result['status'] == 'SUCCESS' else 1,
                'skipped_jobs': 0 if result['status'] != 'SKIPPED' else 1,
                'all_results': [result]
            }
        
        if args.create_report and summary['successful_jobs'] > 0:
            extractor.create_comparison_report(summary)
        
        if summary['failed_jobs'] > 0:
            print(f"\n⚠️  {summary['failed_jobs']} extraction(s) failed - check logs for details")
            exit(1)
        elif summary['successful_jobs'] == 0:
            print(f"\n⚠️  No successful extractions - check model availability")
            exit(1)
        else:
            print(f"\n✅ All {summary['successful_jobs']} extraction(s) completed successfully!")
            exit(0)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Extraction interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Extraction failed with error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
