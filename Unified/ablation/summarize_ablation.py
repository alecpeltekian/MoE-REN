#!/usr/bin/env python3


import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_all_fold_results(base_dir: str, model_type: str, task: str) -> pd.DataFrame:
    
    all_results = []
    
    for fold in range(5):
        fold_dir = Path(base_dir) / model_type / task / f"fold_{fold}"
        result_file = fold_dir / "ablation_table.csv"
        
        if result_file.exists():
            df = pd.read_csv(result_file)
            df['fold'] = fold
            all_results.append(df)
        else:
            print(f"Warning: Fold {fold} results not found")
    
    if not all_results:
        raise ValueError(f"No ablation results found in {base_dir}/{model_type}/{task}")
    
    return pd.concat(all_results, ignore_index=True)


def create_ablation_summary_table(combined_df: pd.DataFrame) -> pd.DataFrame:
    
    
    numeric_cols = ['test_auc', 'test_accuracy', 'best_val_auc', 
                   'mean_active_experts', 'std_active_experts',
                   'expert_usage_variance', 'mean_entropy', 'mean_cosine_similarity']
    
    available_cols = [col for col in numeric_cols if col in combined_df.columns]
    
    summary_data = []
    
    for config in combined_df['config'].unique():
        config_df = combined_df[combined_df['config'] == config]
        
        row = {'config': config}
        
        for col in available_cols:
            values = config_df[col].dropna()
            if len(values) > 0:
                row[f'{col}_mean'] = float(values.mean())
                row[f'{col}_std'] = float(values.std())
                row[f'{col}_min'] = float(values.min())
                row[f'{col}_max'] = float(values.max())
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by mean test AUC
    if 'test_auc_mean' in summary_df.columns:
        summary_df = summary_df.sort_values('test_auc_mean', ascending=False)
    
    return summary_df


def analyze_regularization_impact(summary_df: pd.DataFrame) -> dict:
    
    
    analysis = {}
    
    # Find baseline
    baseline_configs = summary_df[summary_df['config'].str.contains('baseline')]
    if len(baseline_configs) > 0:
        baseline = baseline_configs.iloc[0]
        baseline_auc = baseline.get('test_auc_mean', 0)
        
        analysis['baseline'] = {
            'config': baseline['config'],
            'test_auc': baseline_auc
        }
        
        # Compare with no regularization
        no_reg_configs = summary_df[summary_df['config'].str.contains('no_regularization')]
        if len(no_reg_configs) > 0:
            no_reg = no_reg_configs.iloc[0]
            no_reg_auc = no_reg.get('test_auc_mean', 0)
            analysis['no_regularization'] = {
                'config': no_reg['config'],
                'test_auc': no_reg_auc,
                'auc_drop': baseline_auc - no_reg_auc,
                'percent_drop': ((baseline_auc - no_reg_auc) / baseline_auc * 100) if baseline_auc > 0 else 0
            }
        
        # Compare individual terms
        for term in ['entropy', 'diversity', 'load_balance']:
            no_term_configs = summary_df[summary_df['config'] == f'no_{term}']
            if len(no_term_configs) > 0:
                no_term = no_term_configs.iloc[0]
                no_term_auc = no_term.get('test_auc_mean', 0)
                analysis[f'no_{term}'] = {
                    'config': no_term['config'],
                    'test_auc': no_term_auc,
                    'auc_drop': baseline_auc - no_term_auc,
                    'percent_drop': ((baseline_auc - no_term_auc) / baseline_auc * 100) if baseline_auc > 0 else 0
                }
    
    return analysis


def find_optimal_weights(summary_df: pd.DataFrame) -> dict:
    
    
    optimal = {}
    
    # Entropy weights
    entropy_configs = summary_df[summary_df['config'].str.startswith('entropy_')]
    if len(entropy_configs) > 0:
        best_entropy = entropy_configs.loc[entropy_configs['test_auc_mean'].idxmax()]
        optimal['entropy'] = {
            'weight': float(best_entropy['config'].split('_')[1]),
            'test_auc': best_entropy.get('test_auc_mean', 0)
        }
    
    # Diversity weights
    diversity_configs = summary_df[summary_df['config'].str.startswith('diversity_')]
    if len(diversity_configs) > 0:
        best_diversity = diversity_configs.loc[diversity_configs['test_auc_mean'].idxmax()]
        optimal['diversity'] = {
            'weight': float(best_diversity['config'].split('_')[1]),
            'test_auc': best_diversity.get('test_auc_mean', 0)
        }
    
    # Load balance weights
    load_balance_configs = summary_df[summary_df['config'].str.startswith('load_balance_')]
    if len(load_balance_configs) > 0:
        best_load_balance = load_balance_configs.loc[load_balance_configs['test_auc_mean'].idxmax()]
        optimal['load_balance'] = {
            'weight': float(best_load_balance['config'].split('_')[2]),
            'test_auc': best_load_balance.get('test_auc_mean', 0)
        }
    
    return optimal


def main():
    parser = argparse.ArgumentParser(description='Summarize ablation results across folds')
    parser.add_argument('--base_dir', type=str, default='/data2/akp4895/ablation_results',
                       help='Base directory containing ablation results')
    parser.add_argument('--model_type', type=str, required=True, choices=['moe', 'gating'],
                       help='Model type')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for summary')
    
    args = parser.parse_args()
    
    print(f"Loading ablation results for {args.model_type}/{args.task}...")
    
    # Load all fold results
    combined_df = load_all_fold_results(args.base_dir, args.model_type, args.task)
    
    # Create summary table
    summary_df = create_ablation_summary_table(combined_df)
    
    # Analyze regularization impact
    impact_analysis = analyze_regularization_impact(summary_df)
    
    # Find optimal weights
    optimal_weights = find_optimal_weights(summary_df)
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(args.base_dir) / args.model_type / args.task
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary table
    summary_file = output_dir / 'ablation_summary_table.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary table to: {summary_file}")
    
    # Save detailed analysis
    analysis = {
        'task': args.task,
        'model_type': args.model_type,
        'summary_table': summary_df.to_dict('records'),
        'regularization_impact': impact_analysis,
        'optimal_weights': optimal_weights,
        'num_folds': len(combined_df['fold'].unique())
    }
    
    analysis_file = output_dir / 'ablation_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=4, default=str)
    print(f"Saved analysis to: {analysis_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ABLATION SUMMARY: {args.model_type.upper()} - {args.task}")
    print(f"{'='*80}")
    
    print(f"\nTop 5 Configurations (by mean test AUC):")
    top5_cols = ['config', 'test_auc_mean', 'test_auc_std', 
                'mean_active_experts_mean', 'expert_usage_variance_mean']
    available_cols = [col for col in top5_cols if col in summary_df.columns]
    print(summary_df[available_cols].head().to_string(index=False))
    
    if impact_analysis:
        print(f"\nRegularization Impact Analysis:")
        if 'baseline' in impact_analysis:
            baseline = impact_analysis['baseline']
            print(f"  Baseline: {baseline['config']} (AUC: {baseline['test_auc']:.4f})")
        
        for key, value in impact_analysis.items():
            if key != 'baseline' and 'auc_drop' in value:
                print(f"  {key}: AUC drop = {value['auc_drop']:.4f} ({value['percent_drop']:.2f}%)")
    
    if optimal_weights:
        print(f"\nOptimal Regularization Weights:")
        for term, values in optimal_weights.items():
            print(f"  {term}: weight = {values['weight']}, AUC = {values['test_auc']:.4f}")


if __name__ == "__main__":
    main()























