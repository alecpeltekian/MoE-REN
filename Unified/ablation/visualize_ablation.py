#!/usr/bin/env python3


import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List


def load_ablation_results(base_dir: str, model_type: str, task: str, fold: int = None) -> pd.DataFrame:
    
    if fold is not None:
        result_file = Path(base_dir) / model_type / task / f"fold_{fold}" / "ablation_table.csv"
    else:
        result_file = Path(base_dir) / model_type / task / "ablation_table.csv"
    
    if not result_file.exists():
        raise FileNotFoundError(f"Ablation table not found: {result_file}")
    
    return pd.read_csv(result_file)


def aggregate_across_folds(base_dir: str, model_type: str, task: str) -> pd.DataFrame:
    
    all_results = []
    
    for fold in range(5):
        try:
            df = load_ablation_results(base_dir, model_type, task, fold)
            df['fold'] = fold
            all_results.append(df)
        except FileNotFoundError:
            print(f"Warning: Fold {fold} results not found, skipping...")
            continue
    
    if not all_results:
        raise ValueError("No fold results found")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Aggregate metrics across folds
    agg_metrics = combined_df.groupby('config').agg({
        'test_auc': ['mean', 'std', 'min', 'max'],
        'test_accuracy': ['mean', 'std'],
        'mean_active_experts': ['mean', 'std'],
        'expert_usage_variance': ['mean', 'std'],
        'mean_entropy': ['mean', 'std'],
        'mean_cosine_similarity': ['mean', 'std']
    }).round(4)
    
    agg_metrics.columns = ['_'.join(col).strip() for col in agg_metrics.columns.values]
    agg_metrics = agg_metrics.reset_index()
    
    return agg_metrics, combined_df


def create_ablation_table_plot(df: pd.DataFrame, output_path: str, title: str = "Ablation Analysis"):
    
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Sort by test AUC
    df_sorted = df.sort_values('test_auc', ascending=False)
    
    # 1. AUC Comparison
    ax = axes[0, 0]
    colors = ['green' if 'baseline' in c else 'blue' if 'no_' in c else 'orange' for c in df_sorted['config']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['test_auc'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['config'], fontsize=9)
    ax.set_xlabel('Test AUC', fontsize=12)
    ax.set_title('Test AUC by Configuration', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, auc) in enumerate(zip(bars, df_sorted['test_auc'])):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
               f'{auc:.4f}', va='center', fontsize=8)
    
    # 2. Expert Activation Frequency
    ax = axes[0, 1]
    if 'mean_active_experts' in df_sorted.columns:
        ax.scatter(df_sorted['test_auc'], df_sorted['mean_active_experts'], 
                  s=100, alpha=0.6, c=colors)
        ax.set_xlabel('Test AUC', fontsize=12)
        ax.set_ylabel('Mean Active Experts', fontsize=12)
        ax.set_title('AUC vs Expert Activation', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        for idx, row in df_sorted.iterrows():
            ax.annotate(row['config'], 
                       (row['test_auc'], row['mean_active_experts']),
                       fontsize=7, alpha=0.7)
    
    # 3. Regularization Impact Comparison
    ax = axes[1, 0]
    if 'baseline_all' in df_sorted['config'].values:
        baseline_auc = df_sorted[df_sorted['config'] == 'baseline_all']['test_auc'].values[0]
        
        comparison_configs = ['baseline_all', 'no_entropy', 'no_diversity', 'no_regularization']
        comparison_data = []
        
        for config in comparison_configs:
            if config in df_sorted['config'].values:
                config_auc = df_sorted[df_sorted['config'] == config]['test_auc'].values[0]
                comparison_data.append({
                    'config': config.replace('_', ' ').title(),
                    'auc': config_auc,
                    'delta': config_auc - baseline_auc
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            bars = ax.bar(comp_df['config'], comp_df['delta'], 
                         color=['green' if d >= 0 else 'red' for d in comp_df['delta']],
                         alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax.set_ylabel('AUC Change from Baseline', fontsize=12)
            ax.set_title('Regularization Term Impact', fontsize=14, fontweight='bold')
            ax.set_xticklabels(comp_df['config'], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            for bar, delta in zip(bars, comp_df['delta']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.001 if delta >= 0 else -0.001),
                       f'{delta:+.4f}', ha='center', va='bottom' if delta >= 0 else 'top', fontsize=9)
    
    # 4. Metrics Heatmap
    ax = axes[1, 1]
    metrics_cols = ['test_auc', 'mean_active_experts', 'expert_usage_variance', 
                   'mean_entropy', 'mean_cosine_similarity']
    available_cols = [col for col in metrics_cols if col in df_sorted.columns]
    
    if available_cols:
        metrics_data = df_sorted[['config'] + available_cols].set_index('config')
        
        # Normalize for heatmap
        metrics_normalized = metrics_data.copy()
        for col in available_cols:
            if col == 'mean_cosine_similarity':
                metrics_normalized[col] = 1 - metrics_normalized[col]
            col_min = metrics_normalized[col].min()
            col_max = metrics_normalized[col].max()
            if col_max > col_min:
                metrics_normalized[col] = (metrics_normalized[col] - col_min) / (col_max - col_min)
        
        sns.heatmap(metrics_normalized.T, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=ax, cbar_kws={'label': 'Normalized Score'})
        ax.set_title('Normalized Metrics Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved ablation visualization to: {output_path}")


def create_weight_sensitivity_plot(df: pd.DataFrame, output_path: str):
    
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Regularization Weight Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Extract weight configurations
    entropy_configs = df[df['config'].str.startswith('entropy_')]
    diversity_configs = df[df['config'].str.startswith('diversity_')]
    load_balance_configs = df[df['config'].str.startswith('load_balance_')] if 'load_balance_weight' in df.columns else pd.DataFrame()
    
    # Entropy weight sensitivity
    if len(entropy_configs) > 0:
        ax = axes[0]
        entropy_weights = [float(c.split('_')[1]) for c in entropy_configs['config']]
        ax.plot(entropy_weights, entropy_configs['test_auc'], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Entropy Weight', fontsize=12)
        ax.set_ylabel('Test AUC', fontsize=12)
        ax.set_title('Entropy Weight Sensitivity', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    
    # Diversity weight sensitivity
    if len(diversity_configs) > 0:
        ax = axes[1]
        diversity_weights = [float(c.split('_')[1]) for c in diversity_configs['config']]
        ax.plot(diversity_weights, diversity_configs['test_auc'], 'o-', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Diversity Weight', fontsize=12)
        ax.set_ylabel('Test AUC', fontsize=12)
        ax.set_title('Diversity Weight Sensitivity', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    
    # Load balance weight sensitivity
    if len(load_balance_configs) > 0:
        ax = axes[2]
        load_balance_weights = [float(c.split('_')[2]) for c in load_balance_configs['config']]
        ax.plot(load_balance_weights, load_balance_configs['test_auc'], 'o-', linewidth=2, markersize=8, color='orange')
        ax.set_xlabel('Load Balance Weight', fontsize=12)
        ax.set_ylabel('Test AUC', fontsize=12)
        ax.set_title('Load Balance Weight Sensitivity', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    else:
        axes[2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved weight sensitivity plot to: {output_path}")


def create_expert_activation_analysis(df: pd.DataFrame, output_path: str):
    
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Expert Activation Pattern Analysis', fontsize=16, fontweight='bold')
    
    # 1. Active experts distribution
    ax = axes[0, 0]
    if 'mean_active_experts' in df.columns:
        ax.hist(df['mean_active_experts'], bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(df['mean_active_experts'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["mean_active_experts"].mean():.2f}')
        ax.set_xlabel('Mean Active Experts', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Active Experts', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # 2. AUC vs Active Experts
    ax = axes[0, 1]
    if 'mean_active_experts' in df.columns and 'test_auc' in df.columns:
        scatter = ax.scatter(df['mean_active_experts'], df['test_auc'], 
                           s=100, alpha=0.6, c=df['test_auc'], cmap='viridis')
        ax.set_xlabel('Mean Active Experts', fontsize=12)
        ax.set_ylabel('Test AUC', fontsize=12)
        ax.set_title('AUC vs Expert Activation', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Test AUC')
    
    # 3. Expert usage variance
    ax = axes[1, 0]
    if 'expert_usage_variance' in df.columns:
        ax.scatter(df['expert_usage_variance'], df['test_auc'], s=100, alpha=0.6)
        ax.set_xlabel('Expert Usage Variance', fontsize=12)
        ax.set_ylabel('Test AUC', fontsize=12)
        ax.set_title('Load Balancing Impact', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # 4. Diversity vs Performance
    ax = axes[1, 1]
    if 'mean_cosine_similarity' in df.columns and 'test_auc' in df.columns:
        ax.scatter(df['mean_cosine_similarity'], df['test_auc'], s=100, alpha=0.6, color='green')
        ax.set_xlabel('Mean Cosine Similarity (Lower = More Diverse)', fontsize=12)
        ax.set_ylabel('Test AUC', fontsize=12)
        ax.set_title('Expert Diversity Impact', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved expert activation analysis to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize ablation analysis results')
    parser.add_argument('--base_dir', type=str, default='/data2/akp4895/ablation_results',
                       help='Base directory containing ablation results')
    parser.add_argument('--model_type', type=str, required=True, choices=['moe', 'gating'],
                       help='Model type')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name')
    parser.add_argument('--fold_idx', type=int, default=None, choices=range(5),
                       help='Specific fold to visualize (None for aggregated)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        if args.fold_idx is not None:
            args.output_dir = Path(args.base_dir) / args.model_type / args.task / f"fold_{args.fold_idx}"
        else:
            args.output_dir = Path(args.base_dir) / args.model_type / args.task
    else:
        args.output_dir = Path(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading ablation results...")
    if args.fold_idx is not None:
        df = load_ablation_results(args.base_dir, args.model_type, args.task, args.fold_idx)
        title = f"Ablation Analysis: {args.model_type.upper()} - {args.task} - Fold {args.fold_idx}"
    else:
        df, _ = aggregate_across_folds(args.base_dir, args.model_type, args.task)
        title = f"Ablation Analysis: {args.model_type.upper()} - {args.task} - Aggregated"
    
    print(f"Creating visualizations...")
    
    # Main ablation table plot
    create_ablation_table_plot(df, args.output_dir / 'ablation_table.png', title)
    
    # Weight sensitivity plot
    create_weight_sensitivity_plot(df, args.output_dir / 'weight_sensitivity.png')
    
    # Expert activation analysis
    create_expert_activation_analysis(df, args.output_dir / 'expert_activation_analysis.png')
    
    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()























