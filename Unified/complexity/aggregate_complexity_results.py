#!/usr/bin/env python3


import os
import json
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict

def load_results(results_dir: str) -> List[Dict]:
    
    results = []
    results_path = Path(results_dir)
    
    for json_file in results_path.glob("complexity_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['result_file'] = str(json_file)
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results

def create_summary_table(results: List[Dict]) -> pd.DataFrame:
    
    rows = []
    
    for result in results:
        model_type = result.get('model_type', 'unknown')
        expert_type = result.get('expert_type', 'N/A')
        
        if model_type == 'moe' and expert_type != 'N/A':
            display_name = f"MoE ({expert_type.upper()})"
        elif model_type == 'gating':
            display_name = "Gating MoE"
        elif model_type.startswith('expert_'):
            expert_name = model_type.replace('expert_', '').upper()
            display_name = f"Expert ({expert_name})"
        elif model_type == 'swinunetr':
            display_name = "Baseline SwinUNETR"
        else:
            display_name = model_type
        
        row = {
            'Model': display_name,
            'Model Type': model_type,
            'Expert Type': expert_type if expert_type != 'N/A' else '',
            'Wall-clock Time (hours)': result.get('wall_clock_hours', 0),
            'GPU-hours': result.get('gpu_hours', 0),
            'Peak Memory (GB)': result.get('peak_memory_mb', 0) / 1024.0,
            'Peak Memory (MB)': result.get('peak_memory_mb', 0),
            'Avg Epoch Time (s)': result.get('avg_epoch_time_seconds', 0),
            'Avg Batch Time (s)': result.get('avg_batch_time_seconds', 0),
            'Total Params': result.get('total_params', 0),
            'Trainable Params': result.get('trainable_params', 0),
            'Model Size (MB)': result.get('model_size_mb', 0),
            'Task': result.get('task', ''),
            'Fold': result.get('fold', ''),
            'Batch Size': result.get('batch_size', ''),
            'Epochs': result.get('num_epochs', '')
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.sort_values('Peak Memory (GB)')

def create_paper_table(results: List[Dict]) -> pd.DataFrame:
    
    rows = []
    
    for result in results:
        model_type = result.get('model_type', 'unknown')
        expert_type = result.get('expert_type', 'N/A')
        
        if model_type == 'moe' and expert_type != 'N/A':
            display_name = f"MoE ({expert_type.upper()})"
        elif model_type == 'gating':
            display_name = "Gating MoE"
        elif model_type.startswith('expert_'):
            expert_name = model_type.replace('expert_', '').upper()
            display_name = f"{expert_name} Expert"
        elif model_type == 'swinunetr':
            display_name = "SwinUNETR Baseline"
        else:
            display_name = model_type
        
        row = {
            'Model': display_name,
            'Training Time (GPU-hours)': f"{result.get('gpu_hours', 0):.3f}",
            'Peak Memory (GB)': f"{result.get('peak_memory_mb', 0) / 1024.0:.2f}",
            'Parameters': f"{result.get('total_params', 0):,}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.sort_values('Peak Memory (GB)')

def main():
    parser = argparse.ArgumentParser(description='Aggregate training complexity results')
    parser.add_argument('--results_dir', type=str, default='./complexity_results',
                       help='Directory containing complexity result JSON files')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output CSV file (default: complexity_summary.csv in results_dir)')
    parser.add_argument('--paper_table', type=str, default=None,
                       help='Output CSV file for paper table (default: complexity_paper_table.csv)')
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return 1
    
    print(f"Loaded {len(results)} result files")
    
    summary_df = create_summary_table(results)
    paper_df = create_paper_table(results)
    
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(args.results_dir, 'complexity_summary.csv')
    
    summary_df.to_csv(output_path, index=False)
    print(f"\n✅ Summary table saved to: {output_path}")
    
    if args.paper_table:
        paper_path = args.paper_table
    else:
        paper_path = os.path.join(args.results_dir, 'complexity_paper_table.csv')
    
    paper_df.to_csv(paper_path, index=False)
    print(f"✅ Paper table saved to: {paper_path}")
    
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"\nTotal Models Tested: {len(results)}")
    print(f"\nMemory Usage:")
    print(f"  Min: {summary_df['Peak Memory (GB)'].min():.2f} GB")
    print(f"  Max: {summary_df['Peak Memory (GB)'].max():.2f} GB")
    print(f"  Mean: {summary_df['Peak Memory (GB)'].mean():.2f} GB")
    print(f"\nTraining Time:")
    print(f"  Min: {summary_df['GPU-hours'].min():.3f} GPU-hours")
    print(f"  Max: {summary_df['GPU-hours'].max():.3f} GPU-hours")
    print(f"  Mean: {summary_df['GPU-hours'].mean():.3f} GPU-hours")
    
    print("\n" + "="*80)
    print("Paper-Ready Table")
    print("="*80)
    print(paper_df.to_string(index=False))
    
    return 0

if __name__ == "__main__":
    exit(main())






