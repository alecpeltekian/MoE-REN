#!/bin/bash

set -e

TASK="has_ILD"
MODEL_TYPE="moe"
FOLD_IDX=0
GPU=1

python ablation/ablation_analysis.py \
    --task $TASK \
    --model_type moe \
    --fold_idx $FOLD_IDX \
    --num_epochs 30 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --gpu $GPU

python ablation/ablation_analysis.py \
    --task $TASK \
    --model_type gating \
    --fold_idx $FOLD_IDX \
    --num_epochs 30 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --gpu $GPU \
    --expert_results_dir /data2/akp4895/unified_cnn_results

python ablation/visualize_ablation.py \
    --base_dir /data2/akp4895/ablation_results \
    --model_type moe \
    --task $TASK \
    --fold_idx $FOLD_IDX
