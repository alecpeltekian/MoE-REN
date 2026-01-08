#!/bin/bash

set -e

TASK="has_ILD"
MODEL_TYPE_MOE="moe"
MODEL_TYPE_GATING="gating"
GPU=1
NUM_EPOCHS=30
BATCH_SIZE=4
LEARNING_RATE=1e-4
EXPERT_RESULTS_DIR="/data2/akp4895/unified_cnn_results"
BASE_OUTPUT_DIR="/data2/akp4895/ablation_results"

echo "=========================================="
echo "Full Ablation Analysis Pipeline"
echo "=========================================="
echo "Task: $TASK"
echo "GPU: $GPU"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "=========================================="
echo ""

run_ablation() {
    local model_type=$1
    local fold=$2
    
    echo ""
    echo "=========================================="
    echo "Running $model_type ablation - Fold $fold"
    echo "=========================================="
    
    if [ "$model_type" == "gating" ]; then
        python ablation/ablation_analysis.py \
            --task $TASK \
            --model_type $model_type \
            --fold_idx $fold \
            --num_epochs $NUM_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --gpu $GPU \
            --expert_results_dir $EXPERT_RESULTS_DIR \
            --output_dir $BASE_OUTPUT_DIR
    else
        python ablation/ablation_analysis.py \
            --task $TASK \
            --model_type $model_type \
            --fold_idx $fold \
            --num_epochs $NUM_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --gpu $GPU \
            --output_dir $BASE_OUTPUT_DIR
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed $model_type fold $fold"
    else
        echo "❌ Failed $model_type fold $fold"
        return 1
    fi
}

echo "=========================================="
echo "MoE Model Ablation (All Folds)"
echo "=========================================="
for fold in {0..4}; do
    run_ablation $MODEL_TYPE_MOE $fold
done

echo ""
echo "=========================================="
echo "Gating MoE Model Ablation (All Folds)"
echo "=========================================="
for fold in {0..4}; do
    run_ablation $MODEL_TYPE_GATING $fold
done

echo ""
echo "=========================================="
echo "Generating Visualizations"
echo "=========================================="
python ablation/visualize_ablation.py \
    --base_dir $BASE_OUTPUT_DIR \
    --model_type $MODEL_TYPE_MOE \
    --task $TASK \
    --fold_idx 0 \
    --output_dir $BASE_OUTPUT_DIR/$MODEL_TYPE_MOE/$TASK

python ablation/visualize_ablation.py \
    --base_dir $BASE_OUTPUT_DIR \
    --model_type $MODEL_TYPE_GATING \
    --task $TASK \
    --fold_idx 0 \
    --output_dir $BASE_OUTPUT_DIR/$MODEL_TYPE_GATING/$TASK

echo ""
echo "=========================================="
echo "Generating Summaries"
echo "=========================================="
python ablation/summarize_ablation.py \
    --base_dir $BASE_OUTPUT_DIR \
    --model_type $MODEL_TYPE_MOE \
    --task $TASK \
    --output_dir $BASE_OUTPUT_DIR/$MODEL_TYPE_MOE/$TASK

python ablation/summarize_ablation.py \
    --base_dir $BASE_OUTPUT_DIR \
    --model_type $MODEL_TYPE_GATING \
    --task $TASK \
    --output_dir $BASE_OUTPUT_DIR/$MODEL_TYPE_GATING/$TASK

echo ""
echo "=========================================="
echo "✅ Full Ablation Analysis Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - MoE: $BASE_OUTPUT_DIR/$MODEL_TYPE_MOE/$TASK"
echo "  - Gating: $BASE_OUTPUT_DIR/$MODEL_TYPE_GATING/$TASK"
