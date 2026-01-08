#!/bin/bash

set -e

TASK="has_ILD"
FOLD=0
NUM_EPOCHS=5
BATCH_SIZE=4
GPU=2
NIFTI_DIR="/data2/akp4895/MultipliedImagesClean"
LOBE_MASKS_DIR="/data2/akp4895/1mm_OrientedImagesSegmented"
LABELS_FILE="/home/akp4895/MoE_MultiModal/Radiomics/data/labels.csv"
OUTPUT_DIR="./complexity_results"

echo "=========================================="
echo "Training Complexity Analysis"
echo "=========================================="
echo "Task: $TASK"
echo "Fold: $FOLD"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "GPU: $GPU"
echo "=========================================="
echo ""

run_complexity() {
    local model_type=$1
    local expert_type=${2:-""}
    
    echo ""
    echo "=========================================="
    echo "Running: $model_type"
    if [ -n "$expert_type" ]; then
        echo "Expert Type: $expert_type"
    fi
    echo "=========================================="
    
    if [ -n "$expert_type" ]; then
        python test_training_complexity.py \
            --model_type $model_type \
            --task $TASK \
            --fold $FOLD \
            --num_epochs $NUM_EPOCHS \
            --batch_size $BATCH_SIZE \
            --gpu $GPU \
            --expert_type $expert_type \
            --nifti_dir $NIFTI_DIR \
            --lobe_masks_dir $LOBE_MASKS_DIR \
            --labels_file $LABELS_FILE \
            --output_dir $OUTPUT_DIR
    else
        python test_training_complexity.py \
            --model_type $model_type \
            --task $TASK \
            --fold $FOLD \
            --num_epochs $NUM_EPOCHS \
            --batch_size $BATCH_SIZE \
            --gpu $GPU \
            --nifti_dir $NIFTI_DIR \
            --lobe_masks_dir $LOBE_MASKS_DIR \
            --labels_file $LABELS_FILE \
            --output_dir $OUTPUT_DIR
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed $model_type"
    else
        echo "❌ Failed $model_type"
        return 1
    fi
}

echo "=========================================="
echo "Baseline SwinUNETR"
echo "=========================================="
run_complexity "swinunetr"

echo ""
echo "=========================================="
echo "MoE Models"
echo "=========================================="
run_complexity "moe" "cnn"
run_complexity "moe" "vit"
run_complexity "moe" "mamba"

echo ""
echo "=========================================="
echo "Gating MoE Model"
echo "=========================================="
run_complexity "gating"

echo ""
echo "=========================================="
echo "Individual Expert Models"
echo "=========================================="
run_complexity "expert_cnn"
run_complexity "expert_vit"
run_complexity "expert_mamba"

echo ""
echo "=========================================="
echo "Aggregating Results"
echo "=========================================="
python aggregate_complexity_results.py --results_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "✅ All Complexity Analyses Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
