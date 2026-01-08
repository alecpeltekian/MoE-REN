#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RADIOMICS_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$RADIOMICS_DIR/src"
LUNGS_EXTRACTION_DIR="$SRC_DIR/lungs_extraction"

if [ $# -ge 1 ]; then
    CONFIG_FILE="$1"
else
    CONFIG_FILE="$RADIOMICS_DIR/config/config.json"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

TASK=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['task'])")
NUM_EXPERTS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['num_experts'])")
GPU=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training']['gpu'])")
BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training']['batch_size'])")
LR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training']['learning_rate'])")
MAX_EPOCHS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training']['max_epochs'])")
PATIENCE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training']['patience'])")

RADIOMICS_FEATURES_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['output_paths']['radiomics_features_dir'])")
LABELS_FILE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['data_paths']['labels_file'])")
NIFTI_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['data_paths']['nifti_dir'])")
LOBE_MASKS_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['data_paths']['lobe_masks_dir'])")
PYRADIOMICS_CONFIG=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['data_paths']['pyradiomics_config'])")

IMPORTANCE_RESULTS_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['output_paths']['importance_results_dir'])")
BASE_LOGDIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['output_paths']['model_results_dir'])")

EXTRACTION_BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['extraction']['batch_size'])")

echo "=========================================="
echo "Radiomics MoE Pipeline"
echo "=========================================="
echo "Task: $TASK"
echo "Number of Experts: $NUM_EXPERTS"
echo "GPU: $GPU"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Max Epochs: $MAX_EPOCHS"
echo "Patience: $PATIENCE"
echo ""
echo "Data Paths:"
echo "  - Radiomics Features Dir: $RADIOMICS_FEATURES_DIR"
echo "  - Labels File: $LABELS_FILE"
echo "  - NIFTI Dir: $NIFTI_DIR"
echo "  - Lobe Masks Dir: $LOBE_MASKS_DIR"
echo "  - PyRadiomics Config: $PYRADIOMICS_CONFIG"
echo ""
echo "Output Paths:"
echo "  - Importance Results: $IMPORTANCE_RESULTS_DIR"
echo "  - Model Results: $BASE_LOGDIR"
echo "=========================================="
echo ""

export CUDA_VISIBLE_DEVICES=$GPU
export LABELS_FILE=$LABELS_FILE
export NIFTI_DIR=$NIFTI_DIR
export LOBE_MASKS_DIR=$LOBE_MASKS_DIR
export BASE_LOGDIR=$BASE_LOGDIR

echo "STEP 1: Extract Radiomics Features (if needed)"
echo "--------------------------------------------------------"

python "$SRC_DIR/extract_radiomics.py" \
    --config "$CONFIG_FILE" \
    --gpu "$GPU"

if [ $? -ne 0 ]; then
    echo "ERROR: Radiomics extraction failed!"
    exit 1
fi

echo ""
echo "STEP 2: Extract Importance Weights (Validation AUCs)"
echo "--------------------------------------------------------"

python "$SRC_DIR/extract_power.py" \
    --config "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo "ERROR: Importance extraction failed!"
    exit 1
fi

IMPORTANCE_FILE="$IMPORTANCE_RESULTS_DIR/fold_specific/${TASK}_fold_auc_results.json"

if [ ! -f "$IMPORTANCE_FILE" ]; then
    echo "ERROR: Importance file not found: $IMPORTANCE_FILE"
    exit 1
fi

echo ""
echo "STEP 3: Train MoE Model"
echo "--------------------------------------------------------"
echo "Using importance file: $IMPORTANCE_FILE"
echo ""

python "$SRC_DIR/modeling_radiomics_moe.py" \
    --config "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo "ERROR: MoE training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="

