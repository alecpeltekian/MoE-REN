# MoE MultiModal Pipeline

Complete pipeline for medical imaging analysis using Mixture-of-Experts (MoE) models with radiomics features and deep learning.

## Overview

This repository contains the complete workflow from image preprocessing to model training and evaluation:

1. **Image Preprocessing** - Orient, resample, segment, and prepare images
2. **Radiomics Pipeline** - Extract radiomics features and train radiomics-based MoE models
3. **Unified MoE Pipeline** - Train CNN/ViT/Mamba experts and combine them using MoE or Gating MoE architectures

## Directory Structure

```
MoE_MultiModal/
├── image_preprocessing/     # Image preprocessing scripts
├── Radiomics_clean/         # Clean radiomics pipeline (recommended)
├── Unified_clean/           # Clean unified MoE pipeline (recommended)
├── baseline/                # Baseline model training scripts
└── README.md                # This file
```

## Complete Pipeline Workflow

### Step 1: Image Preprocessing

Preprocess raw medical images for analysis.

```bash
cd image_preprocessing

# 1. Orient images to standard orientation
python 01_OrientImages.py --input_dir /path/to/raw/images --output_dir /path/to/oriented

# 2. Resample to 1mm isotropic resolution
python 02_1mmImages.py --input_dir /path/to/oriented --output_dir /path/to/1mm

# 3. Segment lungs and lobes
python 03_SegmentImages.py --input_dir /path/to/1mm --output_dir /path/to/segmented

# 4. Multiply images (if needed for augmentation)
python 04_MultiplyImages.py --input_dir /path/to/1mm --output_dir /path/to/multiplied
```

**Output**: Preprocessed images ready for feature extraction and model training.

### Step 2: Radiomics Pipeline

Extract radiomics features and train radiomics-based MoE models.

```bash
cd Radiomics_clean

# Edit config/config.json to set:
# - num_experts: 2 (lungs), 5 (lobes), or 7 (lobes + lungs)
# - data_paths: paths to images, masks, labels
# - training: GPU, batch size, learning rate, etc.

# Run the complete pipeline
bash scripts/run_pipeline.sh
```

**Pipeline Steps**:
1. Extracts radiomics features from regions (lobes/lungs)
2. Calculates importance weights (validation AUCs) for each region
3. Trains MoE model with SwinUNETR backbone

**Configuration**: Edit `config/config.json`

**Output**: 
- Radiomics features: `{radiomics_features_dir}/`
- Importance weights: `{importance_results_dir}/fold_specific/`
- Trained models: `{model_results_dir}/`

See `Radiomics_clean/README.md` for detailed documentation.

### Step 3: Unified MoE Pipeline

Train individual experts (CNN/ViT/Mamba) and combine them using MoE or Gating MoE.

#### Option A: Gating MoE Pipeline

Uses learned gating functions to combine experts.

```bash
cd Unified_clean/pipeline

# Edit config/config_gating.json to set:
# - expert_type: "cnn", "vit", or "mamba"
# - num_experts: 2, 5, or 7
# - data_paths: paths to images, masks, labels
# - training: GPU, batch size, learning rate, etc.

# Run the complete pipeline
bash scripts/run_gating_pipeline.sh
```

**Pipeline Steps**:
1. Train individual experts (CNN/ViT/Mamba)
2. Extract gating functions from trained experts
3. Train Gating MoE model with SwinUNETR backbone
4. Evaluate ensemble performance

#### Option B: Standard MoE Pipeline

Uses learned attention weights to combine experts.

```bash
cd Unified_clean/pipeline

# Edit config/config_moe.json to set:
# - expert_type: "cnn", "vit", or "mamba"
# - num_experts: 2, 5, or 7
# - data_paths: paths to images, masks, labels
# - training: GPU, batch size, learning rate, etc.

# Run the complete pipeline
bash scripts/run_moe_pipeline.sh
```

**Pipeline Steps**:
1. Train individual experts (CNN/ViT/Mamba)
2. Train MoE model with learned weights
3. Evaluate ensemble performance

**Configuration**: Edit `pipeline/config/config_gating.json` or `pipeline/config/config_moe.json`

**Output**:
- Expert models: `{expert_results_dir}/`
- Gating functions: `{gating_results_dir}/` (Gating pipeline only)
- MoE models: `{moe_results_dir}/`
- Ensemble results: `{output_dir}/ensemble_results/`

See `Unified_clean/README.md` for detailed documentation.

## Quick Start Examples

### Example 1: Radiomics Pipeline (2 Lung Experts)

```bash
cd Radiomics_clean
# Edit config/config.json: set num_experts=2
bash scripts/run_pipeline.sh
```

### Example 2: Unified Gating MoE (CNN Experts, 5 Lobes)

```bash
cd Unified_clean/pipeline
# Edit config/config_gating.json: set expert_type="cnn", num_experts=5
bash scripts/run_gating_pipeline.sh
```

### Example 3: Unified MoE (ViT Experts, 7 Regions)

```bash
cd Unified_clean/pipeline
# Edit config/config_moe.json: set expert_type="vit", num_experts=7
bash scripts/run_moe_pipeline.sh
```

## Running Individual Steps

### Radiomics Pipeline

The pipeline automatically runs all steps, but you can run individual scripts:

```bash
cd Radiomics_clean/src

# Extract radiomics features
python extract_radiomics.py --config ../config/config.json --gpu 0

# Extract importance weights
python extract_power.py --config ../config/config.json

# Train MoE model
python modeling_radiomics_moe.py --config ../config/config.json --importance_file /path/to/importance.json
```

### Unified Pipeline

Run specific steps:

```bash
cd Unified_clean/pipeline

# Gating Pipeline - Run step 1 only (train experts)
python run_gating_pipeline.py --config config/config_gating.json --step 1

# Gating Pipeline - Run step 2 only (extract gating)
python run_gating_pipeline.py --config config/config_gating.json --step 2

# MoE Pipeline - Run step 1 only (train experts)
python run_moe_pipeline.py --config config/config_moe.json --step 1

# MoE Pipeline - Run step 2 only (train MoE)
python run_moe_pipeline.py --config config/config_moe.json --step 2
```

## Configuration Files

### Radiomics Pipeline Config

`Radiomics_clean/config/config.json`:
- `task`: Task name (e.g., "has_ILD")
- `num_experts`: 2, 5, or 7
- `data_paths`: nifti_dir, lobe_masks_dir, labels_file, pyradiomics_config
- `output_paths`: radiomics_features_dir, importance_results_dir, model_results_dir
- `training`: gpu, batch_size, learning_rate, max_epochs, patience

### Unified Pipeline Configs

`Unified_clean/pipeline/config/config_gating.json` or `config_moe.json`:
- `task`: Task name (e.g., "has_ILD")
- `expert_type`: "cnn", "vit", or "mamba"
- `num_experts`: 2, 5, or 7
- `data_paths`: nifti_dir, lobe_masks_dir, labels_file
- `training`: num_epochs, batch_size, learning_rate, early_stopping_patience
- `gpu`: GPU device ID

## Additional Tools

### Ablation Analysis

```bash
cd Unified_clean/ablation
bash run_full_ablation.sh
```

### Complexity Analysis

```bash
cd Unified_clean/complexity
bash run_complexity_analysis.sh
```

### Baseline Models

```bash
cd baseline
python 01_train_model.py        # Baseline SwinUNETR
python 02_train_model_cnn.py   # CNN baseline
python 03_train_model_vit.py    # ViT baseline
python 04_train_model_mamba.py  # Mamba baseline
```

## Requirements

### Core Dependencies

- Python 3.x
- PyTorch
- MONAI
- scikit-learn
- pandas
- numpy

### Radiomics Pipeline

- PyRadiomics
- xgboost

### Unified Pipeline

- (Optional) mamba-ssm for Mamba models

## Model Architectures

### Backbone Models

All MoE models use **SwinUNETR** as the backbone architecture.

### Expert Types

- **CNN**: Convolutional neural network experts
- **ViT**: Vision Transformer experts
- **Mamba**: Mamba (state-space model) experts

### Expert Configurations

- **2 experts**: Left Lung, Right Lung
- **5 experts**: 5 Lobes (LUL, LLL, RUL, RML, RLL)
- **7 experts**: 5 Lobes + 2 Lungs

## Output Locations

### Radiomics Pipeline

- Features: `{radiomics_features_dir}/`
- Importance: `{importance_results_dir}/fold_specific/`
- Models: `{model_results_dir}/{task}_moe_{num_experts}expert_*/`

### Unified Pipeline

- Experts: `{expert_results_dir}/unified_{expert_type}_results/`
- Gating: `{gating_results_dir}/unified_{expert_type}_results/` (Gating only)
- MoE: `{moe_results_dir}/{expert_type}_{num_experts}experts/`
- Ensemble: `{output_dir}/ensemble_results/`

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install required packages
2. **GPU out of memory**: Reduce batch_size in config
3. **File not found**: Check data_paths in config files
4. **Import errors**: Ensure you're in the correct directory and Python path is set

### Getting Help

- Check individual README files in each pipeline directory
- Review config files for path settings
- Check log files in output directories

## Citation

If you use this code, please cite the relevant papers for:
- SwinUNETR architecture
- PyRadiomics feature extraction
- Mixture-of-Experts methodology

