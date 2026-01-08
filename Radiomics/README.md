# Imaging Radiomics Pipeline

Implementation of the imaging-only radiomics pipeline using Mixture-of-Experts (MoE) model.

## Overview

This pipeline extracts radiomics features and trains a generalizable MoE model for medical imaging classification tasks. Supports 2, 5, or 7 experts:
- **2 experts**: Left Lung, Right Lung
- **5 experts**: 5 Lobes (LUL, LLL, RUL, RML, RLL)
- **7 experts**: 5 Lobes + 2 Lungs

## Pipeline Steps

1. **Extract Radiomics Features**: Extracts radiomics features from regions (lobes and/or lungs) using PyRadiomics
2. **Extract Importance Weights**: Calculates validation AUCs for each region to determine importance weights
3. **Train MoE Model**: Trains a Mixture-of-Experts model with the specified number of experts

## Directory Structure

```
Radiomics_clean/
├── config/
│   └── config.json                              # Configuration file
├── src/
│   ├── modeling_radiomics_moe.py               # General MoE training (2/5/7 experts)
│   ├── extract_radiomics.py                    # Unified radiomics extraction (lobes/lungs)
│   └── extract_power.py                        # Unified importance analysis (lobes/lungs)
├── scripts/
│   └── run_pipeline.sh                         # Main pipeline script
└── README.md
```

## Usage

### Basic Usage

```bash
cd /home/akp4895/MoE_MultiModal/Radiomics_clean
bash scripts/run_pipeline.sh
```

The script will automatically use `config/config.json` if no config file is specified.

### With Custom Config

```bash
bash scripts/run_pipeline.sh /path/to/custom_config.json
```

### Configuration File

Edit `config/config.json` to set:

- `task`: Task name (e.g., "has_ILD")
- `num_experts`: Number of experts (2, 5, or 7)
- `training.gpu`: GPU device ID
- `training.batch_size`: Training batch size
- `training.learning_rate`: Learning rate
- `training.max_epochs`: Maximum training epochs
- `training.patience`: Early stopping patience
- `data_paths`: All data paths (nifti_dir, lobe_masks_dir, labels_file, pyradiomics_config)
- `output_paths`: All output paths (radiomics_features_dir, importance_results_dir, model_results_dir)

## Requirements

- Python 3.x
- PyTorch
- MONAI
- PyRadiomics
- scikit-learn
- pandas
- numpy
- xgboost

## Output

- Radiomics features: `{radiomics_features_dir}/` (organized by region)
- Importance weights: `{importance_results_dir}/fold_specific/{task}_fold_auc_results.json`
- Trained models: `{model_results_dir}/{task}_moe_{num_experts}expert_*/`

