# Unified MoE Pipeline

## Usage

### Gating Pipeline

```bash
cd pipeline
bash scripts/run_gating_pipeline.sh
```

### MoE Pipeline

```bash
cd pipeline
bash scripts/run_moe_pipeline.sh
```

## Configuration

Edit `pipeline/config/config_gating.json` or `pipeline/config/config_moe.json`:
- `task`: Task name (e.g., "has_ILD")
- `expert_type`: Model type ("cnn", "vit", or "mamba")
- `num_experts`: Number of experts (2, 5, or 7)
- `data_paths`: All data paths
- `training`: Training parameters
- `gpu`: GPU device ID

## Directory Structure

- `src/` - Core source code
- `pipeline/` - Pipeline scripts and configs
- `ablation/` - Ablation analysis scripts
- `complexity/` - Complexity analysis scripts
