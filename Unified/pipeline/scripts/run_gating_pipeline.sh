#!/bin/bash
# Gating MoE Pipeline Bash Script
# Runs the complete Gating MoE pipeline with configuration from config_gating.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PIPELINE_DIR}/config/config_gating.json"

echo "=========================================="
echo "Gating MoE Pipeline Execution"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "=========================================="
echo ""

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

cd "$PIPELINE_DIR"

python run_gating_pipeline.py --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Gating MoE Pipeline completed successfully!"
else
    echo ""
    echo "❌ Gating MoE Pipeline failed!"
    exit 1
fi






