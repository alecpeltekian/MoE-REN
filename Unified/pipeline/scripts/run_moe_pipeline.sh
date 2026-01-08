#!/bin/bash
# MoE Pipeline Bash Script
# Runs the complete MoE pipeline with configuration from config_moe.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PIPELINE_DIR}/config/config_moe.json"

echo "=========================================="
echo "MoE Pipeline Execution"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "=========================================="
echo ""

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

cd "$PIPELINE_DIR"

python run_moe_pipeline.py --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ MoE Pipeline completed successfully!"
else
    echo ""
    echo "❌ MoE Pipeline failed!"
    exit 1
fi






