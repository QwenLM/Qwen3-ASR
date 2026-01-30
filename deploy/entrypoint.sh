#!/bin/bash
set -e

# Entrypoint script for Qwen3-ASR server

echo "=========================================="
echo "Qwen3-ASR Server Starting..."
echo "=========================================="
echo "Model: ${MODEL_NAME:-Qwen/Qwen3-ASR-1.7B}"
echo "Device: ${DEVICE:-cuda:0}"
echo "Port: ${PORT:-8000}"
echo "=========================================="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "=========================================="
fi

# Download model if needed (pre-download to avoid timeout during startup)
if [ "${PRELOAD_MODEL:-true}" = "true" ]; then
    echo "Pre-downloading model..."
    python -c "
from huggingface_hub import snapshot_download
import os
model = os.environ.get('MODEL_NAME', 'Qwen/Qwen3-ASR-1.7B')
print(f'Downloading {model}...')
snapshot_download(model)
print('Model downloaded successfully')
"
    echo "=========================================="
fi

# Start the server
exec python server.py
