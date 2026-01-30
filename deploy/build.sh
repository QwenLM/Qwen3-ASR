#!/bin/bash
# Build and run Qwen3-ASR Docker image

set -e

IMAGE_NAME="${IMAGE_NAME:-qwen3-asr}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-asr-server}"

# Model configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-ASR-1.7B}"  # or Qwen/Qwen3-ASR-0.6B
PORT="${PORT:-8000}"

echo "=========================================="
echo "Building Qwen3-ASR Docker Image"
echo "=========================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Model: ${MODEL_NAME}"
echo "=========================================="

# Build image
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "To run the container:"
echo ""
echo "  docker run --gpus all -d \\"
echo "    --name ${CONTAINER_NAME} \\"
echo "    -p ${PORT}:8000 \\"
echo "    -e MODEL_NAME=${MODEL_NAME} \\"
echo "    -v \${HOME}/.cache/huggingface:/app/models \\"
echo "    ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To check logs:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo ""
echo "To stop:"
echo "  docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}"
echo ""
