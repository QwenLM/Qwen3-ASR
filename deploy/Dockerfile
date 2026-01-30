# Qwen3-ASR Docker Deployment
# Base image with CUDA support
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV MODEL_NAME=Qwen/Qwen3-ASR-1.7B

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libsndfile1 \
    libsox-dev \
    sox \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create models directory
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
ENTRYPOINT ["./entrypoint.sh"]
