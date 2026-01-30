# Qwen3-ASR Docker Deployment
# Use smaller base image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV MODEL_NAME=Qwen/Qwen3-ASR-1.7B

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    libsndfile1 \
    libsox-dev \
    sox \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy and install qwen_asr package
COPY qwen_asr/ /app/qwen_asr/
COPY pyproject.toml /app/

# Copy requirements and install
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e /app

# Copy application code
COPY deploy/server.py .
COPY deploy/entrypoint.sh .

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
