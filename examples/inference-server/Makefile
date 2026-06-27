IMAGE              ?= qwenllm/qwen3-asr:cu128
CONTAINER          ?= qwen3-asr-fast
PORT               ?= 8907
HF_HOME            ?= /var/lib/docker/container_volumes/hf_models
VLLM_CACHE_ROOT    ?= $(HF_HOME)/vllm_cache
FILES_DIR          ?= files
TEST_URL_LIST      ?= https://raw.githubusercontent.com/kyr0/asr-sample-files/refs/heads/main/index.m3u
GPU_MEM_UTIL       ?= 0.15
ASR_MODEL_NAME     ?= Qwen/Qwen3-ASR-1.7B
ALIGNER_MODEL_NAME ?= Qwen/Qwen3-ForcedAligner-0.6B

.PHONY: build up down logs health test setup-local test-streaming test-batch test-aligner benchmark-streaming benchmark-batch

build:
	docker build --rm -t $(IMAGE) .

up:
	@echo "Starting container with:"
	@echo "  IMAGE:              $(IMAGE)"
	@echo "  CONTAINER:          $(CONTAINER)"
	@echo "  PORT:               $(PORT)"
	@echo "  HF_HOME:            $(HF_HOME)"
	@echo "  VLLM_CACHE_ROOT:    $(VLLM_CACHE_ROOT)"
	@echo "  GPU_MEM_UTIL:       $(GPU_MEM_UTIL)"
	@echo "  ASR_MODEL_NAME:     $(ASR_MODEL_NAME)"
	@echo "  ALIGNER_MODEL_NAME: $(ALIGNER_MODEL_NAME)"
	@read -p "Proceed? [y/N] " ans && [ $${ans:-N} = y ] || (echo "Aborted." && exit 1)
	mkdir -p $(FILES_DIR)
	docker run -d --name $(CONTAINER) \
	  --gpus all --ipc=host \
	  -p $(PORT):8000 \
	  -e HF_HOME="/data/shared/hf_models" \
	  -e VLLM_CACHE_ROOT="/data/shared/hf_models/vllm_cache" \
	  -e ENABLE_ASR_MODEL=true \
	  -e ENABLE_ALIGNER_MODEL=true \
	  -e ASR_MODEL_NAME=$(ASR_MODEL_NAME) \
	  -e ALIGNER_MODEL_NAME=$(ALIGNER_MODEL_NAME) \
	  -e GPU_MEMORY_UTILIZATION=$(GPU_MEM_UTIL) \
	  -v $(PWD):/data/shared/Qwen3-ASR \
	  -v $(HF_HOME):/data/shared/hf_models \
	  $(IMAGE)
	@$(MAKE) logs

down:
	-docker rm -f $(CONTAINER)

logs:
	docker logs -f $(CONTAINER)

health:
	@echo "Checking health..."
	@curl -s http://localhost:$(PORT)/health | jq .

test:
	@echo "Running verification test..."
	@mkdir -p $(FILES_DIR)
	@# Download samples if missing
	@for ext in m4a mp3 wav; do \
		if [ ! -f "$(FILES_DIR)/reference.$$ext" ]; then \
			echo "Downloading reference.$$ext..."; \
			wget -q -O $(FILES_DIR)/reference.$$ext https://raw.githubusercontent.com/kyr0/asr-sample-files/refs/heads/main/reference.$$ext; \
		fi \
	done
	
	@pass=0; fail=0; \
	for ext in m4a mp3 wav; do \
		file="$(FILES_DIR)/reference.$$ext"; \
		echo "Testing $$file"; \
		resp=$$(curl -s -X POST -F "files=@$$file" "http://localhost:$(PORT)/transcribe?language=de"); \
		echo "Response: $$resp"; \
		if echo "$$resp" | grep -iq "referenz"; then \
			echo "PASS: 'referenz' found."; \
			pass=$$((pass+1)); \
		else \
			echo "FAIL: 'referenz' not found."; \
			fail=$$((fail+1)); \
		fi; \
	done; \
	echo "Results: $$pass PASSED, $$fail FAILED"; \
	if [ $$fail -gt 0 ]; then exit 1; fi

setup-local:
	@echo "Setting up local Python environment..."
	@if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
	@. .venv/bin/activate && pip install -r requirements.txt
	@echo "Setting up local Node.js environment..."
	@npm install --silent

test-streaming:
	@echo "Running Python Streaming Test..."
	@. .venv/bin/activate && python client-streaming.py -e ws://127.0.0.1:$(PORT)/transcribe-streaming -f $(FILES_DIR)/reference.pcm
	@echo "\nRunning Node.js Streaming Test..."
	@node client-streaming.js -e ws://127.0.0.1:$(PORT)/transcribe-streaming -f $(FILES_DIR)/reference.pcm

test-batch:
	@echo "Running Batch Verification Test..."
	@# Send multiple files (reusing the same file 3 times for batching test)
	@curl -s -X POST "http://127.0.0.1:$(PORT)/transcribe?language=de" \
		-F "files=@$(FILES_DIR)/reference.m4a" \
		-F "files=@$(FILES_DIR)/reference.mp3" \
		-F "files=@$(FILES_DIR)/reference.wav" | jq .

test-aligner:
	@echo "Running Forced Alignment Verification Test..."
	@curl -s -X POST "http://127.0.0.1:$(PORT)/transcribe?language=de&forced_alignment=true" \
		-F "files=@$(FILES_DIR)/reference.wav" | jq .

benchmark-streaming:
	@echo "Running Streaming Benchmark (Concurrency: 4, Requests: 20)..."
	@. .venv/bin/activate && python benchmark.py --mode streaming --url ws://127.0.0.1:$(PORT)/transcribe-streaming --file $(FILES_DIR)/reference.pcm --clients 4 --requests 20

benchmark-batch:
	@echo "Running Batch Benchmark (Concurrency: 4, Requests: 20)..."
	@. .venv/bin/activate && python benchmark.py --mode batch --url http://127.0.0.1:$(PORT)/transcribe --file $(FILES_DIR)/reference.wav --clients 4 --requests 20

