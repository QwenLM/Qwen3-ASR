# Fast Qwen3-ASR Inference Server (vLLM Backend, FastAPI async processing)

A containerized Qwen3-ASR inferenceserver using FastAPI and vLLM.
Provides HTTP `/transcribe` and WebSocket `/transcribe-streaming` endpoints.

## Setup

1. **Build the image**:

```bash
make build
```

2. **Start the server**:
```bash
make up
```

Server will be available at `http://localhost:8907` by default.

**!!FIRST START NOTICE!!** Downloading, CUDA graph compile, loading the model into VRAM and warmup can take a few minutes depending on your internet speed and GPU.

3. **Check logs**:
   ```bash
   make logs
   ```
   Wait for "Application startup complete".

## Configuration

Environment variables can be set in `Makefile` or passed to `make up`:

- `ENABLE_ASR_MODEL` (default: true)
- `ENABLE_ALIGNER_MODEL` (default: true)
- `ASR_MODEL_NAME` (default: Qwen/Qwen3-ASR-1.7B)
- `ALIGNER_MODEL_NAME` (default: Qwen/Qwen3-ForcedAligner-0.6B)

## Endpoints

### `POST /transcribe`
Upload an audio file for ASR.

- **URL**: `http://localhost:8907/transcribe`
- **Method**: POST (Multipart/Form-Data)
- **Parameters**: 
  - `file`: Audio file
  - `language`: Target language (e.g. `de`)
  - `forced_alignment`: true/false

### `WS /transcribe-streaming`
Stream raw PCM audio for real-time transcription.

- **URL**: `ws://localhost:8907/transcribe-streaming`
- **Protocol**: Send JSON `start`, then binary PCM chunks (16k, 16-bit, mono), then JSON `stop`.

### `GET /health`

- **URL**: `http://localhost:8907/health`
- **Method**: GET

## Testing

### Health Check
Get the stats and model loading status.

```bash
make health
```

```bash
Checking health...
{
  "status": "ready",
  "limits": {
    "max_concurrent_decode": 4,
    "max_concurrent_infer": 1,
    "threadpool_workers": 320
  },
  "memory": {
    "ram_total_mb": 515491,
    "ram_available_mb": 489283,
    "ram_percent": 5.1,
    "gpu_allocated_mb": 1756,
    "gpu_reserved_mb": 1766
  }
}
```

**RESULTS: How much VRAM does Qwen3-ASR require?**:
`Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B` together take < 2GB VRAM at PEAK - even with 80 concurrent streams in processing.

### Automated Test
Runs a verification script using cached sample files (downloads them once to `files/`).

```bash
make test
```

```bash
Running verification test...
Testing files/reference.m4a
Response: [{"text":"Das ist ein Referenztext.","language":"German"}]
PASS: 'referenz' found.
Testing files/reference.mp3
Response: [{"text":"Das ist ein Referenztext.","language":"German"}]
PASS: 'referenz' found.
Testing files/reference.wav
Response: [{"text":"Das ist ein Referenztext.","language":"German"}]
PASS: 'referenz' found.
Results: 3 PASSED, 0 FAILED
```

### Streaming Client Tests
Scripts `client-streaming.py` and `client-streaming.js` are provided to test the streaming endpoint.
The `make test-streaming` target will automatically convert a sample MP3 to PCM and run both clients.

```bash
# Install local dependencies (Python venv + Node modules) - only required for testing
make setup-local

# Run streaming tests (requires container to be UP)
make test-streaming
```

```bash
Running Python Streaming Test...
Connecting to ws://127.0.0.1:8907/transcribe-streaming...
Audio Duration: 2.06s
Streaming files/reference.pcm...
[19:00:53.518] [Server Ready]
[19:00:53.520] [Partial] (): Finished sending audio.

[19:00:53.598] [Final] (German): Das ist ein Referenztext.

Processing Time: 0.09s
Real-Time Factor (RTF): 0.0457

Running Node.js Streaming Test...
Connecting to ws://127.0.0.1:8907/transcribe-streaming...
Audio Duration: 2.06s
Connected.

[19:00:53.661] [ready] {"type":"ready"}
Finished sending audio.
[19:00:53.664] [Partial] 
[19:00:53.741] [Final] Das ist ein Referenztext.

Processing Time: 0.09s
Real-Time Factor (RTF): 0.0431

Disconnected.
```

**RESULTS: How FAST is Qwen3-ASR?**:
Qwen-ASR runs at roughly 20x real-time on a NVIDIA H200 NVL with Flash Attention 2.

### Forced Aligner Tests

```bash
make test-aligner
```

```bash
Running Forced Alignment Verification Test...
[
  {
    "text": "Das ist ein Referenztext.",
    "language": "German",
    "timestamps": {
      "items": [
        {
          "text": "Das",
          "start_time": 0.16,
          "end_time": 0.32
        },
        {
          "text": "ist",
          "start_time": 0.32,
          "end_time": 0.48
        },
        {
          "text": "ein",
          "start_time": 0.48,
          "end_time": 0.64
        },
        {
          "text": "Referenztext",
          "start_time": 0.64,
          "end_time": 1.68
        }
      ]
    }
  }
]
```

**Manual Usage**:

*Note: The clients expect raw PCM audio (16k, 16-bit, mono). MP3/WAV files must be converted first.*

**Convert Audio to PCM**:
```bash
ffmpeg -i files/reference.mp3 -f s16le -ac 1 -ar 16000 files/reference.pcm
```

**Python Client**:
```bash
source .venv/bin/activate
python client-streaming.py -e ws://localhost:8907/transcribe-streaming -f files/reference.pcm
```

**Node.js Client**:
```bash
node client-streaming.js -e ws://localhost:8907/transcribe-streaming -f files/reference.pcm
```

## Benchmarking

You can benchmark this server by running two commands:

### Benchmark batching

Defaults to 4 clients running 20 transcriptions in parallel.

```bash
make benchmark-batch
```

#### Results

1x NVIDIA H200 NVL:

```bash
Running Batch Benchmark (Concurrency: 4, Requests: 20)...
Starting Benchmark: 4 clients, 20 requests each.

--- Benchmark Results ---
Total Requests:     80
Successful:         80
Errors:             0
Total Wall Time:    2.81s
Avg QPS:            28.51

-- Latency (sec) --
Avg:  0.138
P50:  0.139
P95:  0.142
```

### Benchmark streaming

Defaults to 4 clients running 20 transcriptions in parallel.

```bash
make benchmark-streaming
```

#### Results

1x NVIDIA H200 NVL:

```bash
Running Streaming Benchmark (Concurrency: 4, Requests: 20)...
Starting Benchmark: 4 clients, 20 requests each.

--- Benchmark Results ---
Total Requests:     80
Successful:         80
Errors:             0
Total Wall Time:    6.09s
Avg QPS:            13.14

-- Latency (sec) --
Avg:  0.229
P50:  0.230
P95:  0.240

-- RTF --
Avg:  0.1109
P50:  0.1122
P95:  0.1179
```

