# scripts/

Utility scripts for running Qwen3-ASR inference locally.

## Prerequisites

```bash
# Models (pick one of the two methods)
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./checkpoints/Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./checkpoints/Qwen3-ForcedAligner-0.6B

# Test audio (optional)
python scripts/download_test_audio.py
```

### Optional: VAD backends for `transcribe_conversation.py`

By default, `transcribe_conversation.py` uses the built-in energy-based VAD (`--vad simple`, no extra dependencies).
To use a neural VAD backend, install the corresponding package:

```bash
# Silero VAD (recommended neural VAD, bundled model — no network needed at runtime)
pip install silero_vad

# TEN VAD (low-latency streaming neural VAD)
pip install git+https://github.com/TEN-framework/ten-vad.git

# FSMN VAD
pip install onnxruntime kaldi-native-fbank
git clone https://github.com/lovemefan/fsmn-vad
cd fsmn-vad && python setup.py install
```

---

## Single-file Transcription

### `transcribe.py` — Transformers backend

```bash
python scripts/transcribe.py -i <audio_file> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path / -mp` | `./checkpoints/Qwen3-ASR-0.6B` | ASR model path |
| `--aligner-path / -ap` | `./checkpoints/Qwen3-ForcedAligner-0.6B` | ForcedAligner path |
| `--input / -i` | *(required)* | Audio file path |
| `--output / -o` | `results/<basename>-asr_result.json` | JSON output path |
| `--language / -l` | auto-detect | Force language, e.g. `Chinese` |
| `--device / -d` | `cuda:0` | Inference device, e.g. `mps`, `cpu` |
| `--dtype` | `bfloat16` | Model dtype: `bfloat16` / `float16` / `float32` |
| `--timestamps / -ts` | off | Enable word-level timestamps |
| `--seperate_channel / -sc` | off | Split multi-channel audio; transcribe each channel separately |

### `transcribe_vllm.py` — vLLM backend

Same interface as `transcribe.py`, with additional options:

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu-memory-util / -gmu` | `0.8` | vLLM GPU memory utilization |
| `--aligner-device / -ad` | `cuda:0` | ForcedAligner device |
| `--max-new-tokens` | `1024` | Max tokens for generation |

### `transcribe_vllm_streaming.py` — vLLM streaming

```bash
python scripts/transcribe_vllm_streaming.py -i <audio_file> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path / -mp` | `./checkpoints/Qwen3-ASR-1.7B` | ASR model path |
| `--input / -i` | *(required)* | Audio file path |
| `--output / -o` | `results/<basename>-asr_result_streaming.json` | JSON output path |
| `--language / -l` | auto-detect | Force language |
| `--gpu-memory-util / -gmu` | `0.8` | vLLM GPU memory utilization |
| `--max-new-tokens` | `32` | Max new tokens per streaming chunk |
| `--chunk-size-sec` | `2.0` | Streaming chunk size (seconds) |
| `--step-ms` | `1000` | Audio feed step (milliseconds) |
| `--seperate_channel / -sc` | off | Split multi-channel audio; transcribe each channel separately |

> Streaming does not support word-level timestamps.

---

## Batch Transcription

### `batch_transcribe.py` — Transformers backend

```bash
python scripts/batch_transcribe.py -i <audio_dir> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path / -mp` | `./checkpoints/Qwen3-ASR-0.6B` | ASR model path |
| `--aligner-path / -ap` | `./checkpoints/Qwen3-ForcedAligner-0.6B` | ForcedAligner path |
| `--input / -i` | *(required)* | Directory of `.wav` files |
| `--output / -o` | `./results/batch_asr_results.tsv` | TSV output path |
| `--language / -l` | auto-detect | Force language |
| `--timestamps / -ts` | off | Enable word-level timestamps |
| `--device / -d` | `cuda:0` | Inference device |
| `--dtype` | `bfloat16` | Model dtype |
| `--aligner-device / -ad` | `cuda:0` | ForcedAligner device |
| `--batch-size / -bs` | `1` | Inference batch size |
| `--seperate_channel / -sc` | off | Split multi-channel audio; transcribe each channel separately |

Output: TSV + `<stem>_summary.json`

### `batch_transcribe_vllm.py` — vLLM backend

Same interface as `batch_transcribe.py`, with `--gpu-memory-util / -gmu` (default `0.8`) instead of `--device / --dtype`.

### `batch_transcribe_vllm_streaming.py` — vLLM streaming

```bash
python scripts/batch_transcribe_vllm_streaming.py -i <audio_dir> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path / -mp` | `./checkpoints/Qwen3-ASR-1.7B` | ASR model path |
| `--input / -i` | *(required)* | Directory of `.wav` files |
| `--output / -o` | `./results/batch_asr_results_vllm_streaming.tsv` | TSV output path |
| `--language / -l` | auto-detect | Force language |
| `--gpu-memory-util / -gmu` | `0.8` | vLLM GPU memory utilization |
| `--max-new-tokens` | `32` | Max new tokens per streaming chunk |
| `--chunk-size-sec` | `2.0` | Streaming chunk size (seconds) |
| `--step-ms` | `1000` | Audio feed step (milliseconds) |
| `--seperate_channel / -sc` | off | Split multi-channel audio; transcribe each channel separately |

---

## Conversation Transcription

### `transcribe_conversation.py` — Multi-channel → conversation JSON

Splits each channel with VAD, transcribes segments independently, then merges all utterances sorted by start time into a multi-turn conversation JSON.

```bash
python scripts/transcribe_conversation.py -i <stereo_audio> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path / -mp` | `./checkpoints/Qwen3-ASR-0.6B` | ASR model path |
| `--aligner-path / -ap` | `./checkpoints/Qwen3-ForcedAligner-0.6B` | ForcedAligner path |
| `--input / -i` | *(required)* | Stereo audio file path |
| `--output / -o` | `results/<basename>-conversation.json` | JSON output path |
| `--language / -l` | auto-detect | Force language |
| `--device / -d` | `cuda:0` | Inference device |
| `--dtype` | `bfloat16` | Model dtype |
| `--channels / -c` | `2` | Number of channels to process |
| `--vad` | `simple` | VAD backend: `simple` / `silero` / `ten-vad` |
| `--vad_model_path` | — | VAD model path (for `silero` / `ten-vad`) |
| `--silence-gap / -sg` | `0.5` | Min silence gap (s) to split utterances (`simple` / `ten-vad`) |
| `--silence-thresh / -st` | `0.01` | RMS energy threshold for silence (`simple` only) |
| `--min-speech / -ms` | `0.2` | Min speech segment duration (s) to keep |

**VAD backends:**

| Backend | Dependency | Notes |
|---------|-----------|-------|
| `simple` | none | Energy-based RMS VAD |
| `silero` | `pip install silero_vad` | Neural VAD; more accurate |
| `ten-vad` | `pip install git+https://github.com/TEN-framework/ten-vad.git` | Low-latency neural VAD |
| `fsmn-vad` | see below | FSMN-based neural VAD |

**Output format:**
```json
{
  "source": "path/to/audio.wav",
  "conversations": [
    {"role": "channel_0", "text": "...", "start": 0.0, "end": 1.2},
    {"role": "channel_1", "text": "...", "start": 0.9, "end": 2.3}
  ]
}
```

**Examples:**

```bash
# Energy VAD (default)
python scripts/transcribe_conversation.py -i data/call.wav -d mps -l Chinese

# Silero VAD
python scripts/transcribe_conversation.py -i data/call.wav -d mps --vad silero

# TEN VAD with wider silence tolerance
python scripts/transcribe_conversation.py -i data/call.wav -d mps --vad ten-vad --silence-gap 0.8
```

---

## VAD Utilities

### `vad_utils.py`

Shared VAD module used by `transcribe_conversation.py`. Provides a unified interface for four backends:

| Backend | Dependency | Description |
|---------|-----------|-------------|
| `simple` | none | Energy-based RMS VAD. Splits on frames where RMS energy falls below `--silence-thresh`. Fast, no model needed. |
| `silero` | `pip install silero_vad` | Neural VAD using Silero model (bundled in pip package). More accurate, especially for noisy audio. Resamples to 16kHz internally. |
| `ten-vad` | `pip install git+https://github.com/TEN-framework/ten-vad.git` | Low-latency streaming neural VAD. Processes audio frame-by-frame at 16kHz with 16ms hop. Uses same `--silence-gap` tolerance window as `simple`. |
| `fsmn-vad` | see install below | FSMN-based neural VAD. Requires 16kHz mono WAV (auto-resampled). Returns millisecond-precise segments. |

Can also be imported directly:

```python
from vad_utils import init_vad, apply_vad

# Initialize once
vad = init_vad("silero")

# Apply to multiple files
for path in audio_files:
    segments = apply_vad(path, vad_type="silero", vad_instance=vad, min_speech_s=0.2)
```
