# Qwen3-ASR ncnn deployment

This directory contains a CPU-first pnnx/ncnn path for Qwen3-ASR-0.6B. The C++
runtime performs the complete offline pipeline:

```text
16 kHz PCM -> log-mel -> audio conv -> audio encoder -> text embedding splice
-> decoder with KV cache -> lm_head -> language and transcript
```

The runtime source is integrated from the Apache-2.0
[`nihui/ncnn_llm`](https://github.com/nihui/ncnn_llm) implementation. The
Qwen3-ASR-specific exporter and fixture evaluator are retained as auditable
conversion tools. See [`NOTICE.md`](NOTICE.md) for provenance.

## Build

Install ncnn and nlohmann_json CMake packages, then run:

```bash
cmake -S ncnn -B build-ncnn-asr -G Ninja \
  -DCMAKE_PREFIX_PATH="/path/to/ncnn;/path/to/nlohmann_json" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-ncnn-asr --parallel
ctest --test-dir build-ncnn-asr --output-on-failure
```

The same CMake project is exercised with GCC on Ubuntu/WSL2 and MSYS2 UCRT64
on Windows. The CLI accepts PCM16 or float32 WAV and downmixes stereo to mono:

```bash
build-ncnn-asr/qwen3_asr_ncnn \
  --model /path/to/qwen3_asr_0.6b \
  --audio sample-16k.wav \
  --max-new-tokens 256
```

The converted model package is also published by the ncnn model zoo mirror:
`https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3_asr_0.6b/`.

## Export

`tools/qwen3_asr_export.py` separates the model into pnnx-friendly graphs for
the mel front end, convolutional audio projection, audio transformer, text
embedding, decoder and lm_head. Use an isolated environment so the pnnx wheel
and PyTorch ABI are paired:

```bash
python ncnn/tools/qwen3_asr_export.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --output-dir build/qwen3_asr_0.6b \
  --pnnx /path/to/venv/bin/pnnx
```

Run `python ncnn/tools/qwen3_asr_export.py --help` for the current model and
shape options.

## Parity contract

`tools/run_pytorch_fixtures.py` records PyTorch text, language, preprocessing
metadata and timings for a fixture manifest. `tools/evaluate_fixtures.py`
compares raw and normalized ncnn transcripts without silently discarding the
original strings. Use at least short Mandarin, longer Mandarin, and mixed
Chinese/English/numeric audio.

The model-independent CTest locks the audio-frame to audio-token calculation
at chunk boundaries (99, 100, 101 and 200 frames), where off-by-one conversion
errors are otherwise easy to miss.
