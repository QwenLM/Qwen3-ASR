# Provenance and licenses

This directory is an independent clean-branch integration for
Tencent/ncnn#6790. No contributor branch or commit history was used as its Git
base.

- The C++ runtime and tokenizer utilities are from `nihui/ncnn_llm` at commit
  `f2f29e4` and remain under its Apache-2.0 license, copied as
  `LICENSE.ncnn_llm`.
- The pnnx exporter and fixture tools are retained from the Apache-2.0 public
  `linkeLi0421/ncnn_llm` `qwen3-asr-export` branch at commit `35174d2`. They
  are committed here without that contributor branch's Git ancestry and are
  not represented as newly invented model architecture code.
- Qwen3-ASR model code and weights remain subject to the upstream Qwen3-ASR
  repository and model licenses.

The independent work in this branch is official-repository integration,
portable CMake packaging, boundary regression tests, Windows/WSL reproduction,
and durable validation evidence.
