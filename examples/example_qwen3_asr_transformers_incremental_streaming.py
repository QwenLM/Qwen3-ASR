# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental bounded-context streaming with Transformers encoder/KV reuse."""

import io
import os
import urllib.request

import numpy as np
import soundfile as sf
import torch

from qwen_asr import Qwen3ASRModel


ASR_MODEL_PATH = os.environ.get("QWEN_ASR_MODEL_PATH", "Qwen/Qwen3-ASR-0.6B")
URL_EN = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
SAMPLE_RATE = 16_000
INPUT_CHUNK_SECONDS = 0.5


def _read_audio(url: str) -> np.ndarray:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        audio, sample_rate = sf.read(
            io.BytesIO(response.read()), dtype="float32", always_2d=False
        )
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        duration = len(audio) / float(sample_rate)
        output_size = int(round(duration * SAMPLE_RATE))
        old_times = np.arange(len(audio), dtype=np.float64) / sample_rate
        new_times = np.arange(output_size, dtype=np.float64) / SAMPLE_RATE
        audio = np.interp(new_times, old_times, audio).astype(np.float32)
    return audio


def main() -> None:
    asr = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    stream = asr.init_transformers_incremental_stream(
        language="English",
        window_seconds=30.0,
        overlap_seconds=2.0,
    )

    audio = _read_audio(URL_EN)
    chunk_samples = int(INPUT_CHUNK_SECONDS * SAMPLE_RATE)
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        for update in stream.feed(chunk):
            print(
                f"[{update.audio_end_sec:6.2f}s] "
                f"language={update.language!r} text={update.text!r}"
            )

    final = stream.finish()
    print(f"[final] language={final.language!r} text={final.text!r}")


if __name__ == "__main__":
    main()
