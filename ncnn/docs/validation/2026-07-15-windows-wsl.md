# Windows + WSL validation — 2026-07-15

This record covers clean native-Windows and WSL2 builds of the `ncnn/`
integration, followed by a real end-to-end transcription with the published
Qwen3-ASR-0.6B ncnn package. Build directories and model weights are outside
the source tree.

## Environment

- Native Windows 11: MSYS2 UCRT64, GCC 16.1.0, CMake 4.4.0
- WSL2: Ubuntu 24.04, GCC 13.3.0, CMake 3.28.3
- ncnn package reported by CMake: `20260715`
- nlohmann_json: Windows 3.12.0; WSL 3.11.3
- Configuration: Release, Ninja, OpenMP enabled

## Native Windows clean build and CTest

```console
> cmake -S ncnn -B C:/Users/zhangzherui/Desktop/Tencent/ncnn-builds/qwen3-asr-official-windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:/Users/zhangzherui/Desktop/Tencent/ncnn-install-ucrt64;C:/msys64/ucrt64"
-- The CXX compiler identification is GNU 16.1.0
-- Found OpenMP_CXX: -fopenmp (found version "5.2")
-- Found ncnn: 20260715
-- Found nlohmann_json: 3.12.0
-- Configuring done
-- Generating done

> cmake --build C:/Users/zhangzherui/Desktop/Tencent/ncnn-builds/qwen3-asr-official-windows --parallel 2
[1/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_runtime.dir/src/sampling.cpp.obj
[2/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_runtime.dir/src/utils/rope_embed.cpp.obj
[3/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_runtime.dir/src/ncnn_llm_asr.cpp.obj
[4/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_runtime.dir/src/ncnn_text_runtime.cpp.obj
[5/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_runtime.dir/src/tokenizer/bpe_tokenizer.cpp.obj
[6/10] Linking CXX static library libqwen3_asr_ncnn_runtime.a
[7/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_tests.dir/tests/test_asr_layout.cpp.obj
[8/10] Linking CXX executable qwen3_asr_ncnn_tests.exe
[9/10] Building CXX object CMakeFiles/qwen3_asr_ncnn.dir/examples/asr_main.cpp.obj
[10/10] Linking CXX executable qwen3_asr_ncnn.exe

> ctest --test-dir C:/Users/zhangzherui/Desktop/Tencent/ncnn-builds/qwen3-asr-official-windows --output-on-failure
1/2 Test #1: qwen3_asr_ncnn_tests .............   Passed    0.75 sec
2/2 Test #2: qwen3_asr_ncnn_help ..............   Passed    0.80 sec
100% tests passed out of 2
```

## WSL2 clean build and CTest

```console
$ cmake -S /mnt/c/Users/zhangzherui/Desktop/Tencent/official-model-forks/Qwen3-ASR/ncnn -B /home/zhangzherui/builds/qwen3-asr-official -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/home/zhangzherui/install/ncnn;/usr"
-- The CXX compiler identification is GNU 13.3.0
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Found ncnn: 20260715
-- Found nlohmann_json: 3.11.3
-- Configuring done
-- Generating done

$ cmake --build /home/zhangzherui/builds/qwen3-asr-official --parallel 2
[1/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_runtime.dir/src/sampling.cpp.o
[2/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_runtime.dir/src/utils/rope_embed.cpp.o
[3/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_runtime.dir/src/ncnn_llm_asr.cpp.o
[4/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_runtime.dir/src/ncnn_text_runtime.cpp.o
[5/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_runtime.dir/src/tokenizer/bpe_tokenizer.cpp.o
[6/10] Linking CXX static library libqwen3_asr_ncnn_runtime.a
[7/10] Building CXX object CMakeFiles/qwen3_asr_ncnn_tests.dir/tests/test_asr_layout.cpp.o
[8/10] Linking CXX executable qwen3_asr_ncnn_tests
[9/10] Building CXX object CMakeFiles/qwen3_asr_ncnn.dir/examples/asr_main.cpp.o
[10/10] Linking CXX executable qwen3_asr_ncnn

$ ctest --test-dir /home/zhangzherui/builds/qwen3-asr-official --output-on-failure
1/2 Test #1: qwen3_asr_ncnn_tests .............   Passed    0.00 sec
2/2 Test #2: qwen3_asr_ncnn_help ..............   Passed    0.00 sec
100% tests passed, 0 tests failed out of 2
```

The Python exporter and fixture tools also pass
`python -m compileall -q ncnn/tools`.

## Real model and audio

The ncnn package was downloaded from
`https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3_asr_0.6b/`. Key SHA-256
digests are retained so another reviewer can identify the exact assets:

| Asset | SHA-256 |
| --- | --- |
| `model.json` | `216f9c9b390fb650154520320afeb5e088aa2b53ba435101bb6e94184e11f2d3` |
| `qwen3_asr_audio_conv.ncnn.bin` | `efc58d2089ccb69c7e8b821a36c98a42ac5dad69cc256f5b23eb8f192291fbb7` |
| `qwen3_asr_audio_encoder.ncnn.bin` | `262ac4d696b39f5ab5e6fa9f2a8c47f03898a05f73c270b55d3685ed3250fc8a` |
| `qwen3_asr_decoder.ncnn.bin` | `1f3ad4f9a32573335523ceff525d1627b60d2d8130af8740ee675861856d61fd` |
| `qwen3_asr_text_embed.ncnn.bin` | `26f178b8c242afd88027584e9f7679215888b0ba35acf8c7c432d8fc2fe11665` |

Input is the official
`https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav`
sample (`f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f`).
The source is mono 48 kHz PCM24; it was converted to the CLI's documented
16 kHz PCM16 contract without trimming:

```console
> ffmpeg -i asr_en.wav -ac 1 -ar 16000 -c:a pcm_s16le asr_en_16k_pcm16.wav
> ffprobe -show_entries stream=codec_name,sample_rate,channels,duration asr_en_16k_pcm16.wav
codec_name=pcm_s16le
sample_rate=16000
channels=1
duration=15.051250
```

Converted input SHA-256:
`9c33540cbba2cf82c2b724f733b002466b87f83c06e20fbe3c4eed83c141edc9`.

## End-to-end ncnn result

```console
> qwen3_asr_ncnn.exe --model qwen3_asr_0.6b --audio asr_en_16k_pcm16.wav --max-new-tokens 128
Audio: asr_en_16k_pcm16.wav  (240820 samples, 16000 Hz)
Loading ASR model from qwen3_asr_0.6b
[ncnn_llm_asr] loaded qwen3_asr: attn_cnt=28 hidden=1024 head_dim=128 theta=1e+06 vocab=151936
[ncnn_llm_asr] audio: samples=240820, mel_frames=1505, audio_tokens=196

==== Result ====
language: English
text: Hmm. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.
exit_code=0
elapsed_seconds=8.314
```

The WSL2 binary was then run against the exact same model directory mounted
from Windows and the exact same converted fixture. It returned the identical
language and raw transcript with exit code 0:

```console
$ /usr/bin/time qwen3_asr_ncnn --model /mnt/c/.../qwen3_asr_0.6b --audio /mnt/c/.../asr_en_16k_pcm16.wav --max-new-tokens 128
language: English
text: Hmm. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.
elapsed_seconds=21.41
max_rss_kb=4762740
exit_code=0
```

## Official PyTorch comparison

An isolated environment (`torch 2.13.0+cpu`, `transformers 4.57.6`,
`qwen-asr 0.0.6`) loaded the official Qwen3-ASR-0.6B safetensors file with
SHA-256
`79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea`.
The fixture runner used the same converted audio, automatic language detection,
greedy generation, and 128 maximum new tokens:

```console
{
  "language": "English",
  "text": "Hmm. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.",
  "total_time_ms": 7273.183107376099,
  "rtf": 0.483227845353449
}
```

The comparison result is `language_exact=true`, `raw_text_exact=true`, and
`normalized_text_exact=true`. Raw generated PyTorch result, mel summaries, and
the machine-readable comparison are retained under this directory. Timing is
not presented as a direct speed comparison: the ncnn stopwatch includes
process startup/model loading, while the PyTorch `total_time_ms` excludes its
separately recorded 6.250 s model load.
