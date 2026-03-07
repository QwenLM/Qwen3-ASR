# Qwen3-ASR

<br>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/logo.png" width="400"/>
<p>

<p align="center">
&nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen3-asr">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen/Qwen3-ASR">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwen.ai/blog?id=qwen3asr">Blog</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://github.com/QwenLM/Qwen3-ASR/blob/main/assets/Qwen3_ASR.pdf">Paper</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen3-ASR">Hugging Face Demo</a>&nbsp&nbsp | &nbsp&nbsp 🖥️ <a href="https://modelscope.cn/studios/Qwen/Qwen3-ASR">ModelScope Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/qwen-speech-recognition">API</a>

</p>

We release **Qwen3-ASR**, a family that includes two powerful all-in-one speech recognition models that support language identification and ASR for 52 languages and dialects, as well as a novel non-autoregressive speech forced-alignment model that can align text–speech pairs in 11 languages.


## News
* 2026.1.29: 🎉🎉🎉 We have released the [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) series (0.6B/1.7B) and the Qwen3-ForcedAligner-0.6B model. Please check out our [blog](https://qwen.ai/blog?id=qwen3asr)!


## Contents <!-- omit in toc -->

- [Overview](#overview)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Released Models Description and Download](#released-models-description-and-download)
- [Quickstart](#quickstart)
  - [Environment Setup](#environment-setup)
  - [Python Package Usage](#python-package-usage)
    - [Quick Inference](#quick-inference)
    - [vLLM Backend](#vllm-backend)
    - [Streaming Inference](#streaming-inference)
    - [ForcedAligner Usage](#forcedaligner-usage)
  - [DashScope API Usage](#dashscope-api-usage)
- [Launch Local Web UI Demo](#launch-local-web-ui-demo)
  - [Gradio Demo](#gradio-demo)
  - [Streaming Demo](#streaming-demo)
- [Deployment with vLLM](#deployment-with-vllm)
- [Fine Tuning](#fine-tuning)
- [Docker](#docker)
- [Evaluation](#evaluation)
<!-- - [Citation](#citation) -->


## Overview

### Introduction

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/qwen3_asr_introduction.png" width="90%"/>
<p>

The Qwen3-ASR family includes Qwen3-ASR-1.7B and Qwen3-ASR-0.6B, which support language identification and ASR for 52 languages and dialects. Both leverage large-scale speech training data and the strong audio understanding capability of their foundation model, Qwen3-Omni. Experiments show that the 1.7B version achieves state-of-the-art performance among open-source ASR models and is competitive with the strongest proprietary commercial APIs. Here are the main features:

* **All-in-one**: Qwen3-ASR-1.7B and Qwen3-ASR-0.6B support language identification and speech recognition for 30 languages and 22 Chinese dialects, so as to English accents from multiple countries and regions.

* **Excellent and Fast**: The Qwen3-ASR family ASR models maintains high-quality and robust recognition under complex acoustic environments and challenging text patterns. Qwen3-ASR-1.7B achieves strong performance on both open-sourced and internal benchmarks. While the 0.6B version achieves accuracy-efficient trade-off, it reaches 2000 times throughput at a concurrency of 128. They both achieve streaming / offline unified inference with single model and support transcribe long audio.

* **Novel and strong forced alignment Solution**: We introduce Qwen3-ForcedAligner-0.6B, which supports timestamp prediction for arbitrary units within up to 5 minutes of speech in 11 languages. Evaluations show its timestamp accuracy surpasses E2E based forced-alignment models.

* **Comprehensive inference toolkit**: In addition to open-sourcing the architectures and weights of the Qwen3-ASR series, we also release a powerful, full-featured inference framework that supports vLLM-based batch inference, asynchronous serving, streaming inference, timestamp prediction, and more.

### Model Architecture

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/overview.jpg" width="100%"/>
<p>


### Released Models Description and Download

Below is an introduction and download information for the Qwen3-ASR models. Please select and download the model that fits your needs.

| Model | Supported Languages | Supported Dialects | Inference Mode | Audio Types |
|---|---|---|---|---|
| Qwen3-ASR-1.7B & Qwen3-ASR-0.6B | Chinese (zh), English (en), Cantonese (yue), Arabic (ar), German (de), French (fr), Spanish (es), Portuguese (pt), Indonesian (id), Italian (it), Korean (ko), Russian (ru), Thai (th), Vietnamese (vi), Japanese (ja), Turkish (tr), Hindi (hi), Malay (ms), Dutch (nl), Swedish (sv), Danish (da), Finnish (fi), Polish (pl), Czech (cs), Filipino (fil), Persian (fa), Greek (el), Hungarian (hu), Macedonian (mk), Romanian (ro) | Anhui, Dongbei, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei, Hunan, Jiangxi, Ningxia, Shandong, Shaanxi, Shanxi, Sichuan, Tianjin, Yunnan, Zhejiang, Cantonese (Hong Kong accent), Cantonese (Guangdong accent), Wu language, Minnan language. | Offline / Streaming | Speech, Singing Voice, Songs with BGM |
| Qwen3-ForcedAligner-0.6B | Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish | -- | NAR | Speech |

During model loading in the `qwen-asr` package or vLLM, model weights will be downloaded automatically based on the model name. However, if your runtime environment does not allow downloading weights during execution, you can use the following commands to manually download the model weights to a local directory:

```bash
# Download through ModelScope (recommended for users in Mainland China)
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B  --local_dir ./Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen3-ForcedAligner-0.6B
# Download through Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./Qwen3-ForcedAligner-0.6B
```


## Quickstart

### Environment Setup

The easiest way to use Qwen3-ASR is to install the `qwen-asr` Python package from PyPI. This will pull in the required runtime dependencies and allow you to load any released Qwen3-ASR model. If you’d like to simplify environment setup further, you can also use our official [Docker image](#docker). The `qwen-asr` package provides two backends: the transformers backend and the vLLM backend. For usage instructions for different backends, please refer to [Python Package Usage](#python-package-usage). We recommend using a **fresh, isolated environment** to avoid dependency conflicts with existing packages. You can create a clean Python 3.12 environment like this:

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
```

Run the following command to get the minimal installation with transformers-backend support:

```bash
pip install -U qwen-asr
```

To enable the vLLM backend for faster inference and streaming support, run:

```bash
pip install -U qwen-asr[vllm]
```

If you want to develop or modify the code locally, install from source in editable mode:

```bash
git clone https://github.com/QwenLM/Qwen3-ASR.git
cd Qwen3-ASR
pip install -e .
# support vLLM backend
# pip install -e ".[vllm]"
```

Additionally, we recommend using FlashAttention 2 to reduce GPU memory usage and accelerate inference speed, especially for long inputs and large batch sizes.

```bash
pip install -U flash-attn --no-build-isolation
```

If your machine has less than 96GB of RAM and lots of CPU cores, run:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention 2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

### Python Package Usage

#### Quick Inference

The `qwen-asr` package provides two backends: **transformers backend** and **vLLM backend**. You can pass audio inputs as a local path, a URL, base64 data, or a `(np.ndarray, sr)` tuple, and run batch inference. To quickly try Qwen3-ASR, you can use `Qwen3ASRModel.from_pretrained(...)` for the transformers backend with the following code:

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    language=None, # set "English" to force the language
)

print(results[0].language)
print(results[0].text)
```

If you want to return timestamps, pass `forced_aligner` and its init kwargs. Here is an example of batch inference with timestamps output:

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
        # attn_implementation="flash_attention_2",
    ),
)

results = model.transcribe(
    audio=[
      "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
      "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    ],
    language=["Chinese", "English"], # can also be set to None for automatic language detection
    return_time_stamps=True,
)

for r in results:
    print(r.language, r.text, r.time_stamps[0])
```

For more detailed usage examples, please refer to the [example code](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_transformers.py) for the transformers backend.

#### vLLM Backend

If you want the fastest inference speed with Qwen3-ASR, we strongly recommend using the vLLM backend by initializing the model with `Qwen3ASRModel.LLM(...)`. Example code is provided below. Note that you must install it via `pip install -U qwen-asr[vllm]`. If you want the model to output timestamps, it’s best to install FlashAttention via `pip install -U flash-attn --no-build-isolation` to speed up inference for the forced aligner model. Remember to wrap your code under `if __name__ == '__main__':` to avoid the `spawn` error described in [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/#python-multiprocessing).

```python
import torch
from qwen_asr import Qwen3ASRModel

if __name__ == '__main__':
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.7,
        max_inference_batch_size=128, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=4096, # Maximum number of tokens to generate. Set a larger value for long audio input.
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
        ),
    )

    results = model.transcribe(
        audio=[
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
        ],
        language=["Chinese", "English"], # can also be set to None for automatic language detection
        return_time_stamps=True,
    )

    for r in results:
        print(r.language, r.text, r.time_stamps[0])
```

For more detailed usage examples, please refer to the [example code](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm.py) for the vLLM backend. In addition, you can start a vLLM server via the `qwen-asr-serve` command, which is a wrapper around `vllm serve`. You can pass any arguments supported by `vllm serve`, for example:

```bash
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000
```

And send requests to the server via:

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
                    },
                }
            ],
        }
    ]
}

response = requests.post(url, headers=headers, json=data, timeout=300)
response.raise_for_status()
content = response.json()['choices'][0]['message']['content']
print(content)

# parse ASR output if you want
from qwen_asr import parse_asr_output
language, text = parse_asr_output(content)
print(language)
print(text)
```

#### Streaming Inference

Qwen3-ASR fully supports streaming inference. Currently, streaming inference is only available with the vLLM backend. Note that streaming inference does not support batch inference or returning timestamps. Please refer to the [example code](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py) for details. You can also launch a streaming web demo through the [guide](#streaming-demo) to experience Qwen3-ASR’s streaming transcription capabilities. 

#### ForcedAligner Usage

`Qwen3-ForcedAligner-0.6B` can align text–speech pairs and return word or character level timestamps. Here is an example of using the forced aligner directly:

```python
import torch
from qwen_asr import Qwen3ForcedAligner

model = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
)

results = model.align(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    text="甚至出现交易几乎停滞的情况。",
    language="Chinese",
)

print(results[0])
print(results[0][0].text, results[0][0].start_time, results[0][0].end_time)
```

In addition, the forced aligner supports local paths / URLs / base64 data / `(np.ndarray, sr)` inputs and batch inference. Please refer to the [example code](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_forced_aligner.py) for details.

### DashScope API Usage

To further explore Qwen3-ASR, we encourage you to try our DashScope API for a faster and more efficient experience. For detailed API information and documentation, please refer to the following:

| API Description | API Documentation (Mainland China) | API Documentation (International) |
|------------------|-----------------------------------|------------------------------------|
| Real-time API for Qwen3-ASR. | [https://help.aliyun.com/zh/model-studio/qwen-real-time-speech-recognition](https://help.aliyun.com/zh/model-studio/qwen-real-time-speech-recognition) | [https://www.alibabacloud.com/help/en/model-studio/qwen-real-time-speech-recognition](https://www.alibabacloud.com/help/en/model-studio/qwen-real-time-speech-recognition) |
| FileTrans API for Qwen3-ASR. | [https://help.aliyun.com/zh/model-studio/qwen-speech-recognition](https://help.aliyun.com/zh/model-studio/qwen-speech-recognition) | [https://www.alibabacloud.com/help/en/model-studio/qwen-speech-recognition](https://www.alibabacloud.com/help/en/model-studio/qwen-speech-recognition) |


## Launch Local Web UI Demo

### Gradio Demo

To launch the Qwen3-ASR web UI gradio demo, install the `qwen-asr` package and run `qwen-asr-demo`. Use the command below for help:

```bash
qwen-asr-demo --help
```

To launch the demo, you can use the following commands:

```bash
# Transformers backend
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --ip 0.0.0.0 --port 8000

# Transformers backend + Forced Aligner (enable timestamps)
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"device_map":"cuda:0","dtype":"bfloat16","max_inference_batch_size":8,"max_new_tokens":256}' \
  --aligner-kwargs '{"device_map":"cuda:0","dtype":"bfloat16"}' \
  --ip 0.0.0.0 --port 8000

# vLLM backend + Forced Aligner (enable timestamps)
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B \
  --backend vllm \
  --cuda-visible-devices 0 \
  --backend-kwargs '{"gpu_memory_utilization":0.7,"max_inference_batch_size":8,"max_new_tokens":2048}' \
  --aligner-kwargs '{"device_map":"cuda:0","dtype":"bfloat16"}' \
  --ip 0.0.0.0 --port 8000
```

Then open `http://<your-ip>:8000`, or access it via port forwarding in tools like VS Code.

#### Backend Notes

This demo supports two backends: transformers and vLLM. All backend-specific initialization parameters should be passed via `--backend-kwargs` as a JSON dict. If not provided, the demo will use sensible defaults.

```bash
# Example: override transformers init args with flash attention
--backend-kwargs '{"device_map":"cuda:0","dtype":"bfloat16","attn_implementation":"flash_attention_2"}'

# Example: override vLLM init args with 65% GPU memory
--backend-kwargs '{"gpu_memory_utilization":0.65}'
```

#### CUDA Device Notes

Because vLLM does not follow `cuda:0` style device selection, this demo selects GPUs by setting `CUDA_VISIBLE_DEVICES` via `--cuda-visible-devices`.

```bash
# Use GPU 0
--cuda-visible-devices 0

# Use GPU 1
--cuda-visible-devices 1
```

#### Timestamps Notes

Timestamps are only available when `--aligner-checkpoint` is provided. If you launch the demo without a forced aligner, the timestamps UI will be hidden automatically.

```bash
# No forced aligner
qwen-asr-demo --asr-checkpoint Qwen/Qwen3-ASR-1.7B

# With forced aligner
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B
```

#### HTTPS Notes

To avoid browser microphone permission issues after deploying the server, it is recommended/required to run the gradio service over HTTPS (especially when accessed remotely or behind modern browsers/gateways). Use `--ssl-certfile` and `--ssl-keyfile` to enable HTTPS. First, generate a private key and a self-signed certificate (valid for 365 days):

```bash
openssl req -x509 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes \
  -subj "/CN=localhost"
```

Then run the demo with HTTPS:

```bash
qwen-asr-demo \
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B \
  --backend transformers \
  --cuda-visible-devices 0 \
  --ip 0.0.0.0 --port 8000 \
  --ssl-certfile cert.pem \
  --ssl-keyfile key.pem \
  --no-ssl-verify
```

Then open `https://<your-ip>:8000` to use it. If your browser shows a warning, that’s expected for self-signed certificates. For production, use a real certificate.

### Streaming Demo

To experience Qwen3-ASR’s streaming transcription capability in a web UI, we provide a minimal Flask-based streaming demo. The demo captures microphone audio in the browser, resamples it to 16,000 Hz, and continuously pushes PCM chunks to the model. Run the demo with the following command:

```bash
qwen-asr-demo-streaming \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8000
```

Then open `http://<your-ip>:8000`, or access it via port forwarding in tools like VS Code.

## Deployment with vLLM

vLLM officially provides day-0 model support for Qwen3-ASR for efficient inference. 

### Installation
You can run Qwen3-ASR with vLLM nightly wheel or docker image. To install the nightly version of vLLM, we recommend using `uv` as the environment manager
```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
uv pip install "vllm[audio]" # For additional audio dependencies
```

### Online Serving
You can easily deploy Qwen3-ASR with vLLM by running the following command
```bash
vllm serve Qwen/Qwen3-ASR-1.7B
```
After the model server is successfully deployed, you can interact with it in multiple ways.

#### Using OpenAI SDK
```python
import base64
import httpx
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Create multimodal chat completion request
response = client.chat.completions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}
                    }
                }
            ]
        }
    ],
)

print(response.choices[0].message.content)
```
This model is also supported on vLLM with OpenAI transcription API.
```python
import httpx
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
audio_file = httpx.get(audio_url).content

transcription = client.audio.transcriptions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    file=audio_file,
)

print(transcription.text)
```

#### Using cURL
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "messages": [
    {"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}
    ]}
    ]
    }'
```

### Offline Inference
See the following example on using vLLM to run offline inference with Qwen3-ASR
```python
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
import base64
import requests

# Initialize the LLM
llm = LLM(
    model="Qwen/Qwen3-ASR-1.7B"
)

# Load audio
audio_asset = AudioAsset("winning_call")

# Create conversation with audio content
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": audio_asset.url}
            }
        ]
    }
]

sampling_params = SamplingParams(temperature=0.01, max_tokens=256)

# Run inference using .chat()
outputs = llm.chat(conversation, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```


## Fine Tuning

Please refer to [Qwen3-ASR-Finetuning](finetuning/) for detailed instructions on fine-tuning Qwen3-ASR.


## Docker

To make it easier to use our `qwen-asr` Python package, we provide a pre-built Docker image: [qwenllm/qwen3-asr](https://hub.docker.com/r/qwenllm/qwen3-asr). You only need to install the GPU driver and download the model files to run the code. Please follow the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to ensure Docker can access your GPU. If you are in Mainland China and have trouble reaching Docker Hub, you may use a registry mirror to accelerate image pulls.

First, pull the image and start a container:

```bash
LOCAL_WORKDIR=/path/to/your/workspace
HOST_PORT=8000
CONTAINER_PORT=80
docker run --gpus all --name qwen3-asr \
    -v /var/run/docker.sock:/var/run/docker.sock -p $HOST_PORT:$CONTAINER_PORT \
    --mount type=bind,source=$LOCAL_WORKDIR,target=/data/shared/Qwen3-ASR \
    --shm-size=4gb \
    -it qwenllm/qwen3-asr:latest
```

After running the command, you will enter the container’s bash shell. Your local workspace (**replace** `/path/to/your/workspace` **with the actual path**) will be mounted inside the container at `/data/shared/Qwen3-ASR`. Port `8000` on the host is mapped to port `80` in the container, so you can access services running in the container via `http://<host-ip>:8000`. Note that services inside the container must bind to `0.0.0.0` (not `127.0.0.1`) for port forwarding to work.

If you exit the container, you can start it again and re-enter it with:

```bash
docker start qwen3-asr
docker exec -it qwen3-asr bash
```

To remove the container completely, run:

```bash
docker rm -f qwen3-asr
```


## Evaluation

During evaluation, we ran inference for all models with `dtype=torch.bfloat16` and set `max_new_tokens=1024` using vLLM. Greedy search was used for all decoding, and none of the tests specified a language parameter. The detailed evaluation results are shown below.

<details>
<summary>ASR Benchmarks on Public Datasets (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th colspan="2" style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">English (en)</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Librispeech<br>clean | other</td>
      <td style="text-align: center;"><strong>1.39</strong> | 3.75</td>
      <td style="text-align: center;">2.89 | 3.56</td>
      <td style="text-align: center;">2.78 | 5.70</td>
      <td style="text-align: center;">1.51 | 3.97</td>
      <td style="text-align: center;">1.68 | 4.03</td>
      <td style="text-align: center;">2.11 | 4.55</td>
      <td style="text-align: center;">1.63 | <strong>3.38</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">GigaSpeech</td>
      <td style="text-align: center;">25.50</td>
      <td style="text-align: center;">9.37</td>
      <td style="text-align: center;">9.55</td>
      <td style="text-align: center;">9.76</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">8.88</td>
      <td style="text-align: center;"><strong>8.45</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-en</td>
      <td style="text-align: center;">9.08</td>
      <td style="text-align: center;">14.49</td>
      <td style="text-align: center;">13.78</td>
      <td style="text-align: center;">9.90</td>
      <td style="text-align: center;">9.90</td>
      <td style="text-align: center;">9.92</td>
      <td style="text-align: center;"><strong>7.39</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-en</td>
      <td style="text-align: center;"><strong>2.40</strong></td>
      <td style="text-align: center;">2.94</td>
      <td style="text-align: center;">6.31</td>
      <td style="text-align: center;">4.08</td>
      <td style="text-align: center;">5.49</td>
      <td style="text-align: center;">4.39</td>
      <td style="text-align: center;">3.35</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">MLS-en</td>
      <td style="text-align: center;">5.12</td>
      <td style="text-align: center;"><strong>3.68</strong></td>
      <td style="text-align: center;">7.09</td>
      <td style="text-align: center;">4.87</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">6.00</td>
      <td style="text-align: center;">4.58</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Tedlium</td>
      <td style="text-align: center;">7.69</td>
      <td style="text-align: center;">6.15</td>
      <td style="text-align: center;">4.91</td>
      <td style="text-align: center;">6.84</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>3.85<strong></td>
      <td style="text-align: center;"><strong>4.50</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">VoxPopuli</td>
      <td style="text-align: center;">10.29</td>
      <td style="text-align: center;">11.36</td>
      <td style="text-align: center;">12.12</td>
      <td style="text-align: center;">12.05</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>9.96<strong></td>
      <td style="text-align: center;"><strong>9.15</strong></td>
    </tr>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese (zh)</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech<br>net | meeting</td>
      <td style="text-align: center;">15.30 | 32.27</td>
      <td style="text-align: center;">14.43 | 13.47</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">9.86 | 19.11</td>
      <td style="text-align: center;">6.35 | -</td>
      <td style="text-align: center;">5.97 | 6.88</td>
      <td style="text-align: center;"><strong>4.97</strong> | <strong>5.88</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">AISHELL-2-test</td>
      <td style="text-align: center;">4.24</td>
      <td style="text-align: center;">11.62</td>
      <td style="text-align: center;">2.85</td>
      <td style="text-align: center;">5.06</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">3.15</td>
      <td style="text-align: center;"><strong>2.71</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">SpeechIO</td>
      <td style="text-align: center;">12.86</td>
      <td style="text-align: center;">5.30</td>
      <td style="text-align: center;">2.93</td>
      <td style="text-align: center;">7.56</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">3.44</td>
      <td style="text-align: center;"><strong>2.88</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-zh</td>
      <td style="text-align: center;">2.44</td>
      <td style="text-align: center;">2.71</td>
      <td style="text-align: center;">2.69</td>
      <td style="text-align: center;">4.09</td>
      <td style="text-align: center;">3.51</td>
      <td style="text-align: center;">2.88</td>
      <td style="text-align: center;"><strong>2.41</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-zh</td>
      <td style="text-align: center;">6.32</td>
      <td style="text-align: center;">7.70</td>
      <td style="text-align: center;">5.95</td>
      <td style="text-align: center;">12.91</td>
      <td style="text-align: center;">6.20</td>
      <td style="text-align: center;">6.89</td>
      <td style="text-align: center;"><strong>5.35</strong></td>
    </tr>
    <tr>
      <td colspan="9" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese Dialect</td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">KeSpeech</td>
      <td style="text-align: center;">26.87</td>
      <td style="text-align: center;">24.71</td>
      <td style="text-align: center;">5.27</td>
      <td style="text-align: center;">28.79</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">7.08</td>
      <td style="text-align: center;"><strong>5.10</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">Fleurs-yue</td>
      <td style="text-align: center;">4.98</td>
      <td style="text-align: center;">9.43</td>
      <td style="text-align: center;">4.98</td>
      <td style="text-align: center;">9.18</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">5.79</td>
      <td style="text-align: center;"><strong>3.98</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-yue</td>
      <td style="text-align: center;">11.36</td>
      <td style="text-align: center;">18.76</td>
      <td style="text-align: center;">13.20</td>
      <td style="text-align: center;">16.23</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">9.50</td>
      <td style="text-align: center;"><strong>7.57</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">CV-zh-tw</td>
      <td style="text-align: center;">6.32</td>
      <td style="text-align: center;">7.31</td>
      <td style="text-align: center;">4.06</td>
      <td style="text-align: center;">7.84</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">5.59</td>
      <td style="text-align: center;"><strong>3.77</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech-Yue<br>short | long</td>
      <td style="text-align: center;">15.62 | 25.29</td>
      <td style="text-align: center;">25.19 | 11.23</td>
      <td style="text-align: center;">9.74 | 11.40</td>
      <td style="text-align: center;">32.26 | 46.64</td>
      <td style="text-align: center;">- | -</td>
      <td style="text-align: center;">7.54 | 9.92</td>
      <td style="text-align: center;"><strong>5.82</strong> | <strong>8.85</strong></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: left;">WenetSpeech-Chuan<br>easy | hard</td>
      <td style="text-align: center;">34.81 | 53.98</td>
      <td style="text-align: center;">43.79 | 67.30</td>
      <td style="text-align: center;"><strong>11.40<strong> | <strong>20.20</strong></td>
      <td style="text-align: center;">14.35 | 26.80</td>
      <td style="text-align: center;">- | -</td>
      <td style="text-align: center;">13.92 | 24.45</td>
      <td style="text-align: center;">11.99 | 21.63</td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>ASR Benchmarks on Internal Datasets (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Accented English</td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Accented English</td>
      <td style="text-align: center;">28.56</td>
      <td style="text-align: center;">23.85</td>
      <td style="text-align: center;">20.41</td>
      <td style="text-align: center;">21.30</td>
      <td style="text-align: center;">19.96</td>
      <td style="text-align: center;"><strong>16.62<strong></td>
      <td style="text-align: center;"><strong>16.07</strong></td>
    </tr>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese Mandarin</td>
    </tr>
    <tr>
      <td style="text-align: left;">Elders&Kids</td>
      <td style="text-align: center;">14.27</td>
      <td style="text-align: center;">36.93</td>
      <td style="text-align: center;">4.17</td>
      <td style="text-align: center;">10.61</td>
      <td style="text-align: center;">4.54</td>
      <td style="text-align: center;">4.48</td>
      <td style="text-align: center;"><strong>3.81</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">ExtremeNoise</td>
      <td style="text-align: center;">36.11</td>
      <td style="text-align: center;">29.06</td>
      <td style="text-align: center;">17.04</td>
      <td style="text-align: center;">63.17</td>
      <td style="text-align: center;">36.55</td>
      <td style="text-align: center;">17.88</td>
      <td style="text-align: center;"><strong>16.17</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">TongueTwister</td>
      <td style="text-align: center;">20.87</td>
      <td style="text-align: center;">4.97</td>
      <td style="text-align: center;">3.47</td>
      <td style="text-align: center;">16.63</td>
      <td style="text-align: center;">9.02</td>
      <td style="text-align: center;">4.06</td>
      <td style="text-align: center;"><strong>2.44</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Mandarin</td>
      <td style="text-align: center;">20.73</td>
      <td style="text-align: center;">12.50</td>
      <td style="text-align: center;">6.61</td>
      <td style="text-align: center;">14.01</td>
      <td style="text-align: center;">7.32</td>
      <td style="text-align: center;">7.06</td>
      <td style="text-align: center;"><strong>6.54</strong></td>
    </tr>
    <tr>
      <td colspan="8" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Chinese Dialect</td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Cantonese</td>
      <td style="text-align: center;">16.05</td>
      <td style="text-align: center;">14.98</td>
      <td style="text-align: center;">7.56</td>
      <td style="text-align: center;">31.04</td>
      <td style="text-align: center;">5.85</td>
      <td style="text-align: center;"><strong>4.80<strong></td>
      <td style="text-align: center;"><strong>4.12</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Dialog-Chinese Dialects</td>
      <td style="text-align: center;">45.37</td>
      <td style="text-align: center;">47.70</td>
      <td style="text-align: center;">19.85</td>
      <td style="text-align: center;">44.55</td>
      <td style="text-align: center;">19.41</td>
      <td style="text-align: center;"><strong>18.24<strong></td>
      <td style="text-align: center;"><strong>15.94</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>Dialect coverage:</strong> Results for <em>Dialog-Accented English</em> are averaged over 16 accents, and results for <em>Dialog-Chinese Dialects</em> are averaged over 22 Chinese dialects.</p>

</details>

<details>
<summary>Multilingual ASR Benchmarks (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GLM-ASR<br>-Nano-2512</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR<br>-MLT-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="6" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Open-sourced Benchmarks</td>
    </tr>
    <tr>
      <td style="text-align: left;">MLS</td>
      <td style="text-align: center;">13.32</td>
      <td style="text-align: center;">8.62</td>
      <td style="text-align: center;">28.70</td>
      <td style="text-align: center;">13.19</td>
      <td style="text-align: center;"><strong>8.55</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">CommonVoice</td>
      <td style="text-align: center;">19.40</td>
      <td style="text-align: center;">10.77</td>
      <td style="text-align: center;">17.25</td>
      <td style="text-align: center;">12.75</td>
      <td style="text-align: center;"><strong>9.18</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MLC-SLM</td>
      <td style="text-align: center;">34.93</td>
      <td style="text-align: center;">15.68</td>
      <td style="text-align: center;">29.94</td>
      <td style="text-align: center;">15.84</td>
      <td style="text-align: center;"><strong>12.74</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs</td>
      <td style="text-align: center;">16.08</td>
      <td style="text-align: center;">5.27</td>
      <td style="text-align: center;">10.03</td>
      <td style="text-align: center;">7.57</td>
      <td style="text-align: center;"><strong>4.90</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs<sup>†</sup></td>
      <td style="text-align: center;">20.05</td>
      <td style="text-align: center;">6.85</td>
      <td style="text-align: center;">31.89</td>
      <td style="text-align: center;">10.37</td>
      <td style="text-align: center;"><strong>6.62</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs<sup>††</sup></td>
      <td style="text-align: center;">24.83</td>
      <td style="text-align: center;"><strong>8.16</strong></td>
      <td style="text-align: center;">47.84</td>
      <td style="text-align: center;">21.80</td>
      <td style="text-align: center;">12.60</td>
    </tr>
    <tr>
      <td colspan="6" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Qwen-ASR Internal Benchmarks</td>
    </tr>
    <tr>
      <td style="text-align: left;">News-Multilingual</td>
      <td style="text-align: center;">49.40</td>
      <td style="text-align: center;">14.80</td>
      <td style="text-align: center;">65.07</td>
      <td style="text-align: center;">17.39</td>
      <td style="text-align: center;"><strong>12.80</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>Language coverage:</strong> <em>MLS</em> includes 8 languages: {da, de, en, es, fr, it, pl, pt}.<br><em>CommonVoice</em> includes 13 languages: {en, zh, yue, zh_TW, ar, de, es, fr, it, ja, ko, pt, ru}.<br><em>MLC-SLM</em> includes 11 languages: {en, fr, de, it, pt, es, ja, ko, ru, th, vi}.<br><em>Fleurs</em> includes 12 languages: {en, zh, yue, ar, de, es, fr, it, ja, ko, pt, ru }.<br><em>Fleurs<sup>†</sup></em> includes 8 additional languages beyond Fleurs: {hi, id, ms, nl, pl, th, tr, vi}.<br><em>Fleurs<sup>††</sup></em> includes 10 additional languages beyond Fleurs<sup>†</sup>: {cs, da, el, fa, fi, fil, hu, mk, ro, sv}.<br><em>News-Multilingual</em> includes 15 languages: {ar, de, es, fr, hi, id, it, ja, ko, nl, pl, pt, ru, th, vi}.</p>

</details>

<details>
<summary>Language Identification Accuracy (%) ↑</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">Whisper-large-v3</th>
      <th style="text-align: center;">Qwen3-ASR-0.6B</th>
      <th style="text-align: center;">Qwen3-ASR-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">MLS</td>
      <td style="text-align: center;"><strong>99.9</strong></td>
      <td style="text-align: center;">99.3</td>
      <td style="text-align: center;"><strong>99.9</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">CommonVoice</td>
      <td style="text-align: center;">92.7</td>
      <td style="text-align: center;"><strong>98.2<strong></td>
      <td style="text-align: center;"><strong>98.7</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MLC-SLM</td>
      <td style="text-align: center;">89.2</td>
      <td style="text-align: center;"><strong>92.7<strong></td>
      <td style="text-align: center;"><strong>94.1</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Fleurs</td>
      <td style="text-align: center;">94.6</td>
      <td style="text-align: center;"><strong>97.1<strong></td>
      <td style="text-align: center;"><strong>98.7</strong></td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">94.1</td>
      <td style="text-align: center;"><strong>96.8<strong></td>
      <td style="text-align: center;"><strong>97.9</strong></td>
    </tr>
  </tbody>
</table>
<p><strong>Language coverage:</strong> The language sets follow Multilingual ASR Benchmarks. Here, Fleurs corresponds to Fleurs<sup>††</sup> in Multilingual ASR Benchmarks and covers 30 languages.</p>

</details>

<details>
<summary>Singing Voice & Song Transcription (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">GPT-4o<br>-Transcribe</th>
      <th style="text-align: center;">Gemini-2.5<br>-Pro</th>
      <th style="text-align: center;">Doubao-ASR<br>-1.0</th>
      <th style="text-align: center;">Whisper<br>-large-v3</th>
      <th style="text-align: center;">Fun-ASR-MLT<br>-Nano</th>
      <th style="text-align: center;">Qwen3-ASR<br>-1.7B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="7" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Singing</td>
    </tr>
    <tr>
      <td style="text-align: left;">M4Singer</td>
      <td style="text-align: center;">16.77</td>
      <td style="text-align: center;">20.88</td>
      <td style="text-align: center;">7.88</td>
      <td style="text-align: center;">13.58</td>
      <td style="text-align: center;">7.29</td>
      <td style="text-align: center;"><strong>5.98</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">MIR-1k-vocal</td>
      <td style="text-align: center;">11.87</td>
      <td style="text-align: center;">9.85</td>
      <td style="text-align: center;">6.56</td>
      <td style="text-align: center;">11.71</td>
      <td style="text-align: center;">8.17</td>
      <td style="text-align: center;"><strong>6.25</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Opencpop</td>
      <td style="text-align: center;">7.93</td>
      <td style="text-align: center;">6.49</td>
      <td style="text-align: center;">3.80</td>
      <td style="text-align: center;">9.52</td>
      <td style="text-align: center;"><strong>2.98</strong></td>
      <td style="text-align: center;">3.08</td>
    </tr>
    <tr>
      <td style="text-align: left;">Popcs</td>
      <td style="text-align: center;">32.84</td>
      <td style="text-align: center;">15.13</td>
      <td style="text-align: center;">8.97</td>
      <td style="text-align: center;">13.77</td>
      <td style="text-align: center;">9.42</td>
      <td style="text-align: center;"><strong>8.52</strong></td>
    </tr>
    <tr>
      <td colspan="7" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Songs with BGM</td>
    </tr>
    <tr>
      <td style="text-align: left;">EntireSongs-en</td>
      <td style="text-align: center;">30.71</td>
      <td style="text-align: center;"><strong>12.18</strong></td>
      <td style="text-align: center;">33.51</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">14.60</td>
    </tr>
    <tr>
      <td style="text-align: left;">EntireSongs-zh</td>
      <td style="text-align: center;">34.86</td>
      <td style="text-align: center;">18.68</td>
      <td style="text-align: center;">23.99</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;">N/A</td>
      <td style="text-align: center;"><strong>13.91</strong></td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>ASR Inference Mode Performance (WER ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;">Model</th>
      <th style="text-align: left;">Infer. Mode</th>
      <th style="text-align: center;">Librispeech</th>
      <th style="text-align: center;">Fleurs-en</th>
      <th style="text-align: center;">Fleurs-zh</th>
      <th style="text-align: center;">Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="text-align: left; vertical-align: middle;">Qwen3-ASR-1.7B</td>
      <td style="text-align: left;">Offline</td>
      <td style="text-align: center;">1.63 | 3.38</td>
      <td style="text-align: center;">3.35</td>
      <td style="text-align: center;">2.41</td>
      <td style="text-align: center;">2.69</td>
    </tr>
    <tr>
      <td style="text-align: left;">Streaming</td>
      <td style="text-align: center;">1.95 | 4.51</td>
      <td style="text-align: center;">4.02</td>
      <td style="text-align: center;">2.84</td>
      <td style="text-align: center;">3.33</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td rowspan="2" style="text-align: left; vertical-align: middle;">Qwen3-ASR-0.6B</td>
      <td style="text-align: left;">Offline</td>
      <td style="text-align: center;">2.11 | 4.55</td>
      <td style="text-align: center;">4.39</td>
      <td style="text-align: center;">2.88</td>
      <td style="text-align: center;">3.48</td>
    </tr>
    <tr>
      <td style="text-align: left;">Streaming</td>
      <td style="text-align: center;">2.54 | 6.27</td>
      <td style="text-align: center;">5.38</td>
      <td style="text-align: center;">3.40</td>
      <td style="text-align: center;">4.40</td>
    </tr>
  </tbody>
</table>

</details>

<details>
<summary>Forced Alignment Benchmarks (AAS ms ↓)</summary>

<table>
  <thead>
    <tr>
      <th style="text-align: left;"></th>
      <th style="text-align: center;">Monotonic-Aligner</th>
      <th style="text-align: center;">NFA</th>
      <th style="text-align: center;">WhisperX</th>
      <th style="text-align: center;">Qwen3-ForcedAligner-0.6B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">MFA-Labeled Raw</td>
    </tr>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">161.1</td>
      <td style="text-align: center;">109.8</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>33.1</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">107.5</td>
      <td style="text-align: center;">92.1</td>
      <td style="text-align: center;"><strong>37.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">100.7</td>
      <td style="text-align: center;">145.3</td>
      <td style="text-align: center;"><strong>41.7</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">122.7</td>
      <td style="text-align: center;">165.1</td>
      <td style="text-align: center;"><strong>46.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">142.7</td>
      <td style="text-align: center;">155.5</td>
      <td style="text-align: center;"><strong>75.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>37.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>38.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">200.7</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>40.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">124.7</td>
      <td style="text-align: center;">108.0</td>
      <td style="text-align: center;"><strong>36.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">161.1</td>
      <td style="text-align: center;">129.8</td>
      <td style="text-align: center;">133.2</td>
      <td style="text-align: center;"><strong>42.9</strong></td>
    </tr>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">MFA-Labeled Concat-300s</td>
    </tr>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">1742.4</td>
      <td style="text-align: center;">235.0</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>36.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">226.7</td>
      <td style="text-align: center;">227.2</td>
      <td style="text-align: center;"><strong>58.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">230.6</td>
      <td style="text-align: center;">2052.2</td>
      <td style="text-align: center;"><strong>53.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">220.3</td>
      <td style="text-align: center;">993.4</td>
      <td style="text-align: center;"><strong>62.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">290.5</td>
      <td style="text-align: center;">5719.4</td>
      <td style="text-align: center;"><strong>81.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>81.3</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>50.0</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">283.3</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>43.0</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">240.2</td>
      <td style="text-align: center;">4549.9</td>
      <td style="text-align: center;"><strong>39.6</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Cross-lingual</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>34.2</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">1742.4</td>
      <td style="text-align: center;">246.7</td>
      <td style="text-align: center;">2708.4</td>
      <td style="text-align: center;"><strong>52.9</strong></td>
    </tr>
    <tr>
      <td colspan="5" style="text-align: left; font-style: italic; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;">Human-Labeled</td>
    </tr>
    <tr>
      <td style="text-align: left;">Raw</td>
      <td style="text-align: center;">49.9</td>
      <td style="text-align: center;">88.6</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>27.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Raw-Noisy</td>
      <td style="text-align: center;">53.3</td>
      <td style="text-align: center;">89.5</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>41.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Concat-60s</td>
      <td style="text-align: center;">51.1</td>
      <td style="text-align: center;">86.7</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>25.3</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Concat-300s</td>
      <td style="text-align: center;">410.8</td>
      <td style="text-align: center;">140.0</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>24.8</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Concat-Cross-lingual</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>42.5</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;"><em>Avg.</em></td>
      <td style="text-align: center;">141.3</td>
      <td style="text-align: center;">101.2</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>32.4</strong></td>
    </tr>
  </tbody>
</table>

</details>


<!-- ## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@article{Qwen3-ASR,
  title={Qwen3-ASR Technical Report},
  author={},
  journal={arXiv preprint arXiv:},
  year={2026}
}
``` -->


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen3-ASR&type=Date)](https://star-history.com/#QwenLM/Qwen3-ASR&Date)

<br>