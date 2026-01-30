# Qwen3-ASR Docker 部署指南

## 快速开始

### 1. 构建镜像

```bash
cd deploy
chmod +x build.sh entrypoint.sh
./build.sh
```

### 2. 运行容器

**使用 docker run:**

```bash
# 使用 1.7B 模型 (推荐，精度更高)
docker run --gpus all -d \
  --name qwen3-asr-server \
  -p 8000:8000 \
  -e MODEL_NAME=Qwen/Qwen3-ASR-1.7B \
  -v ~/.cache/huggingface:/app/models \
  --shm-size=4g \
  qwen3-asr:latest

# 使用 0.6B 模型 (更快，显存更小)
docker run --gpus all -d \
  --name qwen3-asr-server \
  -p 8000:8000 \
  -e MODEL_NAME=Qwen/Qwen3-ASR-0.6B \
  -v ~/.cache/huggingface:/app/models \
  --shm-size=4g \
  qwen3-asr:latest
```

**使用 docker-compose:**

```bash
# 默认使用 1.7B 模型
docker-compose up -d

# 使用 0.6B 模型
MODEL_NAME=Qwen/Qwen3-ASR-0.6B docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 3. 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# API 文档
open http://localhost:8000/docs
```

## API 使用

### 1. 文件上传转写

```bash
curl -X POST http://localhost:8000/transcribe/file \
  -F "file=@audio.wav" \
  -F "language=" \
  -F "return_timestamps=false"
```

### 2. Base64 音频转写

```bash
# 将音频编码为 base64
AUDIO_B64=$(base64 -w 0 audio.wav)

curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d "{
    \"audio_base64\": \"${AUDIO_B64}\",
    \"language\": null,
    \"return_timestamps\": false
  }"
```

### 3. Python 客户端示例

```python
import base64
import requests

# 读取音频文件
with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# 发送请求
response = requests.post(
    "http://localhost:8000/transcribe",
    json={
        "audio_base64": audio_b64,
        "language": None,  # 自动检测语言
        "return_timestamps": False,
    }
)

result = response.json()
print(f"识别结果: {result['text']}")
print(f"语言: {result['language']}")
print(f"RTF: {result['rtf']:.4f}")
```

### 4. 批量转写

```python
import base64
import requests

# 准备多个音频
audios = []
for file in ["audio1.wav", "audio2.wav", "audio3.wav"]:
    with open(file, "rb") as f:
        audios.append(base64.b64encode(f.read()).decode())

# 批量请求
response = requests.post(
    "http://localhost:8000/transcribe/batch",
    json={
        "audios": audios,
        "languages": None,  # 自动检测
    }
)

result = response.json()
for i, r in enumerate(result["results"]):
    print(f"[{i}] {r['language']}: {r['text']}")

print(f"\n平均 RTF: {result['average_rtf']:.4f}")
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `Qwen/Qwen3-ASR-1.7B` | 模型名称 |
| `DEVICE` | `cuda:0` | 运行设备 |
| `DTYPE` | `bfloat16` | 数据类型 |
| `MAX_BATCH_SIZE` | `32` | 最大批处理大小 |
| `MAX_NEW_TOKENS` | `1024` | 最大生成 token 数 |
| `PORT` | `8000` | 服务端口 |
| `PRELOAD_MODEL` | `true` | 启动时预下载模型 |

## 模型选择

| 模型 | 显存需求 | RTF | 适用场景 |
|------|----------|-----|----------|
| Qwen3-ASR-0.6B | ~2GB | ~0.08 | 资源有限、实时性要求高 |
| Qwen3-ASR-1.7B | ~5GB | ~0.08 | 追求最高精度 |

## GPU 要求

- 最低显存: 4GB (0.6B 模型)
- 推荐显存: 8GB+ (1.7B 模型)
- CUDA 版本: 12.0+

## 停止服务

```bash
# docker run 方式
docker stop qwen3-asr-server
docker rm qwen3-asr-server

# docker-compose 方式
docker-compose down
```

## 常见问题

### 1. GPU 内存不足

使用 0.6B 模型：
```bash
MODEL_NAME=Qwen/Qwen3-ASR-0.6B docker-compose up -d
```

### 2. 模型下载慢

挂载本地 HuggingFace 缓存：
```bash
-v ~/.cache/huggingface:/app/models
```

或使用 ModelScope 镜像：
```bash
-e HF_ENDPOINT=https://hf-mirror.com
```

### 3. 服务启动慢

首次启动需要下载模型，约需 1-5 分钟。后续启动会使用缓存。
