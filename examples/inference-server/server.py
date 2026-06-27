import os
import json
import io
import asyncio
import logging
import subprocess
from typing import Optional, List, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import time

import uvicorn
import numpy as np
import soundfile as sf
import torch
import psutil
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import Qwen-ASR components
try:
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
except ImportError:
    print("Warning: qwen_asr not found.")
    Qwen3ASRModel = None
    Qwen3ForcedAligner = None

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Config
# -----------------------------
def get_env_bool(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).lower() in ("true", "1", "yes", "on")

MAX_CONCURRENT_DECODE = int(os.getenv("MAX_CONCURRENT_DECODE", "4"))
MAX_CONCURRENT_INFER = int(os.getenv("MAX_CONCURRENT_INFER", "1"))  # GPU: usually 1
THREADPOOL_WORKERS = int(os.getenv("THREADPOOL_WORKERS", str((os.cpu_count() or 4) * 5)))

# Streaming buffering/throttling
STREAM_MIN_SAMPLES = int(os.getenv("STREAM_MIN_SAMPLES", "1600"))  # 100ms @ 16kHz
PARTIAL_INTERVAL_MS = int(os.getenv("PARTIAL_INTERVAL_MS", "120"))  # throttle partials
STREAM_EXPECT_SR = int(os.getenv("STREAM_EXPECT_SR", "16000"))

# -----------------------------
# App state
# -----------------------------
models = {}
model_status = "starting"
model_ready_event = asyncio.Event()

decode_sem = asyncio.Semaphore(MAX_CONCURRENT_DECODE)
infer_sem = asyncio.Semaphore(MAX_CONCURRENT_INFER)

# -----------------------------
# Helpers
# -----------------------------
async def to_thread_limited(sem: asyncio.Semaphore, fn, *args, **kwargs):
    async with sem:
        return await asyncio.to_thread(fn, *args, **kwargs)

def map_language(lang_code: Optional[str]) -> Optional[str]:
    """Map ISO code to Qwen full name."""
    if lang_code is None:
        return None
    mapping = {
        "en": "English", "de": "German", "fr": "French", "es": "Spanish",
        "it": "Italian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
        "ru": "Russian", "pt": "Portuguese", "nl": "Dutch", "tr": "Turkish",
        "sv": "Swedish", "id": "Indonesian", "vi": "Vietnamese",
        "hi": "Hindi", "ar": "Arabic",
    }
    return mapping.get(lang_code.lower(), lang_code)

def read_audio_file(file_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Sync decode. Must be called via asyncio.to_thread (or threadpool).
    soundfile first; fallback to ffmpeg for mp3/m4a/etc.
    """
    try:
        with io.BytesIO(file_bytes) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr
    except Exception:
        process = subprocess.Popen(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = process.communicate(input=file_bytes)
        if process.returncode != 0:
            raise ValueError(f"FFmpeg decoding failed: {err.decode(errors='ignore')}")
        with io.BytesIO(out) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr

# -----------------------------
# Model loading
# -----------------------------
async def load_models_background():
    global model_status
    logger.info("Background task: Loading models...")
    model_status = "loading_models"

    async def _load_asr():
        global model_status
        if not get_env_bool("ENABLE_ASR_MODEL", "true"):
            logger.info("ASR Model disabled via ENABLE_ASR_MODEL.")
            return
        if Qwen3ASRModel is None:
            raise RuntimeError("qwen_asr not installed (Qwen3ASRModel missing).")

        model_name = os.getenv("ASR_MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
        logger.info(f"Loading ASR Model: {model_name}...")
        gpu_mem = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.75"))
        max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "4096"))

        try:
            models["asr"] = await asyncio.to_thread(
                Qwen3ASRModel.LLM,
                model=model_name,
                gpu_memory_utilization=gpu_mem,
                max_new_tokens=max_new_tokens,
            )
            logger.info("ASR Model loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load ASR model: {e}")
            model_status = "error"
            raise

    async def _load_aligner():
        global model_status
        if not get_env_bool("ENABLE_ALIGNER_MODEL", "true"):
            logger.info("Aligner Model disabled via ENABLE_ALIGNER_MODEL.")
            return
        if Qwen3ForcedAligner is None:
            raise RuntimeError("qwen_asr not installed (Qwen3ForcedAligner missing).")

        aligner_name = os.getenv("ALIGNER_MODEL_NAME", "Qwen/Qwen3-ForcedAligner-0.6B")
        logger.info(f"Loading Aligner Model: {aligner_name}...")

        try:
            models["aligner"] = await asyncio.to_thread(
                Qwen3ForcedAligner.from_pretrained,
                aligner_name,
                dtype=torch.bfloat16,
                device_map="cuda:0",
            )
            logger.info("Aligner Model loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load Aligner model: {e}")
            model_status = "error"
            raise

    try:
        await asyncio.gather(_load_asr(), _load_aligner())
    except Exception:
        # model_status already set to "error" by loaders
        model_ready_event.set()  # don't hang endpoints
        return

    # Warmup (best-effort)
    if "asr" in models:
        logger.info("Warming up ASR model (best-effort)...")
        model_status = "warming_up"
        try:
            dummy_wav = np.zeros(16000, dtype=np.float32)
            dummy_sr = 16000

            async with infer_sem:
                await asyncio.to_thread(
                    models["asr"].transcribe,
                    audio=[(dummy_wav, dummy_sr)],
                    language=["English"],
                    return_time_stamps=False,
                )

            async with infer_sem:
                state = await asyncio.to_thread(
                    models["asr"].init_streaming_state,
                    unfixed_chunk_num=2,
                    unfixed_token_num=5,
                    chunk_size_sec=2.0,
                )

            warmup_chunks = [320, 640, 1024, 3200] + [3200] * 25
            for n in warmup_chunks:
                async with infer_sem:
                    await asyncio.to_thread(models["asr"].streaming_transcribe, dummy_wav[:n], state)

            async with infer_sem:
                await asyncio.to_thread(models["asr"].finish_streaming_transcribe, state)

            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    model_status = "ready"
    model_ready_event.set()
    logger.info("Server is ready to accept requests.")

# -----------------------------
# Lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Qwen3-ASR Server...")

    # Bigger threadpool helps when decoding + websocket buffering + other to_thread calls happen together.
    executor = ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS)
    app.state.executor = executor
    asyncio.get_running_loop().set_default_executor(executor)

    task = asyncio.create_task(load_models_background())
    try:
        yield
    finally:
        # Shutdown
        task.cancel()
        models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        executor.shutdown(wait=False, cancel_futures=True)
        logger.info("Shutdown complete.")

# -----------------------------
# App
# -----------------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
async def health():
    mem = psutil.virtual_memory()
    info = {
        "status": model_status,
        "limits": {
            "max_concurrent_decode": MAX_CONCURRENT_DECODE,
            "max_concurrent_infer": MAX_CONCURRENT_INFER,
            "threadpool_workers": THREADPOOL_WORKERS,
        },
        "memory": {
            "ram_total_mb": mem.total // (1024 * 1024),
            "ram_available_mb": mem.available // (1024 * 1024),
            "ram_percent": mem.percent,
        },
    }
    if torch.cuda.is_available():
        info["memory"]["gpu_allocated_mb"] = torch.cuda.memory_allocated() // (1024 * 1024)
        info["memory"]["gpu_reserved_mb"] = torch.cuda.memory_reserved() // (1024 * 1024)
    return info

@app.post("/transcribe")
async def transcribe(
    files: List[UploadFile] = File(...),
    language: Optional[str] = Query(None, description="Language code (e.g. en, de, fr). None for auto-detect."),
    forced_alignment: bool = Query(False, description="Enable forced alignment (timestamps)"),
):
    await model_ready_event.wait()

    if model_status != "ready":
        raise HTTPException(status_code=503, detail=f"Server not ready: {model_status}")
    if "asr" not in models:
        raise HTTPException(status_code=503, detail="ASR model is not enabled or failed to load.")

    full_lang = map_language(language)

    async def decode_one(f: UploadFile):
        content = await f.read()
        return await to_thread_limited(decode_sem, read_audio_file, content)

    # Decode concurrently (limited)
    try:
        audio_batch = await asyncio.gather(*(decode_one(f) for f in files))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    # Inference (explicitly limited, because GPU concurrency is not free)
    try:
        async with infer_sem:
            results = await asyncio.to_thread(
                models["asr"].transcribe,
                audio=audio_batch,
                language=[full_lang] * len(audio_batch),
                return_time_stamps=False,
            )

        response_list = []

        if forced_alignment:
            if "aligner" not in models:
                raise HTTPException(status_code=503, detail="Aligner model is not enabled or failed to load.")

            texts = [r.text for r in results]

            async with infer_sem:
                alignment_results = await asyncio.to_thread(
                    models["aligner"].align,
                    audio=audio_batch,
                    text=texts,
                    language=[full_lang] * len(audio_batch),
                )

            for i, res in enumerate(results):
                response_list.append(
                    {"text": res.text, "language": res.language, "timestamps": alignment_results[i]}
                )
        else:
            for res in results:
                response_list.append({"text": res.text, "language": res.language})

        return response_list

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/transcribe-streaming")
async def websocket_endpoint(
    ws: WebSocket,
    language: Optional[str] = Query(None),
    forced_alignment: bool = Query(False),  # kept for API symmetry; not yet used in streaming
):
    await ws.accept()

    # do wait until we know the outcome
    await model_ready_event.wait()

    if model_status != "ready" or "asr" not in models:
        await ws.close(code=1011, reason=f"Server not ready: {model_status}")
        return

    full_lang = map_language(language)
    client_sr = None
    started = False

    # Init streaming state off event loop + limited concurrency (GPU touch)
    try:
        async with infer_sem:
            state = await asyncio.to_thread(
                models["asr"].init_streaming_state,
                unfixed_chunk_num=2,
                unfixed_token_num=5,
                chunk_size_sec=2.0,
            )
    except Exception as e:
        logger.exception(f"Failed to init streaming state: {e}")
        await ws.close(code=1011, reason="init_streaming_state failed")
        return

    # Send ready
    try:
        await ws.send_json({"type": "ready"})
    except Exception:
        return

    buf_parts: List[np.ndarray] = []
    buf_n = 0
    last_partial_ts = 0.0

    async def flush_and_infer(send_partial: bool):
        nonlocal buf_parts, buf_n, last_partial_ts
        if buf_n <= 0:
            return
        chunk = np.concatenate(buf_parts, axis=0) if len(buf_parts) > 1 else buf_parts[0]

        async with infer_sem:
            await asyncio.to_thread(models["asr"].streaming_transcribe, chunk, state)

        if send_partial:
            now = time.monotonic()
            if (now - last_partial_ts) * 1000.0 >= PARTIAL_INTERVAL_MS:
                await ws.send_json({"type": "partial", "text": state.text, "language": state.language})

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg["type"] != "websocket.receive":
                continue

            # Control messages
            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    data = None

                if isinstance(data, dict):
                    t = data.get("type")

                    if t == "start":
                        started = True
                        client_sr = int(data.get("sample_rate_hz", 0)) if data.get("sample_rate_hz") else None
                        fmt = data.get("format")

                        if client_sr != STREAM_EXPECT_SR or fmt not in (None, "pcm_s16le"):
                            await ws.send_json(
                                {"type": "error", "message": f"Only pcm_s16le @ {STREAM_EXPECT_SR}Hz supported"}
                            )
                            await ws.close(code=1003)
                            return

                        # Optional: acknowledge language selection
                        if full_lang is not None:
                            await ws.send_json({"type": "info", "message": f"language={full_lang}"})
                        continue

                    if t == "stop":
                        # Flush remainder, finish, send final
                        await flush_and_infer(send_partial=False)
                        async with infer_sem:
                            await asyncio.to_thread(models["asr"].finish_streaming_transcribe, state)

                        await ws.send_json({"type": "final", "text": state.text, "language": state.language})
                        await ws.close(code=1000)
                        return

            # Audio frames
            if msg.get("bytes"):
                if not started:
                    # Require explicit start so we can validate format.
                    await ws.send_json({"type": "error", "message": "Send {type:'start', format:'pcm_s16le', sample_rate_hz:16000} first"})
                    await ws.close(code=1002)
                    return

                chunk_bytes = msg["bytes"]
                # int16 mono little-endian -> float32 [-1, 1]
                audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                if audio_int16.size == 0:
                    continue

                audio_f32 = audio_int16.astype(np.float32) / 32768.0
                buf_parts.append(audio_f32)
                buf_n += audio_f32.size

                if buf_n >= STREAM_MIN_SAMPLES:
                    await flush_and_infer(send_partial=True)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"WS Error: {e}")
        try:
            await ws.close(code=1011, reason="internal error")
        except Exception:
            pass

if __name__ == "__main__":
    # NOTE: for GPU models, keep workers=1 unless you deliberately replicate the model per worker.
    uvicorn.run(app, host="0.0.0.0", port=8000)
