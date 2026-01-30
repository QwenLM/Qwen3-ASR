# coding=utf-8
"""
Qwen3-ASR FastAPI Server

Provides REST API for speech recognition using Qwen3-ASR models.
"""

import os
import io
import time
import base64
import tempfile
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
import librosa
import soundfile as sf
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Global model instance
asr_model = None


class TranscribeRequest(BaseModel):
    """Request model for transcription with base64 audio."""
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    language: Optional[str] = Field(None, description="Force language (e.g., 'Chinese', 'English')")
    return_timestamps: bool = Field(False, description="Return word-level timestamps")


class TranscribeResponse(BaseModel):
    """Response model for transcription."""
    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Detected or forced language")
    duration: float = Field(..., description="Audio duration in seconds")
    inference_time: float = Field(..., description="Inference time in seconds")
    rtf: float = Field(..., description="Real-time factor")
    timestamps: Optional[List[dict]] = Field(None, description="Word-level timestamps")


class BatchTranscribeRequest(BaseModel):
    """Request model for batch transcription."""
    audios: List[str] = Field(..., description="List of base64 encoded audio data")
    languages: Optional[List[str]] = Field(None, description="List of forced languages")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    device: str
    gpu_memory_used: Optional[str] = None
    gpu_memory_total: Optional[str] = None


def get_gpu_memory():
    """Get GPU memory usage."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(', ')
            return f"{used} MiB", f"{total} MiB"
    except Exception:
        pass
    return None, None


def load_model():
    """Load the ASR model."""
    global asr_model
    
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
    device = os.environ.get("DEVICE", "cuda:0")
    dtype = os.environ.get("DTYPE", "bfloat16")
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}, Dtype: {dtype}")
    
    # Import here to avoid issues during import
    from qwen_asr import Qwen3ASRModel
    
    start_time = time.time()
    asr_model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=dtype_map.get(dtype, torch.bfloat16),
        device_map=device,
        max_inference_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "32")),
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "1024")),
    )
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s")
    
    # Warmup
    logger.info("Warming up model...")
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    _ = asr_model.transcribe(audio=(dummy_audio, 16000), language=None)
    logger.info("Model warmup complete")
    
    return asr_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting Qwen3-ASR server...")
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down Qwen3-ASR server...")


# Create FastAPI app
app = FastAPI(
    title="Qwen3-ASR API",
    description="Speech recognition API using Qwen3-ASR models",
    version="1.0.0",
    lifespan=lifespan,
)


def decode_audio(audio_data: str) -> tuple:
    """Decode base64 audio data to numpy array."""
    # Handle data URL format
    if audio_data.startswith("data:"):
        # Extract base64 part from data URL
        header, audio_data = audio_data.split(",", 1)
    
    # Decode base64
    audio_bytes = base64.b64decode(audio_data)
    
    # Load audio using soundfile
    with io.BytesIO(audio_bytes) as f:
        try:
            wav, sr = sf.read(f, dtype="float32")
        except Exception:
            # Try with librosa for other formats
            f.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                wav, sr = librosa.load(tmp.name, sr=None)
    
    return np.asarray(wav, dtype=np.float32), int(sr)


def get_audio_duration(wav: np.ndarray, sr: int) -> float:
    """Get audio duration in seconds."""
    return len(wav) / sr


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
    device = os.environ.get("DEVICE", "cuda:0")
    gpu_used, gpu_total = get_gpu_memory()
    
    return HealthResponse(
        status="healthy" if asr_model is not None else "loading",
        model=model_name,
        device=device,
        gpu_memory_used=gpu_used,
        gpu_memory_total=gpu_total,
    )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """
    Transcribe audio from base64 encoded data.
    
    Supports WAV, MP3, FLAC, and other common audio formats.
    """
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Decode audio
        wav, sr = decode_audio(request.audio_base64)
        duration = get_audio_duration(wav, sr)
        
        # Transcribe
        start_time = time.time()
        results = asr_model.transcribe(
            audio=(wav, sr),
            language=request.language,
            return_time_stamps=request.return_timestamps,
        )
        inference_time = time.time() - start_time
        
        result = results[0]
        rtf = inference_time / duration if duration > 0 else 0
        
        # Format timestamps if present
        timestamps = None
        if result.time_stamps:
            timestamps = [
                {
                    "text": ts.text,
                    "start": ts.start_time,
                    "end": ts.end_time,
                }
                for ts in result.time_stamps
            ]
        
        return TranscribeResponse(
            text=result.text,
            language=result.language,
            duration=duration,
            inference_time=inference_time,
            rtf=rtf,
            timestamps=timestamps,
        )
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/file", response_model=TranscribeResponse)
async def transcribe_file(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Force language"),
    return_timestamps: bool = Form(False, description="Return timestamps"),
):
    """
    Transcribe audio from uploaded file.
    
    Supports WAV, MP3, FLAC, and other common audio formats.
    """
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Read file content
        content = await file.read()
        
        # Load audio
        with io.BytesIO(content) as f:
            try:
                wav, sr = sf.read(f, dtype="float32")
            except Exception:
                # Try with librosa for other formats
                f.seek(0)
                with tempfile.NamedTemporaryFile(suffix=f".{file.filename.split('.')[-1]}", delete=True) as tmp:
                    tmp.write(content)
                    tmp.flush()
                    wav, sr = librosa.load(tmp.name, sr=None)
        
        wav = np.asarray(wav, dtype=np.float32)
        duration = get_audio_duration(wav, sr)
        
        # Transcribe
        start_time = time.time()
        results = asr_model.transcribe(
            audio=(wav, sr),
            language=language,
            return_time_stamps=return_timestamps,
        )
        inference_time = time.time() - start_time
        
        result = results[0]
        rtf = inference_time / duration if duration > 0 else 0
        
        # Format timestamps if present
        timestamps = None
        if result.time_stamps:
            timestamps = [
                {
                    "text": ts.text,
                    "start": ts.start_time,
                    "end": ts.end_time,
                }
                for ts in result.time_stamps
            ]
        
        return TranscribeResponse(
            text=result.text,
            language=result.language,
            duration=duration,
            inference_time=inference_time,
            rtf=rtf,
            timestamps=timestamps,
        )
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/batch")
async def transcribe_batch(request: BatchTranscribeRequest):
    """
    Batch transcribe multiple audio files.
    
    More efficient than calling /transcribe multiple times.
    """
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Decode all audios
        audio_data = []
        durations = []
        for audio_b64 in request.audios:
            wav, sr = decode_audio(audio_b64)
            audio_data.append((wav, sr))
            durations.append(get_audio_duration(wav, sr))
        
        # Prepare languages
        languages = request.languages
        if languages is None:
            languages = [None] * len(audio_data)
        elif len(languages) != len(audio_data):
            raise HTTPException(
                status_code=400,
                detail="Number of languages must match number of audios"
            )
        
        # Batch transcribe
        start_time = time.time()
        results = asr_model.transcribe(
            audio=audio_data,
            language=languages,
            return_time_stamps=False,
        )
        total_inference_time = time.time() - start_time
        
        # Format results
        responses = []
        for i, result in enumerate(results):
            dur = durations[i]
            # Estimate per-sample inference time proportionally
            per_sample_time = total_inference_time * (dur / sum(durations))
            rtf = per_sample_time / dur if dur > 0 else 0
            
            responses.append({
                "text": result.text,
                "language": result.language,
                "duration": dur,
                "inference_time": per_sample_time,
                "rtf": rtf,
            })
        
        return {
            "results": responses,
            "total_inference_time": total_inference_time,
            "total_duration": sum(durations),
            "average_rtf": total_inference_time / sum(durations) if sum(durations) > 0 else 0,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Qwen3-ASR API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/transcribe": "Transcribe base64 audio (POST)",
            "/transcribe/file": "Transcribe uploaded file (POST)",
            "/transcribe/batch": "Batch transcribe (POST)",
            "/docs": "API documentation",
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )
