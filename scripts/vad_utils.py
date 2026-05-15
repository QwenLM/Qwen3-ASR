# coding=utf-8
"""
VAD (Voice Activity Detection) utilities for Qwen3-ASR scripts.

Provides a unified interface supporting four backends:
  - 'simple'   : Energy-based VAD (no extra dependencies)
  - 'silero'   : Silero VAD (requires: pip install silero_vad)
  - 'ten-vad'  : TEN VAD (requires: pip install git+https://github.com/TEN-framework/ten-vad.git)
  - 'fsmn-vad' : FSMN VAD (requires: pip install fsmnvad)

Typical usage (one-time init, then apply per audio file):

    vad = init_vad("silero")
    for audio_path in files:
        segments = apply_vad(audio_path, vad_instance=vad, min_speech_s=0.2)
"""

from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf


def init_vad(vad_type: str = "simple", vad_model_path: Optional[str] = None) -> Any:
    """
    Initialize and return a VAD instance.  Call once before processing multiple files.

    Parameters
    ----------
    vad_type       : 'simple' | 'silero' | 'ten-vad' | 'fsmn-vad'
    vad_model_path : reserved for future model-based backends

    Returns
    -------
    - 'simple'   : None  (no state needed)
    - 'silero'   : (model, get_speech_timestamps) tuple
    - 'ten-vad'  : ten_vad.TenVad class (instance created fresh per audio to reset state)
    - 'fsmn-vad' : FSMNVad instance
    """
    if vad_type == "simple":
        return None
    elif vad_type == "silero":
        # Prefer the silero_vad pip package (has bundled model, no network needed).
        # Fall back to torch.hub if the package is not installed.
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps as _gst
            model = load_silero_vad(onnx=False)
            return (model, _gst)
        except ImportError:
            import os
            import torch
            repo = os.path.expanduser(vad_model_path) if vad_model_path else "snakers4/silero-vad"
            model, utils = torch.hub.load(
                repo_or_dir=repo,
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            return (model, utils[0])  # (model, get_speech_timestamps)
    elif vad_type == "ten-vad":
        from ten_vad import TenVad
        return TenVad  # return the class; instance created per audio to reset internal state
    elif vad_type == "fsmn-vad":
        from fsmnvad import FSMNVad
        return FSMNVad()
    else:
        raise ValueError(f"Unknown vad_type: {vad_type!r}. Choose from 'simple', 'silero', 'ten-vad', 'fsmn-vad'.")


def apply_vad(
    audio_path: str,
    vad_type: str = "simple",
    vad_instance: Any = None,
    silence_gap_s: float = 0.5,
    silence_thresh: float = 0.01,
    min_speech_s: float = 0.2,
) -> List[Dict[str, float]]:
    """
    Detect speech segments in an audio file.

    Parameters
    ----------
    audio_path    : path to a mono WAV file (any sample rate)
    vad_type      : 'simple' | 'silero' | 'ten-vad' | 'fsmn-vad'
    vad_instance  : pre-initialized VAD object from init_vad(); if None, initializes on the fly
    silence_gap_s : (simple/ten-vad) min silence gap (s) to split segments
    silence_thresh: (simple) RMS energy threshold for silence
    min_speech_s  : minimum speech segment duration (s) to keep

    Returns
    -------
    List of {"start": float, "end": float} dicts in seconds.
    """
    if vad_instance is None:
        vad_instance = init_vad(vad_type)

    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    if vad_type == "simple":
        return _energy_vad(audio, sample_rate, silence_gap_s, silence_thresh, min_speech_s)
    elif vad_type == "silero":
        model, get_speech_timestamps = vad_instance
        return _silero_vad(audio, sample_rate, min_speech_s, model, get_speech_timestamps)
    elif vad_type == "ten-vad":
        TenVad = vad_instance
        return _ten_vad(audio, sample_rate, min_speech_s, TenVad, silence_gap_s)
    elif vad_type == "fsmn-vad":
        return _fsmn_vad(audio_path, min_speech_s, vad_instance)
    else:
        raise ValueError(f"Unknown vad_type: {vad_type!r}. Choose from 'simple', 'silero', 'ten-vad', 'fsmn-vad'.")


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _energy_vad(
    audio: np.ndarray,
    sample_rate: int,
    silence_gap_s: float,
    silence_thresh: float,
    min_speech_s: float,
    frame_ms: int = 20,
) -> List[Dict[str, float]]:
    """Simple energy-based VAD using per-frame RMS."""
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = (len(audio) + frame_size - 1) // frame_size

    speech_mask = []
    for i in range(n_frames):
        frame = audio[i * frame_size: (i + 1) * frame_size].astype(np.float64)
        rms = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0.0
        speech_mask.append(rms > silence_thresh)

    min_silence_frames = max(1, int(silence_gap_s * 1000 / frame_ms))
    min_speech_frames  = max(1, int(min_speech_s  * 1000 / frame_ms))

    segments = []
    in_speech = False
    start_frame = 0
    silence_streak = 0

    for i, is_speech in enumerate(speech_mask):
        if is_speech:
            if not in_speech:
                start_frame = i
                in_speech = True
            silence_streak = 0
        else:
            if in_speech:
                silence_streak += 1
                if silence_streak >= min_silence_frames:
                    end_frame = i - silence_streak + 1
                    if (end_frame - start_frame) >= min_speech_frames:
                        segments.append({
                            "start": start_frame * frame_ms / 1000,
                            "end":   end_frame   * frame_ms / 1000,
                        })
                    in_speech = False
                    silence_streak = 0

    if in_speech:
        end_frame = n_frames
        if (end_frame - start_frame) >= min_speech_frames:
            segments.append({
                "start": start_frame * frame_ms / 1000,
                "end":   len(audio) / sample_rate,
            })

    return segments


def _silero_vad(
    audio: np.ndarray,
    sample_rate: int,
    min_speech_s: float,
    model,
    get_speech_timestamps,
) -> List[Dict[str, float]]:
    """Silero VAD. Resamples to 16kHz if needed. Uses pre-loaded model."""
    import torch

    target_sr = 16000
    if sample_rate != target_sr:
        dur = len(audio) / float(sample_rate)
        n_new = int(round(dur * target_sr))
        if n_new <= 0:
            return []
        x_old = np.linspace(0.0, dur, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, dur, num=n_new, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    else:
        audio = audio.astype(np.float32)

    wav_tensor = torch.from_numpy(audio)
    speech_timestamps = get_speech_timestamps(
        wav_tensor,
        model,
        sampling_rate=target_sr,
        min_speech_duration_ms=int(min_speech_s * 1000),
        return_seconds=True,
    )

    return [{"start": t["start"], "end": t["end"]} for t in speech_timestamps]


def _ten_vad(
    audio: np.ndarray,
    sample_rate: int,
    min_speech_s: float,
    TenVad,
    silence_gap_s: float = 0.5,
) -> List[Dict[str, float]]:
    """
    TEN VAD (https://github.com/TEN-framework/ten-vad).

    Processes audio frame-by-frame at 16kHz with hop_size=256 (16ms/frame).
    A fresh TenVad instance is created per call to reset internal state.
    Uses a silence_gap_s tolerance window (like energy VAD) so that brief
    pauses within a sentence don't split it into separate segments.
    """
    target_sr = 16000
    hop_size = 256  # 16ms at 16kHz

    # Resample to 16kHz if needed
    if sample_rate != target_sr:
        dur = len(audio) / float(sample_rate)
        n_new = int(round(dur * target_sr))
        if n_new <= 0:
            return []
        x_old = np.linspace(0.0, dur, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, dur, num=n_new, endpoint=False)
        audio16k = np.interp(x_new, x_old, audio).astype(np.float32)
    else:
        audio16k = audio.astype(np.float32)

    # Convert float32 [-1, 1] to int16
    audio_int16 = (audio16k * 32767).clip(-32768, 32767).astype(np.int16)

    vad = TenVad(hop_size=hop_size, threshold=0.5)
    n_frames = len(audio_int16) // hop_size
    frame_dur_s = hop_size / target_sr  # 0.016s per frame

    min_speech_frames  = max(1, int(min_speech_s  / frame_dur_s))
    min_silence_frames = max(1, int(silence_gap_s / frame_dur_s))

    segments = []
    in_speech = False
    speech_start_s = 0.0
    speech_frame_count = 0
    silence_streak = 0

    for i in range(n_frames):
        frame = audio_int16[i * hop_size: (i + 1) * hop_size]
        _, flag = vad.process(frame)
        t = i * frame_dur_s

        if flag == 1:
            if not in_speech:
                speech_start_s = t
                in_speech = True
                speech_frame_count = 0
            speech_frame_count += 1
            silence_streak = 0
        else:
            if in_speech:
                silence_streak += 1
                if silence_streak >= min_silence_frames:
                    end_t = t - silence_streak * frame_dur_s
                    if speech_frame_count >= min_speech_frames:
                        segments.append({
                            "start": speech_start_s,
                            "end":   end_t,
                        })
                    in_speech = False
                    speech_frame_count = 0
                    silence_streak = 0

    # flush last segment
    if in_speech and speech_frame_count >= min_speech_frames:
        segments.append({
            "start": speech_start_s,
            "end":   n_frames * frame_dur_s,
        })

    return segments


def _fsmn_vad(
    audio_path: str,
    min_speech_s: float,
    vad_instance,
) -> List[Dict[str, float]]:
    """
    FSMN VAD (https://github.com/alibaba-damo-academy/FunASR).

    segments_offline() requires a 16kHz mono WAV file and returns
    [[start_ms, end_ms], ...] in milliseconds.
    """
    from pathlib import Path
    import tempfile
    import soundfile as sf2

    audio_path_obj = Path(audio_path)

    # FSMN VAD requires 16kHz mono WAV; resample/convert if needed
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    need_resample = (sr != 16000)
    need_mono = (audio.ndim == 2)

    if need_resample or need_mono:
        if need_mono:
            audio = audio.mean(axis=1)
        if need_resample:
            dur = len(audio) / float(sr)
            n_new = int(round(dur * 16000))
            if n_new <= 0:
                return []
            x_old = np.linspace(0.0, dur, num=len(audio), endpoint=False)
            x_new = np.linspace(0.0, dur, num=n_new, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf2.write(tmp.name, audio, 16000)
        audio_path_obj = Path(tmp.name)

    raw = vad_instance.segments_offline(audio_path_obj)  # [[start_ms, end_ms], ...]

    min_speech_ms = min_speech_s * 1000
    return [
        {"start": seg[0] / 1000.0, "end": seg[1] / 1000.0}
        for seg in raw
        if (seg[1] - seg[0]) >= min_speech_ms
    ]
