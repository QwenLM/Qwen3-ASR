# coding=utf-8
"""
VAD (Voice Activity Detection) utilities for Qwen3-ASR scripts.

Provides a unified apply_vad() interface supporting three backends:
  - 'simple'  : Energy-based VAD (no extra dependencies)
  - 'silero'  : Silero VAD (requires torch.hub / silero-vad package)
  - 'ten-vad' : TEN VAD (requires: pip install git+https://github.com/TEN-framework/ten-vad.git)
"""

from typing import List, Dict
import numpy as np
import soundfile as sf


def apply_vad(
    audio_path: str,
    vad_type: str = "simple",
    vad_model_path=None,
    silence_gap_s: float = 0.5,
    silence_thresh: float = 0.01,
    min_speech_s: float = 0.2,
) -> List[Dict[str, float]]:
    """
    Detect speech segments in an audio file.

    Parameters
    ----------
    audio_path      : path to a mono WAV file (any sample rate)
    vad_type        : 'simple' | 'silero' | 'ten-vad'
    vad_model_path  : reserved for future model-based backends (unused currently)
    silence_gap_s   : (simple) min silence gap (s) to split segments
    silence_thresh  : (simple) RMS energy threshold for silence
    min_speech_s    : minimum speech segment duration (s) to keep

    Returns
    -------
    List of {"start": float, "end": float} dicts in seconds.
    """
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    if vad_type == "simple":
        return _energy_vad(audio, sample_rate, silence_gap_s, silence_thresh, min_speech_s)
    elif vad_type == "silero":
        return _silero_vad(audio, sample_rate, min_speech_s)
    elif vad_type == "ten-vad":
        return _ten_vad(audio, sample_rate, min_speech_s)
    else:
        raise ValueError(f"Unknown vad_type: {vad_type!r}. Choose from 'simple', 'silero', 'ten-vad'.")


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
) -> List[Dict[str, float]]:
    """Silero VAD. Resamples to 16kHz if needed."""
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

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]

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
) -> List[Dict[str, float]]:
    """
    TEN VAD (https://github.com/TEN-framework/ten-vad).

    Install: pip install -U git+https://github.com/TEN-framework/ten-vad.git

    Processes audio frame-by-frame at 16kHz with hop_size=256 (16ms/frame).
    Accumulates speech frames (flag==1) into segments.
    """
    from ten_vad import TenVad

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

    segments = []
    in_speech = False
    speech_start_s = 0.0
    min_speech_frames = max(1, int(min_speech_s / frame_dur_s))
    speech_frame_count = 0

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
        else:
            if in_speech:
                if speech_frame_count >= min_speech_frames:
                    segments.append({
                        "start": speech_start_s,
                        "end":   t,
                    })
                in_speech = False
                speech_frame_count = 0

    # flush last segment
    if in_speech and speech_frame_count >= min_speech_frames:
        segments.append({
            "start": speech_start_s,
            "end":   n_frames * frame_dur_s,
        })

    return segments
