# coding=utf-8
"""
Qwen3-ASR two-channel conversation transcription script (Transformers backend).
Transcribes a stereo audio file by processing each channel independently:
  1. Energy-based VAD splits each channel into utterance segments (no extra model needed).
  2. Each segment is transcribed separately (fast, no forced aligner required).
  3. All segments are merged and sorted by start time into a conversation JSON.

Usage:
  python scripts/transcribe_conversation.py -i <audio_file> [OPTIONS]

  --model-path/-mp        ASR model path (default: ./checkpoints/Qwen3-ASR-0.6B)
  --aligner-path/-ap      ForcedAligner path (default: ./checkpoints/Qwen3-ForcedAligner-0.6B)
  --input/-i              Stereo audio file path (required)
  --output/-o             JSON output path (default: results/<basename>-conversation.json)
  --language/-l           Force language, e.g. "Chinese", "English"; auto-detect if not set
  --device/-d             Inference device, e.g. "mps", "cuda:0", "cpu" (default: cuda:0)
  --dtype                 Model dtype: bfloat16 / float16 / float32 (default: bfloat16)
  --silence-gap/-sg       Min silence duration (s) between words to split utterances (default: 0.5)
  --silence-thresh/-st    RMS energy threshold for silence detection (default: 0.01)
  --min-speech/-ms        Min speech segment duration (s) to keep (default: 0.2)
  --channels/-c           Number of channels to process (default: 2)

Output format:
  {
    "source": "...",
    "conversations": [
      {"role": "channel_0", "text": "...", "start": 0.0, "end": 1.2},
      {"role": "channel_1", "text": "...", "start": 0.9, "end": 2.3},
      ...
    ]
  }
"""

import argparse
import json
import os
import tempfile
import time

import numpy as np
import soundfile as sf
import torch

from qwen_asr import Qwen3ASRModel


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR two-channel conversation transcription tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", "-mp", default="./checkpoints/Qwen3-ASR-0.6B", help="ASR model path")
    parser.add_argument("--aligner-path", "-ap", default="./checkpoints/Qwen3-ForcedAligner-0.6B", help="ForcedAligner path")
    parser.add_argument("--input", "-i", required=True, help="Stereo audio file path")
    parser.add_argument("--output", "-o", default=None, help="JSON output path (default: results/<basename>-conversation.json)")
    parser.add_argument("--language", "-l", default=None, help='Force language, e.g. "Chinese", "English"')
    parser.add_argument("--device", "-d", default="cuda:0", help='Inference device, e.g. "mps", "cuda:0", "cpu"')
    parser.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE_MAP.keys()), help="Model dtype")
    parser.add_argument("--silence-gap", "-sg", type=float, default=0.5, dest="silence_gap",
                        help="Min silence duration (s) between words to split utterances")
    parser.add_argument("--silence-thresh", "-st", type=float, default=0.01, dest="silence_thresh",
                        help="RMS energy threshold below which a frame is considered silent")
    parser.add_argument("--min-speech", "-ms", type=float, default=0.2, dest="min_speech",
                        help="Min speech segment duration (s); shorter segments are discarded")
    parser.add_argument("--channels", "-c", type=int, default=2,
                        help="Number of channels to process")
    return parser.parse_args()


def _energy_vad(audio: np.ndarray, sample_rate: int,
                silence_gap_s: float,
                silence_thresh: float,
                min_speech_s: float,
                frame_ms: int = 20) -> list:
    """
    Simple energy-based VAD.  Returns list of {"start": float, "end": float} dicts (seconds).

    Algorithm:
      - Compute per-frame RMS energy (frame = frame_ms ms).
      - Frames below silence_thresh are "silent".
      - A new segment starts when speech resumes after a silent gap >= silence_gap_s.
      - Segments shorter than min_speech_s are dropped.
    """
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = (len(audio) + frame_size - 1) // frame_size

    # RMS per frame
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

    # flush last segment
    if in_speech:
        end_frame = n_frames
        if (end_frame - start_frame) >= min_speech_frames:
            segments.append({
                "start": start_frame * frame_ms / 1000,
                "end":   len(audio) / sample_rate,
            })

    return segments


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        raise ValueError(f"--input must be a file, got: {args.input!r}")

    audio_data, sample_rate = sf.read(args.input, always_2d=True)
    total_dur_s = audio_data.shape[0] / sample_rate
    num_channels = audio_data.shape[1]
    channels_to_process = min(args.channels, num_channels)
    if num_channels < args.channels:
        print(f"[warning] audio has {num_channels} channel(s), processing {channels_to_process}")

    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_path = args.output or f"results/{basename}-conversation.json"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    dtype = _DTYPE_MAP[args.dtype]
    print(f"[config] model:         {args.model_path}")
    print(f"[config] device:        {args.device}  dtype: {args.dtype}")
    print(f"[config] silence_gap:   {args.silence_gap}s")
    print(f"[config] silence_thresh:{args.silence_thresh}")
    print(f"[config] min_speech:    {args.min_speech}s")
    print(f"[config] channels:      {channels_to_process}")
    print(f"[input]  {args.input}  ({num_channels} ch, {total_dur_s:.1f}s)")

    t0 = time.perf_counter()
    asr = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=args.device,
        forced_aligner=args.aligner_path,
        forced_aligner_kwargs=dict(dtype=dtype, device_map=args.device),
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    print(f"[timing] model load: {time.perf_counter() - t0:.3f}s")

    all_utterances = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for ch in range(channels_to_process):
            channel_audio = audio_data[:, ch]

            # VAD: find speech segments for this channel
            segments = _energy_vad(
                channel_audio, sample_rate,
                silence_gap_s=args.silence_gap,
                silence_thresh=args.silence_thresh,
                min_speech_s=args.min_speech,
            )
            print(f"\n[channel {ch}] {len(segments)} segment(s) detected by VAD")

            if not segments:
                print(f"[channel {ch}] no speech detected, skipping")
                continue

            # Transcribe each segment separately
            for seg_idx, seg in enumerate(segments):
                start_sample = int(seg["start"] * sample_rate)
                end_sample   = int(seg["end"]   * sample_rate)
                seg_audio    = channel_audio[start_sample:end_sample]

                tmp_wav = os.path.join(tmpdir, f"ch{ch}_seg{seg_idx}.wav")
                sf.write(tmp_wav, seg_audio, sample_rate)

                t1 = time.perf_counter()
                results = asr.transcribe(
                    audio=tmp_wav,
                    language=args.language,
                    return_time_stamps=False,
                )
                elapsed = time.perf_counter() - t1

                text = results[0].text.strip()
                print(f"[channel {ch}] seg {seg_idx:03d} "
                      f"[{seg['start']:.2f}s-{seg['end']:.2f}s] "
                      f"({elapsed:.2f}s) -> {text!r}")

                if text:
                    all_utterances.append({
                        "role":  f"channel_{ch}",
                        "text":  text,
                        "start": seg["start"],
                        "end":   seg["end"],
                    })

    # Sort by start time; break ties by channel index for determinism
    all_utterances.sort(key=lambda u: (u["start"], u["role"]))

    output = {
        "source":        args.input,
        "filename":      os.path.basename(args.input),
        "channels":      channels_to_process,
        "silence_gap":   args.silence_gap,
        "silence_thresh": args.silence_thresh,
        "conversations": all_utterances,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[result] {len(all_utterances)} utterance(s) in conversation")
    for u in all_utterances[:8]:
        print(f"  [{u['start']:.2f}-{u['end']:.2f}] {u['role']}: {u['text']!r}")
    if len(all_utterances) > 8:
        print(f"  ... ({len(all_utterances) - 8} more)")
    print(f"\n[output] JSON written: {output_path}")


if __name__ == "__main__":
    main()
