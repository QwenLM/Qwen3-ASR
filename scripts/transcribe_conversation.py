# coding=utf-8
"""
Qwen3-ASR two-channel conversation transcription script (Transformers backend).
Transcribes a stereo audio file by processing each channel independently:
  1. VAD splits each channel into utterance segments.
  2. Each segment is transcribed separately (fast, no forced aligner required).
  3. All segments are merged and sorted by start time into a conversation JSON.

Usage:
  python scripts/transcribe_conversation.py -i <audio_file> [OPTIONS]

  --model-path/-mp        ASR model path (default: ./checkpoints/Qwen3-ASR-0.6B)
  --aligner-path/-ap      ForcedAligner path (default: ./checkpoints/Qwen3-ForcedAligner-0.6B)
  --input/-i              Stereo audio file path (required)
  --output/-o             JSON output path (default: results/<basename>.<model_name>.<vad>.<aligner_name>.json)
  --language/-l           Force language, e.g. "Chinese", "English"; auto-detect if not set
  --device/-d             Inference device, e.g. "mps", "cuda:0", "cpu" (default: cuda:0)
  --dtype                 Model dtype: bfloat16 / float16 / float32 (default: bfloat16)
  --silence-gap/-sg       Min silence duration (s) between words to split utterances (default: 0.5)
  --silence-thresh/-st    RMS energy threshold for silence detection (default: 0.01)
  --min-speech/-ms        Min speech segment duration (s) to keep (default: 0.2)
  --channels/-c           Number of channels to process (default: 2)
  --vad                   VAD backend: simple / silero / ten-vad / fsmn-vad (default: simple)
  --vad_model_path        Path to VAD model (reserved for future use)

Output format:
  {
    "source": "...",
    "filename": "...",
    "channels": 2,
    "audio_dur_s": 120.0,
    "transcribe_s": 20.0,
    "rtf": 0.167,
    "rtfx": 6.0,
    "vad_s": 0.5,
    "vad_rtf": 0.004,
    "vad_rtfx": 240.0,
    "model_name": "Qwen3-ASR-0.6B",
    "vad_model": "simple-vad",
    "aligner_model": "Qwen3-ForcedAligner-0.6B",
    "conversations": [
      {"role": "channel_0", "text": "...", "start": 0.0, "end": 1.2},
      {"role": "channel_1", "text": "...", "start": 0.9, "end": 2.3},
      ...
    ]
  }
"""

import argparse
import json
import logging
import os
import tempfile
import time

import soundfile as sf
import torch

from qwen_asr import Qwen3ASRModel
from vad_utils import apply_vad, init_vad


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--output", "-o", default=None, help="JSON output path (default: results/<basename>.<model_name>.<vad>.<aligner_name>.json)")
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
    parser.add_argument("--vad", default="simple", choices=["simple", "simple-vad", "silero", "silero-vad", "ten", "ten-vad", "fsmn", "fsmn-vad"],
                        help="VAD backend: simple (energy), silero, ten-vad, or fsmn-vad")
    parser.add_argument("--vad_model_path", default=None,
                        help="Path to VAD model (reserved for future use)")
    
    args = parser.parse_args()
    if "fsmn" in args.vad:
        args.vad = "fsmn-vad"
    elif "ten" in args.vad:
        args.vad = "ten-vad"
    elif "silero" in args.vad:
        args.vad = "silero-vad"
    elif "simple" in args.vad:
        args.vad = "simple-vad"

    return args


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        raise ValueError(f"--input must be a file, got: {args.input!r}")

    audio_data, sample_rate = sf.read(args.input, always_2d=True)
    total_dur_s = audio_data.shape[0] / sample_rate
    num_channels = audio_data.shape[1]
    channels_to_process = min(args.channels, num_channels)
    if num_channels < args.channels:
        logger.warning("[warning] audio has %d channel(s), processing %d", num_channels, channels_to_process)

    basename = os.path.splitext(os.path.basename(args.input))[0]
    model_name = os.path.basename(os.path.normpath(args.model_path))
    aligner_name = os.path.basename(os.path.normpath(args.aligner_path))
    output_path = args.output or f"results/{basename}.{model_name}.{args.vad}.{aligner_name}.json"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    dtype = _DTYPE_MAP[args.dtype]
    logger.info("[config] model=%s  aligner=%s", args.model_path, args.aligner_path)
    logger.info("[config] device=%s  dtype=%s  vad=%s", args.device, args.dtype, args.vad)
    if args.vad == "simple-vad":
        logger.info("[config] silence_gap=%.2fs  silence_thresh=%.4f", args.silence_gap, args.silence_thresh)
    logger.info("[config] min_speech=%.2fs  channels=%d", args.min_speech, channels_to_process)
    logger.info("[input]  %s  (%d ch, %.1fs)", args.input, num_channels, total_dur_s)

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
    logger.info("[timing] model loaded: %.3fs", time.perf_counter() - t0)

    logger.info("[vad] initializing %s VAD...", args.vad)
    t_vad = time.perf_counter()
    vad_instance = init_vad(args.vad, args.vad_model_path)
    logger.info("[vad] initialized in %.3fs", time.perf_counter() - t_vad)

    all_utterances = []
    total_vad_s = 0.0
    t_transcribe = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        for ch in range(channels_to_process):
            channel_audio = audio_data[:, ch]

            # Write channel audio to a temp WAV for apply_vad()
            tmp_ch_wav = os.path.join(tmpdir, f"ch{ch}_full.wav")
            sf.write(tmp_ch_wav, channel_audio, sample_rate)

            # VAD: find speech segments for this channel
            t_vad_ch = time.perf_counter()
            segments = apply_vad(
                tmp_ch_wav,
                vad_type=args.vad,
                vad_instance=vad_instance,
                silence_gap_s=args.silence_gap,
                silence_thresh=args.silence_thresh,
                min_speech_s=args.min_speech,
            )
            total_vad_s += time.perf_counter() - t_vad_ch
            logger.info("[channel %d] %d segment(s) detected by %s VAD", ch, len(segments), args.vad)

            if not segments:
                logger.info("[channel %d] no speech detected, skipping", ch)
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
                logger.info("[channel %d] seg %03d [%.2fs-%.2fs] (%.2fs) -> %r",
                            ch, seg_idx, seg["start"], seg["end"], elapsed, text)

                if text:
                    all_utterances.append({
                        "role":  f"channel_{ch}",
                        "text":  text,
                        "start": seg["start"],
                        "end":   seg["end"],
                    })

    # Sort by start time; break ties by channel index for determinism
    all_utterances.sort(key=lambda u: (u["start"], u["role"]))

    transcribe_s = time.perf_counter() - t_transcribe
    rtf = transcribe_s / total_dur_s if total_dur_s > 0 else None
    vad_rtf = total_vad_s / total_dur_s if total_dur_s > 0 else None
    logger.info("[timing] transcribe total: %.3fs", transcribe_s)
    logger.info("[timing] vad total:        %.3fs", total_vad_s)
    if rtf is not None:
        logger.info("[RTF]    RTF=%.4f  RTFx=%.2f", rtf, 1 / rtf)
    if vad_rtf is not None:
        logger.info("[VAD RTF] RTF=%.4f  RTFx=%.2f", vad_rtf, 1 / vad_rtf)

    output = {
        "source":        args.input,
        "filename":      os.path.basename(args.input),
        "channels":      channels_to_process,
        "audio_dur_s":   round(total_dur_s, 3),
        "transcribe_s":  round(transcribe_s, 3),
        "rtf":           round(rtf, 4) if rtf is not None else None,
        "rtfx":          round(1 / rtf, 2) if rtf else None,
        "vad_s":         round(total_vad_s, 3),
        "vad_rtf":       round(vad_rtf, 4) if vad_rtf is not None else None,
        "vad_rtfx":      round(1 / vad_rtf, 2) if vad_rtf else None,
        "model_name":    model_name,
        "vad_model":     args.vad,
        "aligner_model": aligner_name,
        "conversations": all_utterances,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("[result] %d utterance(s) in conversation", len(all_utterances))
    for u in all_utterances[:8]:
        logger.info("  [%.2f-%.2f] %s: %r", u["start"], u["end"], u["role"], u["text"])
    if len(all_utterances) > 8:
        logger.info("  ... (%d more)", len(all_utterances) - 8)
    logger.info("[output] saved: %s", output_path)


if __name__ == "__main__":
    main()
