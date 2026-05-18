# coding=utf-8
"""
Qwen3-ASR single-file transcription script (Transformers backend, macOS/MPS).
Transcribes a single audio file and outputs a JSON with full timestamps and timing stats.

Usage:
  python scripts/transcribe.py -i <audio_file> [OPTIONS]

  --model-path/-mp      ASR model path (default: ./checkpoints/Qwen3-ASR-0.6B)
  --aligner-path/-ap    ForcedAligner path (default: ./checkpoints/Qwen3-ForcedAligner-0.6B)
  --input/-i            Audio file path (required)
  --output/-o           JSON output path (default: results/<input_basename>.<model_name>.no_vad.<aligner_name>.json)
  --language/-l         Force language, e.g. "Chinese", "English"; auto-detect if not set
  --timestamps/-ts      Enable word-level timestamps
  --silence-gap/-sg     Silence gap (s) to split word-level timestamps into segments (default: 0.5; 0 = no split)
  --device/-d           Inference device, e.g. "mps", "cuda:0", "cpu" (default: cuda:0)
  --dtype               Model dtype: bfloat16 / float16 / float32 (default: bfloat16)
  --seperate_channel/-sc  Split multi-channel audio and transcribe each channel separately

Output format (json):
  {
    "source": "...",
    "filename": "...",
    "audio_dur_s": 12.34,
    "model_load_s": 1.0,
    "transcribe_s": 1.23,
    "align_s": 0.5,
    "rtf": 0.1,
    "rtfx": 10.0,
    "align_rtf": 0.04,
    "align_rtfx": 25.0,
    "model_name": "Qwen3-ASR-0.6B",
    "vad_model": "no_vad",
    "aligner_model": "Qwen3-ForcedAligner-0.6B",
    "language": "Chinese",
    "text": "full transcription text",
    "segments": [
      {"text": "segment text", "start": 0.0, "end": 2.5},
      ...
    ],
    "words": [
      {"text": "字", "start": 0.12, "end": 0.36},
      ...
    ]
  }
  segments: word-level timestamps aggregated into sentence segments by silence_gap.
            Empty list when --timestamps is not set.
  words: raw word-level timestamps; empty list when --timestamps is not set.
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


def _audio_duration(path: str) -> float:
    return sf.info(path).duration


def _timed(label: str, fn, *args, audio_duration_s: float = 0.0, **kwargs):
    """Run fn and return (result, elapsed_s), logging elapsed time and RTF."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    logger.info("[timing] %s: %.3fs", label, elapsed)
    if audio_duration_s > 0:
        rtf = elapsed / audio_duration_s
        logger.info("[RTF]    RTF=%.4f  RTFx=%.2f", rtf, 1 / rtf)
    return result, elapsed


def _split_timestamps_by_gap(time_stamps, silence_gap_s: float) -> list:
    """
    Aggregate word-level time_stamps into sentence segments by silence gap.
    Each ts has .text, .start_time, .end_time (seconds).
    Returns [{"text": ..., "start": ..., "end": ...}, ...].
    When silence_gap_s <= 0, returns a single segment covering all words.
    """
    if not time_stamps:
        return []

    segments = []
    seg_words = [time_stamps[0]]

    for ts in time_stamps[1:]:
        gap = ts.start_time - seg_words[-1].end_time
        if silence_gap_s > 0 and gap >= silence_gap_s:
            segments.append({
                "text":  "".join(w.text for w in seg_words),
                "start": round(seg_words[0].start_time, 3),
                "end":   round(seg_words[-1].end_time, 3),
            })
            seg_words = [ts]
        else:
            seg_words.append(ts)

    if seg_words:
        segments.append({
            "text":  "".join(w.text for w in seg_words),
            "start": round(seg_words[0].start_time, 3),
            "end":   round(seg_words[-1].end_time, 3),
        })

    return [s for s in segments if s["text"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR single-file transcription tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", "-mp", default="./checkpoints/Qwen3-ASR-0.6B", help="ASR model path")
    parser.add_argument("--aligner-path", "-ap", default="./checkpoints/Qwen3-ForcedAligner-0.6B", help="ForcedAligner path")
    parser.add_argument("--input", "-i", required=True, help="Audio file path")
    parser.add_argument("--output", "-o", default=None, help="JSON output path (default: results/<input_basename>.<model_name>.no_vad.<aligner_name>.json)")
    parser.add_argument("--language", "-l", default=None, help='Force language, e.g. "Chinese", "English"')
    parser.add_argument("--timestamps", "-ts", action="store_true", help="Enable word-level timestamps")
    parser.add_argument("--silence-gap", "-sg", type=float, default=0.5, dest="silence_gap",
                        help="Silence gap (s) to split word-level timestamps into segments; 0 = no split")
    parser.add_argument("--device", "-d", default="cuda:0", help='Inference device, e.g. "mps", "cuda:0", "cpu"')
    parser.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE_MAP.keys()), help="Model dtype")
    parser.add_argument("--seperate_channel", "-sc", action="store_true", default=False,
                        help="Split multi-channel audio and transcribe each channel separately")
    return parser.parse_args()


def _build_output(args, audio_path, r, audio_dur_s, model_load_s, transcribe_s, model_name, aligner_name):
    """Build the output dict for a single transcription result."""
    align_s = transcribe_s if args.timestamps else None
    rtf = transcribe_s / audio_dur_s if audio_dur_s > 0 else None
    align_rtf = align_s / audio_dur_s if (align_s and audio_dur_s > 0) else None

    words_list = (
        [{"text": ts.text, "start": ts.start_time, "end": ts.end_time}
         for ts in r.time_stamps]
        if r.time_stamps else []
    )
    segments = _split_timestamps_by_gap(r.time_stamps, args.silence_gap) if r.time_stamps else []

    return {
        "source":        audio_path,
        "filename":      os.path.basename(audio_path),
        "audio_dur_s":   round(audio_dur_s, 3),
        "model_load_s":  round(model_load_s, 3),
        "transcribe_s":  round(transcribe_s, 3),
        "align_s":       round(align_s, 3) if align_s is not None else None,
        "rtf":           round(rtf, 4) if rtf is not None else None,
        "rtfx":          round(1.0 / rtf, 2) if rtf else None,
        "align_rtf":     round(align_rtf, 4) if align_rtf is not None else None,
        "align_rtfx":    round(1.0 / align_rtf, 2) if align_rtf else None,
        "model_name":    model_name,
        "vad_model":     "no_vad",
        "aligner_model": aligner_name,
        "language":      r.language,
        "text":          r.text,
        "segments":      segments,
        "words":         words_list,
    }


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        raise ValueError(f"--input must be a file, got: {args.input!r}")

    basename = os.path.splitext(os.path.basename(args.input))[0]
    model_name = os.path.basename(os.path.normpath(args.model_path))
    aligner_name = os.path.basename(os.path.normpath(args.aligner_path))
    dtype = _DTYPE_MAP[args.dtype]
    logger.info("[config] model=%s  aligner=%s", args.model_path, args.aligner_path)
    logger.info("[config] device=%s  dtype=%s", args.device, args.dtype)
    logger.info("[config] timestamps=%s  silence_gap=%.2fs", args.timestamps, args.silence_gap)
    logger.info("[input]  %s", args.input)

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
    model_load_s = time.perf_counter() - t0
    logger.info("[timing] model loaded: %.3fs", model_load_s)

    if args.seperate_channel:
        audio_data, sample_rate = sf.read(args.input, always_2d=True)
        num_channels = audio_data.shape[1]
        logger.info("[channel] detected %d channel(s), transcribing separately", num_channels)

        with tempfile.TemporaryDirectory() as tmpdir:
            for ch in range(num_channels):
                channel_audio = audio_data[:, ch]
                tmp_wav = os.path.join(tmpdir, f"channel{ch}.wav")
                sf.write(tmp_wav, channel_audio, sample_rate)

                audio_dur_s = len(channel_audio) / sample_rate
                logger.info("[channel %d] transcribing...", ch)
                results, transcribe_s = _timed(
                    f"transcribe channel {ch}",
                    asr.transcribe,
                    audio=tmp_wav,
                    language=args.language,
                    return_time_stamps=args.timestamps,
                    audio_duration_s=audio_dur_s,
                )

                r = results[0]
                output = _build_output(args, args.input, r, audio_dur_s, model_load_s, transcribe_s, model_name, aligner_name)
                output["channel"] = ch

                if args.output:
                    out_base, out_ext = os.path.splitext(args.output)
                    output_path = f"{out_base}.channel{ch}{out_ext}"
                else:
                    output_path = f"results/{basename}.{model_name}.no_vad.channel{ch}.{aligner_name}.json"
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

                logger.info("[result] language=%s  segments=%d  words=%d",
                            output["language"], len(output["segments"]), len(output["words"]))
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                logger.info("[output] saved: %s", output_path)
    else:
        output_path = args.output or f"results/{basename}.{model_name}.no_vad.{aligner_name}.json"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        audio_dur_s = _audio_duration(args.input)
        results, transcribe_s = _timed(
            f"transcribe {os.path.basename(args.input)}",
            asr.transcribe,
            audio=args.input,
            language=args.language,
            return_time_stamps=args.timestamps,
            audio_duration_s=audio_dur_s,
        )

        r = results[0]
        output = _build_output(args, args.input, r, audio_dur_s, model_load_s, transcribe_s, model_name, aligner_name)

        logger.info("[result] language=%s  segments=%d  words=%d",
                    output["language"], len(output["segments"]), len(output["words"]))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("[output] saved: %s", output_path)


if __name__ == "__main__":
    main()
