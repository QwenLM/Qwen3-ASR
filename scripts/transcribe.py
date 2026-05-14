# coding=utf-8
"""
Qwen3-ASR single-file transcription script (Transformers backend, macOS/MPS).
Transcribes a single audio file and outputs a JSON with full timestamps and timing stats.

Usage:
  python scripts/transcribe.py -i <audio_file> [OPTIONS]

  --model-path/-mp      ASR model path (default: ./checkpoints/Qwen3-ASR-0.6B)
  --aligner-path/-ap    ForcedAligner path (default: ./checkpoints/Qwen3-ForcedAligner-0.6B)
  --input/-i            Audio file path (required)
  --output/-o           JSON output path (default: results/<input_basename>-asr_result.json)
  --language/-l         Force language, e.g. "Chinese", "English"; auto-detect if not set
  --timestamps/-ts      Enable word-level timestamps
  --device/-d           Inference device, e.g. "mps", "cuda:0", "cpu" (cuda:0)
  --dtype               Model dtype: bfloat16 / float16 / float32 (default: bfloat16)
  --seperate_channel/-sc  Split multi-channel audio and transcribe each channel separately
"""

import argparse
import json
import os
import tempfile
import time

import soundfile as sf
import torch

from qwen_asr import Qwen3ASRModel


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def _audio_duration(path: str) -> float:
    return sf.info(path).duration


def _timed(label: str, fn, *args, audio_duration_s: float = 0.0, **kwargs):
    """Run fn and return (result, elapsed_s), printing elapsed time and RTF."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"[timing] {label}: {elapsed:.3f}s")
    if audio_duration_s > 0:
        rtf = elapsed / audio_duration_s
        print(f"[RTF]    RTF={rtf:.4f}, which means it can transcribe {1/rtf:.2f} seconds audio in 1 second")
    return result, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR single-file transcription tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", "-mp", default="./checkpoints/Qwen3-ASR-0.6B", help="ASR model path")
    parser.add_argument("--aligner-path", "-ap", default="./checkpoints/Qwen3-ForcedAligner-0.6B", help="ForcedAligner path")
    parser.add_argument("--input", "-i", required=True, help="Audio file path")
    parser.add_argument("--output", "-o", default=None, help="JSON output path (default: results/<input_basename>-asr_result.json)")
    parser.add_argument("--language", "-l", default=None, help='Force language, e.g. "Chinese", "English"')
    parser.add_argument("--timestamps", "-ts", action="store_true", help="Enable word-level timestamps")
    parser.add_argument("--device", "-d", default="cuda:0", help='Inference device, e.g. "mps", "cuda:0", "cpu"')
    parser.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE_MAP.keys()), help="Model dtype")
    parser.add_argument("--seperate_channel", "-sc", action="store_true", default=False,
                        help="Split multi-channel audio and transcribe each channel separately")
    return parser.parse_args()


def _build_output(args, audio_path, r, audio_dur_s, model_load_s, transcribe_s):
    """Build the output dict for a single transcription result."""
    align_s = transcribe_s if args.timestamps else None
    rtf = transcribe_s / audio_dur_s if audio_dur_s > 0 else None
    align_rtf = align_s / audio_dur_s if (align_s and audio_dur_s > 0) else None
    return {
        "source":        audio_path,
        "filename":      os.path.basename(audio_path),
        "language":      r.language,
        "text":          r.text,
        "audio_dur_s":   round(audio_dur_s, 3),
        "model_load_s":  round(model_load_s, 3),
        "transcribe_s":  round(transcribe_s, 3),
        "align_s":       round(align_s, 3) if align_s is not None else None,
        "rtf":           round(rtf, 4) if rtf is not None else None,
        "align_rtf":     round(align_rtf, 4) if align_rtf is not None else None,
        "time_stamps":   (
            [{"text": ts.text, "start": ts.start_time, "end": ts.end_time}
             for ts in r.time_stamps]
            if r.time_stamps else []
        ),
    }


def _print_result(output):
    print(f"\n[result] language={output['language']!r}")
    print(f"[result] text={output['text']!r}")
    if output["time_stamps"]:
        head = output["time_stamps"][0]
        tail = output["time_stamps"][-1]
        print(f"[result] ts_first: {head['text']!r} {head['start']}->{head['end']} s")
        print(f"[result] ts_last : {tail['text']!r} {tail['start']}->{tail['end']} s")


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        raise ValueError(f"--input must be a file, got: {args.input!r}")

    basename = os.path.splitext(os.path.basename(args.input))[0]
    dtype = _DTYPE_MAP[args.dtype]
    print(f"[config] model:   {args.model_path}")
    print(f"[config] aligner: {args.aligner_path}")
    print(f"[config] device:  {args.device}  dtype: {args.dtype}")
    print(f"[input]  {args.input}")

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
    print(f"[timing] model load: {model_load_s:.3f}s")

    if args.seperate_channel:
        # Read all channels from the audio file
        audio_data, sample_rate = sf.read(args.input, always_2d=True)
        num_channels = audio_data.shape[1]
        print(f"[channel] detected {num_channels} channel(s), transcribing separately")

        with tempfile.TemporaryDirectory() as tmpdir:
            for ch in range(num_channels):
                channel_audio = audio_data[:, ch]
                tmp_wav = os.path.join(tmpdir, f"channel{ch}.wav")
                sf.write(tmp_wav, channel_audio, sample_rate)

                audio_dur_s = len(channel_audio) / sample_rate
                print(f"\n[channel] processing channel {ch} ...")
                results, transcribe_s = _timed(
                    f"transcribe channel {ch}",
                    asr.transcribe,
                    audio=tmp_wav,
                    language=args.language,
                    return_time_stamps=args.timestamps,
                    audio_duration_s=audio_dur_s,
                )

                r = results[0]
                output = _build_output(args, args.input, r, audio_dur_s, model_load_s, transcribe_s)
                output["channel"] = ch

                if args.output:
                    # Insert _channel{ch} before the last extension
                    out_base, out_ext = os.path.splitext(args.output)
                    output_path = f"{out_base}_channel{ch}{out_ext}"
                else:
                    output_path = f"results/{basename}_channel{ch}-asr_result.json"
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

                _print_result(output)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                print(f"[output] JSON written: {output_path}")
    else:
        output_path = args.output or f"results/{basename}-asr_result.json"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        audio_dur_s = _audio_duration(args.input)
        results, transcribe_s = _timed(
            "transcribe",
            asr.transcribe,
            audio=args.input,
            language=args.language,
            return_time_stamps=args.timestamps,
            audio_duration_s=audio_dur_s,
        )

        r = results[0]
        output = _build_output(args, args.input, r, audio_dur_s, model_load_s, transcribe_s)

        _print_result(output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[output] JSON written: {output_path}")


if __name__ == "__main__":
    main()
