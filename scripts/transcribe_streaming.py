# coding=utf-8
"""
Qwen3-ASR single-file streaming transcription script (Transformers backend).
Reads a local audio file and performs streaming inference, printing partial results
as audio chunks are fed in. Final result is saved as JSON.

Usage:
  python scripts/transcribe_streaming.py -i <audio_file> [OPTIONS]

  --model-path/-mp        ASR model path (default: ./checkpoints/Qwen3-ASR-0.6B)
  --input/-i              Audio file path (required)
  --output/-o             JSON output path (default: results/<input_basename>.<model_name>.no_vad.no_aligner.json)
  --language/-l           Force language, e.g. "Chinese", "English"; auto-detect if not set
  --device/-d             Inference device, e.g. "mps", "cuda:0", "cpu" (default: cuda:0)
  --dtype                 Model dtype: bfloat16 / float16 / float32 (default: bfloat16)
  --max-new-tokens        Max new tokens per streaming chunk (default: 32)
  --chunk-size-sec        Streaming chunk size in seconds (default: 2.0)
  --step-ms               Audio feed step in milliseconds (default: 1000)
  --seperate_channel/-sc  Split multi-channel audio and transcribe each channel separately

Output format (json):
  {
    "source": "...",
    "filename": "...",
    "audio_dur_s": 12.34,
    "model_load_s": 1.0,
    "transcribe_s": 1.23,
    "rtf": 0.1,
    "rtfx": 10.0,
    "model_name": "Qwen3-ASR-0.6B",
    "vad_model": "no_vad",
    "aligner_model": "no_aligner",
    "align_rtf": null,
    "align_rtfx": null,
    "language": "Chinese",
    "text": "full transcription text",
    "step_ms": 1000,
    "chunk_size_sec": 2.0,
    "total_calls": 12,
    "chunks": [
      {"call": 1, "is_final": false, "text": "partial result"},
      ...
    ]
  }

Note:
  Streaming does not support ForcedAligner (no word-level timestamps).
  align_rtf and align_rtfx are always null.
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


def _load_wav_16k(path: str) -> np.ndarray:
    """Load an audio file and resample to 16 kHz if needed (mono)."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if sr == 16000:
        return wav
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * 16000))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR single-file streaming transcription tool (Transformers backend)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", "-mp", default="./checkpoints/Qwen3-ASR-0.6B", help="ASR model path")
    parser.add_argument("--input", "-i", required=True, help="Audio file path")
    parser.add_argument("--output", "-o", default=None, help="JSON output path (default: results/<input_basename>.<model_name>.no_vad.no_aligner.json)")
    parser.add_argument("--language", "-l", default=None, help='Force language, e.g. "Chinese", "English"')
    parser.add_argument("--device", "-d", default="cuda:0", help='Inference device, e.g. "mps", "cuda:0", "cpu"')
    parser.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE_MAP.keys()), help="Model dtype")
    parser.add_argument("--max-new-tokens", type=int, default=32, dest="max_new_tokens", help="Max new tokens per streaming chunk")
    parser.add_argument("--chunk-size-sec", type=float, default=2.0, dest="chunk_size_sec", help="Streaming chunk size in seconds")
    parser.add_argument("--step-ms", type=int, default=1000, dest="step_ms", help="Audio feed step in milliseconds")
    parser.add_argument("--seperate_channel", "-sc", action="store_true", default=False,
                        help="Split multi-channel audio and transcribe each channel separately")
    return parser.parse_args()


def _stream_transcribe(asr, wav16k, args, model_name, model_load_s, audio_path, output_path):
    """Run streaming transcription on a 1-D 16kHz numpy array and write JSON output."""
    sr = 16000
    step = int(round(args.step_ms / 1000.0 * sr))
    audio_dur_s = wav16k.shape[0] / float(sr)

    state = asr.init_streaming_state(
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        chunk_size_sec=args.chunk_size_sec,
        language=args.language,
    )

    t1 = time.perf_counter()
    pos = 0
    call_id = 0
    chunks_out = []
    while pos < wav16k.shape[0]:
        seg = wav16k[pos: pos + step]
        pos += seg.shape[0]
        is_final = (pos >= wav16k.shape[0])
        call_id += 1
        asr.streaming_transcribe(seg, state)
        text = state.text or ""
        chunks_out.append({"call": call_id, "is_final": is_final, "text": text})
        print(f"[call {call_id:03d}] language={state.language!r} text={text!r}")

    asr.finish_streaming_transcribe(state)
    transcribe_s = time.perf_counter() - t1

    print(f"[final]  language={state.language!r} text={state.text!r}")
    print(f"[timing] transcribe: {transcribe_s:.3f}s")
    rtf = transcribe_s / audio_dur_s if audio_dur_s > 0 else None
    rtfx = audio_dur_s / transcribe_s if audio_dur_s > 0 else None
    if rtf is not None:
        print(f"[RTF]    RTF={rtf:.4f}  RTFx={rtfx:.2f}x")

    output = {
        "source":         audio_path,
        "filename":       os.path.basename(audio_path),
        "audio_dur_s":    round(audio_dur_s, 3),
        "model_load_s":   round(model_load_s, 3),
        "transcribe_s":   round(transcribe_s, 3),
        "rtf":            round(rtf, 4) if rtf is not None else None,
        "rtfx":           round(rtfx, 2) if rtfx is not None else None,
        "model_name":     model_name,
        "vad_model":      "no_vad",
        "aligner_model":  "no_aligner",
        "align_rtf":      None,
        "align_rtfx":     None,
        "language":       state.language,
        "text":           state.text,
        "step_ms":        args.step_ms,
        "chunk_size_sec": args.chunk_size_sec,
        "total_calls":    call_id,
        "chunks":         chunks_out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[output] JSON written: {output_path}")


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        raise ValueError(f"--input must be a file, got: {args.input!r}")

    basename = os.path.splitext(os.path.basename(args.input))[0]
    model_name = os.path.basename(os.path.normpath(args.model_path))
    dtype = _DTYPE_MAP[args.dtype]

    print(f"[config] model:          {args.model_path}")
    print(f"[config] device:         {args.device}  dtype: {args.dtype}")
    print(f"[config] chunk_size_sec: {args.chunk_size_sec}")
    print(f"[config] step_ms:        {args.step_ms}")
    print(f"[input]  {args.input}")

    t0 = time.perf_counter()
    asr = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=args.device,
        max_inference_batch_size=32,
        max_new_tokens=args.max_new_tokens,
    )
    model_load_s = time.perf_counter() - t0
    print(f"[timing] model load: {model_load_s:.3f}s")

    if args.seperate_channel:
        audio_data, sample_rate = sf.read(args.input, always_2d=True)
        num_channels = audio_data.shape[1]
        print(f"[channel] detected {num_channels} channel(s), transcribing separately")

        with tempfile.TemporaryDirectory() as tmpdir:
            for ch in range(num_channels):
                channel_audio = audio_data[:, ch]
                tmp_wav = os.path.join(tmpdir, f"channel{ch}.wav")
                sf.write(tmp_wav, channel_audio, sample_rate)

                print(f"\n[channel] processing channel {ch} ...")
                wav16k = _load_wav_16k(tmp_wav)

                if args.output:
                    out_base, out_ext = os.path.splitext(args.output)
                    output_path = f"{out_base}.channel{ch}{out_ext}"
                else:
                    output_path = f"results/{basename}.{model_name}.no_vad.channel{ch}.no_aligner.json"

                _stream_transcribe(asr, wav16k, args, model_name, model_load_s, args.input, output_path)
    else:
        output_path = args.output or f"results/{basename}.{model_name}.no_vad.no_aligner.json"
        wav16k = _load_wav_16k(args.input)
        _stream_transcribe(asr, wav16k, args, model_name, model_load_s, args.input, output_path)


if __name__ == "__main__":
    main()
