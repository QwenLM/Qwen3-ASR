# coding=utf-8
"""
Qwen3-ASR batch streaming transcription script (vLLM backend).
Processes all .wav files in a directory with streaming inference and outputs a TSV
and a summary JSON.

Usage:
  python scripts/batch_transcribe_vllm_streaming.py -i <audio_dir> [OPTIONS]

  --model-path/-mp        ASR model path (default: ./checkpoints/Qwen3-ASR-1.7B)
  --input/-i              Audio directory (required)
  --output/-o             TSV output path (default: ./results/batch_asr_results_vllm_streaming.tsv)
                          Summary file is derived automatically as <stem>_summary.json
  --language/-l           Force language, e.g. "Chinese", "English"; auto-detect if not set
  --gpu-memory-util/-gmu  vLLM GPU memory utilization (default: 0.8)
  --max-new-tokens        Max new tokens per streaming chunk (default: 32)
  --chunk-size-sec        Streaming chunk size in seconds (default: 2.0)
  --step-ms               Audio feed step in milliseconds (default: 1000)
  --seperate_channel/-sc  Split multi-channel audio, transcribe each channel separately

Note:
  Requires vLLM extra:
    pip install qwen-asr[vllm]
  Streaming does not support ForcedAligner (no word-level timestamps).
  Files are processed one at a time (streaming is inherently sequential per sample).
"""

import argparse
import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import soundfile as sf

from qwen_asr import Qwen3ASRModel


_TSV_FIELDS = [
    "filename",      # file name (no path)
    "source",        # full audio file path
    "channel",       # channel index (empty when seperate_channel disabled)
    "audio_dur_s",   # audio duration (seconds)
    "model_load_s",  # model load time (seconds)
    "transcribe_s",  # transcription time (seconds)
    "rtf",           # Real-Time Factor = transcribe_s / audio_dur_s
    "rtfx",          # Inverse RTF = audio_dur_s / transcribe_s
    "language",      # detected language
    "text",          # transcription text
    "total_calls",   # number of streaming calls
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StreamingResult:
    source: str
    language: Optional[str]
    text: str
    audio_dur_s: float
    model_load_s: float
    transcribe_s: float
    total_calls: int
    channel: Optional[int] = None

    @property
    def rtf(self) -> Optional[float]:
        if self.audio_dur_s > 0:
            return self.transcribe_s / self.audio_dur_s
        return None

    @property
    def rtfx(self) -> Optional[float]:
        if self.audio_dur_s > 0 and self.transcribe_s > 0:
            return self.audio_dur_s / self.transcribe_s
        return None


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _load_wav_16k(path: str) -> np.ndarray:
    """Load a wav file and resample to 16 kHz if needed."""
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


def _collect_wav_files(directory: str) -> List[str]:
    """Recursively collect all .wav files in directory, sorted by filename."""
    wav_files = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if fname.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, fname))
    return wav_files


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------

def _write_tsv(output_path: str, rows: List[StreamingResult]) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "filename":     os.path.basename(row.source),
                "source":       row.source,
                "channel":      str(row.channel) if row.channel is not None else "",
                "audio_dur_s":  f"{row.audio_dur_s:.3f}" if row.audio_dur_s else "",
                "model_load_s": f"{row.model_load_s:.3f}",
                "transcribe_s": f"{row.transcribe_s:.3f}",
                "rtf":          f"{row.rtf:.4f}" if row.rtf is not None else "",
                "rtfx":         f"{row.rtfx:.2f}" if row.rtfx is not None else "",
                "language":     row.language or "",
                "text":         row.text,
                "total_calls":  str(row.total_calls),
            })
    print(f"[output] TSV written: {output_path}")


def _summary_path(tsv_path: str) -> str:
    base, _ = os.path.splitext(tsv_path)
    return base + "_summary.json"


def _write_summary(output_path: str, rows: List[StreamingResult], args) -> None:
    total_audio = sum(r.audio_dur_s for r in rows)
    total_transcribe = sum(r.transcribe_s for r in rows)

    summary = {
        "model_path":         args.model_path,
        "gpu_memory_util":    args.gpu_memory_util,
        "step_ms":            args.step_ms,
        "chunk_size_sec":     args.chunk_size_sec,
        "total_files":        len(rows),
        "total_audio_dur_s":  round(total_audio, 3),
        "total_transcribe_s": round(total_transcribe, 3),
        "overall_rtf":        round(total_transcribe / total_audio, 4) if total_audio > 0 else None,
        "overall_rtfx":       round(total_audio / total_transcribe, 2) if total_transcribe > 0 else None,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[output] Summary written: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR batch streaming transcription tool (vLLM backend)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", "-mp", default="./checkpoints/Qwen3-ASR-1.7B", help="ASR model path")
    parser.add_argument("--input", "-i", required=True, help="Audio directory")
    parser.add_argument("--output", "-o", default="./results/batch_asr_results_vllm_streaming.tsv", help="TSV output path (summary derived as <stem>_summary.json)")
    parser.add_argument("--language", "-l", default=None, help='Force language, e.g. "Chinese", "English"')
    parser.add_argument("--gpu-memory-util", "-gmu", type=float, default=0.8, dest="gpu_memory_util", help="vLLM GPU memory utilization")
    parser.add_argument("--max-new-tokens", type=int, default=32, dest="max_new_tokens", help="Max new tokens per streaming chunk")
    parser.add_argument("--chunk-size-sec", type=float, default=2.0, dest="chunk_size_sec", help="Streaming chunk size in seconds")
    parser.add_argument("--step-ms", type=int, default=1000, dest="step_ms", help="Audio feed step in milliseconds")
    parser.add_argument("--seperate_channel", "-sc", action="store_true", default=False,
                        help="Split multi-channel audio and transcribe each channel separately")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.input):
        raise ValueError(f"--input must be a directory, got: {args.input!r}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    wav_files = _collect_wav_files(args.input)
    if not wav_files:
        print(f"[warning] No .wav files found in {args.input!r}")
        return

    print(f"[config] model:           {args.model_path}")
    print(f"[config] gpu_memory_util: {args.gpu_memory_util}")
    print(f"[config] chunk_size_sec:  {args.chunk_size_sec}")
    print(f"[config] step_ms:         {args.step_ms}")
    print(f"[input]  {args.input}  ({len(wav_files)} .wav files)")

    t0 = time.perf_counter()
    asr = Qwen3ASRModel.LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_util,
        max_new_tokens=args.max_new_tokens,
    )
    model_load_s = time.perf_counter() - t0
    print(f"[timing] model load: {model_load_s:.3f}s")

    sr = 16000
    step = int(round(args.step_ms / 1000.0 * sr))

    all_rows: List[StreamingResult] = []

    if args.seperate_channel:
        tmp_root = tempfile.mkdtemp()
        expanded: List[tuple] = []  # (tmp_wav, orig_path, channel_idx)
        for wav_path in wav_files:
            audio_data, sample_rate = sf.read(wav_path, always_2d=True)
            num_channels = audio_data.shape[1]
            for ch in range(num_channels):
                tmp_wav = os.path.join(tmp_root, f"{os.path.splitext(os.path.basename(wav_path))[0]}_ch{ch}.wav")
                sf.write(tmp_wav, audio_data[:, ch], sample_rate)
                expanded.append((tmp_wav, wav_path, ch))
    else:
        expanded = [(f, f, None) for f in wav_files]

    try:
        for tmp_wav, orig_path, ch in expanded:
            ch_info = f" (channel {ch})" if ch is not None else ""
            print(f"\n[file] {os.path.basename(orig_path)}{ch_info}")
            wav16k = _load_wav_16k(tmp_wav)
            audio_dur_s = wav16k.shape[0] / 16000.0

            state = asr.init_streaming_state(
                unfixed_chunk_num=2,
                unfixed_token_num=5,
                chunk_size_sec=args.chunk_size_sec,
                language=args.language,
            )

            t1 = time.perf_counter()
            pos = 0
            call_id = 0
            while pos < wav16k.shape[0]:
                seg = wav16k[pos: pos + step]
                pos += seg.shape[0]
                call_id += 1
                asr.streaming_transcribe(seg, state)
                print(f"[call {call_id:03d}] language={state.language!r} text={state.text!r}")

            asr.finish_streaming_transcribe(state)
            transcribe_s = time.perf_counter() - t1

            print(f"[final]  language={state.language!r} text={state.text!r}")
            print(f"[timing] transcribe: {transcribe_s:.3f}s")
            if audio_dur_s > 0:
                rtf = transcribe_s / audio_dur_s
                rtfx = audio_dur_s / transcribe_s
                print(f"[RTF]    RTF={rtf:.4f}  RTFx={rtfx:.2f}x")

            all_rows.append(StreamingResult(
                source=orig_path,
                language=state.language,
                text=state.text,
                audio_dur_s=audio_dur_s,
                model_load_s=model_load_s,
                transcribe_s=transcribe_s,
                total_calls=call_id,
                channel=ch,
            ))
    finally:
        if args.seperate_channel:
            import shutil
            shutil.rmtree(tmp_root, ignore_errors=True)

    print()
    _write_tsv(args.output, all_rows)
    _write_summary(_summary_path(args.output), all_rows, args)


if __name__ == "__main__":
    main()
