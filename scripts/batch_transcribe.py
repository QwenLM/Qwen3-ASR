# coding=utf-8
"""
Qwen3-ASR batch transcription script (Transformers backend, macOS/MPS).
Processes all .wav files in a directory and outputs a TSV and a summary JSON.

Usage:
  python scripts/batch_transcribe.py -i <audio_dir> [OPTIONS]

  --model-path/-mp   ASR model path (default: ./checkpoints/Qwen3-ASR-0.6B)
  --aligner-path/-ap ForcedAligner path (default: ./checkpoints/Qwen3-ForcedAligner-0.6B)
  --input/-i         Audio directory (required)
  --output/-o        TSV output path (default: ./results/batch_asr_results.tsv)
                     Summary file is derived automatically as <stem>_summary.json
  --language/-l      Force language, e.g. "Chinese", "English"; auto-detect if not set
  --word-timestamps/-wts  Enable word-level timestamps
  --device/-d        Inference device, e.g. "mps", "cuda:0", "cpu" (cuda:0)
  --dtype            Model dtype: bfloat16 / float16 / float32 (default: bfloat16)
  --batch-size/-bs   Inference batch size (default: 1)
  --max-new-tokens   Max new tokens for generation (default: 1024)
  --seperate_channel/-sc  Split multi-channel audio, transcribe each channel separately
"""

import argparse
import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

import soundfile as sf
import torch

from qwen_asr import Qwen3ASRModel


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}

_TSV_FIELDS = [
    "filename",      # file name (no path)
    "source",        # full audio file path
    "channel",       # channel index (empty when seperate_channel disabled)
    "audio_dur_s",   # audio duration (seconds)
    "model_load_s",  # model load time (seconds)
    "transcribe_s",  # transcription time (seconds)
    "align_s",       # alignment time (seconds; empty when timestamps disabled)
    "rtf",           # Real-Time Factor = transcribe_s / audio_dur_s
    "rtfx",          # Inverse RTF = audio_dur_s / transcribe_s
    "align_rtf",     # Align RTF = align_s / audio_dur_s (empty when timestamps disabled)
    "align_rtfx",    # Inverse Align RTF = audio_dur_s / align_s (empty when timestamps disabled)
    "model_name",    # ASR model name
    "vad_model",     # VAD model name (no_vad when VAD disabled)
    "aligner_model", # ForcedAligner model name
    "language",      # detected language
    "text",          # transcription text
    "words",         # word-level timestamps as JSON (empty when timestamps disabled)
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TimedResult:
    source: str
    language: Optional[str]
    text: str
    audio_dur_s: float
    model_load_s: float
    transcribe_s: float
    align_s: Optional[float] = None
    words: list = field(default_factory=list)
    channel: Optional[int] = None
    model_name: str = ""
    vad_model: str = "no_vad"
    aligner_model: str = ""

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

    @property
    def align_rtf(self) -> Optional[float]:
        if self.align_s is not None and self.audio_dur_s > 0:
            return self.align_s / self.audio_dur_s
        return None

    @property
    def align_rtfx(self) -> Optional[float]:
        if self.align_s is not None and self.audio_dur_s > 0 and self.align_s > 0:
            return self.audio_dur_s / self.align_s
        return None


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _audio_duration(path: str) -> float:
    return sf.info(path).duration


def _collect_wav_files(directory: str) -> List[str]:
    """Recursively collect all .wav files in directory, sorted by filename."""
    wav_files = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if fname.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, fname))
    return wav_files


# ---------------------------------------------------------------------------
# Timed inference
# ---------------------------------------------------------------------------

def _timed(label: str, fn, *args, audio_duration_s: float = 0.0, **kwargs):
    """Run fn and return (result, elapsed_s), printing elapsed time and RTF."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"[timing] {label}: {elapsed:.3f}s")
    if audio_duration_s > 0:
        rtf = elapsed / audio_duration_s
        rtfx = audio_duration_s / elapsed
        print(f"[RTF]    RTF={rtf:.4f}  RTFx={rtfx:.2f}x")
    return result, elapsed


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------

def _print_asr_results(title: str, results) -> None:
    print(f"\n===== {title} =====")
    for i, r in enumerate(results):
        print(f"[sample {i}] language={r.language!r}")
        print(f"[sample {i}] text={r.text!r}")
        if r.time_stamps is not None and len(r.time_stamps) > 0:
            head = r.time_stamps[0]
            tail = r.time_stamps[-1]
            print(f"[sample {i}] ts_first: {head.text!r} {head.start_time}->{head.end_time} s")
            print(f"[sample {i}] ts_last : {tail.text!r} {tail.start_time}->{tail.end_time} s")


def _write_tsv(output_path: str, rows: List[TimedResult]) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            ts_json = (
                json.dumps(
                    [{"text": ts.text, "start": ts.start_time, "end": ts.end_time}
                     for ts in row.words],
                    ensure_ascii=False,
                )
                if row.words else ""
            )
            writer.writerow({
                "filename":     os.path.basename(row.source),
                "source":       row.source,
                "channel":      str(row.channel) if row.channel is not None else "",
                "audio_dur_s":  f"{row.audio_dur_s:.3f}" if row.audio_dur_s else "",
                "model_load_s": f"{row.model_load_s:.3f}",
                "transcribe_s": f"{row.transcribe_s:.3f}",
                "align_s":      f"{row.align_s:.3f}" if row.align_s is not None else "",
                "rtf":          f"{row.rtf:.4f}" if row.rtf is not None else "",
                "rtfx":         f"{row.rtfx:.2f}" if row.rtfx is not None else "",
                "align_rtf":    f"{row.align_rtf:.4f}" if row.align_rtf is not None else "",
                "align_rtfx":   f"{row.align_rtfx:.2f}" if row.align_rtfx is not None else "",
                "model_name":   row.model_name,
                "vad_model":    row.vad_model,
                "aligner_model": row.aligner_model,
                "language":     row.language or "",
                "text":         row.text,
                "words":        ts_json,
            })
    print(f"[output] TSV written: {output_path}")


def _summary_path(tsv_path: str) -> str:
    base, _ = os.path.splitext(tsv_path)
    return base + "_summary.json"


def _write_summary(output_path: str, rows: List[TimedResult], args) -> None:
    total_audio = sum(r.audio_dur_s for r in rows)
    total_transcribe = sum(r.transcribe_s for r in rows)
    total_align = sum(r.align_s for r in rows if r.align_s is not None) or None

    summary = {
        "model_path":         args.model_path,
        "aligner_path":       args.aligner_path,
        "device":             args.device,
        "dtype":              args.dtype,
        "batch_size":         args.batch_size,
        "word_timestamps":    args.word_timestamps,
        "total_files":        len(rows),
        "total_audio_dur_s":  round(total_audio, 3),
        "total_transcribe_s": round(total_transcribe, 3),
        "overall_rtf":        round(total_transcribe / total_audio, 4) if total_audio > 0 else None,
        "overall_rtfx":       round(total_audio / total_transcribe, 2) if total_transcribe > 0 else None,
        "total_align_s":      round(total_align, 3) if total_align is not None else None,
        "overall_align_rtf":  round(total_align / total_audio, 4) if total_align and total_audio > 0 else None,
        "overall_align_rtfx": round(total_audio / total_align, 2) if total_align and total_audio > 0 else None,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[output] Summary written: {output_path}")


def _make_timed_results(
    asr_results,
    sources: List[str],
    audio_durs: List[float],
    model_load_s: float,
    transcribe_s: float,
    align_s: Optional[float] = None,
) -> List[TimedResult]:
    return [
        TimedResult(
            source=sources[i] if i < len(sources) else "",
            language=r.language,
            text=r.text,
            audio_dur_s=audio_durs[i] if i < len(audio_durs) else 0.0,
            model_load_s=model_load_s,
            transcribe_s=transcribe_s,
            align_s=align_s,
            words=r.time_stamps or [],
        )
        for i, r in enumerate(asr_results)
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR batch transcription tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", "-mp", default="./checkpoints/Qwen3-ASR-0.6B", help="ASR model path")
    parser.add_argument("--aligner-path", "-ap", default="./checkpoints/Qwen3-ForcedAligner-0.6B", help="ForcedAligner path")
    parser.add_argument("--input", "-i", required=True, help="Audio directory")
    parser.add_argument("--output", "-o", default="./results/batch_asr_results.tsv", help="TSV output path (summary derived as <stem>_summary.json)")
    parser.add_argument("--language", "-l", default=None, help='Force language, e.g. "Chinese", "English"')
    parser.add_argument("--word-timestamps", "-wts", action="store_true", dest="word_timestamps", help="Enable word-level timestamps")
    parser.add_argument("--device", "-d", default="cuda:0", help='Inference device, e.g. "mps", "cuda:0", "cpu"')
    parser.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE_MAP.keys()), help="Model dtype")
    parser.add_argument("--batch-size", "-bs", type=int, default=1, dest="batch_size", help="Inference batch size")
    parser.add_argument("--max-new-tokens", type=int, default=1024, dest="max_new_tokens", help="Max new tokens for generation")
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

    dtype = _DTYPE_MAP[args.dtype]
    model_name = os.path.basename(os.path.normpath(args.model_path))
    aligner_name = os.path.basename(os.path.normpath(args.aligner_path))
    print(f"[config] model:      {args.model_path}")
    print(f"[config] aligner:    {args.aligner_path}")
    print(f"[config] device:     {args.device}  dtype: {args.dtype}")
    print(f"[config] batch_size: {args.batch_size}")
    print(f"[input]  {args.input}  ({len(wav_files)} .wav files)")

    t0 = time.perf_counter()
    asr = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=args.device,
        forced_aligner=args.aligner_path,
        forced_aligner_kwargs=dict(dtype=dtype, device_map=args.device),
        max_inference_batch_size=32,
        max_new_tokens=args.max_new_tokens,
    )
    model_load_s = time.perf_counter() - t0
    print(f"[timing] model load: {model_load_s:.3f}s")

    all_rows: List[TimedResult] = []

    if args.seperate_channel:
        # Expand each wav file into per-channel temp files
        tmp_root = tempfile.mkdtemp()
        expanded: List[tuple] = []  # (tmp_wav_path, original_path, channel_idx)
        for wav_path in wav_files:
            audio_data, sample_rate = sf.read(wav_path, always_2d=True)
            num_channels = audio_data.shape[1]
            for ch in range(num_channels):
                tmp_wav = os.path.join(tmp_root, f"{os.path.splitext(os.path.basename(wav_path))[0]}_ch{ch}.wav")
                sf.write(tmp_wav, audio_data[:, ch], sample_rate)
                expanded.append((tmp_wav, wav_path, ch))
        tmp_files = [e[0] for e in expanded]
    else:
        expanded = [(f, f, None) for f in wav_files]
        tmp_files = wav_files

    try:
        for batch_start in range(0, len(tmp_files), args.batch_size):
            batch_entries = expanded[batch_start: batch_start + args.batch_size]
            batch_tmp = [e[0] for e in batch_entries]
            batch_orig = [e[1] for e in batch_entries]
            batch_chs = [e[2] for e in batch_entries]
            durs = [_audio_duration(f) for f in batch_tmp]
            label = f"batch [{batch_start}~{batch_start + len(batch_entries) - 1}]"
            ch_info = f" channels={batch_chs}" if args.seperate_channel else ""
            print(f"\n[batch] {label}: {[os.path.basename(f) for f in batch_orig]}{ch_info}")

            results, elapsed = _timed(
                label,
                asr.transcribe,
                audio=batch_tmp,
                language=args.language,
                return_time_stamps=args.word_timestamps,
                audio_duration_s=sum(durs),
            )
            _print_asr_results(label, results)
            align_s = elapsed if args.word_timestamps else None
            timed = _make_timed_results(results, batch_orig, durs, model_load_s, elapsed, align_s)
            for tr, ch in zip(timed, batch_chs):
                tr.channel = ch
                tr.model_name = model_name
                tr.vad_model = "no_vad"
                tr.aligner_model = aligner_name
            all_rows += timed
    finally:
        if args.seperate_channel:
            import shutil
            shutil.rmtree(tmp_root, ignore_errors=True)

    print()
    _write_tsv(args.output, all_rows)
    _write_summary(_summary_path(args.output), all_rows, args)


if __name__ == "__main__":
    main()
