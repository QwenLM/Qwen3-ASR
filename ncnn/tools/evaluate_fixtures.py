#!/usr/bin/env python3
"""Evaluate Qwen3-ASR fixture outputs.

This script is intentionally lightweight: it consumes result JSON files already
produced by qwen3_asr_main and optional PyTorch baseline JSON files. It does not
run PyTorch or ncnn by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


CJK_PUNCT = "，。！？、；：（）《》“”‘’【】"
ASCII_PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")
SPACE_RE = re.compile(r"\s+")
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>|<[^>]+>")


def load_json(path: str | None) -> Any | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def strip_qwen3_asr_prefix(text: str) -> str:
    text = text.strip()
    if text.startswith("language "):
        # Common raw form: "language Chinese你好。" or "language EnglishHello."
        text = re.sub(r"^language\s+(Chinese|English|Japanese|Korean|Cantonese)", "", text).strip()
    return text


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = SPECIAL_TOKEN_RE.sub("", text)
    text = strip_qwen3_asr_prefix(text)
    text = text.replace("\u3000", " ")
    text = SPACE_RE.sub(" ", text).strip()
    return text


def normalized_text(text: str) -> str:
    text = clean_text(text).lower()
    for ch in CJK_PUNCT:
        text = text.replace(ch, "")
    text = ASCII_PUNCT_RE.sub("", text)
    text = SPACE_RE.sub("", text)
    return text


def semantic_normalized_text(text: str) -> str:
    text = normalized_text(text)
    # Keep this list intentionally small and explicit. It documents common ASR
    # spellings for English abbreviations in mixed Chinese/English speech, but
    # strict normalized_text still remains the output-contract gate.
    replacements = {
        "openiappi": "openaiapi",
        "openiap": "openaiapi",
        "openiaip": "openaiapi",
        "openaiappi": "openaiapi",
        "apiapi": "api",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def result_texts(result: dict[str, Any] | None) -> dict[str, str]:
    if not result:
        return {"raw": "", "clean": "", "normalized": "", "semantic_normalized": ""}
    raw = str(result.get("raw_text") or result.get("text") or "")
    text = str(result.get("text") or raw)
    return {
        "raw": raw,
        "clean": clean_text(text),
        "normalized": normalized_text(text),
        "semantic_normalized": semantic_normalized_text(text),
    }


def expected_texts(text: str) -> dict[str, str]:
    return {
        "raw": text,
        "clean": clean_text(text),
        "normalized": normalized_text(text),
        "semantic_normalized": semantic_normalized_text(text),
    }


def mel_summary_first(summary: Any | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    if isinstance(summary, list):
        return summary[0] if summary else None
    if isinstance(summary, dict):
        return summary
    return None


def compare_mel(ncnn_mel: Any | None, pytorch_mel: Any | None) -> dict[str, Any]:
    ncnn = mel_summary_first(ncnn_mel)
    pytorch = mel_summary_first(pytorch_mel)
    out: dict[str, Any] = {
        "ncnn_present": ncnn is not None,
        "pytorch_present": pytorch is not None,
        "status": "missing_pytorch" if ncnn is not None and pytorch is None else "missing"
    }
    if ncnn is None:
        return out
    out.update({
        "sample_rate": ncnn.get("sample_rate"),
        "shape": [ncnn.get("h"), ncnn.get("w")],
        "dtype": ncnn.get("dtype"),
        "mean": ncnn.get("mean"),
        "std": ncnn.get("std"),
        "min": ncnn.get("min"),
        "max": ncnn.get("max"),
        "first_values": ncnn.get("first_values", [])[:6],
    })
    if pytorch is None:
        return out

    checks = []
    for key in ("sample_rate", "dtype", "h", "w", "mel_bins", "frames", "n_fft", "hop_length"):
        checks.append(ncnn.get(key) == pytorch.get(key))
    out["status"] = "ok" if all(checks) else "mismatch"
    out["pytorch_shape"] = [pytorch.get("h"), pytorch.get("w")]
    for key in ("mean", "std", "min", "max"):
        nv = ncnn.get(key)
        pv = pytorch.get(key)
        if isinstance(nv, (int, float)) and isinstance(pv, (int, float)):
            out[f"{key}_abs_diff"] = abs(float(nv) - float(pv))
    return out


def module_status(result: dict[str, Any] | None, module_summary: Any | None) -> dict[str, Any]:
    out = {
        "speech_encoder_summary": "missing",
        "projector_or_adaptor_summary": "missing",
        "first_decoder_logits_summary": "missing",
        "module_summary_file": module_summary is not None,
        "pytorch_speech_encoder_summary": "missing",
        "pytorch_projector_or_adaptor_summary": "missing",
        "pytorch_first_decoder_logits_summary": "missing",
    }
    if isinstance(result, dict):
        chunks = result.get("chunks", [])
        if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
            if "audio_embedding" in chunks[0]:
                out["speech_encoder_summary"] = "ncnn_audio_embedding_present"
            first_step = chunks[0].get("first_step")
            if isinstance(first_step, dict) and first_step.get("available"):
                if "merged_embeds" in first_step:
                    out["projector_or_adaptor_summary"] = "ncnn_merged_embedding_present"
                if "selected_logits" in first_step:
                    out["first_decoder_logits_summary"] = "ncnn_selected_logits_present"
        if "audio_embedding" in result:
            out["speech_encoder_summary"] = "ncnn_audio_embedding_present"
        first_step = result.get("first_step")
        if isinstance(first_step, dict) and first_step.get("available"):
            if "merged_embeds" in first_step:
                out["projector_or_adaptor_summary"] = "ncnn_merged_embedding_present"
            if "selected_logits" in first_step:
                out["first_decoder_logits_summary"] = "ncnn_selected_logits_present"
        if "lm_head" in result:
            out["first_decoder_logits_summary"] = "ncnn_lm_head_smoke_present"
    if isinstance(module_summary, dict):
        chunks = module_summary.get("chunks", [])
        first_chunk = chunks[0] if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict) else {}
        if "audio_embedding" in first_chunk:
            out["pytorch_speech_encoder_summary"] = "present"
        first_step = first_chunk.get("first_step")
        if isinstance(first_step, dict) and first_step.get("available"):
            if "merged_embeds" in first_step:
                out["pytorch_projector_or_adaptor_summary"] = "present"
            if "selected_logits" in first_step:
                out["pytorch_first_decoder_logits_summary"] = "present"
        for key in ("speech_encoder_summary", "projector_or_adaptor_summary", "first_decoder_logits_summary"):
            if key in module_summary:
                out[key] = "present"
    return out


def first_chunk(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    chunks = value.get("chunks", [])
    if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
        return chunks[0]
    return value


def summary_shape(summary: dict[str, Any] | None) -> list[int | None]:
    if not isinstance(summary, dict):
        return []
    if isinstance(summary.get("shape"), list):
        return [int(x) for x in summary["shape"]]
    if summary.get("dims") == 1 and summary.get("w") is not None:
        return [int(summary["w"])]
    if summary.get("h") is not None and summary.get("w") is not None:
        return [int(summary["h"]), int(summary["w"])]
    if summary.get("w") is not None:
        return [int(summary["w"])]
    return []


def compare_summary_stats(ncnn: dict[str, Any] | None, pytorch: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ncnn_shape": summary_shape(ncnn),
        "pytorch_shape": summary_shape(pytorch),
    }
    out["shape_match"] = bool(out["ncnn_shape"]) and out["ncnn_shape"] == out["pytorch_shape"]
    for key in ("mean", "std", "min", "max"):
        nv = ncnn.get(key) if isinstance(ncnn, dict) else None
        pv = pytorch.get(key) if isinstance(pytorch, dict) else None
        if isinstance(nv, (int, float)) and isinstance(pv, (int, float)):
            out[f"{key}_abs_diff"] = abs(float(nv) - float(pv))
    n_first = ncnn.get("first_values", []) if isinstance(ncnn, dict) else []
    p_first = pytorch.get("first_values", []) if isinstance(pytorch, dict) else []
    if isinstance(n_first, list) and isinstance(p_first, list) and n_first and p_first:
        count = min(len(n_first), len(p_first))
        diffs = [abs(float(n_first[i]) - float(p_first[i])) for i in range(count)]
        out["first_values_count"] = count
        out["first_values_max_abs_diff"] = max(diffs)
        out["first_values_mean_abs_diff"] = sum(diffs) / count
    return out


def module_compare(result: dict[str, Any] | None, module_summary: Any | None) -> dict[str, Any]:
    ncnn_chunk = first_chunk(result)
    pytorch_chunk = first_chunk(module_summary)
    ncnn_first = ncnn_chunk.get("first_step") if isinstance(ncnn_chunk, dict) else None
    pytorch_first = pytorch_chunk.get("first_step") if isinstance(pytorch_chunk, dict) else None
    out: dict[str, Any] = {
        "available": isinstance(ncnn_first, dict) and isinstance(pytorch_first, dict),
        "next_token_match": None,
        "prompt_len_match": None,
        "summaries": {},
    }
    if not out["available"]:
        return out
    out["next_token_match"] = ncnn_first.get("next_token") == pytorch_first.get("next_token")
    out["prompt_len_match"] = ncnn_first.get("prompt_len") == pytorch_first.get("prompt_len")
    out["audio_embedding"] = compare_summary_stats(
        ncnn_chunk.get("audio_embedding") if isinstance(ncnn_chunk, dict) else None,
        pytorch_chunk.get("audio_embedding") if isinstance(pytorch_chunk, dict) else None,
    )
    for key in ("text_embeds", "merged_embeds", "hidden", "logits", "selected_logits"):
        out["summaries"][key] = compare_summary_stats(ncnn_first.get(key), pytorch_first.get(key))
    shape_checks = [out["audio_embedding"].get("shape_match")]
    shape_checks.extend(v.get("shape_match") for v in out["summaries"].values())
    out["shape_match"] = all(bool(x) for x in shape_checks)
    return out


def chunk_summary(result: dict[str, Any] | None) -> str:
    if not result:
        return ""
    chunks = result.get("chunks", [])
    if not isinstance(chunks, list):
        return ""
    return str(len(chunks))


def short(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    ncnn_result = load_json(fixture.get("ncnn_result_json"))
    pytorch_result = load_json(fixture.get("pytorch_result_json"))
    ncnn_mel = load_json(fixture.get("ncnn_mel_summary_json"))
    pytorch_mel = load_json(fixture.get("pytorch_mel_summary_json"))
    modules = load_json(fixture.get("module_summary_json"))

    expected = expected_texts(fixture["expected_text"])
    ncnn = result_texts(ncnn_result)
    pytorch = result_texts(pytorch_result)

    ncnn_pass = bool(ncnn["normalized"]) and ncnn["normalized"] == expected["normalized"]
    ncnn_semantic_pass = (
        bool(ncnn["semantic_normalized"]) and
        ncnn["semantic_normalized"] == expected["semantic_normalized"]
    )
    pytorch_pass = None
    pytorch_semantic_pass = None
    if pytorch_result is not None:
        pytorch_pass = bool(pytorch["normalized"]) and pytorch["normalized"] == expected["normalized"]
        pytorch_semantic_pass = (
            bool(pytorch["semantic_normalized"]) and
            pytorch["semantic_normalized"] == expected["semantic_normalized"]
        )

    return {
        "id": fixture["id"],
        "category": fixture["category"],
        "audio": fixture["audio"],
        "expected": expected,
        "ncnn": ncnn,
        "pytorch": pytorch if pytorch_result is not None else None,
        "ncnn_pass": ncnn_pass,
        "ncnn_semantic_pass": ncnn_semantic_pass,
        "pytorch_pass": pytorch_pass,
        "pytorch_semantic_pass": pytorch_semantic_pass,
        "ncnn_chunks": chunk_summary(ncnn_result),
        "ncnn_rtf": ncnn_result.get("rtf") if isinstance(ncnn_result, dict) else None,
        "chunking_strategy": ncnn_result.get("chunking_strategy") if isinstance(ncnn_result, dict) else None,
        "mel": compare_mel(ncnn_mel, pytorch_mel),
        "modules": module_status(ncnn_result, modules),
        "module_compare": module_compare(ncnn_result, modules),
    }


def markdown_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "| fixture | category | ncnn strict | ncnn semantic | PyTorch strict | PyTorch semantic | chunks | RTF | normalized expected | normalized ncnn | normalized PyTorch | notes |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for r in results:
        rtf = r["ncnn_rtf"]
        rtf_text = f"{rtf:.2f}" if isinstance(rtf, (int, float)) and math.isfinite(rtf) else ""
        notes = []
        if r["mel"]["status"] == "missing_pytorch":
            notes.append("PyTorch mel 待补")
        elif r["mel"]["status"] == "ok":
            notes.append("mel summary present")
        elif r["mel"]["status"] == "mismatch":
            notes.append("mel summary shape/metadata mismatch")
        if r["chunking_strategy"]:
            notes.append(str(r["chunking_strategy"]))
        if not r["ncnn_pass"] and r["ncnn_semantic_pass"]:
            notes.append("semantic pass; strict postprocess/output contract still differs")
        modules = r["modules"]
        if modules.get("pytorch_first_decoder_logits_summary") == "present":
            notes.append("PyTorch module summary present")
        module_cmp = r["module_compare"]
        if module_cmp.get("available"):
            if module_cmp.get("shape_match") and module_cmp.get("next_token_match"):
                notes.append("module summary aligned")
            else:
                notes.append("module summary mismatch")
        pytorch = r["pytorch"] or {"normalized": ""}
        pytorch_strict = "N/A" if r["pytorch_pass"] is None else ("PASS" if r["pytorch_pass"] else "FAIL")
        pytorch_semantic = (
            "N/A" if r["pytorch_semantic_pass"] is None
            else ("PASS" if r["pytorch_semantic_pass"] else "FAIL")
        )
        lines.append(
            "| `{id}` | {category} | {ncnn_strict} | {ncnn_semantic} | {pytorch_strict} | {pytorch_semantic} | {chunks} | {rtf} | `{expected}` | `{ncnn}` | `{pytorch}` | {notes} |".format(
                id=r["id"],
                category=r["category"],
                ncnn_strict="PASS" if r["ncnn_pass"] else "FAIL",
                ncnn_semantic="PASS" if r["ncnn_semantic_pass"] else "FAIL",
                pytorch_strict=pytorch_strict,
                pytorch_semantic=pytorch_semantic,
                chunks=r["ncnn_chunks"],
                rtf=rtf_text,
                expected=short(r["expected"]["normalized"]),
                ncnn=short(r["ncnn"]["normalized"]),
                pytorch=short(pytorch["normalized"]),
                notes=", ".join(notes),
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default="tests/qwen3_asr/fixtures.local.json")
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    args = parser.parse_args()

    fixture_doc = load_json(args.fixtures)
    if not fixture_doc:
        print(f"failed to load fixtures: {args.fixtures}", file=sys.stderr)
        return 2

    results = [evaluate_fixture(f) for f in fixture_doc.get("fixtures", [])]
    report = {
        "schema_version": 1,
        "fixtures": results,
        "summary": {
            "total": len(results),
            "ncnn_pass": sum(1 for r in results if r["ncnn_pass"]),
            "ncnn_fail": sum(1 for r in results if not r["ncnn_pass"]),
            "ncnn_semantic_pass": sum(1 for r in results if r["ncnn_semantic_pass"]),
            "ncnn_semantic_fail": sum(1 for r in results if not r["ncnn_semantic_pass"]),
            "pytorch_available": sum(1 for r in results if r["pytorch"] is not None),
        },
    }

    md = markdown_report(results)
    print(md)
    print()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).write_text(md + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
