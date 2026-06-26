# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A gradio demo for Qwen3 ASR models.
"""

import argparse
import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import torch
from qwen_asr import Qwen3ASRModel
from scipy.io.wavfile import write as wav_write


def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    """
    Accept gradio audio:
      - {"sampling_rate": int, "data": np.ndarray}
      - (sr, np.ndarray)  [some gradio versions]
    Return: (wav_float32_mono, sr)
    """
    if audio is None:
        return None

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    if isinstance(audio, tuple) and len(audio) == 2:
        a0, a1 = audio
        if isinstance(a0, int):
            sr = int(a0)
            wav = _normalize_audio(a1)
            return wav, sr
        if isinstance(a1, int):
            wav = _normalize_audio(a0)
            sr = int(a1)
            return wav, sr

    return None


def _parse_audio_any(audio: Any) -> Union[str, Tuple[np.ndarray, int]]:
    if audio is None:
        raise ValueError("Audio is required.")
    at = _audio_to_tuple(audio)
    if at is not None:
        return at
    raise ValueError("Unsupported audio input format.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-asr-demo",
        description=(
            "Launch a Gradio demo for Qwen3 ASR models (Transformers / vLLM).\n\n"
            "Examples:\n"
            "  qwen-asr-demo --asr-checkpoint Qwen/Qwen3-ASR-1.7B\n"
            "  qwen-asr-demo --asr-checkpoint Qwen/Qwen3-ASR-1.7B --aligner-checkpoint Qwen/Qwen3-ForcedAligner-0.6B\n"
            "  qwen-asr-demo --backend vllm --cuda-visible-devices 0\n"
            "  qwen-asr-demo --backend transformers --backend-kwargs '{\"device_map\":\"cuda:0\",\"dtype\":\"bfloat16\",\"attn_implementation\":\"flash_attention_2\"}'\n"
            "  qwen-asr-demo --backend vllm --backend-kwargs '{\"gpu_memory_utilization\":0.85}'\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )

    parser.add_argument("--asr-checkpoint", required=True, help="Qwen3-ASR model checkpoint path or HF repo id.")
    parser.add_argument(
        "--aligner-checkpoint",
        default=None,
        help="Qwen3-ForcedAligner checkpoint path or HF repo id (optional; enables timestamps when provided).",
    )

    parser.add_argument(
        "--campplus-model",
        default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512/campplus.onnx",
        help="Campplus model path for speaker diarization (optional).",
    )

    parser.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "vllm"],
        help="Backend for ASR model loading (default: transformers).",
    )

    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        help=(
            "Set CUDA_VISIBLE_DEVICES for the demo process (default: 0). "
            "Use e.g. '0' or '1'"
        ),
    )

    parser.add_argument(
        "--backend-kwargs",
        default=None,
        help=(
            "JSON dict for backend-specific kwargs excluding checkpoints.\n"
            "Examples:\n"
            "  transformers: '{\"device_map\":\"cuda:0\",\"dtype\":\"bfloat16\",\"attn_implementation\":\"flash_attention_2\",\"max_inference_batch_size\":32}'\n"
            "  vllm        : '{\"gpu_memory_utilization\":0.8,\"max_inference_batch_size\":32}'\n"
        ),
    )
    parser.add_argument(
        "--aligner-kwargs",
        default=None,
        help=(
            "JSON dict for forced aligner kwargs (only used when --aligner-checkpoint is set).\n"
            "Example: '{\"dtype\":\"bfloat16\",\"device_map\":\"cuda:0\"}'\n"
        ),
    )

    # Gradio server args
    parser.add_argument("--ip", default="0.0.0.0", help="Server bind IP for Gradio (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Server port for Gradio (default: 8000).")
    parser.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to create a public Gradio link (default: disabled).",
    )
    parser.add_argument("--concurrency", type=int, default=16, help="Gradio queue concurrency (default: 16).")

    # HTTPS args
    parser.add_argument("--ssl-certfile", default=None, help="Path to SSL certificate file for HTTPS (optional).")
    parser.add_argument("--ssl-keyfile", default=None, help="Path to SSL key file for HTTPS (optional).")
    parser.add_argument(
        "--ssl-verify/--no-ssl-verify",
        dest="ssl_verify",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to verify SSL certificate (default: enabled).",
    )

    return parser


def _parse_json_dict(s: Optional[str], *, name: str) -> Dict[str, Any]:
    if s is None or not str(s).strip():
        return {}
    try:
        obj = json.loads(s)
    except Exception as e:
        raise ValueError(f"Invalid JSON for {name}: {e}")
    if not isinstance(obj, dict):
        raise ValueError(f"{name} must be a JSON object (dict).")
    return obj


def _apply_cuda_visible_devices(cuda_visible_devices: str) -> None:
    v = (cuda_visible_devices or "").strip()
    if not v:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = v


def _default_backend_kwargs(backend: str) -> Dict[str, Any]:
    if backend == "transformers":
        return dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=4,
            max_new_tokens=512,
        )
    else:
        return dict(
            gpu_memory_utilization=0.8,
            max_inference_batch_size=4,
            max_new_tokens=4096,
        )


def _default_aligner_kwargs() -> Dict[str, Any]:
    return dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(override)
    return out


def _coerce_special_types(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k == "dtype" and isinstance(v, str):
            out[k] = _dtype_from_str(v)
        else:
            out[k] = v
    return out


def _make_timestamp_html(audio_upload: Any, timestamps: Any) -> str:
    """
    Build HTML with per-token audio slices, using base64 data URLs.
    Expect timestamps as list[dict] with keys: text, start_time, end_time (ms).
    """
    at = _audio_to_tuple(audio_upload)
    if at is None:
        raise ValueError("Audio input is required for visualization.")
    audio, sr = at

    if not timestamps:
        return "<div style='color:#666'>No timestamps to visualize.</div>"
    if not isinstance(timestamps, list):
        raise ValueError("Timestamps must be a list (JSON array).")

    html_content = """
    <style>
        .word-alignment-container { display: flex; flex-wrap: wrap; gap: 10px; }
        .word-box {
            border: 1px solid #ddd; border-radius: 8px; padding: 10px;
            background-color: #f9f9f9; box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            text-align: center;
        }
        .word-text { font-size: 18px; font-weight: 700; margin-bottom: 5px; }
        .word-time { font-size: 12px; color: #666; margin-bottom: 8px; }
        .word-audio audio { width: 140px; height: 30px; }
        details { border: 1px solid #ddd; border-radius: 6px; padding: 10px; background-color: #f7f7f7; }
        summary { font-weight: 700; cursor: pointer; }
    </style>
    """

    html_content += """
    <details open>
        <summary>Timestamps Visualization (时间戳可视化结果）</summary>
        <div class="word-alignment-container" style="margin-top: 14px;">
    """

    for item in timestamps:
        if not isinstance(item, dict):
            continue
        word = str(item.get("text", "") or "")
        start = item.get("start_time", None)
        end = item.get("end_time", None)
        speaker = item.get("speaker", None)
        if start is None or end is None:
            continue

        start = float(start)
        end = float(end)
        if end <= start:
            continue

        start_sample = max(0, int(start * sr))
        end_sample = min(len(audio), int(end * sr))
        if end_sample <= start_sample:
            continue

        seg = audio[start_sample:end_sample]
        seg_i16 = (np.clip(seg, -1.0, 1.0) * 32767.0).astype(np.int16)

        mem = io.BytesIO()
        wav_write(mem, sr, seg_i16)
        mem.seek(0)
        b64 = base64.b64encode(mem.read()).decode("utf-8")
        audio_src = f"data:audio/wav;base64,{b64}"

        speaker_label = f"Speaker {speaker}" if speaker is not None else ""
        speaker_style = f"border-left: 4px solid hsl({(speaker % 12) * 30}, 70%, 50%);" if speaker is not None else ""

        html_content += f"""
        <div class="word-box" style="{speaker_style}">
            <div class="word-text">{word}</div>
            <div class="word-time">{start:.2f} - {end:.2f} s</div>
            <div class="word-audio">
                <audio controls preload="none" src="{audio_src}"></audio>
            </div>
            {f'<div style="font-size: 11px; color: #888; margin-top: 4px;">{speaker_label}</div>' if speaker_label else ''}
        </div>
        """

    html_content += "</div></details>"
    return html_content


def build_demo(
    asr: Qwen3ASRModel,
    asr_ckpt: str,
    backend: str,
    aligner_ckpt: Optional[str] = None,
) -> gr.Blocks:
    supported_langs_raw = asr.get_supported_languages()
    lang_choices_disp, lang_map = _build_choices_and_map([x for x in supported_langs_raw])
    lang_choices = ["Auto"] + lang_choices_disp

    has_aligner = bool(aligner_ckpt)

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )
    css = ".gradio-container {max-width: none !important;}"

    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(
            f"""
# Qwen3 ASR Demo
**Backend:** `{backend}`  
**ASR Checkpoint:** `{asr_ckpt}`  
**Forced Aligner:** `{aligner_ckpt if aligner_ckpt else "(none)"}`  
"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                audio_in = gr.Audio(label="Audio Input (上传音频)", type="numpy")
                lang_in = gr.Dropdown(
                    label="Language (语种)",
                    choices=lang_choices,
                    value="Auto",
                    interactive=True,
                )
                if has_aligner:
                    ts_in = gr.Checkbox(
                        label="Return Timestamps (是否返回时间戳)",
                        value=True,
                    )
                else:
                    ts_in = gr.State(False)

                btn = gr.Button("Transcribe (识别)", variant="primary")

            with gr.Column(scale=2):
                out_lang = gr.Textbox(label="Detected Language", lines=1)
                out_text = gr.Textbox(label="Result Text", lines=12)

            if has_aligner:
                with gr.Column(scale=3):
                    out_ts = gr.JSON(label="Timestamps（时间戳结果）")
                    viz_btn = gr.Button("Visualize Timestamps (可视化时间戳)", variant="secondary")
            else:
                with gr.Column(scale=3):
                    out_ts = gr.State(None)
                    viz_btn = gr.State(None)

        # Put the visualization panel below the three columns
        if has_aligner:
            with gr.Row():
                out_ts_html = gr.HTML(label="Timestamps Visualization (时间戳可视化结果)")
        else:
            out_ts_html = gr.State("")

        def run(audio_upload: Any, lang_disp: str, return_ts: bool):
            audio_obj = _parse_audio_any(audio_upload)

            language = None
            if lang_disp and lang_disp != "Auto":
                language = lang_map.get(lang_disp, lang_disp)

            return_ts = bool(return_ts) and has_aligner

            results = asr.transcribe(
                audio=audio_obj,
                language=language,
                return_time_stamps=return_ts,
            )
            if not isinstance(results, list) or len(results) != 1:
                raise RuntimeError(
                    f"Unexpected result size: {type(results)} "
                    f"len={len(results) if isinstance(results, list) else 'N/A'}"
                )

            r = results[0]

            if has_aligner:
                ts_payload = None
                if return_ts:
                    ts_payload = [
                        dict(
                            text=getattr(t, "text", None),
                            start_time=getattr(t, "start_time", None),
                            end_time=getattr(t, "end_time", None),
                            speaker=getattr(t, "speaker", None),
                        )
                        for t in (getattr(r, "time_stamps", None) or [])
                    ]
                return (
                    getattr(r, "language", "") or "",
                    getattr(r, "text", "") or "",
                    gr.update(value=ts_payload) if return_ts else gr.update(value=None),
                    gr.update(value=""),  # clear html on each transcribe
                )
            else:
                return (
                    getattr(r, "language", "") or "",
                    getattr(r, "text", "") or "",
                )

        def visualize(audio_upload: Any, timestamps_json: Any):
            return _make_timestamp_html(audio_upload, timestamps_json)

        if has_aligner:
            btn.click(
                run,
                inputs=[audio_in, lang_in, ts_in],
                outputs=[out_lang, out_text, out_ts, out_ts_html],
            )
            viz_btn.click(
                visualize,
                inputs=[audio_in, out_ts],
                outputs=[out_ts_html],
            )
        else:
            btn.click(
                run,
                inputs=[audio_in, lang_in, ts_in],
                outputs=[out_lang, out_text],
            )

    return demo


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _apply_cuda_visible_devices(args.cuda_visible_devices)

    backend = args.backend
    asr_ckpt = args.asr_checkpoint
    aligner_ckpt = args.aligner_checkpoint

    user_backend_kwargs = _parse_json_dict(args.backend_kwargs, name="--backend-kwargs")
    user_aligner_kwargs = _parse_json_dict(args.aligner_kwargs, name="--aligner-kwargs")

    # Add campplus model path to aligner kwargs
    if hasattr(args, "campplus_model") and args.campplus_model:
        user_aligner_kwargs["campplus_model"] = args.campplus_model

    backend_kwargs = _merge_dicts(_default_backend_kwargs(backend), user_backend_kwargs)
    backend_kwargs = _coerce_special_types(backend_kwargs)

    forced_aligner = None
    forced_aligner_kwargs = None
    if aligner_ckpt:
        forced_aligner = aligner_ckpt
        aligner_kwargs = _merge_dicts(_default_aligner_kwargs(), user_aligner_kwargs)
        forced_aligner_kwargs = _coerce_special_types(aligner_kwargs)

    if backend == "transformers":
        asr = Qwen3ASRModel.from_pretrained(
            asr_ckpt,
            forced_aligner=forced_aligner,
            forced_aligner_kwargs=forced_aligner_kwargs,
            **backend_kwargs,
        )
    else:
        asr = Qwen3ASRModel.LLM(
            model=asr_ckpt,
            forced_aligner=forced_aligner,
            forced_aligner_kwargs=forced_aligner_kwargs,
            **backend_kwargs,
        )

    demo = build_demo(asr, asr_ckpt, backend, aligner_ckpt=aligner_ckpt)

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        ssl_verify=True if args.ssl_verify else False,
    )
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
