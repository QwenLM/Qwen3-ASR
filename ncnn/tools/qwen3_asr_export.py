from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoConfig, AutoModel, AutoProcessor


class Qwen3ASRAudioEncoderTS(nn.Module):
    def __init__(self, audio_tower: nn.Module, example_frames: int):
        super().__init__()
        self.audio_tower = audio_tower
        # Keep chunking static and batch-1 so pnnx can lower the audio path to ncnn.
        self.chunk_frames = int(audio_tower.n_window) * 2
        self.full_chunks = example_frames // self.chunk_frames
        self.tail_frames = example_frames % self.chunk_frames
        self.after_full_chunk_frames = self._after_cnn_frames(self.chunk_frames)
        self.after_tail_frames = self._after_cnn_frames(self.tail_frames) if self.tail_frames else 0
        self.after_total_frames = self.full_chunks * self.after_full_chunk_frames + self.after_tail_frames

    @staticmethod
    def _after_cnn_frames(input_frames: int) -> int:
        input_frames_leave = input_frames % 100
        feat_frames = (input_frames_leave - 1) // 2 + 1
        output_frames = ((feat_frames - 1) // 2 + 1 - 1) // 2 + 1 + (input_frames // 100) * 13
        return int(output_frames)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        chunks = []
        for i in range(self.full_chunks):
            start = i * self.chunk_frames
            chunks.append(input_features[:, start : start + self.chunk_frames])
        if self.tail_frames:
            start = self.full_chunks * self.chunk_frames
            tail = input_features[:, start : start + self.tail_frames]
            chunks.append(F.pad(tail, (0, self.chunk_frames - self.tail_frames)))

        valid_chunks = []
        for chunk in chunks[: self.full_chunks]:
            valid_chunks.append(self._encode_conv_chunk(chunk, self.after_full_chunk_frames))
        if self.tail_frames:
            valid_chunks.append(self._encode_conv_chunk(chunks[self.full_chunks], self.after_tail_frames))
        hidden_states = torch.cat(valid_chunks, dim=1)

        for encoder_layer in self.audio_tower.layers:
            hidden_states = self._encode_layer(encoder_layer, hidden_states)

        hidden_states = self.audio_tower.ln_post(hidden_states)
        hidden_states = self.audio_tower.proj1(hidden_states)
        hidden_states = self.audio_tower.act(hidden_states)
        return self.audio_tower.proj2(hidden_states)

    def _encode_conv_chunk(self, chunk: torch.Tensor, valid_frames: int) -> torch.Tensor:
        hidden_states = chunk.unsqueeze(0).unsqueeze(0)
        hidden_states = F.gelu(self.audio_tower.conv2d1(hidden_states))
        hidden_states = F.gelu(self.audio_tower.conv2d2(hidden_states))
        hidden_states = F.gelu(self.audio_tower.conv2d3(hidden_states))

        b, c, f, t = hidden_states.size()
        hidden_states = self.audio_tower.conv_out(
            hidden_states.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )
        positional_embedding = (
            self.audio_tower.positional_embedding.positional_embedding[: hidden_states.shape[1], :]
            .unsqueeze(0)
            .to(hidden_states.dtype)
        )
        hidden_states = hidden_states + positional_embedding
        return hidden_states[:, :valid_frames, :]

    @staticmethod
    def _encode_layer(encoder_layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = encoder_layer.self_attn_layer_norm(hidden_states)

        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]
        num_heads = encoder_layer.self_attn.num_heads
        head_dim = encoder_layer.self_attn.head_dim
        query_states = encoder_layer.self_attn.q_proj(hidden_states).view(batch_size, seq_length, num_heads, head_dim)
        key_states = encoder_layer.self_attn.k_proj(hidden_states).view(batch_size, seq_length, num_heads, head_dim)
        value_states = encoder_layer.self_attn.v_proj(hidden_states).view(batch_size, seq_length, num_heads, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=0.0,
            scale=encoder_layer.self_attn.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        hidden_states = residual + encoder_layer.self_attn.out_proj(attn_output)

        residual = hidden_states
        hidden_states = encoder_layer.final_layer_norm(hidden_states)
        hidden_states = encoder_layer.fc1(hidden_states)
        hidden_states = encoder_layer.activation_fn(hidden_states)
        hidden_states = encoder_layer.fc2(hidden_states)
        return residual + hidden_states


class Qwen3ASRTextEmbedTS(nn.Module):
    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.embed_tokens = text_model.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class Qwen3ASRTextBackboneTS(nn.Module):
    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.text_model = text_model

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        hidden_states = inputs_embeds

        cache_position = torch.arange(seq_len, device=inputs_embeds.device)
        position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
        text_position_ids = position_ids[0]

        mask_value = torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.full((seq_len, seq_len), mask_value, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)

        position_embeddings = self.text_model.rotary_emb(hidden_states, position_ids)
        for layer in self.text_model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        return self.text_model.norm(hidden_states)


class Qwen3ASRTextPrefillKVTS(nn.Module):
    def __init__(self, text_model: nn.Module, num_layers: int):
        super().__init__()
        self.text_model = text_model
        self.num_layers = int(num_layers)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (q * cos) + (Qwen3ASRTextPrefillKVTS._rotate_half(q) * sin), (k * cos) + (
            Qwen3ASRTextPrefillKVTS._rotate_half(k) * sin
        )

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def _layer_forward(
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn = layer.self_attn

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = Qwen3ASRTextPrefillKVTS._apply_rotary(query_states, key_states, cos, sin)

        if past_k is not None and past_v is not None:
            key_states = torch.cat((past_k, key_states), dim=2)
            value_states = torch.cat((past_v, value_states), dim=2)

        attn_key = Qwen3ASRTextPrefillKVTS._repeat_kv(key_states, int(attn.num_key_value_groups))
        attn_value = Qwen3ASRTextPrefillKVTS._repeat_kv(value_states, int(attn.num_key_value_groups))
        attn_weights = torch.matmul(query_states, attn_key.transpose(2, 3)) * float(attn.scaling)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : attn_key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, attn_value)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        hidden_states = residual + attn.o_proj(attn_output)

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)
        return hidden_states, key_states, value_states

    @staticmethod
    def _causal_mask(seq_len: int, kv_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        mask_value = torch.finfo(dtype).min
        mask = torch.full((seq_len, kv_len), mask_value, dtype=dtype, device=device)
        past_len = kv_len - seq_len
        for i in range(seq_len):
            mask[i, : past_len + i + 1] = 0
        return mask.view(1, 1, seq_len, kv_len)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        del attention_mask
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        hidden_states = inputs_embeds

        cache_position = torch.arange(seq_len, device=inputs_embeds.device)
        position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
        causal_mask = self._causal_mask(seq_len, seq_len, inputs_embeds.dtype, inputs_embeds.device)
        position_embeddings = self.text_model.rotary_emb(hidden_states, position_ids)

        flat: List[torch.Tensor] = [hidden_states]
        caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.text_model.layers:
            hidden_states, key_states, value_states = self._layer_forward(
                layer, hidden_states, causal_mask, position_embeddings
            )
            caches.append((key_states, value_states))

        hidden_states = self.text_model.norm(hidden_states)
        flat[0] = hidden_states
        for key_states, value_states in caches:
            flat.append(key_states)
            flat.append(value_states)
        return tuple(flat)


class Qwen3ASRTextDecodeKVTS(nn.Module):
    def __init__(self, text_model: nn.Module, num_layers: int):
        super().__init__()
        self.text_model = text_model
        self.num_layers = int(num_layers)

    def forward(self, inputs_embeds: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *flat_cache: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        seq_len = inputs_embeds.shape[1]
        past_len = flat_cache[0].shape[2]
        hidden_states = inputs_embeds

        position_embeddings = (cos, sin)

        flat: List[torch.Tensor] = [hidden_states]
        for i in range(self.num_layers):
            past_k = flat_cache[i * 2]
            past_v = flat_cache[i * 2 + 1]
            hidden_states, key_states, value_states = Qwen3ASRTextPrefillKVTS._layer_forward(
                self.text_model.layers[i], hidden_states, None, position_embeddings, past_k, past_v
            )
            flat.append(key_states)
            flat.append(value_states)

        hidden_states = self.text_model.norm(hidden_states)
        flat[0] = hidden_states
        return tuple(flat)


class Qwen3ASRLmHeadTS(nn.Module):
    def __init__(self, lm_head: nn.Module):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


@dataclass
class Qwen3ASRExportBundle:
    root: Path
    processor_dir: Path
    audio_encoder_path: Path
    text_embed_path: Path
    text_backbone_path: Path
    text_prefill_kv_path: Path
    text_decode_kv_path: Path
    lm_head_path: Path
    manifest_path: Path


def _resolve_dtype(name: Optional[str], default: torch.dtype) -> torch.dtype:
    if not name:
        return default
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    return str(value)


def _token_to_string(token: Any) -> str:
    if isinstance(token, dict):
        return str(token.get("content", ""))
    if token is None:
        return ""
    return str(token)


def _extract_special_tokens(tokenizer_config: Dict[str, Any]) -> Dict[str, Any]:
    special: Dict[str, Any] = {}
    for key in ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"]:
        special[key] = _token_to_string(tokenizer_config.get(key, ""))

    additional = tokenizer_config.get("additional_special_tokens", [])
    if isinstance(additional, list):
        special["additional_special_tokens"] = [_token_to_string(t) for t in additional if _token_to_string(t)]
    else:
        special["additional_special_tokens"] = []
    return special


def _write_bpe_tokenizer_files(processor_dir: Path, out_dir: Path) -> Dict[str, Any]:
    tokenizer_json_path = processor_dir / "tokenizer.json"
    tokenizer_config_path = processor_dir / "tokenizer_config.json"

    if not tokenizer_json_path.exists():
        return {
            "type": "huggingface",
            "processor_dir": processor_dir.name,
            "vocab_file": "",
            "merges_file": "",
            "bos": "",
            "eos": "",
            "pad": "",
            "unk": "",
            "additional_special_tokens": [],
        }

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    model = tokenizer_data.get("model", {})
    vocab = model.get("vocab", {})
    merges = model.get("merges", [])

    vocab_path = out_dir / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in sorted(vocab.items(), key=lambda item: item[1]):
            f.write(f"{token}\n")

    merges_path = out_dir / "merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            if isinstance(merge, list) and len(merge) == 2:
                f.write(f"{merge[0]} {merge[1]}\n")
            elif isinstance(merge, str):
                f.write(f"{merge}\n")

    tokenizer_config: Dict[str, Any] = {}
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
    special = _extract_special_tokens(tokenizer_config)

    return {
        "type": "bbpe",
        "processor_dir": processor_dir.name,
        "vocab_file": vocab_path.name,
        "merges_file": merges_path.name,
        "bos": special.get("bos_token", ""),
        "eos": special.get("eos_token", ""),
        "pad": special.get("pad_token", ""),
        "unk": special.get("unk_token", ""),
        "additional_special_tokens": special.get("additional_special_tokens", []),
    }


def _device_from_arg(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _register_qwen3_asr_auto_classes() -> None:
    try:
        from qwen_asr.core.transformers_backend import (  # type: ignore
            Qwen3ASRConfig,
            Qwen3ASRForConditionalGeneration,
            Qwen3ASRProcessor,
        )
        from transformers import AutoConfig, AutoModel, AutoProcessor

        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    except Exception:
        # Newer model repos may provide native/remote-code registrations.
        pass


def _normalize_qwen3_asr_config(config: Any) -> Any:
    thinker_config = getattr(config, "thinker_config", config)
    text_config = getattr(thinker_config, "text_config", None)
    if text_config is not None and getattr(text_config, "rope_scaling", None) is None:
        text_config.rope_scaling = {
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
        }
    return config


def _load_registered_qwen3_asr_config(model_id: str) -> Any:
    config_path = Path(model_id) / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)

        if (
            raw_config.get("model_type") == "qwen3_asr"
            and "thinker_config" not in raw_config
            and isinstance(raw_config.get("audio_config"), dict)
            and isinstance(raw_config.get("text_config"), dict)
        ):
            from qwen_asr.core.transformers_backend import Qwen3ASRConfig  # type: ignore

            thinker_config = {
                "audio_config": raw_config["audio_config"],
                "text_config": raw_config["text_config"],
                "audio_token_id": raw_config.get("audio_token_id", 151646),
                "audio_start_token_id": raw_config.get("audio_start_token_id", 151647),
                "user_token_id": raw_config.get("user_token_id", 872),
                "initializer_range": raw_config.get("initializer_range", 0.02),
            }
            config = Qwen3ASRConfig(
                thinker_config=thinker_config,
                support_languages=raw_config.get("support_languages"),
                **{
                    k: v
                    for k, v in raw_config.items()
                    if k not in {"audio_config", "text_config", "thinker_config", "support_languages"}
                },
            )
            return _normalize_qwen3_asr_config(config)

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return _normalize_qwen3_asr_config(config)


def _load_model_and_processor(model_id: str, device: str, dtype: torch.dtype) -> Tuple[Any, Any]:
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    load_on_device = device.startswith("cuda")
    if load_on_device:
        load_kwargs["device_map"] = {"": device}

    try:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)
        processor = AutoProcessor.from_pretrained(model_id, fix_mistral_regex=True, trust_remote_code=True)
    except Exception:
        _register_qwen3_asr_auto_classes()
        config = _load_registered_qwen3_asr_config(model_id)
        model = AutoModel.from_pretrained(model_id, config=config, **load_kwargs)
        processor = AutoProcessor.from_pretrained(model_id, fix_mistral_regex=True, trust_remote_code=True)
    if not load_on_device:
        model = model.to(device)
    model = model.eval()
    return model, processor


def _get_thinker(model: nn.Module) -> nn.Module:
    if hasattr(model, "thinker"):
        return model.thinker
    return model


def _get_text_model(thinker: nn.Module) -> nn.Module:
    if hasattr(thinker, "model"):
        return thinker.model
    raise AttributeError("Could not find Qwen3-ASR text model at thinker.model")


def _get_audio_tower(thinker: nn.Module) -> nn.Module:
    if hasattr(thinker, "audio_tower"):
        return thinker.audio_tower
    raise AttributeError("Could not find Qwen3-ASR audio encoder at thinker.audio_tower")


def _get_lm_head(thinker: nn.Module, model: nn.Module) -> nn.Module:
    if hasattr(thinker, "lm_head"):
        return thinker.lm_head
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise AttributeError("Could not find Qwen3-ASR lm_head")


def _get_thinker_config(model: nn.Module) -> Any:
    cfg = model.config
    return getattr(cfg, "thinker_config", cfg)


def _trace_audio_encoder(
    audio_tower: nn.Module,
    out_path: Path,
    device: str,
    dtype: torch.dtype,
    num_mel_bins: int,
    example_frames: int,
) -> None:
    wrapper = Qwen3ASRAudioEncoderTS(audio_tower, example_frames=example_frames).to(device).eval()
    example_input = torch.randn(num_mel_bins, example_frames, device=device, dtype=dtype)

    traced = torch.jit.trace(wrapper, example_input, strict=False)
    traced.save(str(out_path))


def _trace_text_embed(
    text_model: nn.Module,
    out_path: Path,
    device: str,
    dtype: torch.dtype,
    vocab_size: int,
    seq_len: int,
) -> None:
    wrapper = Qwen3ASRTextEmbedTS(text_model).to(device).eval()
    example_ids = torch.randint(0, vocab_size, (1, seq_len), device=device, dtype=torch.long)
    traced = torch.jit.trace(wrapper, example_ids, strict=False)
    traced.save(str(out_path))


def _trace_text_backbone(
    text_model: nn.Module,
    out_path: Path,
    device: str,
    dtype: torch.dtype,
    hidden_size: int,
    seq_len: int,
) -> None:
    if hasattr(text_model, "config"):
        text_model.config._attn_implementation = "eager"
    wrapper = Qwen3ASRTextBackboneTS(text_model).to(device).eval()
    example_embeds = torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype)
    example_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)
    traced = torch.jit.trace(wrapper, (example_embeds, example_mask), strict=False)
    traced.save(str(out_path))


def _trace_text_kv(
    text_model: nn.Module,
    prefill_path: Path,
    decode_path: Path,
    device: str,
    dtype: torch.dtype,
    hidden_size: int,
    prefill_len: int,
    cache_len: int,
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
) -> None:
    if hasattr(text_model, "config"):
        text_model.config._attn_implementation = "eager"
        text_model.config.use_cache = True

    prefill = Qwen3ASRTextPrefillKVTS(text_model, num_layers=num_layers).to(device).eval()
    prefill_embeds = torch.randn(1, prefill_len, hidden_size, device=device, dtype=dtype)
    prefill_mask = torch.ones(1, prefill_len, device=device, dtype=torch.long)
    traced_prefill = torch.jit.trace(prefill, (prefill_embeds, prefill_mask), strict=False)
    traced_prefill.save(str(prefill_path))

    decode = Qwen3ASRTextDecodeKVTS(text_model, num_layers=num_layers).to(device).eval()
    decode_embeds = torch.randn(1, 1, hidden_size, device=device, dtype=dtype)
    decode_cos = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    decode_sin = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    flat_cache: List[torch.Tensor] = []
    for _ in range(num_layers):
        flat_cache.append(torch.randn(1, num_key_value_heads, cache_len, head_dim, device=device, dtype=dtype))
        flat_cache.append(torch.randn(1, num_key_value_heads, cache_len, head_dim, device=device, dtype=dtype))
    traced_decode = torch.jit.trace(decode, tuple([decode_embeds, decode_cos, decode_sin] + flat_cache), strict=False)
    traced_decode.save(str(decode_path))


def _trace_lm_head(
    lm_head: nn.Module,
    out_path: Path,
    device: str,
    dtype: torch.dtype,
    hidden_size: int,
    seq_len: int,
) -> None:
    wrapper = Qwen3ASRLmHeadTS(lm_head).to(device).eval()
    example_hidden = torch.randn(1, seq_len, hidden_size, device=device, dtype=dtype)
    traced = torch.jit.trace(wrapper, example_hidden, strict=False)
    traced.save(str(out_path))


def _run_pnnx(pnnx_bin: str, pt_path: Path, extra_args: Optional[list[str]] = None) -> None:
    cmd = [pnnx_bin, str(pt_path)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, cwd=str(pt_path.parent), check=True)


def _relax_decode_cache_reshape(param_path: Path, cache_len: int, head_dim: int, num_attention_heads: int) -> None:
    """Allow the one-token KV decoder to feed its growing cache back into itself."""
    if not param_path.exists():
        return

    fixed = f"0={head_dim} 1={cache_len + 1} 2={num_attention_heads}"
    relaxed = f"0={head_dim} 1=-1 2={num_attention_heads}"
    lines = param_path.read_text().splitlines()
    replaced = 0
    out_lines = []
    for line in lines:
        if line.startswith("Reshape") and fixed in line:
            line = line.replace(fixed, relaxed)
            replaced += 1
        out_lines.append(line)

    param_path.write_text("\n".join(out_lines) + "\n")
    if replaced == 0:
        raise RuntimeError(f"did not relax any decode cache reshape in {param_path}")
    print(f"Relaxed {replaced} decode cache reshape layers in {param_path.name}")


def _save_pnnx_input(path: Path, value: np.ndarray) -> str:
    np.save(path, value)
    return f"input={path.name}"


def export_qwen3_asr(
    model_id: str,
    out_dir: str = "./export_qwen3_asr",
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    export_audio_encoder: bool = True,
    export_text_stack: bool = True,
    convert_ncnn: bool = False,
    pnnx_bin: str = "pnnx",
    audio_frames: Optional[int] = None,
    text_seq_len: int = 8,
    export_kv_cache: bool = False,
    kv_cache_len: Optional[int] = None,
) -> Qwen3ASRExportBundle:
    device = _device_from_arg(device)
    default_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    torch_dtype = _resolve_dtype(dtype, default_dtype)
    if convert_ncnn and torch_dtype != torch.float32:
        raise ValueError("Qwen3-ASR pnnx/ncnn conversion currently requires --dtype fp32.")

    model, processor = _load_model_and_processor(model_id, device=device, dtype=torch_dtype)

    root = Path(out_dir)
    processor_dir = root / "processor"
    root.mkdir(parents=True, exist_ok=True)
    processor_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(processor_dir))

    thinker = _get_thinker(model)
    text_model = _get_text_model(thinker)
    audio_tower = _get_audio_tower(thinker)
    lm_head = _get_lm_head(thinker, model)
    thinker_cfg = _get_thinker_config(model)
    audio_cfg = thinker_cfg.audio_config
    text_cfg = thinker_cfg.text_config

    audio_encoder_path = root / "audio_encoder.pt"
    text_embed_path = root / "text_embed.pt"
    text_backbone_path = root / "text_backbone.pt"
    text_prefill_kv_path = root / "text_prefill_kv.pt"
    text_decode_kv_path = root / "text_decode_kv.pt"
    lm_head_path = root / "lm_head.pt"

    if export_audio_encoder:
        example_frames = int(audio_frames or getattr(audio_cfg, "max_source_positions", 256) or 256)
        if example_frames <= 0:
            raise ValueError("--audio-frames must be positive.")
        _trace_audio_encoder(
            audio_tower,
            audio_encoder_path,
            device=device,
            dtype=torch_dtype,
            num_mel_bins=int(audio_cfg.num_mel_bins),
            example_frames=max(16, example_frames),
        )

    if export_text_stack:
        _trace_text_embed(
            text_model,
            text_embed_path,
            device=device,
            dtype=torch_dtype,
            vocab_size=int(text_cfg.vocab_size),
            seq_len=text_seq_len,
        )
        _trace_text_backbone(
            text_model,
            text_backbone_path,
            device=device,
            dtype=torch_dtype,
            hidden_size=int(text_cfg.hidden_size),
            seq_len=text_seq_len,
        )
        if export_kv_cache:
            num_attention_heads = int(text_cfg.num_attention_heads)
            num_key_value_heads = int(getattr(text_cfg, "num_key_value_heads", num_attention_heads))
            head_dim = int(getattr(text_cfg, "head_dim", int(text_cfg.hidden_size) // num_attention_heads))
            _trace_text_kv(
                text_model,
                text_prefill_kv_path,
                text_decode_kv_path,
                device=device,
                dtype=torch_dtype,
                hidden_size=int(text_cfg.hidden_size),
                prefill_len=text_seq_len,
                cache_len=int(kv_cache_len or max(1, text_seq_len - 1)),
                num_layers=int(text_cfg.num_hidden_layers),
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
            )
        _trace_lm_head(
            lm_head,
            lm_head_path,
            device=device,
            dtype=torch_dtype,
            hidden_size=int(text_cfg.hidden_size),
            seq_len=text_seq_len,
        )

    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_manifest = _write_bpe_tokenizer_files(processor_dir, root)
    manifest: Dict[str, Any] = {
        "model_type": "qwen3_asr",
        "params": {
            "audio_encoder_pt": audio_encoder_path.name if export_audio_encoder else "",
            "audio_encoder_param": "audio_encoder.ncnn.param" if export_audio_encoder else "",
            "audio_encoder_bin": "audio_encoder.ncnn.bin" if export_audio_encoder else "",
            "text_embed_pt": text_embed_path.name if export_text_stack else "",
            "text_embed_param": "text_embed.ncnn.param" if export_text_stack else "",
            "text_embed_bin": "text_embed.ncnn.bin" if export_text_stack else "",
            "text_backbone_pt": text_backbone_path.name if export_text_stack else "",
            "text_backbone_param": "text_backbone.ncnn.param" if export_text_stack else "",
            "text_backbone_bin": "text_backbone.ncnn.bin" if export_text_stack else "",
            "text_prefill_kv_pt": text_prefill_kv_path.name if export_text_stack and export_kv_cache else "",
            "text_prefill_kv_param": "text_prefill_kv.ncnn.param" if export_text_stack and export_kv_cache else "",
            "text_prefill_kv_bin": "text_prefill_kv.ncnn.bin" if export_text_stack and export_kv_cache else "",
            "text_decode_kv_pt": text_decode_kv_path.name if export_text_stack and export_kv_cache else "",
            "text_decode_kv_param": "text_decode_kv.ncnn.param" if export_text_stack and export_kv_cache else "",
            "text_decode_kv_bin": "text_decode_kv.ncnn.bin" if export_text_stack and export_kv_cache else "",
            "lm_head_pt": lm_head_path.name if export_text_stack else "",
            "lm_head_param": "lm_head.ncnn.param" if export_text_stack else "",
            "lm_head_bin": "lm_head.ncnn.bin" if export_text_stack else "",
            "processor_dir": processor_dir.name,
        },
        "tokenizer": tokenizer_manifest,
        "setting": {
            "audio_token_id": int(thinker_cfg.audio_token_id),
            "audio_start_token_id": int(thinker_cfg.audio_start_token_id),
            "audio_end_token_id": int(thinker_cfg.audio_end_token_id),
            "user_token_id": int(thinker_cfg.user_token_id),
            "audio_config": _to_jsonable(audio_cfg.to_dict() if hasattr(audio_cfg, "to_dict") else audio_cfg.__dict__),
            "text_config": _to_jsonable(text_cfg.to_dict() if hasattr(text_cfg, "to_dict") else text_cfg.__dict__),
            "text_seq_len": int(text_seq_len),
            "audio_frames": int(max(16, example_frames)) if export_audio_encoder else 0,
            "kv_cache_len": int(kv_cache_len or max(1, text_seq_len - 1)),
            "kv_cache_exported": bool(export_kv_cache),
            "support_languages": _to_jsonable(getattr(model.config, "support_languages", None)),
        },
    }

    if tokenizer is not None:
        manifest["tokenizer"].update(
            {
                "audio_token": getattr(tokenizer, "audio_token", ""),
                "audio_bos_token": getattr(tokenizer, "audio_bos_token", ""),
                "audio_eos_token": getattr(tokenizer, "audio_eos_token", ""),
            }
        )

    manifest_path = root / "model.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if convert_ncnn:
        if export_audio_encoder:
            audio_features_input = root / "pnnx_audio_features.npy"
            np.save(audio_features_input, np.zeros((int(audio_cfg.num_mel_bins), max(16, example_frames)), dtype=np.float32))
            _run_pnnx(
                pnnx_bin,
                audio_encoder_path,
                [
                    f"input={audio_features_input.name}",
                    "fp16=0",
                ],
            )
        if export_text_stack:
            text_ids_input = root / "pnnx_text_input_ids.npy"
            _save_pnnx_input(text_ids_input, np.arange(text_seq_len, dtype=np.int64).reshape(1, text_seq_len))
            _run_pnnx(pnnx_bin, text_embed_path, [f"input={text_ids_input.name}", "fp16=0"])
            _run_pnnx(pnnx_bin, text_backbone_path, [f"inputshape=[1,{text_seq_len},{int(text_cfg.hidden_size)}],[1,{text_seq_len}]", "fp16=0"])
            if export_kv_cache:
                num_attention_heads = int(text_cfg.num_attention_heads)
                num_key_value_heads = int(getattr(text_cfg, "num_key_value_heads", num_attention_heads))
                head_dim = int(getattr(text_cfg, "head_dim", int(text_cfg.hidden_size) // num_attention_heads))
                cache_len = int(kv_cache_len or max(1, text_seq_len - 1))
                cache_shapes = ",".join(
                    [f"[1,{num_key_value_heads},{cache_len},{head_dim}]" for _ in range(int(text_cfg.num_hidden_layers) * 2)]
                )
                _run_pnnx(
                    pnnx_bin,
                    text_prefill_kv_path,
                    [f"inputshape=[1,{text_seq_len},{int(text_cfg.hidden_size)}],[1,{text_seq_len}]", "fp16=0"],
                )
                _run_pnnx(
                    pnnx_bin,
                    text_decode_kv_path,
                    [
                        f"inputshape=[1,1,{int(text_cfg.hidden_size)}],[1,1,{head_dim}],[1,1,{head_dim}],{cache_shapes}",
                        "fp16=0",
                    ],
                )
                _relax_decode_cache_reshape(
                    root / "text_decode_kv.ncnn.param",
                    cache_len=cache_len,
                    head_dim=head_dim,
                    num_attention_heads=num_attention_heads,
                )
            _run_pnnx(pnnx_bin, lm_head_path, [f"inputshape=[1,{text_seq_len},{int(text_cfg.hidden_size)}]", "fp16=0"])

    return Qwen3ASRExportBundle(
        root=root,
        processor_dir=processor_dir,
        audio_encoder_path=audio_encoder_path,
        text_embed_path=text_embed_path,
        text_backbone_path=text_backbone_path,
        text_prefill_kv_path=text_prefill_kv_path,
        text_decode_kv_path=text_decode_kv_path,
        lm_head_path=lm_head_path,
        manifest_path=manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR TorchScript bundles and metadata.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--out-dir", default="./export_qwen3_asr", help="Output directory.")
    parser.add_argument("--device", default=None, help="Torch device to run export on.")
    parser.add_argument("--dtype", default=None, help="Override export dtype: fp32, fp16, bf16.")
    parser.add_argument("--no-audio-encoder", action="store_true", help="Skip exporting the audio encoder.")
    parser.add_argument("--no-text-stack", action="store_true", help="Skip exporting the text stack.")
    parser.add_argument("--convert-ncnn", action="store_true", help="Run pnnx on exported TorchScript modules.")
    parser.add_argument("--pnnx-bin", default="pnnx", help="Path to pnnx executable.")
    parser.add_argument("--audio-frames", type=int, default=None, help="Static log-mel frame count for traced audio encoder.")
    parser.add_argument("--text-seq-len", type=int, default=8, help="Static text sequence length for traced text modules.")
    parser.add_argument("--export-kv-cache", action="store_true", help="Also export cache-aware text prefill/decode modules.")
    parser.add_argument("--kv-cache-len", type=int, default=None, help="Static past length for the traced one-token KV decoder.")
    args = parser.parse_args()

    bundle = export_qwen3_asr(
        model_id=args.model_id,
        out_dir=args.out_dir,
        device=args.device,
        dtype=args.dtype,
        export_audio_encoder=not args.no_audio_encoder,
        export_text_stack=not args.no_text_stack,
        convert_ncnn=args.convert_ncnn,
        pnnx_bin=args.pnnx_bin,
        audio_frames=args.audio_frames,
        text_seq_len=args.text_seq_len,
        export_kv_cache=args.export_kv_cache,
        kv_cache_len=args.kv_cache_len,
    )

    print(f"Saved manifest: {bundle.manifest_path}")
    if bundle.audio_encoder_path.exists():
        print(f"Saved audio encoder: {bundle.audio_encoder_path}")
    if bundle.text_embed_path.exists():
        print(f"Saved text embed: {bundle.text_embed_path}")
    if bundle.text_backbone_path.exists():
        print(f"Saved text backbone: {bundle.text_backbone_path}")
    if bundle.text_prefill_kv_path.exists():
        print(f"Saved text KV prefill: {bundle.text_prefill_kv_path}")
    if bundle.text_decode_kv_path.exists():
        print(f"Saved text KV decode: {bundle.text_decode_kv_path}")
    if bundle.lm_head_path.exists():
        print(f"Saved lm head: {bundle.lm_head_path}")


if __name__ == "__main__":
    main()
