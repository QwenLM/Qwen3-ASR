# coding=utf-8
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
"""Experimental bounded-context Qwen3-ASR streaming for the Transformers backend.

This backend accepts a Hugging Face ``Qwen3ASRModel.from_pretrained`` instance.
It intentionally does not use the vLLM ``Qwen3ASRModel.LLM`` wrapper because
that wrapper does not expose the audio-tower outputs or decoder KV needed for
incremental reuse.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import math
import time
import unicodedata
from typing import Any, List, Optional, Tuple

import numpy as np
import torch


SAMPLE_RATE = 16_000
UPDATE_SECONDS = 2.0
_END_IDS = {151645, 151643}
_PUNCTUATION = set("，。！？、；：“”‘’\"'")


@dataclass(frozen=True)
class IncrementalKVUpdate:
    """Complete transcript snapshot emitted on one 2-second update tick."""

    text: str
    window_text: str
    audio_start_sec: float
    audio_end_sec: float
    window_duration_sec: float
    decode_wall_sec: float
    final: bool = False
    epoch_index: int = 0
    kv_tokens: int = 0
    audio_kv_tokens: int = 0
    cache_rebuilt: bool = False
    feature_invalidated_blocks: Tuple[int, ...] = ()
    language: str = ""


def _positions(start: int, length: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(start, start + length, device=device).view(1, 1, -1)
    return pos.expand(3, 1, -1)


def _append_embeds(thinker: Any, embeddings: torch.Tensor, cache: Any, start: int):
    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(0)
    output = thinker(
        inputs_embeds=embeddings,
        position_ids=_positions(start, embeddings.shape[1], embeddings.device),
        past_key_values=cache,
        use_cache=True,
    )
    return output.past_key_values, output.logits[:, -1, :]


def _greedy_decode(thinker, cache, logits, max_new_tokens: int, next_position: int):
    generated: list[int] = []
    for _ in range(max_new_tokens):
        token = logits.argmax(dim=-1, keepdim=True)
        token_id = int(token.item())
        generated.append(token_id)
        if token_id in _END_IDS:
            break
        output = thinker(
            input_ids=token,
            position_ids=_positions(next_position, 1, token.device),
            past_key_values=cache,
            use_cache=True,
        )
        next_position += 1
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
    return generated


def _normalized_with_positions(text: str) -> tuple[str, list[int]]:
    kept = [(ch, index) for index, ch in enumerate(text) if not ch.isspace() and ch not in _PUNCTUATION]
    return "".join(ch for ch, _ in kept), [index for _, index in kept]


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x3040 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
    )


def _join_text(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if unicodedata.category(right[0]).startswith("P"):
        return left + right
    if _is_cjk(left[-1]) or _is_cjk(right[0]):
        return left + right
    if left[-1].isalnum() and right[0].isalnum():
        return left + " " + right
    return left + right


def _merge_overlap(committed: str, current: str) -> str:
    """Merge the 2-second epoch overlap while preserving display formatting."""
    if not committed:
        return current.strip()
    if not current:
        return committed.strip()
    left, _ = _normalized_with_positions(committed)
    right, right_positions = _normalized_with_positions(current)
    if not left or not right:
        return _join_text(committed, current)
    max_overlap = min(24, len(left), len(right))
    matched_prefix = 0
    for size in range(max_overlap, 2, -1):
        if left[-size:] == right[:size]:
            matched_prefix = size
            break
    if not matched_prefix:
        left_tail = left[-max_overlap:]
        right_head = right[:max_overlap]
        matcher = SequenceMatcher(None, left_tail, right_head, autojunk=False)
        candidates = [
            block
            for block in matcher.get_matching_blocks()
            if block.size >= 3
            and block.b <= 1
            and len(left_tail) - (block.a + block.size) <= 2
        ]
        if candidates:
            block = max(candidates, key=lambda item: (item.size, -item.b))
            matched_prefix = block.b + block.size
    if not matched_prefix:
        return _join_text(committed, current)

    cut = right_positions[matched_prefix - 1] + 1
    remainder = current[cut:]
    while remainder and (remainder[0].isspace() or remainder[0] in _PUNCTUATION):
        remainder = remainder[1:]
    return _join_text(committed, remainder)


class TransformersIncrementalStreamingASR:
    """Streaming Qwen3-ASR with fixed context and reusable encoder/decoder KV.

    ``asr`` must be a Hugging Face Qwen3ASRModel loaded with
    ``Qwen3ASRModel.from_pretrained``. Input is 16 kHz mono PCM (float in
    [-1, 1] or int16). The full transcript is returned on every update; words
    inside the active 30-second epoch can be revised.

    The update cadence is intentionally fixed at 2 seconds. ``window_seconds``
    and ``overlap_seconds`` control the bounded audio context independently.
    The validated profile is 30 seconds of context with 2 seconds of overlap.
    """

    def __init__(
        self,
        asr: Any,
        *,
        language: Optional[str] = None,
        context: str = "",
        window_seconds: float = 30.0,
        update_seconds: float = UPDATE_SECONDS,
        overlap_seconds: float = 2.0,
        rollback_tokens: int = 8,
        max_new_tokens: int = 512,
        feature_mean_threshold: float = 0.10,
        feature_changed_fraction_threshold: float = 0.90,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Qwen3-ASR streaming expects {SAMPLE_RATE} Hz audio")
        if abs(update_seconds - UPDATE_SECONDS) > 1e-6:
            raise ValueError("ASR update period is fixed at 2 seconds")
        if window_seconds <= overlap_seconds or overlap_seconds < update_seconds:
            raise ValueError("window_seconds must exceed overlap_seconds >= update_seconds")
        stride = window_seconds - overlap_seconds
        for value, name in ((window_seconds, "window_seconds"), (overlap_seconds, "overlap_seconds"), (stride, "epoch stride")):
            if abs(value / update_seconds - round(value / update_seconds)) > 1e-6:
                raise ValueError(f"{name} must be a multiple of the 2-second update period")
        if rollback_tokens < 0 or max_new_tokens < 1:
            raise ValueError("rollback_tokens must be nonnegative and max_new_tokens positive")

        self.asr = asr
        self.model = asr.model.eval()
        self.thinker = self.model.thinker
        self.language = language
        self.context = context or ""
        self.window_seconds = float(window_seconds)
        self.update_seconds = UPDATE_SECONDS
        self.overlap_seconds = float(overlap_seconds)
        self.epoch_stride_seconds = stride
        self.rollback_tokens = rollback_tokens
        self.max_new_tokens = max_new_tokens
        self.feature_mean_threshold = feature_mean_threshold
        self.feature_changed_fraction_threshold = feature_changed_fraction_threshold
        self.sample_rate = sample_rate
        self.window_samples = int(round(window_seconds * sample_rate))
        self.stride_samples = int(round(stride * sample_rate))
        self.hop_samples = int(round(UPDATE_SECONDS * sample_rate))
        self.device = self.model.device
        self.dtype = self.model.dtype
        self.audio_token_id = asr.processor.tokenizer.convert_tokens_to_ids(asr.processor.audio_token)
        self.block_frames = int(self.thinker.audio_tower.config.n_window_infer)
        self.prompt = asr._build_text_prompt(context=self.context, force_language=self.language)

        self._ring = np.empty((0,), dtype=np.float32)
        self._pending = np.empty((0,), dtype=np.float32)
        self._total_samples = 0
        self._closed = False
        self._last_update: Optional[IncrementalKVUpdate] = None
        self._text = ""
        self._language = ""
        self._committed_text = ""
        self._epoch_index: Optional[int] = None
        self._epoch_start_sample = 0
        self._epoch_text = ""
        self._reset_epoch_state()
        self.update_count = 0
        self.total_encoder_input_feature_frames = 0
        self.total_naive_prefix_replay_feature_frames = 0
        self.feature_invalidation_count = 0
        self.decoder_kv_rebuild_count = 0
        self.max_audio_kv_tokens = 0
        self.max_persistent_kv_tokens = 0

    def _reset_epoch_state(self) -> None:
        self.prefix_ids = None
        self.prefix_embeds = None
        self.suffix_embeds = None
        self.prefix_len = 0
        self.persistent_cache = None
        self.finalized_frames = 0
        self.finalized_tokens = 0
        self.cached_feature_blocks: list[torch.Tensor] = []
        self.cached_audio_embeddings: list[torch.Tensor] = []
        self.prior_generated: list[int] = []

    @property
    def text(self) -> str:
        return self._text

    @property
    def language(self) -> str:
        return self._language

    @property
    def buffered_audio_seconds(self) -> float:
        return len(self._ring) / self.sample_rate

    @property
    def audio_seconds(self) -> float:
        return (self._total_samples + len(self._pending)) / self.sample_rate

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def warmup(self, seconds: float = 2.0) -> float:
        """Warm model kernels before live audio; this audio is discarded."""
        if self._closed:
            raise RuntimeError("Cannot warm up after finish()")
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        silence = np.zeros(int(round(seconds * self.sample_rate)), dtype=np.float32)
        batch = self.asr.processor(text=[self.prompt], audio=[silence], return_tensors="pt")
        self._sync()
        started = time.perf_counter()
        with torch.inference_mode():
            self.model.generate(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                input_features=batch["input_features"].to(device=self.device, dtype=self.dtype),
                feature_attention_mask=batch["feature_attention_mask"].to(self.device),
                max_new_tokens=4,
                do_sample=False,
            )
        self._sync()
        return time.perf_counter() - started

    def feed(self, pcm16k: np.ndarray) -> List[IncrementalKVUpdate]:
        if self._closed:
            raise RuntimeError("Cannot feed audio after finish()")
        if pcm16k is None:
            raise ValueError("pcm16k must not be None")
        x = np.asarray(pcm16k)
        if x.ndim > 1:
            if x.shape[-1] == 1:
                x = x.reshape(-1)
            else:
                raise ValueError("Expected mono audio; downmix stereo before calling feed()")
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        updates: list[IncrementalKVUpdate] = []
        cursor = 0
        while cursor < len(x):
            needed = self.hop_samples - len(self._pending)
            take = min(needed, len(x) - cursor)
            self._pending = np.concatenate((self._pending, x[cursor : cursor + take]))
            cursor += take
            if len(self._pending) == self.hop_samples:
                self._append_audio(self._pending)
                self._pending = np.empty((0,), dtype=np.float32)
                updates.append(self._decode_update(final=False))
        return updates

    def finish(self) -> IncrementalKVUpdate:
        if self._last_update is not None and not len(self._pending):
            self._closed = True
            final = replace(self._last_update, final=True, decode_wall_sec=0.0)
            self._last_update = final
            return final
        if self._closed:
            raise RuntimeError("Stream is already closed")
        if len(self._pending):
            self._append_audio(self._pending)
            self._pending = np.empty((0,), dtype=np.float32)
        self._closed = True
        if not len(self._ring):
            final = IncrementalKVUpdate("", "", 0.0, 0.0, 0.0, 0.0, final=True)
            self._last_update = final
            return final
        final = self._decode_update(final=True)
        self._last_update = final
        return final

    def _append_audio(self, samples: np.ndarray) -> None:
        n = len(samples)
        self._total_samples += n
        if n >= self.window_samples:
            self._ring = np.array(samples[-self.window_samples :], dtype=np.float32, copy=True)
        else:
            self._ring = np.concatenate((self._ring, samples))[-self.window_samples :].copy()

    def _epoch_for_time(self, end_sec: float) -> int:
        # Keep the old epoch through its 30s endpoint; switch at the next
        # 2-second tick. This yields one decode per global tick and preserves
        # exactly 2 seconds of audio overlap between adjacent epochs.
        epochs_after_first = math.ceil((end_sec - self.window_seconds - 1e-6) / self.epoch_stride_seconds)
        return max(0, int(epochs_after_first))

    def _decode_update(self, *, final: bool) -> IncrementalKVUpdate:
        from qwen_asr.inference.utils import parse_asr_output

        total_samples = self._total_samples
        end_sec = total_samples / self.sample_rate
        window_start_sample = total_samples - len(self._ring)
        window_start_sec = window_start_sample / self.sample_rate
        epoch_index = self._epoch_for_time(end_sec)
        epoch_start_sample = epoch_index * self.stride_samples
        epoch_start_sec = epoch_start_sample / self.sample_rate
        if epoch_start_sample < window_start_sample:
            raise RuntimeError("The ring buffer no longer contains the active epoch overlap")
        offset = epoch_start_sample - window_start_sample
        epoch_audio = self._ring[offset:].copy()
        epoch_duration = len(epoch_audio) / self.sample_rate

        if self._epoch_index is None:
            self._epoch_index = epoch_index
            self._epoch_start_sample = epoch_start_sample
        elif epoch_index != self._epoch_index:
            self._committed_text = _merge_overlap(self._committed_text, self._epoch_text)
            self._epoch_index = epoch_index
            self._epoch_start_sample = epoch_start_sample
            self._epoch_text = ""
            self._reset_epoch_state()

        self._sync()
        tick_start = time.perf_counter()
        batch = self.asr.processor(text=[self.prompt], audio=[epoch_audio], return_tensors="pt")
        current_ids = batch["input_ids"].to(self.device)
        feature_len = int(batch["feature_attention_mask"].sum().item())
        current_features = batch["input_features"][0, :, :feature_len].float().cpu().contiguous()

        if self.prefix_ids is None:
            audio_mask = current_ids == self.audio_token_id
            audio_positions = audio_mask[0].nonzero(as_tuple=False).flatten()
            first_audio = int(audio_positions[0].item())
            after_audio = int(audio_positions[-1].item()) + 1
            self.prefix_ids = current_ids[:, :first_audio]
            suffix_ids = current_ids[:, after_audio:]
            self.prefix_len = self.prefix_ids.shape[1]
            with torch.inference_mode():
                self.prefix_embeds = self.thinker.get_input_embeddings()(self.prefix_ids)
                self.suffix_embeds = self.thinker.get_input_embeddings()(suffix_ids)
                self.persistent_cache, _ = _append_embeds(self.thinker, self.prefix_embeds, None, 0)

        full_blocks_end = (feature_len // self.block_frames) * self.block_frames
        self.total_naive_prefix_replay_feature_frames += feature_len
        changed_blocks: list[int] = []
        for block_index, old_features in enumerate(self.cached_feature_blocks):
            start = block_index * self.block_frames
            new_features = current_features[:, start : start + self.block_frames]
            delta = (new_features - old_features).abs()
            mean_delta = float(delta.mean().item())
            changed_fraction = float((delta > 1e-3).float().mean().item())
            if (
                mean_delta > self.feature_mean_threshold
                or changed_fraction > self.feature_changed_fraction_threshold
            ):
                changed_blocks.append(block_index)

        cache_rebuilt = False
        encoder_frames_this_update = 0
        with torch.inference_mode():
            if changed_blocks:
                self.feature_invalidation_count += 1
                self.prior_generated = []
                for block_index in changed_blocks:
                    start = block_index * self.block_frames
                    block_cpu = current_features[:, start : start + self.block_frames].contiguous()
                    block = block_cpu.to(device=self.device, dtype=self.dtype)
                    self.cached_audio_embeddings[block_index] = self.thinker.audio_tower(
                        block,
                        feature_lens=torch.tensor([block.shape[-1]], device=self.device),
                    ).last_hidden_state
                    self.cached_feature_blocks[block_index] = block_cpu.clone()
                    encoder_frames_this_update += int(block.shape[-1])

                self.persistent_cache, _ = _append_embeds(
                    self.thinker, self.prefix_embeds, None, 0
                )
                self.finalized_tokens = 0
                for encoded in self.cached_audio_embeddings:
                    self.persistent_cache, _ = _append_embeds(
                        self.thinker,
                        encoded,
                        self.persistent_cache,
                        self.prefix_len + self.finalized_tokens,
                    )
                    self.finalized_tokens += int(encoded.shape[0])
                self.finalized_frames = len(self.cached_feature_blocks) * self.block_frames
                cache_rebuilt = True
                self.decoder_kv_rebuild_count += 1

            while self.finalized_frames < full_blocks_end:
                start = self.finalized_frames
                block_cpu = current_features[:, start : start + self.block_frames].contiguous()
                block = block_cpu.to(device=self.device, dtype=self.dtype)
                encoded = self.thinker.audio_tower(
                    block,
                    feature_lens=torch.tensor([block.shape[-1]], device=self.device),
                ).last_hidden_state
                self.cached_feature_blocks.append(block_cpu.clone())
                self.cached_audio_embeddings.append(encoded)
                encoder_frames_this_update += int(block.shape[-1])
                self.persistent_cache, _ = _append_embeds(
                    self.thinker,
                    encoded,
                    self.persistent_cache,
                    self.prefix_len + self.finalized_tokens,
                )
                self.finalized_tokens += int(encoded.shape[0])
                self.finalized_frames += int(block.shape[-1])

            active_frames = feature_len - self.finalized_frames
            active = None
            if active_frames:
                active_features = current_features[:, self.finalized_frames : feature_len].to(
                    device=self.device, dtype=self.dtype
                )
                active = self.thinker.audio_tower(
                    active_features,
                    feature_lens=torch.tensor([active_frames], device=self.device),
                ).last_hidden_state
                encoder_frames_this_update += active_frames

            branch_cache = deepcopy(self.persistent_cache)
            branch_start = self.prefix_len + self.finalized_tokens
            active_tokens = int(active.shape[0]) if active is not None else 0
            if active is not None:
                branch_cache, _ = _append_embeds(self.thinker, active, branch_cache, branch_start)
            audio_kv_tokens = self.finalized_tokens + active_tokens
            self.max_audio_kv_tokens = max(self.max_audio_kv_tokens, audio_kv_tokens)

            prefix_tokens = [token for token in self.prior_generated if token not in _END_IDS]
            rollback = min(self.rollback_tokens, max(0, len(prefix_tokens) - 2))
            if rollback:
                prefix_tokens = prefix_tokens[:-rollback]
            branch_parts = [self.suffix_embeds]
            if prefix_tokens:
                prefix_tensor = torch.tensor([prefix_tokens], device=self.device, dtype=torch.long)
                branch_parts.append(self.thinker.get_input_embeddings()(prefix_tensor))
            suffix_and_prefix = torch.cat(branch_parts, dim=1)
            suffix_start = self.prefix_len + self.finalized_tokens + active_tokens
            branch_cache, logits = _append_embeds(
                self.thinker, suffix_and_prefix, branch_cache, suffix_start
            )
            newly_generated = _greedy_decode(
                self.thinker,
                branch_cache,
                logits,
                self.max_new_tokens,
                suffix_start + suffix_and_prefix.shape[1],
            )

        output_ids = prefix_tokens + newly_generated
        self.prior_generated = output_ids
        decoded = self.asr.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self._language, self._epoch_text = parse_asr_output(decoded)
        self._text = _merge_overlap(self._committed_text, self._epoch_text)
        self._sync()
        latency = time.perf_counter() - tick_start
        self.total_encoder_input_feature_frames += encoder_frames_this_update
        self.update_count += 1
        persistent_kv_tokens = int(self.persistent_cache.get_seq_length())
        self.max_persistent_kv_tokens = max(self.max_persistent_kv_tokens, persistent_kv_tokens)
        update = IncrementalKVUpdate(
            text=self._text,
            window_text=self._epoch_text,
            audio_start_sec=epoch_start_sec,
            audio_end_sec=end_sec,
            window_duration_sec=epoch_duration,
            decode_wall_sec=latency,
            final=final,
            epoch_index=epoch_index,
            kv_tokens=persistent_kv_tokens,
            audio_kv_tokens=audio_kv_tokens,
            cache_rebuilt=cache_rebuilt,
            feature_invalidated_blocks=tuple(changed_blocks),
            language=self._language,
        )
        self._last_update = update
        return update
