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
import os
import logging
import soundfile as sf
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import regex
import nagisa
import torch
import torchaudio

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor
import torchaudio.compliance.kaldi as kaldi
from qwen_asr.inference.cluster_backend import ClusterBackend
import onnxruntime

logger = logging.getLogger(__name__)


def _resolve_campplus_model_path(campplus_model: str) -> str:
    if not campplus_model:
        return None
    if os.path.isfile(campplus_model):
        return campplus_model
    if "/" in campplus_model and not os.path.exists(campplus_model):
        parts = campplus_model.split("/")
        if len(parts) >= 2:
            repo_id = "/".join(parts[:-1])
            filename = parts[-1]
            try:
                from modelscope.hub.file_download import model_file_download
                cache_path = model_file_download(
                    model_id=repo_id,
                    file_path=filename,
                )
                logger.info(f"Downloaded campplus model from ModelScope: {campplus_model} -> {cache_path}")
                return cache_path
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to download from ModelScope: {e}")
            try:
                from huggingface_hub import hf_hub_download
                cache_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                )
                logger.info(f"Downloaded campplus model from HuggingFace: {campplus_model} -> {cache_path}")
                return cache_path
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to download from HuggingFace: {e}")
            raise FileNotFoundError(
                f"Cannot find campplus model at '{campplus_model}'. "
                f"Please install modelscope (pip install modelscope) or huggingface_hub (pip install huggingface_hub), "
                f"or provide a valid local path."
            )
    return campplus_model
from .utils import (
    AudioLike,
    ensure_list,
    normalize_audios,
    SAMPLE_RATE,
)


class Qwen3ForceAlignProcessor():
    def __init__(self):
        ko_dict_path = os.path.join(os.path.dirname(__file__), "assets", "korean_dict_jieba.dict")
        ko_scores = {}
        with open(ko_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                word = line.split()[0]
                ko_scores[word] = 1.0
        self.ko_score = ko_scores
        self.ko_tokenizer = None

    def is_kept_char(self, ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            return True
        return False

    def clean_token(self, token: str) -> str:
        return "".join(ch for ch in token if self.is_kept_char(ch))

    def is_cjk_char(self, ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF   # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # Extension A
            or 0x20000 <= code <= 0x2A6DF  # Extension B
            or 0x2A700 <= code <= 0x2B73F  # Extension C
            or 0x2B740 <= code <= 0x2B81F  # Extension D
            or 0x2B820 <= code <= 0x2CEAF  # Extension E
            or 0xF900 <= code <= 0xFAFF    # Compatibility Ideographs
        )

    def tokenize_chinese_mixed(self, text: str) -> List[str]:
        tokens: List[str] = []
        current_latin: List[str] = []

        def flush_latin():
            nonlocal current_latin
            if current_latin:
                token = "".join(current_latin)
                cleaned = self.clean_token(token)
                if cleaned:
                    tokens.append(cleaned)
                current_latin = []

        for ch in text:
            if self.is_cjk_char(ch):
                flush_latin()
                tokens.append(ch)
            else:
                if self.is_kept_char(ch):
                    current_latin.append(ch)
                else:
                    flush_latin()

        flush_latin()

        return tokens

    def tokenize_japanese(self, text: str) -> List[str]:
        words = nagisa.tagging(text).words
        tokens: List[str] = []
        for w in words:
            cleaned = self.clean_token(w)
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def tokenize_korean(self, ko_tokenizer, text: str) -> List[str]:
        raw_tokens = ko_tokenizer.tokenize(text)
        tokens: List[str] = []
        for w in raw_tokens:
            w_clean = self.clean_token(w)
            if w_clean:
                tokens.append(w_clean)
        return tokens

    def split_segment_with_chinese(self, seg: str) -> List[str]:
        tokens: List[str] = []
        buf: List[str] = []

        def flush_buf():
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if self.is_cjk_char(ch):
                flush_buf()
                tokens.append(ch)
            else:
                buf.append(ch)

        flush_buf()
        return tokens

    def tokenize_space_lang(self, text: str) -> List[str]:
        tokens: List[str] = []
        for seg in text.split():
            cleaned = self.clean_token(seg)
            if cleaned:
                tokens.extend(self.split_segment_with_chinese(cleaned))
        return tokens

    def fix_timestamp(self, data) -> List[int]:
        data = data.tolist()
        n = len(data)

        dp = [1] * n
        parent = [-1] * n
        
        for i in range(1, n):
            for j in range(i):
                if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        
        max_length = max(dp)
        max_idx = dp.index(max_length)
        
        lis_indices = []
        idx = max_idx
        while idx != -1:
            lis_indices.append(idx)
            idx = parent[idx]
        lis_indices.reverse()

        is_normal = [False] * n
        for idx in lis_indices:
            is_normal[idx] = True
        
        result = data.copy()
        i = 0
        
        while i < n:
            if not is_normal[i]:
                j = i
                while j < n and not is_normal[j]:
                    j += 1
                
                anomaly_count = j - i
                
                if anomaly_count <= 2:
                    left_val = None
                    for k in range(i - 1, -1, -1):
                        if is_normal[k]:
                            left_val = result[k]
                            break
                    
                    right_val = None
                    for k in range(j, n):
                        if is_normal[k]:
                            right_val = result[k]
                            break
                    
                    for k in range(i, j):
                        if left_val is None:
                            result[k] = right_val
                        elif right_val is None:
                            result[k] = left_val
                        else:
                            result[k] = left_val if (k - (i - 1)) <= ((j) - k) else right_val
                
                else:
                    left_val = None
                    for k in range(i - 1, -1, -1):
                        if is_normal[k]:
                            left_val = result[k]
                            break

                    right_val = None
                    for k in range(j, n):
                        if is_normal[k]:
                            right_val = result[k]
                            break
                    
                    if left_val is not None and right_val is not None:
                        step = (right_val - left_val) / (anomaly_count + 1)
                        for k in range(i, j):
                            result[k] = left_val + step * (k - i + 1)
                    elif left_val is not None:
                        for k in range(i, j):
                            result[k] = left_val
                    elif right_val is not None:
                        for k in range(i, j):
                            result[k] = right_val
                
                i = j
            else:
                i += 1

        return [int(res) for res in result]

    def encode_timestamp(self, text: str, language: str) -> List[str]:
        language = language.lower()

        if language.lower() == "japanese":
            word_list = self.tokenize_japanese(text)
        elif language.lower() == "korean":
            if self.ko_tokenizer is None:
                from soynlp.tokenizer import LTokenizer
                self.ko_tokenizer = LTokenizer(scores=self.ko_score)
            word_list = self.tokenize_korean(self.ko_tokenizer, text)
        else:
            word_list = self.tokenize_space_lang(text)
        
        input_text = "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
        input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text

        return word_list, input_text

    def parse_timestamp(self, word_list, timestamp):
        timestamp_output = []

        timestamp_fixed = self.fix_timestamp(timestamp)
        for i, word in enumerate(word_list):
            start_time = timestamp_fixed[i * 2]
            end_time = timestamp_fixed[i * 2 + 1]
            timestamp_output.append({
                "text": word,
                "start_time": start_time,
                "end_time": end_time
            })
        
        return timestamp_output


@dataclass(frozen=True)
class ForcedAlignItem:
    """
    One aligned item span.

    Attributes:
        text (str):
            The aligned unit (cjk character or word) produced by the forced aligner processor.
        start_time (float):
            Start time in seconds.
        end_time (float):
            End time in seconds.
        speaker (Optional[int]):
            Speaker label (cluster ID) for this segment.
    """
    text: str
    start_time: int
    end_time: int
    speaker: Optional[int] = None


@dataclass(frozen=True)
class ForcedAlignResult:
    """
    Forced alignment output for one sample.

    Attributes:
        items (List[ForcedAlignItem]):
            Aligned token spans.
    """
    items: List[ForcedAlignItem]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> ForcedAlignItem:
        return self.items[idx]


class Qwen3ForcedAligner:
    """
    A HuggingFace-style wrapper for Qwen3-ForcedAligner model inference.

    This wrapper provides:
      - `from_pretrained()` initialization via HuggingFace AutoModel/AutoProcessor
      - audio input normalization (path/URL/base64/(np.ndarray, sr))
      - batch and single-sample forced alignment
      - structured output with attribute access (`.text`, `.start_time`, `.end_time`)
    """

    def __init__(
        self,
        model: Qwen3ASRForConditionalGeneration,
        processor: Qwen3ASRProcessor,
        aligner_processor: Qwen3ForceAlignProcessor,
        campplus_model: str = None,
    ):
        self.model = model
        self.processor = processor
        self.aligner_processor = aligner_processor
        # refer to https://github.com/FunAudioLLM/CosyVoice.git
        if campplus_model:
            resolved_path = _resolve_campplus_model_path(campplus_model)
            self.campplus_model = resolved_path
            self.cb_model = ClusterBackend().to('cpu')
            option = onnxruntime.SessionOptions()
            option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            option.intra_op_num_threads = 1
            self.campplus_session = onnxruntime.InferenceSession(resolved_path, sess_options=option, providers=["CPUExecutionProvider"])

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        self.timestamp_token_id = int(model.config.timestamp_token_id)
        self.timestamp_segment_time = float(model.config.timestamp_segment_time)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "Qwen3ForcedAligner":
        """
        Load Qwen3-ForcedAligner model and initialize processors.

        This method:
          1) Registers config/model/processor for HF auto classes.
          2) Loads the model using `AutoModel.from_pretrained(...)`.
          3) Initializes:
             - HF processor (`AutoProcessor.from_pretrained(...)`)
             - forced alignment text processor (`Qwen3ForceAlignProcessor()`)

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory.
            **kwargs:
                Forwarded to `AutoModel.from_pretrained(...)`.
                Typical examples: device_map="cuda:0", dtype=torch.bfloat16.

        Returns:
            Qwen3ForcedAligner:
                Initialized wrapper instance.
        """
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

        # Extract campplus_model from kwargs before passing to AutoModel
        campplus_model = kwargs.pop('campplus_model', None)

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3ASRForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3ASRForConditionalGeneration."
            )

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True)
        aligner_processor = Qwen3ForceAlignProcessor()

        return cls(model=model, processor=processor, aligner_processor=aligner_processor, campplus_model=campplus_model)

    def _to_structured_items(self, timestamp_output: List[Dict[str, Any]]) -> ForcedAlignResult:
        items: List[ForcedAlignItem] = []
        for it in timestamp_output:
            items.append(
                ForcedAlignItem(
                    text=str(it.get("text", "")),
                    start_time=int(it.get("start_time", 0)),
                    end_time=int(it.get("end_time", 0)),
                )
            )
        return ForcedAlignResult(items=items)
    
    @staticmethod
    def load_wav(wav, target_sr, min_sr=16000):
        if isinstance(wav, str):
            speech, sample_rate = sf.read(wav)
            speech = torch.from_numpy(speech).float().unsqueeze(0)
        else:
            speech = torch.from_numpy(wav).float().unsqueeze(0)
            sample_rate = target_sr
        
        if sample_rate != target_sr:
            assert sample_rate >= min_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech
    
    @torch.inference_mode()
    def align(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        text: Union[str, List[str]],
        language: Union[str, List[str]],
    ) -> List[ForcedAlignResult]:
        """
        Run forced alignment for batch or single sample.

        Args:
            audio:
                Audio input(s). Each item supports:
                  - local path / https URL / base64 string
                  - (np.ndarray, sr)
                All audios will be converted into mono 16k float32 arrays in [-1, 1].
            text:
                Transcript(s) for alignment.
            language:
                Language(s) for each sample (e.g., "Chinese", "English").

        Returns:
            List[ForcedAlignResult]:
                One result per sample. Each result contains `items`, and each token can be accessed via
                `.text`, `.start_time`, `.end_time`.
        """
        texts = ensure_list(text)
        languages = ensure_list(language)
        norm_audio = normalize_audios(audio)

        if len(languages) == 1 and len(norm_audio) > 1:
            languages = languages * len(norm_audio)

        if not (len(norm_audio) == len(texts) == len(languages)):
            raise ValueError(
                f"Batch size mismatch: audio={len(norm_audio)}, text={len(texts)}, language={len(languages)}"
            )

        word_lists = []
        aligner_input_texts = []
        for t, lang in zip(texts, languages):
            word_list, aligner_input_text = self.aligner_processor.encode_timestamp(t, lang)
            word_lists.append(word_list)
            aligner_input_texts.append(aligner_input_text)

        inputs = self.processor(
            text=aligner_input_texts,
            audio=norm_audio,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        logits = self.model.thinker(**inputs).logits
        output_ids = logits.argmax(dim=-1)

        results: List[ForcedAlignResult] = []
        for input_id, output_id, word_list in zip(inputs["input_ids"], output_ids, word_lists):
            masked_output_id = output_id[input_id == self.timestamp_token_id]
            timestamp_ms = (masked_output_id * self.timestamp_segment_time).to("cpu").numpy()
            timestamp_output = self.aligner_processor.parse_timestamp(word_list, timestamp_ms)
            if not self.campplus_model:
                for it in timestamp_output:
                    it['start_time'] = round(it['start_time'] / 1000.0, 3)
                    it['end_time'] = round(it['end_time'] / 1000.0, 3)
            results.append(self._to_structured_items(timestamp_output))
        if not self.campplus_model:
            return results
        
        assert len(results) == len(texts)
        previous_time_offset = 0
        reorganized_audio_ndarray = []
        reorganized_text_segments = []
        # Extract audio segments based on aligned timestamps and punctuation-split text segments
        for text, align_result, audio_ndarray, language in zip(texts, results, norm_audio, languages):

            text_segments_by_punc = regex.split(r'[\p{P}]', text)
            offset = 0
            max_idx = len(align_result) - 1
            previous_time_offset = align_result[-1].end_time + previous_time_offset
            previous_time_start = None
            for seg in text_segments_by_punc:
                if not seg:
                    continue
                seg_len = len(seg)
                start = align_result[min(offset, max_idx)].start_time if not previous_time_start else previous_time_start
                end = align_result[min(offset + seg_len - 1, max_idx)].end_time
                
                start_sample = int(start * (SAMPLE_RATE / 1000))
                end_sample = int(end * (SAMPLE_RATE / 1000))
                
                # kaldi.fbank uses 25ms window by default, skip segments shorter than that
                fbank_window_samples = int(25 / 1000 * SAMPLE_RATE)
                if end_sample < start_sample + fbank_window_samples:
                    previous_time_start = start
                    offset += seg_len
                    continue
                
                
                audio_tensor = audio_ndarray[start_sample:end_sample]
                reorganized_audio_ndarray.append(audio_tensor)
                reorganized_text_segments.append((seg, start, end, language))
                
                offset += seg_len
                previous_time_start = None

        aligner_input_texts = []
        for t, _, _, lang in reorganized_text_segments:
            word_list, aligner_input_text = self.aligner_processor.encode_timestamp(t, lang)
            word_lists.append(word_list)
            aligner_input_texts.append(aligner_input_text)
        
        # Extract speaker embeddings from audio segments and cluster them to assign speaker labels.
        embeddings = []
        valid_segment_indices = []
        for idx, wav in enumerate(reorganized_audio_ndarray):
            speech = self.load_wav(wav, target_sr=SAMPLE_RATE)
            feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=SAMPLE_RATE)
            feat = feat - feat.mean(dim=0, keepdim=True)
            embedding = self.campplus_session.run(None, {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten()
            embeddings.append(torch.from_numpy(embedding))
            valid_segment_indices.append(idx)
        
        if len(embeddings) > 0:
            embeddings = torch.stack(embeddings)
            spk_labels = self.cb_model(embeddings)
        else:
            spk_labels = np.array([], dtype=int)
        
        segment_idx = 0
        current_offset = 0
        final_results = []
        for text, align_result in zip(texts, results):
            new_items = []
            for item in align_result:
                while segment_idx < len(reorganized_text_segments):
                    seg, seg_start, seg_end, _ = reorganized_text_segments[segment_idx]
                    adjusted_start = seg_start - current_offset
                    adjusted_end = seg_end - current_offset
                    if item.start_time >= adjusted_start and item.end_time <= adjusted_end:
                        if segment_idx in valid_segment_indices:
                            label_idx = valid_segment_indices.index(segment_idx)
                            speaker = int(spk_labels[label_idx])
                        else:
                            speaker = None
                        new_item = ForcedAlignItem(
                            text=item.text,
                            start_time=item.start_time,
                            end_time=item.end_time,
                            speaker=speaker
                        )
                        new_items.append(new_item)
                        break
                    elif item.end_time > adjusted_end:
                        segment_idx += 1
                    else:
                        new_item = ForcedAlignItem(
                            text=item.text,
                            start_time=item.start_time,
                            end_time=item.end_time,
                            speaker=None
                        )
                        new_items.append(new_item)
                        break
                else:
                    new_item = ForcedAlignItem(
                        text=item.text,
                        start_time=item.start_time,
                        end_time=item.end_time,
                        speaker=None
                    )
                    new_items.append(new_item)
            if align_result:
                current_offset += align_result[-1].end_time
            final_results.append(ForcedAlignResult(items=new_items))
        
        return final_results
    
    def get_supported_languages(self) -> Optional[List[str]]:
        """
        List supported language names for the current model.

        This is a thin wrapper around `self.model.get_support_languages()`.
        If the underlying model does not expose language constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported language names (lowercased), if available.
                - None if the model does not provide supported languages.
        """
        fn = getattr(self.model, "get_support_languages", None)
        if not callable(fn):
            return None

        langs = fn()
        if langs is None:
            return None

        return sorted({str(x).lower() for x in langs})
