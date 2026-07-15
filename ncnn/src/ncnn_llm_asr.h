#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

#include "ncnn_llm_base.h"
#include "ncnn_llm_gpt.h"
#include "ncnn_text_runtime.h"
#include "utils/tokenizer/bpe_tokenizer.h"
#include "utils/rope_embed.h"

// Qwen3-ASR-0.6B end-to-end ncnn runtime:
//   pcm(16k mono) -> mel(STFT+gemm) -> audio_conv(per 100-frame chunk) -> audio_encoder
//   -> splice into text_embed(audio_pad slots) -> decoder(KV cache) -> lm_head -> greedy decode.
// Reuses the shared text runtime (llm_run_text_embed / llm_run_decoder_with_kv / llm_run_lm_head),
// plain RoPE cache (generate_rope_embed_cache, theta 1e6) and the bbpe tokenizer.
class ncnn_llm_asr : public ncnn_llm_base {
public:
    ncnn_llm_asr(const std::string& model_path, bool use_vulkan = false, int num_threads = 0);

    // Encode audio + prompt, run the decoder prefill, return a context holding the KV cache
    // and the first generated token. `pcm` is mono float PCM in [-1, 1] at 16 kHz.
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::vector<float>& pcm);

    // Autoregressive greedy/sampled decode. Streams decoded non-special token text via
    // `callback` (may be null). If `out_ids` is non-null, every generated token id (until
    // eos, exclusive) is appended to it.
    std::shared_ptr<ncnn_llm_gpt_ctx> generate(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx,
                                               const GenerateConfig& cfg,
                                               std::function<void(const std::string&)> callback,
                                               std::vector<int>* out_ids = nullptr);

    // Convenience: prefill + generate, split the output at <asr_text> into language / text.
    std::string transcribe(const std::vector<float>& pcm,
                           const GenerateConfig& cfg,
                           std::string* language_out = nullptr);

    bool ok() const { return ok_; }
    const std::string& model_type() const { return model_type_; }

    // Post-CNN audio token count for a mel window of `n` frames (matches the HF processor).
    static int feat_out_len(int n);

private:
    // ---- nets ----
    std::shared_ptr<ncnn::Net> mel_net_;
    std::shared_ptr<ncnn::Net> audio_conv_net_;
    std::shared_ptr<ncnn::Net> audio_encoder_net_;
    std::shared_ptr<ncnn::Net> text_embed_net_;
    std::shared_ptr<ncnn::Net> text_decoder_net_;
    std::shared_ptr<ncnn::Net> lm_head_net_;
    std::shared_ptr<BpeTokenizer> bpe_;
    std::unordered_set<int> additional_special_id_set_;

    std::string model_type_ = "qwen3_asr";

    // ---- settings ----
    int attn_cnt_ = 28;
    int hidden_size_ = 1024;
    int head_dim_ = 128;
    float rope_theta_ = 1000000.0f;
    int vocab_size_ = 151936;

    int audio_pad_id_ = 151676;
    int audio_start_id_ = 151669;
    int audio_end_id_ = 151670;
    int asr_text_id_ = 151704;
    int special_id_begin_ = 151643;
    std::unordered_set<int> eos_ids_;

    // audio front-end
    int sample_rate_ = 16000;
    int n_fft_ = 400;
    int hop_length_ = 160;
    int n_mels_ = 128;
    int chunk_frames_ = 100;
    int tokens_per_chunk_ = 13;

    // prompt id template around the L audio-pad tokens
    std::vector<int> prefix_ids_;
    std::vector<int> suffix_ids_;

    // ---- helpers ----
    ncnn::Mat wav_to_logmel(const float* pcm, int n) const;   // -> (w = F frames, h = n_mels)
    ncnn::Mat run_audio(const ncnn::Mat& logmel) const;       // -> (w = hidden_size, h = L)
};
