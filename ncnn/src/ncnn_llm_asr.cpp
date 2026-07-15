#include "ncnn_llm_asr.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>

using nlohmann::json;

ncnn_llm_asr::ncnn_llm_asr(const std::string& model_path, bool use_vulkan, int num_threads)
    : ncnn_llm_base(use_vulkan, num_threads > 0 ? num_threads : 4) {
    try {
        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            if (!ifs) {
                printf("[ncnn_llm_asr] cannot open %s/model.json\n", model_path.c_str());
                ok_ = false;
                return;
            }
            ifs >> config;
        }
        if (config.contains("model_type")) {
            model_type_ = config["model_type"].get<std::string>();
        }

        mel_net_ = std::make_shared<ncnn::Net>();
        audio_conv_net_ = std::make_shared<ncnn::Net>();
        audio_encoder_net_ = std::make_shared<ncnn::Net>();
        text_embed_net_ = std::make_shared<ncnn::Net>();
        text_decoder_net_ = std::make_shared<ncnn::Net>();
        lm_head_net_ = std::make_shared<ncnn::Net>();

        std::shared_ptr<ncnn::Net> nets[] = {mel_net_, audio_conv_net_, audio_encoder_net_,
                                             text_embed_net_, text_decoder_net_, lm_head_net_};
        for (auto& n : nets) {
            if (num_threads > 0) n->opt.num_threads = num_threads;
            // Pure fp32 CPU path: the Qwen3-ASR residual stream is large-magnitude, keep full precision.
            n->opt.use_fp16_packed = false;
            n->opt.use_fp16_storage = false;
            n->opt.use_fp16_arithmetic = false;
            n->opt.use_bf16_storage = false;
        }

        auto p = [&](const char* key) {
            return model_path + "/" + config["params"][key].get<std::string>();
        };
        if (mel_net_->load_param(p("mel_param").c_str()) != 0 ||
            mel_net_->load_model(p("mel_bin").c_str()) != 0 ||
            audio_conv_net_->load_param(p("audio_conv_param").c_str()) != 0 ||
            audio_conv_net_->load_model(p("audio_conv_bin").c_str()) != 0 ||
            audio_encoder_net_->load_param(p("audio_encoder_param").c_str()) != 0 ||
            audio_encoder_net_->load_model(p("audio_encoder_bin").c_str()) != 0 ||
            text_embed_net_->load_param(p("text_embed_param").c_str()) != 0 ||
            text_embed_net_->load_model(p("text_embed_bin").c_str()) != 0 ||
            text_decoder_net_->load_param(p("text_decoder_param").c_str()) != 0 ||
            text_decoder_net_->load_model(p("text_decoder_bin").c_str()) != 0 ||
            lm_head_net_->load_param(p("lm_head_param").c_str()) != 0 ||
            lm_head_net_->load_model(p("lm_head_bin").c_str()) != 0) {
            printf("[ncnn_llm_asr] failed to load one or more nets\n");
            ok_ = false;
            return;
        }

        // ---- tokenizer ----
        std::string type = config["tokenizer"].value("type", std::string("bbpe"));
        std::string vocab_file = model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
        std::string merges_file = model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();
        bpe_ = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
            vocab_file, merges_file, SpecialTokensConfig{}, false, true, type == "bbpe"));
        auto add_tokens = config["tokenizer"]["additional_special_tokens"].get<std::vector<std::string>>();
        for (const auto& t : add_tokens) bpe_->AddAdditionalSpecialToken(t);
        for (int id : bpe_->additional_special_token_ids()) additional_special_id_set_.insert(id);
        if (config["tokenizer"].contains("eos_ids")) {
            for (auto& v : config["tokenizer"]["eos_ids"]) eos_ids_.insert(v.get<int>());
        }

        // ---- settings ----
        auto& s = config["setting"];
        attn_cnt_ = s.value("attn_cnt", attn_cnt_);
        hidden_size_ = s.value("hidden_size", hidden_size_);
        head_dim_ = s.value("head_dim", head_dim_);
        vocab_size_ = s.value("vocab_size", vocab_size_);
        if (s.contains("rope")) rope_theta_ = s["rope"].value("rope_theta", rope_theta_);
        audio_pad_id_ = s.value("audio_pad_token_id", audio_pad_id_);
        audio_start_id_ = s.value("audio_start_token_id", audio_start_id_);
        audio_end_id_ = s.value("audio_end_token_id", audio_end_id_);
        asr_text_id_ = s.value("asr_text_token_id", asr_text_id_);
        special_id_begin_ = s.value("special_token_id_begin", special_id_begin_);
        if (s.contains("audio")) {
            auto& a = s["audio"];
            sample_rate_ = a.value("sampling_rate", sample_rate_);
            n_fft_ = a.value("n_fft", n_fft_);
            hop_length_ = a.value("hop_length", hop_length_);
            n_mels_ = a.value("n_mels", n_mels_);
            chunk_frames_ = a.value("chunk_frames", chunk_frames_);
            tokens_per_chunk_ = a.value("tokens_per_chunk", tokens_per_chunk_);
        }

        // Prompt template (measured against the HF processor / apply_chat_template):
        //   <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
        //     [ L x <|audio_pad|> ]
        //   <|audio_end|><|im_end|>\n<|im_start|>assistant\n
        prefix_ids_ = {151644, 8948, 198, 151645, 198, 151644, 872, 198, audio_start_id_};
        suffix_ids_ = {audio_end_id_, 151645, 198, 151644, 77091, 198};

        printf("[ncnn_llm_asr] loaded %s: attn_cnt=%d hidden=%d head_dim=%d theta=%g vocab=%d\n",
               model_type_.c_str(), attn_cnt_, hidden_size_, head_dim_, rope_theta_, vocab_size_);
    } catch (const std::exception& e) {
        printf("[ncnn_llm_asr] init error: %s\n", e.what());
        ok_ = false;
    }
}

int ncnn_llm_asr::feat_out_len(int n) {
    // Matches export_modules.feat_out_len, which uses Python floor division.
    auto fdiv = [](int a, int b) {
        int q = a / b;
        if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
        return q;
    };
    int leave = n % 100;
    int fl = fdiv(leave - 1, 2) + 1;
    return fdiv(fdiv(fl - 1, 2) + 1 - 1, 2) + 1 + (n / 100) * 13;
}

ncnn::Mat ncnn_llm_asr::wav_to_logmel(const float* pcm, int n) const {
    ncnn::Mat audio(n);
    memcpy(audio.data, pcm, (size_t)n * sizeof(float));

    ncnn::Mat mel;  // raw power mel, (w = frames, h = n_mels)
    {
        ncnn::Extractor ex = mel_net_->create_extractor();
        ex.input("in0", audio);
        ex.extract("out0", mel);
    }

    int F = n / hop_length_;      // drop the trailing center-pad frame (matches Whisper stft[...,:-1])
    if (F > mel.w) F = mel.w;
    if (F < 0) F = 0;

    ncnn::Mat out(F, n_mels_);    // (w = F, h = n_mels)
    float gmax = -1e30f;
    for (int i = 0; i < n_mels_; i++) {
        const float* src = mel.row(i);
        float* dst = out.row(i);
        for (int j = 0; j < F; j++) {
            float v = src[j];
            if (v < 1e-10f) v = 1e-10f;
            v = std::log10(v);
            dst[j] = v;
            if (v > gmax) gmax = v;
        }
    }
    const float floor_v = gmax - 8.0f;
    for (int i = 0; i < n_mels_; i++) {
        float* dst = out.row(i);
        for (int j = 0; j < F; j++) {
            float v = dst[j];
            if (v < floor_v) v = floor_v;
            dst[j] = (v + 4.0f) / 4.0f;
        }
    }
    return out;
}

ncnn::Mat ncnn_llm_asr::run_audio(const ncnn::Mat& logmel) const {
    const int F = logmel.w;
    int L = feat_out_len(F);
    const int D = (F + chunk_frames_ - 1) / chunk_frames_;

    // Per 100-frame chunk -> audio_conv -> (w = 896, h = tokens_per_chunk).
    std::vector<ncnn::Mat> toks;
    toks.reserve(D);
    int conv_w = 0;
    for (int c = 0; c < D; c++) {
        ncnn::Mat chunk(chunk_frames_, n_mels_);  // (w = 100, h = 128) single channel
        chunk.fill(0.0f);
        const int s = c * chunk_frames_;
        const int cnt = std::min(chunk_frames_, F - s);
        for (int i = 0; i < n_mels_; i++) {
            const float* src = logmel.row(i);
            float* dst = chunk.row(i);
            if (cnt > 0) memcpy(dst, src + s, (size_t)cnt * sizeof(float));
        }
        ncnn::Mat convout;
        {
            ncnn::Extractor ex = audio_conv_net_->create_extractor();
            ex.input("in0", chunk);
            ex.extract("out0", convout);  // (w = 896, h = 13)
        }
        conv_w = convout.w;
        toks.push_back(convout);
    }

    // Concat over the token axis, then slice to L real tokens.
    ncnn::Mat hidden(conv_w, L);
    int written = 0;
    for (auto& t : toks) {
        for (int r = 0; r < t.h && written < L; r++, written++) {
            memcpy(hidden.row(written), t.row(r), (size_t)conv_w * sizeof(float));
        }
    }

    ncnn::Mat audio_emb;  // (w = 1024, h = L)
    {
        ncnn::Extractor ex = audio_encoder_net_->create_extractor();
        ex.input("in0", hidden);
        ex.extract("out0", audio_emb);
    }
    return audio_emb;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_asr::prefill(const std::vector<float>& pcm) {
    ncnn::Mat logmel = wav_to_logmel(pcm.data(), (int)pcm.size());  // (F, 128)
    ncnn::Mat audio_emb = run_audio(logmel);                         // (1024, L)
    const int L = audio_emb.h;
    printf("[ncnn_llm_asr] audio: samples=%d, mel_frames=%d, audio_tokens=%d\n",
           (int)pcm.size(), logmel.w, L);

    // Assemble prompt ids: prefix + L audio-pad + suffix.
    std::vector<int> token_ids;
    token_ids.reserve(prefix_ids_.size() + L + suffix_ids_.size());
    for (int t : prefix_ids_) token_ids.push_back(t);
    const int first_audio_index = (int)token_ids.size();
    for (int i = 0; i < L; i++) token_ids.push_back(audio_pad_id_);
    for (int t : suffix_ids_) token_ids.push_back(t);
    const int seq_len = (int)token_ids.size();

    // Text embeddings, then overwrite the audio-pad slots with the encoder output.
    ncnn::Mat token_embed = llm_run_text_embed(*text_embed_net_, token_ids);  // (1024, seq_len)
    for (int i = 0; i < L; i++) {
        memcpy(token_embed.row(first_audio_index + i), audio_emb.row(i),
               (size_t)hidden_size_ * sizeof(float));
    }

    // Plain RoPE cache, positions 0..seq_len-1, half width (head_dim/2, seq_len).
    ncnn::Mat cos_cache, sin_cache;
    generate_rope_embed_cache(seq_len, head_dim_, 0, cos_cache, sin_cache, rope_theta_);

    // Full causal mask (seq_len, seq_len): 0 on/under diagonal, large negative above.
    ncnn::Mat mask(seq_len, seq_len);
    mask.fill(0.0f);
    for (int i = 0; i < seq_len; i++) {
        float* row = mask.row(i);
        for (int j = i + 1; j < seq_len; j++) row[j] = -1e38f;
    }

    KVCache kv_cache;
    ncnn::Mat decode_out = llm_run_decoder_with_kv(*text_decoder_net_, token_embed, mask,
                                                   cos_cache, sin_cache, kv_cache, attn_cnt_, true);
    ncnn::Mat last_hidden = decode_out.row_range(seq_len - 1, 1);
    ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, last_hidden);
    int next_token_id = argmax1d(logits);

    auto ctx = std::make_shared<ncnn_llm_gpt_base_ctx>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;
    ctx->position_id = seq_len;
    return ctx;
}

std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_asr::generate(
    const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in,
    const GenerateConfig& cfg,
    std::function<void(const std::string&)> callback,
    std::vector<int>* out_ids) {

    auto ctx = ctx_in->clone();
    std::unordered_set<int> history;
    history.insert(ctx->cur_token);

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        int tok = ctx->cur_token;
        if (eos_ids_.count(tok)) break;

        if (out_ids) out_ids->push_back(tok);

        bool is_special = (tok >= special_id_begin_) ||
                          (additional_special_id_set_.find(tok) != additional_special_id_set_.end());
        if (!is_special && callback) {
            std::string token_text = bpe_->decode({tok}, true);
            if (!token_text.empty()) callback(token_text);
        }

        // Single-token embedding + single-position RoPE.
        ncnn::Mat cur_embed = llm_run_text_embed(*text_embed_net_, tok);
        ncnn::Mat cos_cache, sin_cache;
        generate_rope_embed_cache(1, head_dim_, ctx->position_id, cos_cache, sin_cache, rope_theta_);
        ctx->position_id++;

        // Incremental mask [1, kv_len+1]: attend to all cached positions + itself.
        ncnn::Mat mask(1, ctx->kv_cache[0].first.h + 1);
        mask.fill(0.0f);

        ncnn::Mat decode_out = llm_run_decoder_with_kv(*text_decoder_net_, cur_embed, mask,
                                                       cos_cache, sin_cache, ctx->kv_cache,
                                                       attn_cnt_, false);
        ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, decode_out);

        LlmTokenSampleConfig sample_cfg;
        sample_cfg.vocab_size = vocab_size_;
        sample_cfg.temperature = cfg.temperature;
        sample_cfg.top_p = cfg.top_p;
        sample_cfg.top_k = cfg.top_k;
        sample_cfg.repetition_penalty = cfg.repetition_penalty;
        sample_cfg.do_sample = cfg.do_sample;
        int next_id = llm_select_next_token(logits, history, sample_cfg);

        ctx->cur_token = next_id;
        history.insert(next_id);
    }
    return ctx;
}

std::string ncnn_llm_asr::transcribe(const std::vector<float>& pcm,
                                     const GenerateConfig& cfg,
                                     std::string* language_out) {
    auto ctx = prefill(pcm);
    std::vector<int> gen_ids;
    generate(ctx, cfg, nullptr, &gen_ids);

    // Output layout: <language> <asr_text> <transcription>.
    auto it = std::find(gen_ids.begin(), gen_ids.end(), asr_text_id_);
    std::vector<int> lang_ids, text_ids;
    if (it != gen_ids.end()) {
        lang_ids.assign(gen_ids.begin(), it);
        text_ids.assign(it + 1, gen_ids.end());
    } else {
        text_ids = gen_ids;
    }
    if (language_out) {
        std::string lang = bpe_->decode(lang_ids, true);
        // Trim, then strip the "language " prefix the model emits (see parse_asr_output).
        size_t b = lang.find_first_not_of(" \t\r\n");
        size_t e = lang.find_last_not_of(" \t\r\n");
        lang = (b == std::string::npos) ? "" : lang.substr(b, e - b + 1);
        const std::string prefix = "language ";
        if (lang.size() >= prefix.size()) {
            std::string head = lang.substr(0, prefix.size());
            std::transform(head.begin(), head.end(), head.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            if (head == prefix) {
                lang = lang.substr(prefix.size());
                size_t nb = lang.find_first_not_of(" \t\r\n");
                lang = (nb == std::string::npos) ? "" : lang.substr(nb);
            }
        }
        *language_out = lang;
    }
    return bpe_->decode(text_ids, true);
}
