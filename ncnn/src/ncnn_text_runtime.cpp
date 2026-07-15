#include "ncnn_text_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <utility>

#include "sampling.h"

ncnn::Mat llm_run_text_embed(ncnn::Net& embed_net, const std::vector<int>& input_ids) {
    ncnn::Mat input_ids_mat((int)input_ids.size(), 1, (void*)input_ids.data());
    input_ids_mat = input_ids_mat.clone();

    ncnn::Mat token_embed;
    ncnn::Extractor ex = embed_net.create_extractor();
    ex.input("in0", input_ids_mat);
    ex.extract("out0", token_embed);
    return token_embed;
}

ncnn::Mat llm_run_text_embed(ncnn::Net& embed_net, int token_id) {
    ncnn::Mat input_id_mat(1, 1, (void*)&token_id);
    input_id_mat = input_id_mat.clone();

    ncnn::Mat token_embed;
    ncnn::Extractor ex = embed_net.create_extractor();
    ex.input("in0", input_id_mat);
    ex.extract("out0", token_embed);
    return token_embed;
}

ncnn::Mat llm_run_decoder_with_kv(ncnn::Net& decoder_net,
                                  const ncnn::Mat& embeds,
                                  const ncnn::Mat& mask,
                                  const ncnn::Mat& cos_cache,
                                  const ncnn::Mat& sin_cache,
                                  KVCache& kv_cache,
                                  int attn_cnt,
                                  bool is_prefill) {
    ncnn::Mat decode_out;
    ncnn::Extractor ex = decoder_net.create_extractor();
    ex.input("in0", embeds);
    ex.input("in1", mask);
    ex.input("in2", cos_cache);
    ex.input("in3", sin_cache);

    if (!is_prefill) {
        for (int i = 0; i < attn_cnt; i++) {
            char name_k_in[32], name_v_in[32];
            std::snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
            std::snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
            ex.input(name_k_in, kv_cache[i].first);
            ex.input(name_v_in, kv_cache[i].second);
        }
    }

    for (int i = 0; i < attn_cnt; i++) {
        char name_k_out[32], name_v_out[32];
        std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
        std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
        ncnn::Mat k_cache, v_cache;
        ex.extract(name_k_out, k_cache);
        ex.extract(name_v_out, v_cache);
        if (is_prefill) {
            kv_cache.emplace_back(std::move(k_cache), std::move(v_cache));
        } else {
            kv_cache[i] = std::make_pair(std::move(k_cache), std::move(v_cache));
        }
    }

    ex.extract("out0", decode_out);
    return decode_out;
}

ncnn::Mat llm_run_lm_head(ncnn::Net& lm_head_net, const ncnn::Mat& hidden_states) {
    ncnn::Mat logits;
    ncnn::Extractor ex = lm_head_net.create_extractor();
    ex.input("in0", hidden_states);
    ex.extract("out0", logits);
    return logits;
}

int llm_select_next_token(const ncnn::Mat& logits,
                          const std::unordered_set<int>& history,
                          const LlmTokenSampleConfig& cfg) {
    const int vocab_size = cfg.vocab_size > 0 ? cfg.vocab_size : logits.w;
    std::vector<float> scores(vocab_size);
    std::memcpy(scores.data(), logits.data, sizeof(float) * vocab_size);

    for (int t : history) {
        if (t < 0 || t >= vocab_size) continue;
        if (scores[t] < 0) {
            scores[t] *= cfg.repetition_penalty;
        } else {
            scores[t] /= cfg.repetition_penalty;
        }
    }

    if (cfg.do_sample != 1 || cfg.temperature <= 0.0f) {
        return (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    softmax_vec(scores, cfg.temperature);
    if (cfg.top_k > 0) apply_top_k(scores, cfg.top_k);
    if (cfg.top_p < 1.0f) apply_top_p(scores, cfg.top_p);

    const float sum = std::accumulate(scores.begin(), scores.end(), 0.0f);
    if (!std::isfinite(sum) || sum <= 0.0f) {
        return (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    return sample_from_probs(scores);
}
