#pragma once

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <mat.h>
#include <net.h>

using KVCache = std::vector<std::pair<ncnn::Mat, ncnn::Mat>>;

inline ncnn::Mat mat_from_int_vector(const std::vector<int>& vec) {
    ncnn::Mat m(static_cast<int>(vec.size()), 1, 1);
    std::memcpy(m.data, vec.data(), vec.size() * sizeof(int));
    return m;
}

inline void add_mats_inplace(ncnn::Mat& a, const ncnn::Mat& b) {
    if (a.w != b.w || a.h != b.h || a.c != b.c) {
        return;
    }
    if (a.elemsize != 4 || b.elemsize != 4) {
        return;
    }
    float* pa = a;
    const float* pb = b;
    for (int i = 0; i < a.total(); ++i) {
        pa[i] += pb[i];
    }
}

inline int argmax1d(const ncnn::Mat& m) {
    const float* p = m;
    int max_idx = 0;
    float max_val = p[0];
    for (int i = 1; i < m.w; ++i) {
        if (p[i] > max_val) {
            max_val = p[i];
            max_idx = i;
        }
    }
    return max_idx;
}

inline ncnn::Mat sinusoidal_positional_embedding(int seq_len, int d_model) {
    int half_dim = d_model / 2;
    ncnn::Mat emb(d_model, seq_len);
    emb.fill(0.0f);

    std::vector<float> inv_freq(half_dim);
    double log_10000 = std::log(10000.0);
    double denom_base = static_cast<double>(std::max(1, half_dim));

    for (int i = 0; i < half_dim; ++i) {
        inv_freq[i] = static_cast<float>(std::exp(static_cast<double>(i) * -(log_10000 / denom_base)));
    }

    for (int i = 0; i < seq_len; ++i) {
        float pos = static_cast<float>(i + 1);
        float* row_ptr = emb.row(i);
        for (int j = 0; j < half_dim; ++j) {
            float angle = pos * inv_freq[j];
            row_ptr[j] = std::sin(angle);
            row_ptr[j + half_dim] = std::cos(angle);
        }
    }
    return emb;
}

inline ncnn::Mat sinusoidal_positional_embedding_for_pos(int position, int d_model) {
    int half_dim = d_model / 2;
    ncnn::Mat emb(d_model);
    emb.fill(0.0f);

    std::vector<float> inv_freq(half_dim);
    double log_10000 = std::log(10000.0);
    double denom_base = static_cast<double>(std::max(1, half_dim));

    for (int i = 0; i < half_dim; ++i) {
        inv_freq[i] = static_cast<float>(std::exp(static_cast<double>(i) * -(log_10000 / denom_base)));
    }

    float* emb_ptr = emb;
    for (int j = 0; j < half_dim; ++j) {
        float angle = static_cast<float>(position) * inv_freq[j];
        emb_ptr[j] = std::sin(angle);
        emb_ptr[j + half_dim] = std::cos(angle);
    }
    if (d_model % 2 != 0) {
        emb_ptr[d_model - 1] = 0.0f;
    }
    return emb;
}

struct SampleConfig {
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    bool do_sample = false;
};

class ncnn_llm_base {
protected:
    bool use_vulkan_ = false;
    int num_threads_ = 4;
    bool ok_ = true;
    std::mt19937 rng_{std::random_device{}()};

    ncnn_llm_base(bool use_vulkan = false, int num_threads = 4)
        : use_vulkan_(use_vulkan), num_threads_(num_threads) {
#if NCNN_VULKAN
        if (use_vulkan_) {
            ncnn::create_gpu_instance();
        }
#endif
    }

    virtual ~ncnn_llm_base() {
#if NCNN_VULKAN
        if (use_vulkan_) {
            ncnn::destroy_gpu_instance();
        }
#endif
    }

    ncnn::Option create_option() const {
        ncnn::Option opt;
        opt.num_threads = num_threads_;
        opt.use_bf16_storage = false;
        opt.use_vulkan_compute = use_vulkan_;
        return opt;
    }

    bool load_net(ncnn::Net& net, const std::string& param_path, const std::string& bin_path) {
        if (net.load_param(param_path.c_str()) != 0 ||
            net.load_model(bin_path.c_str()) != 0) {
            ok_ = false;
            return false;
        }
        return true;
    }

    int sample_logits(const ncnn::Mat& logits, const SampleConfig& cfg) {
        if (!cfg.do_sample) {
            return argmax1d(logits);
        }

        std::vector<float> probs(logits.w);
        const float* p = logits;
        for (int i = 0; i < logits.w; ++i) {
            probs[i] = p[i];
        }

        softmax_vec(probs, cfg.temperature);

        if (cfg.top_k > 0) {
            apply_top_k(probs, cfg.top_k);
        }

        if (cfg.top_p < 1.0f) {
            apply_top_p(probs, cfg.top_p);
        }

        return sample_from_probs(probs);
    }

private:
    void softmax_vec(std::vector<float>& logits, float temperature) {
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.f;
        for (float& x : logits) {
            x = std::exp((x - max_logit) / temperature);
            sum += x;
        }
        for (float& x : logits) x /= sum;
    }

    void apply_top_k(std::vector<float>& probs, int k) {
        if (k <= 0 || k >= (int)probs.size()) return;
        std::vector<float> tmp = probs;
        std::nth_element(tmp.begin(), tmp.end() - k, tmp.end());
        float threshold = tmp[tmp.size() - k];
        for (float& p : probs) if (p < threshold) p = 0.f;
    }

    void apply_top_p(std::vector<float>& probs, float p) {
        if (p >= 1.0f) return;
        std::vector<std::pair<float,int>> v;
        v.reserve(probs.size());
        for (int i = 0; i < (int)probs.size(); ++i) {
            v.emplace_back(probs[i], i);
        }
        std::sort(v.begin(), v.end(), std::greater<>());

        float cum = 0.f;
        size_t cutoff = v.size();
        for (size_t i = 0; i < v.size(); ++i) {
            cum += v[i].first;
            if (cum >= p) {
                cutoff = i + 1;
                break;
            }
        }
        std::vector<char> keep(probs.size(), 0);
        for (size_t i = 0; i < cutoff; ++i) {
            keep[v[i].second] = 1;
        }
        for (int i = 0; i < (int)probs.size(); ++i) {
            if (!keep[i]) probs[i] = 0.f;
        }
    }

    int sample_from_probs(const std::vector<float>& probs) {
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return dist(rng_);
    }

public:
    bool ok() const { return ok_; }
};