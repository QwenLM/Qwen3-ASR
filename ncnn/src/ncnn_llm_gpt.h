#pragma once

#include <array>
#include <cassert>
#include <cstdio>
#include <exception>
#include <functional>
#include <locale>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <unordered_set>
#include <algorithm>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

#include "ncnn_llm_base.h"
#include "utils/tokenizer/bpe_tokenizer.h"
#include "utils/rope_embed.h"
#include "utils/prompt.h"
#include "utils/gdr.h"
#include "utils/image_utils.h"

using nlohmann::json;

struct GenerateConfig {
    int max_new_tokens = 4096;
    float temperature = 0.3f;
    float top_p = 0.8f;
    int top_k = 50;
    float repetition_penalty = 1.1f;
    int do_sample = 1;

    std::function<nlohmann::json(const nlohmann::json&)> tool_callback = nullptr;

    bool debug = false;
};

class ncnn_llm_gpt_ctx {
public:
    virtual ~ncnn_llm_gpt_ctx() = default;

    virtual std::shared_ptr<ncnn_llm_gpt_ctx> clone() const = 0;

    KVCache kv_cache;
    int cur_token = 0;
    int position_id = 0;
};

class ncnn_llm_gpt_base_ctx : public ncnn_llm_gpt_ctx {
public:
    std::shared_ptr<ncnn_llm_gpt_ctx> clone() const override {
        auto dst = std::make_shared<ncnn_llm_gpt_base_ctx>();
        dst->kv_cache.resize(kv_cache.size());
        for (size_t i = 0; i < kv_cache.size(); ++i) {
            dst->kv_cache[i].first = kv_cache[i].first;
            dst->kv_cache[i].second = kv_cache[i].second;
        }
        dst->cur_token = cur_token;
        dst->position_id = position_id;
        return dst;
    }
};

class qwen3_5_ctx : virtual public ncnn_llm_gpt_ctx {
public:
    std::vector<ncnn::Mat> sconv_cache;
    std::vector<ncnn::Mat> gdr_cache;

    std::shared_ptr<ncnn_llm_gpt_ctx> clone() const override {
        auto dst = std::make_shared<qwen3_5_ctx>();
        dst->kv_cache.resize(kv_cache.size());
        for (size_t i = 0; i < kv_cache.size(); ++i) {
            dst->kv_cache[i].first = kv_cache[i].first;
            dst->kv_cache[i].second = kv_cache[i].second;
        }
        dst->sconv_cache = sconv_cache;
        dst->gdr_cache = gdr_cache;
        dst->cur_token = cur_token;
        dst->position_id = position_id;
        return dst;
    }
};

class ncnn_llm_gpt {
private:
    std::shared_ptr<ncnn::Net> decoder_net;
    std::shared_ptr<ncnn::Net> embed_net;
    std::shared_ptr<ncnn::Net> proj_out_net;
    std::shared_ptr<ncnn::Net> vision_embed_patch;
    std::shared_ptr<ncnn::Net> vision_embed_pos;
    std::shared_ptr<ncnn::Net> vision_encoder;
    std::shared_ptr<BpeTokenizer> bpe;

protected:
    std::string model_type;
    int bos = 0;
    int eos = 0;
    int tool_call_id = -1;
    int tool_call_end_id = -1;
    int think_id = -1;
    int think_end_id = -1;
    int attn_cnt = 32;
    int sconv_cnt = 0;
    int gdr_cnt = 0;
    int rope_head_dim = 64;

    enum RoPE_Type {
        RoPE = 0,
        LongRoPE = 1,
        NTK_RoPE = 2,
        YARN_RoPE = 3
    } rope_type;
    float rope_theta = 100000.0f;

    RopeScalingParams ntk_scaling_params;

    std::vector<float> short_factor;
    std::vector<float> long_factor;
    int original_max_position_embeddings = 0;

    int image_pad_id = -1;
    int patch_size = 14;
    int patch_dim = 1280;
    int max_num_patches = 49152;
    int spatial_merge_size = 2;

    enum Vision_Type {
        VISION_CLOSE = 0,
        VISION_VIT = 1,
        VISION_QWEN3_5_VL = 2
    } vision_type;

    enum VisionRoPE_Type {
        mRoPE = 0
    } vision_rope_type;

    std::vector<int> mrope_section;
    std::vector<nlohmann::json> tools;

public:
    ncnn_llm_gpt(const std::string& model_path, bool use_vulkan = false, int num_threads = 0, int vulkan_device = 0);

    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text) const;
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text, const ncnn::Mat& bgr, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const;
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& input_text, const std::shared_ptr<ncnn_llm_gpt_ctx> ctx) const;
    std::shared_ptr<ncnn_llm_gpt_ctx> generate(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in, const GenerateConfig& cfg, std::function<void(const std::string&)> callback) const;

    std::shared_ptr<ncnn_llm_gpt_ctx> define_tools(const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx, const std::vector<nlohmann::json>& tools, const std::string& system_prompt = "You are a helpful assistant.");

    template<typename T>
    static constexpr const char* json_type_name() {
        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, long> || std::is_same_v<T, long long>) return "integer";
        else if constexpr (std::is_same_v<T, bool>) return "boolean";
        else if constexpr (std::is_floating_point_v<T>) return "number";
        else return "string";
    }

    template<typename Ret, typename... Args>
    static nlohmann::json make_function_tool(const std::string& name, const std::string& description, const std::array<std::string, sizeof...(Args)>& arg_names)
    {
        assert(arg_names.size() == sizeof...(Args));
        nlohmann::json properties = nlohmann::json::object();
        size_t idx = 0;
        ((
            properties[arg_names[idx]] = nlohmann::json{{"type", json_type_name<Args>()}, {"description", ""}},
            ++idx
        ), ...);

        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", description},
                {"parameters", {
                    {"type", "object"},
                    {"properties", properties},
                    {"required", arg_names}
                }}
            }}
        };
    }

private:
    int get_scaled_image_size(float scale, int size, int effective_patch_size) const;
    void get_image_size_for_patches(int image_height, int image_width, int patch_size, int max_num_patches, int& target_height, int& target_width) const;
    ncnn::Mat bgr_to_pixel_values(const ncnn::Mat& bgr) const;
    ncnn::Mat reorder_patches_for_merge(const ncnn::Mat& pixel_values, int h_patches, int w_patches, int merge_size = 2) const;
    void get_window_index(int num_patches_w, int num_patches_h, std::vector<int>& window_index, std::vector<int>& cu_window_seqlens) const;
    int get_visiual_features(const ncnn::Mat& bgr, ncnn::Mat& image_embeds, int& num_patches_w, int& num_patches_h) const;
};
