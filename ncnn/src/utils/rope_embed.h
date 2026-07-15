#pragma once

#include <mat.h>
#include <vector>

struct RopeScalingParams {
    float alpha;
    float beta_fast;
    float beta_slow;
    float factor;
    float mscale;
    float mscale_all_dim;
};

void generate_ntk_rope_embed_cache(
    int seqlen,
    int embed_dim,
    int position_id,
    ncnn::Mat& cos_cache,
    ncnn::Mat& sin_cache,
    float rope_theta,
    const RopeScalingParams& scaling_params
);

void generate_yarn_rope_embed_cache(
    int seqlen,
    int embed_dim,
    int position_id,
    ncnn::Mat& cos_cache,
    ncnn::Mat& sin_cache,
    float rope_theta,
    const RopeScalingParams& scaling_params
);

void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta = 100000);

void generate_rope_embed_cache_full(int seqlen, int head_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta = 100000);

void generate_rope_embed_cache_LongRoPE(int seqlen,
                                      int embed_dim,
                                      int position_id,
                                      ncnn::Mat& cos_cache,
                                      ncnn::Mat& sin_cache,
                                      float rope_theta,
                                      const float* SHORT_FACTOR,
                                      const float* LONG_FACTOR,
                                      int ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768);

void inject_image_embeds(std::vector<int>& token_ids, ncnn::Mat& token_embed, int& image_pad_index, int image_pad_id, const ncnn::Mat& image_embeds);

void generate_rope_embed_cache_vision_mrope(int seqlen,
                                          int embed_dim,
                                          int position_id,
                                          int image_pad_index,
                                          int image_embeds_size,
                                          int num_patches_w,
                                          int spatial_merge_size,
                                          const std::vector<int>& mrope_section,
                                          ncnn::Mat& cos_cache,
                                          ncnn::Mat& sin_cache,
                                          float rope_theta = 100000);

void generate_rope_embed_cache_vision_mrope_interleaved(int seqlen,
                                                        int embed_dim,
                                                        int position_id,
                                                        int image_pad_index,
                                                        int image_embeds_size,
                                                        int num_patches_w,
                                                        ncnn::Mat& cos_cache,
                                                        ncnn::Mat& sin_cache,
                                                        float rope_theta = 100000);

// HunyuanOCR 4-axis xdrope. pos4 = { linear, w, h, t }, each of length seq_len.
// Produces cos/sin caches of shape (rope_head_dim/2, seq_len) with an NTK-alpha base:
//   base = rope_theta * alpha^(rope_head_dim/(rope_head_dim-2));
//   inv_freq[j] = 1 / base^(2j/rope_head_dim);
// dim j uses axis = xdrope_section prefix (e.g. [16,16,16,16] -> j/16).
void generate_hunyuan_xdrope_cos_sin(const std::vector<int>* pos4,
                                     int seq_len,
                                     int rope_head_dim,
                                     const std::vector<int>& xdrope_section,
                                     float rope_theta,
                                     float alpha,
                                     ncnn::Mat& cos_cache,
                                     ncnn::Mat& sin_cache);