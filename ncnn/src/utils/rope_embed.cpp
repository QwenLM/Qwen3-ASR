#include "rope_embed.h"
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdio>

static float yarn_ramp(float low, float high, float val) {
    if (val < low) return 0.0f;
    if (val > high) return 1.0f;
    return (val - low) / (high - low);
}

void generate_yarn_rope_embed_cache(
    int seqlen,
    int embed_dim,
    int position_id,
    ncnn::Mat& cos_cache,
    ncnn::Mat& sin_cache,
    float rope_theta,
    const RopeScalingParams& scaling_params
) {
    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    std::vector<float> inv_freq(embed_dim / 2);

    float scale = scaling_params.alpha;
    float beta_fast = scaling_params.beta_fast;
    float beta_slow = scaling_params.beta_slow;

    for (int i = 0; i < embed_dim / 2; i++) {
        float freq = 1.0f / powf(rope_theta, (float)(i * 2) / (float)embed_dim);

        float r = (float)(i * 2) / (float)embed_dim;

        float ramp = yarn_ramp(beta_slow, beta_fast, (float)embed_dim / (i * 2 + 1));
        float base_correction = powf(scale, (float)embed_dim / ((float)embed_dim - 2.0f));
        float ntk_theta = rope_theta * base_correction;
        inv_freq[i] = 1.0f / powf(ntk_theta, (float)(i * 2) / (float)embed_dim);
    }

    for (int i = 0; i < seqlen; i++) {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++) {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];

            float mscale = scaling_params.mscale;
            if (fabs(mscale - 1.0f) > 1e-6) {
                *cos_ptr++ = cosf(t) * mscale;
                *sin_ptr++ = sinf(t) * mscale;
            } else {
                *cos_ptr++ = cosf(t);
                *sin_ptr++ = sinf(t);
            }
        }
    }
}

void generate_ntk_rope_embed_cache(
    int seqlen,
    int embed_dim,
    int position_id,
    ncnn::Mat& cos_cache,
    ncnn::Mat& sin_cache,
    float rope_theta,
    const RopeScalingParams& scaling_params
) {
    float dim = (float)embed_dim;
    float alpha = scaling_params.alpha;

    float base_correction = powf(alpha, dim / (dim - 2.0f));
    float ntk_theta = rope_theta * base_correction;

    std::vector<float> inv_freq(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; i++)
    {
        inv_freq[i] = 1.f / powf(ntk_theta, (float)(i * 2) / dim);
    }

    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++)
        {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];

            *cos_ptr++ = cosf(t);
            *sin_ptr++ = sinf(t);
        }
    }
}

void generate_rope_embed_cache(int seqlen, int embed_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta)
{
    std::vector<float> inv_freq(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; i++)
    {
        inv_freq[i] = 1.f / powf(rope_theta, (float)(i * 2) / embed_dim);
    }

    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++)
        {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];
            const float cos_val = cosf(t);
            const float sin_val = sinf(t);
            *cos_ptr++ = cos_val;
            *sin_ptr++ = sin_val;
        }
    }
}

void generate_rope_embed_cache_full(int seqlen, int head_dim, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache, float rope_theta)
{
    int half_dim = head_dim / 2;
    std::vector<float> inv_freq(half_dim);
    for (int i = 0; i < half_dim; i++)
    {
        inv_freq[i] = 1.f / powf(rope_theta, (float)(i * 2) / head_dim);
    }

    cos_cache.create(head_dim, seqlen);
    sin_cache.create(head_dim, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < half_dim; j++)
        {
            const int pos = position_id + i;
            const float t = pos * inv_freq[j];
            const float cos_val = cosf(t);
            const float sin_val = sinf(t);
            cos_ptr[j] = cos_val;
            cos_ptr[j + half_dim] = cos_val;
            sin_ptr[j] = sin_val;
            sin_ptr[j + half_dim] = sin_val;
        }
    }
}

void generate_rope_embed_cache_vision_mrope_interleaved(int seqlen,
                                                        int embed_dim,
                                                        int position_id,
                                                        int image_pad_index,
                                                        int image_embeds_size,
                                                        int num_patches_w,
                                                        ncnn::Mat& cos_cache,
                                                        ncnn::Mat& sin_cache,
                                                        float rope_theta)
{
    const int merge_size = 2;
    const int mrope[3] = {11, 11, 10};
    const float partial_rotary_factor = 0.5f;
    const int rotate_dim = embed_dim * partial_rotary_factor;

    std::vector<float> inv_freq(rotate_dim / 2);
    for (int i = 0; i < rotate_dim / 2; i++)
    {
        inv_freq[i] = 1.f / powf(rope_theta, (float)(i * 2) / rotate_dim);
    }

    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++)
        {
            if (j < rotate_dim / 2)
            {
                int pos = position_id;
                if (i < image_pad_index)
                {
                    pos += i;
                }
                else if (i >= image_pad_index + image_embeds_size)
                {
                    pos += i - image_embeds_size + (num_patches_w / merge_size);
                }
                else
                {
                    int which_pos = 0;

                    if (j < mrope[1] * 3 && (j % 3 == 1))
                    {
                        which_pos = 1;
                    }
                    else if (j < mrope[2] * 3 && (j % 3 == 2))
                    {
                        which_pos = 2;
                    }

                    if (which_pos == 0)
                    {
                        pos += image_pad_index;
                    }
                    else if (which_pos == 1)
                    {
                        int hid = (i - image_pad_index) / (num_patches_w / merge_size);
                        pos += image_pad_index + hid;
                    }
                    else
                    {
                        int wid = (i - image_pad_index) % (num_patches_w / merge_size);
                        pos += image_pad_index + wid;
                    }
                }

                const float t = pos * inv_freq[j];
                const float cos_val = cosf(t);
                const float sin_val = sinf(t);
                *cos_ptr++ = cos_val;
                *sin_ptr++ = sin_val;
            }
            else
            {
                *cos_ptr++ = 1.0f;
                *sin_ptr++ = 0.0f;
            }
        }
    }
}

static inline float compute_scaling_factor(int max_position_embeddings, int ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768) {
    float scale = static_cast<float>(max_position_embeddings) / static_cast<float>(ORIGINAL_MAX_POSITION_EMBEDDINGS);
    return std::sqrt(1.0f + std::log(scale) / std::log(static_cast<float>(ORIGINAL_MAX_POSITION_EMBEDDINGS)));
}

void generate_rope_embed_cache_LongRoPE(int seqlen,
                                      int embed_dim,
                                      int position_id,
                                      ncnn::Mat& cos_cache,
                                      ncnn::Mat& sin_cache,
                                      float rope_theta,
                                      const float* SHORT_FACTOR,
                                      const float* LONG_FACTOR,
                                      int ORIGINAL_MAX_POSITION_EMBEDDINGS)
{
    if (embed_dim % 2 != 0 || seqlen <= 0) {
        cos_cache.release();
        sin_cache.release();
        return;
    }

    const int half_dim = embed_dim / 2;

    cos_cache.create(half_dim, seqlen);
    sin_cache.create(half_dim, seqlen);

    if (cos_cache.empty() || sin_cache.empty()) {
        return;
    }

    float* cos_ptr = cos_cache.channel(0);
    float* sin_ptr = sin_cache.channel(0);

    std::vector<float> inv_freq(half_dim);
    for (int j = 0; j < half_dim; ++j) {
        float exponent = (2.0f * j) / static_cast<float>(embed_dim);
        inv_freq[j] = 1.0f / std::pow(rope_theta, exponent);
    }

    const float* ext_factor = (seqlen > ORIGINAL_MAX_POSITION_EMBEDDINGS) ? LONG_FACTOR : SHORT_FACTOR;
    const float scaling_factor = compute_scaling_factor(ORIGINAL_MAX_POSITION_EMBEDDINGS);

    for (int i = 0; i < seqlen; ++i) {
        int t = position_id + i;
        float* row_cos = cos_ptr + i * half_dim;
        float* row_sin = sin_ptr + i * half_dim;

        for (int j = 0; j < half_dim; ++j) {
            float freq = (static_cast<float>(t) * inv_freq[j]) / ext_factor[j];
            row_cos[j] = std::cos(freq) * scaling_factor;
            row_sin[j] = std::sin(freq) * scaling_factor;
        }
    }
}

void inject_image_embeds(std::vector<int>& token_ids, ncnn::Mat& token_embed, int& image_pad_index, int image_pad_id, const ncnn::Mat& image_embeds)
{
    image_pad_index = -1;
    if (image_embeds.empty())
    {
        return;
    }

    std::vector<int> token_ids_injected(token_ids.size() - 1 + image_embeds.h);
    ncnn::Mat token_embed_injected(token_embed.w, token_embed.h - 1 + image_embeds.h);

    for (int i = 0; i < (int)token_ids.size(); i++)
    {
        if (token_ids[i] == image_pad_id)
        {
            image_pad_index = i;

            // inject token ids
            memcpy(token_ids_injected.data(), token_ids.data(), i * sizeof(int));
            memset(token_ids_injected.data() + i, image_pad_id, image_embeds.h * sizeof(int));
            memcpy(token_ids_injected.data() + i + image_embeds.h, token_ids.data() + i + 1, (token_ids.size() - 1 - i) * sizeof(int));

            // inject token embed
            memcpy(token_embed_injected.row(0), token_embed.row(0), i * token_embed.w * sizeof(float));
            memcpy(token_embed_injected.row(i), image_embeds.row(0), image_embeds.h * token_embed.w * sizeof(float));
            memcpy(token_embed_injected.row(i + image_embeds.h), token_embed.row(i + 1), (token_ids.size() - 1 - i) * token_embed.w * sizeof(float));

            break;
        }
    }

    token_ids = token_ids_injected;
    token_embed = token_embed_injected;
}

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
                                          float rope_theta)
{
    if (mrope_section.size() < 3) return;

    std::vector<float> inv_freq(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; i++)
    {
        inv_freq[i] = 1.f / powf(rope_theta, (float)(i * 2) / embed_dim);
    }

    cos_cache.create(embed_dim / 2, seqlen);
    sin_cache.create(embed_dim / 2, seqlen);

    for (int i = 0; i < seqlen; i++)
    {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);

        for (int j = 0; j < embed_dim / 2; j++)
        {
            int pos = position_id;
            if (i < image_pad_index)
            {
                pos += i;
            }
            else if (i >= image_pad_index + image_embeds_size)
            {
                pos += i - image_embeds_size + (num_patches_w / spatial_merge_size);
            }
            else
            {
                // inside image tokens range
                if (j < mrope_section[0])
                {
                    // temporal
                    pos += image_pad_index;
                }
                else if (j < mrope_section[0] + mrope_section[1])
                {
                    // height
                    int hid = (i - image_pad_index) / (num_patches_w / spatial_merge_size);
                    pos += image_pad_index + hid;
                }
                else
                {
                    // width
                    int wid = (i - image_pad_index) % (num_patches_w / spatial_merge_size);
                    pos += image_pad_index + wid;
                }
            }

            const float t = pos * inv_freq[j];
            const float cos_val = cosf(t);
            const float sin_val = sinf(t);
            *cos_ptr++ = cos_val;
            *sin_ptr++ = sin_val;
        }
    }
}
void generate_hunyuan_xdrope_cos_sin(const std::vector<int>* pos4,
                                     int seq_len,
                                     int rope_head_dim,
                                     const std::vector<int>& xdrope_section,
                                     float rope_theta,
                                     float alpha,
                                     ncnn::Mat& cos_cache,
                                     ncnn::Mat& sin_cache) {
    const int half = rope_head_dim / 2;
    cos_cache.create(half, seq_len);
    sin_cache.create(half, seq_len);

    // NTK-alpha base: base = rope_theta * alpha^(head_dim/(head_dim-2))
    const double base_corr = std::pow((double)alpha,
                                      (double)rope_head_dim / ((double)rope_head_dim - 2.0));
    const double ntk_theta = (double)rope_theta * base_corr;

    std::vector<float> inv_freq(half);
    for (int j = 0; j < half; j++) {
        inv_freq[j] = (float)(1.0 / std::pow(ntk_theta, (double)(2 * j) / (double)rope_head_dim));
    }

    // axis index per dim, from the (possibly unequal) section widths
    std::vector<int> axis_of_dim(half, (int)xdrope_section.size() - 1);
    {
        int d = 0;
        for (int a = 0; a < (int)xdrope_section.size() && d < half; a++) {
            for (int s = 0; s < xdrope_section[a] && d < half; s++) {
                axis_of_dim[d++] = a;
            }
        }
    }

    for (int i = 0; i < seq_len; i++) {
        float* cos_ptr = cos_cache.row(i);
        float* sin_ptr = sin_cache.row(i);
        for (int j = 0; j < half; j++) {
            const int a = axis_of_dim[j];
            const float pos = (float)pos4[a][i];
            const float t = pos * inv_freq[j];
            cos_ptr[j] = cosf(t);
            sin_ptr[j] = sinf(t);
        }
    }
}
