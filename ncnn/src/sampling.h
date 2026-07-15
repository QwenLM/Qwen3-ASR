#pragma once

#include <vector>

void softmax_vec(std::vector<float>& logits, float temperature);
void apply_top_k(std::vector<float>& probs, int k);
void apply_top_p(std::vector<float>& probs, float p);
int sample_from_probs(const std::vector<float>& probs);
