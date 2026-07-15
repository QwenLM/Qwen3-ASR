#pragma once
#include <string>
#include <optional>

struct SpecialTokensConfig {
    std::optional<std::string> bos_token;
    std::optional<std::string> eos_token;
    std::optional<std::string> unk_token;
    std::optional<std::string> sep_token;
    std::optional<std::string> pad_token;
    std::optional<std::string> cls_token;
    std::optional<std::string> mask_token;
};

struct SpecialTokenIds {
    int bos_id = -1;
    int eos_id = -1;
    int unk_id = -1;
    int sep_id = -1;
    int pad_id = -1;
    int cls_id = -1;
    int mask_id = -1;
};