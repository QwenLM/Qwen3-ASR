#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <functional>
#include <utility>
#include <limits>
#include <mutex>
#include <map>
#include "tokenizer_types.h"

class BpeTokenizer {
public:
    static BpeTokenizer LoadFromFiles(const std::string& vocab_path,
                                      const std::string& merges_path,
                                      const SpecialTokensConfig& spec,
                                      bool add_special_if_missing = true,
                                      bool fallback_to_chars = true,
                                      bool use_byte_encoder = false);

    std::vector<int> encode(const std::string& text,
                            bool add_bos = false,
                            bool add_eos = false,
                            bool add_cls = false,
                            bool add_sep = false) const;

    std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;

    size_t vocab_size() const { return id_to_token_.size(); }
    const std::vector<std::string>& id_to_token() const { return id_to_token_; }
    const std::unordered_map<std::string, int>& token_to_id() const { return token_to_id_; }
    const SpecialTokenIds& special_ids() const { return special_ids_; }
    bool fallback_to_chars() const { return fallback_to_chars_; }

    void AddAdditionalSpecialToken(const std::string& token,
                                   bool add_if_missing = true);

    void SetAdditionalSpecialTokens(const std::vector<std::string>& tokens,
                                    bool add_if_missing = true);

    const std::vector<int>& additional_special_token_ids() const {
        return additional_special_token_ids_;
    }

    BpeTokenizer(const BpeTokenizer&) = delete;
    BpeTokenizer& operator=(const BpeTokenizer&) = delete;
    BpeTokenizer(BpeTokenizer&& other) noexcept;
    BpeTokenizer& operator=(BpeTokenizer&& other) noexcept;

private:
    BpeTokenizer() = default;

    static std::vector<std::string> LoadVocab(const std::string& vocab_path);
    static std::unordered_map<std::string, int> BuildTokenToId(const std::vector<std::string>& id_to_token);
    static std::unordered_map<std::string, int> LoadMergesRank(const std::string& merges_path);
    void EnsureSpecialTokens(const SpecialTokensConfig& spec, bool add_if_missing);
    static std::vector<std::string> PretokenizeSentencePiece(const std::string& text);

    const std::vector<std::string>& BpeForPieceCached(const std::string& piece) const;
    std::vector<std::string> BpeForPiece(const std::string& piece) const;

    void TokensToIds(const std::vector<std::string>& tokens, std::vector<int>& out) const;

    static bool IsAsciiSpace(unsigned char c);
    static bool IsUnicodeSpace(uint32_t cp);
    static bool NextUtf8(const std::string& s, size_t& i, uint32_t& cp, size_t& cp_len);
    static std::vector<std::string> Utf8Chars(const std::string& s);

    void InitByteMaps();
    std::string ByteEncode(const std::string& text) const;
    std::string ByteDecode(const std::string& text) const;

    static std::string PairKey(const std::string& a, const std::string& b);

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merges_rank_;

    SpecialTokenIds special_ids_;
    bool fallback_to_chars_ = true;

    std::vector<std::string> additional_special_tokens_;
    std::vector<int> additional_special_token_ids_;
    std::unordered_map<std::string, int> additional_special_token_to_id_;
    std::unordered_set<int> additional_special_id_set_;

    bool use_byte_encoder_ = false;
    std::vector<uint32_t> byte_encoder_;
    std::map<uint32_t, uint8_t> byte_decoder_;

    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
    mutable std::mutex cache_mu_;
};