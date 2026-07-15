#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ncnn_llm_asr.h"
#include "utf8_args.h"

// Minimal WAV reader: mono/stereo, PCM16 (fmt 1) or IEEE float32 (fmt 3).
// Returns mono float samples in [-1, 1]; sets sample_rate. Empty vector on failure.
static std::vector<float> read_wav(const std::string& path, int& sample_rate) {
    std::vector<float> pcm;
    sample_rate = 0;
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return pcm; }
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() < 44 || memcmp(buf.data(), "RIFF", 4) != 0 || memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "not a WAV file: %s\n", path.c_str());
        return pcm;
    }
    auto rd_u32 = [&](size_t o) { uint32_t v; memcpy(&v, buf.data() + o, 4); return v; };
    auto rd_u16 = [&](size_t o) { uint16_t v; memcpy(&v, buf.data() + o, 2); return v; };

    uint16_t audio_format = 0, num_channels = 1, bits = 16;
    size_t data_off = 0, data_len = 0;
    size_t pos = 12;
    while (pos + 8 <= buf.size()) {
        char id[5] = {0}; memcpy(id, buf.data() + pos, 4);
        uint32_t sz = rd_u32(pos + 4);
        size_t body = pos + 8;
        if (memcmp(id, "fmt ", 4) == 0 && body + 16 <= buf.size()) {
            audio_format = rd_u16(body + 0);
            num_channels = rd_u16(body + 2);
            sample_rate  = (int)rd_u32(body + 4);
            bits         = rd_u16(body + 14);
        } else if (memcmp(id, "data", 4) == 0) {
            data_off = body;
            data_len = sz;
            if (data_off + data_len > buf.size()) data_len = buf.size() - data_off;
            break;
        }
        pos = body + sz + (sz & 1);  // chunks are word-aligned
    }
    if (data_off == 0 || num_channels == 0) { fprintf(stderr, "no data chunk in %s\n", path.c_str()); return pcm; }

    const int ch = num_channels;
    std::vector<float> inter;  // interleaved samples
    if (audio_format == 3 && bits == 32) {
        size_t n = data_len / 4;
        inter.resize(n);
        for (size_t i = 0; i < n; i++) memcpy(&inter[i], buf.data() + data_off + i * 4, 4);
    } else if (audio_format == 1 && bits == 16) {
        size_t n = data_len / 2;
        inter.resize(n);
        for (size_t i = 0; i < n; i++) {
            int16_t s; memcpy(&s, buf.data() + data_off + i * 2, 2);
            inter[i] = s / 32768.0f;
        }
    } else {
        fprintf(stderr, "unsupported WAV format=%u bits=%u (need PCM16 or float32)\n", audio_format, bits);
        return pcm;
    }

    // Downmix to mono.
    size_t frames = inter.size() / ch;
    pcm.resize(frames);
    for (size_t i = 0; i < frames; i++) {
        float acc = 0.0f;
        for (int c = 0; c < ch; c++) acc += inter[i * ch + c];
        pcm[i] = acc / ch;
    }
    return pcm;
}

int main(int argc, char** argv) {
    enable_utf8_console();
    std::vector<std::string> args = get_utf8_args(argc, argv);

    std::string model_path = "assets/qwen3_asr_0.6b";
    std::string audio_path;
    int max_new_tokens = 256;

    for (size_t i = 1; i < args.size(); i++) {
        const std::string& a = args[i];
        if (a == "--help" || a == "-h") {
            printf("Usage: %s --audio <wav_16k_mono> [--model <model_path>] [--max-new-tokens N]\n", argv[0]);
            return 0;
        }
        if (a == "--model" && i + 1 < args.size()) model_path = args[++i];
        else if (a == "--audio" && i + 1 < args.size()) audio_path = args[++i];
        else if (a == "--max-new-tokens" && i + 1 < args.size()) {
            try {
                max_new_tokens = std::stoi(args[++i]);
            } catch (...) {
                fprintf(stderr, "Invalid --max-new-tokens value\n");
                return 1;
            }
        }
    }

    if (audio_path.empty() || max_new_tokens <= 0) {
        fprintf(stderr, "Usage: %s --audio <wav_16k_mono> [--model <model_path>] [--max-new-tokens N]\n", argv[0]);
        return 1;
    }

    int sr = 0;
    std::vector<float> pcm = read_wav(audio_path, sr);
    if (pcm.empty()) { fprintf(stderr, "Failed to read audio: %s\n", audio_path.c_str()); return 1; }
    printf("Audio: %s  (%d samples, %d Hz)\n", audio_path.c_str(), (int)pcm.size(), sr);
    if (sr != 16000) {
        fprintf(stderr, "WARNING: sample rate is %d Hz, expected 16000. Resample first for correct results.\n", sr);
    }

    printf("Loading ASR model from %s\n", model_path.c_str());
    ncnn_llm_asr asr(model_path, false, 4);
    if (!asr.ok()) { fprintf(stderr, "Failed to load ASR model\n"); return 1; }

    GenerateConfig cfg;
    cfg.max_new_tokens = max_new_tokens;
    cfg.temperature = 0.0f;
    cfg.top_p = 0.00001f;
    cfg.top_k = 1;
    cfg.repetition_penalty = 1.0f;
    cfg.do_sample = 0;

    std::string language;
    std::string text = asr.transcribe(pcm, cfg, &language);

    printf("\n==== Result ====\n");
    printf("language: %s\n", language.c_str());
    printf("text: %s\n", text.c_str());
    return 0;
}
