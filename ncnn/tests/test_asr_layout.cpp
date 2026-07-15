#include "ncnn_llm_asr.h"

#include <cstdio>

int main() {
    struct Case {
        int mel_frames;
        int audio_tokens;
    };
    const Case cases[] = {
        {99, 13},
        {100, 13},
        {101, 14},
        {200, 26},
    };

    for (const Case& item : cases) {
        const int actual = ncnn_llm_asr::feat_out_len(item.mel_frames);
        if (actual != item.audio_tokens) {
            std::fprintf(stderr, "feat_out_len(%d): expected %d, got %d\n",
                         item.mel_frames, item.audio_tokens, actual);
            return 1;
        }
    }

    std::puts("Qwen3-ASR audio-token layout tests passed");
    return 0;
}
