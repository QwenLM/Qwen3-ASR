# coding=utf-8
"""
Download Qwen3-ASR test audio files to local disk.

Usage:
  python scripts/download_test_audio.py [--output-dir ./test_audio]
"""

import argparse
import os
import urllib.request

TEST_AUDIO = {
    "asr_zh.wav": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    "asr_en.wav": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
}


def download(url: str, dest: str, timeout: int = 30) -> None:
    print(f"  {url}")
    print(f"  -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"  done ({os.path.getsize(dest) / 1024:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen3-ASR test audio files")
    parser.add_argument(
        "--output-dir", "-o",
        default="./test_audio",
        help="Output directory (default: ./test_audio)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[download] output dir: {args.output_dir}\n")

    for fname, url in TEST_AUDIO.items():
        dest = os.path.join(args.output_dir, fname)
        if os.path.exists(dest):
            print(f"[skip] already exists: {dest}")
            continue
        download(url, dest)

    print(f"\n[done] pass --input {args.output_dir} to batch_transcribe.py")


if __name__ == "__main__":
    main()
