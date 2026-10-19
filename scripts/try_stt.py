"""Step 1: hearing alone. Whisper turns a recording into text, timed and checked.

Run:  uv run python -m scripts.try_stt                         (uses recordings/tts.wav from try_tts)
      uv run python -m scripts.try_stt "C:\\path\\to\\me.m4a" --model small.en

If a .txt with the same name sits next to the audio (try_tts writes one), the transcript is
compared with it and the word error rate (WER) is printed. Pipecat calls Whisper with the default
beam size of 5; this script also times beam size 1 so you can see what the setting costs.
"""

import argparse
import os
import re
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from agent import config
from agent.gpu import enable_cuda_dlls

DEFAULT_AUDIO = config.ROOT / "recordings" / "tts.wav"


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_error_rate(expected: str, heard: str) -> float:
    """(substitutions + deletions + insertions) / words expected, by edit distance over words."""
    ref, hyp = words(expected), words(heard)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", nargs="?", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--model", default=config.STT_MODEL, help='e.g. "small.en", "large-v3-turbo"')
    parser.add_argument("--device", default=config.STT_DEVICE, help='"cuda" or "cpu"')
    parser.add_argument("--runs", type=int, default=3, help="timed runs after one warm-up")
    args = parser.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"{args.audio} not found. Run first:  uv run python -m scripts.try_tts")

    enable_cuda_dlls()
    from faster_whisper import WhisperModel, decode_audio

    compute_type = config.STT_COMPUTE_TYPE if args.device == "cuda" else "int8"
    t0 = time.perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)
    print(f"{args.model} loaded in {time.perf_counter() - t0:.1f}s on {args.device} ({compute_type})")

    audio = decode_audio(str(args.audio), sampling_rate=16000)  # what Pipecat hands Whisper: 16 kHz mono
    audio_s = len(audio) / 16000
    print(f"audio: {args.audio.name}, {audio_s:.1f}s\n")

    def transcribe(beam_size: int) -> str:
        segments, _ = model.transcribe(audio, language="en", beam_size=beam_size)
        return " ".join(s.text.strip() for s in segments)  # transcribe is lazy: this loop does the work

    t0 = time.perf_counter()
    transcribe(5)
    print(f"warm-up (cold): {(time.perf_counter() - t0) * 1000:.0f} ms\n")

    expected_file = args.audio.with_suffix(".txt")
    expected = expected_file.read_text(encoding="utf-8") if expected_file.exists() else None

    for beam_size in (5, 1):
        runs = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            text = transcribe(beam_size)
            runs.append((time.perf_counter() - t0) * 1000)
        ms = statistics.median(runs)
        note = "  (Pipecat default)" if beam_size == 5 else ""
        print(f"beam_size={beam_size}: {ms:.0f} ms, {audio_s * 1000 / ms:.0f}x real time{note}")
        print(f"  heard:    {text}")
        if expected:
            print(f"  expected: {expected}")
            print(f"  WER:      {word_error_rate(expected, text):.0%}")
        print()


if __name__ == "__main__":
    main()
