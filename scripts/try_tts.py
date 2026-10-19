"""Step 1: the voice alone. Kokoro turns a reply into speech, timed.

Run:  uv run python -m scripts.try_tts
      uv run python -m scripts.try_tts --voice am_michael --text "Your payment is due on Friday."

In a live call Pipecat sends the LLM's reply to Kokoro one sentence at a time, so what the caller
waits for is the *first sentence*. This script makes each sentence separately, like Pipecat does,
and saves the whole reply to recordings/tts.wav (plus tts.txt, the text, for try_stt to check against).
"""

import argparse
import statistics
import time
import wave

import numpy as np
from kokoro_onnx import Kokoro
from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files
from pipecat.utils.string import match_endofsentence

from agent import config

DEFAULT_TEXT = (
    "Thank you for confirming, Mr. Sharma. "
    "Your loan payment of four thousand five hundred rupees was due on the fifth. "
    "Would you be able to pay it by this Friday?"
)
OUT_WAV = config.ROOT / "recordings" / "tts.wav"


def split_sentences(text: str) -> list[str]:
    """Cut where Pipecat's sentence aggregator would, using its own function (knows "Mr." isn't an end)."""
    sentences, rest = [], text.strip()
    while rest:
        end = match_endofsentence(rest) or len(rest)
        sentences.append(rest[:end].strip())
        rest = rest[end:].strip()
    return sentences


def load_kokoro() -> Kokoro:
    model_path = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
    voices_path = KOKORO_CACHE_DIR / "voices-v1.0.bin"
    _ensure_model_files(model_path, voices_path)
    return Kokoro(str(model_path), str(voices_path))


def save_wav(path, samples: np.ndarray, rate: int):
    path.parent.mkdir(exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice", default=config.TTS_VOICE)
    parser.add_argument("--runs", type=int, default=3, help="timed runs after one warm-up")
    args = parser.parse_args()

    t0 = time.perf_counter()
    kokoro = load_kokoro()
    print(f"Kokoro loaded in {time.perf_counter() - t0:.1f}s (CPU, ONNX), voice '{args.voice}'\n")

    sentences = split_sentences(args.text)
    t0 = time.perf_counter()
    kokoro.create(sentences[0], voice=args.voice, lang="en-us")  # warm-up: first run is always slower
    print(f"warm-up (cold first sentence): {(time.perf_counter() - t0) * 1000:.0f} ms\n")

    timings: list[list[float]] = [[] for _ in sentences]
    for _ in range(args.runs):
        audio = []
        for i, sentence in enumerate(sentences):
            t0 = time.perf_counter()
            samples, rate = kokoro.create(sentence, voice=args.voice, lang="en-us")
            timings[i].append((time.perf_counter() - t0) * 1000)
            audio.append(samples)

    print(f"{'sentence':<60} {'speech':>7} {'made in':>8} {'x real time':>12}")
    for sentence, samples, runs in zip(sentences, audio, timings):
        speech_s = len(samples) / rate
        made_ms = statistics.median(runs)
        label = sentence if len(sentence) <= 58 else sentence[:55] + "..."
        print(f"{label:<60} {speech_s:>6.1f}s {made_ms:>6.0f}ms {speech_s * 1000 / made_ms:>11.1f}x")

    total_speech = sum(len(a) for a in audio) / rate
    total_ms = sum(statistics.median(r) for r in timings)
    print(f"\nfirst audio ready after: {statistics.median(timings[0]):.0f} ms  (median of {args.runs})")
    print(f"whole reply: {total_speech:.1f}s of speech made in {total_ms:.0f} ms")
    print("  (while sentence 1 plays, sentence 2 is being made, so only the first one makes the caller wait)")

    save_wav(OUT_WAV, np.concatenate(audio), rate)
    OUT_WAV.with_suffix(".txt").write_text(args.text, encoding="utf-8")
    print(f"\nsaved {OUT_WAV.relative_to(config.ROOT)} (listen: Invoke-Item {OUT_WAV.relative_to(config.ROOT)})")


if __name__ == "__main__":
    main()
