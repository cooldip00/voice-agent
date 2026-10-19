"""Step 1: the brain alone. Qwen on Ollama streams a reply, timed.

Run:  uv run python -m scripts.try_llm
      uv run python -m scripts.try_llm --model qwen2.5:3b --say "I lost my job, I can't pay this month."

Three numbers matter for a voice agent:
- time to first token: how soon the model starts answering;
- time to first sentence: when Pipecat can hand the first sentence to the voice (Kokoro);
- tokens per second: how fast the rest arrives (it only has to beat speaking speed).
Uses Ollama's native /api/chat because it reports load and generation times. In the bot, Pipecat
talks to the same server through its OpenAI-compatible /v1 endpoint.
"""

import argparse
import json
import statistics
import time

import requests
from pipecat.utils.string import match_endofsentence

from agent import config

# Placeholder persona; the real prompt and its rules come in step 4.
SYSTEM_PROMPT = (
    "You are Asha, a polite phone agent for Greenfield Bank, calling about an overdue loan payment. "
    "This is a voice call: reply in one or two short sentences, no lists, no markdown."
)
DEFAULT_SAY = "Oh, sorry, I forgot about it. Can I pay next week?"


def stream_reply(model: str, user_text: str) -> dict:
    """Send one turn and time the stream as it arrives."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}],
        "stream": True,
        "keep_alive": "10m",  # stay in VRAM between calls instead of the default 5 minutes
    }
    t0 = time.perf_counter()
    first_token = first_sentence = None
    text = ""
    with requests.post(f"{config.OLLAMA_BASE}/api/chat", json=body, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece and first_token is None:
                first_token = time.perf_counter() - t0
            text += piece
            if first_sentence is None and match_endofsentence(text):  # Pipecat's own check
                first_sentence = time.perf_counter() - t0
            if chunk.get("done"):
                final = chunk
    total = time.perf_counter() - t0
    return {
        "text": text.strip(),
        "first_token_ms": first_token * 1000,
        "first_sentence_ms": (first_sentence or total) * 1000,
        "total_ms": total * 1000,
        "load_ms": final.get("load_duration", 0) / 1e6,
        "tokens": final.get("eval_count", 0),
        "tokens_per_s": final.get("eval_count", 0) / max(final.get("eval_duration", 1) / 1e9, 1e-9),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--say", default=DEFAULT_SAY, help="what the caller says")
    parser.add_argument("--runs", type=int, default=3, help="timed runs after one warm-up")
    args = parser.parse_args()

    try:
        requests.get(f"{config.OLLAMA_BASE}/api/tags", timeout=5).raise_for_status()
    except requests.RequestException:
        raise SystemExit(f"Ollama not reachable at {config.OLLAMA_BASE}. Start the Ollama app.")

    print(f"model: {args.model}\ncaller: {args.say!r}\n")
    warm = stream_reply(args.model, args.say)
    loaded = f"model load {warm['load_ms']:.0f} ms" if warm["load_ms"] > 500 else "model was already in VRAM"
    print(f"warm-up: first token after {warm['first_token_ms']:.0f} ms ({loaded})\n")

    results = [stream_reply(args.model, args.say) for _ in range(args.runs)]
    for i, r in enumerate(results, 1):
        print(f"run {i}: {r['text']}")

    def med(key: str) -> float:
        return statistics.median(r[key] for r in results)

    print(f"\nmedian of {args.runs} warm runs")
    print(f"  time to first token:    {med('first_token_ms'):6.0f} ms")
    print(f"  time to first sentence: {med('first_sentence_ms'):6.0f} ms   <- Kokoro can start here")
    print(f"  whole reply:            {med('total_ms'):6.0f} ms   ({med('tokens'):.0f} tokens)")
    print(f"  speed:                  {med('tokens_per_s'):6.0f} tokens/s  (speech is about 3-4 tokens/s)")


if __name__ == "__main__":
    main()
