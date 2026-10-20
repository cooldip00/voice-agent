"""Every setting in one place, so experiments change one line."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Hearing (faster-whisper)
STT_MODEL = "Systran/faster-distil-whisper-medium.en"   # English-only; "small.en" if VRAM gets tight
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "int8_float16"

# Brain (Ollama)
OLLAMA_BASE = "http://127.0.0.1:11434"  # not "localhost": on Windows it tries IPv6 first and adds ~2 s per request
LLM_MODEL = os.environ.get("LLM", "qwen2.5:7b")   # step 5 compares: $env:LLM="qwen2.5:3b"
LLM_MODELS_NEEDED = ["qwen2.5:7b", "qwen2.5:3b"]
LLM_KEEP_ALIVE = "30m"           # how long Ollama keeps the model in VRAM after the startup warm-up
LLM_TEMPERATURE = 0.2            # low: steadier tool calls, fewer invented facts (Qwen's default is 0.7)

# Voice (Kokoro through onnxruntime-gpu; Pipecat caches its files in ~/.cache/pipecat/kokoro-onnx)
TTS_VOICE = "af_heart"
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda")   # step 5b: 4-6x faster than "cpu" per sentence

# Turn-taking
VAD_STOP_SECS = float(os.environ.get("VAD_STOP_SECS", "0.2"))   # Smart Turn works best with a short VAD stop
USER_IDLE_SECS = 8.0             # caller quiet this long after the bot spoke: it asks (prompts.STILL_THERE), then hangs up

# Fake bank and call rules
BANK_DB = ROOT / "data" / "bank.db"   # made by: uv run python -m scripts.seed_bank
CALL_HOURS = (8, 19)             # a call may start from 08:00 and must start before 19:00 (RBI rule for recovery calls)
PROMISE_MAX_DAYS = 7             # a promised payment date must be at most this many days away
CALLBACK_MAX_DAYS = 7            # same for a callback
MAX_VERIFY_ATTEMPTS = 3          # wrong dates of birth before the identity check locks
MIN_PROMISE_FRACTION = 0.25      # a promise below this share of the amount due goes to a human instead

# The call
BANK_NAME = "Kosha Finance"      # made up
CUSTOMER_ID = int(os.environ.get("CUSTOMER", "1"))          # who the browser call is placed to; 1 = Rahul
ALLOW_ANY_HOUR = os.environ.get("ALLOW_ANY_HOUR") == "1"    # dev only: skip the calling-hours check

# Outputs
CALLS_DIR = ROOT / "calls"
METRICS_DIR = ROOT / "metrics"
