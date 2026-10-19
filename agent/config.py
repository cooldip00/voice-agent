"""Every setting in one place, so experiments change one line."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Hearing (faster-whisper)
STT_MODEL = "Systran/faster-distil-whisper-medium.en"   # English-only; "small.en" if VRAM gets tight
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "int8_float16"

# Brain (Ollama)
OLLAMA_BASE = "http://127.0.0.1:11434"  # not "localhost": on Windows it tries IPv6 first and adds ~2 s per request
LLM_MODEL = "qwen2.5:7b"         # compare with "qwen2.5:3b" in step 5
LLM_MODELS_NEEDED = ["qwen2.5:7b", "qwen2.5:3b"]

# Voice (Kokoro, runs on CPU through ONNX; Pipecat caches its files in ~/.cache/pipecat/kokoro-onnx)
TTS_VOICE = "af_heart"

# Turn-taking
VAD_STOP_SECS = 0.2              # Smart Turn works best with a short VAD stop

# Outputs
CALLS_DIR = ROOT / "calls"
METRICS_DIR = ROOT / "metrics"
