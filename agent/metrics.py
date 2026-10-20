"""How long each bot reply took, and where the time went: one JSON line per reply in metrics/<call_id>.jsonl.

Pipecat's UserBotLatencyObserver does the timing. For each reply it splits the wait, from the caller
falling silent to the bot's first audio, into named parts that add up to the total: endpointing wait
(VAD stop_secs), transcription, turn detection (Smart Turn), LLM inference, tool call, speech synthesis,
output transport. This file stores them; scripts/latency_report.py turns them into p50/p95.

Not measured: the network and the browser's playback buffer after the bot sends its first audio.
"""

import json
import subprocess
from pathlib import Path

import requests
from loguru import logger
from pipecat.observers.user_bot_latency_observer import LatencyBreakdown, UserBotLatencyObserver

from agent import config


def turn_row(breakdown: LatencyBreakdown) -> dict:
    """One reply as a JSON-ready dict. Parts with the same key (turn completion can be two) are added up."""
    stages: dict[str, float] = {}
    for c in breakdown.contributions:
        stages[c.key] = round(stages.get(c.key, 0.0) + c.duration_secs, 4)
    return {
        "kind": "turn",
        # "user_silence" is a reply; "client_connected" is the greeting, timed from the browser connecting.
        "from": breakdown.measured_from.value if breakdown.measured_from else None,
        "total": round(breakdown.total_secs, 4),
        "stages": stages,
        "tools": [f.function_name for f in breakdown.function_calls],
    }


def gpu_used_mb() -> int | None:
    """GPU memory in use on the whole card (Whisper + Ollama + anything else), or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def ollama_loaded() -> list[dict]:
    """Models Ollama holds right now, with how much of each sits in VRAM (the rest runs on the CPU)."""
    try:
        models = requests.get(f"{config.OLLAMA_BASE}/api/ps", timeout=5).json().get("models", [])
    except (requests.RequestException, ValueError):
        return []
    return [{"model": m["name"], "size_mb": m["size"] // 2**20, "vram_mb": m.get("size_vram", 0) // 2**20}
            for m in models]


class LatencyLog:
    """Writes one call's replies to metrics/<call_id>.jsonl. The first line says what was measured."""

    def __init__(self, call_id: str, directory: Path = config.METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{call_id}.jsonl"
        self.call_id = call_id
        self.observer = UserBotLatencyObserver()
        self.observer.add_event_handler("on_latency_breakdown", self._on_breakdown)
        self._write({"kind": "setup", "llm": config.LLM_MODEL, "stt": config.STT_MODEL,
                     "vad_stop_secs": config.VAD_STOP_SECS, "tts_device": config.TTS_DEVICE})

    def _write(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"call_id": self.call_id, **row}) + "\n")

    async def _on_breakdown(self, observer, breakdown: LatencyBreakdown) -> None:
        if not breakdown.contributions:
            return
        self._write(turn_row(breakdown))
        logger.info("Reply latency:\n  " + "\n  ".join(breakdown.turn_contribution_lines()))

    def close(self) -> None:
        """Called once the call ends, while the models are still loaded."""
        self._write({"kind": "gpu", "used_mb": gpu_used_mb(), "ollama": ollama_loaded()})
