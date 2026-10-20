"""p50/p95 of each part of the bot's reply time, from the calls in metrics/, one column per setup.

A setup is the LLM, Whisper model and VAD stop_secs a call ran with (the first line of each file), so
runs with different settings never mix. Below the latency: what the same calls did (from calls/), for
judging a faster setup on more than speed.

usage: uv run python -m scripts.latency_report [metrics/*.jsonl metrics/before-split/*.jsonl ...]
"""

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from agent import config

# In the order they happen in a reply; anything else Pipecat reports goes after, in the order first seen.
STAGES = ["endpointing_wait", "transcription", "turn_detection", "llm_inference", "llm_tool_call",
          "function_handler", "turn_completion", "sentence_aggregation", "speech_synthesis",
          "output_transport", "pipeline"]


def percentile(values: list[float], q: float) -> float:
    """The q-th percentile (0-100), interpolating between the two nearest values, like numpy's default."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    rank = (len(ordered) - 1) * q / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def setup_name(setup: dict) -> str:
    return f"{setup['llm']}, vad {setup['vad_stop_secs']}s, {setup['stt'].split('/')[-1]}"


def load(paths: list[Path]) -> dict[str, dict]:
    """setup name -> {"replies": [turn rows], "greetings": [...], "calls": [call ids], "gpu": [rows]}."""
    runs: dict[str, dict] = defaultdict(lambda: {"replies": [], "greetings": [], "calls": [], "gpu": []})
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or rows[0].get("kind") != "setup":
            continue
        # A subfolder of metrics/ is an archived experiment: its own column, even with the same settings.
        folder = "" if path.parent.resolve() == config.METRICS_DIR.resolve() else f" [{path.parent.name}]"
        run = runs[setup_name(rows[0]) + folder]
        run["calls"].append(rows[0]["call_id"])
        for row in rows[1:]:
            if row["kind"] == "turn":
                run["replies" if row["from"] == "user_silence" else "greetings"].append(row)
            elif row["kind"] == "gpu":
                run["gpu"].append(row)
    return dict(runs)


def stage_cell(replies: list[dict], key: str) -> str:
    values = [r["stages"][key] for r in replies if key in r["stages"]]
    if not values:
        return "-"
    shown = f"{percentile(values, 50) * 1000:.0f} / {percentile(values, 95) * 1000:.0f}"
    return shown if len(values) == len(replies) else f"{shown} ({len(values)}x)"


def latency_table(runs: dict[str, dict]) -> list[str]:
    names = list(runs)
    seen = {k for run in runs.values() for r in run["replies"] for k in r["stages"]}
    keys = [k for k in STAGES if k in seen] + sorted(seen - set(STAGES))
    lines = ["| ms, p50 / p95 | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for key in keys:
        lines.append(f"| {key.replace('_', ' ')} | " + " | ".join(stage_cell(runs[n]["replies"], key) for n in names) + " |")
    def total(replies: list[dict], bold: bool = False) -> str:
        values = [r["total"] for r in replies]
        if not values:
            return "-"
        cell = f"{percentile(values, 50) * 1000:.0f} / {percentile(values, 95) * 1000:.0f}"
        return f"**{cell}**" if bold else f"{cell} ({len(values)}x)"

    lines.append("| **voice to voice** | " + " | ".join(total(runs[n]["replies"], bold=True) for n in names) + " |")
    # A reply that calls a tool waits for the whole call to be written: about 3x the LLM time of a plain
    # one. Split, so a run with more tool replies doesn't look slower than it is.
    lines.append("| - plain replies | " + " | ".join(total([r for r in runs[n]["replies"] if not r["tools"]])
                                                   for n in names) + " |")
    lines.append("| - after a tool call | " + " | ".join(total([r for r in runs[n]["replies"] if r["tools"]])
                                                        for n in names) + " |")
    lines.append("| replies (calls) | " + " | ".join(f"{len(runs[n]['replies'])} ({len(runs[n]['calls'])})" for n in names) + " |")
    return lines


def behaviour_table(runs: dict[str, dict], calls_dir: Path) -> list[str]:
    names = list(runs)
    lines = ["| per call | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    cells: dict[str, list[str]] = defaultdict(list)
    for n in names:
        records = [json.loads(p.read_text(encoding="utf-8")) for c in runs[n]["calls"]
                   if (p := calls_dir / f"{c}.json").exists()]
        tool_calls = [t for r in records for t in r["tool_calls"]]
        refused = sum(1 for t in tool_calls if not t["result"].get("ok"))
        gpu = [g["used_mb"] for g in runs[n]["gpu"] if g.get("used_mb")]
        ollama = {m["model"]: m for g in runs[n]["gpu"] for m in g["ollama"]}
        per = max(len(records), 1)
        cells["outcomes"].append(", ".join(f"{k} {v}" for k, v in Counter(r["outcome"] for r in records).most_common()))
        cells["tool calls"].append(f"{len(tool_calls) / per:.1f} ({refused} refused)")
        cells["LLM slips"].append(f"{sum(len(r['llm_slips']) for r in records) / per:.1f}")
        cells["GPU used, MB"].append(str(max(gpu)) if gpu else "-")
        cells["LLM in VRAM, MB"].append(", ".join(f"{m['vram_mb']} of {m['size_mb']}" for m in ollama.values()) or "-")
    for label, row in cells.items():
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    return lines


def main(args: list[str]) -> None:
    paths = [Path(p) for a in args for p in glob.glob(a)] if args else sorted(config.METRICS_DIR.glob("*.jsonl"))
    runs = load(paths)
    if not runs:
        print(f"No calls measured yet in {config.METRICS_DIR}; make a call with the bot first.")
        return
    print("\n".join(latency_table(runs)))
    print("\nA stage that only happens in some replies shows how many, like (5x). Each column's p50s don't add up "
          "to its total: a percentile of a sum is not the sum of the percentiles.\n")
    print("\n".join(behaviour_table(runs, config.CALLS_DIR)))


if __name__ == "__main__":
    main(sys.argv[1:])
