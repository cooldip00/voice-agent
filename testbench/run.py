"""Call the live bot once per scenario (or --runs times), score every call, print the results.

usage: uv run python -m testbench.run [scenario ...] [--runs N] [--no-bot]

Starts the bot itself (unless --no-bot: then one must be running on --url), reseeds the bank before each
call, and saves everything under testbench/results/<time>/: report.json, the bot's log, and one WAV per
call with both sides. About 1.5 minutes per call.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import requests

from agent import bank, config
from scripts.latency_report import percentile
from testbench import scenario as scenarios
from testbench.score import score

RESULTS_DIR = Path(__file__).parent / "results"
BANK_TABLES = ("promises", "callbacks", "disputes", "escalations")


def bot_running(url: str) -> bool:
    try:
        requests.get(url, timeout=2)
        return True
    except requests.RequestException:
        return False


def start_bot(log_path: Path, url: str) -> subprocess.Popen:
    env = {**os.environ, "ALLOW_ANY_HOUR": "1", "PYTHONIOENCODING": "utf-8"}
    log = log_path.open("wb")
    proc = subprocess.Popen([sys.executable, "-m", "agent.bot"], cwd=config.ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT)
    deadline = time.time() + 240
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"The bot stopped while starting; see {log_path}")
        if b"Uvicorn running" in log_path.read_bytes():
            return proc
        time.sleep(0.5)
    proc.kill()
    raise SystemExit(f"The bot didn't start in 4 minutes; see {log_path}")


def wait_for_record(customer: int, since: float, timeout: float = 30) -> dict | None:
    """The record the bot saves when the call ends: the newest calls/*-c<customer>.json written after `since`."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = [p for p in config.CALLS_DIR.glob(f"*-c{customer}.json") if p.stat().st_mtime >= since]
        if found:
            time.sleep(0.5)   # let the write finish
            return json.loads(max(found, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8"))
        time.sleep(0.5)
    return None


def reply_secs(call_id: str) -> list[float]:
    path = config.METRICS_DIR / f"{call_id}.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r["total"] for r in rows if r.get("kind") == "turn" and r.get("from") == "user_silence"]


def bank_rows(call_id: str) -> dict[str, int]:
    conn = bank.connect(config.BANK_DB)
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t} WHERE call_id = ?", (call_id,)).fetchone()[0]
                for t in BANK_TABLES}
    finally:
        conn.close()


async def run(names: list[str], runs: int, url: str, out: Path) -> list[dict]:
    from testbench.caller import Hearing, Voice, call

    chosen = scenarios.load_all(names)
    print("Loading the caller's voice and ears (CPU)...", flush=True)
    voice, hearing = Voice(), Hearing()
    for s in chosen:
        for line in s.lines:
            voice(line)

    results = []
    for r in range(1, runs + 1):
        for s in chosen:
            s = scenarios.load(scenarios.SCENARIO_DIR / f"{s.name}.toml")   # fresh: rules count their uses
            bank.create(config.BANK_DB, date.today()).close()
            started = time.time()
            log = await call(s, voice, hearing, url, out / f"{s.name}-{r}.wav")
            record = wait_for_record(s.customer, started)
            if record is None:
                print(f"FAIL  {s.name}-{r}: no call record (did the bot crash? see bot.log)", flush=True)
                results.append({"scenario": s.name, "run": r, "passed": False, "checks": [], "caller": log})
                save_report(out, results)
                continue
            secs = reply_secs(record["call_id"])
            checks = score(s.expect, record, log, secs, bank_rows(record["call_id"]), date.today())
            passed = all(c.ok for c in checks)
            failed = "; ".join(f"{c.name}: {c.detail}" for c in checks if not c.ok)
            print(f"{'PASS' if passed else 'FAIL'}  {s.name}-{r:<3} {record['outcome']:15s} {failed}", flush=True)
            results.append({"scenario": s.name, "run": r, "call_id": record["call_id"], "passed": passed,
                            "checks": [c.__dict__ for c in checks], "reply_secs": secs, "caller": log})
            save_report(out, results)
    return results


def save_report(out: Path, results: list[dict]) -> None:
    """After every call, so a run stopped halfway (Ctrl+C, a crash) still leaves what it scored."""
    setup = {"llm": config.LLM_MODEL, "stt": config.STT_MODEL, "vad_stop_secs": config.VAD_STOP_SECS,
             "tts_device": config.TTS_DEVICE}
    (out / "report.json").write_text(json.dumps({"setup": setup, "results": results}, indent=2), encoding="utf-8")


def summary(results: list[dict]) -> list[str]:
    by_scenario = defaultdict(list)
    for res in results:
        by_scenario[res["scenario"]].append(res["passed"])
    lines = ["", "| scenario | passed |", "|---|---|"]
    lines += [f"| {name} | {sum(p)} / {len(p)} |" for name, p in by_scenario.items()]
    total = sum(r["passed"] for r in results)
    lines.append(f"| **all** | **{total} / {len(results)}** |")
    failures = Counter(c["name"] for r in results for c in r["checks"] if not c["ok"])
    if failures:
        lines += ["", "Checks that failed: " + ", ".join(f"{name} ({n}x)" for name, n in failures.most_common())]
    secs = [s for r in results for s in r.get("reply_secs", [])]
    if secs:
        lines.append(f"Voice to voice over {len(secs)} replies: p50 {percentile(secs, 50) * 1000:.0f} ms, "
                     f"p95 {percentile(secs, 95) * 1000:.0f} ms")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="*", help="names from testbench/scenarios (default: all)")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--no-bot", action="store_true", help="use the bot already running on --url")
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    args = parser.parse_args()

    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    bot = None
    if not args.no_bot:
        if bot_running(args.url):
            raise SystemExit(f"A bot is already running on {args.url}. Stop it, or pass --no-bot to use it.")
        print("Starting the bot...", flush=True)
        bot = start_bot(out / "bot.log", args.url)
    try:
        results = asyncio.run(run(args.scenarios, args.runs, args.url, out))
    finally:
        if bot:
            bot.terminate()
            bot.wait(timeout=30)
    save_report(out, results)
    print("\n".join(summary(results)))
    print(f"\nSaved in {out}")


if __name__ == "__main__":
    main()
