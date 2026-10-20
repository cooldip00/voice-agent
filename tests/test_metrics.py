"""Latency rows and the report's maths. No models: the breakdowns are built by hand."""

import asyncio
import json

import pytest
from pipecat.observers.user_bot_latency_observer import (
    FunctionCallMetrics,
    LatencyBreakdown,
    LatencyContribution,
    LatencyOwnerKind,
    MeasuredFrom,
)

from agent.metrics import LatencyLog, turn_row
from scripts.latency_report import latency_table, load, percentile


def part(key: str, secs: float) -> LatencyContribution:
    return LatencyContribution(key=key, label=key, owner="x", owner_kind=LatencyOwnerKind.SERVICE,
                               start_time=0.0, duration_secs=secs)


def reply(*parts, tools=()) -> LatencyBreakdown:
    return LatencyBreakdown(
        contributions=list(parts), total_secs=sum(p.duration_secs for p in parts),
        measured_from=MeasuredFrom.USER_SILENCE,
        function_calls=[FunctionCallMetrics(function_name=t, start_time=0.0, duration_secs=0.01) for t in tools])


def test_turn_row_adds_up_repeated_parts():
    row = turn_row(reply(part("transcription", 0.3), part("turn_completion", 0.1), part("turn_completion", 0.05),
                         tools=["record_promise"]))
    assert row == {"kind": "turn", "from": "user_silence", "total": 0.45,
                   "stages": {"transcription": 0.3, "turn_completion": 0.15}, "tools": ["record_promise"]}


def test_greeting_is_marked_apart():
    greeting = LatencyBreakdown(contributions=[part("first_request", 0.2)], total_secs=0.2,
                                measured_from=MeasuredFrom.CLIENT_CONNECTED)
    assert turn_row(greeting)["from"] == "client_connected"


def test_percentile_interpolates():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == pytest.approx(4.8)
    assert percentile([2.0], 95) == 2.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_log_and_report_round_trip(tmp_path):
    log = LatencyLog("call-1", directory=tmp_path)
    for secs in (0.5, 0.7, 0.9):
        asyncio.run(log._on_breakdown(None, reply(part("llm_inference", secs), part("speech_synthesis", 0.1))))
    asyncio.run(log._on_breakdown(None, LatencyBreakdown()))   # nothing measured: no row
    rows = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["setup", "turn", "turn", "turn"]

    runs = load([log.path])
    (name, run), = runs.items()
    assert len(run["replies"]) == 3 and run["calls"] == ["call-1"]
    table = "\n".join(latency_table(runs))
    assert "| llm inference | 700 / 880 |" in table
    assert "| **voice to voice** | **800 / 980** |" in table
