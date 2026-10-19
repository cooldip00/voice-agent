from datetime import datetime

import pytest

from agent.call_state import (
    CallState,
    as_rupees,
    check_callback,
    check_outcome,
    check_promise,
    parse_date,
    within_call_hours,
)
from tests.conftest import NOW, TODAY


@pytest.mark.parametrize(
    "hour, minute, allowed",
    [(7, 59, False), (8, 0, True), (18, 59, True), (19, 0, False), (23, 0, False)],
)
def test_call_hours(hour, minute, allowed):
    assert within_call_hours(datetime(2026, 10, 20, hour, minute)) is allowed


@pytest.mark.parametrize("text", ["2026-10-20"])
def test_parse_date_accepts_iso(text):
    assert parse_date(text) == TODAY


@pytest.mark.parametrize("text", ["20261020", "20/10/2026", "next Friday", "2026-02-30", "", None, 20261020])
def test_parse_date_rejects_everything_else(text):
    assert parse_date(text) is None


@pytest.mark.parametrize("value, rupees", [(4500, 4500), (4500.0, 4500), ("4500", 4500), ("4,500", 4500)])
def test_as_rupees_accepts(value, rupees):
    assert as_rupees(value) == rupees


@pytest.mark.parametrize("value", [4500.5, "four thousand", True, None, [4500]])
def test_as_rupees_rejects(value):
    assert as_rupees(value) is None


@pytest.mark.parametrize(
    "amount, day",
    [(4500, "2026-10-20"), (1, "2026-10-27"), (2000, "2026-10-23")],
)
def test_promise_ok(amount, day):
    assert check_promise(amount, day, TODAY, amount_due=4500) is None


@pytest.mark.parametrize(
    "amount, day, problem",
    [
        (0, "2026-10-23", "at least 1"),
        (-100, "2026-10-23", "at least 1"),
        (4501, "2026-10-23", "more than"),
        (4500.5, "2026-10-23", "whole number"),
        (4500, "2026-10-19", "past"),
        (4500, "2026-10-28", "within 7 days"),  # day 8
        (4500, "23/10/2026", "YYYY-MM-DD"),
    ],
)
def test_promise_rejected(amount, day, problem):
    assert problem in check_promise(amount, day, TODAY, amount_due=4500)


@pytest.mark.parametrize("when", ["2026-10-20T17:00", "2026-10-21T08:00", "2026-10-27T10:00"])
def test_callback_ok(when):
    assert check_callback(when, NOW) is None


@pytest.mark.parametrize(
    "when, problem",
    [
        ("2026-10-20T10:00", "past"),
        ("2026-10-20T11:00", "past"),          # exactly now
        ("2026-10-20T19:00", "between"),
        ("2026-10-21T07:30", "between"),
        ("2026-10-28T10:00", "within 7 days"),
        ("tomorrow at 5", "YYYY-MM-DDTHH:MM"),
    ],
)
def test_callback_rejected(when, problem):
    assert problem in check_callback(when, NOW)


def test_outcome_must_match_what_was_recorded():
    s = CallState(call_id="c", customer_id=1)
    assert "no promise" in check_outcome(s, "promise_to_pay")
    assert "no callback" in check_outcome(s, "callback")
    assert "no dispute" in check_outcome(s, "dispute")
    assert "verified customer" in check_outcome(s, "refused")
    assert check_outcome(s, "wrong_person") is None
    assert check_outcome(s, "no_resolution") is None
    assert "one of" in check_outcome(s, "paid_in_full")

    s.verified = True
    assert "not the wrong person" in check_outcome(s, "wrong_person")
    assert check_outcome(s, "refused") is None
    s.promise = {"amount": 4500, "pay_by": "2026-10-23"}
    assert check_outcome(s, "promise_to_pay") is None


def test_lock_after_three_wrong_attempts():
    s = CallState(call_id="c", customer_id=1, verify_attempts=2)
    assert not s.locked
    s.verify_attempts = 3
    assert s.locked
    s.verified = True
    assert not s.locked
