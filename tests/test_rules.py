import json
from datetime import datetime, timedelta

import pytest

from agent.call_state import (
    CallState,
    amounts_said,
    as_rupees,
    caller_amounts,
    caller_day,
    check_callback,
    check_outcome,
    call_record,
    check_promise,
    hardship_said,
    join_caller_turns,
    parse_date,
    resolve_birth_date,
    resolve_day,
    resolve_time,
    transcript,
    unsafe_to_say,
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
    [(4500, "2026-10-20"), (1125, "2026-10-27"), (2000, "2026-10-23")],   # 1125 = 25% of 4500
)
def test_promise_ok(amount, day):
    assert check_promise(amount, day, TODAY, amount_due=4500) is None


@pytest.mark.parametrize(
    "amount, day, problem",
    [
        (0, "2026-10-23", "at least 1"),
        (-100, "2026-10-23", "at least 1"),
        (1124, "2026-10-23", "below the minimum of 1125"),
        (500, "2026-10-23", "human agent"),
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
        ("2026-10-20T19:00", "before 19:00"),
        ("2026-10-21T07:30", "from 8:00"),
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
    assert "nothing was escalated" in check_outcome(s, "escalated")
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


def test_minimum_rounds_up():
    assert check_promise(525, "2026-10-23", TODAY, amount_due=2100) is None       # 25% of 2100 = 525
    assert "minimum of 26" in check_promise(25, "2026-10-23", TODAY, amount_due=101)  # 25.25 -> 26


# --- end-of-call record ------------------------------------------------------------------------

MESSAGES = [
    {"role": "system", "content": "You are Asha"},
    {"role": "assistant", "content": "Hello, this is Asha from Kosha Finance."},
    {"role": "user", "content": "Yes, speaking."},
    {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "verify_identity"}}]},
    {"role": "tool", "tool_call_id": "1", "content": '{"ok": true}'},
    {"role": "assistant", "content": "   "},
]


def test_transcript_keeps_only_spoken_turns():
    assert transcript(MESSAGES) == [
        {"role": "bot", "text": "Hello, this is Asha from Kosha Finance."},
        {"role": "caller", "text": "Yes, speaking."},
    ]


def test_transcript_leaves_out_silenced_sentences():
    messages = [{"role": "assistant", "content": "I understand. It's about your overdue EMI. When is good?"},
                {"role": "assistant", "content": "It's about your overdue EMI."}]
    assert transcript(messages, ["It's about your overdue EMI. "]) == [
        {"role": "bot", "text": "I understand. When is good?"}]


def test_call_record_comes_from_state_not_from_the_llm():
    s = CallState(call_id="c", customer_id=1, verified=True, notice_given=True, outcome="promise_to_pay",
                  promise={"amount": 4500, "pay_by": "2026-10-23"})
    # the LLM claiming something in the transcript changes nothing in the record
    said = MESSAGES + [{"role": "assistant", "content": "I've noted 9000 rupees by next month."}]
    record = call_record(s, said, NOW, NOW + timedelta(seconds=95))
    assert {k: record[k] for k in ("outcome", "promise_date", "amount", "reason", "callback")} == {
        "outcome": "promise_to_pay", "promise_date": "2026-10-23", "amount": 4500, "reason": None, "callback": None,
    }
    assert record["ended_by"] == "agent" and record["seconds"] == 95.0
    json.dumps(record)   # must be serializable


def test_call_record_when_the_caller_hangs_up():
    s = CallState(call_id="c", customer_id=1, escalation="lost his job")
    record = call_record(s, [], NOW, NOW)
    # no end_call, so the outcome comes from what the tools recorded
    assert record["outcome"] == "escalated" and record["ended_by"] == "caller"
    assert record["reason"] == "lost his job" and record["amount"] is None
    assert call_record(CallState(call_id="c", customer_id=1), [], NOW, NOW)["outcome"] == "no_resolution"


@pytest.mark.parametrize("text", [
    "I'm calling about your loan.",
    "Your EMI of 4500 rupees is overdue.",
    "It's regarding a payment.",
])
def test_no_debt_talk_before_verification(text):
    assert "before identity" in unsafe_to_say(text, CallState(call_id="t", customer_id=1))
    assert unsafe_to_say(text, CallState(call_id="t", customer_id=1, verified=True)) is None


def test_no_format_talk():
    assert "format" in unsafe_to_say("Please say it in YYYY-MM-DD format.", CallState(call_id="t", customer_id=1))


def test_no_claim_before_a_tool_recorded_it():
    state = CallState(call_id="t", customer_id=1, verified=True)
    assert "claims" in unsafe_to_say("I've recorded your promise.", state)
    assert unsafe_to_say("Please note, this call is recorded.", state) is None
    assert unsafe_to_say("So 4500 rupees by Friday, is that right?", state) is None
    state.promise = {"amount": 4500, "pay_by": "2026-10-23"}
    assert unsafe_to_say("I've recorded your promise.", state) is None



def test_no_dates_that_no_tool_gave():
    state = CallState(call_id="t", customer_id=1, verified=True)
    assert "date no tool gave" in unsafe_to_say("Can you pay by next Friday, 2026-10-27?", state)
    state.tool_calls.append({"name": "record_promise", "arguments": {},
                             "result": {"ok": True, "amount": 4500, "pay_by": "2026-10-23"}})
    assert unsafe_to_say("That is 4500 rupees by 2026-10-23.", state) is None
    assert unsafe_to_say("Or by 2026-10-27?", state) is not None

@pytest.mark.parametrize("day, expected", [
    ("today", "2026-10-20"), ("Tomorrow", "2026-10-21"),
    ("Friday", "2026-10-23"), ("this friday", "2026-10-23"), ("next Friday", "2026-10-23"),
    ("Tuesday", "2026-10-27"),          # today is a Tuesday, so "Tuesday" means next week's
    ("Friday 2026-10-27", "2026-10-23"),   # the caller's word wins over the LLM's date
    ("the 23rd", "2026-10-23"), ("on the 23rd", "2026-10-23"), ("23rd of October", "2026-10-23"),
    ("23 October", "2026-10-23"), ("October 23rd", "2026-10-23"), ("the 20th", "2026-10-20"),
    ("the 5th", "2026-11-05"),          # the 5th has passed this month, so next month's
    ("January 3rd", "2027-01-03"),      # January has passed this year, so next year's
])
def test_resolve_day(day, expected):
    assert resolve_day(day, TODAY).isoformat() == expected


@pytest.mark.parametrize("day", ["someday", "23/10/2026", "", "Fri", "23", "the 32nd", "the 31st of November",
                                 "2026-10-23"])   # no YYYY-MM-DD: the LLM passed today's date for "Friday"
def test_resolve_day_rejects(day):
    assert resolve_day(day, TODAY) is None


@pytest.mark.parametrize("turns, expected", [
    (["I can pay the full amount this Friday.", "Yes, that's right."], "2026-10-23"),   # the turn before "yes"
    (["Okay, call her tomorrow at five in the evening."], "2026-10-21"),
    (["I can pay on the 23rd.", "Sorry, no, Thursday.", "Yes."], "2026-10-22"),          # the newest day wins
    (["Friday the 23rd is fine."], "2026-10-23"),                                        # two words, one day
])
def test_caller_day(turns, expected):
    assert caller_day(turns, TODAY).isoformat() == expected


@pytest.mark.parametrize("text, expected", [
    ("I can only pay 500 rupees this week.", {500}),
    ("4,500 this Friday", {4500}),
    ("four thousand five hundred", {4500}),
    ("forty five hundred", {4500}),
    ("45 hundred", {4500}),
    ("5k by Monday", {5000}),
    ("1.5 lakh", {150_000}),
    ("two lakh and fifty thousand", {250_000}),
    ("500, maybe 600", {500, 600}),
    ("On the 23rd at 5.", set()),      # a day and a time, not amounts
    ("One moment please.", set()),
])
def test_amounts_said(text, expected):
    assert amounts_said(text) == expected


@pytest.mark.parametrize("turns, expected", [
    (["I can only pay 500 rupees this week.", "No, I really can't pay more.", "Yes, please."], {500}),
    (["I can pay 2000.", "Sorry, make it 3000.", "Yes."], {3000}),         # the newest amount wins
    (["I'll pay the full amount on Friday.", "Yes."], {6800}),
    (["This Friday.", "Yes."], None),                                      # only a day: no amount named
])
def test_caller_amounts(turns, expected):
    assert caller_amounts(turns, 6800) == expected


@pytest.mark.parametrize("turns", [
    ["Not today, maybe Friday.", "Yes."],         # two days: code doesn't guess
    ["Yes, that's right."],                       # no day
    [],
    ["I can pay this Friday.", "Hmm.", "Okay.", "Yes."],   # older than the last 3 turns
])
def test_caller_day_none(turns):
    assert caller_day(turns, TODAY) is None


def test_resolve_day_of_month_across_a_year_end():
    assert resolve_day("the 2nd", TODAY.replace(month=12)).isoformat() == "2027-01-02"
    assert resolve_day("the 31st", TODAY.replace(month=11, day=5)) is None   # November has no 31st


@pytest.mark.parametrize("text", [
    "1990-03-05", "5th of March, 1990", "the 5th of March 1990", "5 March 1990", "March 5th, 1990",
    "March 5 1990", "05/03/1990", "5.3.1990", "5-3-1990", "5th of March. 1990.",
])
def test_resolve_birth_date(text):
    assert resolve_birth_date(text).isoformat() == "1990-03-05"


@pytest.mark.parametrize("text", ["5th of March", "1990", "31st of February 1990", "13/13/1990", "5 Smarch 1990", ""])
def test_resolve_birth_date_rejects(text):
    assert resolve_birth_date(text) is None


@pytest.mark.parametrize("text, expected", [
    ("17:00", (17, 0)), ("5 pm", (17, 0)), ("5:30 PM", (17, 30)), ("10 a.m.", (10, 0)),
    ("5", (17, 0)), ("11", (11, 0)), ("12 pm", (12, 0)),
])
def test_resolve_time(text, expected):
    assert resolve_time(text) == expected


@pytest.mark.parametrize("text", ["evening", "25:00", "5:75", ""])
def test_resolve_time_rejects(text):
    assert resolve_time(text) is None


def test_back_to_back_caller_turns_are_joined():
    messages = [
        {"role": "assistant", "content": "What is your date of birth?"},
        {"role": "user", "content": "14th of July."},
        {"role": "user", "content": " 1985"},
        {"role": "assistant", "content": "Thank you."},
        {"role": "user", "content": "Okay."},
    ]
    joined = join_caller_turns(messages)
    assert [m["content"] for m in joined] == ["What is your date of birth?", "14th of July. 1985", "Thank you.", "Okay."]
    assert messages[1]["content"] == "14th of July."   # the originals are left alone


@pytest.mark.parametrize("text", ["I lost my job last month.", "I'm unemployed right now.", "My salary is delayed.",
                                  "My father is in the hospital.", "I had an accident.", "I was laid off."])
def test_hardship_said(text):
    assert hardship_said(text)


@pytest.mark.parametrize("text", ["I'll pay 4500 on Friday.", "I already paid on the 15th.",
                                  "Call me back tomorrow at five.", "My job keeps me busy, call later."])
def test_not_hardship(text):
    assert not hardship_said(text)


def test_json_is_not_spoken():
    # A bench call had Qwen speak a made-up tool result.
    state = CallState(call_id="t", customer_id=1, verified=True)
    assert unsafe_to_say('"ok": false, "message": "Call ended by user."}', state) == "writes JSON"
    assert unsafe_to_say("Okay, I understand.", state) is None


# --- regressions -------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("the day after tomorrow", TODAY + timedelta(days=2)),   # not tomorrow
    ("a week from today", TODAY + timedelta(days=7)),        # not today
    ("in a week", TODAY + timedelta(days=7)),
    ("the 5th may work", TODAY.replace(month=11, day=5)),    # "may" the verb, not the month
    ("the 5th of May", TODAY.replace(year=2027, month=5, day=5)),
    ("May 5th", TODAY.replace(year=2027, month=5, day=5)),
])
def test_resolve_day_regressions(text, expected):
    assert resolve_day(text, TODAY) == expected


def test_two_days_are_no_answer_even_with_today():
    assert resolve_day("not today, maybe Friday", TODAY) is None
    assert caller_day(["Not today, maybe Friday."], TODAY) is None


def test_a_date_with_a_year_is_not_a_day():
    # "the 5th of March, 1990" is a date of birth, not a day to pay on.
    assert resolve_day("the 5th of March, 1990", TODAY) is None
    assert caller_day(["Friday.", "It's the 5th of March, 1990."], TODAY) == TODAY + timedelta(days=3)


@pytest.mark.parametrize("text, expected", [
    ("I can't pay 4500, only 2000.", {2000}),
    ("I don't have 4500 but I can pay 2000.", {2000}),
    ("I can pay 4500, not 5000.", {4500}),
    ("I can't pay more than 2000.", {2000}),
    ("On the 5th of November, 2026 I'll pay 3000.", {3000}),   # 2026 is a year
    ("I may pay 2000 on Friday.", {2000}),                     # "may" isn't the month
])
def test_amounts_turned_down_or_years_are_not_named(text, expected):
    assert amounts_said(text) == expected


def test_callback_window_counts_days():
    # Asked Tuesday 11:00: next Tuesday 17:00 is 7 days away, as for a promise.
    assert check_callback("2026-10-27T17:00", NOW) is None
    assert "within 7 days" in check_callback("2026-10-28T09:00", NOW)


def test_seven_pm_is_refused_with_words_that_say_why():
    assert resolve_time("7") == (19, 0)
    problem = check_callback("2026-10-21T19:00", NOW)
    assert "before 19:00" in problem and "between" not in problem


def test_no_resolution_or_refused_after_something_was_recorded():
    state = CallState(call_id="t", customer_id=1, verified=True, promise={"amount": 4500, "pay_by": "2026-10-23"})
    for outcome in ("no_resolution", "refused"):
        assert "promise_to_pay" in check_outcome(state, outcome)
    assert check_outcome(state, "promise_to_pay") is None


@pytest.mark.parametrize("text", ["My father died last week.", "I was fired.", "There was a death in the family.",
                                  "I've been at a funeral."])
def test_hardship_said_regressions(text):
    assert hardship_said(text)


def test_today_may_be_said():
    # The prompt gives the LLM today's date; saying it is not inventing one.
    state = CallState(call_id="t", customer_id=1, verified=True)
    assert unsafe_to_say("Today is 2026-10-20.", state, TODAY) is None
    assert unsafe_to_say("Today is 2026-10-20.", state) == "states a date no tool gave"
    assert unsafe_to_say("Pay by 2026-10-27.", state, TODAY) == "states a date no tool gave"
