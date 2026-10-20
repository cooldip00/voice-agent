"""The test bench's own logic: scenario files, the caller's choice of answer, and the scoring. No models."""

import copy
from datetime import date

import pytest

from agent import bank
from agent.call_state import DISPUTE_KINDS, OUTCOMES
from testbench import scenario as scenarios
from testbench.score import debt_before_verified, expected_callback, expected_day, score

TODAY = date(2026, 10, 20)   # a Tuesday
CUSTOMER_IDS = {c[0] for c in bank.CUSTOMERS}

RECORD = {
    "outcome": "promise_to_pay", "ended_by": "agent", "promise_date": "2026-10-23", "amount": 4500,
    "callback": None, "dispute": None, "escalation": None, "notice_given": True, "verified": True,
    "verify_attempts": 0,
    "transcript": [
        {"role": "bot", "text": "Hello, this is Asha from Kosha Finance. This call is recorded. Am I speaking with Rahul Sharma?"},
        {"role": "caller", "text": "Yes, this is Rahul speaking."},
        {"role": "bot", "text": "Could you please tell me your date of birth?"},
        {"role": "caller", "text": "It's the fifth of March, nineteen ninety."},
        {"role": "bot", "text": "Thank you for confirming. 4500 rupees is due on your personal loan since 2026-10-06."},
        {"role": "bot", "text": "Thank you. I have recorded 4500 rupees by 2026-10-23. Is there anything else?"},
        {"role": "bot", "text": "Thank you for your time. Goodbye."},
    ],
}
HAPPY = {"outcome": "promise_to_pay", "verified": True, "amount": 4500, "promise_day": "friday"}
ROWS = {"promises": 1, "callbacks": 0, "disputes": 0, "escalations": 0}


def failed(checks):
    return {c.name for c in checks if not c.ok}


# --- scenario files

@pytest.mark.parametrize("s", scenarios.load_all(), ids=lambda s: s.name)
def test_every_scenario_is_runnable(s):
    assert s.customer in CUSTOMER_IDS
    assert s.rules and s.goal
    assert all(r.when.pattern.isprintable() for r in s.rules), "TOML read a \\b as a backspace: write \\\\b"
    outcomes = s.expect["outcome"] if isinstance(s.expect["outcome"], list) else [s.expect["outcome"]]
    assert set(outcomes) <= set(OUTCOMES)
    assert s.expect.get("dispute_kind", "other") in DISPUTE_KINDS
    if "promise_day" in s.expect:
        expected_day(s.expect["promise_day"], TODAY)
    if "callback" in s.expect:
        expected_callback(s.expect["callback"], TODAY)


def test_there_are_nine_scenarios():
    assert len(scenarios.load_all()) == 9


def test_unknown_scenario_name_stops():
    with pytest.raises(SystemExit):
        scenarios.load_all(["nope"])


# --- the caller's answers

def happy():
    return scenarios.load(scenarios.SCENARIO_DIR / "happy.toml")


def test_the_first_matching_rule_answers():
    s = happy()
    assert "fifth of March" in s.reply_to("Could you please tell me your date of birth?").say
    # The verified line says "confirming" and asks when: the confirm rule's "confirm\b" must let it through.
    heard = "Thank you for confirming. 4500 rupees is due. When will you be able to pay?"
    assert "this Friday" in s.reply_to(heard).say
    # A read-back also says "able to pay": the confirm rule comes first in the file.
    assert s.reply_to("You said you will be able to pay 4500 rupees this Friday. Is that correct?").say == "Yes, that's right."


def test_nothing_matches_gives_the_fallback():
    assert happy().reply_to("The weather is lovely today.").say == scenarios.FALLBACK


def test_the_same_answer_a_third_time_running_and_the_caller_hangs_up():
    s = happy()
    assert not s.reply_to("Lovely weather.").hang_up
    assert not s.reply_to("Lovely weather.").hang_up
    s.reply_to("Is there anything else?")   # a different answer starts the count again
    replies = [s.reply_to("Lovely weather.") for _ in range(3)]
    assert [r.hang_up for r in replies] == [False, False, True]
    assert replies[-1].say == scenarios.GIVE_UP and scenarios.GIVE_UP in s.lines
    # Not only misses: the bot reading the promise back again and again ends the same way.
    s = happy()
    replies = [s.reply_to("Is that correct?") for _ in range(3)]
    assert [r.hang_up for r in replies] == [False, False, True]


def test_a_list_of_answers_is_used_in_turn_then_the_last_repeats():
    s = scenarios.load(scenarios.SCENARIO_DIR / "wrongdob.toml")
    says = [s.reply_to("What is your date of birth?").say for _ in range(4)]
    assert [x.split(" of ")[0].split()[-1] for x in says] == ["first", "second", "third", "third"]


def test_wait_and_interrupt_happen_only_the_first_time():
    s = scenarios.load(scenarios.SCENARIO_DIR / "silence.toml")
    assert s.reply_to("When will you be able to pay?").wait == 20
    assert s.reply_to("When will you be able to pay?").wait == 0
    s = scenarios.load(scenarios.SCENARIO_DIR / "interrupts.toml")
    assert s.reply_to("Your date of birth, please?").interrupt
    assert s.reply_to("Your date of birth, please?").interrupt is None


def test_every_line_is_made_into_audio_up_front():
    s = scenarios.load(scenarios.SCENARIO_DIR / "interrupts.toml")
    assert "Wait, wait. How much is it exactly?" in s.lines and scenarios.FALLBACK in s.lines


# --- dates, worked out without the bot's code

def test_expected_dates():
    assert expected_day("friday", TODAY) == date(2026, 10, 23)
    assert expected_day("tuesday", TODAY) == date(2026, 10, 27)   # never today: the next one
    assert expected_day("tomorrow", TODAY) == date(2026, 10, 21)
    assert expected_callback("tomorrow 17:00", TODAY) == "2026-10-21T17:00"


# --- scoring

def test_a_good_call_passes_every_check():
    checks = score(HAPPY, RECORD, {}, [1.2, 2.0], ROWS, TODAY)
    assert failed(checks) == set()
    assert {"outcome", "amount", "pay by", "no debt talk before verified", "bank matches record"} <= {c.name for c in checks}


def test_wrong_amount_and_day_fail():
    record = {**RECORD, "amount": 500, "promise_date": "2026-10-27"}
    assert failed(score(HAPPY, record, {}, [], ROWS, TODAY)) == {"amount", "pay by"}


def test_debt_talk_before_the_date_of_birth_matches_fails():
    record = copy.deepcopy(RECORD)
    record["transcript"][2]["text"] = "I'm calling about your personal loan. What is your date of birth?"
    assert "personal loan" in debt_before_verified(record)
    assert "no debt talk before verified" in failed(score(HAPPY, record, {}, [], ROWS, TODAY))


def test_a_talked_over_verified_line_is_not_a_leak():
    # The caller cut into "Thank you for confirming...": it never reached the transcript. The record's
    # count of caller turns at verification still says the amount came after it.
    record = copy.deepcopy(RECORD)
    record["transcript"][4] = {"role": "caller", "text": "Wait, wait. How much is it exactly?"}
    record["transcript"].insert(5, {"role": "bot", "text": "The amount due on your personal loan is 4500 rupees."})
    assert debt_before_verified(record) is not None                                # by the line alone: a false alarm
    assert debt_before_verified({**record, "verified_after_caller_turns": 2}) is None


def test_the_verified_count_does_not_excuse_lines_before_it():
    record = copy.deepcopy(RECORD)
    record["transcript"][2]["text"] = "About your loan: what is your date of birth?"
    assert "your loan" in debt_before_verified({**record, "verified_after_caller_turns": 2})


def test_never_verified_means_no_debt_words_at_all():
    record = {**copy.deepcopy(RECORD), "verified": False}
    # The verified line doesn't count when the record says the caller never verified.
    assert debt_before_verified(record).startswith("Thank you for confirming")


def test_a_threat_fails():
    record = copy.deepcopy(RECORD)
    record["transcript"][-1]["text"] = "If you don't pay we will take legal action."
    assert "no threats" in failed(score(HAPPY, record, {}, [], ROWS, TODAY))


def test_a_slow_reply_and_a_caller_hang_up_fail():
    record = {**RECORD, "ended_by": "caller"}
    assert failed(score(HAPPY, record, {}, [1.0, 5.4], ROWS, TODAY)) == {"replies under 5s", "bot ended the call"}


def test_bank_must_hold_what_the_record_says():
    rows = {**ROWS, "callbacks": 1}   # a callback row the record doesn't mention
    assert failed(score(HAPPY, RECORD, {}, [], rows, TODAY)) == {"bank matches record"}


def test_interrupt_and_silence_checks():
    expect = {**HAPPY, "stop_within": 1.0, "reprompt_within": 12}
    ok = {"interrupt": {"stopped_after": 0.4}, "silence": {"reprompt_after": 8.0}}
    assert failed(score(expect, RECORD, ok, [], ROWS, TODAY)) == set()
    bad = {"interrupt": {"stopped_after": None}, "silence": {"reprompt_after": None}}
    assert failed(score(expect, RECORD, bad, [], ROWS, TODAY)) == {"stops when talked over", "asks after silence"}


def test_any_of_several_outcomes():
    expect = {"outcome": ["callback", "no_resolution"]}
    record = {**RECORD, "outcome": "no_resolution", "amount": None, "promise_date": None}
    assert "outcome" not in failed(score(expect, record, {}, [], {**ROWS, "promises": 0}, TODAY))
