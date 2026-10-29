from agent import prompts
from agent.call_state import CallState, unsafe_to_say
from agent.speech import speakable
from tests.conftest import TODAY

LOAN = {"customer_name": "Rahul Sharma", "product": "personal loan", "emi": 4500, "amount_due": 4500,
        "due_date": "2026-10-06", "days_overdue": 14, "last_payment": None}


def test_greeting_has_the_notice_and_no_debt():
    line = prompts.greeting("Rahul Sharma")
    assert prompts.is_notice(line)
    assert unsafe_to_say(line, CallState(call_id="t", customer_id=1)) is None


def test_lines_spoken_by_code_come_out_in_words():
    assert speakable(prompts.after_verify({"ok": True, "loan": LOAN}), TODAY) == (
        "Thank you for confirming. four thousand five hundred rupees is due on your personal loan "
        "since the sixth of October. When will you be able to pay?")
    assert speakable(prompts.after_promise({"ok": True, "amount": 4500, "pay_by": "2026-10-23"}), TODAY) == (
        "Thank you. I have recorded four thousand five hundred rupees by Friday, the twenty-third of October. "
        "Is there anything else?")


def test_system_prompt_lists_no_dates():
    # With a list of dates Qwen 7B still picked wrong ones; tools now take the caller's own words.
    text = prompts.system_prompt("Rahul Sharma", TODAY)
    assert "Today is Tuesday 2026-10-20." in text and "2026-10-23" not in text


def test_farewell():
    for text in ("Have a good day, Rahul.", "Goodbye!", "The call will now end.", "This call has ended.",
                 "Take care.", prompts.GOODBYE):
        assert prompts.is_farewell(text), text
    for text in ("Is there anything else?", "When will you be able to pay?", prompts.greeting("Rahul Sharma"),
                 "I'll take care of the payment on Friday."):
        assert not prompts.is_farewell(text), text


def test_lines_that_close_the_call():
    # Said by code before GOODBYE. Not farewells themselves (the guard would stop them once the outcome is
    # set), nothing about the debt (a callback may be for a third party), and readable aloud.
    callback = {"ok": True, "call_at": "2026-10-21T17:00"}
    state = CallState(call_id="t", customer_id=1, tool_calls=[{"name": "schedule_callback", "result": callback}])
    lines = [line(callback) for _, line in prompts.CLOSES_CALL.values()]
    for text in lines + [prompts.LOCKED, prompts.IDLE_END, *prompts.STILL_THERE]:
        assert not prompts.is_farewell(text), text
        assert unsafe_to_say(text, state) is None, text
    assert {outcome for outcome, _ in prompts.CLOSES_CALL.values()} == {"callback", "dispute", "escalated"}
    assert speakable(prompts.after_callback({"call_at": "2026-10-21T17:00"}), TODAY) == (
        "Thank you. We will call you back Wednesday, the twenty-first of October, at five PM.")


def test_identity_prompt_and_tools_say_nothing_about_the_debt():
    # Before the DOB matches the LLM must not know why we call: it can't leak what it doesn't know.
    import json
    import re

    from agent.call_state import _DEBT_WORDS
    from agent.tools import IDENTITY_SCHEMAS, LOCKED

    assert not _DEBT_WORDS.search(prompts.identity_prompt("Anjali Gupta", TODAY))
    assert not _DEBT_WORDS.search(json.dumps([s.to_default_dict() for s in IDENTITY_SCHEMAS]))
    assert not _DEBT_WORDS.search(LOCKED["message"])
    assert {s.name for s in IDENTITY_SCHEMAS} == {"verify_identity", "schedule_callback", "end_call"}
    assert re.search(r"overdue EMI", prompts.system_prompt("Anjali Gupta", TODAY))


def test_personal_matter_line_passes_the_guard_before_verification():
    line = prompts.personal_matter("Anjali Gupta")
    assert unsafe_to_say(line, CallState(call_id="t", customer_id=6)) is None
