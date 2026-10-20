import json

import pytest

from agent.tools import SCHEMAS, TOOL_NAMES, CallTools
from tests.conftest import RAHUL_DOB

LOAN_FIELDS = ("amount_due", "emi", "due_date", "product", "last_payment", "4500", "PL-1001")

GATED = [
    ("get_loan_details", {}),
    ("record_promise", {"amount": 4500, "day": "Friday"}),
    ("flag_dispute", {"kind": "already_paid", "reason": "paid yesterday"}),
    ("escalate_to_human", {"reason": "lost his job"}),
]


def leaks_loan(result: dict) -> bool:
    text = json.dumps(result)
    return any(field in text for field in LOAN_FIELDS)


# --- identity gate -----------------------------------------------------------------------------

@pytest.mark.parametrize("name, args", GATED)
def test_gated_tools_refuse_before_verification(tools, conn, name, args):
    result = tools.call(name, args)
    assert result["error"] == "not_verified"
    assert not leaks_loan(result)
    # nothing was written either
    for table in ("promises", "disputes", "escalations"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


@pytest.mark.parametrize("name, args", GATED)
def test_gated_tools_refuse_after_wrong_dob(tools, name, args):
    tools.verify_identity("1990-03-06")
    assert tools.call(name, args)["error"] == "not_verified"


def test_right_dob_unlocks_loan_details(tools):
    result = tools.verify_identity(RAHUL_DOB)
    assert result["verified"] and result["loan"]["amount_due"] == 4500   # the loan comes with the success
    loan = tools.get_loan_details()
    assert loan["ok"] and loan["amount_due"] == 4500 and loan["days_overdue"] == 14
    assert loan["customer_name"] == "Rahul Sharma"


def test_wrong_dob_counts_down_then_locks(tools):
    first = tools.verify_identity("1990-03-06")
    assert first["error"] == "no_match" and first["attempts_left"] == 2
    assert tools.verify_identity("1991-03-05")["attempts_left"] == 1
    assert tools.verify_identity("1990-05-03")["error"] == "locked"
    # once locked, even the right date of birth is refused, and so is everything gated
    assert tools.verify_identity(RAHUL_DOB)["error"] == "locked"
    assert tools.get_loan_details()["error"] == "locked"


def test_no_identity_check_before_the_recording_notice(tools, state):
    state.notice_given = False   # the caller talked over the greeting
    result = tools.verify_identity(RAHUL_DOB)
    assert result["error"] == "notice_not_given" and "recorded" in result["message"]
    assert not state.verified and state.verify_attempts == 0
    state.notice_given = True
    assert tools.verify_identity(RAHUL_DOB)["verified"]


def test_dob_as_the_caller_said_it(tools, state):
    assert tools.verify_identity("5th of March, 1990")["verified"]


def test_dob_without_a_year_is_not_counted(tools, state):
    result = tools.verify_identity("5th of March")
    assert result["error"] == "bad_format" and "year" in result["message"]
    assert state.verify_attempts == 0


def test_failed_check_never_hints_at_the_real_dob(tools):
    for guess in ("1990-03-06", "1990-04-05"):
        assert RAHUL_DOB not in json.dumps(tools.verify_identity(guess))


def test_another_customers_dob_does_not_verify(tools):
    assert tools.verify_identity("1988-11-21")["error"] == "no_match"   # Priya's, customer 2


def test_no_tool_takes_a_customer_or_loan_id():
    for schema in SCHEMAS:
        assert not any("id" in p.lower() for p in schema.properties), schema.name


# --- after verification ------------------------------------------------------------------------

def test_record_promise_saves_to_bank(verified, conn, state):
    result = verified.record_promise(4500, "the 23rd")
    assert result == {"ok": True, "amount": 4500, "pay_by": "2026-10-23"}
    row = conn.execute("SELECT loan_id, call_id, amount, pay_by FROM promises").fetchone()
    assert tuple(row) == ("PL-1001", "test-call", 4500, "2026-10-23")
    assert state.promise == {"amount": 4500, "pay_by": "2026-10-23"}


@pytest.mark.parametrize("amount, day", [(0, "Friday"), (500, "Friday"), (5000, "Friday"),
                                         (4500, "the 28th"),        # 8 days away
                                         (4500, "2026-10-23")])     # the LLM's date, not the caller's words
def test_invalid_promise_is_refused_and_not_saved(verified, conn, amount, day):
    assert verified.record_promise(amount, day)["error"] == "invalid_promise"
    assert conn.execute("SELECT COUNT(*) FROM promises").fetchone()[0] == 0


def test_flag_dispute(verified, conn):
    assert verified.flag_dispute("already_paid", "  paid on the 15th by UPI  ")["reason"] == "paid on the 15th by UPI"
    assert verified.flag_dispute("lost_job", "x")["error"] == "invalid_dispute"
    assert verified.flag_dispute("other", "   ")["error"] == "invalid_dispute"
    assert conn.execute("SELECT COUNT(*) FROM disputes").fetchone()[0] == 1


def test_escalate_then_end_call(verified, conn, state):
    assert verified.escalate_to_human("  ")["error"] == "invalid_escalation"
    assert verified.end_call("escalated")["error"] == "invalid_outcome"
    assert verified.escalate_to_human("lost his job, can pay 500 this week")["ok"]
    assert tuple(conn.execute("SELECT loan_id, reason FROM escalations").fetchone()) == (
        "PL-1001", "lost his job, can pay 500 this week")
    assert verified.end_call("escalated") == {"ok": True, "outcome": "escalated"}
    assert state.outcome == "escalated"


# --- no identity needed ------------------------------------------------------------------------

def test_callback_works_unverified_and_reveals_nothing(tools, conn):
    result = tools.schedule_callback("tomorrow", "5 pm")
    assert result == {"ok": True, "call_at": "2026-10-21T17:00"}
    assert not leaks_loan(result)
    assert tuple(conn.execute("SELECT customer_id, call_at FROM callbacks").fetchone()) == (1, "2026-10-21T17:00")


def test_callback_outside_hours_refused(tools):
    assert tools.schedule_callback("tomorrow", "21:00")["error"] == "invalid_callback"


def test_end_call_checks_outcome(tools, state):
    assert tools.end_call("promise_to_pay")["error"] == "invalid_outcome"
    assert tools.end_call("wrong_person") == {"ok": True, "outcome": "wrong_person"}
    assert state.outcome == "wrong_person"


# --- dispatcher --------------------------------------------------------------------------------

def test_call_by_name_logs_every_call(tools, state):
    tools.call("verify_identity", {"date_of_birth": RAHUL_DOB})
    tools.call("get_loan_details", {})
    assert [c["name"] for c in state.tool_calls] == ["verify_identity", "get_loan_details"]
    assert state.tool_calls[1]["result"]["ok"]


def test_call_unknown_tool_or_bad_arguments(tools):
    assert tools.call("delete_loan", {})["error"] == "unknown_tool"
    assert tools.call("_gate", {})["error"] == "unknown_tool"         # private methods aren't tools
    assert tools.call("verify_identity", {})["error"] == "bad_arguments"
    assert tools.call("verify_identity", {"dob": RAHUL_DOB})["error"] == "bad_arguments"


def test_schemas_match_methods():
    assert TOOL_NAMES == {"verify_identity", "get_loan_details", "record_promise",
                          "schedule_callback", "flag_dispute", "escalate_to_human", "end_call"}
    for name in TOOL_NAMES:
        assert callable(getattr(CallTools, name))


def test_unknown_customer_rejected(state, conn):
    state.customer_id = 99
    with pytest.raises(ValueError):
        CallTools(state, conn)


def test_promise_by_weekday_gets_its_date_from_code(verified, state):
    assert verified.record_promise(4500, "this Friday")["pay_by"] == "2026-10-23"
    assert state.promise == {"amount": 4500, "pay_by": "2026-10-23"}


def test_llm_date_is_replaced_by_the_callers_day(verified, state):
    # Qwen passed today's date for "this Friday"; the caller's words decide.
    state.caller_turns = ["Okay, I can pay the full amount this Friday.", "Yes, that's right."]
    assert verified.record_promise(4500, "2026-10-20")["pay_by"] == "2026-10-23"
    assert state.llm_slips == ["day not in the caller's words: '2026-10-20', used the caller's 2026-10-23"]


def test_llm_date_is_refused_when_the_caller_named_two_days(verified, conn, state):
    state.caller_turns = ["Not today, maybe Friday.", "Yes."]
    assert verified.record_promise(4500, "2026-10-20")["error"] == "invalid_promise"
    assert conn.execute("SELECT COUNT(*) FROM promises").fetchone()[0] == 0


def test_amount_the_caller_never_said_is_refused(verified, conn, state):
    # Step 5, live: the caller offered 500, Qwen read back the prompt's example amount, "Yes, please." recorded it.
    state.caller_turns = ["I lost my job. I can only pay 2000 rupees this week.",
                          "No, I really can't pay more than that right now.", "Yes, please."]
    result = verified.record_promise(4500, "Friday")
    assert result["error"] == "invalid_promise" and "2000" in result["message"]
    assert conn.execute("SELECT COUNT(*) FROM promises").fetchone()[0] == 0
    assert state.llm_slips == ["amount not in the caller's words: 4500, caller said [2000]"]
    assert verified.record_promise(2000, "Friday")["ok"]


def test_a_day_alone_promises_the_amount_due(verified, state):
    # "4500 rupees is due. When can you pay?" "This Friday." means all of it, and only that.
    state.caller_turns = ["This Friday.", "Yes."]
    assert "hasn't said an amount" in verified.record_promise(3000, "Friday")["message"]
    assert verified.record_promise(4500, "Friday")["ok"]


def test_birth_year_is_not_an_amount(tools, state):
    state.caller_turns = ["Yes, speaking.", "It's the 5th of March, 1990."]
    assert tools.verify_identity(RAHUL_DOB)["verified"]
    state.caller_turns += ["Friday is fine.", "Yes."]
    assert tools.record_promise(4500, "Friday")["ok"]


def test_callback_day_from_the_callers_words(tools, state):
    state.caller_turns = ["Call her tomorrow at five in the evening."]
    assert tools.schedule_callback("2026-10-22", "17:00")["call_at"] == "2026-10-21T17:00"


def test_callback_needs_a_day_and_time_it_can_read(tools):
    assert tools.schedule_callback("later", "17:00")["error"] == "invalid_callback"
    assert tools.schedule_callback("2026-10-21", "17:00")["error"] == "invalid_callback"
    assert tools.schedule_callback("Friday", "evening")["error"] == "invalid_callback"
