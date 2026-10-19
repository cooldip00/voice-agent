import json

import pytest

from agent.tools import SCHEMAS, TOOL_NAMES, CallTools
from tests.conftest import RAHUL_DOB

LOAN_FIELDS = ("amount_due", "emi", "due_date", "product", "last_payment", "4500", "PL-1001")

GATED = [
    ("get_loan_details", {}),
    ("record_promise", {"amount": 4500, "date": "2026-10-23"}),
    ("flag_dispute", {"kind": "already_paid", "reason": "paid yesterday"}),
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
    for table in ("promises", "disputes"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


@pytest.mark.parametrize("name, args", GATED)
def test_gated_tools_refuse_after_wrong_dob(tools, name, args):
    tools.verify_identity("1990-03-06")
    assert tools.call(name, args)["error"] == "not_verified"


def test_right_dob_unlocks_loan_details(tools):
    assert tools.verify_identity(RAHUL_DOB) == {"ok": True, "verified": True}
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


def test_badly_formatted_dob_is_not_counted(tools, state):
    assert tools.verify_identity("5 March 1990")["error"] == "bad_format"
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
    result = verified.record_promise(4500, "2026-10-23")
    assert result == {"ok": True, "amount": 4500, "pay_by": "2026-10-23"}
    row = conn.execute("SELECT loan_id, call_id, amount, pay_by FROM promises").fetchone()
    assert tuple(row) == ("PL-1001", "test-call", 4500, "2026-10-23")
    assert state.promise == {"amount": 4500, "pay_by": "2026-10-23"}


@pytest.mark.parametrize("amount, day", [(0, "2026-10-23"), (5000, "2026-10-23"), (4500, "2026-10-28")])
def test_invalid_promise_is_refused_and_not_saved(verified, conn, amount, day):
    assert verified.record_promise(amount, day)["error"] == "invalid_promise"
    assert conn.execute("SELECT COUNT(*) FROM promises").fetchone()[0] == 0


def test_flag_dispute(verified, conn):
    assert verified.flag_dispute("already_paid", "  paid on the 15th by UPI  ")["reason"] == "paid on the 15th by UPI"
    assert verified.flag_dispute("lost_job", "x")["error"] == "invalid_dispute"
    assert verified.flag_dispute("other", "   ")["error"] == "invalid_dispute"
    assert conn.execute("SELECT COUNT(*) FROM disputes").fetchone()[0] == 1


# --- no identity needed ------------------------------------------------------------------------

def test_callback_works_unverified_and_reveals_nothing(tools, conn):
    result = tools.schedule_callback("2026-10-21T17:00")
    assert result == {"ok": True, "call_at": "2026-10-21T17:00"}
    assert not leaks_loan(result)
    assert tuple(conn.execute("SELECT customer_id, call_at FROM callbacks").fetchone()) == (1, "2026-10-21T17:00")


def test_callback_outside_hours_refused(tools):
    assert tools.schedule_callback("2026-10-21T21:00")["error"] == "invalid_callback"


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
                          "schedule_callback", "flag_dispute", "end_call"}
    for name in TOOL_NAMES:
        assert callable(getattr(CallTools, name))


def test_unknown_customer_rejected(state, conn):
    state.customer_id = 99
    with pytest.raises(ValueError):
        CallTools(state, conn)
