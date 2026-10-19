"""The six tools the LLM can call during a collections call, and their descriptions for Pipecat.

Every tool acts on the customer this call was placed to (`state.customer_id`); none takes a customer
or loan id from the LLM. Loan details and anything recorded against the loan need a verified identity.
Results are dicts; a refusal is {"ok": False, "error": <code>, "message": <what to do instead>} and
never contains loan data. The tools are wired into the bot in step 4.
"""

import sqlite3
from datetime import datetime
from typing import Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from agent import bank, config
from agent.call_state import (
    DISPUTE_KINDS,
    OUTCOMES,
    CallState,
    as_rupees,
    check_callback,
    check_outcome,
    check_promise,
    parse_date,
)

NOT_VERIFIED = {
    "ok": False,
    "error": "not_verified",
    "message": "Identity is not verified. Ask for the date of birth first. Do not share or discuss loan details.",
}
LOCKED = {
    "ok": False,
    "error": "locked",
    "message": "Too many wrong dates of birth. Do not discuss the loan. Offer a callback, then end the call.",
}


def _refused(error: str, message: str) -> dict:
    return {"ok": False, "error": error, "message": message}


class CallTools:
    def __init__(self, state: CallState, conn: sqlite3.Connection, clock: Callable[[], datetime] = datetime.now):
        self.state = state
        self.conn = conn
        self.clock = clock   # tests pass a fixed time
        self.customer = bank.get_customer(conn, state.customer_id)
        self.loan = bank.get_loan(conn, state.customer_id)
        if self.customer is None or self.loan is None:
            raise ValueError(f"no customer or loan with id {state.customer_id}")

    def call(self, name: str, arguments: dict) -> dict:
        """Run a tool by name, as the LLM asked. Bad names or arguments become refusals, not crashes."""
        handler = getattr(self, name, None) if name in TOOL_NAMES else None
        if handler is None:
            result = _refused("unknown_tool", f"There is no tool called {name}.")
        else:
            try:
                result = handler(**(arguments or {}))
            except TypeError as e:
                result = _refused("bad_arguments", str(e))
        self.state.tool_calls.append({"name": name, "arguments": arguments, "result": result})
        return result

    def _gate(self) -> dict | None:
        if self.state.locked:
            return LOCKED
        if not self.state.verified:
            return NOT_VERIFIED
        return None

    def verify_identity(self, date_of_birth: str) -> dict:
        if self.state.verified:
            return {"ok": True, "verified": True}
        if self.state.locked:
            return LOCKED
        dob = parse_date(date_of_birth)
        if dob is None:
            return _refused("bad_format", "Send the date of birth as YYYY-MM-DD. This try was not counted.")
        if dob.isoformat() == self.customer["date_of_birth"]:
            self.state.verified = True
            return {"ok": True, "verified": True}
        self.state.verify_attempts += 1
        if self.state.locked:
            return LOCKED
        left = config.MAX_VERIFY_ATTEMPTS - self.state.verify_attempts
        return {"ok": False, "error": "no_match", "attempts_left": left,
                "message": "That date of birth does not match. Ask the caller to repeat it. Do not hint at the right one."}

    def get_loan_details(self) -> dict:
        if refusal := self._gate():
            return refusal
        today = self.clock().date()
        due = datetime.fromisoformat(self.loan["due_date"]).date()
        return {
            "ok": True,
            "customer_name": self.customer["name"],
            "product": self.loan["product"],
            "emi": self.loan["emi"],
            "amount_due": self.loan["amount_due"],
            "due_date": self.loan["due_date"],
            "days_overdue": (today - due).days,
            "last_payment": bank.last_payment(self.conn, self.loan["id"]),
        }

    def record_promise(self, amount, date: str) -> dict:
        if refusal := self._gate():
            return refusal
        if problem := check_promise(amount, date, self.clock().date(), self.loan["amount_due"]):
            return _refused("invalid_promise", problem)
        rupees, pay_by = as_rupees(amount), parse_date(date)
        bank.add_promise(self.conn, self.loan["id"], self.state.call_id, rupees, pay_by)
        self.state.promise = {"amount": rupees, "pay_by": pay_by.isoformat()}
        return {"ok": True, **self.state.promise}

    def schedule_callback(self, call_at: str) -> dict:
        # No identity needed: whoever answers may ask us to call back, and nothing about the loan is said.
        if problem := check_callback(call_at, self.clock()):
            return _refused("invalid_callback", problem)
        when = datetime.fromisoformat(call_at)
        bank.add_callback(self.conn, self.state.customer_id, self.state.call_id, when)
        self.state.callback = when.isoformat(timespec="minutes")
        return {"ok": True, "call_at": self.state.callback}

    def flag_dispute(self, kind: str, reason: str) -> dict:
        if refusal := self._gate():
            return refusal
        if kind not in DISPUTE_KINDS:
            return _refused("invalid_dispute", f"kind must be one of {', '.join(DISPUTE_KINDS)}")
        reason = (reason or "").strip()[:500]
        if not reason:
            return _refused("invalid_dispute", "give the caller's reason in a few words")
        bank.add_dispute(self.conn, self.loan["id"], self.state.call_id, kind, reason)
        self.state.dispute = {"kind": kind, "reason": reason}
        return {"ok": True, **self.state.dispute}

    def end_call(self, outcome: str) -> dict:
        if problem := check_outcome(self.state, outcome):
            return _refused("invalid_outcome", problem)
        self.state.outcome = outcome
        return {"ok": True, "outcome": outcome}


SCHEMAS = [
    FunctionSchema(
        name="verify_identity",
        description="Check the caller's date of birth. Call this before discussing the loan in any way.",
        properties={"date_of_birth": {"type": "string", "description": "Date of birth as YYYY-MM-DD"}},
        required=["date_of_birth"],
    ),
    FunctionSchema(
        name="get_loan_details",
        description="Get the verified customer's loan: amount due, due date, days overdue, last payment.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="record_promise",
        description="Record the customer's promise to pay an amount by a date (at most 7 days from today).",
        properties={
            "amount": {"type": "integer", "description": "Amount in whole rupees"},
            "date": {"type": "string", "description": "Payment date as YYYY-MM-DD"},
        },
        required=["amount", "date"],
    ),
    FunctionSchema(
        name="schedule_callback",
        description="Schedule a call back when the person asks to be called later.",
        properties={"call_at": {"type": "string", "description": "Local time as YYYY-MM-DDTHH:MM, 08:00 to 19:00"}},
        required=["call_at"],
    ),
    FunctionSchema(
        name="flag_dispute",
        description="Record that the customer disputes the debt, for a human to check.",
        properties={
            "kind": {"type": "string", "enum": list(DISPUTE_KINDS)},
            "reason": {"type": "string", "description": "The customer's reason, in a few words"},
        },
        required=["kind", "reason"],
    ),
    FunctionSchema(
        name="end_call",
        description="End the call with its outcome, after saying goodbye.",
        properties={"outcome": {"type": "string", "enum": list(OUTCOMES)}},
        required=["outcome"],
    ),
]
TOOL_NAMES = {s.name for s in SCHEMAS}
TOOLS = ToolsSchema(standard_tools=SCHEMAS)
