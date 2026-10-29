"""The seven tools the LLM can call during a collections call, and their descriptions for Pipecat.

Every tool acts on the customer this call was placed to (`state.customer_id`); none takes a customer
or loan id from the LLM. Loan details and anything recorded against the loan need a verified identity.
Results are dicts; a refusal is {"ok": False, "error": <code>, "message": <what to do instead>} and
never contains loan data. The bot (agent/bot.py) registers each one with the LLM.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from agent import bank, config
from agent.call_state import (
    DISPUTE_KINDS,
    OUTCOMES,
    CallState,
    as_rupees,
    caller_amounts,
    caller_day,
    check_callback,
    check_outcome,
    check_promise,
    resolve_birth_date,
    resolve_day,
    resolve_time,
)

NOT_VERIFIED = {
    "ok": False,
    "error": "not_verified",
    "message": "Identity is not verified. Ask for the date of birth first. Do not share or discuss loan details.",
}
NO_NOTICE = {
    "ok": False,
    "error": "notice_not_given",
    "message": "The caller has not heard the recording notice. First say: Please note, this call is recorded. "
               "Then ask for the date of birth again.",
}
LOCKED = {
    "ok": False,
    "error": "locked",
    "message": "Too many wrong dates of birth. Do not say why you are calling. The system ends the call.",
}
CALL_ENDED = {
    "ok": False,
    "error": "call_ended",
    "message": "The call is ending. Say nothing more.",
}


def _refused(error: str, message: str) -> dict:
    return {"ok": False, "error": error, "message": message}


def _reason(text) -> str:
    """The LLM's free text, trimmed. Not a string (Qwen may send a list or null) counts as none given."""
    return text.strip()[:500] if isinstance(text, str) else ""


class CallTools:
    def __init__(self, state: CallState, conn: sqlite3.Connection, clock: Callable[[], datetime] = datetime.now):
        self.state = state
        self.conn = conn
        self.clock = clock   # tests pass a fixed time
        self.customer = bank.get_customer(conn, state.customer_id)
        self.loan = bank.get_loan(conn, state.customer_id)
        if self.customer is None or self.loan is None:
            raise ValueError(f"no customer or loan with id {state.customer_id}")

    @classmethod
    def open(cls, state: CallState, db_path: Path) -> "CallTools":
        """Connects to the bank for one call. If the customer isn't there, the connection is closed again:
        left open, it locks bank.db on Windows and the test bench can't reseed it."""
        conn = bank.connect(db_path)
        try:
            return cls(state, conn)
        except Exception:
            conn.close()
            raise

    def call(self, name: str, arguments: dict) -> dict:
        """Run a tool by name, as the LLM asked. Bad names or arguments become refusals, not crashes."""
        handler = getattr(self, name, None) if name in TOOL_NAMES else None
        if handler is None:
            result = _refused("unknown_tool", f"There is no tool called {name}.")
        elif self.state.outcome is not None:
            # end_call ran (or code closed the call): a second end_call or a late promise would change the
            # outcome or add a bank row after the goodbye.
            result = CALL_ENDED
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
        if not self.state.notice_given:
            return NO_NOTICE
        dob = resolve_birth_date(date_of_birth)
        if dob is None:
            return _refused("bad_format", "Ask the caller for their full date of birth: day, month and year. "
                                          "This try was not counted.")
        if dob.isoformat() == self.customer["date_of_birth"]:
            self.state.verified = True
            self.state.verified_at_turn = len(self.state.caller_turns)
            # The loan comes with the success, so the LLM can't skip get_loan_details and guess the facts.
            return {"ok": True, "verified": True, "loan": self._loan_facts()}
        self.state.verify_attempts += 1
        if self.state.locked:
            return LOCKED
        left = config.MAX_VERIFY_ATTEMPTS - self.state.verify_attempts
        return {"ok": False, "error": "no_match", "attempts_left": left,
                "message": "That date of birth does not match. Ask the caller to repeat it. Do not hint at the right one."}

    def get_loan_details(self) -> dict:
        if refusal := self._gate():
            return refusal
        return {"ok": True, **self._loan_facts()}

    def _loan_facts(self) -> dict:
        today = self.clock().date()
        due = datetime.fromisoformat(self.loan["due_date"]).date()
        return {
            "customer_name": self.customer["name"],
            "product": self.loan["product"],
            "emi": self.loan["emi"],
            "amount_due": self.loan["amount_due"],
            "due_date": self.loan["due_date"],
            "days_overdue": (today - due).days,
            "last_payment": bank.last_payment(self.conn, self.loan["id"]),
        }

    def _day(self, day, today):
        """The LLM's day if it is in the caller's words, else the day the caller named (the slip is noted).
        Not from turns up to the date of birth: "the 5th of March" there is a birthday."""
        if (on := resolve_day(day, today)) is not None:
            return on
        if (on := caller_day(self.state.caller_turns[self.state.verified_at_turn:], today)) is not None:
            self.state.llm_slips.append(f"day not in the caller's words: {day!r}, used the caller's {on.isoformat()}")
        return on

    def _amount_not_said(self, rupees: int) -> str | None:
        """Refuse an amount the caller didn't say. Qwen 7B once asked a caller who offered 500 for "4500 on the
        27th" (the prompt's example amount), heard "No", read it back anyway, got "Yes, please" and recorded
        it. Unlike the day, the caller's amount isn't used instead: below the minimum it needs a human."""
        due = self.loan["amount_due"]
        said = caller_amounts(self.state.caller_turns[self.state.verified_at_turn:], due)
        if rupees in (said or {due}):
            return None
        self.state.llm_slips.append(f"amount not in the caller's words: {rupees}, caller said {sorted(said or [])}")
        if said is None:
            return "Nothing recorded; don't read this out. The caller hasn't said an amount: ask how much they can pay"
        return (f"Nothing recorded; don't read this out. The caller said {' or '.join(map(str, sorted(said)))} "
                f"rupees, not {rupees}. Never suggest an amount; use theirs")

    def record_promise(self, amount, day: str) -> dict:
        if refusal := self._gate():
            return refusal
        today = self.clock().date()
        pay_by = self._day(day, today)
        if pay_by is None:
            return _refused("invalid_promise", "Nothing recorded; don't read this out. Call record_promise again "
                                               "with day in the caller's words, like Friday, tomorrow or the 23rd")
        if problem := check_promise(amount, pay_by.isoformat(), today, self.loan["amount_due"]):
            return _refused("invalid_promise", problem)
        rupees = as_rupees(amount)
        if problem := self._amount_not_said(rupees):
            return _refused("invalid_promise", problem)
        if self.state.promise:   # the caller changed their mind ("make it Saturday"): one promise per call
            bank.update_promise(self.conn, self.state.call_id, rupees, pay_by)
        else:
            bank.add_promise(self.conn, self.loan["id"], self.state.call_id, rupees, pay_by)
        self.state.promise = {"amount": rupees, "pay_by": pay_by.isoformat()}
        return {"ok": True, **self.state.promise}

    def schedule_callback(self, day: str, time: str) -> dict:
        # No identity needed: whoever answers may ask us to call back, and nothing about the loan is said.
        if self.state.callback:   # once per call: a second call returns the first, no second bank row
            return {"ok": True, "call_at": self.state.callback}
        now = self.clock()
        on, at = self._day(day, now.date()), resolve_time(time)
        if on is None or at is None:
            return _refused("invalid_callback", "Nothing scheduled; don't read this out. Call schedule_callback again "
                                                "with day in the caller's words, like tomorrow, and time like 17:00")
        call_at = f"{on.isoformat()}T{at[0]:02d}:{at[1]:02d}"
        if problem := check_callback(call_at, now):
            return _refused("invalid_callback", problem)
        when = datetime.fromisoformat(call_at)
        bank.add_callback(self.conn, self.state.customer_id, self.state.call_id, when)
        self.state.callback = when.isoformat(timespec="minutes")
        return {"ok": True, "call_at": self.state.callback}

    def flag_dispute(self, kind: str, reason: str) -> dict:
        if refusal := self._gate():
            return refusal
        if self.state.dispute:   # in a bench call Qwen flagged the same dispute twice: two bank rows
            return {"ok": True, **self.state.dispute}
        if kind not in DISPUTE_KINDS:
            return _refused("invalid_dispute", f"kind must be one of {', '.join(DISPUTE_KINDS)}")
        reason = _reason(reason)
        if not reason:
            return _refused("invalid_dispute", "give the caller's reason in a few words")
        bank.add_dispute(self.conn, self.loan["id"], self.state.call_id, kind, reason)
        self.state.dispute = {"kind": kind, "reason": reason}
        return {"ok": True, **self.state.dispute}

    def escalate_to_human(self, reason: str) -> dict:
        if refusal := self._gate():
            return refusal
        if self.state.escalation:
            return {"ok": True, "reason": self.state.escalation}
        reason = _reason(reason)
        if not reason:
            return _refused("invalid_escalation", "give the reason in a few words")
        bank.add_escalation(self.conn, self.loan["id"], self.state.call_id, reason)
        self.state.escalation = reason
        return {"ok": True, "reason": reason}

    def update_escalation(self, reason: str) -> None:
        """Code only, not a tool for the LLM: the caller went on after the hardship code escalated on, so the
        note for the human gets all of it ("I lost my job. I can pay 500 this week.")."""
        reason = _reason(reason)
        if self.state.escalation and reason:
            bank.update_escalation(self.conn, self.state.call_id, reason)
            self.state.escalation = reason

    def end_call(self, outcome: str) -> dict:
        if problem := check_outcome(self.state, outcome):
            return _refused("invalid_outcome", problem)
        self.state.outcome = outcome
        return {"ok": True, "outcome": outcome}


# No YYYY-MM-DD: offered one, Qwen 7B passed today's date for "this Friday" in 4 of 20 runs (resolve_day).
SPOKEN_DAY = "The day as the caller said it: today, tomorrow, a weekday like Friday, or a day of the month like the 23rd"

SCHEMAS = [
    FunctionSchema(
        name="verify_identity",
        description="Check the caller's date of birth. Call this before discussing the loan in any way. "
                    "On success it also returns the loan details.",
        properties={"date_of_birth": {"type": "string", "description": "The date of birth as the caller said it"}},
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
            "day": {"type": "string", "description": SPOKEN_DAY},
        },
        required=["amount", "day"],
    ),
    FunctionSchema(
        name="schedule_callback",
        description="Schedule a call back when the person asks to be called later.",
        properties={
            "day": {"type": "string", "description": SPOKEN_DAY},
            "time": {"type": "string", "description": "Time like 17:00 or 5 pm, from 08:00 and before 19:00"},
        },
        required=["day", "time"],
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
        name="escalate_to_human",
        description="Pass the case to a human agent: hardship (lost job, illness), a request for a human, "
                    "or a promise below the minimum.",
        properties={"reason": {"type": "string", "description": "What the customer said, in a few words"}},
        required=["reason"],
    ),
    FunctionSchema(
        name="end_call",
        description="End the call with its outcome. Do not say goodbye yourself; the system says it.",
        properties={"outcome": {"type": "string", "enum": list(OUTCOMES)}},
        required=["outcome"],
    ),
]
TOOL_NAMES = {s.name for s in SCHEMAS}
TOOLS = ToolsSchema(standard_tools=SCHEMAS)

# What the LLM sees until the date of birth matches: three tools whose descriptions say nothing about a
# debt (the full list would give it away: record_promise, flag_dispute...). The bot swaps in TOOLS after.
IDENTITY_SCHEMAS = [
    FunctionSchema(
        name="verify_identity",
        description="Check the caller's date of birth to confirm who they are.",
        properties=SCHEMAS[0].properties,
        required=SCHEMAS[0].required,
    ),
    next(s for s in SCHEMAS if s.name == "schedule_callback"),
    FunctionSchema(
        name="end_call",
        description="End the call with its outcome. Do not say goodbye yourself; the system says it.",
        properties={"outcome": {"type": "string", "enum": ["callback", "wrong_person", "no_resolution"]}},
        required=["outcome"],
    ),
]
IDENTITY_TOOLS = ToolsSchema(standard_tools=IDENTITY_SCHEMAS)
