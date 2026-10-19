"""Per-call state and the call rules. Plain Python, no models: the rules hold even if the LLM misbehaves.

Each check returns None when the input is fine, or a short reason the LLM can act on.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from agent import config

OUTCOMES = ("promise_to_pay", "callback", "dispute", "wrong_person", "refused", "no_resolution")
DISPUTE_KINDS = ("already_paid", "wrong_amount", "other")

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ISO_MINUTE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


@dataclass
class CallState:
    call_id: str
    customer_id: int                 # who we are calling: set by code when the call starts, never by the LLM
    verified: bool = False
    verify_attempts: int = 0         # wrong answers only; a badly formatted date doesn't count
    promise: dict | None = None      # {"amount": 4500, "pay_by": "2026-10-23"}
    callback: str | None = None      # "2026-10-21T17:00"
    dispute: dict | None = None      # {"kind": "already_paid", "reason": "..."}
    outcome: str | None = None
    tool_calls: list[dict] = field(default_factory=list)   # every call, for the end-of-call record and scoring

    @property
    def locked(self) -> bool:
        return not self.verified and self.verify_attempts >= config.MAX_VERIFY_ATTEMPTS


def within_call_hours(now: datetime) -> bool:
    start, end = config.CALL_HOURS
    return start <= now.hour < end


def parse_date(text: str) -> date | None:
    """Strict YYYY-MM-DD. Python's fromisoformat alone also accepts forms like '20261020'."""
    if not isinstance(text, str) or not _ISO_DATE.fullmatch(text):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:  # 2026-02-30
        return None


def parse_minute(text: str) -> datetime | None:
    """Strict YYYY-MM-DDTHH:MM."""
    if not isinstance(text, str) or not _ISO_MINUTE.fullmatch(text):
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def as_rupees(amount) -> int | None:
    """Whole rupees. The LLM may send 4500, 4500.0 or "4500"; all mean the same."""
    if isinstance(amount, bool):
        return None
    if isinstance(amount, str):
        amount = amount.replace(",", "").strip()
        try:
            amount = float(amount)
        except ValueError:
            return None
    if isinstance(amount, (int, float)) and float(amount).is_integer():
        return int(amount)
    return None


def check_promise(amount, pay_by: str, today: date, amount_due: int) -> str | None:
    rupees = as_rupees(amount)
    if rupees is None:
        return "amount must be a whole number of rupees"
    if rupees < 1:
        return "amount must be at least 1 rupee"
    if rupees > amount_due:
        return f"amount is more than the {amount_due} rupees due"
    day = parse_date(pay_by)
    if day is None:
        return "date must be YYYY-MM-DD"
    if day < today:
        return "date is in the past"
    if day > today + timedelta(days=config.PROMISE_MAX_DAYS):
        return f"date must be within {config.PROMISE_MAX_DAYS} days; offer a callback instead"
    return None


def check_callback(call_at: str, now: datetime) -> str | None:
    when = parse_minute(call_at)
    if when is None:
        return "time must be YYYY-MM-DDTHH:MM"
    if when <= now:
        return "time is in the past"
    if when > now + timedelta(days=config.CALLBACK_MAX_DAYS):
        return f"time must be within {config.CALLBACK_MAX_DAYS} days"
    if not within_call_hours(when):
        start, end = config.CALL_HOURS
        return f"time must be between {start}:00 and {end}:00"
    return None


def check_outcome(state: CallState, outcome: str) -> str | None:
    """The outcome must match what the call actually recorded."""
    if outcome not in OUTCOMES:
        return f"outcome must be one of {', '.join(OUTCOMES)}"
    if outcome == "promise_to_pay" and state.promise is None:
        return "no promise was recorded; call record_promise first"
    if outcome == "callback" and state.callback is None:
        return "no callback was scheduled; call schedule_callback first"
    if outcome == "dispute" and state.dispute is None:
        return "no dispute was flagged; call flag_dispute first"
    if outcome == "wrong_person" and state.verified:
        return "identity was verified, so this is not the wrong person"
    if outcome == "refused" and not state.verified:
        return "only a verified customer can refuse"
    return None
