"""Per-call state and the call rules. Plain Python, no models: the rules hold even if the LLM misbehaves.

Each check returns None when the input is fine, or a short reason the LLM can act on.
"""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from agent import config

OUTCOMES = ("promise_to_pay", "callback", "dispute", "escalated", "wrong_person", "refused", "no_resolution")
DISPUTE_KINDS = ("already_paid", "wrong_amount", "other")

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ISO_MINUTE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


@dataclass
class CallState:
    call_id: str
    customer_id: int                 # who we are calling: set by code when the call starts, never by the LLM
    notice_given: bool = False       # the recording notice finished playing; set by the bot, from the audio
    verified: bool = False
    verify_attempts: int = 0         # wrong answers only; a badly formatted date doesn't count
    promise: dict | None = None      # {"amount": 4500, "pay_by": "2026-10-23"}
    callback: str | None = None      # "2026-10-21T17:00"
    dispute: dict | None = None      # {"kind": "already_paid", "reason": "..."}
    escalation: str | None = None    # reason handed to a human, e.g. "lost his job, can pay 500 this week"
    outcome: str | None = None
    tool_calls: list[dict] = field(default_factory=list)   # every call, for the end-of-call record and scoring
    llm_slips: list[str] = field(default_factory=list)     # e.g. a tool call written as text; counted in step 5
    silenced: list[str] = field(default_factory=list)      # sentences the LLM wrote that the guards didn't let out

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
    minimum = math.ceil(amount_due * config.MIN_PROMISE_FRACTION)
    if rupees < minimum:
        return (f"amount is below the minimum of {minimum} rupees; do not record it. "
                "Ask if they can pay the minimum, or offer to pass the case to a human agent")
    if rupees > amount_due:
        return f"amount is more than the {amount_due} rupees due"
    day = parse_date(pay_by)
    if day is None:
        return "date must be YYYY-MM-DD"
    if day < today:
        return f"{day.isoformat()} is in the past; if the caller named a weekday, pass the weekday itself, like Friday"
    if day > today + timedelta(days=config.PROMISE_MAX_DAYS):
        return f"date must be within {config.PROMISE_MAX_DAYS} days; offer a callback instead"
    return None


WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def resolve_day(day, today: date) -> date | None:
    """The day as the caller said it ("Friday", "tomorrow") -> a date. Code does this, not the LLM: Qwen 7B
    turned "this Friday" into 2026-10-27, a Tuesday, even with the dates listed in its prompt.
    A weekday name means the next one after today (1 to 7 days ahead); a YYYY-MM-DD date alone is taken as is."""
    text = str(day).strip().lower()
    words = re.findall(r"[a-z]+", text)
    # The caller's word beats a date next to it: "Friday 2026-10-27" means Friday.
    if "today" in words:
        return today
    if "tomorrow" in words:
        return today + timedelta(days=1)
    for word in words:
        if word.capitalize() in WEEKDAYS:
            return today + timedelta(days=(WEEKDAYS.index(word.capitalize()) - today.weekday()) % 7 or 7)
    return parse_date(text)


MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december")
_DAY_MONTH_YEAR = re.compile(r"(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+),?\s+(\d{4})")
_MONTH_DAY_YEAR = re.compile(r"([a-z]+)\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})")
_NUMERIC = re.compile(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})")


def resolve_birth_date(text) -> date | None:
    """A date of birth as the caller said it -> a date: "1st of January, 1990", "January 1 1990",
    "01/01/1990" (day first, as in India) or YYYY-MM-DD. Asking Qwen 7B for YYYY-MM-DD made it ask the
    caller for that format instead of calling the tool."""
    text = str(text).strip().lower()
    if (iso := parse_date(text)) is not None:
        return iso
    if m := _NUMERIC.fullmatch(text):
        day, month, year = m[1], MONTHS[int(m[2]) - 1] if 1 <= int(m[2]) <= 12 else "", m[3]
    else:
        text = " ".join(text.replace(".", " ").split())   # Whisper writes "5th of March. 1990."
        if m := _DAY_MONTH_YEAR.fullmatch(text):
            day, month, year = m[1], m[2], m[3]
        elif m := _MONTH_DAY_YEAR.fullmatch(text):
            month, day, year = m[1], m[2], m[3]
        else:
            return None
    if month not in MONTHS:
        return None
    try:
        return date(int(year), MONTHS.index(month) + 1, int(day))
    except ValueError:  # 31st of February
        return None


def resolve_time(text) -> tuple[int, int] | None:
    """"17:00", "5 pm", "5:30 PM" -> (hour, minute). A bare 1 to 7 means PM: calls end at 19:00."""
    m = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*(?:([ap])\.?\s*m\.?)?\s*", str(text), re.IGNORECASE)
    if m is None:
        return None
    hour, minute, ampm = int(m[1]), int(m[2] or 0), (m[3] or "").lower()
    if ampm == "p" and hour < 12:
        hour += 12
    elif ampm == "a" and hour == 12:
        hour = 0
    elif not ampm and 1 <= hour <= 7:
        hour += 12
    if hour > 23 or minute > 59:
        return None
    return hour, minute


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
    if outcome == "escalated" and state.escalation is None:
        return "nothing was escalated; call escalate_to_human first"
    if outcome == "wrong_person" and state.verified:
        return "identity was verified, so this is not the wrong person"
    if outcome == "refused" and not state.verified:
        return "only a verified customer can refuse"
    return None


_DEBT_WORDS = re.compile(r"\b(loans?|emis?|overdue|payments?|pay|paid|dues?|debts?|outstanding|rupees|amount|"
                         r"instal+ments?|arrears)\b", re.IGNORECASE)
_FORMAT_TALK = re.compile(r"\b(yyyy|format)\b", re.IGNORECASE)
_CLAIM = re.compile(r"\b(recorded|noted|registered|logged|scheduled|escalated|flagged)\b", re.IGNORECASE)
_NOTICE = re.compile(r"\bcall (is|is being|will be) recorded\b", re.IGNORECASE)


def unsafe_to_say(text: str, state: CallState) -> str | None:
    """Why a sentence the LLM wrote must not be spoken, or None if it may be.

    The last line of defence for rules the prompt asks for but can't guarantee. It works on words, so it
    is coarse on purpose: blocking a harmless sentence costs a moment of silence, saying a harmful one
    can't be taken back.
    """
    if not state.verified and _DEBT_WORDS.search(text):
        return "mentions the debt before identity is verified"
    if _FORMAT_TALK.search(text):
        return "asks the caller for a date format"
    # Dates come from tools; in a test call Qwen offered a hardship caller "next Friday, 2026-10-27" (a Tuesday).
    given = set(_ISO_DATE.findall(json.dumps([c["result"] for c in state.tool_calls])))
    if any(d not in given for d in _ISO_DATE.findall(text)):
        return "states a date no tool gave"
    nothing_recorded = not (state.promise or state.callback or state.dispute or state.escalation)
    if nothing_recorded and _CLAIM.search(_NOTICE.sub("", text)):
        return "claims something was recorded before any tool recorded it"
    return None


def transcript(messages: list, silenced: list[str] = ()) -> list[dict]:
    """What was said on the call, from the LLM context: caller and bot text, no tool plumbing. The bot keeps
    sentences the guards silenced out of the context; any that got in anyway are taken out here, so the
    transcript is what the caller heard."""
    turns = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            text = m.get("content")
            if not isinstance(text, str):
                continue
            if m["role"] == "assistant":
                for sentence in silenced:
                    text = text.replace(sentence.strip(), "")
            text = " ".join(text.split())
            if text:
                turns.append({"role": "caller" if m["role"] == "user" else "bot", "text": text})
    return turns


def join_caller_turns(messages: list) -> list:
    """Back-to-back caller messages -> one. A pause mid-answer ("14th of July." ... "1985") ends a turn,
    and with the answer split in two Qwen 7B called verify_identity 1 time in 10; joined, 10 in 10."""
    joined = []
    for m in messages:
        prev = joined[-1] if joined else None
        if (isinstance(m, dict) and isinstance(prev, dict) and m.get("role") == prev.get("role") == "user"
                and isinstance(m.get("content"), str) and isinstance(prev.get("content"), str)):
            joined[-1] = {**prev, "content": f"{prev['content'].strip()} {m['content'].strip()}"}
        else:
            joined.append(m)
    return joined


def final_outcome(state: CallState) -> str:
    """end_call's outcome; if the call ended without one, what the tools recorded says what happened."""
    if state.outcome:
        return state.outcome
    if state.promise:
        return "promise_to_pay"
    if state.dispute:
        return "dispute"
    if state.escalation:
        return "escalated"
    if state.callback:
        return "callback"
    return "no_resolution"


def call_record(state: CallState, messages: list, started: datetime, ended: datetime) -> dict:
    """The end-of-call record. Built from what the tools accepted, never from what the LLM said."""
    promise = state.promise or {}
    return {
        "call_id": state.call_id,
        "customer_id": state.customer_id,
        "outcome": final_outcome(state),
        "ended_by": "agent" if state.outcome else "caller",   # no end_call = the caller hung up first
        "promise_date": promise.get("pay_by"),
        "amount": promise.get("amount"),
        "reason": (state.dispute or {}).get("reason") or state.escalation,
        "callback": state.callback,
        "dispute": state.dispute,
        "escalation": state.escalation,
        "notice_given": state.notice_given,
        "verified": state.verified,
        "verify_attempts": state.verify_attempts,
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
        "seconds": round((ended - started).total_seconds(), 1),
        "tool_calls": state.tool_calls,
        "llm_slips": state.llm_slips,
        "transcript": transcript(messages, state.silenced),
    }
