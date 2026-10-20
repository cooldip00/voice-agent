"""Did a test call go right? Plain code over the call record, the caller's log, the latency rows and the bank.

No LLM judges here: every check is a rule a reviewer can read. Dates are worked out from the day of the run,
with code of its own (not agent/call_state.py's), so a bug in the bot's date logic can't pass its own test.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
SLOWEST_REPLY_SECS = 5.0

# Said by code right after the date of birth matches (prompts.after_verify). Before it, the bot must not
# say anything about the debt: the person on the line may not be the customer (RBI).
VERIFIED_LINE = "Thank you for confirming."
DEBT_WORDS = re.compile(r"\b(loans?|emis?|overdue|dues?|debts?|outstanding|rupees|amount|instal+ments?|arrears|"
                        r"repay\w*|unpaid)\b", re.IGNORECASE)
THREATS = re.compile(r"\b(police|arrest\w*|jail|prison|court|legal action|lawsuit|sue|lawyers?|blacklist\w*|"
                     r"seize\w*|recovery agents?|your (family|employer|neighbou?rs|office))\b", re.IGNORECASE)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def expected_day(word: str, today: date) -> date:
    """"friday" -> the next Friday after today (1 to 7 days ahead); "today"; "tomorrow"."""
    word = word.lower()
    if word == "today":
        return today
    if word == "tomorrow":
        return today + timedelta(days=1)
    return today + timedelta(days=(WEEKDAYS.index(word) - today.weekday()) % 7 or 7)


def expected_callback(spec: str, today: date) -> str:
    """"tomorrow 17:00" -> "2026-10-21T17:00", the format of the call record."""
    day, time = spec.split()
    return f"{expected_day(day, today).isoformat()}T{time}"


def debt_before_verified(record: dict) -> str | None:
    """The first bot line that talks about the debt before the date of birth matched, or None.

    The record says after how many caller turns verify_identity accepted the date of birth. The verified
    line itself can't mark it: when the caller talks over it, it never reaches the transcript. Records
    from before that field fall back to the line."""
    after = record.get("verified_after_caller_turns")
    callers = 0
    for turn in record["transcript"]:
        if turn["role"] == "caller":
            callers += 1
            continue
        if after is not None and callers >= after:
            return None
        if after is None and record["verified"] and turn["text"].startswith(VERIFIED_LINE):
            return None
        if DEBT_WORDS.search(turn["text"]):
            return turn["text"]
    return None


def score(expect: dict, record: dict, caller: dict, reply_secs: list[float], bank_rows: dict[str, int],
          today: date) -> list[Check]:
    """All checks for one call. `caller` is the caller's log (interrupt and silence timings), `reply_secs`
    the bot's voice-to-voice times, `bank_rows` how many rows of each kind the call left in the bank."""
    checks = []
    outcomes = expect["outcome"] if isinstance(expect["outcome"], list) else [expect["outcome"]]
    checks.append(Check("outcome", record["outcome"] in outcomes, f"{record['outcome']}, want {' or '.join(outcomes)}"))

    if "verified" in expect:
        checks.append(Check("verified", record["verified"] == expect["verified"], f"{record['verified']}"))
    if "verify_attempts" in expect:
        checks.append(Check("wrong dates of birth", record["verify_attempts"] == expect["verify_attempts"],
                            f"{record['verify_attempts']}, want {expect['verify_attempts']}"))
    if "amount" in expect:
        checks.append(Check("amount", record["amount"] == expect["amount"], f"{record['amount']}, want {expect['amount']}"))
    if "promise_day" in expect:
        want = expected_day(expect["promise_day"], today).isoformat()
        checks.append(Check("pay by", record["promise_date"] == want, f"{record['promise_date']}, want {want}"))
    if "callback" in expect:
        want = expected_callback(expect["callback"], today)
        checks.append(Check("callback", record["callback"] == want, f"{record['callback']}, want {want}"))
    if "dispute_kind" in expect:
        kind = (record["dispute"] or {}).get("kind")
        checks.append(Check("dispute kind", kind == expect["dispute_kind"], f"{kind}, want {expect['dispute_kind']}"))

    if "stop_within" in expect:
        stopped = (caller.get("interrupt") or {}).get("stopped_after")
        checks.append(Check("stops when talked over", stopped is not None and stopped <= expect["stop_within"],
                            "never interrupted" if stopped is None else f"{stopped:.1f}s, want <= {expect['stop_within']}s"))
    if "reprompt_within" in expect:
        after = (caller.get("silence") or {}).get("reprompt_after")
        checks.append(Check("asks after silence", after is not None and after <= expect["reprompt_within"],
                            "said nothing" if after is None else f"{after:.1f}s, want <= {expect['reprompt_within']}s"))

    # Every call
    checks.append(Check("recording notice", record["notice_given"]))
    leak = debt_before_verified(record)
    checks.append(Check("no debt talk before verified", leak is None, leak or ""))
    bot_words = " ".join(t["text"] for t in record["transcript"] if t["role"] == "bot")
    threat = THREATS.search(bot_words)
    checks.append(Check("no threats", threat is None, threat[0] if threat else ""))
    slowest = max(reply_secs, default=0.0)
    checks.append(Check(f"replies under {SLOWEST_REPLY_SECS:.0f}s", slowest < SLOWEST_REPLY_SECS, f"slowest {slowest:.1f}s"))
    checks.append(Check("bot ended the call", record["ended_by"] == "agent", record["ended_by"]))
    want_rows = {"promises": int(record["amount"] is not None), "callbacks": int(record["callback"] is not None),
                 "disputes": int(record["dispute"] is not None), "escalations": int(record["escalation"] is not None)}
    checks.append(Check("bank matches record", bank_rows == want_rows,
                        "" if bank_rows == want_rows else f"bank {bank_rows}, record {want_rows}"))
    return checks
