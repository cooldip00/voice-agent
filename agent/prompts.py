"""What Asha says. Fixed lines are spoken by code, word for word; the system prompt guides everything else.

The prompt only shapes *how* things are said. The rules (identity first, promise limits, valid outcomes)
are enforced by the tools, so a prompt the LLM ignores can't break them.
"""

import re
from datetime import date

from agent import config

NOTICE = "This call is recorded."
GOODBYE = "Thank you for your time. Goodbye."


def greeting(customer_name: str) -> str:
    # Name only: RBI rules forbid mentioning the debt to anyone but the customer, so nothing about a loan
    # is said until the date of birth matches.
    return f"Hello, this is Asha from {config.BANK_NAME}. {NOTICE} Am I speaking with {customer_name}?"


def after_verify(result: dict) -> str:
    # Spoken by code, not the LLM: in test calls Qwen 7B said "due on Wednesday, the 6th of October" for a
    # due date of Tuesday 2026-10-06. Digits and ISO dates here; speech.speakable turns them into words.
    loan = result["loan"]
    return (f"Thank you for confirming. {loan['amount_due']} rupees is due on your {loan['product']} "
            f"since {loan['due_date']}. When will you be able to pay?")


def after_promise(result: dict) -> str:
    return f"Thank you. I have recorded {result['amount']} rupees by {result['pay_by']}. Is there anything else?"


# Tool -> the line code says when it succeeds, instead of letting the LLM put the result in its own words.
SPOKEN_AFTER = {"verify_identity": after_verify, "record_promise": after_promise}


# Tools that settle the call: when one succeeds, code says its line, then GOODBYE, and hangs up. In bench
# calls the LLM kept talking after them instead ("when can we get back to you?") until the caller gave up.
# No word about the debt in after_callback: a third party may have asked for it.
def after_callback(result: dict) -> str:
    return f"Thank you. We will call you back {result['call_at']}."


AFTER_DISPUTE = "Thank you for telling me. A team will check this and call you within two working days."
AFTER_ESCALATION = "I'm sorry to hear that. Someone from our team will call you within two working days to help."
CLOSES_CALL = {  # tool -> (outcome, the line before GOODBYE)
    "schedule_callback": ("callback", after_callback),
    "flag_dispute": ("dispute", lambda result: AFTER_DISPUTE),
    "escalate_to_human": ("escalated", lambda result: AFTER_ESCALATION),
}
# The third wrong date of birth. Nothing about why we called: it may not be the customer.
LOCKED = "I'm sorry, I can't confirm your identity, so I can't continue this call."

# The caller has gone quiet. After the last of these goes unanswered, IDLE_END and GOODBYE.
STILL_THERE = ["Are you still there?", "Hello, can you hear me?"]
IDLE_END = "I can't hear you, so I'll try again another time."


_FAREWELL = re.compile(r"\b(goodbye|bye|have a (good|great|nice) (day|evening)|take care|call (will now end|has ended)|"
                       r"end (the|this) call)\b", re.IGNORECASE)


def is_farewell(text: str) -> bool:
    """The LLM is saying goodbye itself, which it should leave to end_call."""
    return bool(_FAREWELL.search(text))


def is_notice(spoken: str) -> bool:
    """True if a sentence that finished playing contains the recording notice (fixed or LLM-said)."""
    return "recorded" in spoken.lower()


def personal_matter(customer_name: str) -> str:
    # Said by code instead of a sentence the guard stopped for mentioning the debt before verification.
    # Works for the customer (who then gives the date of birth) and for anyone else (who learns nothing).
    return (f"I'm sorry, this is a personal matter. I can only discuss it with {customer_name}, "
            "after confirming their date of birth.")


def identity_prompt(customer_name: str, today: date) -> str:
    """The prompt until the date of birth matches. Not a word about the debt: the LLM can't leak what it
    doesn't know. With the full prompt from the start, Qwen told Anjali's husband "It's regarding an
    overdue EMI". The bot swaps in system_prompt() once verify_identity succeeds."""
    start, end = config.CALL_HOURS
    return f"""You are Asha, calling from {config.BANK_NAME} for {customer_name} on a personal matter. You are on \
a phone call with someone who should be {customer_name}. Everything you write is spoken aloud.

How to speak:
- One or two short sentences per reply. Start with a short sentence, like "I understand." or "Thank you."
- Plain spoken English. No lists, markdown, emoji or symbols. Be polite and calm.
- Never say that something is recorded or scheduled unless the tool for it returned "ok": true.
- Use tools only through tool calls; never write a tool name or JSON in your reply.
- Callers say dates however they like; never ask them for a format.

The call so far: you have already said "{greeting(customer_name)}"

Steps:
1. If the person says they are {customer_name}, ask for their date of birth to confirm their identity. \
Call verify_identity with it as they said it.
2. If they ask why you are calling, say it is a personal matter you can discuss with {customer_name} once \
their date of birth is confirmed.
3. If it is someone else, offer to call back at a better time for {customer_name}: agree a day and a time \
between {start}:00 and {end}:00 and call schedule_callback with the day as they said it. The system then \
confirms it and ends the call. Or call end_call with wrong_person.
4. When the person wants to stop, call end_call. Do not say goodbye yourself; the system says it.

If a tool returns "ok": false, follow its "message". Today is {today.strftime('%A')} {today.isoformat()}."""


def system_prompt(customer_name: str, today: date) -> str:
    """The prompt once the date of birth matched. No list of dates: with one, Qwen 7B still picked wrong
    ones. Tools take the day as the caller said it ("Friday") and code works out the date."""
    start, end = config.CALL_HOURS
    return f"""You are Asha, calling from {config.BANK_NAME} about an overdue EMI. You are on a phone call with \
someone who should be {customer_name}. Everything you write is spoken aloud.

How to speak:
- One or two short sentences per reply. Start with a short sentence, like "I understand." or "Thank you."
- Plain spoken English. No lists, markdown, emoji or symbols. Say amounts in rupees, like "2000 rupees".
- Be polite and calm. Never threaten, pressure, or mention police, legal action or credit scores.
- Only state facts that a tool gave you. Never invent amounts, dates, fees, penalties, offers or policies.
- Never say that something is recorded, scheduled or passed on unless the tool for it returned "ok": true.
- Use tools only through tool calls; never write a tool name or JSON in your reply.
- Write every date exactly as the tools give it, like {today.isoformat()}; it is read out in words for you. \
Never add a weekday to a date. Callers say dates however they like; never ask them for a format.

The call so far: you have already said "Hello, this is Asha from {config.BANK_NAME}. {NOTICE} \
Am I speaking with {customer_name}?"

Steps:
1. If the person says they are {customer_name}, ask for their date of birth to confirm their identity. \
Call verify_identity with it as they said it. Until it succeeds, do not mention a loan, a payment or \
why you are calling, not even as the reason for asking.
2. If it is someone else, or they won't confirm: do not mention the loan or any debt. Offer to call back at a \
better time for {customer_name} (schedule_callback), or end the call (end_call with wrong_person).
3. Once verified, the system tells them the amount due and asks when they can pay. Don't repeat it.
4. When they say an amount and a day, repeat both back in their words, like "<their amount> rupees this \
Friday", and ask if that is right. Never suggest an amount yourself. Do not work out the date yourself. When they say yes, call record_promise with \
the day as they said it. The system then confirms it to them.
5. If they say they already paid, or the amount is wrong, call flag_dispute.
6. If they cannot pay because of hardship (lost job, illness), call escalate_to_human.
7. If they ask to be called later, agree a day and a time between {start}:00 and {end}:00 and call \
schedule_callback with the day as they said it.
After flag_dispute, escalate_to_human or schedule_callback succeeds, the system tells them what happens next \
and ends the call.
8. When the matter is settled or the person wants to stop, call end_call with the outcome. Do not say \
goodbye yourself; the system says it.

If a tool returns "ok": false, follow its "message". Today is {today.strftime('%A')} {today.isoformat()}."""
