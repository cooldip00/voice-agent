"""Text -> words Kokoro reads correctly. Applied only to what TTS speaks; the context keeps the original text.

Kokoro reads "2026-10-23" digit by digit and doesn't know Indian grouping ("1,25,000"). The LLM and the
tools work in digits and ISO dates, so we turn those into words here, just before speaking:

    "4,500 rupees"        -> "four thousand five hundred rupees"
    "Rs. 1,25,000"        -> "one lakh twenty-five thousand rupees"
    "2026-10-23"          -> "Friday, the twenty-third of October"   (weekday if within a week of today)
    "1990-03-05"          -> "the fifth of March, nineteen ninety"   (year only if not this year)
    "2026-10-21T17:00"    -> "Wednesday, the twenty-first of October, at five PM"
    "Wednesday, the 6th of October" -> "the sixth of October"         (the LLM's weekday is dropped)
"""

import re
from datetime import date, datetime

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
           "nine": "ninth", "twelve": "twelfth"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]

# A weekday the LLM put before a date is dropped: it is often wrong ("Wednesday, the 6th of October" when
# the 6th is a Tuesday), and date_words adds the right one from the date itself.
_WEEKDAY = r"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
_MONTH = "|".join(MONTHS)
_DATETIME = re.compile(rf"\b{_WEEKDAY}(\d{{4}}-\d{{2}}-\d{{2}})[T ](\d{{2}}):(\d{{2}})\b")
_DATE = re.compile(rf"\b{_WEEKDAY}(\d{{4}}-\d{{2}}-\d{{2}})\b")
_DAY_OF_MONTH = re.compile(rf"\b{_WEEKDAY}(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+({_MONTH})(?:,?\s+(\d{{4}}))?\b")
_MONTH_DAY = re.compile(rf"\b{_WEEKDAY}({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}})\b)?")
_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(\s*[AaPp]\.?[Mm]\.?)?")
_NUM = r"\d{1,3}(?:,\d{2,3})+|\d+"   # 4500, 4,500 or 1,25,000
_RUPEES_BEFORE = re.compile(rf"(?:₹|\bRs\.?|\bINR)\s?({_NUM})(?:\.00)?(?:\s*rupees)?", re.IGNORECASE)
_RUPEES_AFTER = re.compile(rf"\b({_NUM})\s*(rupees|rupee)\b", re.IGNORECASE)
_GROUPED = re.compile(r"\b\d{1,3}(?:,\d{2,3})+\b")


def number_words(n: int) -> str:
    """Indian system: 1,25,000 -> one lakh twenty-five thousand."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + number_words(n % 100) if n % 100 else "")
    for size, name in ((10_000_000, "crore"), (100_000, "lakh"), (1000, "thousand")):
        if n >= size:
            rest = n % size
            return number_words(n // size) + " " + name + (" " + number_words(rest) if rest else "")
    raise AssertionError("unreachable")


def ordinal_words(n: int) -> str:
    words = number_words(n)
    head, sep, last = words.rpartition("-")
    if last in ORDINAL:
        last = ORDINAL[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def year_words(y: int) -> str:
    """1990 -> nineteen ninety, 2026 -> twenty twenty-six, 2005 -> two thousand five."""
    if 2000 <= y < 2010:
        return number_words(y)
    high, low = divmod(y, 100)
    return number_words(high) + " " + ("hundred" if low == 0 else
                                       ("oh " + ONES[low]) if low < 10 else number_words(low))


def date_words(d: date, today: date) -> str:
    text = f"the {ordinal_words(d.day)} of {MONTHS[d.month - 1]}"
    if 0 <= (d - today).days <= 7:
        text = f"{d.strftime('%A')}, {text}"
    if d.year != today.year:
        text += f", {year_words(d.year)}"
    return text


def time_words(hour: int, minute: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    h = hour % 12 or 12
    if minute == 0:
        return f"{number_words(h)} {suffix}"
    m = number_words(minute) if minute >= 10 else "oh " + ONES[minute]
    return f"{number_words(h)} {m} {suffix}"


def speakable(text: str, today: date) -> str:
    def _datetime(m):
        d = _parse(m.group(1))
        return m.group(0) if d is None else f"{date_words(d, today)}, at {time_words(int(m[2]), int(m[3]))}"

    def _date(m):
        d = _parse(m.group(1))
        return m.group(0) if d is None else date_words(d, today)

    def _spoken_date(m, day: str, month: str, year: str | None):
        try:
            d = date(int(year) if year else today.year, MONTHS.index(month) + 1, int(day))
        except ValueError:
            return m.group(0)
        return date_words(d, today)

    def _time(m):
        hour, minute, ampm = int(m[1]), int(m[2]), (m[3] or "").strip().lower().replace(".", "")
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return time_words(hour, minute)

    def _rupees(m):
        return number_words(int(m.group(1).replace(",", ""))) + " rupees"

    text = _DATETIME.sub(_datetime, text)
    text = _DATE.sub(_date, text)
    text = _DAY_OF_MONTH.sub(lambda m: _spoken_date(m, m[1], m[2], m[3]), text)
    text = _MONTH_DAY.sub(lambda m: _spoken_date(m, m[2], m[1], m[3]), text)
    text = _TIME.sub(_time, text)
    text = _RUPEES_BEFORE.sub(_rupees, text)
    text = _RUPEES_AFTER.sub(_rupees, text)
    text = _GROUPED.sub(lambda m: number_words(int(m.group(0).replace(",", ""))), text)
    return text


def _parse(text: str) -> date | None:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None
