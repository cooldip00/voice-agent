from datetime import date

import pytest

from agent.speech import number_words, ordinal_words, speakable, year_words
from tests.conftest import TODAY   # Tuesday 2026-10-20


@pytest.mark.parametrize("n, words", [
    (0, "zero"), (15, "fifteen"), (40, "forty"), (99, "ninety-nine"), (100, "one hundred"),
    (4500, "four thousand five hundred"), (4200, "four thousand two hundred"),
    (125000, "one lakh twenty-five thousand"), (10000000, "one crore"), (2100005, "twenty-one lakh five"),
])
def test_number_words_indian_system(n, words):
    assert number_words(n) == words


@pytest.mark.parametrize("n, words", [(1, "first"), (2, "second"), (3, "third"), (12, "twelfth"),
                                      (20, "twentieth"), (23, "twenty-third"), (31, "thirty-first")])
def test_ordinals(n, words):
    assert ordinal_words(n) == words


@pytest.mark.parametrize("y, words", [(1990, "nineteen ninety"), (2026, "twenty twenty-six"),
                                      (2005, "two thousand five"), (1905, "nineteen oh five")])
def test_years(y, words):
    assert year_words(y) == words


@pytest.mark.parametrize("text, spoken", [
    ("You owe 4500 rupees.", "You owe four thousand five hundred rupees."),
    ("You owe 4,500 rupees.", "You owe four thousand five hundred rupees."),
    ("That is Rs. 1,25,000 in total.", "That is one lakh twenty-five thousand rupees in total."),
    ("Pay \u20b94500 today.", "Pay four thousand five hundred rupees today."),
    ("By 2026-10-23, please.", "By Friday, the twenty-third of October, please."),
    ("It was due on 2026-10-06.", "It was due on the sixth of October."),
    ("Born 1990-03-05?", "Born the fifth of March, nineteen ninety?"),
    ("I'll call on 2026-10-21T17:00.", "I'll call on Wednesday, the twenty-first of October, at five PM."),
    ("At 5:30 pm then.", "At five thirty PM then."),
    ("At 09:05.", "At nine oh five AM."),
    ("Thank you for your time.", "Thank you for your time."),
    ("It was due on Wednesday, the 6th of October.", "It was due on the sixth of October."),
    ("Due on October 6, 2026.", "Due on the sixth of October."),
    ("By Friday, 2026-10-27.", "By Tuesday, the twenty-seventh of October."),
    ("Pay by the 23rd of October.", "Pay by Friday, the twenty-third of October."),
    ("Born 5th of March, 1990.", "Born the fifth of March, nineteen ninety."),
    ("May I have your date of birth?", "May I have your date of birth?"),
])
def test_speakable(text, spoken):
    assert speakable(text, TODAY) == spoken


def test_bad_date_left_alone():
    assert speakable("on 2026-02-30", TODAY) == "on 2026-02-30"


def test_year_said_only_when_not_this_year():
    assert speakable("2027-01-02", date(2026, 12, 30)) == "Saturday, the second of January, twenty twenty-seven"
