# Step 3: fake bank, tools, call rules

No models in this step. It is plain Python that the bot will call in step 4, plus tests that prove
the rules hold. Code: [agent/bank.py](../../agent/bank.py), [agent/call_state.py](../../agent/call_state.py),
[agent/tools.py](../../agent/tools.py), tests in [tests/](../../tests/).

| File | What it holds |
|---|---|
| `agent/bank.py` | SQLite fake bank: customers, loans, payments, and what calls record (promises, callbacks, disputes) |
| `agent/call_state.py` | Per-call state (verified? attempts? what was recorded?) and every rule as a plain function |
| `agent/tools.py` | The 6 tools the LLM may call, and their descriptions for Pipecat |
| `scripts/seed_bank.py` | Creates `data/bank.db` |

## 1. Create the fake bank
```powershell
cd "D:\ResumeProject\voice-agent"
uv run python -m scripts.seed_bank
```
Real output:
```
created data\bank.db (dates relative to 2026-10-19)

id  name           date of birth  loan        due   overdue  scenario
1   Rahul Sharma   1990-03-05     PL-1001    4500     14 d   happy path
2   Priya Nair     1988-11-21     TW-1002    3200     10 d   says she already paid
3   Arjun Mehta    1985-07-14     PL-1003    6800     20 d   lost his job
4   Sneha Reddy    1993-01-30     CD-1004    4200     35 d   disputes the amount
5   Vikram Singh   1979-09-09     PL-1005    5600      7 d   asks to be called later
6   Anjali Gupta   1996-05-18     TW-1006    2900     12 d   someone else answers
```
All names, numbers and dates are made up. Due dates are counted back from today, so re-run this
whenever you want fresh dates; it replaces the old bank, including anything calls recorded.
Sneha's amount is two EMIs because she missed two.

## 2. Run the tests
```powershell
uv run pytest
```
Real output:
```
........................................................................ [ 98%]
.                                                                        [100%]
73 passed in 2.05s
```
`uv run pytest -v` lists each test by name. The tests use their own temporary bank and a fixed
"today" (20 Oct 2026, 11:00), so they never touch `data/bank.db` and pass on any date.

**Do the tests really catch a broken rule?** Checked by switching the identity gate off on purpose:
7 tool tests failed (`test_gated_tools_refuse_before_verification` and others). So a change that
leaks loan data before verification cannot pass the tests unnoticed.

## 3. The tools
| Tool | Needs identity? | Refuses when |
|---|---|---|
| `verify_identity(date_of_birth)` | – | not `YYYY-MM-DD` (not counted as a try); wrong (counted); after 3 wrong tries it **locks** for the rest of the call |
| `get_loan_details()` | yes | not verified |
| `record_promise(amount, date)` | yes | amount < 1 or above the amount due; not whole rupees; date in the past or more than 7 days away |
| `schedule_callback(call_at)` | **no** | not `YYYY-MM-DDTHH:MM`; in the past; more than 7 days away; outside 08:00–19:00 |
| `flag_dispute(kind, reason)` | yes | kind not `already_paid` / `wrong_amount` / `other`; empty reason |
| `end_call(outcome)` | no | the outcome doesn't match what was recorded (for example `promise_to_pay` with no promise) |

`schedule_callback` needs no identity on purpose: whoever answers may say "call later", and it tells
them nothing about the loan. The limits (hours, days, attempts) live in [agent/config.py](../../agent/config.py).

## 4. Try the tools by hand
Paste this whole block into PowerShell (run `seed_bank` first):
```powershell
@'
from agent import bank, config
from agent.call_state import CallState
from agent.tools import CallTools

conn = bank.connect(config.BANK_DB)
tools = CallTools(CallState(call_id="demo", customer_id=1), conn)
for name, args in [
    ("get_loan_details", {}),
    ("verify_identity", {"date_of_birth": "1990-03-06"}),
    ("verify_identity", {"date_of_birth": "1990-03-05"}),
    ("get_loan_details", {}),
    ("record_promise", {"amount": 9000, "date": "2026-10-22"}),
    ("end_call", {"outcome": "promise_to_pay"}),
]:
    print(f"{name}({args})\n  -> {tools.call(name, args)}")
'@ | uv run python -
```
Real output:
```
get_loan_details({})
  -> {'ok': False, 'error': 'not_verified', 'message': 'Identity is not verified. Ask for the date of birth first. Do not share or discuss loan details.'}
verify_identity({'date_of_birth': '1990-03-06'})
  -> {'ok': False, 'error': 'no_match', 'attempts_left': 2, 'message': 'That date of birth does not match. Ask the caller to repeat it. Do not hint at the right one.'}
verify_identity({'date_of_birth': '1990-03-05'})
  -> {'ok': True, 'verified': True}
get_loan_details({})
  -> {'ok': True, 'customer_name': 'Rahul Sharma', 'product': 'personal loan', 'emi': 4500, 'amount_due': 4500, 'due_date': '2026-10-05', 'days_overdue': 14, 'last_payment': {'paid_on': '2026-09-05', 'amount': 4500}}
record_promise({'amount': 9000, 'date': '2026-10-22'})
  -> {'ok': False, 'error': 'invalid_promise', 'message': 'amount is more than the 4500 rupees due'}
end_call({'outcome': 'promise_to_pay'})
  -> {'ok': False, 'error': 'invalid_outcome', 'message': 'no promise was recorded; call record_promise first'}
```
Each refusal says **what to do instead**. In step 4, the LLM reads these messages and acts on them.
The dates depend on the day you seeded. Change the list to try other cases: a callback at
`2026-10-20T21:00`, three wrong dates of birth, `customer_id=2`.

Look at what the calls recorded:
```powershell
@'
import sqlite3
db = sqlite3.connect("data/bank.db")
for table in ("promises", "callbacks", "disputes"):
    print(table, db.execute(f"SELECT * FROM {table}").fetchall())
'@ | uv run python -
```
After the block above, all three are empty: the 9,000-rupee promise was refused, so nothing was saved.

## When something fails
| Symptom | Fix |
|---|---|
| `sqlite3.OperationalError: no such table` | Run `uv run python -m scripts.seed_bank` first |
| `ModuleNotFoundError: No module named 'agent'` | Run from the project root, with `-m` (`python -m scripts.seed_bank`, not `python scripts\seed_bank.py`) |
| `pytest` not found | `uv sync` installs the dev tools, including pytest |
| `PermissionError` when re-seeding | Something still has `data\bank.db` open (another Python, a DB viewer). Close it |
