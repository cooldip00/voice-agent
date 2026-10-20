# Step 6: the test bench

One command phones the live bot once per scenario, plays a caller that answers what it hears, and scores
every call with plain code. No human listens and no LLM judges.
Code: [testbench/run.py](../../testbench/run.py), [testbench/caller.py](../../testbench/caller.py),
[testbench/scenario.py](../../testbench/scenario.py), [testbench/score.py](../../testbench/score.py),
[testbench/scenarios/](../../testbench/scenarios/).

| File | What changed in step 6 |
|---|---|
| `testbench/run.py` | New. Starts the bot, reseeds the bank before each call, places the calls, scores them, prints a table, saves `testbench/results/<time>/` |
| `testbench/caller.py` | New. The caller: WebRTC like the browser, Kokoro `am_michael` to speak, Whisper `tiny.en` to listen, both on the CPU. Times interrupts and silence |
| `testbench/scenario.py` | New. Reads a scenario file; picks the caller's answer to each bot turn; hangs up after saying the same line 3 times running |
| `testbench/score.py` | New. The checks: outcome, amount, dates, no debt talk before the date of birth matched, no threats, replies under 5 s, bot ended the call, bank matches the record |
| `testbench/scenarios/*.toml` | New. 9 scenarios: who is called, how the caller answers, what the call must end with |
| `agent/bot.py` | The customer comes from the browser's offer (`request_data: {"customer": N}`); the call id ends in `-c<customer>` |
| `agent/call_state.py` | The record says after how many caller turns the date of birth matched (`verified_after_caller_turns`) |
| `tests/test_testbench.py` | New. The bench's own logic without models: scenario files, the caller's choices, every check |

## 1. Run it
Close any bot you started yourself (the bench starts its own) and keep Ollama running.
```powershell
uv run python -m testbench.run                     # all 9 scenarios once, about 15 minutes
uv run python -m testbench.run happy hardship      # only these
uv run python -m testbench.run --runs 3            # each 3 times: about 40 minutes
uv run python -m testbench.run --no-bot            # use a bot already running on --url (default http://127.0.0.1:7860)
```
The bench sets `ALLOW_ANY_HOUR=1` for its bot, so it runs outside calling hours too. It reseeds `data\bank.db`
before every call, so each call starts from the same bank.

## 2. What it prints
One line per call as it ends, then a table:
```
PASS  happy-1   promise_to_pay
FAIL  hardship-1   no_resolution   outcome: no_resolution, want escalated; bot ended the call: caller
FAIL  silence-1   promise_to_pay  asks after silence: said nothing
...
| scenario | passed |
|---|---|
| already_paid | 0 / 3 |
...
| **all** | **14 / 27** |

Checks that failed: bot ended the call (9x), outcome (3x), asks after silence (3x)
Voice to voice over 150 replies: p50 1372 ms, p95 2395 ms
```
A line reads: scenario-run, the outcome the bot recorded, then every failed check with what it got and wanted.

## 3. What it saves
`testbench\results\<date-time>\` (gitignored):
- `report.json`: the setup (LLM, Whisper, VAD wait, Kokoro device), and per call the checks, reply times and
  the caller's log: what it heard, what it said, interrupt and silence timings, whether it gave up.
- `<scenario>-<run>.wav`: both sides on one track. **Listen to every failed call** before blaming the bot.
- `bot.log`: the bot's own log for the whole run.

The bot's call records (`calls\`) and latency rows (`metrics\`) are written as in steps 4 and 5.

Read one failed call's conversation:
```powershell
@'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))["results"]
for c in r:
    if c["scenario"] == sys.argv[2] and c["run"] == int(sys.argv[3]):
        for t in c["caller"]["turns"]:
            print("BOT:   ", t["bot"]); print("CALLER:", t["caller"])
'@ | uv run python - testbench\results\20261020-145403\report.json hardship 1
```

## 4. The 9 scenarios
| Scenario | Customer | The call must end with |
|---|---|---|
| happy | 1 Rahul | promise of 4500 this Friday |
| interrupts | 1 Rahul | the same, after the caller talks over the amount line; the bot stops within 1 s |
| silence | 1 Rahul | the same, after 20 s of quiet; the bot speaks up within 12 s |
| wrongdob | 1 Rahul | three wrong dates of birth: no word about the loan, no promise |
| wrong_person | 6 Anjali | her husband answers: a callback for her, no word about the loan |
| call_later | 5 Vikram | callback tomorrow 11:00 |
| already_paid | 2 Priya | a dispute, not a promise |
| dispute | 4 Sneha | a dispute about the amount |
| hardship | 3 Arjun | escalated to a human |

Every call is also checked for: recording notice, no threats, every reply under 5 s, the bot hung up (not the
caller), and the bank holds exactly the rows the record names.

### Writing a scenario
Copy one in `testbench\scenarios\`. The caller picks the **first** rule whose `when` regex is in what it heard,
so put narrow rules (a read-back: `is that (right|correct)`) above broad ones (`pay`). Write a regex backslash
twice in TOML: `"confirm\\b"`. Answer only what a reasonable person would say: a caller tuned until the bot
passes is testing the caller. Then `uv run pytest tests\test_testbench.py`; it fails on a backspace in a pattern,
an unknown customer, or an outcome the bot can't record.

## 5. Results (20 Oct 2026, 7B, VAD 0.2, Kokoro on the GPU)
`--runs 3`, 27 calls, 41 minutes:

| Scenario | Passed | What went wrong |
|---|---|---|
| happy | 3 / 3 | |
| interrupts | 3 / 3 | the bot's audio stopped 0.67–0.71 s after the caller cut in |
| call_later | 2 / 3 | once the callback was booked, the bot never ended the call; it read a tool result aloud ("call ended by user") |
| dispute | 2 / 3 | dispute recorded, then kept asking "when can we get back to you?" |
| wrong_person | 2 / 3 | callback booked, then kept answering "what is this about?" without ending |
| wrongdob | 2 / 3 | after the third wrong date it offered a callback again and again instead of ending |
| already_paid | 0 / 3 | dispute recorded, then "a team will check this" on repeat until the caller gave up |
| hardship | 0 / 3 | never escalated: offered a payment plan, booked a callback, or said goodbye without hanging up |
| silence | 0 / 3 | never spoke during the 20 s of quiet: the bot has no silence timer |
| **all** | **14 / 27** | voice to voice p50 1372 ms, p95 2395 ms; slowest reply 4.2 s |

The same 9 scenarios passed 5, 3 and 6 times in the three rounds. One run of each is not enough to call a fix.

Every failure is the bot's, in three groups (the fixes are step 6b):
1. **The job is done but the call goes on** (9 checks). After `flag_dispute`, `schedule_callback` or the identity
   lock, nothing in code says goodbye and hangs up; the LLM keeps talking.
2. **Hardship isn't escalated** (3). "Lost my job" with a partial payment is left to the LLM, which bargains.
3. **No re-prompt after silence** (3).

Mistakes the bench made along the way, now fixed (the study note has the detail): a TOML `"\b"` read as a backspace
broke rules; broad rules (`pay`) caught read-backs; a talked-over line looked like the bot leaking the debt; the
caller looped on its fallback forever, so it now gives up after the same line 3 times.

## 6. Step 6b: code closes the call
The three groups above, fixed in code rather than in the prompt:

| File | What changed in step 6b |
|---|---|
| `agent/bot.py` | `close_call`: records the outcome, says a fixed line and the goodbye, hangs up. Called when `flag_dispute`, `escalate_to_human` or `schedule_callback` succeeds, and on the identity lock. A caller turn that names hardship is escalated by code before the LLM sees it. 8 s of silence: "Are you still there?", twice, then the call ends |
| `agent/prompts.py` | The lines code says: `CLOSES_CALL`, `LOCKED`, `STILL_THERE`, `IDLE_END`. The prompt says the system ends the call after those tools |
| `agent/call_state.py` | `hardship_said` (lost job, illness, a death...). The guard stops text that looks like JSON |
| `agent/tools.py` | `flag_dispute`, `escalate_to_human`, `schedule_callback` record once per call; a second call returns the first |
| `agent/config.py` | `USER_IDLE_SECS = 8.0` |
| `testbench/scenario.py`, `caller.py` | A rule can hang up (`hang_up = true`); the log tells `caller_hung_up` (said bye) from `caller_gave_up` (a loop) |
| `testbench/scenarios/call_later.toml` | The caller hangs up after "Bye" |

Nothing new to run: `uv run pytest` (270 tests), then the bench as in § 1.

**Results, same setup, `--runs 3`, 27 calls, 29 minutes (results `20261020-155200`):**

| | Step 6 | Step 6b |
|---|---|---|
| calls passed | 14 / 27 | **27 / 27** |
| already_paid, hardship, silence | 0 / 3 each | 3 / 3 each |
| call_later, dispute, wrong_person, wrongdob | 2 / 3 each | 3 / 3 each |
| voice to voice p50 / p95 | 1372 / 2395 ms (150 replies) | 1316 / 1783 ms (98 replies) |
| slowest reply | 4.2 s | 1.9 s |
| interrupt: bot stopped after | 0.67–0.71 s | 0.65–0.70 s |
| silence: bot asked after | never | 6.1 s of the caller's quiet (8 s after the bot stopped; the caller counts from 2 s) |

Fewer replies and a faster p95: calls end when their business is done, so the long, slow loops are gone.

What the bench still shows:
- **Hardship fires on the first sentence.** Smart Turn ended the caller's turn at "I lost my job last month."
  Code escalated at once, and the bot's closing line talked over "I can only pay five hundred rupees this week".
  The escalation note for the human lacks the 500. It passed, because the check is "escalated", in 3 of 3.
- **The LLM still slips; the guards catch it.** In one happy call Qwen said "this Friday, 2026-10-20" (a Tuesday);
  the date guard silenced it and the call passed.
- **27/27 is these 9 scenarios.** The hardship words and the closing lines were written with them in view.

## When it breaks
| You see | Why | Fix |
|---|---|---|
| `The bot didn't start in 4 minutes` | Model download, or another bot holds port 7860 | Read `bot.log` in the results folder; close the other bot, or use `--no-bot` |
| `FAIL ...: no call record` | The bot crashed or the call never connected | `bot.log`: look for the traceback near that call's time |
| `WinError 10054` / `RemoteProtocolError` while loading | Hugging Face dropped the download of `tiny.en` | Run again: the caller uses the cached copy first and only goes online when there is none |
| A rule never fires; pytest says `TOML read a \b as a backspace` | `"\b"` in a TOML string is a backspace character | Write `"\\b"` |
| Caller answers `Sorry, could you say that again?` to a clear question | No rule matched what tiny.en heard (it hears "Asha" as "a shaft") | Look at `heard` in `report.json`; widen the rule's regex |
| `bot ended the call: caller` | The caller gave up (same line 3 times), or the bot stopped talking without hanging up | Read the turns (§ 3): a loop means the bot didn't end the call |
| Every reply is slow | Another program is on the GPU, or a second Ollama model is loaded | `ollama ps`; close heavy programs and run again |
| A browser call ends with "I'm sorry to hear that..." when you only mentioned being ill in passing | `hardship_said` is broad on purpose: a false match hands the case to a person | Expected. Narrow `_HARDSHIP` in `agent/call_state.py` only with a test for the case |
| The bot asks "Are you still there?" while you think | 8 s of quiet after it spoke | `USER_IDLE_SECS` in `agent/config.py` |
