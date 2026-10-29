# Step 4: the collections call

A full call in the browser: greeting with the recording notice, date of birth, amount due, then a promise, callback,
dispute or escalation, then goodbye and hang-up. Each call leaves `calls/<call_id>.json`, built from what the tools
accepted, never from what the LLM said.
Code: [agent/bot.py](../../agent/bot.py), [agent/prompts.py](../../agent/prompts.py),
[agent/speech.py](../../agent/speech.py), [agent/call_state.py](../../agent/call_state.py), [agent/tools.py](../../agent/tools.py).

| File | What changed in step 4 |
|---|---|
| `agent/prompts.py` | Two prompts: `identity_prompt` (no word about the debt) until the DOB matches, then `system_prompt` with the call steps; the fixed lines code speaks (greeting, amount due, promise confirmation, personal matter, goodbye) |
| `agent/bot.py` | Tools wired to the LLM; guards on every sentence before it is spoken; the call record saved at the end |
| `agent/speech.py` | Digits and ISO dates into words just before Kokoro speaks ("2026-10-23" → "Friday, the twenty-third of October") |
| `agent/call_state.py` | `resolve_day` / `resolve_time` / `resolve_birth_date` (the caller's words → dates, in code); `unsafe_to_say` (the guards); `call_record` |
| `agent/tools.py` | `escalate_to_human` (7th tool); tools take dates **as the caller said them** ("Friday", "5th of March 1990"); `IDENTITY_TOOLS`, the 3 tools shown before the DOB matches |

## 1. Make a call
```powershell
cd "D:\ResumeProject\voice-agent"
uv run python -m scripts.seed_bank        # fresh due dates (optional, but resets what earlier calls recorded)
uv run python -m agent.bot
```
Wait for `Uvicorn running on http://localhost:7860` (about 20 s: Whisper, Kokoro and Qwen load and warm up first),
open http://localhost:7860, allow the microphone, click **Connect**.

| Setting | PowerShell, before `agent.bot` | Default |
|---|---|---|
| Who is called | `$env:CUSTOMER=3` (ids from `seed_bank`: 1 Rahul … 6 Anjali) | 1, Rahul |
| Outside 08:00–19:00 | `$env:ALLOW_ANY_HOUR=1` (dev only) | calls refused: `Not calling: it is 21:30, calls are allowed 8:00-19:00` |

Remove them with `Remove-Item Env:CUSTOMER`, `Remove-Item Env:ALLOW_ANY_HOUR`.

A call to Rahul (DOB 1990-03-05, owes 4500), as heard:
```
bot   : Hello, this is Asha from Kosha Finance. This call is recorded. Am I speaking with Rahul Sharma?
caller: Yes, this is Rahal speaking.                       ← Whisper's spelling; names don't matter, the DOB does
bot   : I understand. Could you please provide me with your date of birth for verification purposes?
caller: It's 5th of March. 1990.
bot   : Thank you for confirming. 4500 rupees is due on your personal loan since 2026-10-06. When will you be able to pay?
caller: Okay, I can pay the full amount this Friday.
bot   : 4500 rupees this Friday, is that correct?
caller: Yes, that's right.
bot   : Thank you. I have recorded 4500 rupees by 2026-10-23. Is there anything else?
caller: No, that's all thank you.
bot   : Great.
bot   : Thank you for your time. Goodbye.                      ← then the bot hangs up
```
Dates are written as ISO in the transcript; the caller hears them in words ("since the sixth of October",
"by Friday, the twenty-third of October").

## 2. Read the record
```powershell
Get-ChildItem calls | Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content
```
Real output (shortened):
```json
{
  "call_id": "20261020-011444-c1",
  "customer_id": 1,
  "outcome": "promise_to_pay",
  "ended_by": "agent",
  "promise_date": "2026-10-23",
  "amount": 4500,
  "reason": null,
  "callback": null,
  "notice_given": true,
  "verified": true,
  "verify_attempts": 0,
  "seconds": 76.9,
  "tool_calls": [
    {"name": "verify_identity", "arguments": {"date_of_birth": "1990-03-05"}, "result": {"ok": true, ...}},
    {"name": "record_promise", "arguments": {"amount": 4500, "day": "Friday"}, "result": {"ok": true, "amount": 4500, "pay_by": "2026-10-23"}},
    {"name": "end_call", "arguments": {"outcome": "promise_to_pay"}, "result": {"ok": true, "outcome": "promise_to_pay"}}
  ],
  "llm_slips": ["goodbye without end_call: Have a nice day, Rahul Sharma."],
  "transcript": [...]
}
```
- `outcome` is end_call's; if the call ended without one, `final_outcome` takes it from what the tools recorded.
- `ended_by: "caller"` means the caller hung up before end_call.
- `llm_slips` are sentences the guards stopped (section 4), plus days code took from the caller's words
  (`day not in the caller's words: '2026-10-20', used the caller's 2026-10-23`). They are counted in step 5.

The same call is also in the bank: `promises` in `data/bank.db` has the row (step 3, section 4 shows how to look).

## 3. Four test calls (run on 20 Oct 2026, Qwen 2.5 7B)
Played into the browser by a scripted caller (Kokoro voice reading the caller's lines), so they can be repeated.

| Customer | Caller does | Record | Notes |
|---|---|---|---|
| 1 Rahul | gives DOB, pays 4500 "this Friday" | `promise_to_pay`, 2026-10-23, 4500, ended by agent | Friday worked out by code |
| 1 Rahul | three wrong DOBs | `no_resolution`, `verify_attempts: 3`, nothing about the loan said | locks after the 3rd; bot offers a callback |
| 3 Arjun | lost his job, can pay 500 | `escalated`, reason "Lost job, can only pay 500 rupees this week." | below the 25% minimum, so no promise |
| 6 Anjali | husband answers, "what is this about?" | `callback` 2026-10-21T17:00, 0 slips | bot: "It's a personal matter…". Before the identity prompt, the guard had to stop "It's regarding an overdue EMI" |

## 4. What code does, and what the LLM does
The LLM handles language. Code owns facts, dates and the rules. Each line below came from a failure in a test call.

| Code does | Because Qwen 7B, in test calls… |
|---|---|
| Keeps the debt out of the LLM until the DOB matches: `identity_prompt` + 3 tools, then `verify_identity` swaps in the full prompt and all 7 tools | told Anjali's husband "It's regarding an overdue EMI" while the prompt said not to |
| Says the amount due and the promise confirmation itself (fixed lines, `SPOKEN_AFTER` in `prompts.py`) | said "due on Wednesday, the 6th of October" for Tuesday 2026-10-06 |
| Works out dates: the LLM passes `day="Friday"` or `"the 23rd"`, `resolve_day` turns it into 2026-10-23 | turned "this Friday" into 2026-10-27 (a Tuesday), even with the dates listed in its prompt |
| Refuses a YYYY-MM-DD `day` and takes the day from the caller's own words instead (`caller_day`): the newest caller turn that names one day | passed today's date for "this Friday" (3 of 20 test runs, 3 of 3 live calls), and once refused, asked the caller to say it "in a different way" instead of retrying |
| Reads the DOB as said (`resolve_birth_date`: "5th of March 1990", "05/03/1990", ISO) | asked callers to "say it in the format YYYY-MM-DD" instead of calling the tool, when the tool wanted ISO |
| Reads back first, records on "yes" (prompt step 4) | called `record_promise` 0/10 right after the caller named a day; 9–10/10 after a read-back and "yes" |
| Joins back-to-back caller messages (`JoinCallerTurns`) | called `verify_identity` 1/10 when a pause split "14th of July." / "1985" into two turns; 10/10 joined |
| Ends the call itself when the LLM says goodbye without `end_call` | often said "Have a nice day" and never called `end_call` |

**Guards** (`to_words` in `bot.py`, Kokoro's text transformer, runs them on every LLM sentence before it is spoken: the tool-name check itself, the rest through `unsafe_to_say` in `call_state.py`):

| Stops a sentence that | Example stopped |
|---|---|
| mentions the debt before the DOB matched (RBI: no disclosure to third parties) | "It's regarding an overdue EMI that needs to be settled." |
| asks the caller for a date format | "Please say it in the format YYYY-MM-DD." |
| has an ISO date no tool returned (today's date is allowed: the prompt gives it) | "Can you manage to pay 6800 rupees by next Friday, 2026-10-27?" |
| claims something was recorded before any tool recorded it | "I see, your date of birth is now recorded as 1990-01-02." |
| writes a tool call as text | `end_call(outcome="promise_to_pay")` |

If the debt guard fires before verification, code says instead, once: "I'm sorry, this is a personal matter. I can only discuss it with Anjali Gupta, after confirming their date of birth." Any other stopped sentence is silent, logged as a warning, added to `llm_slips`, and kept out of the LLM context and the
transcript, so the LLM doesn't build on something the caller never heard. The guards match words, so they are coarse
on purpose: silencing a harmless sentence costs a moment of silence; saying a harmful one can't be undone.

## 5. Logs you will see
| Log line | Meaning |
|---|---|
| `Recording notice played` | the notice finished playing; `verify_identity` refuses until then |
| `Tool record_promise({'amount': 4500, 'day': 'Friday'}) -> {...} [3 ms]` | a tool call and its result |
| `WARNING Not speaking, LLM mentions the debt before identity is verified: '...'` | a guard stopped a sentence |
| `updated settings fields: {'system_instruction'}` | the DOB matched; the LLM now has the full prompt |
| `WARNING LLM said goodbye without end_call, ending the call: '...'` | code ended the call instead |
| `ERROR QuietKokoroTTS#0 error: TTS context ... completed with no audio` | **harmless**: follows a silenced sentence (Pipecat expects audio for every sentence) |
| `Call record: calls\20261020-011444-c1.json  outcome=promise_to_pay` | the record was saved |

## When something fails
| Symptom | Fix |
|---|---|
| Browser connects, nothing is said, log says `Not calling: it is ...` | Outside 08:00–19:00. `$env:ALLOW_ANY_HOUR=1` for a test |
| `FileNotFoundError: data\bank.db missing` | `uv run python -m scripts.seed_bank` |
| `[Errno 10048]` / port 7860 already in use | An old bot is still running. `Get-NetTCPConnection -LocalPort 7860 \| % { Stop-Process -Id $_.OwningProcess }` |
| The bot keeps asking for the DOB, `verify_identity` refuses with `notice_not_given` | You talked over the greeting, so the recording notice never finished playing. Let it finish |
| The bot never says the amount | The DOB didn't match: check the record's `tool_calls`. DOBs are in the `seed_bank` table (step 3) |
| Long silences after you speak | Normal for some turns: the sentence was stopped by a guard (see the log), or Qwen is thinking (measured in step 5) |
| `promise_date` is `null` though you agreed a day | The bot must read it back and hear "yes" before it records. Answer the read-back |
