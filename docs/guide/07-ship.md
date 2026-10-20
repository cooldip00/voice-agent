# Step 7: README, diagram, demo video

| File | What it is |
|---|---|
| `README.md` | Problem, results, how it works, measured numbers, honest limits, how to run it |
| `docs/architecture.md` | The pipeline and where code takes over from the LLM, as Mermaid diagrams (GitHub draws them) |
| `docs/guide/07-ship.md` | This page: the demo shot list and the checks before pushing |

## 1. Check the diagrams
GitHub renders Mermaid in Markdown. To see it before pushing, open `README.md` in VS Code and press
`Ctrl+Shift+V` (the built-in preview needs the "Markdown Preview Mermaid Support" extension), or paste the
` ```mermaid ` block into https://mermaid.live.

## 2. Record the demo (2 minutes)
Set up first:
```powershell
cd "D:\ResumeProject\voice-agent"
uv run python -m scripts.seed_bank
uv run python -m agent.bot
```
Record the screen with Win+Alt+R (Xbox Game Bar), or OBS if you want the microphone and the bot on separate
tracks. Show the browser and a terminal side by side; make the terminal font large.

| Time | Show | Say (short) |
|---|---|---|
| 0:00–0:10 | `nvidia-smi` and `ollama ps` | "Everything runs on this laptop GPU. No cloud, no API keys." |
| 0:10–0:45 | Happy call to Rahul: DOB, "this Friday", goodbye, hang-up | Let the call speak. Point at the reply speed. |
| 0:45–1:00 | `calls/<id>.json` | "The record comes from the tools, not from what the model said: 4500, the 23rd." |
| 1:00–1:20 | A second call: cut the bot off mid-sentence | "It stops within 0.7 s." |
| 1:20–1:40 | `$env:CUSTOMER=3`, restart, Arjun: "I lost my job... I can only pay 500" | "Hardship goes to a person, by code. The note keeps the 500." |
| 1:40–2:00 | `testbench/results/20261020-164649` printout and the latency table in the README | "27 of 27 scripted calls; p95 1.8 s. The limits are in the README." |

Tips:
- Do the calls between 08:00 and 19:00, or set `$env:ALLOW_ANY_HOUR=1` (the README says it's dev only).
- Wear headphones: the bot's voice from the speakers reaches the mic, and VAD hears it as you cutting in.
- Do a dry run; the first call after starting is the slowest (models warm up).
- Keep the video under 100 MB so it can go in the repo, or upload it (YouTube unlisted) and link it from the README.

## 3. Before pushing
```powershell
git status --short                       # nothing unexpected
git ls-files | Select-String "\.db$|calls/|metrics/|results/|docs/study|\.wav$"   # must print nothing
uv run pytest                            # all pass
```
Then make the repo on GitHub (empty, no README), and:
```powershell
git remote add origin https://github.com/<you>/voice-agent.git
git push -u origin main
```

## When it breaks
| You see | Why | Fix |
|---|---|---|
| The README shows the Mermaid source as text | Not on GitHub, or the fence isn't ` ```mermaid ` | GitHub renders it; local previews need an extension |
| The bot interrupts itself in the video | Speaker audio reaches the mic | Headphones |
| `Not calling: it is 19:30...` | The calling-hours rule | Record earlier, or `$env:ALLOW_ANY_HOUR=1` |
