# Run guide

How to run and check each piece of the voice agent, step by step. Every command here was run on the
dev laptop (RTX 3070 Ti Laptop, 8 GB) and the output shown is real.

`docs/guide/` is the **how** (commands, expected output, what to try, what to do when it breaks).
`docs/study/` is the **why** (design choices, questions); it stays out of git.

| Step | Guide | What you can run | Status |
|---|---|---|---|
| 0 | [00-setup.md](00-setup.md) | Environment check; each model alone: Ollama, Kokoro, Whisper, Silero VAD, Smart Turn | done |
| 1 | [01-parts.md](01-parts.md) | `try_tts` / `try_stt` / `try_llm`, each timed | done |
| 2 | [02-bot.md](02-bot.md) | `agent.bot`: browser call with interruptions | done |
| 3 | [03-bank-and-tools.md](03-bank-and-tools.md) | `seed_bank`, `pytest`, the 6 tools by hand | done |
| 4 | [04-collections-call.md](04-collections-call.md) | Full browser call; end-of-call JSON in `calls/`; the guards | done |
| 5 | [05-latency.md](05-latency.md) | `latency_report.py`: delay per stage, 3B vs 7B, VAD wait | done |
| 5b | [05-latency.md § 5](05-latency.md#5-step-5b-kokoro-on-the-gpu) | Kokoro on the GPU: `check_env` shows the provider | done |
| 6 | [06-testbench.md](06-testbench.md) | `testbench/run.py`, 9 scenarios, scored by code | done |
| 6b | [06-testbench.md § 6](06-testbench.md#6-step-6b-code-closes-the-call) | Code ends the call after settling tools, hardship and silence: 27/27 | done |
| 7 | [07-ship.md](07-ship.md) | README, `docs/architecture.md`, demo shot list, checks before pushing | done (video: later) |

## Rules for every command
- Open PowerShell and run from the project root: `cd "D:\ResumeProject\voice-agent"`.
- Run Python through `uv run ...` so it uses the project's `.venv`. No need to activate anything.
- The Ollama app must be running (tray icon). `ollama list` fails if it isn't.
- Snippets that start with `@'` and end with `'@ | uv run python -` are pasted whole into PowerShell.
  The closing `'@` must be at the very start of its line.
- Generated audio goes in `recordings/` (gitignored).
