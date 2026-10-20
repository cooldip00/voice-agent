# Step 5: measure the delay, tune, 3B vs 7B

How long the caller waits after they stop speaking, split into its parts, from real calls. Then two settings
compared on those numbers: the LLM (Qwen 7B vs 3B) and how long the VAD waits for silence.
Code: [agent/metrics.py](../../agent/metrics.py), [scripts/latency_report.py](../../scripts/latency_report.py),
[agent/bot.py](../../agent/bot.py) (`QuietKokoroTTS`), [agent/speech.py](../../agent/speech.py) (`sentences`).

| File | What changed in step 5 |
|---|---|
| `agent/metrics.py` | New. Pipecat's `UserBotLatencyObserver` on the pipeline; one JSON line per bot reply in `metrics/<call_id>.jsonl`, plus GPU memory at the end of the call |
| `scripts/latency_report.py` | New. p50/p95 per stage and voice to voice, one column per setup; what the same calls did (outcomes, refused tool calls, slips, VRAM) |
| `agent/bot.py` | Observer added; Kokoro now speaks a fixed line **one sentence at a time** (the biggest win of this step) |
| `agent/config.py` | `LLM` and `VAD_STOP_SECS` can be set from PowerShell, so a comparison needs no code change |

## 1. See the delay of each reply
Make a call as in step 4. After every bot reply the log prints where the time went:
```
Reply latency:
   0.200s  endpointing wait     [config: VAD stop_secs]       ← silence the VAD waits for
   0.343s  transcription        [SharedWhisperSTT#0]          ← Whisper, after the caller stopped
   0.052s  turn detection       [config: user turn strategies] ← Smart Turn: is the thought finished?
   0.309s  LLM inference        [OLLamaLLMService#0]          ← until Qwen's first words
   0.038s  sentence aggregation [config: text_aggregation_mode] ← until the first full sentence
   0.510s  speech synthesis     [QuietKokoroTTS#0]            ← Kokoro, until the first audio
   0.004s  pipeline             [pipecat]                     ← frames moving between the parts
   1.456s  TOTAL
```
The parts add up to the total. The clock starts when the caller **actually** went quiet (VAD stop time minus
`stop_secs`) and stops when the bot's first audio leaves for the browser. Not included: the network and the
browser's playback buffer. Locally that is a few ms; on a phone line add 100–300 ms.

A reply after a tool call has `LLM tool call` and `function handler` lines instead of the sentence ones. A reply
spoken by code (the amount due, the promise read-back) shows its synthesis as `pipeline`: Pipecat can't name it.

## 2. The report
```powershell
uv run python -m scripts.latency_report
```
It reads every `metrics\*.jsonl`. Each column is a setup (LLM, VAD wait, Whisper model, from the first line of each
file), so runs with different settings never mix. The cells are `p50 / p95` in ms; `(24x)` means the stage
happened in 24 of the replies. The greeting is left out: it is timed from the browser connecting, not from silence.

To keep old runs as their own column, move them to a subfolder and name it in the command:
```powershell
mkdir metrics\before-split; move metrics\*.jsonl metrics\before-split\
uv run python -m scripts.latency_report "metrics\before-split\*.jsonl" "metrics\*.jsonl"
```

## 3. Compare a setting
```powershell
ollama stop qwen2.5:7b                 # only the model under test in VRAM
$env:LLM="qwen2.5:3b"                  # or: $env:VAD_STOP_SECS="0.4"
uv run python -m agent.bot             # make the calls, then run the report
Remove-Item Env:LLM                    # back to qwen2.5:7b
```

## 4. Results (20 Oct 2026, RTX 3070 Ti Laptop)
Every column is 12 calls: 4 scripted scenarios × 3. The caller is a script that speaks over WebRTC with a second
Kokoro voice and waits for the bot to finish (it becomes the step 6 test bench). Same audio for every setup.

**Voice to voice, ms (p50 / p95):**

| | 7B, VAD 0.2, before the fix | **7B, VAD 0.2** | 3B, VAD 0.2 | 7B, VAD 0.3 | 7B, VAD 0.4 |
|---|---|---|---|---|---|
| all replies | 2406 / 4573 | **1831 / 3093** | 2123 / 4007 | 2012 / 3314 | 2237 / 4192 |
| plain replies | 1697 / 2474 | **1585 / 2542** | 1560 / 4331 | 1633 / 3266 | 2011 / 3133 |
| after a tool call | 2884 / 4671 | **2315 / 3186** | 2398 / 3713 | 2544 / 4453 | 2916 / 4857 |

**Where the time goes** (7B, VAD 0.2, p50 ms): speech synthesis 914 · transcription 362 · LLM 325 (plain reply 236,
tool call 787) · endpointing 200 · sentence aggregation 116 · turn detection 9.

**3B vs 7B** (same 12 calls):

| | 7B | 3B |
|---|---|---|
| Voice to voice p50 / p95 | **1831 / 3093 ms** | 2123 / 4007 ms |
| Calls that ended right | **10 / 12** | 7 / 12 |
| Tool calls refused | 9 | 18 |
| GPU memory, whole card | 5.4 GB | **2.9 GB** |
| Words per LLM reply | 19 | 26 |

**VAD stop_secs** (7B):

| stop_secs | Caller lines cut in two | Mean reply, plain / after a tool | Calls right |
|---|---|---|---|
| **0.2** (kept) | 23 of 53 | 1786 / 2334 ms | 10 / 12 |
| 0.3 | 15 of 56 | 1935 / 2575 ms | 10 / 12 |
| 0.4 | 8 of 57 | 2096 / 2887 ms | 12 / 12 |

What the numbers say (the why is in the study note):
- **Sentence by sentence** took p50 from 2.4 s to 1.8 s and p95 from 4.6 s to 3.1 s. Kokoro had been making the
  whole 146-character amount-due line before the first sound: 3 s, once 18 s.
- **The 3B is slower**, not faster: its replies are longer and Kokoro's time grows with the sentence. It also
  sent `"day: "` instead of `"day"` in all 3 wrong-person calls, so no callback was booked.
- **VAD 0.2 stays.** Longer waits stop the cuts (every date of birth splits at the comma: "5th of March." | "1990.")
  but cost every reply. The cut halves are joined before the LLM answers (`JoinCallerTurns`, step 4) and no call
  failed because of a cut.
- Medians from about 27 replies move by 0.1–0.2 s between runs on their own. Differences smaller than that are noise.

## 5. Step 5b: Kokoro on the GPU
Kokoro was the largest part of each reply. It now runs on the GPU through `onnxruntime-gpu`, which replaces the
CPU package (`pyproject.toml`, `[tool.uv] override-dependencies`). Silero VAD and Smart Turn stay on the CPU.

Check where it runs:
```powershell
uv run python -m scripts.check_env      # "Kokoro voice: ... made in 155 ms on CUDAExecutionProvider"
uv run python -m agent.bot              # log: "Kokoro loaded on CUDAExecutionProvider and warm in 2.4s"
```
Back to the CPU, for a comparison: `$env:TTS_DEVICE = "cpu"`. The report adds `tts cpu` / `tts cuda` to each column.

**Same 12 calls, 7B, VAD 0.2, after the amount fix (ms, p50 / p95):**

| | Kokoro on CPU | **Kokoro on GPU** |
|---|---|---|
| speech synthesis | 699 / 1484 | **331 / 479** |
| **voice to voice** | 1808 / 2770 | **1304 / 2445** |
| plain replies | 1521 / 2380 | **1242 / 1612** |
| after a tool call | 2249 / 3065 | **1800 / 2516** |
| calls right | 10 / 12 | 12 / 12 |
| GPU memory, whole card | 5.4 GB | 6.5 GB |

- Each sentence is 4–6× faster alone (585 → 158 ms for 25 characters, 2648 → 433 ms for 106).
- The p50 dropped by half a second. The p95 dropped less, because the slowest replies are now the ones that wait
  for a tool call to be written: LLM time, not voice.
- 12/12 is the LLM's luck, not the GPU's: the 2 hardship calls that failed before were LLM choices. Step 6 checks it.
- Kokoro takes about 1 GB of VRAM. About 1.5 GB of the 8 GB is still free.

## When it breaks
| You see | Why | Fix |
|---|---|---|
| `Kokoro voice: ... on CPUExecutionProvider` (FAIL) | onnxruntime couldn't start CUDA and quietly used the CPU | `uv sync` (needs `onnxruntime-gpu`, not `onnxruntime`); `uv pip list` must not show plain `onnxruntime` |
| `Failed to load ...cublas64_12.dll: [WinError 127]` | Whisper was loaded before Kokoro | Keep Kokoro first in `agent/bot.py` (it is); harmless either way |
| `No calls measured yet in ...\metrics` | No call finished with the new code | Make a call; one file per call appears in `metrics\` |
| A column you didn't expect | A setting was still set in PowerShell | `Get-ChildItem Env:LLM, Env:VAD_STOP_SECS`; remove with `Remove-Item Env:...` |
| 3B numbers look like 7B, or VRAM is 7+ GB | Both models loaded | `ollama ps`; `ollama stop qwen2.5:7b` before the 3B run |
| One reply of 5+ s with `pipeline` large | Kokoro stalled (on the CPU: another program busy) | Look at `QuietKokoroTTS#0 TTFB` in the log; close heavy programs and repeat the run |
