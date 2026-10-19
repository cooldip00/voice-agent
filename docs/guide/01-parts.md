# Step 1: each part alone, timed

Three scripts, one per part of a call. Each loads its model, does one untimed warm-up run, then
times 3 runs and prints the median. Run them in this order (`try_stt` listens to what `try_tts` made).

| Script | Part | What it times |
|---|---|---|
| `scripts/try_tts.py` | Voice (Kokoro, CPU) | Each sentence of a reply, and when the first one is ready |
| `scripts/try_stt.py` | Hearing (Whisper, GPU) | Transcription at beam size 5 and 1, plus word error rate |
| `scripts/try_llm.py` | Brain (Qwen on Ollama, GPU) | Time to first token, time to first sentence, tokens/s |

```powershell
cd "D:\ResumeProject\voice-agent"
```

> **Run one at a time.** Whisper and Qwen share the GPU. Running two scripts at once made Qwen's
> numbers wrong in testing.

## 1. Voice: `try_tts`
```powershell
uv run python -m scripts.try_tts
uv run python -m scripts.try_tts --voice am_michael
uv run python -m scripts.try_tts --text "Your EMI of Rs. 4,500 is due on 05/11."
uv run python -m scripts.try_tts --runs 5
```
Real output:
```
Kokoro loaded in 1.3s (CPU, ONNX), voice 'af_heart'

warm-up (cold first sentence): 1126 ms

sentence                                                      speech  made in  x real time
Thank you for confirming, Mr. Sharma.                           2.2s    795ms         2.8x
Your loan payment of four thousand five hundred rupees ...      4.3s   1383ms         3.1x
Would you be able to pay it by this Friday?                     2.1s    737ms         2.8x

first audio ready after: 795 ms  (median of 3)
whole reply: 8.6s of speech made in 2915 ms
```
Saves `recordings/tts.wav` (listen with `Invoke-Item recordings\tts.wav`) and `recordings/tts.txt` (the text).

**How to read it**
- `x real time` = seconds of speech ÷ seconds to make it. Anything above 1× can keep up with playback.
- **Only `first audio ready after` makes the caller wait.** Sentence 2 is made while sentence 1 plays.
- Time to make a sentence grows with its length (about 0.35 s per second of speech). **A short first
  sentence means a faster first sound.** We'll use that in the step 4 prompt.
- Sentences are split with Pipecat's own `match_endofsentence`, so "Mr. Sharma" stays in one piece.

**Try**
- The `--text` with "Rs. 4,500" and "05/11" above. Listen: how does it read them? Measured: a short line
  became **6.0 s of speech**, and `try_stt` heard it back as *"Your EMI of RS, 4,500 is due on 0 5
  slash 11."* A real agent must write amounts and dates out as words before speaking ("four thousand
  five hundred rupees", "the fifth of November"). This is text normalization, and it goes in step 4.
  This overwrites `recordings/tts.wav`; run plain `try_tts` again before `try_stt`.
- `--voice am_michael` / `bf_emma`: is speed the same for every voice?

## 2. Hearing: `try_stt`
```powershell
uv run python -m scripts.try_stt                                    # recordings/tts.wav from try_tts
uv run python -m scripts.try_stt "$env:USERPROFILE\Documents\Sound Recordings\Recording.m4a"
uv run python -m scripts.try_stt --model small.en
uv run python -m scripts.try_stt --device cpu                       # how slow without the GPU?
```
Real output:
```
Systran/faster-distil-whisper-medium.en loaded in 3.1s on cuda (int8_float16)
audio: tts.wav, 8.6s

warm-up (cold): 1537 ms

beam_size=5: 295 ms, 29x real time  (Pipecat default)
  heard:    Thank you for confirming Mr. Sharma, your loan payment of 4,500 rupees was due on the fifth. Would you be able to pay it by this Friday?
  expected: Thank you for confirming, Mr. Sharma. Your loan payment of four thousand five hundred rupees was due on the fifth. Would you be able to pay it by this Friday?
  WER:      13%

beam_size=1: 178 ms, 48x real time
  ...same text, WER 13%
```
**How to read it**
- **WER** (word error rate) = (words swapped + missed + added) ÷ words expected. It only prints when
  a `.txt` with the same name sits next to the audio. For your own recording, create one, for example
  `Recording.txt` holding exactly what you said.
- **The 13% isn't a hearing mistake.** Whisper wrote "4,500" where the text said "four thousand five
  hundred". That counts as 4 wrong words out of 30. Before the step 6 test bench uses WER, both sides
  need numbers written the same way.
- **Beam size**: 5 tries several candidate transcripts and keeps the best; 1 takes the single most
  likely word each time. Same text here, 40% faster. Clean synthetic audio is easy, so test your own
  voice before choosing.

**Try**
- Record yourself (Windows Sound Recorder) reading the default sentence, write it into a `.txt` next to
  the recording, and compare WER at beam 5 and beam 1.
- `--device cpu`: this is the number to quote when someone asks "why a GPU?". Measured on the same
  8.6 s clip at beam 5: **4,625 ms on the CPU vs 295 ms on the GPU**, about 16× slower. On the CPU, the
  caller would wait over 4 s just for the bot to *hear* them.

## 3. Brain: `try_llm`
```powershell
uv run python -m scripts.try_llm
uv run python -m scripts.try_llm --model qwen2.5:3b
uv run python -m scripts.try_llm --say "I lost my job, I can't pay this month."
uv run python -m scripts.try_llm --say "Who is this? How did you get my number?"
```
Real output (7B, with the 3B unloaded first: `ollama stop qwen2.5:3b`):
```
model: qwen2.5:7b
caller: 'Oh, sorry, I forgot about it. Can I pay next week?'

warm-up: first token after 199 ms (model was already in VRAM)

run 1: I understand, but to avoid any late fees, can you make the payment as soon as possible this week?
run 2: I understand, but to avoid any fees, can we set up the payment as soon as possible this week?
run 3: I understand, but to avoid any fees, would you like to make the payment right away? We can set up an auto-debit to prevent future issues.

median of 3 warm runs
  time to first token:       191 ms
  time to first sentence:    473 ms   <- Kokoro can start here
  whole reply:               506 ms   (23 tokens)
  speed:                      75 tokens/s  (speech is about 3-4 tokens/s)
```
**How to read it**
- **Time to first token**: the model has read the prompt and starts answering.
- **Time to first sentence**: the first complete sentence (Pipecat's `match_endofsentence`) is ready.
  This is when the voice can start, so it's the LLM number that matters most for a voice agent.
- **Tokens/s** only has to stay well above speaking speed (~3–4 tokens/s). 75 or 122 makes no
  difference to the caller.
- `model load ... ms` on the warm-up line means the model wasn't in VRAM. The script asks Ollama to
  keep it loaded for 10 minutes (`keep_alive`).

**7B vs 3B, same prompt (measured)**
| | qwen2.5:7b | qwen2.5:3b |
|---|---|---|
| Load from cold | 4.9 s | 2.8 s |
| Time to first token | 191 ms | 241 ms |
| Time to first sentence | 473 ms | 378 ms |
| Speed | 75 tokens/s | 122 tokens/s |
| VRAM | 4.7 GB | 2.2 GB |
| Behaviour | Asks for payment this week | "Sure, you can pay next week" — gives in, no date or amount |

Both invent things the bank never said ("late fees", "auto-debit", "interest"). Step 3's tools and
step 4's prompt exist to stop that.

**Try**
- The two `--say` lines above. Does either model reveal or invent loan details? (They have none to
  reveal yet, so anything specific is made up.)
- `ollama ps` while a model is loaded; then `ollama stop qwen2.5:7b` and run `try_llm` again to see
  the cold load in the warm-up line.

## Putting the numbers together
What the caller would wait between their last word and the bot's first sound, measured warm:

| Stage | Time | From |
|---|---|---|
| VAD: notice silence | 200 ms | `VAD_STOP_SECS` in config |
| Smart Turn: decide "finished" | (~78 ms) | runs **at the same time** as Whisper and is faster, so it adds nothing |
| Whisper: transcribe the caller | ~180–300 ms | `try_stt` (8.6 s clip; a short reply is faster) |
| Qwen 7B: first sentence | ~470 ms | `try_llm` |
| Kokoro: first sentence | ~800 ms | `try_tts` |
| **Total** | **~1.7 s** | browser and audio buffers not included yet |

Smart Turn and Whisper both start when VAD reports silence; Pipecat waits for both, then calls the LLM.

Under about 1 s feels natural on a phone call; 2 s feels slow. **Kokoro on the CPU is the biggest
single piece.** Step 5 measures the real total in live calls and attacks the largest stages first.

## When something fails
| Symptom | Fix |
|---|---|
| `recordings\tts.wav not found` | Run `try_tts` first, or pass a file: `try_stt "path\to\file.wav"` |
| `Ollama not reachable` | Start the Ollama app |
| First token ~2 s even when warm | `OLLAMA_BASE` must be `127.0.0.1`, not `localhost` (Windows tries IPv6 first: measured 2,049 ms vs 3 ms per request) |
| LLM numbers jump around | Another script is using the GPU, or two models are loaded (`ollama ps`; `ollama stop <name>`) |
| `Voice xx not found` | Use a name from the 54 voices (see step 0 guide, section 4) |
| `--device cpu` very slow | Expected. That's the point of the test |
