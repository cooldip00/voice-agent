# Step 2: the talking bot in the browser

One command starts a local web server. You open the page, allow the mic, and talk to the bot:
browser mic → Silero VAD + Smart Turn → Whisper → Qwen 7B → Kokoro → browser speaker.
Code: [agent/bot.py](../../agent/bot.py), settings in [agent/config.py](../../agent/config.py).

## 1. Start it
```powershell
cd "D:\ResumeProject\voice-agent"
uv run python -m agent.bot
```
Real output (trimmed; it takes about 11 s before the server accepts a call):
```
INFO | load_whisper  - Whisper loaded and warm in 3.7s
INFO | load_kokoro   - Kokoro loaded and warm in 2.0s
INFO | warm_llm      - qwen2.5:7b loaded in Ollama in 5.5s
INFO:     Uvicorn running on http://localhost:7860 (Press CTRL+C to quit)
```
The three models load **once**, before the server starts. Every call reuses them, so the first
call is as fast as the tenth.

## 2. Call it
1. Open **http://localhost:7860** (it redirects to `/client`, Pipecat's built-in test page).
2. Click **Connect** and allow the microphone.
3. The bot says *"Hi, this is Asha. How can I help you today?"*. Talk back.
4. **Use headphones.** With laptop speakers, the mic can pick up the bot's voice, and the bot may
   take that as you interrupting it.
5. Stop the server with `Ctrl+C`.

The bot is a placeholder assistant for now. It knows nothing about loans; that comes in steps 3–4.

## 3. Reading the log
Each turn prints a burst of lines. The ones that matter, in order (real lines from a test call):

| Line | Meaning |
|---|---|
| `User started speaking (strategy: VADUserTurnStartStrategy)` | Silero VAD heard speech |
| `broadcasting interruption` | Printed every time you start talking. If the bot was talking or thinking, it stops |
| `Transcription: [ Would you be able to pay it by this Friday? ]` | What Whisper heard |
| `SharedWhisperSTT TTFB: 0.583s` | From VAD "silence" to the transcript |
| `End of Turn result: EndOfTurnState.COMPLETE` | Smart Turn thinks you have finished. `INCOMPLETE` means it keeps listening |
| `User stopped speaking (strategy: TurnAnalyzerUserTurnStopStrategy)` | The turn is over; the LLM is called now |
| `Generating chat from context [...]` | The whole conversation sent to Qwen |
| `OLLamaLLMService TTFB: 0.352s` | Qwen's first token |
| `Generating TTS [You're welcome!]` | One sentence handed to Kokoro |
| `KokoroTTSService TTFB: 0.464s` | Kokoro's first audio for that sentence |

**TTFB** (time to first byte) is the time a stage takes to produce its first output. **Processing
time** is the time until it finished. The caller waits for the TTFBs; processing overlaps with playback.

## 4. What a scripted test call showed
Before handing this over, I tested it with a fake caller: a script that connects over WebRTC like a
browser does, plays a WAV as its "mic" and records what the bot says back. The WAV was the step 1
Kokoro line, *"Thank you for confirming, Mr. Sharma. Your loan payment of four thousand five hundred
rupees was due on the fifth. Would you be able to pay it by this Friday?"*

| Measure | Result |
|---|---|
| Greeting, Kokoro TTFB | 0.905 s (first sentence of the call) |
| Whisper TTFB / processing | ~0.58 s / 0.19–0.42 s per piece |
| Qwen TTFB (final turn) | 0.352 s |
| Kokoro TTFB, "You're welcome!" | 0.464 s (a short first sentence → a fast first sound) |
| **Voice to voice** | **~1.7 s** from the caller's last word to the bot's first audio (server side; browser buffers add a little) |
| Talk-over | none: the recording has the bot speaking only before and after the caller |

**Surprise: Smart Turn ended the turn 4 times inside one utterance.** The synthetic voice drops its
pitch at commas, and Smart Turn took each drop as "finished":
```
Transcription: [ Thank you for confirming Mr. ]                     -> COMPLETE, LLM called
Transcription: [ Sharma, your loan payment of 4,500 rupees. ]       -> COMPLETE, LLM called
Transcription: [ was due on the fifth. ]                            -> COMPLETE, LLM called
Transcription: [ Would you be able to pay it by this Friday? ]      -> COMPLETE, LLM called
```
Each time, the "caller" kept talking within 0.05–0.5 s, VAD fired `broadcasting interruption`,
and the half-started reply was thrown away before any audio played. The final context still had
all four pieces, and the bot answered the whole thing. So: wasted LLM calls, nothing audible. A real
voice may behave differently; check it with your own (section 5).

## 4b. What a real browser call showed (laptop speakers, no headphones)
The test page's **karaoke** view shows each bot sentence white once spoken and grey if it was cut off.
Comparing it with the next `Generating chat from context [...]` log line:

- **The context keeps only whole sentences that finished playing.** The bot said *"Great! ~~What can
  I help you with?~~"* and the context stored `'Great!'`. A sentence cut off halfway is dropped
  entirely, even though the caller heard part of it.
- **The bot heard itself.** Three "user" turns were its own words picked up by the mic:
  *"How can I assist you today?"* came back as *"I'll assist you today."*, *"…just let me know."* as
  *"Just let me know."*. One echo cut off the bot's own reply. The browser's echo cancellation wasn't
  enough with laptop speakers. Headphones fix it.

## 5. Try
- **Interrupt it.** Ask something with a long answer ("Tell me about the history of India"), then
  start talking while it speaks. It should stop within a fraction of a second, and the log shows
  `broadcasting interruption`.
- **Pause mid-sentence.** Say "I can pay… [2 s pause] …on Friday." Does it jump in during the pause?
  Look for `INCOMPLETE` vs `COMPLETE` in the log.
- **Count early ends with your own voice.** Read the Mr. Sharma line above at a normal pace. How many
  `Transcription:` lines does one sentence become?
- **The 3B model.** Set `LLM_MODEL = "qwen2.5:3b"` in `agent/config.py`, restart, and compare the
  `OLLamaLLMService TTFB` lines and the answers.
- **A slower VAD.** Set `VAD_STOP_SECS = 0.5`. Each reply starts ~0.3 s later. Do the early ends
  go away? Put it back to 0.2 afterwards.

## When something fails
| Symptom | Fix |
|---|---|
| `ConnectionError` at `warm_llm` on startup | Start the Ollama app, then run again |
| `error while attempting to bind on address ... 7860` | Another bot is still running. Close it, or find it: `Get-NetTCPConnection -LocalPort 7860` then `Stop-Process -Id <OwningProcess>`. Or use `--port 7861` |
| Page connects but the bot never hears you | Mic blocked: click the icon left of the address bar → allow Microphone, reload |
| The bot keeps stopping itself mid-sentence | It hears its own voice through the speakers. Use headphones |
| First reply after a long idle takes ~5 s | Ollama unloaded Qwen. `ollama ps` shows how long it stays (`UNTIL`). Requests through Pipecat may reset it to Ollama's default 5 minutes (not verified yet); step 5 fixes it with the `OLLAMA_KEEP_ALIVE` setting |
| Call ends by itself after 5 minutes of silence | Pipecat's idle timeout (300 s). Reconnect |
| `Data channel not ready, queuing message` | Harmless. Shown before the page's data channel opens (and always with the test script, which has none) |
