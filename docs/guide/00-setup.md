# Step 0: check the setup, then try each model alone

Every piece of the agent can be run on its own. Do them in this order: the voice (Kokoro) makes the
WAV files that the other checks listen to.

| # | Piece | Runs on | Command type |
|---|---|---|---|
| 1 | Everything at once | — | `uv run python -m scripts.check_env` |
| 2 | GPU | — | `nvidia-smi` |
| 3 | Brain: Ollama + Qwen | GPU | `ollama` CLI, HTTP API |
| 4 | Voice: Kokoro | CPU | Python snippet → WAV |
| 5 | Hearing: Whisper | GPU | Python snippet on a WAV |
| 6 | "Is someone speaking?": Silero VAD | CPU | Python snippet on a WAV |
| 7 | "Have they finished?": Smart Turn | CPU | Python snippet on two WAVs |

```powershell
cd "D:\ResumeProject\voice-agent"
```

## 1. Everything at once
```powershell
uv run python -m scripts.check_env
```
Expected: every line `OK`, then `All green.`

```
OK    GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB, 610.62
OK    CUDA libraries: cuBLAS 12 and cuDNN 9 found in the venv
OK    Whisper on GPU: Systran/faster-distil-whisper-medium.en loaded in 2.9s, 1 s of audio in 662 ms, ~699 MB VRAM
OK    Ollama: running at http://127.0.0.1:11434
OK    model qwen2.5:7b: pulled
OK    model qwen2.5:3b: pulled
OK    Kokoro voice: voice 'af_heart': 1.6s of speech made in 1022 ms on CPU
OK    Silero VAD + Smart Turn: both models load on CPU
All green.
```
Exit code is 0 when green and 1 when anything fails (`$LASTEXITCODE` shows it), so scripts can rely on it.

## 2. GPU
```powershell
nvidia-smi                      # one snapshot: driver, VRAM used, which processes use it
nvidia-smi -l 1                 # refresh every second; Ctrl+C to stop
```
**Try:** keep `nvidia-smi -l 1` open in a second window while you run sections 3 and 5. You will see
VRAM jump by about 4.7 GB when Qwen 7B loads and about 0.7 GB for Whisper. Our budget is 8 GB.

## 3. Brain: Ollama + Qwen
```powershell
ollama list                     # models on disk
ollama ps                       # models loaded in VRAM right now, and when they unload
ollama run qwen2.5:7b --verbose "In one sentence: why should a voice agent keep replies short?"
ollama stop qwen2.5:7b          # unload it from VRAM now
```
`--verbose` prints timings after the reply. Real output from a cold start:
```
total duration:       5.2271858s
load duration:        4.7247272s     <- model copied into VRAM (only when it wasn't loaded)
prompt eval rate:     260.75 tokens/s  <- reading the prompt
eval count:           22 token(s)
eval rate:            66.88 tokens/s   <- writing the reply
```
And `ollama ps` straight after:
```
NAME          ID              SIZE      PROCESSOR    CONTEXT    UNTIL
qwen2.5:7b    845dbda0ea48    4.7 GB    100% GPU     4096       4 minutes from now
```
**What to notice**
- **Load 4.7 s, reply 0.33 s.** A cold model is the single biggest delay in the whole system.
- `UNTIL 4 minutes from now`: Ollama unloads a model after 5 idle minutes. So a live agent must warm
  the LLM at startup *and* keep it loaded (`keep_alive`), not only warm Whisper.
- `100% GPU`: if this ever says `CPU` or a split like `30%/70% CPU/GPU`, VRAM ran out and it will be
  much slower.

The same thing through the HTTP API (`try_llm` in step 1 uses its `/api/chat` sibling):
```powershell
$body = @{ model = "qwen2.5:7b"; prompt = "Say hello in five words."; stream = $false } | ConvertTo-Json
$r = Invoke-RestMethod -Uri http://127.0.0.1:11434/api/generate -Method Post -Body $body -ContentType "application/json"
"$($r.response)  | load $([math]::Round($r.load_duration/1e6)) ms, $($r.eval_count) tokens in $([math]::Round($r.eval_duration/1e6)) ms"
```
Real output (model already warm): `Hello, nice to meet you.  | load 204 ms, 8 tokens in 106 ms`

**Try**
- `ollama run qwen2.5:3b --verbose "..."` with the same prompt. Compare `eval rate` and the answer.
- Add `keep_alive = -1` to `$body` (never unload), then check `ollama ps`. Use `ollama stop` to undo.

## 4. Voice: Kokoro → WAV files
Makes three WAVs in `recordings/`: one long line, plus a finished and an unfinished sentence for
section 7.
```powershell
@'
import wave
from pathlib import Path
import numpy as np
from kokoro_onnx import Kokoro
from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR
from agent import config

kokoro = Kokoro(str(KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"), str(KOKORO_CACHE_DIR / "voices-v1.0.bin"))
Path("recordings").mkdir(exist_ok=True)
lines = {
    "hello": "Hi, this is Asha from Greenfield Bank, calling about your loan payment.",
    "complete": "I can pay on Friday.",
    "cut_off": "I can pay on, um,",
}
for name, text in lines.items():
    samples, rate = kokoro.create(text, voice=config.TTS_VOICE, lang="en-us")
    with wave.open(f"recordings/{name}.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"recordings/{name}.wav  {len(samples) / rate:.1f}s  {text!r}")
'@ | uv run python -
```
Real output:
```
recordings/hello.wav  4.0s  'Hi, this is Asha from Greenfield Bank, calling about your loan payment.'
recordings/complete.wav  1.4s  'I can pay on Friday.'
recordings/cut_off.wav  1.4s  'I can pay on, um,'
```
Listen: `Invoke-Item recordings\hello.wav`

Kokoro returns float samples between -1 and 1 at 24 kHz. A WAV file stores 16-bit integers, hence the
`* 32767`.

**Try**
- Another voice: change `voice=config.TTS_VOICE` to `voice="am_michael"` (male, US) or `"bf_emma"`
  (female, UK). There are 54; the English ones start with `af_`, `am_`, `bf_`, `bm_`. To list them:
  add `print(kokoro.get_voices())`.
- Time it: wrap `kokoro.create(...)` in `time.perf_counter()` and compare speech length with how long it
  took to make. Step 1 does this properly.

## 5. Hearing: Whisper on the GPU
```powershell
@'
import os, time
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from agent.gpu import enable_cuda_dlls
enable_cuda_dlls()                      # must run before faster_whisper loads the GPU libraries
from faster_whisper import WhisperModel
from agent import config

model = WhisperModel(config.STT_MODEL, device=config.STT_DEVICE, compute_type=config.STT_COMPUTE_TYPE)
for path in ["recordings/hello.wav", "recordings/hello.wav"]:   # twice: cold, then warm
    t0 = time.perf_counter()
    segments, info = model.transcribe(path, beam_size=1, language="en")
    text = " ".join(s.text.strip() for s in segments)
    print(f"{(time.perf_counter() - t0) * 1000:.0f} ms  {info.duration:.1f}s audio  ->  {text}")
'@ | uv run python -
```
Real output:
```
1394 ms  4.0s audio  ->  Hi, this is a shaft from Greenfield Bank, calling about your loan payment.
180 ms  4.0s audio  ->  Hi, this is a shaft from Greenfield Bank, calling about your loan payment.
```
**What to notice**
- **Cold 1,394 ms, warm 180 ms**: same file, same model. Warm-up again.
- **"Asha" came out as "a shaft".** Setting `hotwords="Asha"`, `initial_prompt="Asha from Greenfield Bank."` or
  `beam_size=5` did not fix it. Names are a known weak spot for speech-to-text, and this test has two
  models that could be at fault. See the exercise below.

**Your own voice:** record with the Windows *Sound Recorder* app. It saves `.m4a`, which is fine:
faster-whisper decodes audio with PyAV, which has ffmpeg built in. Put the path in place of
`recordings/hello.wav`. Sound Recorder saves to `Documents\Sound Recordings` by default.

**Try**
- Replace `config.STT_MODEL` in the snippet with `"small.en"` and compare speed and accuracy (or use
  `try_stt --model small.en` from step 1).
- Exercise: is the "a shaft" mistake Whisper's fault or Kokoro's? Listen to `hello.wav`, then record
  yourself saying the sentence and transcribe that. Tell me what you find.

## 6. Silero VAD: "is someone speaking?"
Checks every 32 ms chunk and prints `#` for speech, `.` for no speech. It uses the same confidence
threshold Pipecat uses in a live call (0.7). Pipecat also checks loudness, which this snippet skips.
```powershell
@'
import numpy as np
from faster_whisper import decode_audio
from pipecat.audio.vad.silero import SileroVADAnalyzer

audio = decode_audio("recordings/hello.wav", sampling_rate=16000)      # float32, 16 kHz, mono
silence = np.zeros(8000, dtype=np.float32)                             # 0.5 s each side
pcm = (np.concatenate([silence, audio, silence]) * 32767).astype(np.int16).tobytes()

vad = SileroVADAnalyzer()
vad.set_sample_rate(16000)
chunk = vad.num_frames_required() * 2      # 512 samples (32 ms), 2 bytes each
bar = ""
for i in range(0, len(pcm) - chunk, chunk):
    bar += "#" if vad.voice_confidence(pcm[i:i + chunk]) > vad.params.confidence else "."
print(bar)
print("one character = 32 ms;  # = speech, . = no speech")
'@ | uv run python -
```
Real output:
```
...................######################################################..#######.##............######################################..##................
```
**What to notice:** the run of `............` in the middle is the comma after "Bank". It lasts about
12 × 32 ms = **384 ms**. Our VAD stop time is 0.2 s, so **VAD alone would have ended the turn at the
comma**. That is why Smart Turn exists (section 7).

**Try:** change `vad.params.confidence` to `0.5` and run it again. The gap shrinks to 10 chunks
(320 ms). A lower threshold counts more sound as speech.

The log line `confidence=0.7 start_secs=0.2 stop_secs=0.2 min_volume=0.6` shows the settings
Pipecat uses in a live call: speech must last 0.2 s to count as "started" and be silent for 0.2 s to
count as "stopped".

## 7. Smart Turn: "have they finished their thought?"
Runs the end-of-turn model on the two short WAVs from section 4.
```powershell
@'
from faster_whisper import decode_audio
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

turn = LocalSmartTurnAnalyzerV3()
turn.set_sample_rate(16000)
for name in ["complete", "cut_off"]:
    audio = decode_audio(f"recordings/{name}.wav", sampling_rate=16000)
    result = turn._predict_endpoint(audio)   # private method: the same call Pipecat makes after each pause
    verdict = "finished" if result["prediction"] == 1 else "not finished"
    print(f"{name:9}  {verdict:13}  p(finished) = {result['probability']:.2f}")
'@ | uv run python -
```
Real output:
```
complete   finished       p(finished) = 0.99
cut_off    not finished   p(finished) = 0.01
```
**What to notice:** both clips are 1.4 s long and end in silence, so VAD can't tell them apart. Smart
Turn can, from the *sound* alone (intonation and pacing in the last 8 s). It never sees any text. It
takes about **78 ms** on the CPU (median of 10 warm runs).

In a live call, VAD reporting 0.2 s of silence starts **Smart Turn and Whisper at the same time**.
Pipecat waits for both, the "finished" verdict and the transcript, before it calls the LLM. If Smart
Turn says "not finished", Pipecat keeps listening and keeps the text so far. After 3 s of silence it
ends the turn anyway (`SmartTurnParams.stop_secs = 3`).

**Try:** record yourself saying "My account number is, uh…" and trailing off, then point the snippet
at that file. Synthetic speech is easy; your voice is the real test.

## Numbers measured today (warm unless marked)
| Piece | Result |
|---|---|
| Qwen 7B load (cold) | 4.7 s |
| Qwen 7B reply | 67 tokens/s, 4.7 GB VRAM |
| Whisper, 4 s of audio | 1,394 ms cold → 180 ms warm, ~0.7 GB VRAM |
| Kokoro, 3.5 s sentence (CPU) | ~1,300 ms |
| Smart Turn | 0.99 on a finished sentence, 0.01 on a cut-off one; ~78 ms per check |

## Where the model files live
| What | Where | Size |
|---|---|---|
| Whisper | `%USERPROFILE%\.cache\huggingface\hub\models--Systran--faster-distil-whisper-medium.en` | 755 MB |
| Kokoro | `%USERPROFILE%\.cache\pipecat\kokoro-onnx\` (`kokoro-v1.0.onnx`, `voices-v1.0.bin`) | 337 MB |
| Qwen | `%USERPROFILE%\.ollama\models` (`ollama rm <name>` to delete) | 4.7 GB / 1.9 GB |
| Silero VAD, Smart Turn | inside the Pipecat package in `.venv` | small |

Delete a cache folder and the next run downloads it again.

## When something fails
| Symptom | Fix |
|---|---|
| `FAIL Ollama: not reachable` | Start the Ollama app; check with `ollama list` |
| `model ... not pulled` | `ollama pull qwen2.5:7b` |
| `Could not load library cudnn...` or `cublas64_12.dll not found` | `enable_cuda_dlls()` must run before any `WhisperModel` is created (simplest: before importing `faster_whisper`); if the DLLs are missing: `uv add nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` |
| `CUDA out of memory` | `ollama stop qwen2.5:7b` to free VRAM, or use `small.en` / the 3B model |
| Paste does nothing / `>>` prompt waits | The closing `'@` isn't at the start of its line. Press Ctrl+C and paste again |
| `ModuleNotFoundError: agent` | You're not in the project root. `cd "D:\ResumeProject\voice-agent"` |
| Slow first run | Expected: downloads and cold start. Run it a second time before judging speed |
