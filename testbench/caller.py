"""The test caller: phones the bot over WebRTC like the browser does, listens, and answers by the scenario.

After each bot turn (the bot spoke, then 2 s of quiet) it writes down what it heard with Whisper tiny.en and
picks its answer with Scenario.reply_to. Its voice is Kokoro's am_michael, not the bot's af_heart. Both of
its models run on the CPU, so the bot keeps the GPU to itself and its timings stay comparable.

It also times the two things the call record can't show:
- interrupt: the caller starts talking 1.5 s into a bot turn; how long until the bot's audio stops.
- silence: the caller says nothing for N s; how long until the bot speaks up.
"""

import asyncio
import bisect
import fractions
import time
import wave
from pathlib import Path

import aiohttp
import av
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from testbench.scenario import Scenario

RATE = 48000
FRAME = 960            # 20 ms
LOUD_RMS = 300         # int16; the bot's silence is digital zero, its speech is in the thousands
TURN_QUIET_SECS = 2.0  # the bot's turn is over after this much quiet
STOPPED_QUIET_SECS = 0.7
INTERRUPT_AFTER_SECS = 1.5


class Voice:
    """The caller's lines as 48 kHz int16 audio, made once per text."""

    def __init__(self):
        from kokoro_onnx import Kokoro
        from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR

        from agent.gpu import onnx_session

        session = onnx_session(KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", "cpu")
        self.kokoro = Kokoro.from_session(session, str(KOKORO_CACHE_DIR / "voices-v1.0.bin"))
        self.cache: dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        if text not in self.cache:
            samples, sr = self.kokoro.create(text, voice="am_michael", lang="en-us")
            t_new = np.arange(int(len(samples) * RATE / sr)) / RATE
            pcm = np.interp(t_new, np.arange(len(samples)) / sr, samples)
            self.cache[text] = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
        return self.cache[text]


class Hearing:
    """Whisper tiny.en on the CPU: rough, but enough to tell "date of birth" from "anything else"."""

    def __init__(self):
        from faster_whisper import WhisperModel

        try:   # cached copy first: a flaky Hugging Face connection shouldn't stop the bench
            self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8", local_files_only=True)
        except Exception:
            self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    def __call__(self, pcm48k: np.ndarray) -> str:
        if len(pcm48k) < RATE // 4:
            return ""
        audio = pcm48k[: len(pcm48k) // 3 * 3].reshape(-1, 3).mean(axis=1).astype(np.float32) / 32768  # 48k -> 16k
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()


class Mouth(MediaStreamTrack):
    """The caller's microphone: plays queued lines in real time, silence otherwise."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = np.zeros(0, np.int16)
        self.pts = 0
        self.start = None
        self.said: list[tuple[float, np.ndarray]] = []

    def say(self, pcm: np.ndarray) -> None:
        self.said.append((time.time(), pcm))
        self.queue = np.concatenate([self.queue, pcm])

    async def recv(self):
        if self.start is None:
            self.start = time.time()
        wait = self.start + self.pts / RATE - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        chunk, self.queue = self.queue[:FRAME], self.queue[FRAME:]
        if len(chunk) < FRAME:
            chunk = np.pad(chunk, (0, FRAME - len(chunk)))
        frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = RATE, self.pts, fractions.Fraction(1, RATE)
        self.pts += FRAME
        return frame


class Ear:
    """The bot's audio as it arrives, with the time of each loud frame."""

    def __init__(self):
        self.frames: list[tuple[float, np.ndarray]] = []
        self.loud: list[float] = []
        self.closed = False

    async def listen(self, track) -> None:
        while True:
            try:
                frame = await track.recv()
            except Exception:   # the bot hung up
                self.closed = True
                return
            now = time.time()
            a = frame.to_ndarray().astype(np.int16).reshape(-1, len(frame.layout.channels)).mean(axis=1)
            self.frames.append((now, a.astype(np.int16)))
            if np.sqrt(np.mean(a.astype(np.float32) ** 2)) > LOUD_RMS:
                self.loud.append(now)

    @property
    def last_loud(self) -> float:
        return self.loud[-1] if self.loud else 0.0

    def first_loud_after(self, t: float) -> float | None:
        i = bisect.bisect_right(self.loud, t)
        return self.loud[i] if i < len(self.loud) else None

    def audio_since(self, t: float) -> np.ndarray:
        parts = [a for at, a in self.frames if at > t]
        return np.concatenate(parts) if parts else np.zeros(0, np.int16)

    async def turn_over(self, since: float, timeout: float = 30.0, longest: float = 90.0) -> bool:
        """Wait for a bot turn that starts after `since` to end. False if the bot never spoke."""
        start = time.time()
        while time.time() - start < longest:
            spoke = self.first_loud_after(since) is not None
            if spoke and time.time() - self.last_loud >= TURN_QUIET_SECS:
                return True
            if self.closed:
                return spoke
            if not spoke and time.time() - start > timeout:
                return False
            await asyncio.sleep(0.05)
        return True

    async def stopped(self, since: float) -> float:
        """After the caller cut in at `since`: seconds until the bot's audio stopped."""
        while time.time() - self.last_loud < STOPPED_QUIET_SECS and not self.closed:
            await asyncio.sleep(0.02)
        return max(0.0, self.last_loud - since)


async def call(scenario: Scenario, voice: Voice, hearing: Hearing, url: str, wav_path: Path | None = None) -> dict:
    """Place one call. Returns the caller's log: what it heard and said, and the interrupt/silence timings."""
    pc = RTCPeerConnection()
    mouth, ear = Mouth(), Ear()
    pc.addTrack(mouth)
    pc.on("track", lambda track: asyncio.ensure_future(ear.listen(track)))
    await pc.setLocalDescription(await pc.createOffer())
    async with aiohttp.ClientSession() as s:
        offer = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type,
                 "request_data": {"customer": scenario.customer}}
        async with s.post(f"{url}/api/offer", json=offer) as r:
            r.raise_for_status()
            answer = await r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    t0 = time.time()
    log: dict = {"scenario": scenario.name, "turns": [], "interrupt": None, "silence": None}

    async def speak(text: str) -> float:
        pcm = voice(text)
        mouth.say(pcm)
        await asyncio.sleep(len(pcm) / RATE)
        return time.time()

    mark = t0
    for _ in range(scenario.max_turns):
        if not await ear.turn_over(mark):
            log["turns"].append({"t": round(time.time() - t0, 1), "bot": "(silent)", "caller": None})
            break
        heard = hearing(ear.audio_since(mark))
        if ear.closed:
            log["turns"].append({"t": round(time.time() - t0, 1), "bot": heard, "caller": None})
            break
        reply = scenario.reply_to(heard)
        log["turns"].append({"t": round(time.time() - t0, 1), "bot": heard, "caller": reply.say})

        if reply.wait:
            quiet_from = time.time()
            while time.time() - quiet_from < reply.wait and not ear.closed:
                await asyncio.sleep(0.1)
            spoke = ear.first_loud_after(quiet_from)
            log["silence"] = {"secs": reply.wait, "reprompt_after": round(spoke - quiet_from, 2) if spoke else None}
            if spoke:
                await ear.turn_over(quiet_from, timeout=0)
            if ear.closed:
                break

        mark = await speak(reply.say)
        if reply.hang_up:   # the caller gave up; nothing the bot says now changes the result
            log["caller_gave_up"] = True
            break

        if reply.interrupt:
            while ear.first_loud_after(mark) is None and time.time() - mark < 30 and not ear.closed:
                await asyncio.sleep(0.02)
            began = ear.first_loud_after(mark)
            if began is None:
                continue
            await asyncio.sleep(max(0.0, began + INTERRUPT_AFTER_SECS - time.time()))
            if time.time() - ear.last_loud > 0.3:   # the bot finished before the caller could cut in
                log["interrupt"] = {"line": reply.interrupt, "stopped_after": None}
                continue
            cut_in = time.time()
            pcm = voice(reply.interrupt)
            mouth.say(pcm)
            stopped_after = await ear.stopped(cut_in)
            log["interrupt"] = {"line": reply.interrupt, "stopped_after": round(stopped_after, 2)}
            log["turns"].append({"t": round(cut_in - t0, 1), "bot": "(talked over)", "caller": reply.interrupt})
            await asyncio.sleep(max(0.0, cut_in + len(pcm) / RATE - time.time()))
            mark = time.time()

    end = time.time() + (0 if log.get("caller_gave_up") else 15)   # let the goodbye play and the bot hang up
    while time.time() < end and not ear.closed and pc.connectionState not in ("closed", "failed"):
        await asyncio.sleep(0.2)
    log["bot_hung_up"] = ear.closed or pc.connectionState in ("closed", "failed")
    log["seconds"] = round(time.time() - t0, 1)
    await pc.close()
    if wav_path:
        _save_wav(wav_path, t0, ear.frames, mouth.said)
    return log


def _save_wav(path: Path, t0: float, bot: list, caller: list) -> None:
    """Both sides on one track, each placed at the time it was heard or said."""
    pieces = [(t, a) for t, a in bot] + [(t, a) for t, a in caller]
    if not pieces:
        return
    length = max(int((t - t0) * RATE) + len(a) for t, a in pieces)
    mix = np.zeros(length, np.int32)
    for t, a in pieces:
        start = int((t - t0) * RATE)
        mix[start:start + len(a)] += a
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes(np.clip(mix, -32768, 32767).astype(np.int16).tobytes())
