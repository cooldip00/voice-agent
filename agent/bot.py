"""The collections call. Browser mic -> VAD + Smart Turn -> Whisper -> Qwen (+ tools) -> Kokoro -> browser speaker.

Run:  uv run python -m agent.bot      then open http://localhost:7860
      $env:CUSTOMER=3                  to call another customer (default 1, Rahul)
      $env:ALLOW_ANY_HOUR=1            to test outside 08:00-19:00

Each call: fixed greeting with the recording notice -> date of birth -> loan -> promise, callback, dispute
or escalation -> end_call -> fixed goodbye -> hang up. The record in calls/<call_id>.json is built from
CallState (what the tools accepted), not from what the LLM says happened.

The heavy models (Whisper on the GPU, Kokoro, Qwen in Ollama) load and warm up once, when this module
starts, before the server accepts a call. Each call then gets its own pipeline, whose services reuse
those models. Silero VAD and Smart Turn are small and keep per-call state, so each call makes its own.
"""

import json
import os
import time
from datetime import datetime

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from agent import config, prompts
from agent.gpu import enable_cuda_dlls

enable_cuda_dlls()  # before faster_whisper is imported

import numpy as np
import requests
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from loguru import logger

import pipecat.services.kokoro.tts as kokoro_tts
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    FunctionCallResultProperties,
    LLMContextFrame,
    LLMUpdateSettingsFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, KokoroTTSService, _ensure_model_files
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from agent import bank
from agent.call_state import (
    CallState,
    call_record,
    final_outcome,
    join_caller_turns,
    unsafe_to_say,
    within_call_hours,
)
from agent.speech import speakable
from agent.tools import IDENTITY_TOOLS, TOOL_NAMES, TOOLS, CallTools


def load_whisper() -> WhisperModel:
    t = time.perf_counter()
    model = WhisperModel(config.STT_MODEL, device=config.STT_DEVICE, compute_type=config.STT_COMPUTE_TYPE)
    segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
    list(segments)  # transcribe is lazy; consuming the segments runs the model
    logger.info(f"Whisper loaded and warm in {time.perf_counter() - t:.1f}s")
    return model


def load_kokoro() -> Kokoro:
    t = time.perf_counter()
    model_path = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
    voices_path = KOKORO_CACHE_DIR / "voices-v1.0.bin"
    _ensure_model_files(model_path, voices_path)
    model = Kokoro(str(model_path), str(voices_path))
    model.create("Hello.", voice=config.TTS_VOICE, lang="en-us")
    logger.info(f"Kokoro loaded and warm in {time.perf_counter() - t:.1f}s")
    return model


def warm_llm() -> None:
    """An empty prompt makes Ollama load the model into VRAM without generating anything."""
    t = time.perf_counter()
    requests.post(
        f"{config.OLLAMA_BASE}/api/generate",
        json={"model": config.LLM_MODEL, "prompt": "", "keep_alive": config.LLM_KEEP_ALIVE},
        timeout=120,
    ).raise_for_status()
    logger.info(f"{config.LLM_MODEL} loaded in Ollama in {time.perf_counter() - t:.1f}s")


WHISPER = load_whisper()
KOKORO = load_kokoro()
warm_llm()


class SharedWhisperSTT(WhisperSTTService):
    """WhisperSTTService that uses the model loaded at startup instead of loading its own."""

    def _load(self):
        self._model = WHISPER


class QuietKokoroTTS(KokoroTTSService):
    """Says nothing for a sentence the guards in to_words blanked out (Kokoro raises an error on "")."""

    async def run_tts(self, text: str, context_id: str):
        if not text.strip():
            return
        async for frame in super().run_tts(text, context_id):
            yield frame


def make_tts() -> KokoroTTSService:
    # KokoroTTSService builds a kokoro_onnx.Kokoro by name in __init__; hand it the one loaded at startup.
    kokoro_tts.Kokoro = lambda *args, **kwargs: KOKORO
    try:
        return QuietKokoroTTS(settings=KokoroTTSService.Settings(voice=config.TTS_VOICE))
    finally:
        kokoro_tts.Kokoro = Kokoro


class PlayedText(FrameProcessor):
    """Sits right after transport.output(), which passes a sentence's TTSTextFrame on only once its audio
    has played. So this sees what the caller actually heard; an interrupted sentence never arrives.

    A sentence the guards in to_words silenced still arrives (with no audio). It is kept out of the
    context: in test calls Qwen kept repeating a silenced "say it as YYYY-MM-DD" it could still see."""

    def __init__(self, state: CallState):
        super().__init__()
        self.state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSTextFrame) and frame.text.strip() in self.state.silenced:
            frame.append_to_context = False
            logger.debug(f"Kept out of the context: {frame.text!r}")
        elif isinstance(frame, TTSTextFrame) and not self.state.notice_given and prompts.is_notice(frame.text):
            self.state.notice_given = True
            logger.info("Recording notice played")
        await self.push_frame(frame, direction)


class JoinCallerTurns(FrameProcessor):
    """Between the user aggregator and the LLM: joins back-to-back caller messages before each LLM call."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            joined = join_caller_turns(frame.context.messages)
            if len(joined) < len(frame.context.messages):
                frame.context.set_messages(joined)
        await self.push_frame(frame, direction)


def register_tools(llm: OLLamaLLMService, tools: CallTools, worker: PipelineWorker,
                   context: LLMContext, call_prompt: str) -> None:
    def make_handler(name: str):
        async def handler(params: FunctionCallParams):
            arguments = dict(params.arguments or {})
            t = time.perf_counter()
            result = tools.call(name, arguments)
            logger.info(f"Tool {name}({arguments}) -> {result} [{(time.perf_counter() - t) * 1000:.0f} ms]")
            if name == "end_call" and result["ok"]:
                # No more LLM: code says the goodbye, then hangs up once it has played.
                await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
                await worker.queue_frames([TTSSpeakFrame(prompts.GOODBYE), EndFrame()])
            elif name == "verify_identity" and "loan" in result:
                # Only now does the LLM learn why we are calling: the full prompt and all seven tools.
                # ("loan" is missing when it was already verified; then there is nothing to swap or say.)
                context.set_tools(TOOLS)
                await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
                await worker.queue_frames([
                    LLMUpdateSettingsFrame(delta=OLLamaLLMService.Settings(system_instruction=call_prompt)),
                    TTSSpeakFrame(prompts.after_verify(result)),
                ])
            elif name in prompts.SPOKEN_AFTER and name != "verify_identity" and result["ok"]:
                # Facts (amount, dates) are said by code; the LLM speaks again after the caller replies.
                await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
                await worker.queue_frames([TTSSpeakFrame(prompts.SPOKEN_AFTER[name](result))])
            else:
                await params.result_callback(result)

        return handler

    for name in TOOL_NAMES:
        llm.register_function(name, make_handler(name))


def save_record(record: dict) -> None:
    config.CALLS_DIR.mkdir(exist_ok=True)
    path = config.CALLS_DIR / f"{record['call_id']}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Call record: {path}  outcome={record['outcome']}")


async def bot(runner_args: RunnerArguments):
    """Called by Pipecat's runner once per browser connection: one connection = one call to one customer."""
    started = datetime.now()
    if not (within_call_hours(started) or config.ALLOW_ANY_HOUR):
        start, end = config.CALL_HOURS
        logger.warning(f"Not calling: it is {started:%H:%M}, calls are allowed {start}:00-{end}:00. "
                       "For a dev test set $env:ALLOW_ANY_HOUR=1")
        return
    if not config.BANK_DB.exists():
        raise FileNotFoundError(f"{config.BANK_DB} missing; run: uv run python -m scripts.seed_bank")

    conn = bank.connect(config.BANK_DB)
    state = CallState(call_id=f"{started:%Y%m%d-%H%M%S}-c{config.CUSTOMER_ID}", customer_id=config.CUSTOMER_ID)
    tools = CallTools(state, conn)
    name = tools.customer["name"]
    logger.info(f"Call {state.call_id} to {name}")

    transport = await create_transport(
        runner_args,
        {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)},
    )

    stt = SharedWhisperSTT(device=config.STT_DEVICE, compute_type=config.STT_COMPUTE_TYPE)
    llm = OLLamaLLMService(
        base_url=f"{config.OLLAMA_BASE}/v1",
        settings=OLLamaLLMService.Settings(
            model=config.LLM_MODEL,
            # Nothing about the debt until the DOB matches; register_tools swaps in system_prompt() then.
            system_instruction=prompts.identity_prompt(name, started.date()),
            temperature=config.LLM_TEMPERATURE,
        ),
    )
    tts = make_tts()
    explained = False   # the personal-matter line, said at most once

    async def to_words(text: str, aggregation_type) -> str:  # only what is spoken; the context keeps digits
        nonlocal explained
        # Qwen sometimes writes a tool call as plain text; Ollama only parses its <tool_call> format.
        problem = "writes a tool call as text" if any(t in text for t in TOOL_NAMES) else unsafe_to_say(text, state)
        if problem:
            # Silenced; PlayedText keeps it out of the context. Counted for step 5.
            logger.warning(f"Not speaking, LLM {problem}: {text!r}")
            state.llm_slips.append(f"{problem}: {text}")
            state.silenced.append(text.strip())
            if problem == "mentions the debt before identity is verified" and not explained:
                # Instead of silence, a fixed line that tells a third party nothing (RBI).
                explained = True
                await worker.queue_frames([TTSSpeakFrame(prompts.personal_matter(name))])
            return ""
        if prompts.is_farewell(text) and text.strip() not in prompts.GOODBYE and state.outcome is not None:
            # end_call already ran and code's goodbye follows; one is enough.
            state.silenced.append(text.strip())
            return ""
        if prompts.is_farewell(text) and state.outcome is None:
            # Qwen often says goodbye instead of calling end_call. Code ends the call the same way end_call
            # would, with the outcome the tools recorded, so the call still hangs up and gets its record.
            logger.warning(f"LLM said goodbye without end_call, ending the call: {text!r}")
            state.llm_slips.append(f"goodbye without end_call: {text}")
            state.silenced.append(text.strip())
            tools.call("end_call", {"outcome": final_outcome(state)})
            await worker.queue_frames([TTSSpeakFrame(prompts.GOODBYE), EndFrame()])
            return ""
        return speakable(text, started.date())

    tts.add_text_transformer(to_words)

    # Turn-taking lives in the user aggregator: Silero VAD hears speech start and stop, and the
    # default stop strategy (Smart Turn v3) decides whether the thought is finished.
    context = LLMContext(tools=IDENTITY_TOOLS)
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=config.VAD_STOP_SECS)),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            JoinCallerTurns(),
            llm,
            tts,
            transport.output(),
            PlayedText(state),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )
    register_tools(llm, tools, worker, context, prompts.system_prompt(name, started.date()))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Caller connected")
        await worker.queue_frames([TTSSpeakFrame(prompts.greeting(name))])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Caller disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        save_record(call_record(state, context.messages, started, datetime.now()))
        conn.close()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
