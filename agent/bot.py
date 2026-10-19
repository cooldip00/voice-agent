"""Step 2: the talking bot. Browser mic -> VAD + Smart Turn -> Whisper -> Qwen -> Kokoro -> browser speaker.

Run:  uv run python -m agent.bot      then open http://localhost:7860

The heavy models (Whisper on the GPU, Kokoro, Qwen in Ollama) load and warm up once, when this module
starts, before the server accepts a call. Each call then gets its own pipeline, whose services reuse
those models. Silero VAD and Smart Turn are small and keep per-call state, so each call makes its own.
"""

import os
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from agent import config
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
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, KokoroTTSService, _ensure_model_files
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

# Placeholder persona; the collections prompt and fixed lines come in step 4.
SYSTEM_PROMPT = (
    "You are Asha, a friendly voice assistant on a phone call. Everything you write is spoken aloud, "
    "so answer in one or two short sentences, in plain words, with no lists, markdown or emoji."
)
GREETING = "Hi, this is Asha. How can I help you today?"


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


def make_tts() -> KokoroTTSService:
    # KokoroTTSService builds a kokoro_onnx.Kokoro by name in __init__; hand it the one loaded at startup.
    kokoro_tts.Kokoro = lambda *args, **kwargs: KOKORO
    try:
        return KokoroTTSService(settings=KokoroTTSService.Settings(voice=config.TTS_VOICE))
    finally:
        kokoro_tts.Kokoro = Kokoro


async def bot(runner_args: RunnerArguments):
    """Called by Pipecat's runner once per browser connection."""
    transport = await create_transport(
        runner_args,
        {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)},
    )

    stt = SharedWhisperSTT(device=config.STT_DEVICE, compute_type=config.STT_COMPUTE_TYPE)
    llm = OLLamaLLMService(
        base_url=f"{config.OLLAMA_BASE}/v1",
        settings=OLLamaLLMService.Settings(model=config.LLM_MODEL, system_instruction=SYSTEM_PROMPT),
    )
    tts = make_tts()

    # Turn-taking lives in the user aggregator: Silero VAD hears speech start and stop, and the
    # default stop strategy (Smart Turn v3) decides whether the thought is finished.
    context = LLMContext()
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
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Caller connected")
        await worker.queue_frames([TTSSpeakFrame(GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Caller disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
