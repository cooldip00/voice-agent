"""Step 0: check every local piece before building anything.

Run:  uv run python -m scripts.check_env
Each check prints OK or FAIL with a fix. The first run downloads the Whisper model
(about 800 MB) and the Kokoro voice files (about 340 MB).
"""

import os
import subprocess
import sys
import time

# Windows without Developer Mode can't make symlinks; Hugging Face then copies files and warns. Harmless.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import requests

from agent import config
from agent.gpu import enable_cuda_dlls, missing_dlls

results: list[tuple[str, bool]] = []


def report(name: str, ok: bool, detail: str):
    results.append((name, ok))
    print(f"{'OK  ' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def gpu_memory_used_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def check_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        report("GPU", False, "nvidia-smi not found. Install the NVIDIA driver.")
        return
    report("GPU", out.returncode == 0, out.stdout.strip() or out.stderr.strip())


def check_cuda_dlls():
    missing = missing_dlls()
    if missing:
        report("CUDA libraries", False,
               f"missing {', '.join(missing)}. Run: uv add nvidia-cublas-cu12 \"nvidia-cudnn-cu12==9.*\"")
    else:
        report("CUDA libraries", True, "cuBLAS 12 and cuDNN 9 found in the venv")


def check_whisper():
    """Load the Whisper model on the GPU and transcribe 1 s of silence.

    Silence is enough: the encoder runs on the GPU either way, which proves the CUDA
    libraries load. Real speech comes in step 1.
    """
    enable_cuda_dlls()
    from faster_whisper import WhisperModel

    before = gpu_memory_used_mb()
    try:
        t0 = time.perf_counter()
        model = WhisperModel(config.STT_MODEL, device=config.STT_DEVICE,
                             compute_type=config.STT_COMPUTE_TYPE)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
        list(segments)  # transcribe is lazy; consuming the generator runs the model
        run_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        report("Whisper on GPU", False,
               f"{type(e).__name__}: {e}. Fallback: set STT_DEVICE='cpu', STT_COMPUTE_TYPE='int8' in agent/config.py")
        return
    after = gpu_memory_used_mb()
    vram = f", ~{after - before} MB VRAM" if before is not None and after is not None else ""
    report("Whisper on GPU", True,
           f"{config.STT_MODEL} loaded in {load_s:.1f}s, 1 s of audio in {run_ms:.0f} ms{vram}")
    del model


def check_ollama():
    try:
        resp = requests.get(f"{config.OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        report("Ollama", False, f"not reachable at {config.OLLAMA_BASE} ({type(e).__name__}). Start the Ollama app.")
        return
    names = {m.get("name", "") for m in resp.json().get("models", [])}
    report("Ollama", True, f"running at {config.OLLAMA_BASE}")
    for model in config.LLM_MODELS_NEEDED:
        report(f"model {model}", model in names,
               "pulled" if model in names else f"not pulled. Run: ollama pull {model}")


def check_kokoro():
    """Download the voice files where Pipecat expects them, then speak one sentence."""
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files
    from kokoro_onnx import Kokoro

    from agent.gpu import onnx_session

    model_path = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
    voices_path = KOKORO_CACHE_DIR / "voices-v1.0.bin"
    try:
        if not (model_path.exists() and voices_path.exists()):
            print("      downloading Kokoro files (about 340 MB)...", flush=True)
        _ensure_model_files(model_path, voices_path)
        session = onnx_session(model_path, config.TTS_DEVICE)
        kokoro = Kokoro.from_session(session, str(voices_path))
        text = "Hello, this is a test call."
        kokoro.create(text, voice=config.TTS_VOICE, lang="en-us")   # the first run sets up CUDA; time the second
        t0 = time.perf_counter()
        samples, sample_rate = kokoro.create(text, voice=config.TTS_VOICE, lang="en-us")
        synth_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        report("Kokoro voice", False, f"{type(e).__name__}: {e}")
        return
    device = session.get_providers()[0]
    on_wanted = config.TTS_DEVICE != "cuda" or device == "CUDAExecutionProvider"
    report("Kokoro voice", on_wanted,
           f"voice '{config.TTS_VOICE}': {len(samples) / sample_rate:.1f}s of speech made in {synth_ms:.0f} ms "
           f"on {device}" + ("" if on_wanted else ". Wanted CUDA: is onnxruntime-gpu installed (uv sync)?"))


def check_turn_taking():
    try:
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
        from pipecat.audio.vad.silero import SileroVADAnalyzer

        SileroVADAnalyzer()
        LocalSmartTurnAnalyzerV3()
    except Exception as e:
        report("Silero VAD + Smart Turn", False, f"{type(e).__name__}: {e}")
        return
    report("Silero VAD + Smart Turn", True, "both models load on CPU")


def main():
    print(f"Python {sys.version.split()[0]}\n", flush=True)
    # Kokoro before Whisper, as in agent/bot.py: the other order makes onnxruntime fail to load cuBLAS.
    for check in (check_gpu, check_cuda_dlls, check_kokoro, check_whisper, check_ollama, check_turn_taking):
        check()
    failed = [name for name, ok in results if not ok]
    print("\nAll green." if not failed else f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
