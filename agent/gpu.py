"""Make the pip-installed CUDA libraries visible to faster-whisper on Windows.

faster-whisper (through CTranslate2) loads cublas64_12.dll and cudnn64_9.dll by name.
Windows only finds a DLL by name in a few places, and the venv's nvidia/*/bin folders
are not among them, so they are added here. Call enable_cuda_dlls() before any Whisper
model is created.
"""

import os
import sys
from pathlib import Path

NVIDIA_DIR = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
DLL_DIRS = [NVIDIA_DIR / "cublas" / "bin", NVIDIA_DIR / "cudnn" / "bin"]
REQUIRED_DLLS = ["cublas64_12.dll", "cudnn64_9.dll"]


def missing_dlls() -> list[str]:
    """Names of required DLLs not found in DLL_DIRS."""
    return [name for name in REQUIRED_DLLS if not any((d / name).exists() for d in DLL_DIRS)]


def enable_cuda_dlls() -> None:
    if os.name != "nt":
        return
    for d in DLL_DIRS:
        if d.is_dir():
            os.add_dll_directory(str(d))
            # Safety net: plain LoadLibrary lookups search PATH, not add_dll_directory folders.
            # Whisper worked without this (tested); cuDNN loads other parts of itself at runtime by name.
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]
