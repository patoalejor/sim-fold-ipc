"""Device selection helper.

Keeps the rest of the code device-agnostic: prefer CUDA when a driver is
present, otherwise fall back to CPU. Warp kernels are identical either way.
"""
from __future__ import annotations

import warp as wp

_INITIALIZED = False


def get_device(prefer_cuda: bool = True) -> str:
    """Return a Warp device string, preferring CUDA when available."""
    global _INITIALIZED
    if not _INITIALIZED:
        wp.init()
        _INITIALIZED = True
    if prefer_cuda and wp.is_cuda_available():
        return "cuda:0"
    return "cpu"
