"""NVTX helpers for Nsight Systems tracing."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch


class NvtxTracer:
    """Lightweight NVTX range helper."""

    def __init__(self, enabled: bool, nvtx: Any | None) -> None:
        self.enabled = enabled
        self._nvtx = nvtx

    @contextmanager
    def range(self, name: str) -> object:
        """Emit an NVTX range when enabled."""
        if not self.enabled:
            yield None
            return
        self._nvtx.range_push(name)
        try:
            yield None
        finally:
            self._nvtx.range_pop()


def build_tracer(enable_nvtx: bool) -> NvtxTracer:
    """Create a tracer, raising if NVTX is requested but unavailable."""
    if not enable_nvtx:
        return NvtxTracer(False, None)
    nvtx = getattr(torch.cuda, "nvtx", None)
    if nvtx is None or not hasattr(nvtx, "range_push") or not hasattr(nvtx, "range_pop"):
        raise SystemExit("NVTX is required for --nsys but torch.cuda.nvtx is unavailable.")
    return NvtxTracer(True, nvtx)
