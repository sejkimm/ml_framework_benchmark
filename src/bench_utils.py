"""Benchmark timing utilities with optional device synchronization."""

from __future__ import annotations

import time
from collections.abc import Callable

DEFAULT_WARMUP = 30
DEFAULT_ITERS = 200


def benchmark_seconds(
    fn: Callable[..., object],
    *args: object,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
    synchronize: Callable[[object | None], None] | None = None,
) -> float:
    """Return average seconds per call for `fn(*args)`."""
    last: object | None = None
    for _ in range(warmup):
        last = fn(*args)
    if synchronize is not None:
        synchronize(last)

    t0 = time.perf_counter()
    for _ in range(iters):
        last = fn(*args)
    if synchronize is not None:
        synchronize(last)
    return (time.perf_counter() - t0) / iters


def format_ms(seconds: float) -> float:
    """Convert seconds to milliseconds."""
    return seconds * 1e3
