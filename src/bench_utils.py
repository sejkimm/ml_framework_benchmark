from __future__ import annotations

import time
from collections.abc import Callable

import torch

DEFAULT_WARMUP = 30
DEFAULT_ITERS = 200


def cuda_sync() -> None:
    torch.cuda.synchronize()


def benchmark_seconds(
    fn: Callable, *args, warmup: int = DEFAULT_WARMUP, iters: int = DEFAULT_ITERS
) -> float:
    for _ in range(warmup):
        fn(*args)
    cuda_sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    cuda_sync()
    return (time.perf_counter() - t0) / iters


def format_ms(seconds: float) -> float:
    return seconds * 1e3

