"""CUDA extension wrapper for SwiGLU."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def _load_ext() -> object:
    cuda_pkg = __package__.rsplit(".", 1)[0]
    src_root = resources.files(cuda_pkg) / "cuda_ext"
    with resources.as_file(src_root) as src_dir:
        sources = [
            str(src_dir / "swiglu_ext.cpp"),
            str(src_dir / "swiglu_ext.cu"),
        ]
        return load(
            name="triton_bench_swiglu_ext",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
            verbose=False,
        )


def swiglu_cuda(x: torch.Tensor) -> torch.Tensor:
    """Compute SwiGLU using a custom CUDA extension."""
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor.")
    if x.dtype != torch.float16:
        raise ValueError("x must be float16.")
    if x.ndim < 2:
        raise ValueError(f"x must be at least 2D (got ndim={x.ndim}).")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x.shape[-1] must be even (got {x.shape[-1]}).")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous.")

    hidden2 = x.shape[-1]
    hidden = hidden2 // 2
    x2 = x.view(-1, hidden2)

    ext = _load_ext()
    y2 = ext.forward(x2)
    return y2.view(*x.shape[:-1], hidden)
