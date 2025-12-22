from __future__ import annotations

from functools import lru_cache
from importlib import resources

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def _load_ext():
    src_root = resources.files("benchmark") / "cuda_ext"
    with resources.as_file(src_root) as src_dir:
        sources = [
            str(src_dir / "residual_rmsnorm_ext.cpp"),
            str(src_dir / "residual_rmsnorm_ext.cu"),
        ]
        return load(
            name="triton_bench_residual_rmsnorm_ext",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
            verbose=False,
        )


def residual_rmsnorm_cuda(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor.")
    if x.dtype != torch.float16:
        raise ValueError("x must be float16.")
    if residual.dtype != x.dtype or weight.dtype != x.dtype:
        raise ValueError("residual and weight must have the same dtype as x.")
    if x.shape != residual.shape:
        raise ValueError(f"x and residual shapes must match (got {x.shape} vs {residual.shape}).")
    if x.shape[-1] != weight.numel():
        raise ValueError(f"weight must have shape ({x.shape[-1]},) (got {tuple(weight.shape)}).")
    if not x.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x, residual, and weight must be contiguous.")

    hidden = x.shape[-1]
    x2 = x.view(-1, hidden)
    r2 = residual.view(-1, hidden)
    w = weight.view(hidden)

    ext = _load_ext()
    y2 = ext.forward(x2, r2, w, float(eps))
    return y2.view_as(x)
