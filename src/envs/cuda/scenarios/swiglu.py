"""SwiGLU benchmark comparing Torch vs Triton vs CUDA extension."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from src.bench_utils import benchmark_seconds, format_ms

DEFAULT_MAX_ABS_TOL = 5e-2


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def swiglu_kernel(
    X_ptr: object,
    Y_ptr: object,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Triton kernel for SwiGLU."""
    row = tl.program_id(0)
    pid = tl.program_id(1)

    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N

    a = tl.load(X_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(X_ptr + row * stride_xm + (cols + N) * stride_xn, mask=mask, other=0.0).to(tl.float32)
    y = a * tl.sigmoid(a) * b
    tl.store(Y_ptr + row * stride_ym + cols * stride_yn, y.to(tl.float16), mask=mask)


def swiglu_triton(x: torch.Tensor) -> torch.Tensor:
    """Compute SwiGLU using Triton."""
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
    y2 = torch.empty((x2.shape[0], hidden), device=x.device, dtype=x.dtype)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (x2.shape[0], triton.cdiv(hidden, meta["BLOCK_N"]))

    swiglu_kernel[grid](
        x2,
        y2,
        stride_xm=x2.stride(0),
        stride_xn=x2.stride(1),
        stride_ym=y2.stride(0),
        stride_yn=y2.stride(1),
        N=hidden,
    )
    return y2.view(*x.shape[:-1], hidden)


def swiglu_torch(x: torch.Tensor) -> torch.Tensor:
    """Compute SwiGLU using Torch."""
    a, b = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


def swiglu_ref(x: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU in FP32."""
    a, b = x.chunk(2, dim=-1)
    y = torch.nn.functional.silu(a.to(torch.float32)) * b.to(torch.float32)
    return y.to(x.dtype)


@dataclass(frozen=True)
class Case:
    """Input shapes for SwiGLU."""

    rows: int
    hidden: int


def _swiglu_cuda_ext(x: torch.Tensor) -> torch.Tensor:
    from src.envs.cuda.ops.swiglu_cuda import swiglu_cuda

    return swiglu_cuda(x)


def _run_case(case: Case, *, device: torch.device, warmup: int, iters: int, backends: set[str]) -> None:
    rows, hidden = case.rows, case.hidden
    x = torch.randn((rows, 2 * hidden), device=device, dtype=torch.float16).contiguous()

    def cuda_sync(_: object | None) -> None:
        torch.cuda.synchronize()

    backend_fns = {
        "torch": swiglu_torch,
        "cuda_ext": _swiglu_cuda_ext,
        "triton": swiglu_triton,
    }
    selected = {name: fn for name, fn in backend_fns.items() if name in backends}

    with torch.no_grad():
        ref = swiglu_ref(x)
        max_abs = 0.0
        for fn in selected.values():
            out = fn(x)
            max_abs = max(max_abs, (ref - out).abs().max().item())
        if max_abs > DEFAULT_MAX_ABS_TOL:
            print(f"[warn] max_abs_diff={max_abs:.4g} (rows,hidden=({rows},{hidden}))")

    secs = {
        name: benchmark_seconds(fn, x, warmup=warmup, iters=iters, synchronize=cuda_sync)
        for name, fn in selected.items()
    }
    torch_sec = secs.get("torch")
    cuda_sec = secs.get("cuda_ext")
    triton_sec = secs.get("triton")

    def ms(sec: float | None) -> float | None:
        return None if sec is None else format_ms(sec)

    torch_ms = ms(torch_sec)
    cuda_ms = ms(cuda_sec)
    triton_ms = ms(triton_sec)

    speedup_cuda = None if (torch_sec is None or cuda_sec is None) else (torch_sec / cuda_sec)
    speedup_triton = None if (torch_sec is None or triton_sec is None) else (torch_sec / triton_sec)

    def fmt(v: float | None, width: int = 10) -> str:
        return f"{v:{width}.3f}" if v is not None else " " * (width - 2) + "NA"

    def fmt_su(v: float | None) -> str:
        return f"{v:8.3f}" if v is not None else "   NA   "

    print(
        f"{rows:6d} {hidden:6d} | {fmt(torch_ms)} | {fmt(cuda_ms)} {fmt_su(speedup_cuda)} |"
        f" {fmt(triton_ms)} {fmt_su(speedup_triton)}"
    )


def run(*, device: torch.device, warmup: int, iters: int, backends: set[str]) -> None:
    """Run SwiGLU benchmarks."""
    if not {"torch", "cuda_ext", "triton"} & backends:
        raise ValueError("backends must include at least one of: torch, cuda_ext, triton")

    cases = [
        Case(rows=4096, hidden=1024),
        Case(rows=4096, hidden=4096),
        Case(rows=16384, hidden=4096),
    ]

    print("\n== swiglu (fp16) ==")
    print(
        f"{'rows':>6} {'hid':>6} | {'torch(ms)':>10} | {'cuda(ms)':>10} {'speedup':>8} |"
        f" {'triton(ms)':>10} {'speedup':>8}"
    )
    print("-" * 80)
    for case in cases:
        _run_case(case, device=device, warmup=warmup, iters=iters, backends=backends)
