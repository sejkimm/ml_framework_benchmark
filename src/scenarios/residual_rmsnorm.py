from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from src.bench_utils import benchmark_seconds, format_ms

DEFAULT_EPS = 1e-6
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
def residual_rmsnorm_kernel(
    X_ptr,
    R_ptr,
    W_ptr,
    Y_ptr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_rm: tl.constexpr,
    stride_rn: tl.constexpr,
    stride_w: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    sum_sq = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R_ptr + row * stride_rm + cols * stride_rn, mask=mask, other=0.0).to(tl.float32)
        z = x + r
        sum_sq += tl.sum(z * z, axis=0)

    inv_rms = tl.rsqrt(sum_sq / N + EPS)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R_ptr + row * stride_rm + cols * stride_rn, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols * stride_w, mask=mask, other=0.0).to(tl.float32)
        y = (x + r) * inv_rms * w
        tl.store(Y_ptr + row * stride_ym + cols * stride_yn, y.to(tl.float16), mask=mask)


def residual_rmsnorm_triton(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, *, eps: float = DEFAULT_EPS
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
    y2 = torch.empty_like(x2)

    grid = (x2.shape[0],)
    residual_rmsnorm_kernel[grid](
        x2,
        r2,
        w,
        y2,
        stride_xm=x2.stride(0),
        stride_xn=x2.stride(1),
        stride_rm=r2.stride(0),
        stride_rn=r2.stride(1),
        stride_w=w.stride(0),
        stride_ym=y2.stride(0),
        stride_yn=y2.stride(1),
        N=hidden,
        EPS=float(eps),
    )
    return y2.view_as(x)


def residual_rmsnorm_torch(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, *, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    z = x + residual
    inv_rms = torch.rsqrt((z * z).mean(dim=-1, keepdim=True) + eps)
    return (z * inv_rms) * weight


def residual_rmsnorm_ref(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, *, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    z = (x + residual).to(torch.float32)
    inv_rms = torch.rsqrt((z * z).mean(dim=-1, keepdim=True) + eps)
    y = (z * inv_rms) * weight.to(torch.float32)
    return y.to(x.dtype)


@dataclass(frozen=True)
class Case:
    rows: int
    hidden: int


def _run_case(case: Case, *, device: torch.device, warmup: int, iters: int, backends: set[str]) -> None:
    rows, hidden = case.rows, case.hidden
    x = torch.randn((rows, hidden), device=device, dtype=torch.float16).contiguous()
    residual = torch.randn((rows, hidden), device=device, dtype=torch.float16).contiguous()
    weight = torch.randn((hidden,), device=device, dtype=torch.float16).contiguous()

    with torch.no_grad():
        ref = residual_rmsnorm_ref(x, residual, weight)

        torch_max_abs = None
        cuda_max_abs = None
        triton_max_abs = None

        if "torch" in backends:
            out = residual_rmsnorm_torch(x, residual, weight)
            torch_max_abs = (ref - out).abs().max().item()
        if "cuda_ext" in backends:
            from src.ops.residual_rmsnorm_cuda import residual_rmsnorm_cuda

            out = residual_rmsnorm_cuda(x, residual, weight, eps=DEFAULT_EPS)
            cuda_max_abs = (ref - out).abs().max().item()
        if "triton" in backends:
            out = residual_rmsnorm_triton(x, residual, weight, eps=DEFAULT_EPS)
            triton_max_abs = (ref - out).abs().max().item()

        max_abs = max(v for v in [torch_max_abs, cuda_max_abs, triton_max_abs] if v is not None)
        if max_abs > DEFAULT_MAX_ABS_TOL:
            print(f"[warn] max_abs_diff={max_abs:.4g} (rows,hidden=({rows},{hidden}))")

    torch_sec = None
    cuda_sec = None
    triton_sec = None

    if "torch" in backends:
        torch_sec = benchmark_seconds(residual_rmsnorm_torch, x, residual, weight, warmup=warmup, iters=iters)
    if "cuda_ext" in backends:
        from src.ops.residual_rmsnorm_cuda import residual_rmsnorm_cuda

        cuda_sec = benchmark_seconds(residual_rmsnorm_cuda, x, residual, weight, warmup=warmup, iters=iters)
    if "triton" in backends:
        triton_sec = benchmark_seconds(residual_rmsnorm_triton, x, residual, weight, warmup=warmup, iters=iters)

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
    if not {"torch", "cuda_ext", "triton"} & backends:
        raise ValueError("backends must include at least one of: torch, cuda_ext, triton")

    cases = [
        Case(rows=4096, hidden=1024),
        Case(rows=4096, hidden=4096),
        Case(rows=8192, hidden=4096),
        Case(rows=16384, hidden=4096),
    ]

    print("\n== residual + rmsnorm (fp16) ==")
    print(f"{'rows':>6} {'hid':>6} | {'torch(ms)':>10} | {'cuda(ms)':>10} {'su':>8} | {'triton(ms)':>10} {'su':>8}")
    print("-" * 80)
    for case in cases:
        _run_case(case, device=device, warmup=warmup, iters=iters, backends=backends)

