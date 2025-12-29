"""Matmul benchmark comparing Torch vs Triton on CUDA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from src.bench_utils import benchmark_seconds, format_ms

DEFAULT_MAX_ABS_TOL = 5e-2
DEFAULT_MAX_REL_TOL = 1e-2


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=5),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr: object,
    B_ptr: object,
    C_ptr: object,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Triton matmul kernel with FP32 accumulation."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute matmul using Triton."""
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Inputs must be CUDA tensors.")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("Inputs must be float16 tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {a.shape} x {b.shape}")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Inputs must be contiguous.")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    matmul_kernel[grid](
        a,
        b,
        c,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
    )
    return c


def torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute matmul using Torch (cuBLAS/cuBLASLt)."""
    return a @ b


def gemm_tflops(M: int, N: int, K: int, sec: float) -> float:
    """Compute TFLOPS for a GEMM given dimensions and runtime in seconds."""
    return (2.0 * M * N * K) / sec / 1e12


@dataclass(frozen=True)
class MatmulCase:
    """Matmul shapes (M, N, K)."""

    M: int
    N: int
    K: int


def _run_case(case: MatmulCase, *, device: torch.device, warmup: int, iters: int, backends: set[str]) -> None:
    M, N, K = case.M, case.N, case.K
    a = torch.randn((M, K), device=device, dtype=torch.float16).contiguous()
    b = torch.randn((K, N), device=device, dtype=torch.float16).contiguous()

    def cuda_sync(_: object | None) -> None:
        torch.cuda.synchronize()

    if {"torch", "triton"} <= backends:
        with torch.no_grad():
            ref = torch_matmul(a, b)
            out = triton_matmul(a, b)
            diff = (ref - out).abs()
            max_abs = diff.max().item()
            ref_abs_max = ref.abs().max().item()
            max_rel = max_abs / max(ref_abs_max, 1e-6)
            if max_abs > DEFAULT_MAX_ABS_TOL and max_rel > DEFAULT_MAX_REL_TOL:
                print(f"[warn] max_abs_diff={max_abs:.4g} max_rel_diff={max_rel:.4g} (M,N,K=({M},{N},{K}))")

    torch_sec = None
    triton_sec = None
    if "torch" in backends:
        torch_sec = benchmark_seconds(torch_matmul, a, b, warmup=warmup, iters=iters, synchronize=cuda_sync)
    if "triton" in backends:
        triton_sec = benchmark_seconds(triton_matmul, a, b, warmup=warmup, iters=iters, synchronize=cuda_sync)

    torch_ms = None if torch_sec is None else format_ms(torch_sec)
    triton_ms = None if triton_sec is None else format_ms(triton_sec)
    torch_t = None if torch_sec is None else gemm_tflops(M, N, K, torch_sec)
    triton_t = None if triton_sec is None else gemm_tflops(M, N, K, triton_sec)
    speedup = None if (torch_sec is None or triton_sec is None) else (torch_sec / triton_sec)

    def fmt(v: float | None, width: int = 10) -> str:
        return f"{v:{width}.3f}" if v is not None else " " * (width - 2) + "NA"

    def fmt_t(v: float | None, width: int) -> str:
        return f"{v:{width}.2f}" if v is not None else " " * (width - 2) + "NA"

    def fmt_su(v: float | None) -> str:
        return f"{v:8.3f}" if v is not None else "   NA   "

    print(
        f"{M:6d} {N:6d} {K:6d} | {fmt(torch_ms)} {fmt_t(torch_t, 13)} | "
        f"{fmt(triton_ms)} {fmt_t(triton_t, 14)} | {fmt_su(speedup)}"
    )


def run(*, device: torch.device, warmup: int, iters: int, backends: set[str]) -> None:
    """Run matmul benchmarks."""
    if not {"torch", "triton"} & backends:
        raise ValueError("backends must include at least one of: torch, triton")

    cases = [
        MatmulCase(1024, 1024, 1024),
        MatmulCase(2048, 2048, 2048),
        MatmulCase(4096, 4096, 4096),
        MatmulCase(8192, 8192, 4096),
        MatmulCase(8192, 8192, 8192),
        MatmulCase(16384, 4096, 4096),
        MatmulCase(16384, 8192, 4096),
    ]

    print("\n== matmul (fp16) ==")
    print(
        f"{'M':>6} {'N':>6} {'K':>6} | {'torch(ms)':>10} {'torch(TFLOPS)':>13} | "
        f"{'triton(ms)':>10} {'triton(TFLOPS)':>14} | {'speedup':>8}"
    )
    print("-" * 80)
    for case in cases:
        _run_case(case, device=device, warmup=warmup, iters=iters, backends=backends)
