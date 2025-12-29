"""Matmul benchmark on macOS backends."""

from __future__ import annotations

from dataclasses import dataclass

from src.bench_utils import benchmark_seconds, format_ms


@dataclass(frozen=True)
class MatmulCase:
    """Matmul shapes (M, N, K)."""

    M: int
    N: int
    K: int


def _bench_torch_matmul(*, M: int, N: int, K: int, warmup: int, iters: int) -> float:
    import torch

    device = torch.device("mps")
    a = torch.randn((M, K), device=device, dtype=torch.float16).contiguous()
    b = torch.randn((K, N), device=device, dtype=torch.float16).contiguous()

    def torch_sync(_: object | None) -> None:
        torch.mps.synchronize()

    return benchmark_seconds(lambda x, y: x @ y, a, b, warmup=warmup, iters=iters, synchronize=torch_sync)


def _bench_mlx_matmul(*, M: int, N: int, K: int, warmup: int, iters: int, seed: int) -> float:
    import mlx.core as mx

    mx.random.seed(seed)
    a = mx.random.normal((M, K)).astype(mx.float16)
    b = mx.random.normal((K, N)).astype(mx.float16)

    def mlx_op(x: object, y: object) -> object:
        out = x @ y
        # MLX is lazy; force compute in the timed loop.
        mx.eval(out)
        return out

    def mlx_sync(_: object | None) -> None:
        mx.synchronize()

    return benchmark_seconds(mlx_op, a, b, warmup=warmup, iters=iters, synchronize=mlx_sync)


def _bench_jax_matmul(*, M: int, N: int, K: int, warmup: int, iters: int, seed: int) -> float:
    import jax
    import jax.numpy as jnp

    metal_devices = [d for d in jax.devices() if d.platform.lower() == "metal"]
    device = metal_devices[0]

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key, 2)
    a = jax.device_put(jax.random.normal(k1, (M, K), dtype=jnp.float16), device)
    b = jax.device_put(jax.random.normal(k2, (K, N), dtype=jnp.float16), device)

    jitted = jax.jit(lambda x, y: x @ y)

    def jax_sync(x: object | None) -> None:
        if x is not None:
            x.block_until_ready()

    return benchmark_seconds(jitted, a, b, warmup=warmup, iters=iters, synchronize=jax_sync)


def _run_case(case: MatmulCase, *, warmup: int, iters: int, backends: set[str], seed: int) -> None:
    M, N, K = case.M, case.N, case.K

    torch_sec = _bench_torch_matmul(M=M, N=N, K=K, warmup=warmup, iters=iters) if "torch" in backends else None
    mlx_sec = _bench_mlx_matmul(M=M, N=N, K=K, warmup=warmup, iters=iters, seed=seed) if "mlx" in backends else None
    jax_sec = _bench_jax_matmul(M=M, N=N, K=K, warmup=warmup, iters=iters, seed=seed) if "jax" in backends else None

    def ms(sec: float | None) -> float | None:
        return None if sec is None else format_ms(sec)

    torch_ms = ms(torch_sec)
    mlx_ms = ms(mlx_sec)
    jax_ms = ms(jax_sec)

    su_mlx = None if (torch_sec is None or mlx_sec is None) else (torch_sec / mlx_sec)
    su_jax = None if (torch_sec is None or jax_sec is None) else (torch_sec / jax_sec)

    def fmt(v: float | None, width: int = 10) -> str:
        return f"{v:{width}.3f}" if v is not None else " " * (width - 2) + "NA"

    def fmt_su(v: float | None) -> str:
        return f"{v:8.3f}" if v is not None else "   NA   "

    print(f"{M:6d} {N:6d} {K:6d} | {fmt(torch_ms)} | {fmt(mlx_ms)} {fmt_su(su_mlx)} | {fmt(jax_ms)} {fmt_su(su_jax)}")


def run(*, warmup: int, iters: int, backends: set[str], seed: int) -> None:
    """Run matmul benchmarks."""
    if not {"torch", "mlx", "jax"} & backends:
        raise ValueError("backends must include at least one of: torch, mlx, jax")

    cases = [
        MatmulCase(1024, 1024, 1024),
        MatmulCase(2048, 2048, 2048),
        MatmulCase(4096, 4096, 4096),
        MatmulCase(8192, 8192, 4096),
    ]

    print("\n== matmul (fp16) ==")
    print(
        f"{'M':>6} {'N':>6} {'K':>6} | {'torch(ms)':>10} | {'mlx(ms)':>10} {'speedup':>8} |"
        f" {'jax(ms)':>10} {'speedup':>8}"
    )
    print("-" * 80)
    for case in cases:
        _run_case(case, warmup=warmup, iters=iters, backends=backends, seed=seed)
