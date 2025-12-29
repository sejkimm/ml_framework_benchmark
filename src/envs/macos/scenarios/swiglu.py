"""SwiGLU benchmark on macOS backends."""

from __future__ import annotations

from dataclasses import dataclass

from src.bench_utils import benchmark_seconds, format_ms


@dataclass(frozen=True)
class Case:
    """Input shapes for SwiGLU."""

    rows: int
    hidden: int


def _bench_torch_swiglu(*, rows: int, hidden: int, warmup: int, iters: int) -> float:
    import torch

    device = torch.device("mps")
    x = torch.randn((rows, 2 * hidden), device=device, dtype=torch.float16).contiguous()

    def torch_op(t: object) -> object:
        a, b = t.chunk(2, dim=-1)
        return torch.nn.functional.silu(a) * b

    def torch_sync(_: object | None) -> None:
        torch.mps.synchronize()

    return benchmark_seconds(torch_op, x, warmup=warmup, iters=iters, synchronize=torch_sync)


def _bench_mlx_swiglu(*, rows: int, hidden: int, warmup: int, iters: int, seed: int) -> float:
    import mlx.core as mx

    mx.random.seed(seed)
    x = mx.random.normal((rows, 2 * hidden)).astype(mx.float16)

    def mlx_op(t: object) -> object:
        a, b = mx.split(t, 2, axis=-1)
        return (a * mx.sigmoid(a)) * b

    def mlx_sync(x: object | None) -> None:
        if x is not None:
            mx.eval(x)

    return benchmark_seconds(mlx_op, x, warmup=warmup, iters=iters, synchronize=mlx_sync)


def _bench_jax_swiglu(*, rows: int, hidden: int, warmup: int, iters: int, seed: int) -> float:
    import jax
    import jax.numpy as jnp

    metal_devices = [d for d in jax.devices() if d.platform.lower() == "metal"]
    device = metal_devices[0]

    key = jax.random.PRNGKey(seed)
    x = jax.device_put(jax.random.normal(key, (rows, 2 * hidden), dtype=jnp.float16), device)

    def jax_op(t: object) -> object:
        a, b = jnp.split(t, 2, axis=-1)
        return jax.nn.silu(a) * b

    jitted = jax.jit(jax_op)

    def jax_sync(x: object | None) -> None:
        if x is not None:
            x.block_until_ready()

    return benchmark_seconds(jitted, x, warmup=warmup, iters=iters, synchronize=jax_sync)


def _run_case(case: Case, *, warmup: int, iters: int, backends: set[str], seed: int) -> None:
    rows, hidden = case.rows, case.hidden

    torch_sec = (
        _bench_torch_swiglu(rows=rows, hidden=hidden, warmup=warmup, iters=iters) if "torch" in backends else None
    )
    mlx_sec = (
        _bench_mlx_swiglu(rows=rows, hidden=hidden, warmup=warmup, iters=iters, seed=seed)
        if "mlx" in backends
        else None
    )
    jax_sec = (
        _bench_jax_swiglu(rows=rows, hidden=hidden, warmup=warmup, iters=iters, seed=seed)
        if "jax" in backends
        else None
    )

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

    print(f"{rows:6d} {hidden:6d} | {fmt(torch_ms)} | {fmt(mlx_ms)} {fmt_su(su_mlx)} | {fmt(jax_ms)} {fmt_su(su_jax)}")


def run(*, warmup: int, iters: int, backends: set[str], seed: int) -> None:
    """Run SwiGLU benchmarks."""
    if not {"torch", "mlx", "jax"} & backends:
        raise ValueError("backends must include at least one of: torch, mlx, jax")

    cases = [
        Case(rows=4096, hidden=1024),
        Case(rows=4096, hidden=4096),
        Case(rows=16384, hidden=4096),
    ]

    print("\n== swiglu (fp16) ==")
    print(
        f"{'rows':>6} {'hid':>6} | {'torch(ms)':>10} | {'mlx(ms)':>10} {'speedup':>8} | {'jax(ms)':>10} {'speedup':>8}"
    )
    print("-" * 80)
    for case in cases:
        _run_case(case, warmup=warmup, iters=iters, backends=backends, seed=seed)
