"""Matmul benchmark on macOS backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.bench_utils import benchmark_seconds, format_ms
from src.envs.macos.correctness import check_outputs


@dataclass(frozen=True)
class MatmulCase:
    """Matmul shapes (M, N, K)."""

    M: int
    N: int
    K: int


DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 5e-2


def matmul_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference matmul in FP32 on CPU."""
    y = a.to(torch.float32) @ b.to(torch.float32)
    return y.to(a.dtype)


def _bench_torch_matmul(*, a: torch.Tensor, b: torch.Tensor, warmup: int, iters: int) -> float:
    torch_op = torch.compile(lambda x, y: x @ y)

    def torch_sync(_: object | None) -> None:
        torch.mps.synchronize()

    return benchmark_seconds(torch_op, a, b, warmup=warmup, iters=iters, synchronize=torch_sync)


def _bench_mlx_matmul(*, a: object, b: object, warmup: int, iters: int) -> float:
    import mlx.core as mx

    def mlx_op(x: object, y: object) -> object:
        out = x @ y
        # MLX is lazy; force compute in the timed loop.
        mx.eval(out)
        return out

    def mlx_sync(_: object | None) -> None:
        mx.synchronize()

    return benchmark_seconds(mlx_op, a, b, warmup=warmup, iters=iters, synchronize=mlx_sync)


def _bench_jax_matmul(*, a: object, b: object, warmup: int, iters: int) -> float:
    import jax

    def jax_sync(x: object | None) -> None:
        if x is not None:
            x.block_until_ready()

    jitted = jax.jit(lambda x, y: x @ y)

    return benchmark_seconds(jitted, a, b, warmup=warmup, iters=iters, synchronize=jax_sync)


def _run_case(case: MatmulCase, *, warmup: int, iters: int, backends: set[str], seed: int) -> None:
    M, N, K = case.M, case.N, case.K

    torch.manual_seed(seed)
    a_cpu = torch.randn((M, K), dtype=torch.float16)
    b_cpu = torch.randn((K, N), dtype=torch.float16)
    a_np = a_cpu.numpy()
    b_np = b_cpu.numpy()

    ref = matmul_ref(a_cpu, b_cpu)

    torch_inputs = None
    mlx_inputs = None
    jax_inputs = None

    if "torch" in backends:
        device = torch.device("mps")
        torch_inputs = (a_cpu.to(device), b_cpu.to(device))

    if "mlx" in backends:
        import mlx.core as mx

        mlx_inputs = (mx.array(a_np), mx.array(b_np))

    if "jax" in backends:
        import jax
        import jax.numpy as jnp

        metal_devices = [d for d in jax.devices() if d.platform.lower() == "metal"]
        device = metal_devices[0]
        jax_inputs = (jax.device_put(jnp.array(a_np), device), jax.device_put(jnp.array(b_np), device))

    with torch.no_grad():
        outputs: dict[str, object] = {}
        if torch_inputs is not None:
            outputs["torch"] = torch_inputs[0] @ torch_inputs[1]
        if mlx_inputs is not None:
            import mlx.core as mx

            out = mlx_inputs[0] @ mlx_inputs[1]
            mx.eval(out)
            outputs["mlx"] = out
        if jax_inputs is not None:
            out = jax_inputs[0] @ jax_inputs[1]
            out.block_until_ready()
            outputs["jax"] = out

        check_outputs(
            ref=ref,
            outputs=outputs,
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
            seed=seed,
            context=f"M={M} N={N} K={K}",
        )

    torch_sec = (
        _bench_torch_matmul(a=torch_inputs[0], b=torch_inputs[1], warmup=warmup, iters=iters)
        if torch_inputs is not None
        else None
    )
    mlx_sec = (
        _bench_mlx_matmul(a=mlx_inputs[0], b=mlx_inputs[1], warmup=warmup, iters=iters)
        if mlx_inputs is not None
        else None
    )
    jax_sec = (
        _bench_jax_matmul(a=jax_inputs[0], b=jax_inputs[1], warmup=warmup, iters=iters)
        if jax_inputs is not None
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
