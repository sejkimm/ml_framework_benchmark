"""Residual + RMSNorm benchmark on macOS backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.bench_utils import benchmark_seconds, format_ms
from src.envs.macos.correctness import check_outputs

DEFAULT_EPS = 1e-6
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 5e-2


@dataclass(frozen=True)
class Case:
    """Input shapes for residual + RMSNorm."""

    rows: int
    hidden: int


def residual_rmsnorm_ref(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reference residual + RMSNorm in FP32 on CPU."""
    z = (x + residual).to(torch.float32)
    inv_rms = torch.rsqrt((z * z).mean(dim=-1, keepdim=True) + DEFAULT_EPS)
    y = (z * inv_rms) * weight.to(torch.float32)
    return y.to(x.dtype)


def _bench_torch_residual_rmsnorm(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    def torch_op(a: object, b: object, w: object) -> object:
        z = a + b
        inv_rms = torch.rsqrt((z * z).mean(dim=-1, keepdim=True) + DEFAULT_EPS)
        return (z * inv_rms) * w

    torch_compiled = torch.compile(torch_op)

    def torch_sync(_: object | None) -> None:
        torch.mps.synchronize()

    return benchmark_seconds(torch_compiled, x, residual, weight, warmup=warmup, iters=iters, synchronize=torch_sync)


def _bench_mlx_residual_rmsnorm(*, x: object, residual: object, weight: object, warmup: int, iters: int) -> float:
    import mlx.core as mx

    def mlx_op(a: object, b: object, w: object) -> object:
        z = a + b
        inv_rms = mx.rsqrt(mx.mean(z * z, axis=-1, keepdims=True) + DEFAULT_EPS)
        out = (z * inv_rms) * w
        # MLX is lazy; force compute in the timed loop.
        mx.eval(out)
        return out

    def mlx_sync(_: object | None) -> None:
        mx.synchronize()

    return benchmark_seconds(mlx_op, x, residual, weight, warmup=warmup, iters=iters, synchronize=mlx_sync)


def _bench_jax_residual_rmsnorm(*, x: object, residual: object, weight: object, warmup: int, iters: int) -> float:
    import jax
    import jax.numpy as jnp

    def jax_op(a: object, b: object, w: object) -> object:
        z = a + b
        inv_rms = jnp.reciprocal(jnp.sqrt(jnp.mean(z * z, axis=-1, keepdims=True) + DEFAULT_EPS))
        return (z * inv_rms) * w

    jitted = jax.jit(jax_op)

    def jax_sync(x: object | None) -> None:
        if x is not None:
            x.block_until_ready()

    return benchmark_seconds(jitted, x, residual, weight, warmup=warmup, iters=iters, synchronize=jax_sync)


def _run_case(case: Case, *, warmup: int, iters: int, backends: set[str], seed: int) -> None:
    rows, hidden = case.rows, case.hidden

    torch.manual_seed(seed)
    x_cpu = torch.randn((rows, hidden), dtype=torch.float16)
    residual_cpu = torch.randn((rows, hidden), dtype=torch.float16)
    weight_cpu = torch.randn((hidden,), dtype=torch.float16)
    x_np = x_cpu.numpy()
    residual_np = residual_cpu.numpy()
    weight_np = weight_cpu.numpy()

    ref = residual_rmsnorm_ref(x_cpu, residual_cpu, weight_cpu)

    torch_inputs = None
    mlx_inputs = None
    jax_inputs = None

    if "torch" in backends:
        device = torch.device("mps")
        torch_inputs = (x_cpu.to(device), residual_cpu.to(device), weight_cpu.to(device))

    if "mlx" in backends:
        import mlx.core as mx

        mlx_inputs = (mx.array(x_np), mx.array(residual_np), mx.array(weight_np))

    if "jax" in backends:
        import jax
        import jax.numpy as jnp

        metal_devices = [d for d in jax.devices() if d.platform.lower() == "metal"]
        device = metal_devices[0]
        jax_inputs = (
            jax.device_put(jnp.array(x_np), device),
            jax.device_put(jnp.array(residual_np), device),
            jax.device_put(jnp.array(weight_np), device),
        )

    with torch.no_grad():
        outputs: dict[str, object] = {}
        if torch_inputs is not None:
            x_t, r_t, w_t = torch_inputs
            z = x_t + r_t
            inv_rms = torch.rsqrt((z * z).mean(dim=-1, keepdim=True) + DEFAULT_EPS)
            outputs["torch"] = (z * inv_rms) * w_t
        if mlx_inputs is not None:
            import mlx.core as mx

            x_m, r_m, w_m = mlx_inputs
            z = x_m + r_m
            inv_rms = mx.rsqrt(mx.mean(z * z, axis=-1, keepdims=True) + DEFAULT_EPS)
            out = (z * inv_rms) * w_m
            mx.eval(out)
            outputs["mlx"] = out
        if jax_inputs is not None:
            import jax.numpy as jnp

            x_j, r_j, w_j = jax_inputs
            z = x_j + r_j
            inv_rms = jnp.reciprocal(jnp.sqrt(jnp.mean(z * z, axis=-1, keepdims=True) + DEFAULT_EPS))
            out = (z * inv_rms) * w_j
            out.block_until_ready()
            outputs["jax"] = out

        check_outputs(
            ref=ref,
            outputs=outputs,
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
            seed=seed,
            context=f"rows={rows} hidden={hidden}",
        )

    torch_sec = (
        _bench_torch_residual_rmsnorm(
            x=torch_inputs[0], residual=torch_inputs[1], weight=torch_inputs[2], warmup=warmup, iters=iters
        )
        if torch_inputs is not None
        else None
    )
    mlx_sec = (
        _bench_mlx_residual_rmsnorm(
            x=mlx_inputs[0], residual=mlx_inputs[1], weight=mlx_inputs[2], warmup=warmup, iters=iters
        )
        if mlx_inputs is not None
        else None
    )
    jax_sec = (
        _bench_jax_residual_rmsnorm(
            x=jax_inputs[0], residual=jax_inputs[1], weight=jax_inputs[2], warmup=warmup, iters=iters
        )
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

    print(f"{rows:6d} {hidden:6d} | {fmt(torch_ms)} | {fmt(mlx_ms)} {fmt_su(su_mlx)} | {fmt(jax_ms)} {fmt_su(su_jax)}")


def run(*, warmup: int, iters: int, backends: set[str], seed: int) -> None:
    """Run residual + RMSNorm benchmarks."""
    if not {"torch", "mlx", "jax"} & backends:
        raise ValueError("backends must include at least one of: torch, mlx, jax")

    cases = [
        Case(rows=4096, hidden=1024),
        Case(rows=4096, hidden=4096),
        Case(rows=8192, hidden=4096),
    ]

    print("\n== residual + rmsnorm (fp16) ==")
    print(
        f"{'rows':>6} {'hid':>6} | {'torch(ms)':>10} | {'mlx(ms)':>10} {'speedup':>8} | {'jax(ms)':>10} {'speedup':>8}"
    )
    print("-" * 80)
    for case in cases:
        _run_case(case, warmup=warmup, iters=iters, backends=backends, seed=seed)
