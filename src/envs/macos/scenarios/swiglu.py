"""SwiGLU benchmark on macOS backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.bench_utils import benchmark_seconds, format_ms
from src.envs.macos.correctness import check_outputs


@dataclass(frozen=True)
class Case:
    """Input shapes for SwiGLU."""

    rows: int
    hidden: int


DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 5e-2


def swiglu_ref(x: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU in FP32 on CPU."""
    a, b = x.chunk(2, dim=-1)
    y = torch.nn.functional.silu(a.to(torch.float32)) * b.to(torch.float32)
    return y.to(x.dtype)


def _bench_torch_swiglu(*, x: torch.Tensor, warmup: int, iters: int) -> float:
    def torch_op(t: object) -> object:
        a, b = t.chunk(2, dim=-1)
        return torch.nn.functional.silu(a) * b

    torch_compiled = torch.compile(torch_op)

    def torch_sync(_: object | None) -> None:
        torch.mps.synchronize()

    return benchmark_seconds(torch_compiled, x, warmup=warmup, iters=iters, synchronize=torch_sync)


def _bench_mlx_swiglu(*, x: object, warmup: int, iters: int) -> float:
    import mlx.core as mx

    def mlx_op(t: object) -> object:
        a, b = mx.split(t, 2, axis=-1)
        out = (a * mx.sigmoid(a)) * b
        # MLX is lazy; force compute in the timed loop.
        mx.eval(out)
        return out

    def mlx_sync(_: object | None) -> None:
        mx.synchronize()

    return benchmark_seconds(mlx_op, x, warmup=warmup, iters=iters, synchronize=mlx_sync)


def _bench_jax_swiglu(*, x: object, warmup: int, iters: int) -> float:
    import jax
    import jax.numpy as jnp

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

    torch.manual_seed(seed)
    x_cpu = torch.randn((rows, 2 * hidden), dtype=torch.float16)
    x_np = x_cpu.numpy()

    ref = swiglu_ref(x_cpu)

    torch_input = None
    mlx_input = None
    jax_input = None

    if "torch" in backends:
        device = torch.device("mps")
        torch_input = x_cpu.to(device)

    if "mlx" in backends:
        import mlx.core as mx

        mlx_input = mx.array(x_np)

    if "jax" in backends:
        import jax
        import jax.numpy as jnp

        metal_devices = [d for d in jax.devices() if d.platform.lower() == "metal"]
        device = metal_devices[0]
        jax_input = jax.device_put(jnp.array(x_np), device)

    with torch.no_grad():
        outputs: dict[str, object] = {}
        if torch_input is not None:
            a, b = torch_input.chunk(2, dim=-1)
            outputs["torch"] = torch.nn.functional.silu(a) * b
        if mlx_input is not None:
            import mlx.core as mx

            a, b = mx.split(mlx_input, 2, axis=-1)
            out = (a * mx.sigmoid(a)) * b
            mx.eval(out)
            outputs["mlx"] = out
        if jax_input is not None:
            import jax
            import jax.numpy as jnp

            a, b = jnp.split(jax_input, 2, axis=-1)
            out = jax.nn.silu(a) * b
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

    torch_sec = _bench_torch_swiglu(x=torch_input, warmup=warmup, iters=iters) if torch_input is not None else None
    mlx_sec = _bench_mlx_swiglu(x=mlx_input, warmup=warmup, iters=iters) if mlx_input is not None else None
    jax_sec = _bench_jax_swiglu(x=jax_input, warmup=warmup, iters=iters) if jax_input is not None else None

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
