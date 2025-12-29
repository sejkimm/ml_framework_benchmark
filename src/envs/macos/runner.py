"""CLI runner for macOS benchmarks (MLX / MPS / JAX Metal)."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as importlib_metadata
import platform
import sys

from src.bench_utils import DEFAULT_ITERS, DEFAULT_WARMUP
from src.envs.macos.scenarios import SCENARIOS

DIVIDER = "-" * 80
BACKEND_CHOICES = ("torch", "mlx", "jax")


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for macOS benchmarks."""
    parser = argparse.ArgumentParser(description="Compare MLX vs PyTorch (MPS) vs JAX (Metal).")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=list(SCENARIOS.keys()),
        choices=sorted(SCENARIOS.keys()),
        help="Which scenarios to run.",
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        default=list(BACKEND_CHOICES),
        choices=BACKEND_CHOICES,
        help="Backends to include (scenario-specific).",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
    return build_parser().parse_args(argv)


def ensure_macos() -> None:
    """Exit if not running on macOS."""
    if sys.platform != "darwin":
        raise SystemExit(f"--env macos requires macOS (sys.platform={sys.platform!r})")


def ensure_backends_available(backends: set[str]) -> None:
    """Exit if any requested backend is unavailable."""
    if "torch" in backends:
        import torch

        if not torch.backends.mps.is_available():
            raise SystemExit("PyTorch MPS is required. torch.backends.mps.is_available() == False")
    if "jax" in backends:
        import jax

        if not any(d.platform.lower() == "metal" for d in jax.devices()):
            raise SystemExit(
                "JAX Metal backend is required. Install jax-metal and ensure jax.devices() includes metal."
            )
    if "mlx" in backends:
        importlib.import_module("mlx.core")


def configure_torch(seed: int) -> None:
    """Seed Torch RNG for repeatability."""
    import torch

    torch.manual_seed(seed)


def print_system_info(args: argparse.Namespace, backends: set[str]) -> None:
    """Print device and benchmark configuration."""
    print(f"Platform: {platform.platform()}")
    print(f"bench: warmup={args.warmup}, iters={args.iters}, backends={','.join(args.backends)}")

    def version_or_note(dist: str) -> str:
        try:
            return importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            return "unknown (package metadata not found)"

    if "torch" in backends:
        import torch

        mps_name = getattr(torch.mps, "get_device_name", None)
        device_name = "mps" if mps_name is None else mps_name(0)
        print(f"torch: {torch.__version__} (device={device_name})")

    if "mlx" in backends:
        print(f"mlx: {version_or_note('mlx')}")

    if "jax" in backends:
        import jax

        metal_devices = [d for d in jax.devices() if d.platform.lower() == "metal"]
        dev = metal_devices[0] if metal_devices else jax.devices()[0]
        print(f"jax: {jax.__version__} (device={dev})")


def run_scenarios(*, args: argparse.Namespace, backends: set[str]) -> None:
    """Run all selected scenarios."""
    for name in args.scenarios:
        SCENARIOS[name](warmup=args.warmup, iters=args.iters, backends=backends, seed=args.seed)


def print_notes() -> None:
    """Print benchmark notes."""
    print("Notes:")
    print(" - The first run includes compile/JIT cost, so warmup is required.")
    print(" - Some frameworks enqueue work asynchronously; timings include a sync after each timed loop.")


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    ensure_macos()

    backends = set(args.backends)
    ensure_backends_available(backends)

    if "torch" in backends:
        configure_torch(args.seed)

    print_system_info(args, backends)
    print(DIVIDER)

    run_scenarios(args=args, backends=backends)

    print(DIVIDER)
    print_notes()


if __name__ == "__main__":
    main()
