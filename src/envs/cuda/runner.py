#!/usr/bin/env python3
"""CLI runner for CUDA (Linux) benchmarks."""

from __future__ import annotations

import argparse

import torch
import triton

try:
    from src.bench_utils import DEFAULT_ITERS, DEFAULT_WARMUP
    from src.envs.cuda.scenarios import SCENARIOS
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Run as a module from the repo root:\n  uv run -m src.envs.cuda.runner --help") from exc

DIVIDER = "-" * 80
BACKEND_CHOICES = ("torch", "cuda_ext", "triton")


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for CUDA benchmarks."""
    parser = argparse.ArgumentParser(description="Compare Torch (cuBLAS) vs Triton vs custom CUDA fused ops.")
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


def ensure_cuda() -> None:
    """Exit if CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. torch.cuda.is_available() == False")


def configure_torch(seed: int) -> None:
    """Configure Torch deterministic settings for CUDA benchmarking."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
    if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
        cudnn_conv.fp32_precision = "ieee"


def print_system_info(args: argparse.Namespace) -> None:
    """Print device and benchmark configuration."""
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}, triton: {triton.__version__}")
    print(f"SM count: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print(f"bench: warmup={args.warmup}, iters={args.iters}, backends={','.join(args.backends)}")


def run_scenarios(*, args: argparse.Namespace, device: torch.device, backends: set[str]) -> None:
    """Run all selected scenarios."""
    for name in args.scenarios:
        SCENARIOS[name](device=device, warmup=args.warmup, iters=args.iters, backends=backends, seed=args.seed)


def print_notes() -> None:
    """Print benchmark notes."""
    print("Notes:")
    print(" - Torch runs under torch.compile (Inductor) for CUDA timing.")
    print(" - For these cases, compiled torch is typically fastest; triton/cuda_ext may trail on GEMM-heavy ops.")
    print(" - Triton and custom CUDA extensions still tend to win when real kernel fusion reduces memory traffic.")
    print(" - The first run includes compile cost, so warmup is required.")


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)

    ensure_cuda()
    configure_torch(args.seed)

    device = torch.device("cuda")
    print_system_info(args)
    print(DIVIDER)

    backends = set(args.backends)
    run_scenarios(args=args, device=device, backends=backends)

    print(DIVIDER)
    print_notes()


if __name__ == "__main__":
    main()
