#!/usr/bin/env python3
"""Legacy CUDA benchmark entry point.

This module is kept for backward compatibility; use `src.envs.cuda.runner` directly.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    """Forward to the CUDA runner."""
    from src.envs.cuda.runner import main as cuda_main

    cuda_main(argv)


if __name__ == "__main__":
    main()
