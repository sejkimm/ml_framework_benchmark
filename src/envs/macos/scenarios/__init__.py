"""macOS benchmark scenarios."""

from __future__ import annotations

from src.envs.macos.scenarios.matmul import run as run_matmul
from src.envs.macos.scenarios.residual_rmsnorm import run as run_residual_rmsnorm
from src.envs.macos.scenarios.swiglu import run as run_swiglu

SCENARIOS = {
    "matmul": run_matmul,
    "residual_rmsnorm": run_residual_rmsnorm,
    "swiglu": run_swiglu,
}
