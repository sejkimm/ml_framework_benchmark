"""CUDA benchmark scenarios."""

from __future__ import annotations

from src.envs.cuda.scenarios.matmul import run as run_matmul
from src.envs.cuda.scenarios.residual_rmsnorm import run as run_residual_rmsnorm
from src.envs.cuda.scenarios.swiglu import run as run_swiglu

SCENARIOS = {
    "matmul": run_matmul,
    "residual_rmsnorm": run_residual_rmsnorm,
    "swiglu": run_swiglu,
}
