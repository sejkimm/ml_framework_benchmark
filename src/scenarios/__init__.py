from __future__ import annotations

from src.scenarios.matmul import run as run_matmul
from src.scenarios.residual_rmsnorm import run as run_residual_rmsnorm
from src.scenarios.swiglu import run as run_swiglu

SCENARIOS = {
    "matmul": run_matmul,
    "residual_rmsnorm": run_residual_rmsnorm,
    "swiglu": run_swiglu,
}

