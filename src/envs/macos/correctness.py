"""macOS correctness checks for benchmark outputs."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def _unravel_index(flat_index: int, shape: Sequence[int]) -> tuple[int, ...]:
    if not shape:
        return ()
    idx = flat_index
    coords_rev: list[int] = []
    for dim in reversed(shape):
        coords_rev.append(idx % dim)
        idx //= dim
    return tuple(reversed(coords_rev))


def _sample_errors(
    ref: "torch.Tensor",
    out: "torch.Tensor",
    diff: "torch.Tensor",
    *,
    max_samples: int,
) -> list[str]:
    if diff.numel() == 0 or max_samples <= 0:
        return []
    k = min(max_samples, diff.numel())
    flat_diff = diff.flatten()
    flat_ref = ref.flatten()
    flat_out = out.flatten()
    topk = flat_diff.topk(k)
    lines: list[str] = []
    for idx, err in zip(topk.indices.tolist(), topk.values.tolist(), strict=True):
        idx_int = int(idx)
        coord = _unravel_index(idx_int, ref.shape)
        ref_val = float(flat_ref[idx_int].item())
        out_val = float(flat_out[idx_int].item())
        denom = max(abs(ref_val), 1e-6)
        rel = float(err) / denom
        lines.append(f"idx={coord} ref={ref_val:.6g} out={out_val:.6g} abs={float(err):.6g} rel={rel:.6g}")
    return lines


def to_torch(value: object) -> "torch.Tensor":
    """Convert backend output to a CPU torch.Tensor."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if hasattr(value, "block_until_ready"):
        with contextlib.suppress(Exception):
            value.block_until_ready()
    if hasattr(value, "to_numpy"):
        array = value.to_numpy()
    elif hasattr(value, "numpy"):
        array = value.numpy()
    else:
        array = np.asarray(value)
    if not array.flags.writeable:
        array = array.copy()
    return torch.from_numpy(array)


def assert_close_with_details(
    *,
    backend: str,
    ref: "torch.Tensor",
    out: "torch.Tensor",
    rtol: float,
    atol: float,
    seed: int | None,
    context: str | None,
    max_samples: int = 3,
) -> None:
    """Assert outputs are close, raising with detailed diagnostics on failure."""
    import torch

    try:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    except AssertionError as exc:
        lines = [f"Correctness check failed for backend={backend}"]
        if context:
            lines.append(f"context={context}")
        if seed is not None:
            lines.append(f"seed={seed}")
        lines.append(f"ref: shape={tuple(ref.shape)} dtype={ref.dtype} device={ref.device}")
        lines.append(f"out: shape={tuple(out.shape)} dtype={out.dtype} device={out.device}")
        lines.append(f"rtol={rtol} atol={atol}")
        if ref.shape != out.shape:
            lines.append("shape mismatch between reference and output")
            raise AssertionError("\n".join(lines)) from exc

        diff = (ref - out).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        denom = ref.abs().clamp_min(1e-6)
        max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
        lines.append(f"max_abs_err={max_abs:.6g} max_rel_err={max_rel:.6g}")
        sample_lines = _sample_errors(ref, out, diff, max_samples=max_samples)
        if sample_lines:
            lines.append("samples:")
            lines.extend(f"  {line}" for line in sample_lines)
        raise AssertionError("\n".join(lines)) from exc


def check_outputs(
    *,
    ref: "torch.Tensor",
    outputs: Mapping[str, object],
    rtol: float,
    atol: float,
    seed: int | None,
    context: str | None,
    max_samples: int = 3,
) -> None:
    """Compare backend outputs against a reference tensor."""
    for name, out in outputs.items():
        out_t = to_torch(out).to(dtype=ref.dtype)
        assert_close_with_details(
            backend=name,
            ref=ref,
            out=out_t,
            rtol=rtol,
            atol=atol,
            seed=seed,
            context=context,
            max_samples=max_samples,
        )
