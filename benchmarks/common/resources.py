#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Device synchronisation and memory measurement for the benchmarks, plus the log-log scaling fit.

Every import of ``torch`` and ``psutil`` is local to the function that needs it, so that this module can be imported
(and the CPU-only parts used) from an environment that has neither, such as the nuSQuIDS one.

Module contents:
    cuda_synchronizer(...)
        The ``synchronize`` hook for ``timing.time_call``: waits for pending CUDA work, does nothing otherwise.
    reset_cuda_peak_memory(...), cuda_peak_memory_gb(...)
        Peak CUDA memory allocated since the last reset, in GiB.
    process_ram_gb()
        Resident memory of the current process, in GiB.
    check_ram_budget(...)
        Raise ``MemoryError`` when the process exceeds a RAM budget.
    fit_loglog_slope(...)
        Slope of the straight line fitted to a quantity against its size on log-log axes (the scaling exponent).
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import numpy as np

_GIB = 1024**3


def cuda_synchronizer(device: object = "cpu") -> Optional[Callable[[], None]]:
    """The ``synchronize`` hook for a device: waits for pending CUDA work, ``None`` for any other device.

    Args:
        device: A ``torch.device`` or its string form (``"cpu"``, ``"cuda"``, ``"cuda:0"``).

    Returns:
        A zero-argument callable that synchronises the device, or ``None`` when the device is not CUDA (or ``torch``
        is not installed), which ``timing.time_call`` accepts as "nothing to wait for".
    """
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch
    except ImportError:
        return None
    target = torch.device(device)  # type: ignore[arg-type]
    return lambda: torch.cuda.synchronize(target)


def reset_cuda_peak_memory(device: object = "cpu") -> None:
    """Reset the peak CUDA memory counter of ``device`` (no effect on any other device).

    Args:
        device: A ``torch.device`` or its string form.
    """
    if not str(device).startswith("cuda"):
        return
    import torch

    torch.cuda.reset_peak_memory_stats(torch.device(device))  # type: ignore[arg-type]


def cuda_peak_memory_gb(device: object = "cpu") -> Optional[float]:
    """Peak CUDA memory allocated since the last reset, in GiB; ``None`` when ``device`` is not CUDA.

    Args:
        device: A ``torch.device`` or its string form.

    Returns:
        The peak allocated memory in GiB, or ``None``.
    """
    if not str(device).startswith("cuda"):
        return None
    import torch

    return torch.cuda.max_memory_allocated(torch.device(device)) / _GIB  # type: ignore[arg-type]


def process_ram_gb() -> Optional[float]:
    """Resident memory of the current process, in GiB; ``None`` when ``psutil`` is not installed."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / _GIB


def check_ram_budget(max_ram_gb: float) -> None:
    """Raise ``MemoryError`` when the resident memory of the process exceeds ``max_ram_gb``.

    The reference packages can exhaust the RAM on large grids; checking the budget between chunks turns that into a
    controlled failure that the benchmark can record, instead of a killed process. Does nothing when ``psutil`` is not
    installed.

    Args:
        max_ram_gb: The budget, in GiB.

    Raises:
        MemoryError: If the process uses more than ``max_ram_gb``.
    """
    used = process_ram_gb()
    if used is not None and used > max_ram_gb:
        raise MemoryError(f"Process RAM {used:.2f} GiB exceeds max_ram_gb={max_ram_gb:.2f} GiB")


def fit_loglog_slope(sizes: Sequence[float], values: Sequence[float]) -> float:
    """Slope of ``log10(values)`` against ``log10(sizes)``: the scaling exponent of a cost.

    Entries that are not strictly positive are ignored (they have no logarithm).

    Args:
        sizes: Problem sizes (e.g. the number of energies).
        values: The measured quantity at each size (e.g. the minimum time).

    Returns:
        The slope of the least-squares line, or ``nan`` when fewer than two usable points remain. A slope of 1 is a
        cost linear in the size; a slope below 1 means the cost is dominated by a fixed overhead.
    """
    x = np.asarray(sizes, dtype=float)
    y = np.asarray(values, dtype=float)
    keep = (x > 0) & (y > 0)
    if keep.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log10(x[keep]), np.log10(y[keep]), 1)[0])
