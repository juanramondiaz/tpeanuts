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

"""Backend-agnostic wall-clock timing: time a call, summarize repeats, compare two results.

Imports nothing from ``tpeanuts`` or the reference packages: ``time_call`` accepts any zero-argument callable
(typically a ``lambda`` or ``functools.partial`` closing over whichever backend's own API is being timed) and does not
know which backend it belongs to. Devices that run asynchronously (CUDA) are handled by the ``synchronize`` hook,
called before and after every timed call; ``benchmarks.common.resources.cuda_synchronizer`` builds it.

The statistic compared between backends is the minimum (the least sensitive to other load on the machine); the median
and the interquartile range are recorded with it, as the measurement protocol of ``benchmarks/README.md`` requires.

Module contents:
    TimingResult
        One callable's repeated wall-clock measurements, with min, median, IQR, mean and std as properties.
    time_call(...)
        Time ``n_repeats`` calls of a zero-argument callable, after an optional warm-up.
    time_build_and_evaluate(...)
        Time object construction and evaluation separately.
    speedup(...)
        Ratio of two ``TimingResult`` statistics.
    format_timing_table(...)
        Small aligned-text summary table, one row per ``TimingResult``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, TypeVar

import numpy as np

_T = TypeVar("_T")

STATISTICS = ("min", "median", "mean")


@dataclass(frozen=True)
class TimingResult:
    """Repeated wall-clock timings of one callable.

    Attributes:
        name: Label for this timing (e.g. ``"tpeanuts vacuum"``), used only for display.
        times_s: Per-repeat wall-clock durations, in seconds, shape ``(n_repeats,)``. Warm-up calls are not included.
    """

    name: str
    times_s: np.ndarray

    @property
    def n_repeats(self) -> int:
        """Number of timed repeats (excluding any warm-up calls)."""
        return int(self.times_s.shape[0])

    @property
    def min_s(self) -> float:
        """Shortest wall-clock time of one call, in seconds (the statistic backends are compared on)."""
        return float(np.min(self.times_s))

    @property
    def median_s(self) -> float:
        """Median wall-clock time of one call, in seconds."""
        return float(np.median(self.times_s))

    @property
    def iqr_s(self) -> float:
        """Interquartile range (75th minus 25th percentile) of the per-call time, in seconds."""
        q75, q25 = np.percentile(self.times_s, [75, 25])
        return float(q75 - q25)

    @property
    def mean_s(self) -> float:
        """Mean wall-clock time per call, in seconds."""
        return float(np.mean(self.times_s))

    @property
    def std_s(self) -> float:
        """Standard deviation of the per-call wall-clock time, in seconds."""
        return float(np.std(self.times_s))

    def to_dict(self) -> Dict[str, float]:
        """The summary statistics as a JSON-serialisable dictionary (the schema of ``results.BenchmarkRecord``)."""
        return {
            "min_s": self.min_s,
            "median_s": self.median_s,
            "iqr_s": self.iqr_s,
            "mean_s": self.mean_s,
            "std_s": self.std_s,
            "n_repeats": self.n_repeats,
        }


def time_call(
    fn: Callable[[], object],
    *,
    n_repeats: int = 7,
    n_warmup: int = 1,
    synchronize: Optional[Callable[[], None]] = None,
) -> TimingResult:
    """Time ``n_repeats`` calls of a zero-argument callable, after an optional warm-up.

    Args:
        fn: A zero-argument callable to time (bind any real arguments first).
        n_repeats: Number of timed repeats (7 by default, the protocol value).
        n_warmup: Number of untimed calls made first (JIT/cache warm-up); ``0`` disables it.
        synchronize: Called before and after each call, so that asynchronous devices (CUDA) have finished the work
            being timed; ``None`` for CPU backends.

    Returns:
        A ``TimingResult`` named after ``fn``'s ``__name__`` (``"<callable>"`` when it has none).

    Raises:
        ValueError: If ``n_repeats`` is not a positive integer or ``n_warmup`` is negative.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be a positive integer, got {n_repeats!r}.")
    if n_warmup < 0:
        raise ValueError(f"n_warmup must not be negative, got {n_warmup!r}.")

    name = getattr(fn, "__name__", "<callable>")
    sync = synchronize if synchronize is not None else _no_synchronize

    for _ in range(n_warmup):
        fn()
    sync()

    times = np.empty(n_repeats, dtype=float)
    for i in range(n_repeats):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times[i] = time.perf_counter() - start

    return TimingResult(name=name, times_s=times)


def time_build_and_evaluate(
    build: Callable[[], _T],
    evaluate: Callable[[_T], object],
    *,
    n_repeats: int = 7,
    n_warmup: int = 1,
    synchronize: Optional[Callable[[], None]] = None,
) -> Tuple[TimingResult, TimingResult]:
    """Time the construction of an object and its evaluation separately.

    ``build`` is timed on its own (a new object per repeat); ``evaluate`` is timed on one object built beforehand, so
    that the evaluation time does not include construction.

    Args:
        build: Zero-argument callable returning the object to evaluate (a medium, a model, a nuSQuIDS instance...).
        evaluate: Callable taking that object.
        n_repeats: Number of timed repeats of each phase.
        n_warmup: Number of untimed calls of each phase made first.
        synchronize: As in ``time_call``.

    Returns:
        ``(build_timing, evaluate_timing)``.
    """
    build_timing = time_call(build, n_repeats=n_repeats, n_warmup=n_warmup, synchronize=synchronize)
    obj = build()
    evaluate_timing = time_call(lambda: evaluate(obj), n_repeats=n_repeats, n_warmup=n_warmup, synchronize=synchronize)
    evaluate_name = getattr(evaluate, "__name__", "<callable>")
    return (
        TimingResult(name=f"{build_timing.name} (build)", times_s=build_timing.times_s),
        TimingResult(name=f"{evaluate_name} (evaluate)", times_s=evaluate_timing.times_s),
    )


def speedup(baseline: TimingResult, candidate: TimingResult, *, statistic: str = "min") -> float:
    """Ratio of one statistic of two ``TimingResult``s, ``baseline / candidate``.

    A value greater than 1 means ``candidate`` is faster than ``baseline``.

    Args:
        baseline: The reference timing (e.g. nuSQuIDS).
        candidate: The timing compared against it (e.g. tpeanuts).
        statistic: ``"min"`` (default, the protocol statistic), ``"median"`` or ``"mean"``.

    Returns:
        ``baseline.<statistic>_s / candidate.<statistic>_s``.

    Raises:
        ValueError: If ``statistic`` is not one of ``STATISTICS``.
        ZeroDivisionError: If the candidate's statistic is exactly 0.
    """
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, got {statistic!r}.")
    return getattr(baseline, f"{statistic}_s") / getattr(candidate, f"{statistic}_s")


def format_timing_table(results: Sequence[TimingResult]) -> str:
    """Render a small aligned-text summary table, one row per ``TimingResult``.

    Args:
        results: ``TimingResult`` instances to display.

    Returns:
        A multi-line string: a header row (``name``, ``min_s``, ``median_s``, ``iqr_s``, ``n_repeats``) and one row per
        entry in ``results``.
    """
    header = f"{'name':<30} {'min_s':>12} {'median_s':>12} {'iqr_s':>12} {'n_repeats':>10}"
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.name:<30} {result.min_s:>12.6g} {result.median_s:>12.6g} {result.iqr_s:>12.6g} "
            f"{result.n_repeats:>10d}"
        )
    return "\n".join(lines)


def _no_synchronize() -> None:
    """Synchronisation hook of CPU backends: nothing to wait for."""
