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

"""Backend-agnostic grid comparison, error metrics, and CSV export.

Imports nothing from ``tpeanuts`` or nuSQuIDS -- these utilities take
two arbitrary callables, each mapping one grid point's coordinates to
a probability vector, and compare them. Which of the two is the
"reference" (typically nuSQuIDS) and which is the "candidate"
(typically tpeanuts) is the caller's own convention; this module does
not know or care.

Uses ``csv.writer`` for export, not pandas -- this project's own
established convention throughout ``tpeanuts`` itself, kept here too
for consistency even though this package sits outside it.

Module contents:
    ComparisonResult
        Grid coordinates plus two backends' own probability arrays,
        with error metrics as properties.
    compare_grid(...)
        Evaluate two probability functions over a grid and assemble a
        ``ComparisonResult``.
    summarize_errors(...)
        Max/RMS absolute and relative error, plus the grid point where
        the max absolute error occurs.
    check_unitarity(...)
        Per-point ``|sum(probabilities) - 1|`` deviation, for either
        backend's own array.
    export_csv(...)
        Write a ``ComparisonResult`` to a CSV file, one row per grid
        point.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple, Union

import numpy as np

GridPoint = Mapping[str, float]
ProbabilityFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class ComparisonResult:
    """One backend-vs-backend comparison over a grid of points.

    Attributes:
        coords: One coordinate mapping per grid point, e.g.
            ``{"E_GeV": 1.0, "baseline_km": 100.0}``, length ``n_points``.
        labels: Component names of the probability vectors, e.g.
            ``("P_nue", "P_numu", "P_nutau")``, length ``n_components``.
        reference: Reference backend's own probabilities, shape
            ``(n_points, n_components)``.
        candidate: Candidate backend's own probabilities, same shape.

    Raises:
        ValueError: If ``reference``/``candidate`` do not share the
            same shape, or that shape does not match
            ``(len(coords), len(labels))``.
    """

    coords: Tuple[Dict[str, float], ...]
    labels: Tuple[str, ...]
    reference: np.ndarray
    candidate: np.ndarray

    def __post_init__(self) -> None:
        expected_shape = (len(self.coords), len(self.labels))
        if self.reference.shape != self.candidate.shape:
            raise ValueError(
                f"reference and candidate must share the same shape; got "
                f"{self.reference.shape} and {self.candidate.shape}."
            )
        if self.reference.shape != expected_shape:
            raise ValueError(
                f"reference/candidate must have shape {expected_shape} "
                f"(len(coords), len(labels)); got {self.reference.shape}."
            )

    @property
    def abs_error(self) -> np.ndarray:
        """``|candidate - reference|``, shape ``(n_points, n_components)``."""
        return np.abs(self.candidate - self.reference)

    @property
    def rel_error(self) -> np.ndarray:
        """``|candidate - reference| / max(|reference|, eps)``, same shape as ``abs_error``.

        The ``eps`` floor (``1e-12``) avoids a spurious ``inf``/``nan``
        wherever the reference probability is exactly (or near) zero --
        a near-zero probability's *absolute* error is the physically
        meaningful quantity there anyway, not a blown-up relative one.
        """
        denom = np.clip(np.abs(self.reference), 1.0e-12, None)
        return self.abs_error / denom


def compare_grid(
    grid_points: Sequence[GridPoint],
    reference_fn: ProbabilityFn,
    candidate_fn: ProbabilityFn,
    *,
    labels: Sequence[str],
) -> ComparisonResult:
    """Evaluate two probability functions over a grid and assemble a ``ComparisonResult``.

    Args:
        grid_points: One coordinate mapping per grid point (see
            ``ComparisonResult.coords``).
        reference_fn: Called as ``reference_fn(**point)`` for each
            ``point`` in ``grid_points``; must return a 1-D array of
            length ``len(labels)``.
        candidate_fn: Same calling convention as ``reference_fn``, the
            backend being validated against it.
        labels: Component names of the returned probability vectors.

    Returns:
        A ``ComparisonResult`` with one row per grid point.

    Raises:
        ValueError: If either function's return value does not have
            shape ``(len(labels),)`` for some grid point.
    """
    reference_rows = []
    candidate_rows = []
    for point in grid_points:
        ref = np.asarray(reference_fn(**point), dtype=float)
        cand = np.asarray(candidate_fn(**point), dtype=float)
        if ref.shape != (len(labels),) or cand.shape != (len(labels),):
            raise ValueError(
                f"reference_fn/candidate_fn must each return shape ({len(labels)},) "
                f"for point {dict(point)!r}; got {ref.shape} and {cand.shape}."
            )
        reference_rows.append(ref)
        candidate_rows.append(cand)

    return ComparisonResult(
        coords=tuple(dict(p) for p in grid_points),
        labels=tuple(labels),
        reference=np.stack(reference_rows, axis=0),
        candidate=np.stack(candidate_rows, axis=0),
    )


def summarize_errors(result: ComparisonResult) -> Dict[str, Union[float, Dict[str, float]]]:
    """Summarize a ``ComparisonResult``'s errors into a few headline numbers.

    Args:
        result: A ``ComparisonResult``.

    Returns:
        ``{"max_abs_error": ..., "rms_abs_error": ..., "max_rel_error": ...,
        "rms_rel_error": ..., "argmax_abs_error_point": {...}}`` -- the
        last entry is the grid point (from ``result.coords``) where
        the single largest absolute-error component occurs.
    """
    abs_error = result.abs_error
    rel_error = result.rel_error
    flat_argmax = int(np.argmax(abs_error))
    point_idx = flat_argmax // abs_error.shape[1]

    return {
        "max_abs_error": float(np.max(abs_error)),
        "rms_abs_error": float(np.sqrt(np.mean(abs_error**2))),
        "max_rel_error": float(np.max(rel_error)),
        "rms_rel_error": float(np.sqrt(np.mean(rel_error**2))),
        "argmax_abs_error_point": dict(result.coords[point_idx]),
    }


def check_unitarity(probabilities: np.ndarray) -> np.ndarray:
    """Per-point ``|sum(probabilities) - 1|`` deviation from unit normalization.

    Args:
        probabilities: Shape ``(n_points, n_components)`` (e.g.
            ``result.reference`` or ``result.candidate``).

    Returns:
        Real array shape ``(n_points,)``, the absolute deviation of
        each row's sum from 1. Every entry should be at or below the
        caller's own numerical tolerance for a properly normalized
        propagation (both backends are expected to conserve total
        probability).
    """
    return np.abs(np.sum(probabilities, axis=-1) - 1.0)


def export_csv(result: ComparisonResult, path: Union[str, Path]) -> None:
    """Write a ``ComparisonResult`` to a CSV file, one row per grid point.

    Columns: every coordinate key (in the first point's own order),
    then ``{label}_reference``/``{label}_candidate``/``{label}_abs_error``/
    ``{label}_rel_error`` per component.

    Args:
        result: A ``ComparisonResult``.
        path: Destination CSV path; parent directories are not created.
    """
    coord_keys = list(result.coords[0].keys()) if result.coords else []
    fieldnames = list(coord_keys)
    for label in result.labels:
        fieldnames += [f"{label}_reference", f"{label}_candidate", f"{label}_abs_error", f"{label}_rel_error"]

    abs_error = result.abs_error
    rel_error = result.rel_error

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for i, point in enumerate(result.coords):
            row = [point[key] for key in coord_keys]
            for j in range(len(result.labels)):
                row += [
                    result.reference[i, j], result.candidate[i, j],
                    abs_error[i, j], rel_error[i, j],
                ]
            writer.writerow(row)
