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

"""The common structure of the validation notebooks: cases with a declared tolerance, and residual statistics.

Every validation notebook compares a computed quantity with a reference (an exact identity, an analytic expression, a
stored result, another code) and reports how far apart they are. This module fixes how that is described and shown, so
that the notebooks of the validation series read the same way: a table of the cases with the tolerance of each and the
reason for it, the statistics of the residual of each case (maximum, 99th percentile, median and root mean square, the
maximum alone being dominated by a single point), the same statistics by region of the grid, a map of the residual, and
the check that the maximum is below the tolerance. Only presentation and bookkeeping live here; the physical
calculations of a notebook go through the tpeanuts API.

Module contents:
    ValidationCase
        One comparison: name, description, tolerance and the cause that justifies the tolerance.
    cases_table(...)
        The cases of a notebook as a table (name, what is compared, tolerance, cause).
    residual_statistics(...)
        Maximum, 99th percentile, median and root mean square of a residual, overall and by region.
    record_case(...)
        Record in a ``CheckLog`` whether the largest residual of a case is below its tolerance.
    plot_residual_map(...)
        Draw a residual over a two-axis grid on a logarithmic colour scale.
    worst_component(...)
        Largest absolute value over the last axis of a difference (the components of a probability).
    largest_difference(...)
        Largest absolute difference of two arrays over their trailing axes (the components of a probability).
    axis_regions(...)
        Masks of the regions of a grid defined by thresholds on its axes (low and high energy, mantle and core).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from tpeanuts.notebooks.notebooks_helper import CheckLog, to_numpy

RESIDUAL_FLOOR = 1.0e-18


@dataclass(frozen=True)
class ValidationCase:
    """One comparison of a validation notebook.

    Attributes:
        name: Short name, used as the label of the check and of the table row.
        description: What is compared with what.
        tolerance: Largest acceptable residual, in the units of the residual.
        cause: Why the tolerance has this value (rounding of a precision, a discretisation, a stated approximation).
    """

    name: str
    description: str
    tolerance: float
    cause: str


def cases_table(cases: Sequence[ValidationCase]) -> pd.DataFrame:
    """The cases of a notebook as a table.

    Args:
        cases: The cases, in the order of the notebook.

    Returns:
        A table with the columns ``Case``, ``Compared``, ``Tolerance`` and ``Cause`` (as text, one row per case).
    """
    return pd.DataFrame(
        [(case.name, case.description, f"{case.tolerance:.0e}", case.cause) for case in cases],
        columns=["Case", "Compared", "Tolerance", "Cause"],
    )


def worst_component(difference: Any) -> np.ndarray:
    """Largest absolute value over the last axis of a difference, the residual of a probability.

    Args:
        difference: Array whose last axis holds the components (the final flavours).

    Returns:
        Array with the last axis reduced.
    """
    return np.abs(to_numpy(difference).astype(float)).max(axis=-1)


def largest_difference(computed: Any, reference: Any, keep_axes: int) -> np.ndarray:
    """Largest absolute difference of two arrays over their trailing axes.

    The residual of a probability is the largest difference of its components, so the leading axes are the grid
    (energy, angle, initial flavour) and the trailing ones the components.

    Args:
        computed: The computed array.
        reference: The reference, the same shape.
        keep_axes: Number of leading axes that are kept; the rest are reduced with the maximum.

    Returns:
        Array shaped like the first ``keep_axes`` axes.
    """
    difference = np.abs(to_numpy(computed).astype(float) - to_numpy(reference).astype(float))
    return difference.reshape(difference.shape[:keep_axes] + (-1,)).max(axis=-1)


def axis_regions(shape: Sequence[int], *splits: tuple) -> dict:
    """Boolean masks of the regions of a grid, each defined by a threshold on one of its axes.

    Args:
        shape: Shape of the residual.
        *splits: One tuple ``(axis, values, threshold, label_below, label_above)`` per split: the points of ``axis``
            whose value in ``values`` is below the threshold form the region ``label_below`` and the others
            ``label_above``. The regions of the splits are returned in the order given, below before above.

    Returns:
        Dict from label to a mask of the shape of the residual.
    """
    regions = {}
    for axis, values, threshold, label_below, label_above in splits:
        below = np.asarray(values) < threshold
        layout = [1] * len(shape)
        layout[axis] = len(below)
        regions[label_below] = np.broadcast_to(below.reshape(layout), tuple(shape))
        regions[label_above] = np.broadcast_to((~below).reshape(layout), tuple(shape))
    return regions


def residual_statistics(
    residual: Any, regions: Optional[Mapping[str, Any]] = None, *, label: str = "all"
) -> pd.DataFrame:
    """Statistics of the absolute value of a residual, overall and by region.

    Args:
        residual: The residual, any shape; only its absolute value is used.
        regions: Optional boolean masks with the shape of ``residual``, by region name (e.g. mantle, core). A region
            with no point is reported with ``NaN`` statistics.
        label: Name of the row that holds the statistics of the whole residual.

    Returns:
        A table with one row per region, the first one being ``label`` for all the points, and the columns ``max``,
        ``p99``, ``median``, ``rms`` and ``n`` (number of points).
    """
    values = np.abs(to_numpy(residual).astype(float))
    groups = {label: np.ones(values.shape, dtype=bool)}
    for name, mask in (regions or {}).items():
        mask = to_numpy(mask).astype(bool)
        if mask.shape != values.shape:
            raise ValueError(f"the mask of region {name!r} has shape {mask.shape}, not {values.shape}")
        groups[name] = mask
    rows = []
    for name, mask in groups.items():
        chosen = values[mask]
        if chosen.size == 0:
            rows.append((name, np.nan, np.nan, np.nan, np.nan, 0))
        else:
            rms = np.sqrt(np.mean(chosen**2))
            rows.append((name, chosen.max(), np.percentile(chosen, 99.0), np.median(chosen), rms, chosen.size))
    return pd.DataFrame(rows, columns=["region", "max", "p99", "median", "rms", "n"]).set_index("region")


def record_case(checks: CheckLog, case: ValidationCase, residual: Any) -> bool:
    """Record whether the largest residual of a case is below its tolerance.

    Args:
        checks: The log of the notebook.
        case: The case, whose name, tolerance and cause are written to the check.
        residual: The residual of the case, any shape.

    Returns:
        ``True`` if the maximum of ``|residual|`` is below ``case.tolerance``.
    """
    worst = float(np.max(np.abs(to_numpy(residual).astype(float)))) if np.size(to_numpy(residual)) else 0.0
    checks.details[case.name] = {
        "worst": worst, "tolerance": case.tolerance, "cause": case.cause, "description": case.description,
    }
    return checks.record(case.name, f"{worst:.1e}", f"< {case.tolerance:.0e} ({case.cause})", worst < case.tolerance)


def plot_residual_map(
    ax: Any, x: Any, y: Any, residual: Any, *, xlabel: str, ylabel: str, title: str, log_x: bool = False,
    vmin: Optional[float] = None, vmax: Optional[float] = None,
) -> Any:
    """Draw ``|residual|`` over a grid on a logarithmic colour scale.

    Args:
        ax: The matplotlib axis.
        x: Coordinates of the columns, length ``residual.shape[1]``.
        y: Coordinates of the rows, length ``residual.shape[0]``.
        residual: The residual, shaped ``(len(y), len(x))``.
        xlabel: Label of the horizontal axis.
        ylabel: Label of the vertical axis.
        title: Title of the panel.
        log_x: ``True`` for a logarithmic horizontal axis (energies).
        vmin: Lower end of the colour scale; defaults to the smallest non-zero residual.
        vmax: Upper end of the colour scale; defaults to the largest residual.

    Returns:
        The mesh, so that the caller can add a colour bar.
    """
    values = np.maximum(np.abs(to_numpy(residual).astype(float)), RESIDUAL_FLOOR)
    low = vmin if vmin is not None else max(float(values.min()), RESIDUAL_FLOOR)
    high = vmax if vmax is not None else max(float(values.max()), low * 10.0)
    mesh = ax.pcolormesh(to_numpy(x), to_numpy(y), values, norm=LogNorm(vmin=low, vmax=high), shading="auto")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return mesh
