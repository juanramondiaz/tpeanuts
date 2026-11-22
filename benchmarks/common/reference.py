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

"""Reference probabilities exchanged between backends, and the agreement check that gates every timing.

The protocol records a timing only for a case whose result agrees with the reference. Each backend runs in its own
process and environment, so the reference travels as a file: tpeanuts evaluates every scenario on a small *check grid*
and writes the probabilities to an ``.npz``; a reference backend (the original peanuts, nuSQuIDS) loads it, evaluates
the same grid with its own code, and compares with ``compare`` before it times anything. This module depends on
``numpy`` only.

The tolerances are per scenario and measured, not wished: each is the largest difference seen between the two
implementations when they were first compared, with a margin, and the physical reason for any difference above rounding
is stated next to it. A tolerance is a statement about two codes agreeing, so it is raised only with that reason.

Module contents:
    CHECK_SIZES
        The sizes of the check grid (small, so it costs nothing next to the timings).
    check_sizes(...)
        The check-grid sizes of one scenario.
    reference_key(...)
        The key of a (scenario, method) array in a reference file.
    save_reference(...), load_reference(...)
        Write and read a reference file: the arrays and a metadata dictionary.
    compare(...)
        The agreement record of two probability arrays, in the schema of ``results.BenchmarkRecord.agreement``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from benchmarks.common.scenarios import SCENARIOS

CHECK_SIZES = {"n_energy": 12, "n_nadir": 5}

_META_KEY = "_meta"
METRIC = "max_abs_probability_difference"


def check_sizes(scenario: str) -> Dict[str, int]:
    """The check-grid sizes of a scenario: ``CHECK_SIZES`` restricted to the axes the scenario has.

    Args:
        scenario: A name in ``scenarios.SCENARIOS``.

    Returns:
        ``{axis: size}`` for the axes of that scenario.

    Raises:
        KeyError: If the scenario is unknown.
    """
    return {axis: CHECK_SIZES[axis] for axis in SCENARIOS[scenario].sizes}


def reference_key(scenario: str, method: Optional[str] = None) -> str:
    """The key of an array in a reference file: ``scenario``, or ``scenario/method`` for method-dependent scenarios.

    Args:
        scenario: Scenario name.
        method: Propagation method name (``"numerical"``, ``"analytic_perturbative"``...), if the scenario has one.

    Returns:
        The key.
    """
    return scenario if method is None else f"{scenario}/{method}"


def save_reference(path: Union[str, Path], arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Path:
    """Write reference probabilities and their metadata to an ``.npz``.

    Args:
        path: Destination; its directory is created if needed.
        arrays: Key (see ``reference_key``) mapped to the probabilities, shaped ``(n_nadir, n_energy, 3)`` or
            ``(n_energy, 3)``.
        meta: JSON-serialisable description (backend, precision, check sizes, discretisation, environment).

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
    np.savez(target, **payload, **{_META_KEY: np.array(json.dumps(meta, sort_keys=True))})
    return target


def load_reference(path: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Read a reference file written by ``save_reference``.

    Args:
        path: The ``.npz`` file.

    Returns:
        ``(arrays, meta)``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != _META_KEY}
        meta = json.loads(str(data[_META_KEY])) if _META_KEY in data.files else {}
    return arrays, meta


def compare(candidate: np.ndarray, reference: np.ndarray, tolerance: float) -> Dict[str, Any]:
    """The agreement record of two probability arrays.

    Args:
        candidate: Probabilities of the backend being timed.
        reference: Probabilities it must agree with, of the same shape.
        tolerance: The largest accepted absolute difference of any probability.

    Returns:
        ``{"metric", "value", "tolerance", "passed"}``; ``passed`` is false if the shapes differ, any value is not
        finite, or the largest absolute difference exceeds ``tolerance``.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or not np.isfinite(candidate).all():
        return {"metric": METRIC, "value": float("inf"), "tolerance": tolerance, "passed": False}
    value = float(np.max(np.abs(candidate - reference)))
    return {"metric": METRIC, "value": value, "tolerance": tolerance, "passed": bool(value <= tolerance)}
