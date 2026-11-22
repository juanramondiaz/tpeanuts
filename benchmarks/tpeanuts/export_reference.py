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

"""Write the tpeanuts reference probabilities that the reference backends must agree with before they are timed.

For every scenario, and for each propagation method the scenario depends on, tpeanuts evaluates the check grid
(``benchmarks.common.reference.CHECK_SIZES``) in float64 on the CPU and the probabilities are saved to an ``.npz``. The
file is read by ``benchmarks.peanuts.run`` and ``benchmarks.nusquids.run``, each from its own process and environment.

The reference is tpeanuts's own result, and the question being asked is whether another code reproduces it: a
disagreement does not say which of the two is right. That is settled by the validation, which compares each against
its own independent check (``validation/``), and the benchmark only refuses to time what does not agree.

Usage, from the repository root::

    python -m benchmarks.tpeanuts.export_reference                       # benchmarks/results/reference_tpeanuts.npz
    python -m benchmarks.tpeanuts.export_reference --nsteps 160 --output elsewhere.npz

nuSQuIDS has one fixed atmosphere (a production height of 22 km), unlike the catalogue's atmosphere scenario, which
averages four heights; the file therefore also holds the atmosphere computed with that single height, under the
``atmosphere_earthatm`` key, which is what ``benchmarks.nusquids.run`` is checked against.

Module contents:
    METHODS_BY_SCENARIO
        The methods each scenario is exported with.
    EARTHATM_KEY, EARTHATM_HEIGHT_M, EARTHATM_METHODS
        The single-height atmosphere reference that nuSQuIDS is compared with.
    compute_reference(...)
        The arrays and metadata of a reference file.
    main(...)
        Command-line entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from benchmarks.common import environment
from benchmarks.common.reference import check_sizes, reference_key, save_reference
from benchmarks.common.scenarios import OSCILLATION
from benchmarks.tpeanuts.cases import DEFAULT_NSTEPS, METHODS, build_case, make_context
from benchmarks.tpeanuts.run import RESULTS_DIR

METHODS_BY_SCENARIO: Dict[str, Tuple[Optional[str], ...]] = {
    "vacuum": (None,),
    "solar": (None,),
    "earth": ("numerical", "analytic_perturbative"),
    "solar_earth": ("numerical", "analytic_perturbative"),
    "atmosphere": ("numerical", "analytic_perturbative"),
}

EARTHATM_KEY = "atmosphere_earthatm"
EARTHATM_HEIGHT_M = 22000.0  # validation.nusquids.adapter.NUSQUIDS_EARTHATM_HEIGHT_M (tested to be equal)
EARTHATM_METHODS = ("numerical",)

DEFAULT_OUTPUT = RESULTS_DIR / "reference_tpeanuts.npz"


def compute_reference(*, nsteps: int = DEFAULT_NSTEPS) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Evaluate every scenario on its check grid, in float64 on the CPU.

    Args:
        nsteps: Chord discretisation of the Earth and atmosphere legs.

    Returns:
        ``(arrays, meta)``: the probabilities by ``reference_key`` and the description of how they were computed.
    """
    context = make_context("cpu", "float64")
    arrays: Dict[str, np.ndarray] = {}
    for scenario, methods in METHODS_BY_SCENARIO.items():
        sizes = check_sizes(scenario)
        for method in methods:
            case = build_case(scenario, sizes, context=context, method=METHODS[method or "numerical"], nsteps=nsteps)
            arrays[reference_key(scenario, method)] = case.probabilities(case.build()).detach().cpu().numpy()
    for method in EARTHATM_METHODS:
        case = build_case(
            "atmosphere",
            check_sizes("atmosphere"),
            context=context,
            method=METHODS[method],
            nsteps=nsteps,
            heights_m=(EARTHATM_HEIGHT_M,),
        )
        arrays[reference_key(EARTHATM_KEY, method)] = case.probabilities(case.build()).detach().cpu().numpy()
    meta = {
        "backend": "tpeanuts",
        "device": "cpu",
        "dtype": "float64",
        "nsteps": nsteps,
        "check_sizes": {scenario: check_sizes(scenario) for scenario in METHODS_BY_SCENARIO},
        "oscillation": OSCILLATION,
        "environment": environment.capture_environment(packages=("numpy", "torch", "tpeanuts")),
    }
    return arrays, meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: compute the reference and write it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nsteps", type=int, default=DEFAULT_NSTEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    arrays, meta = compute_reference(nsteps=args.nsteps)
    path = save_reference(args.output, arrays, meta)
    print(f"wrote {len(arrays)} reference arrays to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
