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

"""Time the original PEANUTS on the benchmark scenarios, after checking that it agrees with the tpeanuts reference.

For every scenario PEANUTS can run, and each of its Earth modes, the check grid is evaluated first and compared with the
tpeanuts reference file (``benchmarks.tpeanuts.export_reference``): only a case that agrees within its tolerance is
timed, the others are recorded as ``"disagrees"``, and the scenarios PEANUTS has no equivalent for are recorded as
``"unavailable"``. The results are written in the same schema as the tpeanuts baseline, so the two files are read
together by the report.

PEANUTS runs in its own process and environment (it needs NumPy, Numba, SciPy, pandas and mpmath, not ``torch``), and
this module never imports ``tpeanuts``::

    python -m benchmarks.tpeanuts.export_reference                                   # in the tpeanuts environment
    python -s -m benchmarks.peanuts.run --smoke --label quick                         # in the peanuts environment
    python -s -m benchmarks.peanuts.run --max-points 100000 --repeats 7 --label peanuts_cpu

Its calls are scalar and single-threaded Numba functions inside a Python loop, so its cost grows linearly with the
number of points; ``--max-points`` (20000 by default) keeps the runs short, and the larger grids are recorded as
``"skipped"``.

Module contents:
    TOLERANCES
        The agreement tolerance of each (scenario, mode), with the physical reason where it is above rounding.
    run_benchmarks(...)
        Check and time the scenarios and return the ``ResultSet``.
    main(...)
        Command-line entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.common import environment, results
from benchmarks.common.reference import check_sizes, compare, load_reference, reference_key
from benchmarks.common.scenarios import SCENARIOS, Scenario, energy_grid_MeV, expand_sizes, nadir_grid_rad
from benchmarks.common.timing import time_build_and_evaluate
from benchmarks.peanuts.adapter import PAIRED_METHODS, UNAVAILABLE, Peanuts, git_state

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "benchmarks" / "results"
DEFAULT_REFERENCE = RESULTS_DIR / "reference_tpeanuts.npz"
DEFAULT_MAX_POINTS = 20_000
DEFAULT_MAX_POINTS_NUMERICAL = 1
# (nadir index, energy index) of the check grid at which the numerical mode is compared: a core crossing and an interior
# chord, since it cannot afford the whole grid.
NUMERICAL_CHECK_POINTS = ((0, 0), (2, 5))
_EARTH_SCENARIOS = ("earth", "solar_earth")

# Largest accepted absolute difference of any probability against the tpeanuts reference, from the first comparison of
# the two codes on the check grid (measured differences in brackets, 2026-11-18), each with a margin:
#
# * vacuum (4.5e-8): the same closed formula in both; the difference is rounding and the value of the constants.
# * solar (2.9e-4): the same adiabatic formula on different tabulations of the production profile and the electron
#   density (PEANUTS reads the B16 file directly, tpeanuts its own packaged copy of the tables).
# * earth and solar_earth, analytical (2.1e-2 and 1.6e-2): both are perturbative evolutors, with different expansions,
#   and each is itself 1.7e-2 to 2.6e-2 away from the exact numerical solution at the worst point (the core-crossing
#   chord, at MeV energies, where the oscillation in the Earth is fastest), so two of them cannot be compared more
#   tightly than that.
# * earth and solar_earth, numerical (2.9e-2 and 1.0e-2, at two points): two exact integrations that should agree to
#   far better than this. tpeanuts is converged (400 against 1600 chord steps: 4.5e-4), and the remaining 2.5e-2 at the
#   core-crossing chord is attributed to the Earth density: PEANUTS uses the five-layer polynomial of hep-ph/9702343,
#   tpeanuts the tabulated PREM. This was not tested by giving both the same density.
TOLERANCES: Dict[Tuple[str, Optional[str]], float] = {
    ("vacuum", None): 1.0e-6,
    ("solar", None): 1.0e-3,
    ("earth", "analytical"): 5.0e-2,
    ("solar_earth", "analytical"): 5.0e-2,
    ("earth", "numerical"): 5.0e-2,
    ("solar_earth", "numerical"): 5.0e-2,
}
DEFAULT_TOLERANCE = 1.0e-3


def run_benchmarks(
    scenarios: Sequence[str],
    *,
    peanuts: Peanuts,
    reference: Dict[str, np.ndarray],
    modes: Sequence[str] = ("analytical",),
    repeats: int = 7,
    smoke: bool = False,
    max_points: int = DEFAULT_MAX_POINTS,
    max_points_numerical: int = DEFAULT_MAX_POINTS_NUMERICAL,
    label: str = "peanuts",
    reference_meta: Optional[Dict[str, Any]] = None,
) -> results.ResultSet:
    """Check each case against the reference and time the ones that agree.

    Args:
        scenarios: Names in ``benchmarks.common.scenarios.SCENARIOS``.
        peanuts: The loaded original package.
        reference: The arrays of the tpeanuts reference file.
        modes: PEANUTS Earth modes to run (``"analytical"``, ``"numerical"``). The numerical one costs about a hundred
            seconds per point, so it is opt-in and has its own, much lower, point limit.
        repeats: Timed repeats per phase (the protocol value is 7).
        smoke: Use the reduced sizes.
        max_points: Skip the cases with more (energy x nadir) points than this; 0 means no limit.
        max_points_numerical: The same limit for the numerical mode.
        label: Name of the run.
        reference_meta: Metadata of the reference file, recorded in the environment of the result.

    Returns:
        The validated result set.

    Raises:
        ValueError: If a scenario or a mode name is unknown.
    """
    unknown = [name for name in scenarios if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios {unknown}; expected a subset of {sorted(SCENARIOS)}.")
    bad_modes = [name for name in modes if name not in PAIRED_METHODS]
    if bad_modes:
        raise ValueError(f"unknown modes {bad_modes}; expected a subset of {sorted(PAIRED_METHODS)}.")
    environment_record = environment.capture_environment(packages=("numpy", "numba", "scipy", "pandas", "mpmath"))
    environment_record["peanuts"] = git_state(peanuts.directory)
    environment_record["reference"] = {
        key: (reference_meta or {}).get(key) for key in ("backend", "dtype", "nsteps", "oscillation")
    }

    records: List[results.BenchmarkRecord] = []
    for name in scenarios:
        scenario = SCENARIOS[name]
        scenario_modes: List[Optional[str]] = list(modes) if name in _EARTH_SCENARIOS else [None]
        for mode in scenario_modes:
            records.extend(
                _run_scenario(
                    scenario,
                    mode,
                    peanuts=peanuts,
                    reference=reference,
                    repeats=repeats,
                    smoke=smoke,
                    max_points=max_points_numerical if mode == "numerical" else max_points,
                )
            )
    result_set = results.ResultSet(label=label, environment=environment_record, records=records)
    result_set.validate()
    return result_set


def _grid(scenario: Scenario, sizes: Dict[str, int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The energy and nadir grids of a case, in MeV and radians."""
    energy = energy_grid_MeV(scenario.energy_kind, sizes["n_energy"])
    nadir = nadir_grid_rad(sizes["n_nadir"]) if "n_nadir" in sizes else None
    return energy, nadir


def _check_arrays(
    peanuts: Peanuts, scenario: Scenario, mode: Optional[str], check: Dict[str, int], reference: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """PEANUTS and the reference at the points of the check: the whole check grid, or a few of its points.

    The numerical mode takes about a hundred seconds per point, so it is checked at ``NUMERICAL_CHECK_POINTS`` only
    (a core-crossing one and an interior one); every other mode is checked on the whole grid.
    """
    energy, nadir = _grid(scenario, check)
    if mode != "numerical":
        return peanuts.evaluate(scenario.name, energy, nadir, mode), reference
    candidate, expected = [], []
    for j, k in NUMERICAL_CHECK_POINTS:
        candidate.append(peanuts.evaluate(scenario.name, energy[[k]], nadir[[j]], mode)[0, 0])
        expected.append(reference[j, k])
    return np.stack(candidate), np.stack(expected)


def _run_scenario(
    scenario: Scenario,
    mode: Optional[str],
    *,
    peanuts: Peanuts,
    reference: Dict[str, np.ndarray],
    repeats: int,
    smoke: bool,
    max_points: int,
) -> List[results.BenchmarkRecord]:
    """The records of one (scenario, mode): unavailable, disagreeing, or the build and evaluate timings per size."""
    options = {"backend_mode": mode, "paired_method": PAIRED_METHODS.get(mode) if mode else None}
    sizes_list = list(expand_sizes(scenario, smoke=smoke))

    def record(sizes: Dict[str, int], phase: str, **fields: Any) -> results.BenchmarkRecord:
        return results.BenchmarkRecord(
            scenario=scenario.name,
            backend="peanuts",
            phase=phase,
            device="cpu",
            dtype="float64",
            size=dict(sizes),
            options=dict(options),
            **fields,
        )

    every_phase = results.PHASES
    if scenario.name in UNAVAILABLE:
        note = UNAVAILABLE[scenario.name]
        return [record(s, p, status="unavailable", note=note) for s in sizes_list for p in every_phase]

    # Check first: the check grid against the tpeanuts reference.
    check = check_sizes(scenario.name)
    key = reference_key(scenario.name, PAIRED_METHODS[mode] if mode else None)
    tolerance = TOLERANCES.get((scenario.name, mode), DEFAULT_TOLERANCE)
    agreement = compare(*_check_arrays(peanuts, scenario, mode, check, reference[key]), tolerance)
    if not agreement["passed"]:
        note = (
            f"disagrees with the tpeanuts reference ({agreement['metric']} {agreement['value']:.3g} > "
            f"{tolerance:.3g}); not timed"
        )
        return [
            record(s, p, status="disagrees", agreement=agreement, note=note) for s in sizes_list for p in every_phase
        ]

    out: List[results.BenchmarkRecord] = []
    for sizes in sizes_list:
        points = sizes["n_energy"] * sizes.get("n_nadir", 1)
        if max_points and points > max_points:
            note = f"skipped: {points} points exceed --max-points={max_points}"
            out.extend(record(sizes, p, status="skipped", note=note) for p in every_phase)
            continue
        energy, nadir = _grid(scenario, sizes)

        def evaluate(loaded: Peanuts, energy=energy, nadir=nadir) -> np.ndarray:
            return loaded.evaluate(scenario.name, energy, nadir, mode)

        try:
            build_timing, evaluate_timing = time_build_and_evaluate(
                lambda: Peanuts(peanuts.directory), evaluate, n_repeats=repeats
            )
        except (MemoryError, ValueError, RuntimeError) as error:
            note = f"failed: {type(error).__name__}: {error}"[:500]
            out.extend(record(sizes, p, status="failed", note=note) for p in every_phase)
            continue
        out.append(record(sizes, "build", timing=build_timing.to_dict(), agreement=agreement))
        out.append(record(sizes, "evaluate", timing=evaluate_timing.to_dict(), agreement=agreement))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: check, time, print a summary and write the JSON under ``benchmarks/results``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS), help="subset of the catalogue")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="tpeanuts reference .npz")
    parser.add_argument("--peanuts-path", type=Path, default=None, help="directory of the original PEANUTS")
    parser.add_argument("--modes", nargs="*", default=["analytical"], choices=sorted(PAIRED_METHODS))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--smoke", action="store_true", help="the reduced sizes of quick runs")
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS, help="skip larger grids; 0 for no limit")
    parser.add_argument("--max-points-numerical", type=int, default=DEFAULT_MAX_POINTS_NUMERICAL)
    parser.add_argument("--label", default="peanuts_cpu_float64")
    parser.add_argument("--output", type=Path, default=None, help=".csv or .json (default: results/<label>.json)")
    results.add_spec_arguments(parser)
    args = parser.parse_args(argv)

    reference, meta = load_reference(args.reference)
    result_set = run_benchmarks(
        args.scenarios,
        peanuts=Peanuts(args.peanuts_path),
        reference=reference,
        modes=args.modes,
        repeats=args.repeats,
        smoke=args.smoke,
        max_points=args.max_points,
        max_points_numerical=args.max_points_numerical,
        label=args.label,
        reference_meta=meta,
    )
    for entry in result_set.records:
        if entry.phase != "evaluate":
            continue
        shown = ", ".join(f"{key}={value}" for key, value in entry.size.items())
        mode = entry.options.get("backend_mode") or "-"
        if entry.status == "ok" and entry.timing is not None:
            print(f"{entry.scenario:12s} {mode:11s} {shown:28s} min {entry.timing['min_s'] * 1e3:10.2f} ms")
        else:
            print(f"{entry.scenario:12s} {mode:11s} {shown:28s} {entry.status}: {entry.note}")
    path = results.save(
        result_set,
        args.output or RESULTS_DIR / f"{args.label}.json",
        spec_hash=args.spec_hash,
        spec_id=args.spec_id,
        smoke=args.smoke,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
