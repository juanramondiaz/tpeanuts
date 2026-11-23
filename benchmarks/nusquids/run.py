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

"""Time nuSQuIDS on the benchmark scenarios, after checking that it agrees with the tpeanuts reference.

For each scenario nuSQuIDS can run, the check grid is evaluated first and compared with the tpeanuts reference file
(``benchmarks.tpeanuts.export_reference``): only a case that agrees within its tolerance is timed, the others are
recorded as ``"disagrees"``, and the scenarios without a nuSQuIDS equivalent are recorded as ``"unavailable"``. The
results are written in the same schema as the other backends, so the report reads them together.

nuSQuIDS runs in its own process and environment (the WSL virtual environment, which has no ``torch`` requirement here),
and this module never imports ``tpeanuts``. The repository lives on the Windows drive and the run on the native WSL
copy, which ``benchmarks.nusquids.sync_wsl`` keeps up to date and fingerprints::

    python -m benchmarks.tpeanuts.export_reference                                   # tpeanuts environment, Windows
    python -m benchmarks.nusquids.sync_wsl                                           # copy the files to WSL
    python -m benchmarks.nusquids.run --smoke --label quick                          # in the nuSQuIDS environment

Its calls evolve one energy at a time, single-threaded, so the cost grows linearly with the number of points;
``--max-points`` (2000 by default) keeps the runs short, and the larger grids are recorded as ``"skipped"``.

Module contents:
    TOLERANCES
        The agreement tolerance of each scenario, with the physical reason where it is above the solver precision.
    REFERENCE_KEYS
        The array of the reference file each scenario is checked against.
    run_benchmarks(...)
        Check and time the scenarios and return the ``ResultSet``.
    parse_scenario_limits(...)
        Read the ``SCENARIO=LIMIT`` items of ``--max-points-scenario``.
    main(...)
        Command-line entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from benchmarks.common import environment, results
from benchmarks.common.reference import check_sizes, compare, load_reference
from benchmarks.common.scenarios import (
    METHOD_SCENARIOS,
    SCENARIOS,
    Scenario,
    energy_grid_MeV,
    expand_sizes,
    nadir_grid_rad,
)
from benchmarks.common.timing import time_build_and_evaluate
from benchmarks.nusquids.adapter import PAIRED_METHOD, UNAVAILABLE, NuSQuIDSBackend, configuration, describe

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "benchmarks" / "results"
DEFAULT_REFERENCE = RESULTS_DIR / "reference_tpeanuts.npz"
DEFAULT_MAX_POINTS = 2_000
DEFAULT_THREADS = 1

# The key of ``benchmarks.tpeanuts.export_reference.EARTHATM_KEY`` (a test checks they are the same).
REFERENCE_KEYS = {
    "vacuum": "vacuum",
    "earth": f"earth/{PAIRED_METHOD}",
    "atmosphere": f"atmosphere_earthatm/{PAIRED_METHOD}",
}

# Largest accepted absolute difference of any probability against the tpeanuts reference. Both codes integrate the
# evolution equation over a PREM Earth, yet differ by up to 1.8e-2 (earth) and 1.5e-2 (atmosphere), always largest for
# chords near the core. It is not the discretisation (tpeanuts at 1600 steps still differs by 1.7e-2 and 1.1e-2), nor
# the solver reuse (it equals the independent per-call adapter exactly). The two PREM tables differ, e.g.
# in the core electron fraction (0.4656 in nuSQuIDS, 0.467 in tpeanuts); that attribution has not been isolated.
# Vacuum is a closed form and agrees to rounding.
TOLERANCES: Dict[str, float] = {"vacuum": 1.0e-6, "earth": 3.0e-2, "atmosphere": 2.5e-2}
DEFAULT_TOLERANCE = 1.0e-3


def run_benchmarks(
    scenarios: Sequence[str],
    *,
    reference: Dict[str, np.ndarray],
    repeats: int = 7,
    smoke: bool = False,
    max_points: int = DEFAULT_MAX_POINTS,
    max_points_by_scenario: Optional[Dict[str, int]] = None,
    label: str = "nusquids",
    reference_meta: Optional[Dict[str, Any]] = None,
    backend_factory: Any = NuSQuIDSBackend,
) -> results.ResultSet:
    """Check each case against the reference and time the ones that agree.

    Args:
        scenarios: Names in ``benchmarks.common.scenarios.SCENARIOS``.
        reference: The arrays of the tpeanuts reference file.
        repeats: Timed repeats per phase (the protocol value is 7).
        smoke: Use the reduced sizes.
        max_points: Skip the cases with more (energy x nadir) points than this; 0 means no limit.
        max_points_by_scenario: Own limit of the scenarios named here, instead of ``max_points``. The cost per point
            differs by orders of magnitude (the Earth at MeV against the atmosphere at GeV), so one limit cannot
            suit both.
        label: Name of the run.
        reference_meta: Metadata of the reference file, recorded in the environment of the result.
        backend_factory: ``NuSQuIDSBackend``, or a stand-in with the same constructor and ``evaluate``.

    Returns:
        The validated result set.

    Raises:
        ValueError: If a scenario name is unknown, in ``scenarios`` or in ``max_points_by_scenario``.
    """
    limits = dict(max_points_by_scenario or {})
    unknown = [name for name in (*scenarios, *limits) if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios {unknown}; expected a subset of {sorted(SCENARIOS)}.")
    environment_record = environment.capture_environment(packages=("numpy", "nusquids"))
    environment_record["nusquids_solver"] = describe(configuration())
    environment_record["reference"] = {
        key: (reference_meta or {}).get(key) for key in ("backend", "dtype", "nsteps", "oscillation")
    }
    records: List[results.BenchmarkRecord] = []
    for name in scenarios:
        records.extend(
            _run_scenario(
                SCENARIOS[name],
                reference=reference,
                repeats=repeats,
                smoke=smoke,
                max_points=limits.get(name, max_points),
                backend_factory=backend_factory,
            )
        )
    result_set = results.ResultSet(label=label, environment=environment_record, records=records)
    result_set.validate()
    return result_set


def _grid(scenario: Scenario, sizes: Dict[str, int]) -> Any:
    energy = energy_grid_MeV(scenario.energy_kind, sizes["n_energy"])
    nadir = nadir_grid_rad(sizes["n_nadir"]) if "n_nadir" in sizes else None
    return energy, nadir


def _run_scenario(
    scenario: Scenario,
    *,
    reference: Dict[str, np.ndarray],
    repeats: int,
    smoke: bool,
    max_points: int,
    backend_factory: Any,
) -> List[results.BenchmarkRecord]:
    """The records of one scenario: unavailable, disagreeing, or the build and evaluate timings per size."""
    options = {"backend_mode": "solver", "paired_method": PAIRED_METHOD if scenario.name in METHOD_SCENARIOS else None}
    sizes_list = list(expand_sizes(scenario, smoke=smoke))

    def record(sizes: Dict[str, int], phase: str, **fields: Any) -> results.BenchmarkRecord:
        return results.BenchmarkRecord(
            scenario=scenario.name,
            backend="nusquids",
            phase=phase,
            device="cpu",
            dtype="float64",
            size=dict(sizes),
            options=dict(options),
            **fields,
        )

    if scenario.name in UNAVAILABLE:
        note = UNAVAILABLE[scenario.name]
        return [record(s, p, status="unavailable", note=note) for s in sizes_list for p in results.PHASES]

    check = check_sizes(scenario.name)
    energy, nadir = _grid(scenario, check)
    tolerance = TOLERANCES.get(scenario.name, DEFAULT_TOLERANCE)
    candidate = backend_factory(scenario.name).evaluate(energy, nadir)
    agreement = compare(candidate, reference[REFERENCE_KEYS[scenario.name]], tolerance)
    if not agreement["passed"]:
        note = (
            f"disagrees with the tpeanuts reference ({agreement['metric']} {agreement['value']:.3g} > "
            f"{tolerance:.3g}); not timed"
        )
        return [
            record(s, p, status="disagrees", agreement=agreement, note=note) for s in sizes_list for p in results.PHASES
        ]

    out: List[results.BenchmarkRecord] = []
    for sizes in sizes_list:
        points = sizes["n_energy"] * sizes.get("n_nadir", 1)
        if max_points and points > max_points:
            note = f"skipped: {points} points exceed --max-points={max_points}"
            out.extend(record(sizes, p, status="skipped", note=note) for p in results.PHASES)
            continue
        grid = _grid(scenario, sizes)
        try:
            build_timing, evaluate_timing = time_build_and_evaluate(
                lambda: backend_factory(scenario.name), lambda loaded, g=grid: loaded.evaluate(*g), n_repeats=repeats
            )
        except (MemoryError, ValueError, RuntimeError) as error:
            note = f"failed: {type(error).__name__}: {error}"[:500]
            out.extend(record(sizes, p, status="failed", note=note) for p in results.PHASES)
            continue
        out.append(record(sizes, "build", timing=build_timing.to_dict(), agreement=agreement))
        out.append(record(sizes, "evaluate", timing=evaluate_timing.to_dict(), agreement=agreement))
    return out


def parse_scenario_limits(items: Sequence[str]) -> Dict[str, int]:
    """Read ``["earth=2000", "atmosphere=0"]`` into ``{"earth": 2000, "atmosphere": 0}``.

    Args:
        items: ``SCENARIO=LIMIT`` strings.

    Returns:
        The limits by scenario.

    Raises:
        ValueError: If an item is not ``name=integer`` with an integer >= 0.
    """
    limits: Dict[str, int] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name or not value.isdigit():
            raise ValueError(f"expected SCENARIO=LIMIT with a non-negative integer, got {item!r}")
        limits[name] = int(value)
    return limits


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: check, time, print a summary and write the JSON under ``benchmarks/results``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS), help="subset of the catalogue")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="tpeanuts reference .npz")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="CPU threads of the process")
    parser.add_argument("--smoke", action="store_true", help="the reduced sizes of quick runs")
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS, help="skip larger grids; 0 for no limit")
    parser.add_argument(
        "--max-points-scenario",
        nargs="*",
        default=[],
        metavar="SCENARIO=LIMIT",
        help="own point limit of a scenario instead of --max-points (e.g. earth=2000 atmosphere=1000000; 0 for none)",
    )
    parser.add_argument("--label", default="nusquids_cpu_float64")
    parser.add_argument("--output", type=Path, default=None, help=".csv or .json (default: results/<label>.json)")
    results.add_spec_arguments(parser)
    args = parser.parse_args(argv)

    environment.set_threads(args.threads)
    reference, meta = load_reference(args.reference)
    result_set = run_benchmarks(
        args.scenarios,
        reference=reference,
        repeats=args.repeats,
        smoke=args.smoke,
        max_points=args.max_points,
        max_points_by_scenario=parse_scenario_limits(args.max_points_scenario),
        label=args.label,
        reference_meta=meta,
    )
    for entry in result_set.records:
        if entry.phase != "evaluate":
            continue
        shown = ", ".join(f"{key}={value}" for key, value in entry.size.items())
        if entry.status == "ok" and entry.timing is not None:
            print(f"{entry.scenario:12s} {shown:28s} min {entry.timing['min_s'] * 1e3:10.2f} ms")
        else:
            print(f"{entry.scenario:12s} {shown:28s} {entry.status}: {entry.note}")
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
