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

"""Time the tpeanuts scenarios of the catalogue and write a ``ResultSet`` JSON (the baseline of plan step B2).

Every scenario is run at every size of its axes (``--smoke`` selects the reduced sizes) and, for the scenarios that
cross the Earth or the atmosphere, with each requested propagation method. For each case the construction of the objects
and the evaluation of the grid are timed apart (``benchmarks.common.timing.time_build_and_evaluate``), with CUDA
synchronised when the device is a GPU, and the peak memory is recorded. A case that cannot be run is recorded, never
dropped: ``"skipped"`` above ``--max-points`` grid points, ``"failed"`` with the error when the method cannot handle the
size.

Usage, from the repository root::

    python -m benchmarks.tpeanuts.run --smoke --label quick
    python -m benchmarks.tpeanuts.run --device cuda --threads 8 --repeats 7 --max-points 0   # the full grid
    python -m benchmarks.tpeanuts.run --scenarios earth --methods numerical analytic

Module contents:
    run_benchmarks(...)
        Run the scenarios and return the ``ResultSet``.
    main(...)
        Command-line entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from benchmarks.common import environment, resources, results
from benchmarks.common.scenarios import METHOD_SCENARIOS, SCENARIOS, Scenario, expand_sizes
from benchmarks.common.timing import time_build_and_evaluate
from benchmarks.tpeanuts.cases import DEFAULT_MAX_WORK, DEFAULT_NSTEPS, METHODS, build_case, make_context
from tpeanuts.physics.evolution.method import PropagationMethod
from tpeanuts.util.context import RuntimeContext

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "benchmarks" / "results"

# The scenarios whose cost depends on the propagation method; the others are timed once.
_METHOD_SCENARIOS = METHOD_SCENARIOS
DEFAULT_MAX_POINTS = 1_000_000


def run_benchmarks(
    scenarios: Sequence[str],
    *,
    context: RuntimeContext,
    methods: Sequence[str] = ("numerical",),
    nsteps: int = DEFAULT_NSTEPS,
    repeats: int = 7,
    smoke: bool = False,
    max_points: int = DEFAULT_MAX_POINTS,
    max_work: int = DEFAULT_MAX_WORK,
    label: str = "tpeanuts",
) -> results.ResultSet:
    """Time the scenarios and collect one record per (scenario, size, method, phase).

    Args:
        scenarios: Names in ``benchmarks.common.scenarios.SCENARIOS``.
        context: Device and precision.
        methods: Propagation methods (keys of ``cases.METHODS``) for the scenarios that depend on it.
        nsteps: Chord discretisation of the Earth and atmosphere legs.
        repeats: Timed repeats per phase (the protocol value is 7).
        smoke: Use the reduced sizes.
        max_points: Skip the cases with more (energy x nadir) points than this; 0 means no limit.
        max_work: Memory budget of one chunk (see ``cases.build_case``).
        label: Name of the run.

    Returns:
        The validated result set, with the environment of this process.

    Raises:
        ValueError: If a scenario or a method name is unknown.
    """
    unknown = [name for name in scenarios if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios {unknown}; expected a subset of {sorted(SCENARIOS)}.")
    bad_methods = [name for name in methods if name not in METHODS]
    if bad_methods:
        raise ValueError(f"unknown methods {bad_methods}; expected a subset of {sorted(METHODS)}.")

    environment_record = environment.capture_environment(packages=("numpy", "torch", "tpeanuts"))
    records: List[results.BenchmarkRecord] = []
    for name in scenarios:
        scenario = SCENARIOS[name]
        scenario_methods = list(methods) if name in _METHOD_SCENARIOS else [None]
        for sizes in expand_sizes(scenario, smoke=smoke):
            for method_name in scenario_methods:
                records.extend(
                    _run_case(
                        scenario,
                        sizes,
                        method_name,
                        context=context,
                        nsteps=nsteps,
                        repeats=repeats,
                        max_points=max_points,
                        max_work=max_work,
                    )
                )
    result_set = results.ResultSet(label=label, environment=environment_record, records=records)
    result_set.validate()
    return result_set


def _run_case(
    scenario: Scenario,
    sizes: Dict[str, int],
    method_name: Optional[str],
    *,
    context: RuntimeContext,
    nsteps: int,
    repeats: int,
    max_points: int,
    max_work: int,
) -> List[results.BenchmarkRecord]:
    """The build and evaluate records of one case; a skipped or failed case gives one record per phase."""
    options = {"nsteps": nsteps if scenario.name in _METHOD_SCENARIOS else None, "max_work": max_work}
    if method_name is not None:
        options["method"] = method_name

    def record(phase: str, **fields) -> results.BenchmarkRecord:
        return results.BenchmarkRecord(
            scenario=scenario.name,
            backend="tpeanuts",
            phase=phase,
            device=context.device.type,
            dtype=str(context.dtype).replace("torch.", ""),
            size=dict(sizes),
            options=dict(options),
            **fields,
        )

    points = sizes["n_energy"] * sizes.get("n_nadir", 1)
    if max_points and points > max_points:
        note = f"skipped: {points} points exceed --max-points={max_points}"
        return [record(phase, status="skipped", note=note) for phase in results.PHASES]

    method = METHODS[method_name] if method_name is not None else PropagationMethod.NUMERICAL
    case = build_case(scenario.name, sizes, context=context, method=method, nsteps=nsteps, max_work=max_work)
    synchronize = resources.cuda_synchronizer(context.device)
    try:
        resources.reset_cuda_peak_memory(context.device)
        build_timing, evaluate_timing = time_build_and_evaluate(
            case.build, case.evaluate, n_repeats=repeats, synchronize=synchronize
        )
    except (MemoryError, ValueError, RuntimeError) as error:  # the method cannot handle this size; record why
        note = f"failed: {type(error).__name__}: {error}"
        return [record(phase, status="failed", note=note[:500]) for phase in results.PHASES]
    peak_cuda = resources.cuda_peak_memory_gb(context.device)
    peak_ram = resources.process_ram_gb()
    return [
        record("build", timing=build_timing.to_dict(), peak_ram_gb=peak_ram),
        record("evaluate", timing=evaluate_timing.to_dict(), peak_cuda_gb=peak_cuda, peak_ram_gb=peak_ram),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: run, print a summary and write the JSON under ``benchmarks/results``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS), help="subset of the catalogue")
    parser.add_argument("--methods", nargs="*", default=["numerical"], choices=sorted(METHODS))
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--nsteps", type=int, default=DEFAULT_NSTEPS)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (default: leave as they are)")
    parser.add_argument("--smoke", action="store_true", help="the reduced sizes of quick runs")
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS, help="skip larger grids; 0 for no limit")
    parser.add_argument("--max-work", type=int, default=DEFAULT_MAX_WORK, help="memory budget of one chunk")
    parser.add_argument("--label", default=None, help="name of the run (default: tpeanuts_<device>_<dtype>)")
    parser.add_argument("--output", type=Path, default=None, help=".csv or .json (default: results/<label>.json)")
    results.add_spec_arguments(parser)
    args = parser.parse_args(argv)

    if args.threads is not None:
        environment.set_threads(args.threads)
    context = make_context(args.device, args.dtype)
    label = args.label or f"tpeanuts_{context.device.type}_{args.dtype}"
    result_set = run_benchmarks(
        args.scenarios,
        context=context,
        methods=args.methods,
        nsteps=args.nsteps,
        repeats=args.repeats,
        smoke=args.smoke,
        max_points=args.max_points,
        max_work=args.max_work,
        label=label,
    )
    for entry in result_set.records:
        if entry.phase != "evaluate":
            continue
        shown = ", ".join(f"{key}={value}" for key, value in entry.size.items())
        method = entry.options.get("method") or "-"
        if entry.status == "ok" and entry.timing is not None:
            print(f"{entry.scenario:12s} {method:22s} {shown:28s} min {entry.timing['min_s'] * 1e3:10.2f} ms")
        else:
            print(f"{entry.scenario:12s} {method:22s} {shown:28s} {entry.status}: {entry.note}")
    path = results.save(
        result_set,
        args.output or RESULTS_DIR / f"{label}.json",
        spec_hash=args.spec_hash,
        spec_id=args.spec_id,
        smoke=args.smoke,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
