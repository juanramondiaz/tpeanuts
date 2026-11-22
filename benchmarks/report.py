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

"""Aggregate the benchmark result files into tables and figures: time against size, speed-ups and scaling exponents.

The runners of the three backends write ``ResultSet`` JSON files with one common schema (``benchmarks.common.results``);
this module only reads them, so it needs neither ``torch`` nor any reference package. Three summaries are built:

* the **times** of every timed record, one row each;
* the **speed-ups** of tpeanuts over each reference backend, matched on scenario, phase and sizes, and on the
  algorithm (the reference record names the tpeanuts method it is paired with in ``options["paired_method"]``); every
  tpeanuts device present is compared, so a GPU speed-up and a CPU one are separate rows and are labelled by device;
* the **scaling exponents**: the slope of log(time) against log(number of points), which says whether the cost is
  dominated by a fixed overhead (below 1) or grows with the grid (about 1).

Only records whose status is ``"ok"`` enter the speed-ups and the exponents. Absences (``unavailable``, ``disagrees``,
``failed``, ``skipped``) are listed with their reason by ``absences``, so that a missing number is never silent. The
compared statistic is the minimum, as the protocol fixes.

Usage, from the repository root::

    python -m benchmarks.report                                    # every JSON in benchmarks/results, tables to stdout
    python -m benchmarks.report --inputs a.json b.json --output-dir benchmarks/report --figures

Module contents:
    Row
        One timed record flattened for tabulation.
    load_results(...)
        Read result files, skipping those that are not result sets.
    rows_of(...)
        The timed rows of a list of result sets.
    speedups(...)
        Speed-up of tpeanuts over each reference record, with the caveats of the pairing.
    scaling_exponents(...)
        Log-log slope of time against points for each series.
    absences(...)
        The records that have no timing, with their status and reason.
    to_markdown(...), write_csv(...)
        Output of the tables.
    plot_times(...), plot_speedups(...)
        Figures (need ``matplotlib``).
    main(...)
        Command-line entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmarks.common import results
from benchmarks.common.resources import fit_loglog_slope

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks" / "results"
REFERENCE_BACKENDS = ("peanuts", "nusquids")

_SPEEDUP_COLUMNS = [
    "scenario",
    "reference",
    "method",
    "size",
    "device",
    "reference_s",
    "tpeanuts_s",
    "speedup",
    "caveat",
]

# Pairings whose two sides do different amounts of work, so that the speed-up is not like for like. Keyed by
# (reference backend, scenario).
CAVEATS = {
    ("nusquids", "atmosphere"): (
        "tpeanuts averages four production heights, nuSQuIDS has one fixed height: the speed-up is a lower bound"
    ),
}


@dataclass(frozen=True)
class Row:
    """One timed record, flattened.

    Attributes:
        label: Label of the result file it came from.
        scenario, backend, phase, device, dtype: As in the record.
        method: Algorithm: the tpeanuts ``method``, or the reference backend's ``backend_mode`` (``None`` if none).
        paired_method: The tpeanuts method a reference record is paired with (``None`` for tpeanuts rows).
        size: The sizes of the record.
        points: Number of (energy x nadir) points.
        min_s, median_s, iqr_s: Timing statistics, in seconds.
    """

    label: str
    scenario: str
    backend: str
    phase: str
    device: str
    dtype: str
    method: Optional[str]
    paired_method: Optional[str]
    size: Tuple[Tuple[str, int], ...]
    points: int
    min_s: float
    median_s: float
    iqr_s: float

    def sizes(self) -> Dict[str, int]:
        """The sizes as a dictionary."""
        return dict(self.size)


def load_results(paths: Iterable[Path]) -> List[results.ResultSet]:
    """Read result files (JSON or CSV); a file that is not a result set (or is of another schema version) is skipped.

    Args:
        paths: JSON and CSV files; the format is the file suffix.

    Returns:
        The validated result sets, in the order of ``paths``.
    """
    loaded = []
    for path in paths:
        try:
            if Path(path).suffix.lower() == ".csv":
                loaded.append(results.load_csv(path))
                continue
            if "records" not in json.loads(Path(path).read_text(encoding="utf-8")):
                continue
            loaded.append(results.load_json(path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return loaded


def _points(sizes: Dict[str, int]) -> int:
    return int(sizes.get("n_energy", 1)) * int(sizes.get("n_nadir", 1))


def rows_of(result_sets: Sequence[results.ResultSet]) -> List[Row]:
    """The timed (``"ok"``) records of the result sets, flattened.

    Args:
        result_sets: As returned by ``load_results``.

    Returns:
        One row per timed record.
    """
    rows = []
    for result_set in result_sets:
        for record in result_set.records:
            if record.status != "ok" or record.timing is None:
                continue
            method = record.options.get("method") or record.options.get("backend_mode")
            rows.append(
                Row(
                    label=result_set.label,
                    scenario=record.scenario,
                    backend=record.backend,
                    phase=record.phase,
                    device=record.device,
                    dtype=record.dtype,
                    method=method,
                    paired_method=record.options.get("paired_method"),
                    size=tuple(sorted(record.size.items())),
                    points=_points(record.size),
                    min_s=float(record.timing["min_s"]),
                    median_s=float(record.timing["median_s"]),
                    iqr_s=float(record.timing["iqr_s"]),
                )
            )
    return rows


def speedups(rows: Sequence[Row], *, phase: str = "evaluate") -> List[Dict[str, Any]]:
    """Speed-up of tpeanuts over each reference record, ``reference min / tpeanuts min``.

    A reference row is matched with the tpeanuts rows of the same scenario, phase and sizes whose ``method`` is the one
    it is paired with (a scenario without a method, such as vacuum, matches the tpeanuts rows without one). Each
    tpeanuts device and precision gives its own entry. A value above 1 means tpeanuts is faster.

    Args:
        rows: As returned by ``rows_of``.
        phase: ``"evaluate"`` or ``"build"``.

    Returns:
        Dictionaries with ``scenario``, ``reference``, ``method``, ``size``, ``points``, ``device``, ``dtype``,
        ``reference_s``, ``tpeanuts_s``, ``speedup`` and ``caveat``, ordered by scenario, reference, method and points.
    """
    own: Dict[Tuple[Any, ...], List[Row]] = {}
    for row in rows:
        if row.backend == "tpeanuts" and row.phase == phase:
            own.setdefault((row.scenario, row.size, row.method), []).append(row)
    out = []
    for ref in rows:
        if ref.backend not in REFERENCE_BACKENDS or ref.phase != phase:
            continue
        for mine in own.get((ref.scenario, ref.size, ref.paired_method), []):
            out.append(
                {
                    "scenario": ref.scenario,
                    "reference": ref.backend,
                    "method": ref.paired_method,
                    "size": dict(ref.size),
                    "points": ref.points,
                    "device": mine.device,
                    "dtype": mine.dtype,
                    "reference_s": ref.min_s,
                    "tpeanuts_s": mine.min_s,
                    "speedup": ref.min_s / mine.min_s,
                    "caveat": CAVEATS.get((ref.backend, ref.scenario)),
                }
            )
    out.sort(key=lambda e: (e["scenario"], e["reference"], e["method"] or "", e["device"], e["points"]))
    return out


def scaling_exponents(rows: Sequence[Row], *, phase: str = "evaluate", min_points: int = 3) -> List[Dict[str, Any]]:
    """The log-log slope of the minimum time against the number of points, for each series of rows.

    A series is a scenario, backend, device, precision and method. The slope is taken against the total number of points
    (energies times angles), which is what the cost follows; where several size combinations have the same number of
    points, the fastest is used.

    Args:
        rows: As returned by ``rows_of``.
        phase: ``"evaluate"`` or ``"build"``.
        min_points: Series with fewer distinct sizes are left out (a slope of two points is not a fit).

    Returns:
        Dictionaries with ``scenario``, ``backend``, ``device``, ``dtype``, ``method``, ``n_sizes`` and ``slope``.
    """
    series: Dict[Tuple[Any, ...], Dict[int, float]] = {}
    for row in rows:
        if row.phase != phase:
            continue
        key = (row.scenario, row.backend, row.device, row.dtype, row.method or row.paired_method)
        best = series.setdefault(key, {})
        best[row.points] = min(row.min_s, best.get(row.points, row.min_s))  # one value per point count
    out = []
    for (scenario, backend, device, dtype, method), by_points in sorted(series.items(), key=lambda kv: str(kv[0])):
        if len(by_points) < min_points:
            continue
        points = sorted(by_points)
        out.append(
            {
                "scenario": scenario,
                "backend": backend,
                "device": device,
                "dtype": dtype,
                "method": method,
                "n_sizes": len(points),
                "slope": fit_loglog_slope(points, [by_points[p] for p in points]),
            }
        )
    return out


def absences(result_sets: Sequence[results.ResultSet]) -> List[Dict[str, Any]]:
    """The records without a timing, one entry per (scenario, backend, status, reason).

    Args:
        result_sets: As returned by ``load_results``.

    Returns:
        Dictionaries with ``scenario``, ``backend``, ``status``, ``note`` and ``count``.
    """
    counts: Dict[Tuple[str, str, str, str], int] = {}
    for result_set in result_sets:
        for record in result_set.records:
            if record.status == "ok":
                continue
            key = (record.scenario, record.backend, record.status, record.note or "")
            counts[key] = counts.get(key, 0) + 1
    return [
        {"scenario": s, "backend": b, "status": st, "note": n, "count": c}
        for (s, b, st, n), c in sorted(counts.items())
    ]


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return "" if value is None else str(value)


def to_markdown(table: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> str:
    """A list of dictionaries as a GitHub Markdown table.

    Args:
        table: The rows.
        columns: The columns to show, in order (default: the keys of the first row).

    Returns:
        The table, or an empty string for no rows.
    """
    if not table:
        return ""
    columns = list(columns or table[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(_format(row.get(c)) for c in columns) + " |" for row in table]
    return "\n".join(lines)


def write_csv(table: Sequence[Dict[str, Any]], path: Path) -> Path:
    """A list of dictionaries as a CSV file (nested dictionaries are written as ``key=value`` lists).

    Args:
        table: The rows.
        path: Destination; its directory is created if needed.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(table[0]) if table else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in table:
            writer.writerow([_format(row.get(c)) for c in columns])
    return path


def plot_times(rows: Sequence[Row], scenario: str, path: Path, *, phase: str = "evaluate") -> Optional[Path]:
    """Minimum time against number of points for one scenario, one line per backend, device and method.

    Args:
        rows: As returned by ``rows_of``.
        scenario: The scenario to plot.
        path: Image file to write.
        phase: ``"evaluate"`` or ``"build"``.

    Returns:
        The path written, or ``None`` if the scenario has no timed rows.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: Dict[Tuple[str, str, Optional[str]], Dict[int, float]] = {}
    for row in rows:
        if row.scenario == scenario and row.phase == phase:
            best = series.setdefault((row.backend, row.device, row.method or row.paired_method), {})
            best[row.points] = min(row.min_s, best.get(row.points, row.min_s))
    if not series:
        return None
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    for (backend, device, method), by_points in sorted(series.items(), key=lambda kv: str(kv[0])):
        points = sorted(by_points)
        name = f"{backend} {device}" + (f" ({method})" if method else "")
        axis.loglog(points, [by_points[p] for p in points], marker="o", label=name)
    axis.set_xlabel("points (energies x angles)")
    axis.set_ylabel(f"minimum {phase} time (s)")
    axis.set_title(scenario)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_speedups(entries: Sequence[Dict[str, Any]], path: Path) -> Optional[Path]:
    """Speed-up of tpeanuts against the number of points, one line per scenario, reference, device and method.

    Args:
        entries: As returned by ``speedups``.
        path: Image file to write.

    Returns:
        The path written, or ``None`` if there are no entries.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not entries:
        return None
    series: Dict[Tuple[str, str, str, Optional[str]], List[Tuple[int, float]]] = {}
    for entry in entries:
        key = (entry["scenario"], entry["reference"], entry["device"], entry["method"])
        series.setdefault(key, []).append((entry["points"], entry["speedup"]))
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    for (scenario, reference, device, method), values in sorted(series.items(), key=lambda kv: str(kv[0])):
        values.sort()
        label = f"{scenario} vs {reference} ({device}" + (f", {method})" if method else ")")
        axis.loglog([v[0] for v in values], [v[1] for v in values], marker="o", label=label)
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_xlabel("points (energies x angles)")
    axis.set_ylabel("speed-up of tpeanuts (reference time / tpeanuts time)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: aggregate, print the tables, and optionally write CSV files and figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="*", type=Path, default=None, help="JSON or CSV results (default: results/)")
    parser.add_argument("--output-dir", type=Path, default=None, help="write CSV tables (and figures) here")
    parser.add_argument("--figures", action="store_true", help="also write the figures (needs matplotlib)")
    parser.add_argument("--phase", default="evaluate", choices=results.PHASES)
    args = parser.parse_args(argv)

    paths = args.inputs if args.inputs else sorted([*RESULTS_DIR.glob("*.json"), *RESULTS_DIR.glob("*.csv")])
    result_sets = load_results(paths)
    if not result_sets:
        print(f"no result files found among {len(list(paths))} candidate(s)", file=sys.stderr)
        return 1
    rows = rows_of(result_sets)
    comparison = speedups(rows, phase=args.phase)
    slopes = scaling_exponents(rows, phase=args.phase)
    missing = absences(result_sets)

    sections = {
        "speedups": (comparison, _SPEEDUP_COLUMNS),
        "scaling": (slopes, None),
        "absences": (missing, None),
    }
    for name, (table, columns) in sections.items():
        print(f"\n## {name}\n")
        print(to_markdown(table, columns) or "(none)")
    if args.output_dir:
        for name, (table, _) in sections.items():
            write_csv(table, args.output_dir / f"{name}.csv")
        if args.figures:
            for scenario in sorted({r.scenario for r in rows}):
                plot_times(rows, scenario, args.output_dir / f"time_{scenario}.png", phase=args.phase)
            plot_speedups(comparison, args.output_dir / "speedups.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
