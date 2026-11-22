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

"""What the benchmark notebooks show: tables of the results read from the CSV files, and the structural checks on them.

A benchmark notebook never measures an expensive benchmark; it reads the CSV that ``cache.load_or_run`` finds (and stops
with the exact commands if it is missing). This module turns the loaded result sets into the tables and figures that the
notebooks display, and into the checks they record. The checks are structural: they say whether the data can be trusted
(they belong to the specification, are final, come from a clean commit, every case was run and agrees with the
reference), never whether a code is fast enough, which is a result to read, not a pass criterion.

Quick mode (``smoke=True``) reads the quick-run CSVs, which are *preliminary* by construction: the checks then expect
that state and the stability, clean-commit and same-machine checks are informative only, so that a notebook runs end to
end before the real measurements exist. With ``smoke=False`` the same checks demand final data.

Module contents:
    load(...)
        The result sets of several specifications, by id.
    provenance_table(...)
        Where each result set came from: file, state, machine, commit, date.
    timing_table(...)
        One row per timed record, in milliseconds, with its stability.
    agreement_table(...)
        The correctness gate: how far each reference backend is from the tpeanuts reference, per scenario.
    speedup_table(...), exponent_table(...), memory_table(...), absence_table(...)
        The derived tables of ``benchmarks.report`` and the recorded peak memory, as data frames.
    device_ratio_table(...), method_ratio_table(...), build_share_table(...), coverage_table(...)
        Ratios between two series of the same machine, the share of the build phase, and the coverage of a set of runs.
    record_checks(...), record_same_machine(...)
        The structural checks of a set of loaded results, recorded in a ``CheckLog``.
    plot_time_vs_points(...)
        Minimum time against the number of points, one line per series, on log-log axes.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from benchmarks import report
from benchmarks.common import cache, results, scenarios, spec

STABILITY_CEILING = 0.5  # IQR / median above which a timing is called noisy
STABLE_FRACTION = 0.9  # fraction of the timings that must be below the ceiling


def load(spec_ids: Sequence[str], *, smoke: bool = False, mode: Optional[str] = None) -> Dict[str, cache.Loaded]:
    """Read the result set of each specification.

    Args:
        spec_ids: Ids of the specifications (``<series>/<name>``).
        smoke: Read the quick-run CSVs (preliminary) instead of the final ones.
        mode: ``cache.MODES`` entry, or ``None`` for ``BENCH_MODE`` / ``auto``.

    Returns:
        ``{spec id: Loaded}``, in the order of ``spec_ids``.

    Raises:
        cache.ResultsError: If a CSV is missing and may not be measured here; the message has the commands.
        KeyError: If an id is not a specification.
    """
    by_id = {s.id: s for s in spec.discover()}
    missing = [i for i in spec_ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown specification(s) {missing}; known: {sorted(by_id)}")
    return {i: cache.load_or_run(by_id[i], mode=mode, smoke=smoke) for i in spec_ids}


def result_sets(loaded: Mapping[str, cache.Loaded]) -> List[results.ResultSet]:
    """The result sets of ``loaded``, in order."""
    return [item.result_set for item in loaded.values()]


def provenance_table(loaded: Mapping[str, cache.Loaded]) -> pd.DataFrame:
    """Where each result set came from: the facts a reader needs to trust the numbers."""
    rows = []
    for spec_id, item in loaded.items():
        info = item.provenance()
        rows.append(
            {
                "Specification": spec_id,
                "State": info["state"],
                "Source": info["source"],
                "Rows": info["rows"],
                "Machine": info["host"] or "-",
                "CPU": info["cpu"] or "-",
                "GPU": info["gpus"] or "-",
                "Commit": (info["commit"] or "-")[:10],
                "Clean": {"True": "no", "False": "yes"}.get(str(info["dirty"]), "unknown"),
                "Date (UTC)": (info["date_utc"] or "-")[:19],
            }
        )
    return pd.DataFrame(rows).set_index("Specification")


def _selected(rows: Iterable[report.Row], scenario: Optional[str], phase: Optional[str]) -> List[report.Row]:
    return [r for r in rows if (scenario is None or r.scenario == scenario) and (phase is None or r.phase == phase)]


def timing_table(
    loaded: Mapping[str, cache.Loaded], *, scenario: Optional[str] = None, phase: Optional[str] = "evaluate"
) -> pd.DataFrame:
    """One row per timed record, in milliseconds.

    Args:
        loaded: As returned by ``load``.
        scenario: Keep only this scenario (all if ``None``).
        phase: ``"evaluate"``, ``"build"`` or ``None`` for both.

    Returns:
        Columns ``label``, ``scenario``, ``backend``, ``method``, ``device``, ``dtype``, ``n_energy``, ``n_nadir``,
        ``points``, ``min_ms``, ``median_ms`` and ``iqr_over_median`` (the stability of the timing).
    """
    rows = _selected(report.rows_of(result_sets(loaded)), scenario, phase)
    columns = [
        "label", "scenario", "backend", "method", "device", "dtype", "phase", "n_energy", "n_nadir", "points",
        "min_ms", "median_ms", "iqr_over_median",
    ]  # fmt: skip
    data = [
        {
            "label": r.label, "scenario": r.scenario, "backend": r.backend,
            "method": r.method or r.paired_method or "-",
            "device": r.device, "dtype": r.dtype, "phase": r.phase,
            "n_energy": r.sizes().get("n_energy", 1), "n_nadir": r.sizes().get("n_nadir", 1), "points": r.points,
            "min_ms": 1e3 * r.min_s, "median_ms": 1e3 * r.median_s,
            "iqr_over_median": r.iqr_s / r.median_s if r.median_s > 0 else float("nan"),
        }
        for r in rows
    ]  # fmt: skip
    return pd.DataFrame(data, columns=columns).sort_values(["scenario", "label", "method", "points"], ignore_index=True)


def agreement_table(loaded: Mapping[str, cache.Loaded]) -> pd.DataFrame:
    """The correctness gate: the largest probability difference to the tpeanuts reference, per scenario and backend.

    Args:
        loaded: As returned by ``load``.

    Returns:
        Columns ``label``, ``scenario``, ``cases`` (with a recorded agreement), ``passed``, ``max_difference`` and
        ``tolerance``. Records without an agreement (the tpeanuts rows, which are the reference) are not counted.
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for item in loaded.values():
        for record in item.result_set.records:
            if record.agreement is None:
                continue
            groups.setdefault((item.result_set.label, record.scenario), []).append(record.agreement)
    data = []
    for (label, scenario), agreements in sorted(groups.items()):
        values = [a["value"] for a in agreements if a.get("value") is not None]
        data.append(
            {
                "label": label, "scenario": scenario, "cases": len(agreements),
                "passed": sum(bool(a.get("passed")) for a in agreements),
                "max_difference": max(values) if values else float("nan"),
                "tolerance": agreements[0].get("tolerance"),
            }
        )  # fmt: skip
    return pd.DataFrame(data, columns=["label", "scenario", "cases", "passed", "max_difference", "tolerance"])


def speedup_table(loaded: Mapping[str, cache.Loaded], *, phase: str = "evaluate") -> pd.DataFrame:
    """The speed-up of tpeanuts over each reference record (``reference min / tpeanuts min``) as a data frame."""
    entries = report.speedups(report.rows_of(result_sets(loaded)), phase=phase)
    frame = pd.DataFrame(
        [
            {
                "scenario": e["scenario"], "reference": e["reference"], "method": e["method"] or "-",
                "n_energy": e["size"].get("n_energy", 1), "n_nadir": e["size"].get("n_nadir", 1),
                "points": e["points"], "tpeanuts": f"{e['device']} {e['dtype']}",
                "reference_ms": 1e3 * e["reference_s"], "tpeanuts_ms": 1e3 * e["tpeanuts_s"], "speedup": e["speedup"],
            }
            for e in entries
        ],
        columns=[
            "scenario", "reference", "method", "n_energy", "n_nadir", "points", "tpeanuts", "reference_ms",
            "tpeanuts_ms", "speedup",
        ],
    )  # fmt: skip
    return frame


def exponent_table(loaded: Mapping[str, cache.Loaded], *, phase: str = "evaluate", min_points: int = 3) -> pd.DataFrame:
    """The log-log slope of time against points for each series (see ``report.scaling_exponents``)."""
    entries = report.scaling_exponents(report.rows_of(result_sets(loaded)), phase=phase, min_points=min_points)
    columns = ["scenario", "backend", "device", "dtype", "method", "n_sizes", "slope"]
    return pd.DataFrame(entries, columns=columns)


def memory_table(loaded: Mapping[str, cache.Loaded]) -> pd.DataFrame:
    """The largest recorded peak memory of each series, in GB (``NaN`` where it was not recorded)."""
    best: Dict[tuple, Dict[str, float]] = {}
    for item in loaded.values():
        for record in item.result_set.records:
            if record.status != "ok":
                continue
            key = (item.result_set.label, record.scenario, record.phase)
            entry = best.setdefault(key, {"peak_ram_gb": np.nan, "peak_cuda_gb": np.nan})
            for name in entry:
                value = getattr(record, name)
                if value is not None:
                    entry[name] = value if np.isnan(entry[name]) else max(entry[name], value)
    data = [{"label": k[0], "scenario": k[1], "phase": k[2], **v} for k, v in sorted(best.items())]
    return pd.DataFrame(data, columns=["label", "scenario", "phase", "peak_ram_gb", "peak_cuda_gb"])


def absence_table(loaded: Mapping[str, cache.Loaded]) -> pd.DataFrame:
    """The records without a timing and why (unavailable, skipped, failed), one row per reason."""
    entries = report.absences(result_sets(loaded))
    return pd.DataFrame(entries, columns=["scenario", "backend", "status", "note", "count"])


def _min_ms(loaded: Mapping[str, cache.Loaded], spec_id: str, phase: str = "evaluate") -> pd.DataFrame:
    """The minimum time of one specification per scenario, method and size, in milliseconds."""
    table = timing_table({spec_id: loaded[spec_id]}, phase=phase)
    keys = ["scenario", "method", "n_energy", "n_nadir", "points"]
    return table.groupby(keys, as_index=False)["min_ms"].min()


def device_ratio_table(loaded: Mapping[str, cache.Loaded], numerator_id: str, denominator_id: str) -> pd.DataFrame:
    """The ratio ``numerator min time / denominator min time`` for every case both specifications timed.

    With the CPU specification as numerator and a CUDA one as denominator the ratio is the GPU speed-up (above 1: the
    GPU is faster). Both must come from the same machine to mean anything.

    Args:
        loaded: As returned by ``load``.
        numerator_id, denominator_id: Ids of the two specifications.

    Returns:
        Columns ``scenario``, ``method``, ``n_energy``, ``n_nadir``, ``points``, ``numerator_ms``,
        ``denominator_ms`` and ``ratio``.
    """
    keys = ["scenario", "method", "n_energy", "n_nadir", "points"]
    merged = _min_ms(loaded, numerator_id).merge(_min_ms(loaded, denominator_id), on=keys, suffixes=("_num", "_den"))
    merged = merged.rename(columns={"min_ms_num": "numerator_ms", "min_ms_den": "denominator_ms"})
    merged["ratio"] = merged["numerator_ms"] / merged["denominator_ms"]
    return merged.sort_values(["scenario", "method", "points"], ignore_index=True)


def method_ratio_table(loaded: Mapping[str, cache.Loaded], spec_id: str, slow: str, fast: str) -> pd.DataFrame:
    """The ratio of the minimum time of method ``slow`` to method ``fast`` of one tpeanuts specification, per case."""
    table = _min_ms(loaded, spec_id)
    keys = ["scenario", "n_energy", "n_nadir", "points"]
    wide = table.pivot_table(index=keys, columns="method", values="min_ms").reset_index()
    if slow not in wide or fast not in wide:
        return pd.DataFrame(columns=keys + [slow, fast, "ratio"])
    wide = wide.dropna(subset=[slow, fast])
    wide["ratio"] = wide[slow] / wide[fast]
    return wide[keys + [slow, fast, "ratio"]].sort_values(["scenario", "points"], ignore_index=True)


def build_share_table(loaded: Mapping[str, cache.Loaded], spec_id: str) -> pd.DataFrame:
    """The share of the build phase in the total time (build + evaluate), per scenario, method and size."""
    both = timing_table({spec_id: loaded[spec_id]}, phase=None)
    keys = ["scenario", "method", "n_energy", "n_nadir", "points"]
    wide = both.groupby(keys + ["phase"])["min_ms"].min().unstack("phase").reset_index()
    if "build" not in wide or "evaluate" not in wide:
        return pd.DataFrame(columns=keys + ["build_ms", "evaluate_ms", "build_share"])
    wide = wide.dropna(subset=["build", "evaluate"]).rename(columns={"build": "build_ms", "evaluate": "evaluate_ms"})
    wide["build_share"] = wide["build_ms"] / (wide["build_ms"] + wide["evaluate_ms"])
    return wide.sort_values(["scenario", "method", "points"], ignore_index=True)


def coverage_table(loaded: Mapping[str, cache.Loaded]) -> pd.DataFrame:
    """The number of timed records of each specification (rows) in each scenario (columns); 0 is a gap."""
    counts: Dict[str, Dict[str, int]] = {}
    for spec_id, item in loaded.items():
        row = counts.setdefault(spec_id, dict.fromkeys(scenarios.SCENARIOS, 0))
        for record in item.result_set.records:
            if record.status == "ok" and record.phase == "evaluate":
                row[record.scenario] = row.get(record.scenario, 0) + 1
    return pd.DataFrame.from_dict(counts, orient="index")


def record_same_machine(
    checks: Any, loaded: Mapping[str, cache.Loaded], spec_ids: Sequence[str], *, smoke: bool = False
) -> None:
    """Record whether the CPU specifications ``spec_ids`` were measured on the same machine.

    A CPU-against-CPU comparison is only meaningful on one machine; the check compares the scheduler machine (or
    ``local``) and the CPU model of the result sets. In quick mode it is informative only: quick runs are taken on a
    workstation, and the same computer reports another CPU text from Windows than from WSL.

    Args:
        checks: The ``CheckLog``.
        loaded: As returned by ``load``.
        spec_ids: The specifications whose machines are compared.
        smoke: Quick mode (the check always passes).
    """
    identity = {i: (loaded[i].meta.get("env_host"), loaded[i].meta.get("env_cpu_model")) for i in spec_ids}
    checks.record(
        "same machine", f"{len(set(identity.values()))} distinct machine(s)",
        "the CPU results being compared were measured on one machine"
        + (" (informative in quick mode)" if smoke else ""),
        smoke or len(set(identity.values())) == 1,
    )  # fmt: skip


def record_checks(
    checks: Any,
    loaded: Mapping[str, cache.Loaded],
    *,
    smoke: bool = False,
    expected_scenarios: Optional[Mapping[str, Sequence[str]]] = None,
) -> None:
    """Record the structural checks of ``loaded`` in ``checks`` (a ``notebooks_helper.CheckLog``).

    Args:
        checks: The ``CheckLog``.
        loaded: As returned by ``load``.
        smoke: Quick mode: the data are expected to be preliminary, and the clean-commit and stability checks are
            informative (always pass) because a quick run is neither.
        expected_scenarios: ``{spec id: scenarios}`` that must have at least one timed record each (the coverage). The
            default is no coverage check.
    """
    expected_state = "preliminary" if smoke else "fresh"
    states = {i: item.state for i, item in loaded.items()}
    checks.record(
        "data state", ", ".join(sorted(set(states.values()))),
        f"every result set is {expected_state}" + (" (quick mode)" if smoke else ""),
        all(s == expected_state for s in states.values()),
    )  # fmt: skip
    checks.record(
        "belongs to the specification", len(loaded), "the recorded spec_hash is the one of the YAML",
        all(item.meta.get("spec_hash") == item.spec.spec_hash(item.smoke) for item in loaded.values()),
    )  # fmt: skip
    dirty = [
        i for i, item in loaded.items() if str(item.meta.get("env_dirty")).lower() != "false"
    ]  # the CSV keeps text
    checks.record(
        "clean commit", f"{len(dirty)} without a clean commit", "every result set was taken from a clean checkout",
        smoke or not dirty,
    )  # fmt: skip
    records = [rec for item in loaded.values() for rec in item.result_set.records]
    failed = sum(rec.status == "failed" for rec in records)
    checks.record("no failed records", failed, "no record ended in an error", failed == 0)
    timed = sum(rec.status == "ok" for rec in records)
    checks.record("records were timed", timed, "at least one timed record", timed > 0)
    gate = agreement_table(loaded)
    bad = int((gate["cases"] - gate["passed"]).sum()) if len(gate) else 0
    checks.record("correctness gate", f"{bad} disagreeing of {int(gate['cases'].sum()) if len(gate) else 0}",
                  "every timed reference record agrees with the tpeanuts reference", bad == 0)  # fmt: skip
    stability = timing_table(loaded, phase=None)["iqr_over_median"].dropna()
    share = float((stability <= STABILITY_CEILING).mean()) if len(stability) else 0.0
    checks.record(
        "timing stability", f"{share:.0%} of timings with IQR/median <= {STABILITY_CEILING}",
        f"at least {STABLE_FRACTION:.0%} of the timings are stable" + (" (informative in quick mode)" if smoke else ""),
        smoke or share >= STABLE_FRACTION,
    )  # fmt: skip
    for spec_id, wanted in (expected_scenarios or {}).items():
        present = {rec.scenario for rec in loaded[spec_id].result_set.records if rec.status == "ok"}
        absent = sorted(set(wanted) - present)
        checks.record(f"coverage {spec_id.split('/')[-1]}", f"missing: {absent or 'none'}",
                      f"a timed record in each of {list(wanted)}", not absent)  # fmt: skip


def plot_time_vs_points(
    ax: Any, table: pd.DataFrame, *, group: Sequence[str] = ("label", "method"), column: str = "min_ms"
) -> None:
    """Draw ``column`` against the number of points, one line per series, on log-log axes.

    Args:
        ax: A matplotlib axes.
        table: A ``timing_table`` (one scenario).
        group: The columns that identify a series.
        column: The value to draw.
    """
    for key, part in table.groupby(list(group), sort=True):
        best = part.groupby("points")[column].min().sort_index()
        ax.plot(best.index, best.values, marker="o", label=" / ".join(str(k) for k in np.atleast_1d(key)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("points (energies x angles)")
    ax.set_ylabel(column.replace("_", " "))
    ax.legend(fontsize=7)
