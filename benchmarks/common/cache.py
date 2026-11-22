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

"""Read a benchmark result if it already exists, measure it only when that is allowed.

A benchmark is expensive, so a notebook must not repeat it when its CSV is there. ``load_or_run`` looks for the CSV of a
specification (next to the YAML, see ``spec.Spec.output_csv``) and decides:

========  ===========================================  =========================================================
mode      CSV present and current                      CSV missing, or present but of another specification
========  ===========================================  =========================================================
``read``  read it                                      never measures: a missing file is an error and a stale
                                                       one is returned marked ``stale`` with a warning
``auto``  read it                                      measures only a ``cheap`` specification; an
                                                       ``expensive`` one stops with the exact commands to run it
``run``   measures it again                            measures it
========  ===========================================  =========================================================

A CSV is *current* when the ``spec_hash`` it recorded equals the one of the specification, so a change to the sizes,
the repetitions, the methods, the precision, the threads or the resources is detected. A CSV that is a quick run
(``smoke``) is *preliminary*: it is returned, but a reader must not present it as a final result. The mode comes from
the argument or, when ``None``, from the ``BENCH_MODE`` environment variable (``auto`` by default).

Measuring from here is possible only for tpeanuts, whose runner uses the environment of the caller; PEANUTS and
nuSQuIDS have their own environments (and the heavy runs belong on the cluster), so for them the call stops with the
commands, or uses the ``runner`` the caller passes. The module imports nothing that needs ``torch`` or a reference
package.

Module contents:
    MODES
        The three modes.
    ResultsError, MissingResultError, StaleResultError, RunUnavailable, RunFailed
        What can go wrong, each with the commands that fix it.
    Loaded
        A result set with where it came from and whether it is current, preliminary or stale.
    inspect(...)
        The state of the CSV of a specification, without reading its records.
    load_or_run(...)
        The decision above.
    default_runner(...)
        Runs the backend of a specification in a subprocess, for tpeanuts.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from benchmarks.common import results
from benchmarks.common.spec import Spec

REPOSITORY = Path(__file__).resolve().parents[2]
MODES = ("read", "auto", "run")
STATES = ("fresh", "stale", "preliminary", "missing")
_RUNNER_MODULES = {
    "tpeanuts": "benchmarks.tpeanuts.run",
    "peanuts": "benchmarks.peanuts.run",
    "nusquids": "benchmarks.nusquids.run",
}


class ResultsError(RuntimeError):
    """Base class of the errors of this module; the message says what to do."""


class MissingResultError(ResultsError):
    """The CSV does not exist and may not be measured here."""


class StaleResultError(ResultsError):
    """The CSV belongs to another specification and may not be measured again here."""


class RunUnavailable(ResultsError):
    """The backend cannot be run from this process (it has its own environment)."""


class RunFailed(ResultsError):
    """The runner finished without leaving a CSV of this specification."""


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY).as_posix()
    except ValueError:
        return str(path)


def _commands(spec: Spec, smoke: bool) -> str:
    """The commands that produce the CSV of ``spec``: render the cluster job, or run the runner by hand."""
    yaml = _relative(spec.path) if spec.path is not None else f"{spec.id}.yaml"
    runner = _RUNNER_MODULES.get(spec.backend, spec.backend)
    argv = " ".join(shlex.quote(arg) for arg in spec.to_argv(smoke=smoke)[1:])
    return (
        f"  cluster:  python benchmarks/condor/render.py {yaml}  then  condor_submit <the .sub it writes>\n"
        f"  by hand:  python -s -m {runner} {argv}   (in the environment of the {spec.backend} backend)"
    )


@dataclass
class Loaded:
    """A result set read from its CSV, or measured now, with its provenance.

    Attributes:
        spec: The specification.
        result_set: The records and the environment they were taken in.
        path: The CSV.
        source: ``"csv"`` (read, nothing was measured) or ``"measured"`` (the runner was called by this call).
        state: ``"fresh"`` (the CSV belongs to this specification), ``"stale"`` (it records another one) or
            ``"preliminary"`` (it is a quick run).
        meta: ``results.csv_metadata`` of the CSV.
        smoke: Whether the quick-run variant was requested.
        notes: Why the state is not ``"fresh"``, if it is not.
    """

    spec: Spec
    result_set: results.ResultSet
    path: Path
    source: str
    state: str
    meta: Dict[str, Any]
    smoke: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the data are final and belong to the specification (``state == "fresh"``)."""
        return self.state == "fresh"

    def provenance(self) -> Dict[str, Any]:
        """The facts a reader needs to trust the numbers: source, state, specification, machine, code and date."""
        meta = self.meta
        return {
            "source": "read from CSV" if self.source == "csv" else "measured now",
            "state": self.state,
            "csv": _relative(self.path),
            "spec": self.spec.id,
            "spec_hash": meta.get("spec_hash"),
            "rows": meta.get("n_rows"),
            "date_utc": meta.get("env_timestamp_utc"),
            "host": meta.get("env_host"),
            "cpu": meta.get("env_cpu_model"),
            "gpus": meta.get("env_gpus"),
            "commit": meta.get("env_commit"),
            "dirty": meta.get("env_dirty"),
            "job": meta.get("env_job"),
        }

    def describe(self) -> str:
        """A short text of ``provenance``, for the first cell of a notebook."""
        info = {key: value for key, value in self.provenance().items() if value not in (None, "")}
        lines = [f"{key:10s} {value}" for key, value in info.items()]
        lines += [f"WARNING    {note}" for note in self.notes]
        return "\n".join(lines)


def inspect(spec: Spec, *, smoke: bool = False) -> Dict[str, Any]:
    """The state of the CSV of a specification.

    Args:
        spec: The specification.
        smoke: Look at the quick-run CSV instead of the real one.

    Returns:
        A dictionary with ``path``, ``state`` (``"missing"``, ``"fresh"``, ``"stale"`` or ``"preliminary"``), ``meta``
        (``None`` when missing) and ``notes`` (the reasons of a state other than ``"fresh"``).

    Raises:
        ResultsError: If the file exists but is not a result CSV.
    """
    path = spec.output_csv(smoke)
    if not path.exists():
        return {"path": path, "state": "missing", "meta": None, "notes": [f"{_relative(path)} does not exist"]}
    try:
        meta = results.csv_metadata(path)
    except ValueError as error:
        raise ResultsError(f"{_relative(path)} is not a benchmark result CSV: {error}") from error
    expected = spec.spec_hash(smoke)
    name = _relative(path)

    def found(state: str, *notes: str) -> Dict[str, Any]:
        return {"path": path, "state": state, "meta": meta, "notes": list(notes)}

    if meta["smoke"] and not smoke:
        return found("preliminary", f"{name} is a quick run (smoke): preliminary data, not a final result")
    if meta["spec_hash"] is None:
        return found("stale", f"{name} records no specification hash, so it cannot be tied to {spec.id}")
    if meta["spec_hash"] != expected:
        return found(
            "stale",
            f"{name} was measured with another configuration of {spec.id}: "
            f"its hash is {meta['spec_hash']}, the specification's is {expected}",
        )
    if smoke:
        return found("preliminary", "a quick run was requested: the data are preliminary")
    return found("fresh")


def default_runner(spec: Spec, *, smoke: bool = False) -> None:
    """Run the backend of ``spec`` in a subprocess of this interpreter and wait for it.

    Only tpeanuts can be run this way: its runner needs the environment of the caller. The CPU threads are passed as
    ``--threads`` (on the cluster ``run.sh`` does it from ``BENCH_THREADS``).

    Args:
        spec: The specification; its CSV is written where ``spec.output_csv`` says.
        smoke: Run the quick variant.

    Raises:
        RunUnavailable: If the backend is not tpeanuts.
        RunFailed: If the runner exits with an error.
    """
    if spec.backend != "tpeanuts":
        raise RunUnavailable(
            f"{spec.id}: the {spec.backend} backend has its own environment and cannot be run from here. Run it with\n"
            + _commands(spec, smoke)
        )
    argv = spec.to_argv(smoke=smoke)
    command = [
        sys.executable,
        "-s",
        "-m",
        _RUNNER_MODULES[spec.backend],
        *argv[1:],
        "--threads",
        str(spec.effective_threads),
    ]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPOSITORY), os.environ.get("PYTHONPATH")])))
    completed = subprocess.run(command, cwd=REPOSITORY, env=env, check=False)
    if completed.returncode != 0:
        raise RunFailed(f"{spec.id}: the runner exited with code {completed.returncode}")


def _measure(spec: Spec, smoke: bool, runner: Optional[Callable[..., None]]) -> Loaded:
    """Call the runner and read the CSV it must have written."""
    (runner or default_runner)(spec, smoke=smoke)
    found = inspect(spec, smoke=smoke)
    if found["state"] == "missing":
        raise RunFailed(f"{spec.id}: the runner finished but {_relative(found['path'])} was not written")
    if found["state"] == "stale":
        raise RunFailed(f"{spec.id}: the runner wrote a CSV of another specification. " + " ".join(found["notes"]))
    return Loaded(
        spec, results.load_csv(found["path"]), found["path"], "measured", found["state"], found["meta"], smoke,
        found["notes"],
    )  # fmt: skip


def load_or_run(
    spec: Spec,
    *,
    mode: Optional[str] = None,
    smoke: bool = False,
    runner: Optional[Callable[..., None]] = None,
) -> Loaded:
    """Read the CSV of ``spec`` if it exists and is current; measure it only where the mode and the cost allow.

    Args:
        spec: The benchmark specification (``spec.load_spec``).
        mode: ``"read"``, ``"auto"`` or ``"run"`` (see the module docstring); the ``BENCH_MODE`` environment variable,
            or ``"auto"``, when ``None``.
        smoke: Work with the quick-run variant and its own CSV.
        runner: ``runner(spec, smoke=...)`` that writes the CSV; ``default_runner`` when ``None``.

    Returns:
        The ``Loaded`` result; check ``loaded.ok`` before presenting the numbers as final.

    Raises:
        ValueError: If the mode is unknown or the specification is not a result benchmark (``export``).
        MissingResultError: If the CSV is missing and may not be measured (mode ``read``, or ``auto`` and expensive).
        StaleResultError: If the CSV is of another specification and an expensive ``auto`` run may not replace it.
        RunUnavailable, RunFailed: From the runner.
    """
    mode = mode or os.environ.get("BENCH_MODE", "auto")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if spec.backend == "export":
        raise ValueError(f"{spec.id} writes the reference file, not a result CSV; there is nothing to load")

    found = inspect(spec, smoke=smoke)
    state = found["state"]
    if mode == "run":
        return _measure(spec, smoke, runner)

    if state in ("fresh", "preliminary") or (state == "stale" and mode == "read"):
        if state == "stale":
            warnings.warn(f"{spec.id}: " + "; ".join(found["notes"]), stacklevel=2)
        return Loaded(
            spec, results.load_csv(found["path"]), found["path"], "csv", state, found["meta"], smoke, found["notes"]
        )

    # Missing, or stale in auto mode: only a cheap specification is measured without being asked.
    reason = "; ".join(found["notes"])
    if mode == "auto" and spec.cost.cost_class == "cheap":
        return _measure(spec, smoke, runner)
    commands = _commands(spec, smoke)
    if mode == "read":
        raise MissingResultError(f"{spec.id}: {reason}. Mode 'read' never measures. To produce it:\n{commands}")
    error = StaleResultError if state == "stale" else MissingResultError
    minutes = f" (about {spec.cost.est_minutes:g} min)" if spec.cost.est_minutes else ""
    raise error(
        f"{spec.id}: {reason}. It is an expensive benchmark{minutes}, so it is not measured on its own; "
        f"set mode='run' (or BENCH_MODE=run) to do it here, or produce it with:\n{commands}"
    )
