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

"""Specification of one benchmark: what is measured, where the CSV goes and how to run it on a cluster (a YAML file).

Every benchmark has one file ``benchmarks/suites/<series>/<name>.yaml``. It is the single source of truth for three
consumers: the notebook that shows the result (is the CSV there and does it belong to this specification?), the local
runner (``to_argv`` is the command line of the backend) and ``benchmarks/condor/render.py`` (the HTCondor submit file).
The CSV is written next to the YAML unless ``output.dir`` says otherwise.

Fields (see ``Spec``): ``name`` and ``series`` (they must agree with the file name and its folder), ``backend``
(``tpeanuts``, ``peanuts``, ``nusquids`` or ``export`` for the tpeanuts reference file), ``scenarios``, ``methods``
(tpeanuts) or ``modes`` (peanuts), ``device``, ``dtype``, ``repeats``, ``max_points`` (and, for nuSQuIDS,
``max_points_by_scenario``: a limit per scenario, because the cost per point differs by orders of magnitude between the
Earth at MeV and the atmosphere at GeV), ``smoke`` (the overrides of a quick run), ``output``, ``cost``
(``cheap`` runs on demand from a notebook, ``expensive`` only on the cluster)
and ``condor`` (resources and the specifications that must have finished first). A key that does not apply to
the backend, or that the schema does not know, is an error, so a misspelt option is never ignored silently.

``spec_hash`` fingerprints what changes the measurement (backend, scenarios, methods, precision, repeats, sizes,
threads, resources and whether it is a quick run) and not where the file is written, so a CSV taken with a different
configuration is detected as stale. ``yaml`` is imported only when a file is read: the shared harness must import in
environments that do not have it.

Module contents:
    SUITES_DIR
        Directory that holds the specifications.
    SpecError
        A specification that does not follow the schema.
    CondorSpec, CostSpec, OutputSpec
        The resource, cost and output sections.
    Spec
        One benchmark: validation, ``spec_hash``, ``to_argv``, ``env_vars`` and the output paths.
    load_spec(...), discover(...)
        Read one file, or every specification under a directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from benchmarks.common.scenarios import SCENARIOS

SUITES_DIR = Path(__file__).resolve().parents[1] / "suites"

BACKENDS = ("tpeanuts", "peanuts", "nusquids", "export")
TPEANUTS_METHODS = ("numerical", "analytic", "analytic_perturbative")
PEANUTS_MODES = ("analytical", "numerical")
DTYPES = ("float32", "float64")
DEVICES = ("cpu", "cuda")
COST_CLASSES = ("cheap", "expensive")
SMOKE_KEYS = ("repeats", "max_points", "max_points_numerical")

# Keys that each backend accepts besides the common ones.
_COMMON_KEYS = ("name", "series", "backend", "description", "optional", "output", "cost", "condor", "smoke")
_BACKEND_KEYS = {
    "tpeanuts": ("scenarios", "methods", "device", "dtype", "nsteps", "repeats", "max_points", "threads", "label"),
    "peanuts": ("scenarios", "modes", "repeats", "max_points", "max_points_numerical", "label"),
    "nusquids": ("scenarios", "repeats", "max_points", "max_points_by_scenario", "threads", "label"),
    "export": ("nsteps",),
}
_MEMORY = re.compile(r"^\d+ ?(MB|GB|TB)$")


class SpecError(ValueError):
    """A specification that does not follow the schema; the message names the file and the field."""


def _only(mapping: Mapping[str, Any], allowed: Tuple[str, ...], where: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise SpecError(f"{where}: unknown or inapplicable key(s) {unknown}; allowed: {sorted(allowed)}")


def _choice(value: Any, options: Tuple[str, ...], where: str) -> str:
    if value not in options:
        raise SpecError(f"{where}: {value!r} is not one of {list(options)}")
    return str(value)


def _integer(value: Any, where: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpecError(f"{where}: must be an integer >= {minimum}, not {value!r}")
    return int(value)


def _names(value: Any, options: Tuple[str, ...], where: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise SpecError(f"{where}: must be a non-empty list")
    return tuple(_choice(item, options, where) for item in value)


@dataclass(frozen=True)
class CondorSpec:
    """Resources of the cluster job and the specifications that must have finished before it starts.

    Attributes:
        request_cpus: CPUs requested (and, for tpeanuts, the CPU threads used).
        request_memory: Memory requested, as HTCondor writes it (``"32 GB"``).
        request_gpus: GPUs requested.
        after: Identifiers (``"<series>/<name>"``) of the specifications whose output this one needs first.
    """

    request_cpus: int = 1
    request_memory: str = "8 GB"
    request_gpus: int = 0
    after: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> "CondorSpec":
        """Validate the ``condor`` section."""
        _only(data, ("request_cpus", "request_memory", "request_gpus", "after"), where)
        memory = str(data.get("request_memory", cls.request_memory))
        if not _MEMORY.match(memory):
            raise SpecError(f"{where}.request_memory: {memory!r} must look like '32 GB'")
        after = data.get("after", [])
        if not isinstance(after, (list, tuple)) or not all(isinstance(a, str) and "/" in a for a in after):
            raise SpecError(f"{where}.after: must be a list of '<series>/<name>' identifiers")
        return cls(
            request_cpus=_integer(data.get("request_cpus", 1), f"{where}.request_cpus"),
            request_memory=memory,
            request_gpus=_integer(data.get("request_gpus", 0), f"{where}.request_gpus", minimum=0),
            after=tuple(after),
        )


@dataclass(frozen=True)
class CostSpec:
    """How expensive the benchmark is: ``cheap`` ones may be run on demand by a notebook, ``expensive`` ones not.

    Attributes:
        cost_class: ``"cheap"`` or ``"expensive"``.
        est_minutes: Estimated wall-clock minutes of the full run (informative; it is not enforced).
    """

    cost_class: str = "expensive"
    est_minutes: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> "CostSpec":
        """Validate the ``cost`` section."""
        _only(data, ("class", "est_minutes"), where)
        minutes = data.get("est_minutes")
        if minutes is not None and (isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or minutes <= 0):
            raise SpecError(f"{where}.est_minutes: must be a positive number, not {minutes!r}")
        return cls(_choice(data.get("class", "expensive"), COST_CLASSES, f"{where}.class"), minutes)


@dataclass(frozen=True)
class OutputSpec:
    """Where the CSV is written.

    Attributes:
        dir: Directory of the CSV, relative to the YAML file; ``"."`` (next to it) by default.
        csv: File name of the CSV; ``<name>.csv`` when empty.
    """

    dir: str = "."
    csv: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> "OutputSpec":
        """Validate the ``output`` section."""
        _only(data, ("dir", "csv"), where)
        csv = str(data.get("csv", ""))
        if csv and (not csv.endswith(".csv") or "/" in csv or "\\" in csv):
            raise SpecError(f"{where}.csv: must be a plain file name ending in .csv, not {csv!r}")
        return cls(str(data.get("dir", ".")), csv)


@dataclass(frozen=True)
class Spec:
    """One benchmark specification (the content of one YAML file).

    Attributes:
        name, series: Identify the benchmark; the file is ``suites/<series>/<name>.yaml``.
        backend: ``tpeanuts``, ``peanuts``, ``nusquids`` or ``export``.
        description: One line for humans.
        optional: Not part of the standard sequence of runs (``render.py --chain`` leaves it out unless asked).
        scenarios: Scenarios of ``common.scenarios.SCENARIOS`` that are timed (the runner's choice when empty).
        methods: tpeanuts propagation methods.
        modes: PEANUTS Earth modes.
        device, dtype: tpeanuts device and precision (the other backends are CPU float64 by construction).
        nsteps: Earth segments of tpeanuts (the runner default when ``None``).
        repeats: Timed repetitions of each case.
        max_points: Grids above this many points are skipped; 0 means no limit.
        max_points_numerical: The same limit for the numerical mode of PEANUTS.
        max_points_by_scenario: nuSQuIDS only. ``{scenario: limit}``; a scenario listed here uses its own limit instead
            of ``max_points`` (0 means no limit). A quick run ignores it and uses the ``max_points`` of ``smoke``.
        threads: CPU threads; the CPUs requested when ``None`` (tpeanuts) and 1 for the single-thread backends.
        label: Name of the run; the specification name when empty.
        smoke: Overrides applied to a quick run (``repeats``, ``max_points``, ``max_points_numerical``).
        output, cost, condor: The corresponding sections.
        path: The YAML file this was read from, if any.
    """

    name: str
    series: str
    backend: str
    description: str = ""
    optional: bool = False
    scenarios: Tuple[str, ...] = ()
    methods: Tuple[str, ...] = ()
    modes: Tuple[str, ...] = ()
    device: str = "cpu"
    dtype: str = "float64"
    nsteps: Optional[int] = None
    repeats: int = 7
    max_points: Optional[int] = None
    max_points_numerical: Optional[int] = None
    max_points_by_scenario: Dict[str, int] = field(default_factory=dict)
    threads: Optional[int] = None
    label: str = ""
    smoke: Dict[str, int] = field(default_factory=dict)
    output: OutputSpec = field(default_factory=OutputSpec)
    cost: CostSpec = field(default_factory=CostSpec)
    condor: CondorSpec = field(default_factory=CondorSpec)
    path: Optional[Path] = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, path: Optional[Path] = None) -> "Spec":
        """Validate a parsed YAML mapping and build the specification.

        Args:
            data: The mapping read from the YAML file.
            path: The file it came from (used in messages and to locate the CSV).

        Returns:
            The validated ``Spec``.

        Raises:
            SpecError: If a key is unknown or does not apply to the backend, a value is not allowed, or the name and
                series do not agree with the path.
        """
        where = str(path) if path is not None else "specification"
        if not isinstance(data, Mapping):
            raise SpecError(f"{where}: the file must hold a mapping")
        backend = _choice(data.get("backend"), BACKENDS, f"{where}: backend")
        accepted = _BACKEND_KEYS[backend]
        _only(data, _COMMON_KEYS + accepted, where)
        for key in ("name", "series"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise SpecError(f"{where}: '{key}' is required")
        if path is not None and (data["name"] != path.stem or data["series"] != path.parent.name):
            raise SpecError(f"{where}: name/series {data['series']}/{data['name']} must match the file path")

        smoke = dict(data.get("smoke", {}) or {})
        _only(smoke, tuple(k for k in SMOKE_KEYS if k in accepted), f"{where}: smoke")
        for key, value in smoke.items():
            _integer(value, f"{where}: smoke.{key}", minimum=0 if key.startswith("max_points") else 1)

        kwargs: Dict[str, Any] = {}
        if "scenarios" in accepted and "scenarios" in data:
            kwargs["scenarios"] = _names(data["scenarios"], tuple(SCENARIOS), f"{where}: scenarios")
        if "methods" in accepted:
            kwargs["methods"] = _names(data.get("methods", ["numerical"]), TPEANUTS_METHODS, f"{where}: methods")
        if "modes" in accepted:
            kwargs["modes"] = _names(data.get("modes", ["analytical"]), PEANUTS_MODES, f"{where}: modes")
        if "device" in accepted:
            kwargs["device"] = _choice(data.get("device", "cpu"), DEVICES, f"{where}: device")
            kwargs["dtype"] = _choice(data.get("dtype", "float64"), DTYPES, f"{where}: dtype")
        if "repeats" in accepted:
            kwargs["repeats"] = _integer(data.get("repeats", 7), f"{where}: repeats")
        for key in ("max_points", "max_points_numerical", "nsteps", "threads"):
            if key in accepted and key in data:
                kwargs[key] = _integer(data[key], f"{where}: {key}", minimum=0 if key.startswith("max_p") else 1)
        if "max_points_by_scenario" in accepted and data.get("max_points_by_scenario") is not None:
            per_scenario = data["max_points_by_scenario"]
            if not isinstance(per_scenario, Mapping) or not per_scenario:
                raise SpecError(f"{where}: max_points_by_scenario must be a non-empty mapping scenario: limit")
            kwargs["max_points_by_scenario"] = {
                _choice(name, tuple(SCENARIOS), f"{where}: max_points_by_scenario"): _integer(
                    limit, f"{where}: max_points_by_scenario.{name}", minimum=0
                )
                for name, limit in per_scenario.items()
            }
        if backend in ("peanuts", "nusquids") and kwargs.get("threads", 1) != 1:
            raise SpecError(f"{where}: {backend} is single-threaded by construction; threads must be 1")
        if backend in ("tpeanuts", "peanuts", "nusquids") and "max_points" not in data:
            raise SpecError(f"{where}: 'max_points' is required (0 means no limit), so the sizes are explicit")

        optional = data.get("optional", False)
        if not isinstance(optional, bool):
            raise SpecError(f"{where}: optional must be true or false, not {optional!r}")
        condor = CondorSpec.from_dict(data.get("condor", {}) or {}, f"{where}: condor")
        if kwargs.get("device") == "cuda" and condor.request_gpus < 1:
            raise SpecError(f"{where}: device 'cuda' needs condor.request_gpus >= 1")
        if kwargs.get("device", "cpu") == "cpu" and condor.request_gpus:
            raise SpecError(f"{where}: a CPU benchmark must not request a GPU")
        return cls(
            name=data["name"],
            series=data["series"],
            backend=backend,
            description=str(data.get("description", "")),
            optional=optional,
            label=str(data.get("label", "")),
            smoke=smoke,
            output=OutputSpec.from_dict(data.get("output", {}) or {}, f"{where}: output"),
            cost=CostSpec.from_dict(data.get("cost", {}) or {}, f"{where}: cost"),
            condor=condor,
            path=path,
            **kwargs,
        )

    # ------------------------------------------------------------------ derived values

    @property
    def id(self) -> str:
        """``"<series>/<name>"``, the identifier other specifications use in ``condor.after``."""
        return f"{self.series}/{self.name}"

    @property
    def effective_threads(self) -> int:
        """CPU threads of the run: the explicit value, else the CPUs requested for tpeanuts, else 1."""
        if self.threads is not None:
            return self.threads
        return self.condor.request_cpus if self.backend in ("tpeanuts", "export") else 1

    def effective(self, smoke: bool = False) -> Dict[str, Any]:
        """The values that define the measurement, with the quick-run overrides applied when ``smoke``.

        Args:
            smoke: Apply the ``smoke`` overrides and mark the run as a quick one.

        Returns:
            A JSON-serialisable dictionary; it is what ``spec_hash`` fingerprints.
        """
        values: Dict[str, Any] = {
            "backend": self.backend,
            "scenarios": list(self.scenarios),
            "methods": list(self.methods),
            "modes": list(self.modes),
            "device": self.device,
            "dtype": self.dtype,
            "nsteps": self.nsteps,
            "repeats": self.repeats,
            "max_points": self.max_points,
            "max_points_numerical": self.max_points_numerical,
            "threads": self.effective_threads,
            "request_cpus": self.condor.request_cpus,
            "request_memory": self.condor.request_memory,
            "request_gpus": self.condor.request_gpus,
            "smoke": bool(smoke),
        }
        if smoke:
            values.update(self.smoke)
        elif self.max_points_by_scenario:
            # Only when set, so the hash of every other specification is unchanged; a quick run never uses it.
            values["max_points_by_scenario"] = dict(sorted(self.max_points_by_scenario.items()))
        return values

    def spec_hash(self, smoke: bool = False) -> str:
        """Short SHA-256 of what defines the measurement (not of where the CSV is written).

        Args:
            smoke: Hash the quick-run variant.

        Returns:
            The first 16 hexadecimal digits.
        """
        text = json.dumps(self.effective(smoke), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @property
    def run_label(self) -> str:
        """The name of the run recorded in the result: the label, else the specification name."""
        return self.label or self.name

    def output_csv(self, smoke: bool = False) -> Path:
        """Path of the CSV: next to the YAML (or in ``output.dir``); a quick run has its own file.

        Args:
            smoke: The path of the quick run, which never overwrites the real one.

        Returns:
            The absolute path when the specification has a file, otherwise relative to the current directory.
        """
        base = (self.path.parent if self.path is not None else Path(".")) / self.output.dir
        stem = self.output.csv[: -len(".csv")] if self.output.csv else self.name
        path = base / f"{stem}{'_smoke' if smoke else ''}.csv"
        return path.resolve() if self.path is not None else path

    def to_argv(self, *, smoke: bool = False, output: Optional[Path] = None) -> List[str]:
        """Command line of the backend in the form ``run.sh`` takes: the backend name, then the flags of its runner.

        Every size and option is written explicitly, so the command does not depend on the defaults of the runner. The
        CPU threads are not an argument: ``run.sh`` reads them from ``BENCH_THREADS`` (see ``env_vars``). The
        specification id and hash are passed so that the CSV records which specification produced it.

        Args:
            smoke: Add ``--smoke`` and the quick-run overrides, and write to the quick-run CSV.
            output: CSV path to write (the specification's own by default).

        Returns:
            The list of arguments.
        """
        eff = self.effective(smoke)
        out = str(output if output is not None else self.output_csv(smoke))
        argv: List[str] = [self.backend]
        if self.backend == "export":
            if self.nsteps is not None:
                argv += ["--nsteps", str(self.nsteps)]
            return argv
        if self.scenarios:
            argv += ["--scenarios", *self.scenarios]
        if self.backend == "tpeanuts":
            argv += ["--methods", *self.methods, "--device", self.device, "--dtype", self.dtype]
            if self.nsteps is not None:
                argv += ["--nsteps", str(self.nsteps)]
        if self.backend == "peanuts":
            argv += ["--modes", *self.modes]
        argv += ["--repeats", str(eff["repeats"]), "--max-points", str(eff["max_points"])]
        if self.backend == "peanuts" and eff["max_points_numerical"] is not None:
            argv += ["--max-points-numerical", str(eff["max_points_numerical"])]
        if self.backend == "nusquids" and eff.get("max_points_by_scenario"):
            argv += ["--max-points-scenario", *(f"{k}={v}" for k, v in eff["max_points_by_scenario"].items())]
        if smoke:
            argv.append("--smoke")
        label = self.run_label + ("_smoke" if smoke else "")
        return argv + ["--label", label, "--output", out, "--spec-id", self.id, "--spec-hash", self.spec_hash(smoke)]

    def env_vars(self) -> Dict[str, str]:
        """Environment variables of the job: ``BENCH_THREADS`` for tpeanuts (``run.sh`` turns it into ``--threads``)."""
        if self.backend in ("tpeanuts", "export"):
            return {"BENCH_THREADS": str(self.effective_threads)}
        return {}


def load_spec(path: Union[str, Path]) -> Spec:
    """Read and validate one specification file.

    Args:
        path: The YAML file (``suites/<series>/<name>.yaml``).

    Returns:
        The ``Spec``, with ``path`` set.

    Raises:
        SpecError: If the file does not follow the schema or cannot be parsed.
    """
    import yaml  # local: the shared harness must import where PyYAML is not installed

    file = Path(path).resolve()
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise SpecError(f"{file}: not valid YAML: {error}") from error
    return Spec.from_dict(data, path=file)


def discover(root: Union[str, Path] = SUITES_DIR) -> List[Spec]:
    """Every specification under ``root``, sorted by identifier; the ``after`` references are checked.

    Args:
        root: Directory searched (``SUITES_DIR`` by default).

    Returns:
        The specifications.

    Raises:
        SpecError: If one is invalid, two share an identifier, or ``condor.after`` names an unknown specification.
    """
    specs = [load_spec(p) for p in sorted(Path(root).rglob("*.yaml"))]
    ids = [s.id for s in specs]
    for duplicated in sorted({i for i in ids if ids.count(i) > 1}):
        raise SpecError(f"duplicated specification {duplicated}")
    for spec in specs:
        for dependency in spec.condor.after:
            if dependency not in ids:
                raise SpecError(f"{spec.id}: condor.after names {dependency}, which does not exist")
    return sorted(specs, key=lambda s: s.id)
