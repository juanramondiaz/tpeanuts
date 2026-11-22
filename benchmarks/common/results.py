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

"""Result schema of the benchmarks: one record per (scenario, backend, phase, size), saved as JSON or CSV.

Every backend runs in its own process and environment and exchanges only these files, so the schema is the contract
between them. The rule of the measurement protocol that a timing is only recorded for a result that agrees with the
reference is part of the schema: a record whose status is ``"ok"`` and whose backend is not tpeanuts must carry a
passed agreement check, and a record with any other status carries no timing.

Module contents:
    SCHEMA_VERSION, BACKENDS, PHASES, STATUSES
        The schema version and the allowed values of the categorical fields.
    BenchmarkRecord
        One measurement (or one explained absence of it).
    ResultSet
        The records of one run, with the environment they were taken in.
    save_json(...), load_json(...)
        Write and read a ``ResultSet``, validating it.
    CSV_COLUMNS
        The columns of the CSV form: the record, the run and the environment flattened for reading.
    save_csv(...), load_csv(...), csv_metadata(...)
        The CSV form: one row per record, readable on its own and lossless (the full environment is kept in
        one column).
    save(...), add_spec_arguments(...)
        What the runners call: pick the format from the file suffix, and the command-line options that tie a
        CSV to the specification that produced it.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

SCHEMA_VERSION = 1

BACKENDS = ("tpeanuts", "peanuts", "nusquids")
PHASES = ("build", "evaluate")
STATUSES = ("ok", "unavailable", "disagrees", "failed", "skipped")

_TIMING_KEYS = ("min_s", "median_s", "iqr_s", "mean_s", "std_s", "n_repeats")


@dataclass
class BenchmarkRecord:
    """One measurement of one scenario with one backend, or the reason why there is none.

    Attributes:
        scenario: Name of the scenario (``scenarios.SCENARIOS``).
        backend: One of ``BACKENDS``.
        phase: ``"build"`` (object construction) or ``"evaluate"``.
        device: ``"cpu"`` or ``"cuda"``.
        dtype: Floating-point precision, e.g. ``"float64"``.
        size: The problem sizes of this record, e.g. ``{"n_energy": 1000, "n_nadir": 10}``.
        status: ``"ok"`` (timed), ``"unavailable"`` (the backend does not support the scenario), ``"disagrees"`` (its
            result does not match the reference, so it is not timed), ``"failed"`` (an error or a RAM budget) or
            ``"skipped"`` (left out on purpose, e.g. above the size limit of the run).
        options: How the case was run, e.g. ``{"method": "numerical", "nsteps": 100}``; free-form.
        timing: The dictionary of ``timing.TimingResult.to_dict``; ``None`` unless the status is ``"ok"``.
        agreement: The check against the reference, ``{"metric": str, "value": float, "tolerance": float,
            "passed": bool}``; ``None`` when there was none.
        peak_cuda_gb: Peak CUDA memory in GiB, if measured.
        peak_ram_gb: Resident memory of the process in GiB, if measured.
        note: Free text; required when the status is not ``"ok"``.
    """

    scenario: str
    backend: str
    phase: str
    device: str
    dtype: str
    size: Dict[str, int]
    status: str = "ok"
    options: Dict[str, Any] = field(default_factory=dict)
    timing: Optional[Dict[str, float]] = None
    agreement: Optional[Dict[str, Any]] = None
    peak_cuda_gb: Optional[float] = None
    peak_ram_gb: Optional[float] = None
    note: str = ""

    def validate(self) -> None:
        """Check the record against the schema and the correctness-first rule.

        Raises:
            ValueError: If a categorical field is outside its allowed values, a timing is missing or present against
                the status, an ``"ok"`` record of a reference backend has no passed agreement check, or a record that
                is not ``"ok"`` has no note.
        """
        where = f"{self.scenario}/{self.backend}/{self.phase}"
        if self.backend not in BACKENDS:
            raise ValueError(f"{where}: backend must be one of {BACKENDS}, got {self.backend!r}.")
        if self.phase not in PHASES:
            raise ValueError(f"{where}: phase must be one of {PHASES}, got {self.phase!r}.")
        if self.status not in STATUSES:
            raise ValueError(f"{where}: status must be one of {STATUSES}, got {self.status!r}.")
        if self.status == "ok":
            if self.timing is None or any(key not in self.timing for key in _TIMING_KEYS):
                raise ValueError(f"{where}: an 'ok' record needs a timing with the keys {_TIMING_KEYS}.")
            if self.backend != "tpeanuts" and not (self.agreement and self.agreement.get("passed") is True):
                raise ValueError(f"{where}: an 'ok' record of a reference backend needs a passed agreement check.")
        else:
            if self.timing is not None:
                raise ValueError(f"{where}: a record with status {self.status!r} must not carry a timing.")
            if not self.note:
                raise ValueError(f"{where}: a record with status {self.status!r} needs a note saying why.")


@dataclass
class ResultSet:
    """The records of one benchmark run and the environment it ran in.

    Attributes:
        label: Name of the run (e.g. ``"tpeanuts_cpu_float64"``).
        environment: The record of ``environment.capture_environment``.
        records: The measurements.
        schema_version: Version of this schema, for readers of old files.
    """

    label: str
    environment: Dict[str, Any]
    records: List[BenchmarkRecord] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        """Validate the schema version and every record.

        Raises:
            ValueError: If the version is not ``SCHEMA_VERSION`` or any record is invalid.
        """
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r} (this code reads {SCHEMA_VERSION}).")
        for record in self.records:
            record.validate()

    def to_dict(self) -> Dict[str, Any]:
        """The result set as a JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResultSet":
        """Rebuild a result set from ``to_dict`` output.

        Args:
            data: The dictionary.

        Returns:
            The result set (not validated; ``load_json`` validates).
        """
        records = [BenchmarkRecord(**record) for record in data["records"]]
        return cls(
            label=data["label"],
            environment=data["environment"],
            records=records,
            schema_version=data["schema_version"],
        )


def save_json(result_set: ResultSet, path: Union[str, Path]) -> Path:
    """Validate a result set and write it as JSON.

    Args:
        result_set: The result set to write.
        path: Destination file; its directory is created if needed.

    Returns:
        The path written.

    Raises:
        ValueError: If the result set is not valid (nothing is written).
    """
    result_set.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result_set.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_json(path: Union[str, Path]) -> ResultSet:
    """Read a result set from JSON and validate it.

    Args:
        path: The file written by ``save_json``.

    Returns:
        The validated result set.

    Raises:
        ValueError: If the content does not satisfy the schema.
    """
    result_set = ResultSet.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    result_set.validate()
    return result_set


# ---------------------------------------------------------------------------------------------------------------------
# CSV form
# ---------------------------------------------------------------------------------------------------------------------

_TIMING_COLUMNS = _TIMING_KEYS
_AGREEMENT_COLUMNS = ("metric", "value", "tolerance", "passed")

# Run and specification first, then the record, then the environment: a few columns flattened so the CSV can be read
# without parsing JSON, and the whole record in ``environment_json`` so that nothing is lost.
CSV_COLUMNS = (
    "label", "schema_version", "spec_id", "spec_hash", "smoke",
    "scenario", "backend", "phase", "device", "dtype", "status", "size", "options",
    *(f"timing_{key}" for key in _TIMING_COLUMNS),
    *(f"agreement_{key}" for key in _AGREEMENT_COLUMNS),
    "peak_cuda_gb", "peak_ram_gb", "note",
    "env_timestamp_utc", "env_host", "env_cpu_model", "env_cpu_count", "env_memory_gb", "env_gpus", "env_cuda_device",
    "env_threads", "env_python", "env_torch", "env_commit", "env_dirty", "env_job",
    "environment_json",
)  # fmt: skip


def _flat_environment(environment: Dict[str, Any]) -> Dict[str, Any]:
    """The readable columns of an environment record (``None`` where the record has no such entry)."""
    hardware = environment.get("hardware") or {}
    scheduler = environment.get("scheduler") or {}
    machine, job = scheduler.get("machine") or {}, scheduler.get("job") or {}
    threads = environment.get("threads") or {}
    return {
        "env_timestamp_utc": environment.get("timestamp_utc"),
        "env_host": machine.get("Machine") or ("local" if not scheduler else None),
        "env_cpu_model": hardware.get("cpu_model"),
        "env_cpu_count": environment.get("cpu_count"),
        "env_memory_gb": hardware.get("memory_gb"),
        "env_gpus": "; ".join(f"{g.get('name')} ({g.get('memory')})" for g in hardware.get("gpus") or []) or None,
        "env_cuda_device": (environment.get("cuda") or {}).get("device"),
        "env_threads": threads.get("torch") or threads.get("OMP_NUM_THREADS"),
        "env_python": environment.get("python"),
        "env_torch": (environment.get("packages") or {}).get("torch"),
        "env_commit": (environment.get("git") or {}).get("commit"),
        "env_dirty": (environment.get("git") or {}).get("dirty"),
        "env_job": f"{job['ClusterId']}.{job.get('ProcId', 0)}" if "ClusterId" in job else None,
    }


def _cell(value: Any) -> Any:
    """A value as a CSV cell: ``None`` is empty, containers are compact JSON."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def save_csv(
    result_set: ResultSet,
    path: Union[str, Path],
    *,
    spec_hash: Optional[str] = None,
    spec_id: Optional[str] = None,
    smoke: bool = False,
) -> Path:
    """Validate a result set and write it as CSV, one row per record.

    Args:
        result_set: The result set to write; it must have at least one record (the run metadata lives in the rows).
        path: Destination file; its directory is created if needed.
        spec_hash: ``Spec.spec_hash`` of the specification that was run, so that a reader can tell a stale file.
        spec_id: ``Spec.id`` of that specification.
        smoke: Whether this is a quick run; a reader must not present it as a final result.

    Returns:
        The path written.

    Raises:
        ValueError: If the result set is not valid or has no records (nothing is written).
    """
    result_set.validate()
    if not result_set.records:
        raise ValueError("a CSV needs at least one record: the run metadata is stored in the rows.")
    flat = _flat_environment(result_set.environment)
    environment_json = json.dumps(result_set.environment, sort_keys=True, separators=(",", ":"))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in result_set.records:
            timing, agreement = record.timing or {}, record.agreement or {}
            row = {
                "label": result_set.label,
                "schema_version": result_set.schema_version,
                "spec_id": spec_id,
                "spec_hash": spec_hash,
                "smoke": bool(smoke),
                "scenario": record.scenario,
                "backend": record.backend,
                "phase": record.phase,
                "device": record.device,
                "dtype": record.dtype,
                "status": record.status,
                "size": record.size,
                "options": record.options,
                **{f"timing_{key}": timing.get(key) for key in _TIMING_COLUMNS},
                **{f"agreement_{key}": agreement.get(key) for key in _AGREEMENT_COLUMNS},
                "peak_cuda_gb": record.peak_cuda_gb,
                "peak_ram_gb": record.peak_ram_gb,
                "note": record.note,
                **flat,
                "environment_json": environment_json,
            }
            writer.writerow({key: _cell(value) for key, value in row.items()})
    return target


def _read_rows(path: Union[str, Path]) -> List[Dict[str, str]]:
    """The rows of a result CSV; a file without the expected columns or without rows is not one."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in CSV_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: not a benchmark result CSV (missing columns {missing[:3]}...).")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: the CSV has no rows.")
    return rows


def _flag(text: str) -> Optional[bool]:
    return None if text == "" else text == "True"


def _number(text: str) -> Optional[float]:
    return None if text == "" else float(text)


def csv_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """The run metadata of a result CSV, without building the records.

    Args:
        path: A file written by ``save_csv``.

    Returns:
        A dictionary with ``label``, ``spec_id`` and ``spec_hash`` (``None`` when absent), ``smoke``, ``n_rows`` and the
        flattened environment columns (``env_*``) of the first row.

    Raises:
        ValueError: If the file is not a result CSV.
    """
    rows = _read_rows(path)
    first = rows[0]
    meta: Dict[str, Any] = {
        "label": first["label"],
        "spec_id": first["spec_id"] or None,
        "spec_hash": first["spec_hash"] or None,
        "smoke": first["smoke"] == "True",
        "n_rows": len(rows),
    }
    meta.update({key: first[key] or None for key in CSV_COLUMNS if key.startswith("env_")})
    return meta


def load_csv(path: Union[str, Path]) -> ResultSet:
    """Read a result set from a CSV written by ``save_csv`` and validate it.

    Args:
        path: The file.

    Returns:
        The validated result set, equal to the one that was written.

    Raises:
        ValueError: If the file is not a result CSV or its content does not satisfy the schema.
    """
    rows = _read_rows(path)
    records = []
    for row in rows:
        timing: Dict[str, float] = {
            key: float(row[f"timing_{key}"]) for key in _TIMING_COLUMNS if row[f"timing_{key}"] != ""
        }
        if "n_repeats" in timing:
            timing["n_repeats"] = int(timing["n_repeats"])
        agreement = {
            "metric": row["agreement_metric"],
            "value": _number(row["agreement_value"]),
            "tolerance": _number(row["agreement_tolerance"]),
            "passed": _flag(row["agreement_passed"]),
        }
        records.append(
            BenchmarkRecord(
                scenario=row["scenario"],
                backend=row["backend"],
                phase=row["phase"],
                device=row["device"],
                dtype=row["dtype"],
                size={key: int(value) for key, value in json.loads(row["size"]).items()},
                status=row["status"],
                options=json.loads(row["options"] or "{}"),
                timing=timing or None,
                agreement=agreement if row["agreement_metric"] != "" else None,
                peak_cuda_gb=_number(row["peak_cuda_gb"]),
                peak_ram_gb=_number(row["peak_ram_gb"]),
                note=row["note"],
            )
        )
    result_set = ResultSet(
        label=rows[0]["label"],
        environment=json.loads(rows[0]["environment_json"]),
        records=records,
        schema_version=int(rows[0]["schema_version"]),
    )
    result_set.validate()
    return result_set


def add_spec_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options that tie a CSV to the specification that produced it (``--spec-hash`` and ``--spec-id``)."""
    parser.add_argument("--spec-hash", default=None, help="spec_hash of the YAML specification (recorded in a CSV)")
    parser.add_argument("--spec-id", default=None, help="<series>/<name> of the YAML specification (recorded in a CSV)")


def save(
    result_set: ResultSet,
    path: Union[str, Path],
    *,
    spec_hash: Optional[str] = None,
    spec_id: Optional[str] = None,
    smoke: bool = False,
) -> Path:
    """Write a result set in the format of the file suffix: ``.csv`` as CSV, anything else as JSON.

    Args:
        result_set: The result set.
        path: Destination file.
        spec_hash: Recorded in a CSV (a JSON has no place for it and ignores it).
        spec_id: Recorded in a CSV (ignored by a JSON).
        smoke: Recorded in a CSV (ignored by a JSON).

    Returns:
        The path written.
    """
    if Path(path).suffix.lower() == ".csv":
        return save_csv(result_set, path, spec_hash=spec_hash, spec_id=spec_id, smoke=smoke)
    return save_json(result_set, path)
