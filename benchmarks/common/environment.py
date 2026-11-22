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

"""Record of the environment a benchmark ran in, and the control of its CPU threads.

A timing is only meaningful together with the machine and the software that produced it, so every result file carries
this record (see ``results.ResultSet``). Nothing here imports a reference package; ``torch`` is imported only if it is
already installed, and only inside the functions that ask about it.

Module contents:
    THREAD_VARIABLES
        The environment variables that fix the CPU thread count of the numerical libraries.
    set_threads(...)
        Fix the CPU thread count of the process.
    capture_environment(...)
        The environment record as a JSON-serialisable dictionary.
    parse_class_ad(...)
        The attributes of an HTCondor ClassAd file that identify a job and its machine.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)

_REPOSITORY = Path(__file__).resolve().parents[2]
_MANIFEST = _REPOSITORY / "MANIFEST.json"

# ClassAd attributes of the job (``_CONDOR_JOB_AD``) and of the machine (``_CONDOR_MACHINE_AD``) that HTCondor writes
# into the sandbox of a running job; only the ones that say which resources the timings were taken with are kept.
_CONDOR_JOB_ATTRIBUTES = (
    "ClusterId",
    "ProcId",
    "RequestCpus",
    "RequestMemory",
    "RequestGpus",
    "RequestDisk",
    "JobUniverse",
)
_CONDOR_MACHINE_ATTRIBUTES = (
    "Machine",
    "Name",
    "Cpus",
    "TotalCpus",
    "TotalSlotCpus",
    "Memory",
    "TotalMemory",
    "GPUs",
    "TotalGPUs",
    "AssignedGPUs",
    "GPUs_DeviceName",
    "GPUs_DriverVersion",
    "GPUs_CUDACapability",
    "GPUs_GlobalMemoryMb",
    "OpSysAndVer",
    "CondorVersion",
)


def set_threads(n_threads: int) -> None:
    """Fix the CPU thread count of the process, for every backend that honours it.

    Sets the thread environment variables (``THREAD_VARIABLES``) and, for ``torch`` and ``numba`` when this process has
    already imported them, their own thread counts. The environment variables only take effect for libraries imported
    afterwards, so call this before importing the backend being timed, or run each backend in a process started with
    the variables already set.

    Args:
        n_threads: Number of threads, at least 1.

    Raises:
        ValueError: If ``n_threads`` is less than 1.
    """
    if n_threads < 1:
        raise ValueError(f"n_threads must be at least 1, got {n_threads!r}.")
    for name in THREAD_VARIABLES:
        os.environ[name] = str(n_threads)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(n_threads)
    if "numba" in sys.modules:
        sys.modules["numba"].set_num_threads(min(n_threads, sys.modules["numba"].config.NUMBA_NUM_THREADS))


def capture_environment(
    packages: Sequence[str] = ("numpy", "torch"), *, repository: Optional[Path] = None
) -> Dict[str, Any]:
    """The environment record of this process, as a JSON-serialisable dictionary.

    Args:
        packages: Distribution names whose installed versions are recorded (``None`` for one that is not installed).
        repository: Repository whose git state is recorded; the one this file belongs to by default.

    Returns:
        A dictionary with the keys ``timestamp_utc``, ``python``, ``platform``, ``machine``, ``cpu_count``,
        ``threads`` (the thread variables and, when this process has imported ``torch``, its thread count),
        ``packages``, ``git`` (``commit`` and ``dirty``, ``None`` when unavailable) and ``cuda`` (``available`` and
        the device name; always unavailable in a process that has not imported ``torch``, which is how the
        reference packages, that do not use it, are recorded without loading it), ``hardware`` (CPU model, memory and
        the GPUs reported by ``nvidia-smi``, which needs no ``torch``), ``scheduler`` (the HTCondor job and machine
        attributes when the process runs inside a job, else ``None``) and ``manifest`` (the fingerprint written by
        ``benchmarks.nusquids.sync_wsl --manifest-only``, ``None`` if there is none).
    """
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "threads": _threads(),
        "packages": {name: _version(name) for name in packages},
        "git": _git_state(repository or _REPOSITORY),
        "cuda": _cuda(),
        "hardware": _hardware(),
        "scheduler": _scheduler(),
        "manifest": _manifest(),
    }


def parse_class_ad(path: Path, attributes: Sequence[str]) -> Dict[str, Any]:
    """The requested attributes of an HTCondor ClassAd file (lines ``Name = value``).

    Only simple ``name = value`` lines are read; string values lose their quotes, numbers become numbers, and anything
    else is kept as text. An unreadable file gives an empty dictionary.

    Args:
        path: The ClassAd file.
        attributes: The attribute names to keep (matched case-insensitively, as ClassAds are).

    Returns:
        The attributes found, by the name given in ``attributes``.
    """
    wanted = {name.lower(): name for name in attributes}
    found: Dict[str, Any] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return found
    for line in lines:
        name, separator, value = line.partition("=")
        key = wanted.get(name.strip().lower())
        if key is None or not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            found[key] = value[1:-1]
            continue
        try:
            found[key] = int(value)
        except ValueError:
            try:
                found[key] = float(value)
            except ValueError:
                found[key] = value
    return found


def _scheduler() -> Optional[Dict[str, Any]]:
    """The HTCondor job and machine of this process, or ``None`` outside a job."""
    job_ad, machine_ad = os.environ.get("_CONDOR_JOB_AD"), os.environ.get("_CONDOR_MACHINE_AD")
    if not (job_ad or machine_ad or os.environ.get("_CONDOR_SLOT")):
        return None
    return {
        "name": "htcondor",
        "slot": os.environ.get("_CONDOR_SLOT"),
        "job": parse_class_ad(Path(job_ad), _CONDOR_JOB_ATTRIBUTES) if job_ad else {},
        "machine": parse_class_ad(Path(machine_ad), _CONDOR_MACHINE_ATTRIBUTES) if machine_ad else {},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _hardware() -> Dict[str, Any]:
    """The CPU model, the memory and the GPUs of this machine, from the operating system and ``nvidia-smi``."""
    cpu_model = platform.processor() or None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    try:
        memory_gb: Optional[float] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (AttributeError, ValueError, OSError):
        memory_gb = None
    gpus = []
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        for line in listing.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3:
                gpus.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    except (OSError, subprocess.SubprocessError):
        pass
    return {"cpu_model": cpu_model, "memory_gb": memory_gb, "gpus": gpus}


def _manifest() -> Optional[Dict[str, Any]]:
    """The fingerprint of the copy of the code (``MANIFEST.json`` at the repository root), without the file list."""
    try:
        data = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    keys = ("tree_sha256", "n_files", "source_commit", "source_dirty", "synced_utc")
    return {key: data.get(key) for key in keys}


def _version(name: str) -> Optional[str]:
    """Installed version of a distribution, or ``None``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _threads() -> Dict[str, Any]:
    """The thread variables as set, and the thread count of ``torch`` when this process has imported it."""
    threads: Dict[str, Any] = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    torch = sys.modules.get("torch")
    threads["torch"] = None if torch is None else torch.get_num_threads()
    return threads


def _git_state(repository: Path) -> Dict[str, Any]:
    """The commit and whether the working tree has uncommitted changes; ``None`` values when git is unavailable."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository, capture_output=True, text=True, check=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": bool(status.strip())}


def _cuda() -> Dict[str, Any]:
    """Whether CUDA is available to ``torch`` (if this process has imported it) and the name of the first device."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return {"available": False, "device": None}
    return {"available": True, "device": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda}
