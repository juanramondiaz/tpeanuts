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

"""Copy the files a nuSQuIDS benchmark needs to the native WSL filesystem, and fingerprint what was copied.

The repository is on a synchronised Windows drive, which the WSL mount handles unreliably (silent write failures), and
nuSQuIDS only runs inside WSL; so the benchmark runs on a native copy that this script refreshes. It copies
``benchmarks/``, ``validation/nusquids/`` (the pure-nuSQuIDS helpers) and the tpeanuts reference file, streams them to
WSL as a tar archive, and writes ``MANIFEST.json`` beside them with the SHA-256 of every file and of the whole tree, the
commit of the source repository and whether it had local changes. ``benchmarks.nusquids.run`` records that fingerprint
in its results, so a result says exactly which code produced it.

Usage, from the repository root on Windows::

    python -m benchmarks.nusquids.sync_wsl                 # to ~/tpeanuts_bench in the default WSL distribution
    python -m benchmarks.nusquids.sync_wsl --target '~/elsewhere'
    python -m benchmarks.nusquids.sync_wsl --manifest-only   # no copy: fingerprint this checkout

Module contents:
    collect_files(...)
        The files to copy, relative to the repository root.
    build_manifest(...)
        The fingerprint of a list of files.
    write_manifest(...)
        Write ``MANIFEST.json`` at the repository root, for a copy that is moved by other means.
    sync(...)
        Stream the files to WSL and write the manifest there.
    main(...)
        Command-line entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = "~/tpeanuts_bench"
_INCLUDED_DIRECTORIES = ("benchmarks", "validation/nusquids")
_REFERENCE = "benchmarks/results/reference_tpeanuts.npz"
_EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
_EXCLUDED_RESULTS_PREFIX = "benchmarks/results/"
_INITS = ("validation/__init__.py",)


def collect_files(root: Path = ROOT) -> List[str]:
    """The files to copy, as sorted POSIX paths relative to ``root``.

    The code of ``benchmarks/`` and ``validation/nusquids/``, the ``validation`` package marker, and the tpeanuts
    reference file; the other result files stay where they were produced.

    Args:
        root: The repository root.

    Returns:
        The relative paths.

    Raises:
        FileNotFoundError: If the reference file has not been exported yet.
    """
    files = set()
    for directory in _INCLUDED_DIRECTORIES:
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root).as_posix()
            if not path.is_file() or _EXCLUDED_PARTS & set(path.parts) or path.suffix in (".pyc", ".pyo"):
                continue
            if relative.startswith(_EXCLUDED_RESULTS_PREFIX):
                continue
            files.add(relative)
    files.update(name for name in _INITS if (root / name).is_file())
    if not (root / _REFERENCE).is_file():
        raise FileNotFoundError(f"{_REFERENCE} does not exist; run 'python -m benchmarks.tpeanuts.export_reference'.")
    files.add(_REFERENCE)
    return sorted(files)


def build_manifest(files: Sequence[str], root: Path = ROOT) -> Dict[str, Any]:
    """The fingerprint of a set of files.

    Args:
        files: Relative paths, as returned by ``collect_files``.
        root: The repository root.

    Returns:
        ``{"files": {path: sha256}, "n_files", "tree_sha256", "source_commit", "source_dirty", "synced_utc"}``; the
        git fields are ``None`` when git is unavailable.
    """
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    tree = hashlib.sha256("".join(f"{name}:{digest}\n" for name, digest in sorted(hashes.items())).encode()).hexdigest()
    commit: Optional[str] = None
    dirty: Optional[bool] = None
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
            or None
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "benchmarks", "validation/nusquids"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "files": hashes,
        "n_files": len(hashes),
        "tree_sha256": tree,
        "source_commit": commit,
        "source_dirty": dirty,
        "synced_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_manifest(root: Path = ROOT) -> Dict[str, Any]:
    """Fingerprint the files of the benchmarks and write ``MANIFEST.json`` at the root of the repository.

    For the copies that are not made by ``sync`` (a ``git clone`` or an ``rsync`` to a cluster): the manifest travels
    with the code, and every result records it (``environment["manifest"]``).

    Args:
        root: The repository root.

    Returns:
        The manifest written.
    """
    manifest = build_manifest(collect_files(root), root)
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def sync(target: str = DEFAULT_TARGET, root: Path = ROOT) -> Dict[str, Any]:
    """Replace the code under ``target`` in WSL with the current files and write the manifest there.

    Only the directories ``benchmarks`` and ``validation`` inside ``target`` are replaced; the result files already
    written there by earlier runs are kept (``benchmarks/results``).

    Args:
        target: Directory in the WSL filesystem (``~`` is expanded by the WSL shell).
        root: The repository root.

    Returns:
        The manifest written.

    Raises:
        RuntimeError: If WSL cannot be run or the transfer fails.
    """
    files = collect_files(root)
    manifest = build_manifest(files, root)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in files:
            archive.add(root / name, arcname=name)
        payload = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    script = (
        f'set -e; D={target}; mkdir -p "$D/benchmarks/results"; '
        'find "$D/benchmarks" -mindepth 1 -maxdepth 1 ! -name results -exec rm -rf {} +; '
        'rm -rf "$D/validation"; tar -x -C "$D"'
    )
    command = ["wsl", "-e", "bash", "-c", script]
    done = subprocess.run(command, input=buffer.getvalue(), capture_output=True, timeout=300)
    if done.returncode != 0:
        raise RuntimeError(f"the transfer to WSL failed: {done.stderr.decode(errors='replace')[:500]}")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: sync and print the fingerprint."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default=DEFAULT_TARGET, help="directory in the WSL filesystem")
    parser.add_argument("--manifest-only", action="store_true", help="only write MANIFEST.json here; no copy to WSL")
    args = parser.parse_args(argv)
    if args.manifest_only:
        manifest = write_manifest()
        print(f"fingerprinted {manifest['n_files']} files; tree sha256 {manifest['tree_sha256'][:16]}")
        return 0
    manifest = sync(args.target)
    print(f"copied {manifest['n_files']} files to WSL:{args.target}; tree sha256 {manifest['tree_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
