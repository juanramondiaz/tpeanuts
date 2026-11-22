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

"""Download, verify and remove the datasets of the registry.

Every download goes to ``<file>.part``, is hashed while it streams, is compared with the registry's
SHA-256 and only then atomically moved into place, so a dataset directory never holds a truncated or
corrupted file under its final name. Files that are already present and valid are skipped. Network
failures and checksum mismatches raise ``DataDownloadError`` with the URL and the destination, so the
file can also be fetched by hand.

Module contents:
    DataDownloadError
        A download failed or its checksum did not match.
    FetchReport, VerifyReport
        What ``fetch_dataset`` and ``verify_dataset`` did / found.
    sha256_of(...)
        Streaming SHA-256 of a file.
    download_file(...)
        Download one file with verification; returns whether it downloaded.
    fetch_dataset(...)
        Download every (selected) file of a dataset, from the registry URLs or from a
        ``source`` directory/URL (a local copy or an institutional mirror).
    dataset_present(...)
        Cheap check (existence and size, no hashing) that a dataset is complete on disk.
    verify_dataset(...)
        Check the files of a dataset against the registry.
    remove_dataset(...)
        Delete a dataset directory.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import os
import shutil
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from tpeanuts.data import dataset_dir
from tpeanuts.data.registry import Dataset, DatasetFile, get_dataset

_CHUNK = 1 << 20
_USER_AGENT = "tpeanuts-data"
ProgressFn = Callable[[str, int, int], None]


class DataDownloadError(RuntimeError):
    """A dataset file could not be downloaded or failed its checksum."""


@dataclass
class FetchReport:
    """Outcome of ``fetch_dataset``.

    Attributes:
        downloaded: Relative paths that were downloaded now.
        skipped: Relative paths already present with the right checksum.
    """

    downloaded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


@dataclass
class VerifyReport:
    """Outcome of ``verify_dataset``.

    Attributes:
        ok: Files present with the right size and checksum.
        missing: Files that are not there.
        corrupted: Files present whose size or checksum differ from the registry.
    """

    ok: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    corrupted: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """True when nothing is missing or corrupted."""
        return not self.missing and not self.corrupted


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of ``path``, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid(path: Path, entry: DatasetFile) -> bool:
    return path.is_file() and path.stat().st_size == entry.size and sha256_of(path) == entry.sha256


def download_file(
    entry: DatasetFile,
    destination: Path,
    *,
    force: bool = False,
    timeout: float = 60.0,
    progress: Optional[ProgressFn] = None,
) -> bool:
    """Download one registry file to ``destination`` and verify it.

    Args:
        entry: Registry description of the file (URL, size, SHA-256).
        destination: Final path; parent directories are created.
        force: Download again even if a valid file is already there.
        timeout: Network timeout in seconds.
        progress: Optional callback ``(relative path, bytes so far, total bytes)``.

    Returns:
        True if the file was downloaded, False if a valid copy was already present.

    Raises:
        DataDownloadError: On a network error or a checksum/size mismatch. The partial file is
            removed and ``destination`` is left untouched.
    """
    if not force and _is_valid(destination, entry):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(entry.url, headers={"User-Agent": _USER_AGENT})
    digest = hashlib.sha256()
    received = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as out:
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if progress is not None:
                    progress(entry.path, received, entry.size)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        partial.unlink(missing_ok=True)
        reason = getattr(error, "reason", error)
        tls = " The server's TLS certificate could not be verified." if isinstance(reason, ssl.SSLError) else ""
        raise DataDownloadError(
            f"Could not download {entry.url}: {error}.{tls} If the data cannot be reached from this machine, "
            f"download the file by hand and place it at {destination} (its SHA-256 must be {entry.sha256}), or "
            "pass --from DIRECTORY_OR_URL with a copy or a mirror: every file is checksum-verified either way."
        ) from error
    if digest.hexdigest() != entry.sha256 or received != entry.size:
        partial.unlink(missing_ok=True)
        raise DataDownloadError(
            f"Checksum mismatch for {entry.url}: got {received} bytes / sha256 {digest.hexdigest()}, "
            f"expected {entry.size} bytes / {entry.sha256}. The file was not stored."
        )
    os.replace(partial, destination)
    return True


def _select(dataset: Dataset, only: Optional[Sequence[str]]) -> List[DatasetFile]:
    if not only:
        return list(dataset.files)
    selected = [f for f in dataset.files if any(fnmatch.fnmatch(f.path, pattern) for pattern in only)]
    if not selected:
        raise KeyError(f"No file of dataset {dataset.name!r} matches {list(only)}.")
    return selected


def _from_source(entry: DatasetFile, source: Union[str, Path]) -> DatasetFile:
    """Point ``entry`` at ``source`` (a directory, or a base URL) instead of its registry URL."""
    text = str(source)
    if text.startswith(("http://", "https://", "file://")):
        url = text.rstrip("/") + "/" + entry.path
    else:
        url = (Path(source) / entry.path).resolve().as_uri()
    return dataclasses.replace(entry, url=url)


def fetch_dataset(
    name: str,
    *,
    only: Optional[Sequence[str]] = None,
    force: bool = False,
    root: Optional[Path] = None,
    source: Union[str, Path, None] = None,
    progress: Optional[ProgressFn] = None,
) -> FetchReport:
    """Download the files of dataset ``name`` that are missing or invalid.

    Args:
        name: Registry name.
        only: Optional glob patterns restricting which files are fetched.
        force: Re-download files even if they are already valid.
        root: Directory to store the dataset in (default: ``tpeanuts.data.dataset_dir(name)``).
        source: Directory or base URL holding the dataset's files (same relative paths as the registry)
            to use instead of the registry URLs: a local copy, or a mirror. The checksums still apply.
        progress: Optional progress callback, see ``download_file``.

    Returns:
        A ``FetchReport``.

    Raises:
        KeyError: Unknown dataset, or ``only`` matches nothing.
        DataDownloadError: A download failed.
    """
    dataset = get_dataset(name)
    target = Path(root) if root is not None else dataset_dir(name)
    report = FetchReport()
    for entry in _select(dataset, only):
        origin = _from_source(entry, source) if source is not None else entry
        if download_file(origin, target / entry.path, force=force, progress=progress):
            report.downloaded.append(entry.path)
        else:
            report.skipped.append(entry.path)
    return report


def dataset_present(name: str, *, root: Optional[Path] = None) -> bool:
    """Tell whether every file of dataset ``name`` is on disk with its registered size.

    Fast (no hashing): meant for deciding whether code or tests that need the dataset can run;
    use ``verify_dataset`` to check the contents.

    Args:
        name: Registry name.
        root: Directory holding the dataset (default: ``tpeanuts.data.dataset_dir(name)``).

    Returns:
        True if all files exist with the registered size.
    """
    dataset = get_dataset(name)
    target = Path(root) if root is not None else dataset_dir(name)
    return all(
        (target / f.path).is_file() and (target / f.path).stat().st_size == f.size for f in dataset.files
    )


def verify_dataset(
    name: str, *, only: Optional[Sequence[str]] = None, root: Optional[Path] = None,
) -> VerifyReport:
    """Check the files of dataset ``name`` on disk against the registry.

    Args:
        name: Registry name.
        only: Optional glob patterns restricting which files are checked.
        root: Directory holding the dataset (default: ``tpeanuts.data.dataset_dir(name)``).

    Returns:
        A ``VerifyReport``.
    """
    dataset = get_dataset(name)
    target = Path(root) if root is not None else dataset_dir(name)
    report = VerifyReport()
    for entry in _select(dataset, only):
        path = target / entry.path
        if not path.is_file():
            report.missing.append(entry.path)
        elif _is_valid(path, entry):
            report.ok.append(entry.path)
        else:
            report.corrupted.append(entry.path)
    return report


def remove_dataset(name: str, *, root: Optional[Path] = None) -> bool:
    """Delete the directory of dataset ``name``.

    Args:
        name: Registry name.
        root: Directory holding the dataset (default: ``tpeanuts.data.dataset_dir(name)``).

    Returns:
        True if a directory was removed, False if there was none.
    """
    get_dataset(name)  # validates the name
    target = Path(root) if root is not None else dataset_dir(name)
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True
