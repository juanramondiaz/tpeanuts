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

"""The ``tpeanuts-data`` command: list, fetch, verify, build and remove tpeanuts' downloadable data.

::

    tpeanuts-data list
    tpeanuts-data info icecube
    tpeanuts-data path icecube
    tpeanuts-data fetch icecube
    tpeanuts-data fetch honda --only "frj-0608-20-12-solmin.d.gz"
    tpeanuts-data fetch honda --only "frj-0608-20-12-solmin.d.gz" --from /path/to/honda/tables
    tpeanuts-data verify icecube
    tpeanuts-data build honda-flux
    tpeanuts-data build honda-flux --from /path/to/honda/tables
    tpeanuts-data clean icecube --yes

Exit codes: 0 success, 1 verification found missing/corrupted files, 2 a usage error (unknown dataset,
refusing a large download), 3 a download or checksum failure.

Module contents:
    main(...)
        Entry point (also registered as the ``tpeanuts-data`` console script).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from tpeanuts.data import dataset_dir, generated_dir
from tpeanuts.data.fetch import DataDownloadError, fetch_dataset, remove_dataset, verify_dataset
from tpeanuts.data.registry import get_dataset, load_registry

_LARGE_DOWNLOAD_BYTES = 100 * 1000 * 1000


def _human(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:.1f} MB" if n_bytes >= 1e5 else f"{n_bytes / 1e3:.1f} kB"


_PROGRESS_STEP = 25  # percent
_reported: dict = {}


def _progress(path: str, received: int, total: int) -> None:
    """Print one line per file at 25 % steps (readable in terminals and in logs)."""
    bucket = 100 if received >= total else int(100 * received / max(total, 1)) // _PROGRESS_STEP * _PROGRESS_STEP
    if bucket and _reported.get(path) != bucket:
        _reported[path] = bucket
        print(f"  {path}: {bucket:3d}% of {_human(total)}", file=sys.stderr, flush=True)


def _cmd_list(_args: argparse.Namespace) -> int:
    print(f"{'dataset':<10}{'files':>7}{'size':>12}  {'on disk':<14}license")
    for name, dataset in sorted(load_registry().items()):
        root = dataset_dir(name)
        present = sum(
            1 for f in dataset.files if (root / f.path).is_file() and (root / f.path).stat().st_size == f.size
        )
        n = len(dataset.files)
        state = "complete" if present == n else (f"{present}/{n}" if present else "absent")
        print(f"{name:<10}{len(dataset.files):>7}{_human(dataset.size):>12}  {state:<14}{dataset.license}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    dataset = get_dataset(args.name)
    print(f"{dataset.name}: {dataset.description}")
    print(f"  license : {dataset.license}")
    print(f"  cite    : {dataset.citation}")
    print(f"  source  : {dataset.homepage}")
    print(f"  files   : {len(dataset.files)} ({_human(dataset.size)})")
    print(f"  stored  : {dataset_dir(dataset.name)}")
    return 0


def _cmd_path(args: argparse.Namespace) -> int:
    get_dataset(args.name)
    print(dataset_dir(args.name))
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    dataset = get_dataset(args.name)
    if not args.only and not args.yes and dataset.size > _LARGE_DOWNLOAD_BYTES:
        print(
            f"Dataset {args.name!r} is {_human(dataset.size)} ({len(dataset.files)} files). Restrict it with "
            "--only PATTERN (e.g. --only 'frj-0608-20-12-solmin.d.gz') or confirm the full download with --yes.",
            file=sys.stderr,
        )
        return 2
    report = fetch_dataset(args.name, only=args.only, force=args.force, source=args.source, progress=_progress)
    print(f"{args.name}: {len(report.downloaded)} downloaded, {len(report.skipped)} already present "
          f"-> {dataset_dir(args.name)}")
    print(f"Cite: {dataset.citation}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    report = verify_dataset(args.name, only=args.only)
    print(f"{args.name}: {len(report.ok)} ok, {len(report.missing)} missing, {len(report.corrupted)} corrupted")
    for label, names in (("missing", report.missing), ("corrupted", report.corrupted)):
        for name in names[:10]:
            print(f"  {label}: {name}")
        if len(names) > 10:
            print(f"  ... and {len(names) - 10} more {label}")
    return 0 if report.complete else 1


def _cmd_clean(args: argparse.Namespace) -> int:
    get_dataset(args.name)
    if not args.yes:
        print(f"This deletes {dataset_dir(args.name)}. Re-run with --yes to confirm.", file=sys.stderr)
        return 2
    print(f"{args.name}: {'removed' if remove_dataset(args.name) else 'nothing to remove'}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    # Imported here: it pulls numpy/torch, which the other commands do not need.
    import tpeanuts.config.default as default
    from tpeanuts.source.atmosphere.honda import SURFACE_FLUX_RAW_FILENAME, build_surface_flux_table

    destination = default.ATMOSPHERE_DATA_DIR / "honda" / "flux" / "honda_flux.csv"
    if destination.is_file() and not args.force:
        print(f"honda-flux: already built at {destination} (use --force to rebuild).")
        return 0
    fetch_dataset("honda", only=[SURFACE_FLUX_RAW_FILENAME], source=args.source, progress=_progress)
    built = build_surface_flux_table(dataset_dir("honda") / SURFACE_FLUX_RAW_FILENAME, destination)
    print(f"honda-flux: built {built} (generated-data dir: {generated_dir()})")
    print(f"Cite: {get_dataset('honda').citation}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tpeanuts-data", description="Manage tpeanuts' downloadable datasets.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the datasets and what is on disk").set_defaults(func=_cmd_list)
    for name, func, help_text in (("info", _cmd_info, "license, citation and source of a dataset"),
                                  ("path", _cmd_path, "print the directory of a dataset")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("name")
        p.set_defaults(func=func)

    p = sub.add_parser("fetch", help="download a dataset (verifying every file)")
    p.add_argument("name")
    p.add_argument("--only", action="append", metavar="PATTERN", help="only files matching this glob (repeatable)")
    p.add_argument("--force", action="store_true", help="download again even if the file is valid")
    p.add_argument("--yes", action="store_true", help="confirm a large full download")
    p.add_argument("--from", dest="source", metavar="DIR_OR_URL",
                   help="take the files from this directory or base URL (a local copy or a mirror)")
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("verify", help="check the files on disk against the registry checksums")
    p.add_argument("name")
    p.add_argument("--only", action="append", metavar="PATTERN")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("clean", help="delete a downloaded dataset")
    p.add_argument("name")
    p.add_argument("--yes", action="store_true", help="confirm the deletion")
    p.set_defaults(func=_cmd_clean)

    p = sub.add_parser("build", help="build a locally generated product")
    p.add_argument("product", choices=["honda-flux"], help="honda-flux: the canonical Honda surface-flux table")
    p.add_argument("--force", action="store_true", help="rebuild even if it exists")
    p.add_argument("--from", dest="source", metavar="DIR_OR_URL",
                   help="take the raw Honda table from this directory or base URL")
    p.set_defaults(func=_cmd_build)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``tpeanuts-data`` command.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        The exit code, see the module docstring.
    """
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return args.func(args)
    except KeyError as error:
        print(error.args[0] if error.args else error, file=sys.stderr)
        return 2
    except DataDownloadError as error:
        print(f"\n{error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
