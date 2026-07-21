#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#  This module is part of the Master's Thesis (MSc Dissertation):
#  - Fast Simulation of Neutrino Oscillations in Matter
#  
#  Author:
#      Juan Ramon Diaz Santos <diazjuan@alumni.uv.es>
#
#  Supervisors:
#      Roberto Ruiz de Austri Bazan <rruiz@ific.uv.es>
#      Michele Lucente <michele.lucente@unibo.it>
#
#  Date:
#      June 2026
# =============================================================================

"""
Readers for Honda/HKKM Atmosphere-neutrino tables.

This module is the parsing layer for the external Honda/HKKM data files: it
contains no propagation physics and does not call into any external Python
package. It only locates, classifies, and parses the plain-text (gzip
compressed) ``.d.gz`` tables that Honda et al. distribute for a given
experimental site, and exposes them as NumPy arrays keyed by energy and
cos(zenith).

Honda flux tables are tabulated as differential fluxes in
(m^2 s sr GeV)^-1. TPeanuts Atmosphere source files use the same spectral
quantity per cm^2, so generator code divides Honda fluxes by 1e4 before
saving.

There are two kinds of Honda tables read here:

    flux tables
        Differential flux Phi(E, cosZ) per neutrino flavour (NuMu, NuMubar,
        NuE, NuEbar), binned in neutrino energy E [GeV] and cos(zenith) at
        the detector. Optionally also binned in azimuth, depending on the
        selected angular mode.

    production-height tables
        For each (energy, cos(zenith)) bin, the quantiles of the altitude
        [km] at which the neutrino's parent meson/muon decayed in the
        atmosphere. These are used to reconstruct a height-differential
        flux Phi(E,h) from the height-integrated flux Phi(E).

Module functions:
    HondaTableSelection
        Dataclass selecting which Honda table variant (site, season, solar
        activity, mountain profile, angular binning) to read.
    find_honda_data_dir(...)
        Locate the local directory containing Honda .d.gz table files.
    classify_table_name(...)
        Parse a Honda filename into site/season/solar/angular metadata.
    choose_flux_file(...)
        Select the flux table file matching a HondaTableSelection.
    choose_height_file(...)
        Select the production-height table file for one flavour.
    load_honda_tables(...)
        Locate and parse one flux table plus the per-particle height tables
        needed by the generator.
    honda_cosz_centers(...)
        Return the fixed cos(zenith) bin centers used by the standard
        20-bin Honda zenith binning.
    zenith_bin_to_cosz_center(...)
        Convert a 1-indexed Honda zenith bin number to its cos(zenith)
        center.
    read_flux_table(...)
        Parse a Honda flux .d.gz file into energy/cosz-indexed flux arrays.
    read_height_table(...)
        Parse a Honda production-height .d.gz file into quantile arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import os
import re
from typing import Any, Optional

import numpy as np

import tpeanuts.config.default as default


FLUX_BLOCK_RE = re.compile(
    r"average flux in \[cosZ\s*=\s*([+-]?\d+(?:\.\d+)?)\s*--\s*([+-]?\d+(?:\.\d+)?),\s*"
    r"phi_Az\s*=\s*([+-]?\d+(?:\.\d+)?)\s*--\s*([+-]?\d+(?:\.\d+)?)\]"
)
HEIGHT_HEADER_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+)$")

HONDA_FLUX_COLUMNS = ("NuMu", "NuMubar", "NuE", "NuEbar")
HONDA_TO_TPEANUTS = {
    "NuE": "nue",
    "NuEbar": "antinue",
    "NuMu": "numu",
    "NuMubar": "antinumu",
}
TPEANUTS_TO_HONDA = {value: key for key, value in HONDA_TO_TPEANUTS.items()}
TPEANUTS_TO_HONDA_HEIGHT = {
    "nue": "nue",
    "antinue": "nuebar",
    "numu": "numu",
    "antinumu": "numubar",
}


@dataclass(frozen=True)
class HondaTableSelection:
    """
    Selects which Honda/HKKM table variant to read for a given site.

    Honda publishes several variants of the flux and production-height
    tables per experimental site, differing in the assumed solar activity,
    season averaging, presence of a nearby mountain (which shields part of
    the downward-going sky), and angular binning. This dataclass identifies
    one such variant; it carries no physics itself and is only used to
    select a filename via classify_table_name/choose_flux_file/
    choose_height_file.

    Attributes:
        site_code: Three-letter Honda site code (e.g. "frj" for Frejus),
            matching the first filename token.
        season_code: Season-averaging code (e.g. "ally" for all-year
            averaged), matching the second filename token.
        solar: Solar activity assumption, "solmin" or "solmax", selecting
            the solar-minimum or solar-maximum flux tables.
        mountain: If True, select the table variant computed with a nearby
            mountain profile shadowing part of the sky.
        angular_mode: Angular binning of the flux table: "zenith+azimuth",
            "azimuth-averaged", or "all-direction-averaged".
        azimuth_averaged_height: If True, prefer the azimuth-averaged
            production-height table file (filename token "aa") when
            selecting a height file.
    """

    site_code: str = "frj"
    season_code: str = "ally"
    solar: str = "solmin"
    mountain: bool = False
    angular_mode: str = "azimuth-averaged"
    azimuth_averaged_height: bool = True


def find_honda_data_dir(path: str | os.PathLike[str] | None = None) -> Path:
    """
    Locate the local directory containing Honda/HKKM ``.d.gz`` table files.

    This only resolves a filesystem path; it does not parse or validate the
    physics content of the tables. Candidates are tried in order: an
    explicit path argument, the ``HONDA_DATA_DIR`` environment variable,
    then ``tpeanuts.config.default.honda_dataset``. The first candidate
    directory that exists and contains at least one ``*.d.gz`` file is
    returned.

    Args:
        path: Optional explicit directory path to try first.

    Returns:
        Path to the first matching directory containing Honda tables.

    Raises:
        FileNotFoundError: If no candidate directory contains ``.d.gz``
            files.
    """
    candidates = []

    if path is not None:
        candidates.append(Path(path))

    env_path = os.environ.get("HONDA_DATA_DIR", None)
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(Path(default.honda_dataset))

    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.d.gz")):
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find Honda .d.gz tables. Set HONDA_DATA_DIR or pass "
        f"honda_data_dir explicitly. Checked: {checked}"
    )


def classify_table_name(name: str) -> dict[str, Any]:
    """
    Parse a Honda table filename into site/season/solar/angular metadata.

    Honda filenames encode the table contents in dash-separated tokens,
    e.g. ``frj-ally-20-12-solmin.d.gz`` for a flux table with 20 zenith bins
    and 12 azimuth bins, or ``frj-ally-aa-numu.d.gz`` for an
    azimuth-averaged muon-neutrino production-height table. This function
    has no physics content; it only decodes those filename conventions so
    that choose_flux_file/choose_height_file can select the correct file
    for a given HondaTableSelection.

    Args:
        name: Honda table filename (with or without directory components),
            ending in ".d.gz".

    Returns:
        Dictionary of metadata describing the table. Always includes "name",
        "site_code", "season_code", and "table_type" ("flux",
        "production_height", or "unknown"). For flux tables, also includes
        "solar", "mountain", "zenith_bins", "azimuth_bins", and
        "angular_mode" ("zenith+azimuth", "azimuth-averaged",
        "all-direction-averaged", or "other"). For production-height
        tables, also includes "flavour" and "azimuth_averaged_height".
    """
    stem = name.removesuffix(".d.gz")
    parts = stem.split("-")
    meta: dict[str, Any] = {
        "name": name,
        "site_code": parts[0] if parts else None,
        "season_code": parts[1] if len(parts) > 1 else None,
        "table_type": "unknown",
    }

    if any(part in {"solmin", "solmax"} for part in parts):
        meta["table_type"] = "flux"
        meta["solar"] = next(part for part in parts if part in {"solmin", "solmax"})
        meta["mountain"] = "mtn" in parts
        bin_tokens = [part for part in parts if re.fullmatch(r"\d{2}", part)]
        meta["zenith_bins"] = int(bin_tokens[0]) if len(bin_tokens) >= 1 else None
        meta["azimuth_bins"] = int(bin_tokens[1]) if len(bin_tokens) >= 2 else None

        if meta["zenith_bins"] == 20 and meta["azimuth_bins"] == 12:
            meta["angular_mode"] = "zenith+azimuth"
        elif meta["zenith_bins"] == 20 and meta["azimuth_bins"] == 1:
            meta["angular_mode"] = "azimuth-averaged"
        elif meta["zenith_bins"] == 1 and meta["azimuth_bins"] == 1:
            meta["angular_mode"] = "all-direction-averaged"
        else:
            meta["angular_mode"] = "other"
    elif any(part in TPEANUTS_TO_HONDA_HEIGHT.values() for part in parts):
        meta["table_type"] = "production_height"
        meta["flavour"] = next(part for part in parts if part in TPEANUTS_TO_HONDA_HEIGHT.values())
        meta["azimuth_averaged_height"] = "aa" in parts

    return meta


def choose_flux_file(
    honda_data_dir: str | os.PathLike[str],
    selection: HondaTableSelection,
) -> Path:
    """
    Select the Honda flux table file matching a HondaTableSelection.

    Args:
        honda_data_dir: Directory (or candidate hint) containing Honda
            ``.d.gz`` files, resolved through find_honda_data_dir.
        selection: Site/season/solar/mountain/angular-mode selection to
            match against each candidate file's parsed metadata.

    Returns:
        Path to the matching flux table file. If several files match, the
        lexicographically first path is returned for determinism.

    Raises:
        FileNotFoundError: If no flux table in the data directory matches
            the selection.
    """
    data_dir = find_honda_data_dir(honda_data_dir)
    matches: list[Path] = []

    for path in data_dir.glob("*.d.gz"):
        meta = classify_table_name(path.name)
        if (
            meta.get("table_type") == "flux"
            and meta.get("site_code") == selection.site_code
            and meta.get("season_code") == selection.season_code
            and meta.get("solar") == selection.solar
            and meta.get("mountain") == selection.mountain
            and meta.get("angular_mode") == selection.angular_mode
        ):
            matches.append(path)

    if not matches:
        raise FileNotFoundError(
            "No Honda flux table matched "
            f"{selection} in {data_dir}."
        )

    return sorted(matches)[0]


def choose_height_file(
    honda_data_dir: str | os.PathLike[str],
    selection: HondaTableSelection,
    particle: str,
) -> Optional[Path]:
    """
    Select the Honda production-height table file for one particle flavour.

    The production-height table gives the distribution of the altitude at
    which the parent meson/muon of a detected neutrino decayed, as a
    function of neutrino energy and cos(zenith). It is used downstream to
    turn the height-integrated Honda flux into a height-differential flux
    Phi(E,h).

    Args:
        honda_data_dir: Directory (or candidate hint) containing Honda
            ``.d.gz`` files, resolved through find_honda_data_dir.
        selection: Site/season selection; only site_code, season_code, and
            azimuth_averaged_height are used here.
        particle: tpeanuts particle/flavour name (e.g. "nue", "antinumu").
            Mapped to the Honda flavour token via TPEANUTS_TO_HONDA_HEIGHT.

    Returns:
        Path to the matching production-height file, or None if the
        particle has no Honda height-table flavour (e.g. tau neutrinos) or
        no matching file is found on disk.
    """
    honda_flavour = TPEANUTS_TO_HONDA_HEIGHT.get(str(particle).lower())
    if honda_flavour is None:
        return None

    data_dir = find_honda_data_dir(honda_data_dir)
    aa_token = "-aa-" if selection.azimuth_averaged_height else "-"
    name = f"{selection.site_code}-{selection.season_code}{aa_token}{honda_flavour}.d.gz"
    path = data_dir / name

    if path.exists():
        return path

    fallback = data_dir / f"{selection.site_code}-{selection.season_code}-{honda_flavour}.d.gz"
    if fallback.exists():
        return fallback

    return None


def _select_height_source(
    height_tables: dict[str, Optional[dict[str, Any]]],
    particle: str,
) -> Optional[dict[str, Any]]:
    """
    Pick a production-height table for a particle, with a tau-neutrino fallback.

    Honda does not publish production-height tables for tau neutrinos
    (they are not produced directly in cosmic-ray air showers). As an
    approximation, nutau/antinutau fall back to the numu/antinumu height
    table, since both come from the same hadronic shower region.

    Args:
        height_tables: Mapping from particle name to parsed Honda
            production-height table (or None if unavailable).
        particle: tpeanuts particle/flavour name.

    Returns:
        The matching production-height table dictionary, or None if no
        table (including no fallback) is available for this particle.
    """
    if height_tables.get(particle) is not None:
        return height_tables[particle]

    key = str(particle).lower()
    if "tau" in key:
        return height_tables.get("numu") if "anti" not in key else height_tables.get("antinumu")

    return None


def load_honda_tables(
    *,
    honda_data_dir: str | os.PathLike[str] | None,
    selection: HondaTableSelection,
    particles: list[str],
) -> dict[str, Any]:
    """
    Locate and parse the Honda flux table and per-particle height tables.

    This loads the data once so that it can be reused across many
    particle/angle combinations, instead of re-reading the gzip files for
    every call.

    Args:
        honda_data_dir: Optional directory hint forwarded to
            find_honda_data_dir.
        selection: Honda table variant (site, season, solar, mountain,
            angular mode) to load.
        particles: List of tpeanuts particle/flavour names for which a
            production-height table should be loaded (entries without a
            matching Honda height table are stored as None).

    Returns:
        Dictionary with keys "data_dir" (resolved Honda data directory as
        str), "flux_table" (parsed flux table dict, see read_flux_table),
        and "height_tables" (dict mapping each requested particle to its
        parsed production-height table, or None).
    """
    data_dir = find_honda_data_dir(honda_data_dir)
    flux_path = choose_flux_file(data_dir, selection)
    flux_table = read_flux_table(flux_path)

    height_tables: dict[str, Optional[dict[str, Any]]] = {}
    for particle in particles:
        path = choose_height_file(data_dir, selection, particle)
        height_tables[particle] = read_height_table(path) if path is not None else None

    return {
        "data_dir": str(data_dir),
        "flux_table": flux_table,
        "height_tables": height_tables,
    }


def honda_cosz_centers() -> np.ndarray:
    """
    Return the cos(zenith) bin centers of the standard 20-bin Honda binning.

    Honda's standard zenith-angle binning divides cos(zenith) in [-1, 1]
    into 20 equal-width bins of width 0.10, ordered from cosZ=0.95
    (near-vertically-downward) to cosZ=-0.95 (near-vertically-upward).
    cos(zenith)=1 corresponds to a neutrino arriving straight down at the
    detector, cos(zenith)=-1 to one arriving straight up through the Earth.

    Returns:
        1D float array of length 20 with the 20 bin-center cos(zenith)
        values, in decreasing order.
    """
    return np.array([0.95 - 0.10 * idx for idx in range(20)], dtype=float)


def zenith_bin_to_cosz_center(zenith_bin: int) -> float:
    """
    Convert a 1-indexed Honda zenith bin number to its cos(zenith) center.

    Args:
        zenith_bin: 1-indexed zenith bin number as it appears in the
            production-height table header (1 = most downward-going bin).

    Returns:
        cos(zenith) value at the center of that bin, consistent with the
        ordering returned by honda_cosz_centers().
    """
    return float(0.95 - 0.10 * (int(zenith_bin) - 1))


def read_flux_table(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """
    Parse a Honda flux ``.d.gz`` file into energy/cos(zenith)-indexed arrays.

    The file is a gzip-compressed plain-text table organized in blocks, each
    headed by a line of the form
    "average flux in [cosZ = c1 -- c2, phi_Az = p1 -- p2]" followed by rows
    of "E_nu(GeV)  NuMu  NuMubar  NuE  NuEbar" differential flux values in
    (m^2 s sr GeV)^-1 (azimuth-averaged tables use one phi_Az block per
    cosZ bin). This function does not convert units or interpolate; it only
    builds dense arrays indexed by the file's native energy and cos(zenith)
    grids, leaving the physical flux values unchanged.

    Args:
        path: Path to a Honda flux ``.d.gz`` file.

    Returns:
        Dictionary with keys "path" (str), "energy_GeV" (1D sorted array of
        unique neutrino energies in GeV), "cosz_center" (1D sorted array of
        unique cos(zenith) bin centers), and "flux_m2" (dict mapping each
        Honda flavour name in HONDA_FLUX_COLUMNS to a 2D array of shape
        (n_cosz, n_energy) with differential flux in (m^2 s sr GeV)^-1).

    Raises:
        ValueError: If no flux rows could be parsed from the file.
    """
    rows: list[dict[str, float]] = []
    current: Optional[dict[str, float]] = None

    with gzip.open(Path(path), "rt", encoding="ascii", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            match = FLUX_BLOCK_RE.search(line)
            if match:
                c1, c2, p1, p2 = map(float, match.groups())
                current = {
                    "cosz_min": min(c1, c2),
                    "cosz_max": max(c1, c2),
                    "cosz_center": 0.5 * (c1 + c2),
                    "phi_min_deg": min(p1, p2),
                    "phi_max_deg": max(p1, p2),
                    "phi_center_deg": 0.5 * (p1 + p2),
                }
                continue

            if current is None or line.startswith("Enu"):
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue

            rows.append(
                {
                    **current,
                    "energy_GeV": values[0],
                    **dict(zip(HONDA_FLUX_COLUMNS, values[1:])),
                }
            )

    if not rows:
        raise ValueError(f"Could not read Honda flux rows from {path}.")

    energies = np.array(sorted({row["energy_GeV"] for row in rows}), dtype=float)
    cosz = np.array(sorted({row["cosz_center"] for row in rows}), dtype=float)
    flux = {
        flavour: np.full((cosz.size, energies.size), np.nan, dtype=float)
        for flavour in HONDA_FLUX_COLUMNS
    }

    e_index = {value: idx for idx, value in enumerate(energies)}
    z_index = {value: idx for idx, value in enumerate(cosz)}

    for row in rows:
        iz = z_index[row["cosz_center"]]
        ie = e_index[row["energy_GeV"]]
        for flavour in HONDA_FLUX_COLUMNS:
            flux[flavour][iz, ie] = row[flavour]

    return {
        "path": str(path),
        "energy_GeV": energies,
        "cosz_center": cosz,
        "flux_m2": flux,
    }


def read_height_table(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """
    Parse a Honda production-height ``.d.gz`` file into quantile arrays.

    The file is a gzip-compressed plain-text table organized in blocks per
    (zenith bin, azimuth bin), each headed by a line with the probability
    levels of the production-height distribution (e.g. 10%, 20%, ...,
    90% quantiles), followed by rows of
    "E_nu(GeV)  h_q1(m)  h_q2(m)  ...  h_qN(m)" giving, for each neutrino
    energy, the altitude (in metres above sea level in the source file,
    converted to km here) at which that fraction of parent mesons/muons
    decayed. This function does not interpolate onto a new energy or
    cos(zenith) grid; it only assembles the native quantile arrays.

    Args:
        path: Path to a Honda production-height ``.d.gz`` file.

    Returns:
        Dictionary with keys "path" (str), "energy_GeV" (1D array, the
        common neutrino-energy grid shared by all zenith blocks),
        "cosz_center" (1D array of cos(zenith) bin centers, one per zenith
        block, via zenith_bin_to_cosz_center), "probabilities" (1D array of
        quantile probability levels in [0, 1], shared by all blocks), and
        "height_quantiles_km" (3D array of shape
        (n_cosz, n_energy, n_probabilities) with production-height
        quantiles in km).

    Raises:
        ValueError: If no height rows could be parsed, or if different
            zenith blocks report inconsistent probability levels or energy
            grids.
    """
    blocks: dict[tuple[int, int], dict[str, Any]] = {}
    current_key: Optional[tuple[int, int]] = None

    with gzip.open(Path(path), "rt", encoding="ascii", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            match = HEIGHT_HEADER_RE.match(line)
            if match:
                _site, _flavour, zenith_bin, azimuth_bin = map(int, match.groups()[:4])
                probabilities = np.array(
                    [float(value) for value in match.group(5).split()],
                    dtype=float,
                )
                current_key = (zenith_bin, azimuth_bin)
                blocks[current_key] = {
                    "probabilities": probabilities,
                    "energy_GeV": [],
                    "height_m": [],
                    "zenith_bin": zenith_bin,
                    "azimuth_bin": azimuth_bin,
                }
                continue

            if current_key is None:
                continue

            parts = line.split()
            probabilities = blocks[current_key]["probabilities"]
            if len(parts) != probabilities.size + 1:
                continue

            values = np.array([float(part) for part in parts], dtype=float)
            blocks[current_key]["energy_GeV"].append(values[0])
            blocks[current_key]["height_m"].append(values[1:])

    if not blocks:
        raise ValueError(f"Could not read Honda production-height rows from {path}.")

    # Build cosz and heights in the order blocks were read (sorted by zenith_bin,
    # which maps to *decreasing* cosz: zenith_bin=1 → cosz=0.95, bin=2 → 0.85, …).
    cosz_unsorted = np.array(
        [zenith_bin_to_cosz_center(key[0]) for key in sorted(blocks)],
        dtype=float,
    )
    probabilities_ref = None
    energy_ref = None
    heights = []

    for key in sorted(blocks):
        block = blocks[key]
        probabilities = block["probabilities"]
        energy = np.asarray(block["energy_GeV"], dtype=float)
        height_km = np.asarray(block["height_m"], dtype=float) * 1.0e-3

        if probabilities_ref is None:
            probabilities_ref = probabilities
        elif not np.allclose(probabilities_ref, probabilities):
            raise ValueError(f"Inconsistent height probabilities in {path}.")

        if energy_ref is None:
            energy_ref = energy
        elif not np.allclose(energy_ref, energy):
            raise ValueError(f"Inconsistent height energy grid in {path}.")

        heights.append(height_km)

    # np.interp requires xp to be strictly increasing.  Re-sort by ascending cosz
    # so that downstream interpolation (generator._interpolate_quantiles) works
    # correctly for any requested zenith angle.
    sort_idx = np.argsort(cosz_unsorted)
    cosz_sorted = cosz_unsorted[sort_idx]
    quantiles_sorted = np.stack(heights, axis=0)[sort_idx]   # (n_cosz, n_E, n_prob)

    return {
        "path": str(path),
        "energy_GeV": energy_ref,
        "cosz_center": cosz_sorted,
        "probabilities": probabilities_ref,
        "height_quantiles_km": quantiles_sorted,
    }
