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

"""Generate height-differential atmospheric-neutrino flux tables ``Phi(E, h, angle)`` from Honda/HKKM tables.

This module reads the Honda/HKKM
external ``.d.gz`` tables -- a height-*integrated* flux table
Phi(E, cosZ) plus, per particle, a production-height quantile table --
and combines them into the same height-*differential* flux
Phi(E, h) = Phi(E; X_obs) * f(h|E, alpha) that ``mceq.py`` produces,
saved with ``tpeanuts.source.atmosphere.io.write_angle_flux_csv`` in
the identical per-angle CSV schema (``energy_GeV, cos_zenith, particle,
flux, altitude_km``) -- one file per requested angle, directly readable
by ``load_atmospheric_flux``, with no format distinction between an
MCEq-sourced and a Honda-sourced table once written to disk.

Ported from v1_0's ``external/honda/{tables,generator,math}.py``, with
full option parity (not a trimmed-down single-variant version): every
Honda table-selection dimension v1_0 exposed via ``HondaTableSelection``
(site, season, solar activity, mountain profile, angular binning,
azimuth-averaged-height preference) is configurable here too,
consolidated into one ``HondaConfig`` dataclass (matching
``tpeanuts.source.atmosphere.mceq.MCEqConfig``'s own "one config, full
parity" shape), with defaults sourced from ``tpeanuts.config.default``.

**Two zenith-angle conventions, exactly as in ``mceq.py``**: ``alpha``
(the surface/table zenith angle Honda's own ``cosZ`` binning is keyed
by -- Honda itself calls this coordinate "theta", but within this
project's own convention it is the *surface* angle, see
``tpeanuts.medium.atmosphere.geometry``'s module docstring) and
``theta`` (the detector zenith angle, at any detector depth).
``generate_honda_flux``'s ``angle_type`` argument accepts either,
converting via ``alpha_surface_to_theta_detector``/
``theta_detector_to_alpha_surface`` exactly as ``generate_mceq_flux``
does; the file this module writes always stores
``cos_zenith = cos(theta_deg)`` (the detector-frame convention), even
though the Honda table lookup itself happens at ``cos(alpha_deg)``.

**Why the raw ``.d.gz`` tables are not read with torch**: Honda's own
file format (irregular text blocks, gzip-compressed, keyed by
non-uniform (energy, zenith, azimuth) grids that differ file to file)
is exactly the kind of ragged, index-heavy parsing NumPy/pure Python
handles directly and torch has no real advantage for -- mirroring why
v1_0 itself never ports this specific layer to torch either. Every
tensor this module hands to ``write_angle_flux_csv`` is still a plain
non-differentiable leaf regardless (built from a NumPy array that was
never part of any autograd graph), the same non-differentiable
-external-source reasoning ``mceq.py``'s own module docstring gives for
its blanket ``@torch.no_grad()`` -- so parsing in NumPy costs nothing
this project's own conventions actually protect against.

**Units**: Honda tables are natively in ``(m^2 s sr GeV)^-1``; every
flux value is converted to ``(cm^2 s sr GeV)^-1`` before being written
(matching v1_0's own generator, and matching ``mceq.py``'s own MCEq
-native unit -- so a downstream reader never has to special-case which
generator produced a given file). This is a genuine unit difference
from the canonical surface table ``honda/flux/honda_flux.csv`` (kept in
Honda's own original ``m^-2`` unit; built from one raw file by
``build_surface_flux_table`` below, which is not part of
``generate_honda_flux``). That table is not shipped with the package (the
Honda terms are unverified, see ``DATA_LICENSES.md``): each user builds it
locally with ``tpeanuts-data build honda-flux``.

Module contents:
    HondaConfig
        Dataclass aggregating every Honda table-selection and
        production-height-grid parameter, with ``config.default``
        -sourced defaults and a ``validate()`` method.
    generate_honda_flux(...)
        Public entry point: build and save one differential-flux CSV
        per requested angle from the Honda/HKKM tables.
    SURFACE_FLUX_RAW_FILENAME
        The one raw Honda table the canonical surface table is built from.
    build_surface_flux_table(...)
        Convert that raw table into the canonical ``honda_flux.csv`` the
        default ``load_atmospheric_flux`` reads (same content as the
        table this project was validated with, apart from LF line endings).
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from tpeanuts.config import default
from tpeanuts.source.atmosphere.io import angle_output_path, write_angle_flux_csv
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.geometry import alpha_surface_to_theta_detector, theta_detector_to_alpha_surface
from tpeanuts.util.tensor import TensorLike, as_tensor

# ============================================================
# Honda filename/particle-name conventions (mirrors v1_0's own tables.py)
# ============================================================

_FLUX_BLOCK_RE = re.compile(
    r"average flux in \[cosZ\s*=\s*([+-]?\d+(?:\.\d+)?)\s*--\s*([+-]?\d+(?:\.\d+)?),\s*"
    r"phi_Az\s*=\s*([+-]?\d+(?:\.\d+)?)\s*--\s*([+-]?\d+(?:\.\d+)?)\]"
)
_HEIGHT_HEADER_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+)$")

HONDA_FLUX_COLUMNS = ("NuMu", "NuMubar", "NuE", "NuEbar")
HONDA_TO_TPEANUTS = {"NuE": "nue", "NuEbar": "antinue", "NuMu": "numu", "NuMubar": "antinumu"}
TPEANUTS_TO_HONDA = {value: key for key, value in HONDA_TO_TPEANUTS.items()}
TPEANUTS_TO_HONDA_HEIGHT = {"nue": "nue", "antinue": "nuebar", "numu": "numu", "antinumu": "numubar"}

_SOLAR_ACTIVITIES = {"solmin", "solmax"}
_ANGULAR_MODES = {"zenith+azimuth", "azimuth-averaged", "all-direction-averaged"}

_M2_TO_CM2_FLUX = 1.0e-4


# ============================================================
# HondaConfig: every Honda table-selection + grid parameter, one dataclass
# ============================================================


@dataclass
class HondaConfig:
    """Every parameter needed to select a Honda/HKKM table variant and reconstruct its height profile.

    Consolidates what v1_0 split across ``HondaTableSelection`` (which
    site/season/solar/mountain/angular-binning table variant to read)
    and the generator's own loose ``h_grid_km`` keyword into one
    object, mirroring ``tpeanuts.source.atmosphere.mceq.MCEqConfig``'s
    own shape. Every field default is read from
    ``tpeanuts.config.default`` at class-definition time.

    Attributes:
        site_code: Three-letter Honda site code (e.g. ``"frj"`` for
            Frejus), the first filename token.
        season_code: Season-averaging code (e.g. ``"ally"`` for
            all-year averaged, ``"0608"`` for 2006-2008), the second
            filename token.
        solar: Solar activity assumption, ``"solmin"`` or ``"solmax"``.
        mountain: If True, select the table variant computed with a
            nearby mountain profile shadowing part of the sky.
        angular_mode: Angular binning of the flux table:
            ``"zenith+azimuth"``, ``"azimuth-averaged"``, or
            ``"all-direction-averaged"``.
        azimuth_averaged_height: If True, prefer the azimuth-averaged
            production-height table file (filename token ``"aa"``) when
            selecting a height file.
        h_grid_km: Strictly increasing 1-D altitude grid, in km, the
            reconstructed production-height profile is reported on.
        energy_grid_GeV: Optional strictly increasing 1-D energy grid,
            in GeV, the flux is reported on. ``None`` (default) uses
            the loaded Honda flux table's own native energy grid.

    Raises:
        ValueError: Via ``validate()``, if ``solar``/``angular_mode``
            is not one of the recognised values, if ``site_code``/
            ``season_code`` is empty, or if ``h_grid_km``/
            ``energy_grid_GeV`` is not a strictly increasing 1-D
            tensor.
    """

    site_code: str = default.ATMOSPHERE_HONDA_SITE_CODE
    season_code: str = default.ATMOSPHERE_HONDA_SEASON_CODE
    solar: str = default.ATMOSPHERE_HONDA_SOLAR
    mountain: bool = default.ATMOSPHERE_HONDA_MOUNTAIN
    angular_mode: str = default.ATMOSPHERE_HONDA_ANGULAR_MODE
    azimuth_averaged_height: bool = default.ATMOSPHERE_HONDA_AZIMUTH_AVERAGED_HEIGHT
    h_grid_km: torch.Tensor = field(
        default_factory=lambda: torch.linspace(
            default.ATMOSPHERE_HONDA_H_GRID_MIN_KM,
            default.ATMOSPHERE_HONDA_H_GRID_MAX_KM,
            default.ATMOSPHERE_HONDA_H_GRID_N,
            dtype=torch.float64,
        )
    )
    energy_grid_GeV: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.h_grid_km = as_tensor(self.h_grid_km, dtype=torch.float64).reshape(-1)
        if self.energy_grid_GeV is not None:
            self.energy_grid_GeV = as_tensor(self.energy_grid_GeV, dtype=torch.float64).reshape(-1)

    def validate(self) -> None:
        """Check every field against its own valid domain.

        Raises:
            ValueError: See this dataclass's own docstring.
        """
        if not self.site_code:
            raise ValueError("site_code must be a non-empty string.")
        if not self.season_code:
            raise ValueError("season_code must be a non-empty string.")
        if self.solar not in _SOLAR_ACTIVITIES:
            raise ValueError(f"Unknown solar={self.solar!r}. Allowed: {_SOLAR_ACTIVITIES}")
        if self.angular_mode not in _ANGULAR_MODES:
            raise ValueError(f"Unknown angular_mode={self.angular_mode!r}. Allowed: {_ANGULAR_MODES}")
        if self.h_grid_km.ndim != 1:
            raise ValueError("h_grid_km must be one-dimensional.")
        if torch.any(torch.diff(self.h_grid_km) <= 0.0):
            raise ValueError("h_grid_km must be strictly increasing.")
        if self.energy_grid_GeV is not None:
            if self.energy_grid_GeV.ndim != 1:
                raise ValueError("energy_grid_GeV must be one-dimensional.")
            if torch.any(torch.diff(self.energy_grid_GeV) <= 0.0):
                raise ValueError("energy_grid_GeV must be strictly increasing.")
            if torch.any(self.energy_grid_GeV <= 0.0):
                raise ValueError("energy_grid_GeV must be strictly positive.")


# ============================================================
# Locate and classify the raw .d.gz tables
# ============================================================


def find_honda_data_dir(honda_data_dir: Union[str, os.PathLike, None] = None) -> Path:
    """Locate the local directory containing Honda/HKKM ``.d.gz`` table files.

    Candidates are tried in order: an explicit ``honda_data_dir``
    argument, the ``HONDA_DATA_DIR`` environment variable, then this
    project's own default raw-table directory
    (``config.default.ATMOSPHERE_HONDA_RAW_DIR``, i.e. ``<datasets dir>/honda``,
    never the installed package: the full archive is ~280 files, ~289 MB,
    and the Honda/HKKM terms of use are unverified, see ``DATA_LICENSES.md``).
    The first candidate that
    exists and contains at least one ``*.d.gz`` file is returned.

    Args:
        honda_data_dir: Optional explicit directory path to try first.

    Returns:
        Path to the first matching directory containing Honda tables.

    Raises:
        FileNotFoundError: If no candidate directory contains ``.d.gz``
            files.
    """
    candidates = []
    if honda_data_dir is not None:
        candidates.append(Path(honda_data_dir))
    env_path = os.environ.get("HONDA_DATA_DIR")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(default.ATMOSPHERE_HONDA_RAW_DIR)

    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.d.gz")):
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find Honda .d.gz tables. Run `tpeanuts-data fetch honda` (or `fetch honda --from DIR` "
        f"with a copy), set HONDA_DATA_DIR, or pass honda_data_dir explicitly. Checked: {checked}"
    )


def _classify_table_name(name: str) -> Dict[str, Any]:
    """Parse a Honda table filename into site/season/solar/angular metadata; mirrors v1_0's own ``classify_table_name``.
    """
    stem = name[: -len(".d.gz")] if name.endswith(".d.gz") else name
    parts = stem.split("-")
    meta: Dict[str, Any] = {
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


def _choose_flux_file(data_dir: Path, config: HondaConfig) -> Path:
    """Select the Honda flux table file matching ``config``; mirrors v1_0's own ``choose_flux_file``."""
    matches: List[Path] = []
    for path in data_dir.glob("*.d.gz"):
        meta = _classify_table_name(path.name)
        if (
            meta.get("table_type") == "flux"
            and meta.get("site_code") == config.site_code
            and meta.get("season_code") == config.season_code
            and meta.get("solar") == config.solar
            and meta.get("mountain") == config.mountain
            and meta.get("angular_mode") == config.angular_mode
        ):
            matches.append(path)

    if not matches:
        raise FileNotFoundError(f"No Honda flux table matched {config!r} in {data_dir}.")
    return sorted(matches)[0]


def _choose_height_file(data_dir: Path, config: HondaConfig, particle: str) -> Optional[Path]:
    """Select the Honda production-height table file for one particle; mirrors v1_0's own ``choose_height_file``."""
    honda_flavour = TPEANUTS_TO_HONDA_HEIGHT.get(str(particle).lower())
    if honda_flavour is None:
        return None

    aa_token = "-aa-" if config.azimuth_averaged_height else "-"
    name = f"{config.site_code}-{config.season_code}{aa_token}{honda_flavour}.d.gz"
    path = data_dir / name
    if path.exists():
        return path

    fallback = data_dir / f"{config.site_code}-{config.season_code}-{honda_flavour}.d.gz"
    if fallback.exists():
        return fallback
    return None


def _select_height_source(
    height_tables: Dict[str, Optional[Dict[str, Any]]], particle: str
) -> Optional[Dict[str, Any]]:
    """Pick a production-height table for a particle, with a tau-neutrino fallback; mirrors v1_0's own
    ``_select_height_source``.
    """
    if height_tables.get(particle) is not None:
        return height_tables[particle]

    key = str(particle).lower()
    if "tau" in key:
        return height_tables.get("numu") if "anti" not in key else height_tables.get("antinumu")
    return None


# ============================================================
# Parse the raw .d.gz tables (NumPy, no torch -- see module docstring)
# ============================================================


def _read_flux_table(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Parse a Honda flux ``.d.gz`` file; mirrors v1_0's own ``read_flux_table``.

    Returns:
        Dict with ``"path"``, ``"energy_GeV"`` (1-D sorted array),
        ``"cosz_center"`` (1-D sorted array), and ``"flux_m2"`` (dict
        mapping each ``HONDA_FLUX_COLUMNS`` name to a
        ``(n_cosz, n_energy)`` array in ``(m^2 s sr GeV)^-1``).

    Raises:
        ValueError: If no flux rows could be parsed from the file.
    """
    rows: List[Dict[str, float]] = []
    current: Optional[Dict[str, float]] = None

    with gzip.open(Path(path), "rt", encoding="ascii", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            match = _FLUX_BLOCK_RE.search(line)
            if match:
                c1, c2, p1, p2 = map(float, match.groups())
                current = {
                    "cosz_min": min(c1, c2), "cosz_max": max(c1, c2), "cosz_center": 0.5 * (c1 + c2),
                    "phi_min_deg": min(p1, p2), "phi_max_deg": max(p1, p2), "phi_center_deg": 0.5 * (p1 + p2),
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
            rows.append({**current, "energy_GeV": values[0], **dict(zip(HONDA_FLUX_COLUMNS, values[1:], strict=True))})

    if not rows:
        raise ValueError(f"Could not read Honda flux rows from {path}.")

    energies = np.array(sorted({row["energy_GeV"] for row in rows}), dtype=float)
    cosz = np.array(sorted({row["cosz_center"] for row in rows}), dtype=float)
    flux = {flavour: np.full((cosz.size, energies.size), np.nan, dtype=float) for flavour in HONDA_FLUX_COLUMNS}

    e_index = {value: idx for idx, value in enumerate(energies)}
    z_index = {value: idx for idx, value in enumerate(cosz)}
    for row in rows:
        iz, ie = z_index[row["cosz_center"]], e_index[row["energy_GeV"]]
        for flavour in HONDA_FLUX_COLUMNS:
            flux[flavour][iz, ie] = row[flavour]

    return {"path": str(path), "energy_GeV": energies, "cosz_center": cosz, "flux_m2": flux}



SURFACE_FLUX_RAW_FILENAME = "frj-0608-20-12-solmin.d.gz"
_SURFACE_PARTICLES = ("numu", "numubar", "nue", "nuebar")


def _surface_table_rows(raw_path: Union[str, os.PathLike]):
    """Yield ``(energy_GeV, cos_zenith, azimuth_deg, particle, flux)`` for every entry of a raw Honda flux table.

    Block centres are used for the angles (``cosZ`` and ``phi_Az`` are bin ranges in the raw file); the
    four columns ``NuMu NuMubar NuE NuEbar`` become the particles ``numu numubar nue nuebar``, in that
    order, for each energy row. Fluxes stay in Honda's own unit, ``(m^2 s sr GeV)^-1``.
    """
    current: Optional[Tuple[float, float]] = None
    with gzip.open(Path(raw_path), "rt", encoding="ascii", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            match = _FLUX_BLOCK_RE.search(line)
            if match:
                c1, c2, p1, p2 = map(float, match.groups())
                current = (0.5 * (c1 + c2), 0.5 * (p1 + p2))
                continue
            if current is None or line.startswith("Enu"):
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                energy, *fluxes = (float(part) for part in parts)
            except ValueError:
                continue
            for particle, flux in zip(_SURFACE_PARTICLES, fluxes, strict=True):
                yield energy, current[0], current[1], particle, flux


def build_surface_flux_table(
    raw_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike, None] = None,
) -> Path:
    """Convert one raw Honda/HKKM flux table into the canonical surface-flux CSV.

    The output has the columns ``energy_GeV, cos_zenith, azimuth_deg, particle, flux`` that
    ``tpeanuts.source.atmosphere.io.load_atmospheric_flux`` reads, with the flux in Honda's own
    ``m^-2 s^-1 sr^-1 GeV^-1``. For ``SURFACE_FLUX_RAW_FILENAME`` the result has exactly the content of the table
    this project was developed and validated with (that file has CRLF line endings,
    this writer uses LF). A ``metadata/source.json`` recording the provenance
    is written next to the ``flux/`` directory.

    Args:
        raw_path: Raw Honda ``.d.gz`` table (one with ``cosZ``/``phi_Az`` flux blocks).
        output_path: Destination CSV. Defaults to ``honda/flux/honda_flux.csv`` under
            ``config.default.ATMOSPHERE_DATA_DIR`` (the per-user generated-data directory).

    Returns:
        The path of the CSV written (atomically: a failed run leaves no partial file).

    Raises:
        ValueError: If no flux rows could be parsed from ``raw_path``.
    """
    raw = Path(raw_path)
    destination = (
        Path(output_path) if output_path is not None
        else default.ATMOSPHERE_DATA_DIR / "honda" / "flux" / "honda_flux.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    n_rows = 0
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["energy_GeV", "cos_zenith", "azimuth_deg", "particle", "flux"])
        for row in _surface_table_rows(raw):
            writer.writerow(row)
            n_rows += 1
    if n_rows == 0:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Could not read Honda flux rows from {raw}.")
    os.replace(partial, destination)

    metadata_path = destination.parent.parent / "metadata" / "source.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({
        "provider": "Honda/HKKM",
        "reference": "M. Honda et al. (2015), arXiv:1502.03916",
        "raw_file": raw.name,
        "flux_unit": "m^-2 s^-1 sr^-1 GeV^-1",
        "canonical_schema": ["energy_GeV", "cos_zenith", "azimuth_deg", "particle", "flux"],
        "particles": list(_SURFACE_PARTICLES),
        "rows": n_rows,
        "generated_by": "tpeanuts.source.atmosphere.honda.build_surface_flux_table",
        "notes": "Honda tabulates surface flux. Built locally; not redistributed (see DATA_LICENSES.md).",
    }, indent=2) + "\n", encoding="utf-8")
    return destination


def _read_height_table(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Parse a Honda production-height ``.d.gz`` file; mirrors v1_0's own ``read_height_table``.

    Returns:
        Dict with ``"path"``, ``"energy_GeV"`` (1-D array, the common
        grid shared by all zenith blocks), ``"cosz_center"`` (1-D
        array, ascending), ``"probabilities"`` (1-D array in ``[0,
        1]``), and ``"height_quantiles_km"`` (``(n_cosz, n_energy,
        n_probabilities)`` array).

    Raises:
        ValueError: If no height rows could be parsed, or if different
            zenith blocks report inconsistent probability levels or
            energy grids.
    """
    blocks: Dict[Tuple[int, int], Dict[str, Any]] = {}
    current_key: Optional[Tuple[int, int]] = None

    with gzip.open(Path(path), "rt", encoding="ascii", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            match = _HEIGHT_HEADER_RE.match(line)
            if match:
                _site, _flavour, zenith_bin, azimuth_bin = map(int, match.groups()[:4])
                probabilities = np.array([float(v) for v in match.group(5).split()], dtype=float)
                current_key = (zenith_bin, azimuth_bin)
                blocks[current_key] = {
                    "probabilities": probabilities, "energy_GeV": [], "height_m": [],
                    "zenith_bin": zenith_bin, "azimuth_bin": azimuth_bin,
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

    # zenith_bin=1 -> cosz=0.95, bin=2 -> 0.85, ... (Honda's own 20-bin convention).
    cosz_unsorted = np.array([0.95 - 0.10 * (key[0] - 1) for key in sorted(blocks)], dtype=float)
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

    sort_idx = np.argsort(cosz_unsorted)
    cosz_sorted = cosz_unsorted[sort_idx]
    quantiles_sorted = np.stack(heights, axis=0)[sort_idx]

    return {
        "path": str(path), "energy_GeV": energy_ref, "cosz_center": cosz_sorted,
        "probabilities": probabilities_ref, "height_quantiles_km": quantiles_sorted,
    }


def _load_honda_tables(data_dir: Path, config: HondaConfig, particles: List[str]) -> Dict[str, Any]:
    """Load the flux table plus every requested particle's height table once; mirrors v1_0's own ``load_honda_tables``.
    """
    flux_path = _choose_flux_file(data_dir, config)
    flux_table = _read_flux_table(flux_path)

    height_tables: Dict[str, Optional[Dict[str, Any]]] = {}
    for particle in particles:
        path = _choose_height_file(data_dir, config, particle)
        height_tables[particle] = _read_height_table(path) if path is not None else None

    return {"flux_table": flux_table, "height_tables": height_tables}


# ============================================================
# Interpolation + production-height density reconstruction (NumPy)
# ============================================================


def _interp_logx(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linearly interpolate in ``log10(x)``; mirrors v1_0's own ``interp_logx``."""
    return np.interp(np.log10(x), np.log10(xp), fp)


def _trapezoid(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return trapezoid(y, x=x, axis=axis)


def _interpolate_flux_cm2(
    flux_table: Dict[str, Any], *, honda_flavour: str, cosz: float, energy_grid: np.ndarray
) -> np.ndarray:
    """Interpolate a Honda flux table onto an energy grid and cos(zenith); mirrors v1_0's own ``interpolate_flux_cm2``.

    Log-linear in energy, linear in cos(zenith) between the native
    Honda zenith bin centers; converted from ``(m^2 s sr GeV)^-1`` to
    ``(cm^2 s sr GeV)^-1``.
    """
    source_energy = flux_table["energy_GeV"]
    source_cosz = flux_table["cosz_center"]
    values = flux_table["flux_m2"][honda_flavour]

    by_cosz = np.empty((source_cosz.size, energy_grid.size), dtype=float)
    for iz in range(source_cosz.size):
        by_cosz[iz] = _interp_logx(energy_grid, source_energy, values[iz])

    flux_m2 = np.empty(energy_grid.size, dtype=float)
    for ie in range(energy_grid.size):
        flux_m2[ie] = np.interp(cosz, source_cosz, by_cosz[:, ie])

    return flux_m2 * _M2_TO_CM2_FLUX


def _interpolate_quantiles(
    height_table: Dict[str, Any], *, cosz: float, energy_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate Honda production-height quantiles onto an energy grid and cos(zenith); mirrors v1_0's own
    ``interpolate_quantiles``.
    """
    source_energy = height_table["energy_GeV"]
    source_cosz = height_table["cosz_center"]
    probabilities = height_table["probabilities"]
    quantiles = height_table["height_quantiles_km"]

    q_energy = np.empty((source_cosz.size, energy_grid.size, probabilities.size), dtype=float)
    for iz in range(source_cosz.size):
        for ip in range(probabilities.size):
            q_energy[iz, :, ip] = _interp_logx(energy_grid, source_energy, quantiles[iz, :, ip])

    sort_idx = np.argsort(source_cosz)
    source_cosz_asc = source_cosz[sort_idx]
    q_energy_asc = q_energy[sort_idx]

    q = np.empty((energy_grid.size, probabilities.size), dtype=float)
    for ie in range(energy_grid.size):
        for ip in range(probabilities.size):
            q[ie, ip] = np.interp(cosz, source_cosz_asc, q_energy_asc[:, ie, ip])

    return probabilities, q


def _density_from_quantiles(
    probabilities: np.ndarray, quantile_heights_km: np.ndarray, h_grid_km: np.ndarray
) -> np.ndarray:
    """Reconstruct a normalized production-height density ``f(h|E)`` from quantiles; mirrors v1_0's own
    ``density_from_quantiles``.

    For each energy row, the (probability, height) quantile pairs
    define a piecewise-linear CDF (anchored at ``h=0, CDF=0`` and the
    grid top, ``CDF=1``); the density is ``dCDF/dh``, clamped
    non-negative and renormalized to integrate to 1. Degenerate rows
    (zero or non-finite norm) fall back to a narrow density at ``h=0``.
    """
    f = np.zeros((quantile_heights_km.shape[0], h_grid_km.size), dtype=float)

    for ie, heights in enumerate(quantile_heights_km):
        order = np.argsort(heights)
        h_sorted = heights[order]
        p_sorted = probabilities[order]

        h_nodes = np.concatenate(([0.0], h_sorted, [max(float(h_grid_km[-1]), float(h_sorted[-1]))]))
        p_nodes = np.concatenate(([0.0], p_sorted, [1.0]))

        unique_h, unique_indices = np.unique(h_nodes, return_index=True)
        unique_p = p_nodes[unique_indices]
        cdf = np.interp(h_grid_km, unique_h, unique_p, left=0.0, right=1.0)
        density = np.gradient(cdf, h_grid_km, edge_order=1)
        density = np.clip(density, 0.0, None)
        norm = _trapezoid(density, x=h_grid_km)

        if norm > 0.0 and np.isfinite(norm):
            density = density / norm
        else:
            density = np.zeros_like(h_grid_km)
            density[0] = 1.0 / max(float(h_grid_km[1] - h_grid_km[0]), 1.0)
            density = density / _trapezoid(density, x=h_grid_km)

        f[ie] = density

    return f


def _flat_height_density(energy_grid: np.ndarray, h_grid_km: np.ndarray) -> np.ndarray:
    """Uniform production-height density, for particles with no Honda height table; mirrors v1_0's own
    ``flat_height_density``.
    """
    density = np.ones((energy_grid.size, h_grid_km.size), dtype=float)
    norm = _trapezoid(density, x=h_grid_km, axis=1)
    return density / norm[:, None]


# ============================================================
# Public entry point
# ============================================================


@torch.no_grad()
def generate_honda_flux(
    angles_deg: TensorLike,
    angle_type: str,
    *,
    depth_m: TensorLike = 0.0,
    particles: Optional[Dict[str, str]] = None,
    config: Optional[HondaConfig] = None,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
    honda_data_dir: Union[str, os.PathLike, None] = None,
) -> List[Path]:
    """Generate and save one differential-flux CSV per requested angle, from Honda/HKKM tables.

    For each angle in ``angles_deg``: resolves both the surface
    (``alpha``) and detector (``theta``) zenith angles via
    ``tpeanuts.medium.atmosphere.geometry`` (using ``depth_m`` only for
    that conversion), interpolates the Honda height-integrated flux
    Phi(E; X_obs) and production-height density f(h|E, alpha) for every
    requested particle, forms Phi(E,h) = Phi(E; X_obs) * f(h|E, alpha),
    and writes a single CSV holding every particle's rows -- in the
    same schema ``tpeanuts.source.atmosphere.io.load_atmospheric_flux``
    already reads, and the same per-angle-file convention
    ``tpeanuts.source.atmosphere.mceq.generate_mceq_flux`` uses. The
    Honda flux and per-particle height tables are loaded once and
    reused across every angle (unlike MCEq, which must reinitialize a
    fresh run per angle, Honda's own tables are angle-independent --
    only the interpolation target changes).

    Args:
        angles_deg: Scalar or 1-D iterable of angles in degrees, in the
            convention named by ``angle_type``.
        angle_type: ``"alpha"`` (surface/Honda-table zenith angle, ``0
            <= alpha_deg < 90``) or ``"theta"`` (detector zenith angle,
            any value convertible to a valid ``alpha_deg`` at the given
            ``depth_m``).
        depth_m: Detector depth below the surface, in metres, used only
            for the ``alpha``/``theta`` conversion. Defaults to
            ``0.0``.
        particles: Mapping from output flavour label to tpeanuts
            particle name (e.g. ``{"numu": "numu"}``). Particles
            without a matching Honda flux flavour get an all-zero flux;
            particles without a height table fall back to a flat
            height density. Defaults to
            ``tpeanuts.config.default.ATMOSPHERE_HONDA_PARTICLES``.
        config: Optional ``HondaConfig``; defaults to ``HondaConfig()``.
        output_dir: Directory the per-angle CSVs are written into.
            Defaults to
            ``tpeanuts/data/atmosphere/<ATMOSPHERE_HONDA_FLUX_DIRNAME>/``.
        overwrite: If False (default) and an angle's output file
            already exists, that angle is skipped without recomputation.
        honda_data_dir: Optional directory hint forwarded to
            ``find_honda_data_dir``.

    Returns:
        List of paths written (or already present, when
        ``overwrite=False`` skipped them), one per requested angle, in
        the same order as ``angles_deg``.

    Raises:
        FileNotFoundError: If no Honda ``.d.gz`` tables can be located,
            or no flux table matches ``config``.
        ValueError: If ``angle_type`` is not ``"alpha"``/``"theta"``, if
            ``config`` fails ``validate()``, or if any angle converts to
            an ``alpha_deg`` outside ``[0, 90)``.
    """
    if angle_type not in ("alpha", "theta"):
        raise ValueError(f"angle_type must be 'alpha' or 'theta', got {angle_type!r}.")

    if config is None:
        config = HondaConfig()
    config.validate()

    if particles is None:
        particles = default.ATMOSPHERE_HONDA_PARTICLES

    if output_dir is None:
        output_dir = default.ATMOSPHERE_DATA_DIR / default.ATMOSPHERE_HONDA_FLUX_DIRNAME
    output_dir = Path(output_dir)

    data_dir = find_honda_data_dir(honda_data_dir)
    tables = _load_honda_tables(data_dir, config, list(particles.values()))
    flux_table = tables["flux_table"]
    height_tables = tables["height_tables"]

    energy_np = (
        np.asarray(flux_table["energy_GeV"], dtype=float)
        if config.energy_grid_GeV is None
        else config.energy_grid_GeV.detach().cpu().numpy()
    )
    h_np = config.h_grid_km.detach().cpu().numpy()

    angles_t = as_tensor(angles_deg, dtype=torch.float64).reshape(-1)
    depth_context = RuntimeContext(device=torch.device("cpu"), dtype=torch.float64)

    written_paths: List[Path] = []
    for angle_value in angles_t.tolist():
        if angle_type == "alpha":
            alpha_deg = float(angle_value)
            theta_deg = float(alpha_surface_to_theta_detector(angle_value, depth_m, context=depth_context).item())
        else:
            theta_deg = float(angle_value)
            alpha_deg = float(theta_detector_to_alpha_surface(angle_value, depth_m, context=depth_context).item())

        if not (0.0 <= alpha_deg < 90.0):
            raise ValueError(
                f"angle {angle_value!r} ({angle_type}) converts to a surface zenith angle "
                f"alpha_deg={alpha_deg}, outside Honda's own valid domain [0, 90) for this "
                "generator -- an upward-going/horizontal trajectory needs the Earth-crossing "
                "pipeline, not a direct Honda table lookup."
            )

        cosz = float(np.cos(np.deg2rad(alpha_deg)))
        cos_zenith = float(np.cos(np.deg2rad(theta_deg)))
        out_path = angle_output_path(output_dir, alpha_deg)

        if out_path.exists() and not overwrite:
            written_paths.append(out_path)
            continue

        per_particle_flux: Dict[str, torch.Tensor] = {}
        for flavour_name, particle in particles.items():
            honda_flavour = TPEANUTS_TO_HONDA.get(str(particle).lower())
            if honda_flavour is None:
                phi_E_obs = np.zeros(energy_np.size, dtype=float)
            else:
                phi_E_obs = _interpolate_flux_cm2(
                    flux_table, honda_flavour=honda_flavour, cosz=cosz, energy_grid=energy_np
                )

            height_table = _select_height_source(height_tables, particle)
            if height_table is None:
                f_Eh = _flat_height_density(energy_np, h_np)
            else:
                probabilities, quantiles = _interpolate_quantiles(height_table, cosz=cosz, energy_grid=energy_np)
                f_Eh = _density_from_quantiles(probabilities, quantiles, h_np)

            per_particle_flux[flavour_name] = torch.as_tensor(phi_E_obs[:, None] * f_Eh, dtype=torch.float64)

        write_angle_flux_csv(
            out_path,
            energy_GeV=torch.as_tensor(energy_np, dtype=torch.float64),
            altitude_km=torch.as_tensor(h_np, dtype=torch.float64),
            per_particle_flux=per_particle_flux,
            cos_zenith=cos_zenith,
        )
        written_paths.append(out_path)

    return written_paths
