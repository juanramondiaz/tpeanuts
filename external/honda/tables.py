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
Readers for Honda/HKKM atmospheric-neutrino tables.

Honda flux tables are tabulated as differential fluxes in
(m^2 s sr GeV)^-1. TPeanuts atmospheric source files use the same spectral
quantity per cm^2, so generator code divides Honda fluxes by 1e4 before
saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import os
import re
from typing import Any, Optional

import numpy as np


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
    site_code: str = "frj"
    season_code: str = "ally"
    solar: str = "solmin"
    mountain: bool = False
    angular_mode: str = "azimuth-averaged"
    azimuth_averaged_height: bool = True


def find_honda_data_dir(path: str | os.PathLike[str] | None = None) -> Path:
    candidates = []

    if path is not None:
        candidates.append(Path(path))

    env_path = os.environ.get("HONDA_DATA_DIR", None)
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path(r"G:\Mi unidad\04.Datasets\Honda"),
            Path(r"G:\Mi unidad\03.Codigo\034.TFM.UV\External\Honda"),
        ]
    )

    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.d.gz")):
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find Honda .d.gz tables. Set HONDA_DATA_DIR or pass "
        f"honda_data_dir explicitly. Checked: {checked}"
    )


def classify_table_name(name: str) -> dict[str, Any]:
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


def honda_cosz_centers() -> np.ndarray:
    return np.array([0.95 - 0.10 * idx for idx in range(20)], dtype=float)


def zenith_bin_to_cosz_center(zenith_bin: int) -> float:
    return float(0.95 - 0.10 * (int(zenith_bin) - 1))


def read_flux_table(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
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

    cosz = np.array(
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

    return {
        "path": str(path),
        "energy_GeV": energy_ref,
        "cosz_center": cosz,
        "probabilities": probabilities_ref,
        "height_quantiles_km": np.stack(heights, axis=0),
    }
