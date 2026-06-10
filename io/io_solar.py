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
solar data input/output helpers for the torch-native solar block.

The torch implementation reads primary solar files from:

    tpeanuts/data/solar

Legacy validation utilities read original peanuts files from:

    tpeanuts/data/peanuts
"""



from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
import torch

from tpeanuts.util.torch_util import _default_device


Tensor = torch.Tensor


def package_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def default_solar_data_dir() -> Path:
    return package_dir() / "data" / "solar"


def default_legacy_data_dir() -> Path:
    return package_dir() / "data" / "peanuts"


def _as_tensor(values, *, device: Union[str, torch.device], dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(values, device=_default_device(device), dtype=dtype)


def load_b16_solar_model(
    path: str | Path | None = None,
    *,
    device: Union[str, torch.device] | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor | dict[str, Tensor]]:
    if path is None:
        path = default_solar_data_dir() / "nudistr_b16_agss09.csv"

    table = pd.read_csv(path)

    radius = _as_tensor(table["radius"].to_numpy(), device=device, dtype=dtype)
    density = 10.0 ** _as_tensor(table["density_log_10"].to_numpy(), device=device, dtype=dtype)

    fractions = {}
    for column in table.columns:
        if column.endswith(" fraction"):
            name = column.replace(" fraction", "")
            fractions[name] = _as_tensor(table[column].to_numpy(), device=device, dtype=dtype)

    return {
        "radius": radius,
        "density": density,
        "fractions": fractions,
    }


def load_b16_fluxes(
    path: str | Path | None = None,
    *,
    device: Union[str, torch.device] | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    if path is None:
        path = default_solar_data_dir() / "fluxes_b16.csv"

    table = pd.read_csv(path)

    return {
        str(row["fraction"]): _as_tensor(row["flux"], device=device, dtype=dtype)
        for _, row in table.iterrows()
    }


def load_spectrum_csv(
    path: str | Path,
    *,
    energy_column: str | None = None,
    spectrum_column: str | None = None,
    device: Union[str, torch.device] | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    table = pd.read_csv(path)

    if energy_column is None:
        energy_column = table.columns[0]

    if spectrum_column is None:
        spectrum_column = table.columns[1]

    return {
        "energy": _as_tensor(table[energy_column].to_numpy(), device=device, dtype=dtype),
        "spectrum": _as_tensor(table[spectrum_column].to_numpy(), device=device, dtype=dtype),
    }


def load_sun_earth_distance(
    path: str | Path | None = None,
    *,
    device: Union[str, torch.device] | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, list[str] | Tensor]:
    if path is None:
        path = default_solar_data_dir() / "sun_earth_distance.csv"

    table = pd.read_csv(path)

    required = {"date", "distance_km", "distance_AU"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "Sun-earth distance table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return {
        "date": [str(value) for value in table["date"].to_list()],
        "distance_km": _as_tensor(table["distance_km"].to_numpy(), device=device, dtype=dtype),
        "distance_AU": _as_tensor(table["distance_AU"].to_numpy(), device=device, dtype=dtype),
    }
