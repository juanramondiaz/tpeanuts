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
earth density input/output utilities for peanuts-torch.

This module contains helper functions for loading earth electron-density
profiles from external files and converting them into EarthDensity
objects.

The expected peanuts density format is a CSV table with columns such as:

    rj,
    alpha,
    beta,
    gamma,
    delta1,
    delta2,
    ...

where:

    rj:
        Dimensionless shell radius.

    alpha, beta, gamma:
        Polynomial density coefficients.

    deltaN:
        Higher-order polynomial coefficients.

The main public function is:

    load_earth_density_from_csv(...)

which loads the CSV file and returns a fully initialized
EarthDensity object.

This module only handles file parsing and tensor construction.
It does not compute shell crossings, trajectory coefficients,
Hamiltonians, or evolution operators.
"""



from __future__ import annotations

from typing import Union, Literal, Optional

import os

import pandas as pd
import torch

import tpeanuts.util.default as default
from tpeanuts.earth.density import EarthDensity

from tpeanuts.util.type import (
    _as_tensor,
)

from tpeanuts.util.math import interp1d_linear

from tpeanuts.earth.exposure_math import (
    make_eta_grid, _daynight_slice
    )

TensorLike = Union[float, int, torch.Tensor]
AngleMode = Literal["Nadir", "Zenith", "CosZenith"]
DayNight = Optional[Literal["day", "night"]]


# ============================================================
# Read earth Denisity CSV File
# ============================================================

def extract_delta_columns(
    table: pd.DataFrame,
) -> list[str]:
    return [
        column_name
        for column_name in table.columns
        if column_name.startswith("delta")
    ]


def parse_density_table(
    table: pd.DataFrame,
    *,
    tabulated_density: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> EarthDensity:
    rj = _as_tensor(
        table["rj"],
        device=device,
        dtype=dtype,
    )

    alpha = _as_tensor(
        table["alpha"],
        device=device,
        dtype=dtype,
    )

    if tabulated_density:

        beta = torch.zeros_like(
            rj,
            device=device,
            dtype=dtype,
        )

        gamma = torch.zeros_like(
            rj,
            device=device,
            dtype=dtype,
        )

        deltas = torch.zeros(
            (0, rj.numel()),
            device=device,
            dtype=dtype,
        )

    else:

        beta = _as_tensor(
            table.get("beta", torch.zeros_like(rj)),
            device=device,
            dtype=dtype,
        )

        gamma = _as_tensor(
            table.get("gamma", torch.zeros_like(rj)),
            device=device,
            dtype=dtype,
        )

        delta_names = extract_delta_columns(table)

        if len(delta_names) == 0:

            deltas = torch.zeros(
                (0, rj.numel()),
                device=device,
                dtype=dtype,
            )

        else:

            deltas = torch.stack(
                [
                    _as_tensor(
                        table[name],
                        device=device,
                        dtype=dtype,
                    )
                    for name in delta_names
                ],
                dim=0,
            )

    return EarthDensity(
        rj=rj,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        deltas=deltas,
        tabulated=tabulated_density,
    )


def load_earth_density_from_csv(
    density_file: str,
    *,
    tabulated_density: bool = default.earth_tabulated_density,
    device: Union[str, torch.device] = default.earth_device,
    dtype: torch.dtype = default.dtype,
) -> EarthDensity:
    device = torch.device(device)

    if not os.path.isfile(density_file):
        raise FileNotFoundError(
            f"density file not found: {density_file}"
        )

    table = pd.read_csv(density_file)

    return parse_density_table(
        table,
        tabulated_density=tabulated_density,
        device=device,
        dtype=dtype,
    )


def attach_csv_loader_to_density_class() -> None:
    EarthDensity.from_csv = staticmethod(
        load_earth_density_from_csv
    )



# ============================================================
# Nadir Exposure Source 2: cache
# ============================================================
def _cache_filename(
    cache_dir: str,
    lam_rad: float,
    d1: float,
    d2: float,
    ns: int,
    daynight: DayNight,
) -> str:
    filename = (
        f"nadir_exposure"
        f"_lam{lam_rad:.8f}"
        f"_d{d1:.3f}-{d2:.3f}"
        f"_ns{ns}"
        f"_dn{daynight or 'all'}"
        f".pt"
    )

    return os.path.join(cache_dir, filename)


@torch.no_grad()
def nadir_exposure_from_cache(
    lam_rad: float,
    d1: float = default.earth_d1,
    d2: float = default.earth_d2,
    ns: int = default.earth_exposure_ns,
    *,
    daynight: DayNight = default.earth_daynight,
    cache_dir: str = default.earth_cache_dir,
    device: Union[str, torch.device] = default.earth_device,
    dtype: torch.dtype = default.dtype,
) ->  tuple[torch.Tensor, torch.Tensor]:
    path = _cache_filename(
        cache_dir,
        lam_rad,
        d1,
        d2,
        ns,
        daynight,
    )


    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cached nadir-exposure file not found: {path}"
        )

    try:
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        data = torch.load(
            path,
            map_location="cpu",
        )

    if "eta" not in data or "exposure" not in data:
        raise KeyError(
            "Cached file must contain keys 'eta' and 'exposure'."
        )

    dev = torch.device(device)

    eta = data["eta"].to(
        device=dev,
        dtype=dtype,
    )

    exposure = data["exposure"].to(
        device=dev,
        dtype=dtype,
    )

    if eta.ndim != 1 or exposure.ndim != 1:
        raise ValueError(
            "Cached 'eta' and 'exposure' must be 1D tensors."
        )

    if eta.numel() != exposure.numel():
        raise ValueError(
            "'eta' and 'exposure' must have the same length."
        )
        
    return eta, exposure

@torch.no_grad()
def save_nadir_exposure_to_cache(
    eta: torch.Tensor,
    exposure: torch.Tensor,
    lam_rad: float,
    d1: float = default.earth_d1,
    d2: float = default.earth_d2,
    ns: Optional[int] = None,
    *,
    daynight: DayNight = default.earth_daynight,
    cache_dir: str = default.earth_legacy_cache_dir,
) -> str:
    os.makedirs(
        cache_dir,
        exist_ok=True,
    )

    if ns is None:

        if daynight is None:
            ns = eta.numel()

        elif daynight in ("day", "night"):
            ns = 2 * eta.numel()

        else:
            raise ValueError(
                "daynight must be None, 'day', or 'night'."
            )

    filename = _cache_filename(
        cache_dir,
        lam_rad,
        d1,
        d2,
        ns,
        daynight,
    )

    path = filename

    data = {
        "eta": eta.detach().cpu(),
        "exposure": exposure.detach().cpu(),
        "metadata": {
            "lam_rad": float(lam_rad),
            "d1": float(d1),
            "d2": float(d2),
            "ns": int(ns),
            "daynight": daynight,
        },
    }

    torch.save(
        data,
        path,
    )

    return path



# ============================================================
# Nadir Exposure Source 3: CSV file
# ============================================================

def _read_csv_exposure_column(filepath: str) -> torch.Tensor:

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath)

    if "Exposure" not in df.columns:
        raise ValueError("CSV file must contain an 'Exposure' column.")

    return torch.tensor(
        df["Exposure"].to_numpy(),
        dtype=torch.float64,
    )

@torch.no_grad()
def _convert_csv_angle_mode(
    raw: torch.Tensor,
    eta: torch.Tensor,
    *,
    angle: AngleMode,
    dtype: torch.dtype,
) -> torch.Tensor:

    raw = raw.to(device=eta.device, dtype=dtype)
    ns = raw.numel()

    if angle == "Nadir":
        return raw

    if angle == "Zenith":
        return torch.flip(raw, dims=(0,))

    if angle == "CosZenith":
        cz = torch.linspace(
            -1.0,
            1.0,
            ns,
            device=eta.device,
            dtype=dtype,
        )

        dcz = cz[1] - cz[0]
        deta = torch.pi / (ns - 1)

        exposure = interp1d_linear(
            x=-torch.cos(eta),
            xp=cz,
            fp=raw,
            left=0.0,
            right=0.0,
            device=eta.device,
            dtype=dtype,
        )

        exposure = exposure * torch.sin(eta) * deta / dcz
        return torch.clamp(exposure, min=0.0)

    raise ValueError("angle must be 'Nadir', 'Zenith' or 'CosZenith'.")


@torch.no_grad()
def nadir_exposure_from_csv(
    filepath: str,
    *,
    angle: AngleMode = default.earth_angle,
    daynight: DayNight = default.earth_daynight,
    device: Union[str, torch.device] = default.earth_device,
    dtype: torch.dtype = default.dtype,
) ->  tuple[torch.Tensor, torch.Tensor]:

    device = torch.device(device)

    raw = _read_csv_exposure_column(filepath)

    eta_full = make_eta_grid(
        raw.numel(),
        daynight=None,
        device=device,
        dtype=dtype,
    )
    
    exposure_full = _convert_csv_angle_mode(
        raw,
        eta_full,
        angle=angle,
        dtype=dtype,
    )

    eta = _daynight_slice(eta_full, raw.numel(), daynight)
    exposure = _daynight_slice(exposure_full, raw.numel(), daynight)
    
    return eta, exposure
