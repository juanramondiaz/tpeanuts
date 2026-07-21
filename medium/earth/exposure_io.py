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
Earth nadir-exposure input/output utilities.

This module contains helper functions for loading and saving nadir-exposure
tables used by ``medium.earth.exposure_table``. It handles cache files and CSV
exposure tables only; model-specific Earth-density files belong to the
perturbative profile model packages.

Module functions:
    nadir_exposure_from_cache(...): Load a cached torch exposure table.
    save_nadir_exposure_to_cache(...): Save eta/exposure tensors to cache.
    nadir_exposure_from_csv(...): Load and convert an exposure CSV table.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import os

import pandas as pd
import torch

import tpeanuts.config.default as default
from tpeanuts.medium.earth.exposure_math import make_eta_grid, _daynight_slice
from tpeanuts.util.math import interp1d_linear


AngleMode = Literal["Nadir", "Zenith", "CosZenith"]
DayNight = Optional[Literal["day", "night"]]


def _cache_filename(
    cache_dir: str,
    lam_rad: float,
    d1: float,
    d2: float,
    ns: int,
    daynight: DayNight,
) -> str:
    """Build the deterministic cache filename for a nadir-exposure table."""
    filename = (
        f"{default.earth_exposure_cache_filename_prefix}"
        f"_lam{lam_rad:.8f}"
        f"_d{d1:.3f}-{d2:.3f}"
        f"_ns{ns}"
        f"_dn{daynight or 'all'}"
        f"{default.torch_default_extension}"
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a cached nadir-exposure table.

    Args:
        lam_rad: Detector latitude in radians.
        d1: Initial day.
        d2: Final day.
        ns: Number of grid samples stored in the cache filename.
        daynight: Optional day/night subset selector.
        cache_dir: Directory containing cached torch files.
        device: Device for returned tensors.
        dtype: Real dtype for returned tensors.

    Returns:
        Tuple ``(eta, exposure)`` loaded on the requested device and dtype.
    """
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
    eta = data["eta"].to(device=dev, dtype=dtype)
    exposure = data["exposure"].to(device=dev, dtype=dtype)

    if eta.ndim != 1 or exposure.ndim != 1:
        raise ValueError("Cached 'eta' and 'exposure' must be 1D tensors.")
    if eta.numel() != exposure.numel():
        raise ValueError("'eta' and 'exposure' must have the same length.")

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
    """Save a nadir-exposure table to the standard torch cache format."""
    os.makedirs(cache_dir, exist_ok=True)

    if ns is None:
        if daynight is None:
            ns = eta.numel()
        elif daynight in ("day", "night"):
            ns = 2 * eta.numel()
        else:
            raise ValueError("daynight must be None, 'day', or 'night'.")

    path = _cache_filename(
        cache_dir,
        lam_rad,
        d1,
        d2,
        ns,
        daynight,
    )

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

    torch.save(data, path)
    return path


def _read_csv_exposure_column(
    filepath: str,
) -> torch.Tensor:
    """Read the raw exposure column from a CSV file."""
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
    """Convert raw CSV exposure samples to the requested angle convention."""
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a CSV exposure table and convert it to a nadir grid."""
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
