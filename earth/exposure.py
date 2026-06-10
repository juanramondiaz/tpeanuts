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
earth nadir-exposure input/output utilities for peanuts-torch.

This module loads or builds nadir-angle exposure tables used to compute
time-averaged earth matter-regeneration probabilities.

The exposure table represents a weight function

    W(eta)

defined on eta in [0, pi]. It is later used by earth/integration.py to compute

    P_int(E) = ∫ d eta W(eta) P_earth(E, eta).

This module provides two main public constructors:

    nadir_exposure_from_file(...)
        Loads an exposure table from an experiment file.

    nadir_exposure_from_latitude_cached(...)
        Computes the exposure using the original peanuts CPU implementation
        once, stores it in a cache file, and reloads it as torch tensors.

The module is intentionally separated from earth/exposure_math.py because it
depends on optional CPU-side libraries such as numpy, pandas, scipy, and the
legacy peanuts implementation.
"""



from __future__ import annotations

from typing import Literal, Optional, Union

import os
from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch

import tpeanuts.util.default as default
from tpeanuts.util.math import interp1d_linear, normalize_trapz
from tpeanuts.earth.exposure_math import (
    IntegralDay,
    make_eta_grid, _daynight_slice
    )
from tpeanuts.io.io_earth import (
    nadir_exposure_from_cache,
    save_nadir_exposure_to_cache,
    nadir_exposure_from_csv,
    _cache_filename
    )

TensorLike = Union[float, int, torch.Tensor]
AngleMode = Literal["Nadir", "Zenith", "CosZenith"]
DayNight = Optional[Literal["day", "night"]]
ExposureSource = Literal["math", "cache", "csv", "legacy"]


@dataclass
class NadirExposureTable:
    """
    GPU-ready nadir-angle exposure table.

    Parameters
    ----------
    eta:
        Nadir-angle grid in radians. Shape: (Neta,).

    exposure:
        Exposure weights evaluated on eta. Shape: (Neta,).
    """

    eta: torch.Tensor
    exposure: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.eta.device

    @property
    def dtype(self) -> torch.dtype:
        return self.eta.dtype

    def normalize_(
        self,
        eps: float = default.earth_normalize_eps,
    ) -> "NadirExposureTable":
        self.exposure = normalize_trapz(
            self.exposure,
            x=self.eta,
            clamp_min=0.0,
            eps=eps,
        )

        return self

    @torch.no_grad()
    def interp(
        self,
        eta_query: torch.Tensor,
    ) -> torch.Tensor:
        return interp1d_linear(
            x=eta_query,
            xp=self.eta,
            fp=self.exposure,
            left=self.exposure[0],
            right=self.exposure[-1],
            device=self.device,
            dtype=self.dtype,
        )


# ============================================================
# Source 1: direct mathematical computation
# ============================================================


@torch.no_grad()
def nadir_exposure_from_math(
    lam_rad: TensorLike,
    *,
    d1: float = default.earth_d1,
    d2: float = default.earth_d2,
    ns: int = default.earth_exposure_ns,
    daynight: DayNight = default.earth_daynight,
    inclination: float = default.earth_inclination,
    device: Union[str, torch.device] = default.earth_device,
    dtype: torch.dtype = default.dtype,
) -> NadirExposureTable:

    device = torch.device(device)

    eta = make_eta_grid(
        ns,
        daynight=daynight,
        device=device,
        dtype=dtype,
    )

    lam = torch.as_tensor(
        lam_rad,
        device=device,
        dtype=dtype,
    )

    exposure = torch.stack(
        [
            IntegralDay(
                eta_i,
                lam,
                d1=d1,
                d2=d2,
                inclination = inclination,
                device=device,
                dtype=dtype,
            )
            for eta_i in eta
        ]
    )
    
    return eta, exposure


# ==============================================================
# Source 4: Compute from Legacy peanuts (for Validation purpose)
# ==============================================================


def nadir_exposure_from_legacy_peanuts(
    lam_rad: float,
    d1: float = default.earth_d1,
    d2: float = default.earth_d2,
    ns: int = default.earth_exposure_ns,
    *,
    daynight: DayNight = default.earth_daynight,
    device: Union[str, torch.device] = default.earth_device,
    dtype: torch.dtype = default.dtype,
) ->  tuple[torch.Tensor, torch.Tensor]:

    from peanuts.time_average import NadirExposure
    
    table = NadirExposure(
        lam=lam_rad,
        d1=d1,
        d2=d2,
        ns=ns,
        normalized=False,
        from_file=None,
        angle="Nadir",
        daynight=daynight,
    )

    eta_np = table[:, 0]
    exposure_np = table[:, 1]


    eta = torch.as_tensor(
        eta_np,
        device=device,
        dtype=dtype,
    )

    exposure = torch.as_tensor(
        exposure_np,
        device=device,
        dtype=dtype,
    )

    return eta, exposure


# ============================================================
# Main public builder
# ============================================================

@torch.no_grad()
def build_nadir_exposure(
    *,
    source: ExposureSource,
    lam_rad: Optional[float] = None,
    d1: float = default.earth_d1,
    d2: float = default.earth_d2,
    ns: int = default.earth_exposure_ns,
    daynight: DayNight = default.earth_daynight,
    inclination: float = default.earth_inclination,
    normalized: bool = default.earth_normalized_exposure,
    csv_path: Optional[str] = None,
    angle: AngleMode = default.earth_angle,
    cache_dir: str = default.earth_legacy_cache_dir,
    use_cache: bool = default.earth_use_cache,
    device: Union[str, torch.device] = default.earth_device,
    dtype: torch.dtype = default.dtype,
) -> NadirExposureTable:
    
    if source == "csv":

        if csv_path is None:
            raise ValueError("csv_path must be provided when source='csv'.")

        eta, exposure = nadir_exposure_from_csv(
            csv_path,
            angle=angle,
            daynight=daynight,
            device=device,
            dtype=dtype,
        )

    elif source == "cache":

        if lam_rad is None:
            raise ValueError("lam_rad must be provided when source='cache'.")

        eta, exposure = nadir_exposure_from_cache(
            cache_dir = cache_dir,
            lam_rad = lam_rad,
            d1 = d1,
            d2 = d2,
            ns = ns,
            device=device,
            dtype=dtype,
        )

    elif source == "math":

        if lam_rad is None:
            raise ValueError("lam_rad must be provided when source='math'.")

        path_cache = _cache_filename(
            cache_dir,
            lam_rad,
            d1,
            d2,
            ns,
            daynight,
        )
                
        if use_cache and os.path.exists(path_cache):
            eta, exposure = nadir_exposure_from_cache(
            cache_dir = cache_dir,
            lam_rad = lam_rad,
            d1 = d1,
            d2 = d2,
            ns = ns,
            device=device,
            dtype=dtype,
        )

        else:

            eta, exposure = nadir_exposure_from_math(
                lam_rad,
                d1=d1,
                d2=d2,
                ns=ns,
                inclination = inclination,
                daynight=daynight,
                device=device,
                dtype=dtype,
            )
            
    elif source == "legacy":

        if lam_rad is None:
            raise ValueError("lam_rad must be provided when source='legacy'.")

        eta, exposure = nadir_exposure_from_legacy_peanuts(
            lam_rad=lam_rad,
            d1=d1,
            d2=d2,
            ns=ns,
            daynight=daynight,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError("source must be 'math', 'cache', 'csv' or 'legacy'.")

    table = NadirExposureTable(
        eta=eta,
        exposure=exposure,
    )

    if normalized:
        table.normalize_()

    if use_cache and source in {"math", "legacy"}:
        if lam_rad is None:
            raise ValueError(f"lam_rad must be provided to cache source={source!r}.")

        save_nadir_exposure_to_cache(
            eta = table.eta,
            exposure = table.exposure,
            lam_rad = lam_rad,
            d1 = d1,
            d2 = d2,
            ns = ns,
            daynight = daynight,
            cache_dir = cache_dir,
            )

    return table
