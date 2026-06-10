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
earth probability integration utilities for peanuts-torch.

This module computes time-averaged or exposure-averaged earth matter
regeneration probabilities.

It sits above:

    earth.probabilities
        Computes P_earth(E, eta).

    time_average / exposure utilities
        Provide the nadir-angle exposure table W(eta).

The main public function is:

    pearth_integrated(...)

which computes

    P_int(E) = ∫ d eta W(eta) P_earth(E, eta).

The implementation supports chunking in eta in order to control memory usage
when evaluating large energy-angle grids on GPU.
"""



from __future__ import annotations

from typing import Optional, Union, Literal
from types import SimpleNamespace

import torch

import tpeanuts.util.default as default
from tpeanuts.earth.probabilities import pearth
from tpeanuts.io.io_earth import (
    nadir_exposure_from_cache,
    save_nadir_exposure_to_cache,
    nadir_exposure_from_csv,
)

Tensor = torch.Tensor
TensorLike = Union[float, int, torch.Tensor]

AngleMode = Literal["Nadir", "Zenith", "CosZenith"]
DayNight = Optional[Literal["day", "night"]]
PearthMethod = Literal["analytical", "numerical"]


def _prepare_energy_grid(
    E_MeV: TensorLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, bool]:
    if torch.is_tensor(E_MeV):
        E = E_MeV.to(device=device, dtype=dtype)
    else:
        E = torch.tensor(E_MeV, device=device, dtype=dtype)

    if E.ndim == 0:
        E = E[None]
        squeeze_E = True
    else:
        squeeze_E = False

    return E, squeeze_E

  

def _load_exposure_table(
    *,
    from_file: Optional[str],
    lam_rad: float,
    d1: float,
    d2: float,
    ns: int,
    cache_dir: str,
    normalized_exposure: bool,
    angle: AngleMode,
    daynight: DayNight,
    device: torch.device,
    dtype: torch.dtype,
):
    if from_file is not None:
        eta, exposure = nadir_exposure_from_csv(
            from_file,
            angle=angle,
            daynight=daynight,
            device=device,
            dtype=dtype,
        )
    else:
        if lam_rad < 0:
            raise ValueError("Provide either from_file or lam_rad >= 0.")

        eta, exposure = nadir_exposure_from_cache(
            lam_rad=lam_rad,
            d1=d1,
            d2=d2,
            ns=ns,
            daynight=daynight,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
        )

    if normalized_exposure:
        norm = torch.trapz(exposure, x=eta)
        exposure = exposure / torch.clamp(norm, min=torch.finfo(dtype).tiny)

    return SimpleNamespace(
        eta=eta,
        exposure=exposure,
    )

@torch.no_grad()
def pearth_integrated(
    nustate: Tensor,
    density: object,
    pmns: object,
    dm21_eV2: TensorLike,
    dm3l_eV2: TensorLike,
    E_MeV: TensorLike,
    depth_m: float,
    *,
    method: PearthMethod = default.earth_method,
    antinu: Union[bool, Tensor] = default.earth_antinu,
    massbasis: bool = default.earth_massbasis,
    lam_rad: float = default.earth_lam_rad,
    d1: float = default.earth_d1,
    d2: float = default.earth_d2,
    ns: int = default.earth_exposure_ns,
    cache_dir: str = default.earth_cache_dir,
    normalized_exposure: bool = default.earth_normalized_exposure,
    from_file: Optional[str] = None,
    angle: AngleMode = default.earth_angle,
    daynight: DayNight = default.earth_daynight,
    device: Union[str, torch.device] = default.earth_device,
    dtype: torch.dtype = default.dtype,
    chunk_eta: Optional[int] = default.earth_chunk_eta,
    reunitarize: bool = default.earth_reunitarize,
    nsteps: int = default.earth_probability_nsteps,
    ode_method: str | None = default.earth_numerical_method,
) -> Tensor:
    dev = torch.device(device)

    if method not in ("analytical", "numerical"):
        raise ValueError("method must be either 'analytical' or 'numerical'.")

    exposure_table = _load_exposure_table(
        from_file=from_file,
        lam_rad=lam_rad,
        d1=d1,
        d2=d2,
        ns=ns,
        cache_dir=cache_dir,
        normalized_exposure=normalized_exposure,
        angle=angle,
        daynight=daynight,
        device=dev,
        dtype=dtype,
    )

    eta_grid = exposure_table.eta
    w_eta = exposure_table.exposure

    deta = torch.pi / ns

    E, squeeze_E = _prepare_energy_grid(
        E_MeV,
        device=dev,
        dtype=dtype,
    )

    n_energy = E.shape[0]

    antinu_t = None
    if torch.is_tensor(antinu):
        antinu_t = antinu.to(device=dev, dtype=torch.bool)

    out = torch.zeros(
        (n_energy, 3),
        device=dev,
        dtype=dtype,
    )

    if chunk_eta is None or chunk_eta <= 0:
        chunk_eta = eta_grid.numel()

    for start in range(0, eta_grid.numel(), chunk_eta):

        eta_chunk = eta_grid[start:start + chunk_eta]
        w_chunk = w_eta[start:start + chunk_eta]

        E_grid = E[:, None].expand(
            n_energy,
            eta_chunk.numel(),
        )

        eta_grid_chunk = eta_chunk[None, :].expand(
            n_energy,
            eta_chunk.numel(),
        )

        antinu_chunk = antinu
        if antinu_t is not None:
            if antinu_t.shape == eta_grid.shape:
                antinu_chunk = antinu_t[start:start + chunk_eta]
            elif antinu_t.ndim >= 1 and antinu_t.shape[-1] == eta_grid.numel():
                antinu_chunk = antinu_t[..., start:start + chunk_eta]
            else:
                antinu_chunk = antinu_t

        if method == "analytical":
            P_chunk = pearth(
                nustate=nustate,
                density=density,
                pmns=pmns,
                dm21_eV2=dm21_eV2,
                dm3l_eV2=dm3l_eV2,
                E_MeV=E_grid,
                eta=eta_grid_chunk,
                depth_m=depth_m,
                method="analytical",
                antinu=antinu_chunk,
                massbasis=massbasis,
                reunitarize=reunitarize,
            )
        else:
            if torch.is_tensor(antinu_chunk) and antinu_chunk.numel() != 1:
                raise ValueError("Numerical earth integration only supports scalar antinu.")

            P_rows = []

            for i_energy in range(n_energy):
                P_eta = []

                for eta_value in eta_chunk:
                    P_eta.append(
                        pearth(
                            nustate=nustate,
                            density=density,
                            pmns=pmns,
                            dm21_eV2=dm21_eV2,
                            dm3l_eV2=dm3l_eV2,
                            E_MeV=E[i_energy],
                            eta=eta_value,
                            depth_m=depth_m,
                            method="numerical",
                            antinu=bool(antinu_chunk.item()) if torch.is_tensor(antinu_chunk) else antinu_chunk,
                            massbasis=massbasis,
                            full_oscillation=False,
                            nsteps=nsteps,
                            ode_method=ode_method,
                            device=dev,
                            dtype=dtype,
                        )
                    )

                P_rows.append(torch.stack(P_eta, dim=0))

            P_chunk = torch.stack(P_rows, dim=0)

        out = out + torch.sum(
            P_chunk * w_chunk[None, :, None],
            dim=1,
        ) * deta

    if squeeze_E:
        return out[0]

    return out
