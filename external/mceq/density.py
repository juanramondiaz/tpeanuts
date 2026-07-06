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
MCEq atmosphere-density utilities.

This module bridges MCEq density-model objects with the tensor-based
TPeanuts code. MCEq itself is CPU/Python based, so calls to

    density_model.get_density(...)

and related MCEq density-model methods are not GPU operations. Tensor inputs
are moved to CPU for the MCEq call and converted back to the requested device.

Main functions:
    get_density_model:
        Return the density_model object attached to an initialized MCEq run.
    get_mass_density_gcm3_from_mceq:
        Evaluate the MCEq mass density at one altitude in km.
    get_mass_overburden_gcm2_from_mceq:
        Evaluate the MCEq mass overburden at one altitude in km.
    atmosphere_mass_density_profile_from_mceq:
        Build a torch tensor with the atmosphere mass-density profile.
    atmosphere_mass_overburden_profile_from_mceq:
        Build a torch tensor with the atmosphere overburden profile.
    save_mceq_density_profile:
        Save a two-column altitude-density table extracted from MCEq.
"""



from __future__ import annotations

import os
from typing import Optional, Union

import torch

from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import default_device

from tpeanuts.external.mceq.config import (
    MCEqModelConfig,
)

from tpeanuts.external.mceq.core import init_mceq


TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Low-level accessors
# ============================================================

def get_density_model(mceq):
    """
    Return the density model attached to an initialized MCEq object.

    Args:
        mceq: Initialized MCEq instance expected to expose a density_model
            attribute.

    Returns:
        The underlying MCEq density-model object.

    Raises:
        AttributeError: If the provided object does not expose density_model.
    """
    if not hasattr(mceq, "density_model"):
        raise AttributeError("The provided mceq object has no density_model attribute.")

    return mceq.density_model


def get_mass_density_gcm3_from_mceq(
    mceq,
    h_km: float,
) -> float:
    """
    Evaluate MCEq atmosphere mass density at a single altitude.

    Args:
        mceq: Initialized MCEq object whose density_model provides
            get_density with altitude in cm.
        h_km: Altitude above sea level in kilometres.

    Returns:
        Mass density in g/cm^3 as a Python float.
    """
    density_model = get_density_model(mceq)

    h_cm = float(h_km) * 1.0e5

    return float(density_model.get_density(h_cm))


def get_mass_overburden_gcm2_from_mceq(
    mceq,
    h_km: float,
) -> float:
    """
    Evaluate MCEq atmosphere mass overburden at a single altitude.

    Args:
        mceq: Initialized MCEq object whose density_model provides
            get_mass_overburden with altitude in cm.
        h_km: Altitude above sea level in kilometres.

    Returns:
        Atmospheric mass overburden in g/cm^2 as a Python float.

    Raises:
        AttributeError: If the selected MCEq density model does not implement
            get_mass_overburden.
    """
    density_model = get_density_model(mceq)

    if not hasattr(density_model, "get_mass_overburden"):
        raise AttributeError(
            "The mceq density model does not provide get_mass_overburden."
        )

    h_cm = float(h_km) * 1.0e5

    return float(density_model.get_mass_overburden(h_cm))


# ============================================================
# Torch-compatible profiles
# ============================================================

@torch.no_grad()
def atmosphere_mass_density_profile_from_mceq(
    h_km: TensorLike,
    mceq=None,
    theta_deg: TensorLike = 0.0,
    config: Optional[MCEqModelConfig] = None,
    interaction_model: Optional[str] = None,
    primary_model: Optional[Union[str, tuple]] = None,
    density_model: Optional[str] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build a tensor-valued atmosphere mass-density profile from MCEq.

    Args:
        h_km: Scalar or tensor-like altitude values in kilometres. The returned
            tensor preserves this shape.
        mceq: Optional initialized MCEq object. When omitted, a new instance is
            created with init_mceq and the supplied configuration arguments.
        theta_deg: Zenith angle passed to init_mceq when mceq is omitted.
        config: Optional MCEq model configuration object.
        interaction_model: Optional interaction-model override for init_mceq.
        primary_model: Optional primary cosmic-ray model override for init_mceq.
        density_model: Optional MCEq density-model override for init_mceq.
        device: Output torch device. None selects CUDA when available, else CPU.
        dtype: Floating dtype used for input conversion and output tensor.

    Returns:
        Tensor of mass density values in g/cm^3 with the same shape as h_km.
    """
    dev = default_device(device)

    h_t = as_tensor(h_km, device=dev, dtype=dtype)
    original_shape = h_t.shape

    if mceq is None:
        mceq = init_mceq(
            theta_deg=theta_deg,
            config=config,
            interaction_model=interaction_model,
            primary_model=primary_model,
            density_model=density_model,
            info=False,
        )

    h_flat_cpu = h_t.detach().cpu().reshape(-1)

    rho_vals = [
        get_mass_density_gcm3_from_mceq(
            mceq=mceq,
            h_km=float(h.item()),
        )
        for h in h_flat_cpu
    ]

    return torch.tensor(
        rho_vals,
        device=dev,
        dtype=dtype,
    ).reshape(original_shape)


@torch.no_grad()
def atmosphere_mass_overburden_profile_from_mceq(
    h_km: TensorLike,
    mceq=None,
    theta_deg: TensorLike = 0.0,
    config: Optional[MCEqModelConfig] = None,
    interaction_model: Optional[str] = None,
    primary_model: Optional[Union[str, tuple]] = None,
    density_model: Optional[str] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build a tensor-valued atmosphere overburden profile from MCEq.

    Args:
        h_km: Scalar or tensor-like altitude values in kilometres. The returned
            tensor preserves this shape.
        mceq: Optional initialized MCEq object. When omitted, a new instance is
            created with init_mceq and the supplied configuration arguments.
        theta_deg: Zenith angle passed to init_mceq when mceq is omitted.
        config: Optional MCEq model configuration object.
        interaction_model: Optional interaction-model override for init_mceq.
        primary_model: Optional primary cosmic-ray model override for init_mceq.
        density_model: Optional MCEq density-model override for init_mceq.
        device: Output torch device. None selects CUDA when available, else CPU.
        dtype: Floating dtype used for input conversion and output tensor.

    Returns:
        Tensor of atmosphere mass overburden values in g/cm^2 with the same
        shape as h_km.
    """
    dev = default_device(device)

    h_t = as_tensor(h_km, device=dev, dtype=dtype)
    original_shape = h_t.shape

    if mceq is None:
        mceq = init_mceq(
            theta_deg=theta_deg,
            config=config,
            interaction_model=interaction_model,
            primary_model=primary_model,
            density_model=density_model,
            info=False,
        )

    h_flat_cpu = h_t.detach().cpu().reshape(-1)

    X_vals = [
        get_mass_overburden_gcm2_from_mceq(
            mceq=mceq,
            h_km=float(h.item()),
        )
        for h in h_flat_cpu
    ]

    return torch.tensor(
        X_vals,
        device=dev,
        dtype=dtype,
    ).reshape(original_shape)


# ============================================================
# Save density profile
# ============================================================

@torch.no_grad()
def save_mceq_density_profile(
    output_path: str,
    mceq=None,
    theta_deg: TensorLike = 0.0,
    h_grid_km: Optional[TensorLike] = None,
    h_min_km: TensorLike = 0.0,
    h_max_km: TensorLike = 100.0,
    n_h: int = 1000,
    overwrite: bool = True,
    config: Optional[MCEqModelConfig] = None,
    interaction_model: Optional[str] = None,
    primary_model: Optional[Union[str, tuple]] = None,
    density_model: Optional[str] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> str:
    """
    Save an altitude-density profile extracted from MCEq to a text file.

    Args:
        output_path: Destination text file path.
        mceq: Optional initialized MCEq object. When omitted, a new instance is
            created with init_mceq and the supplied configuration arguments.
        theta_deg: Zenith angle passed to init_mceq when mceq is omitted.
        h_grid_km: Optional explicit altitude grid in kilometres.
        h_min_km: Minimum altitude in kilometres used when h_grid_km is omitted.
        h_max_km: Maximum altitude in kilometres used when h_grid_km is omitted.
        n_h: Number of altitude samples used when h_grid_km is omitted.
        overwrite: Whether an existing output file may be replaced.
        config: Optional MCEq model configuration object.
        interaction_model: Optional interaction-model override for init_mceq.
        primary_model: Optional primary cosmic-ray model override for init_mceq.
        density_model: Optional MCEq density-model override for init_mceq.
        device: Working/output torch device for tensor construction.
        dtype: Floating dtype used for altitude and density tensors.

    Returns:
        The output path as a string.

    Raises:
        FileExistsError: If output_path exists and overwrite is False.
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")

    output_dir = os.path.dirname(output_path)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    dev = default_device(device)

    if h_grid_km is None:
        h_min_t = as_tensor(h_min_km, device=dev, dtype=dtype)
        h_max_t = as_tensor(h_max_km, device=dev, dtype=dtype)

        h_grid_t = torch.linspace(
            float(h_min_t.item()),
            float(h_max_t.item()),
            int(n_h),
            device=dev,
            dtype=dtype,
        )

    else:
        h_grid_t = as_tensor(h_grid_km, device=dev, dtype=dtype)

    rho_t = atmosphere_mass_density_profile_from_mceq(
        h_km=h_grid_t,
        mceq=mceq,
        theta_deg=theta_deg,
        config=config,
        interaction_model=interaction_model,
        primary_model=primary_model,
        density_model=density_model,
        device=dev,
        dtype=dtype,
    )

    h_cpu = h_grid_t.detach().cpu().reshape(-1)
    rho_cpu = rho_t.detach().cpu().reshape(-1)

    header = (
        "# Atmospheric density profile extracted from mceq\n"
        "# Columns:\n"
        "# h_km    rho_gcm3\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)

        for h_val, rho_val in zip(h_cpu, rho_cpu):
            f.write(
                f"{h_val.item():.10e}    {rho_val.item():.10e}\n"
            )

    return output_path
