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
Diagnostics and validation utilities for atmospheric height-flux datasets.

This module provides consistency checks for:

    - normalization
    - NaN / Inf detection
    - monotonicity
    - tensor shapes
    - positivity
    - finite values
    - profile reconstruction quality
"""



from __future__ import annotations

from typing import Dict, Optional, Union

import torch

from tpeanuts.util.type import _as_tensor
from tpeanuts.util.torch_util import _default_device


TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Generic tensor checks
# ============================================================

def tensor_summary(
    x: TensorLike,
    *,
    name: str = "tensor",
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict:
    dev = _default_device(device)

    x_t = _as_tensor(
        x,
        device=dev,
        dtype=dtype,
    )

    return {
        "name": name,
        "shape": tuple(x_t.shape),
        "dtype": str(x_t.dtype),
        "device": str(x_t.device),
        "min": float(torch.min(x_t).item()),
        "max": float(torch.max(x_t).item()),
        "mean": float(torch.mean(x_t).item()),
        "std": float(torch.std(x_t).item()),
        "has_nan": bool(torch.isnan(x_t).any().item()),
        "has_inf": bool(torch.isinf(x_t).any().item()),
    }


def check_no_nan_inf(
    x: TensorLike,
    *,
    name: str = "tensor",
    raise_error: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> bool:
    dev = _default_device(device)

    x_t = _as_tensor(
        x,
        device=dev,
        dtype=dtype,
    )

    valid = (
        not torch.isnan(x_t).any()
        and not torch.isinf(x_t).any()
    )

    if (not valid) and raise_error:
        raise ValueError(
            f"{name} contains NaN or Inf values."
        )

    return valid


def check_monotonic_increasing(
    x: TensorLike,
    *,
    strict: bool = True,
    raise_error: bool = False,
    name: str = "tensor",
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> bool:
    dev = _default_device(device)

    x_t = _as_tensor(
        x,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    dx = torch.diff(x_t)

    if strict:
        valid = bool(torch.all(dx > 0.0).item())
    else:
        valid = bool(torch.all(dx >= 0.0).item())

    if (not valid) and raise_error:
        raise ValueError(
            f"{name} is not monotonic increasing."
        )

    return valid


def check_positive(
    x: TensorLike,
    *,
    allow_zero: bool = True,
    raise_error: bool = False,
    name: str = "tensor",
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> bool:
    dev = _default_device(device)

    x_t = _as_tensor(
        x,
        device=dev,
        dtype=dtype,
    )

    if allow_zero:
        valid = bool(torch.all(x_t >= 0.0).item())
    else:
        valid = bool(torch.all(x_t > 0.0).item())

    if (not valid) and raise_error:
        raise ValueError(
            f"{name} contains negative values."
        )

    return valid


# ============================================================
# Profile normalization
# ============================================================

@torch.no_grad()
def compute_profile_normalization(
    h_grid_km: TensorLike,
    f_Eh: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    dev = _default_device(device)

    h_t = _as_tensor(
        h_grid_km,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    f_t = _as_tensor(
        f_Eh,
        device=dev,
        dtype=dtype,
    )

    if f_t.ndim != 2:
        raise ValueError("f_Eh must have shape (n_E, n_h).")

    if f_t.shape[1] != h_t.numel():
        raise ValueError(
            "f_Eh.shape[1] must match len(h_grid_km)."
        )

    return torch.trapezoid(
        f_t,
        x=h_t,
        dim=1,
    )


@torch.no_grad()
def check_profile_normalization(
    h_grid_km: TensorLike,
    f_Eh: TensorLike,
    *,
    atol: float = 1.0e-3,
    rtol: float = 1.0e-3,
    raise_error: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
):
    norm_E = compute_profile_normalization(
        h_grid_km=h_grid_km,
        f_Eh=f_Eh,
        device=device,
        dtype=dtype,
    )

    target = torch.ones_like(norm_E)

    valid = bool(
        torch.allclose(
            norm_E,
            target,
            atol=atol,
            rtol=rtol,
        )
    )

    if (not valid) and raise_error:
        max_dev = torch.max(torch.abs(norm_E - 1.0))

        raise ValueError(
            f"Profile normalization failed. "
            f"Maximum deviation = {float(max_dev.item())}"
        )

    return {
        "valid": valid,
        "norm_E": norm_E,
        "max_abs_deviation": float(
            torch.max(torch.abs(norm_E - 1.0)).item()
        ),
    }


# ============================================================
# flux reconstruction checks
# ============================================================

@torch.no_grad()
def reconstruct_flux_from_profile(
    h_grid_km: TensorLike,
    phi_Eh: TensorLike,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
):
    dev = _default_device(device)

    h_t = _as_tensor(
        h_grid_km,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    phi_t = _as_tensor(
        phi_Eh,
        device=dev,
        dtype=dtype,
    )

    return torch.trapezoid(
        phi_t,
        x=h_t,
        dim=1,
    )


@torch.no_grad()
def check_flux_reconstruction(
    h_grid_km: TensorLike,
    phi_Eh: TensorLike,
    phi_E_obs: TensorLike,
    *,
    atol: float = 1.0e-3,
    rtol: float = 1.0e-3,
    raise_error: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
):
    dev = _default_device(device)

    phi_obs_t = _as_tensor(
        phi_E_obs,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    phi_rec = reconstruct_flux_from_profile(
        h_grid_km=h_grid_km,
        phi_Eh=phi_Eh,
        device=dev,
        dtype=dtype,
    )

    valid = bool(
        torch.allclose(
            phi_rec,
            phi_obs_t,
            atol=atol,
            rtol=rtol,
        )
    )

    residual = phi_rec - phi_obs_t

    max_abs = float(torch.max(torch.abs(residual)).item())

    denom = torch.clamp(
        torch.abs(phi_obs_t),
        min=1.0e-30,
    )

    max_rel = float(
        torch.max(torch.abs(residual) / denom).item()
    )

    if (not valid) and raise_error:
        raise ValueError(
            "flux reconstruction check failed. "
            f"max_rel_error={max_rel:.3e}"
        )

    return {
        "valid": valid,
        "phi_reconstructed": phi_rec,
        "residual": residual,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
    }


# ============================================================
# Result-level diagnostics
# ============================================================

@torch.no_grad()
def diagnose_result(
    result: Dict,
    *,
    profile_atol: float = 1.0e-3,
    profile_rtol: float = 1.0e-3,
    reconstruction_atol: float = 1.0e-3,
    reconstruction_rtol: float = 1.0e-3,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
):
    diagnostics = {}

    # --------------------------------------------------------
    # Basic tensor diagnostics
    # --------------------------------------------------------
    tensor_keys = [
        "E_grid_GeV",
        "X_grid_gcm2",
        "h_grid_km",
        "flux_XE",
        "flux_smooth_XE",
        "dPhi_dX_XE",
        "f_Eh",
        "phi_E_obs",
        "phi_Eh",
    ]

    diagnostics["tensor_summaries"] = {}

    for key in tensor_keys:

        if key in result:

            diagnostics["tensor_summaries"][key] = (
                tensor_summary(
                    result[key],
                    name=key,
                    device=device,
                    dtype=dtype,
                )
            )

    # --------------------------------------------------------
    # Grid checks
    # --------------------------------------------------------
    diagnostics["grid_checks"] = {
        "E_grid_monotonic": (
            check_monotonic_increasing(
                result["E_grid_GeV"],
                name="E_grid_GeV",
                device=device,
                dtype=dtype,
            )
            if "E_grid_GeV" in result
            else None
        ),

        "X_grid_monotonic": (
            check_monotonic_increasing(
                result["X_grid_gcm2"],
                name="X_grid_gcm2",
                device=device,
                dtype=dtype,
            )
            if "X_grid_gcm2" in result
            else None
        ),

        "h_grid_monotonic": (
            check_monotonic_increasing(
                result["h_grid_km"],
                name="h_grid_km",
                device=device,
                dtype=dtype,
            )
            if "h_grid_km" in result
            else None
        ),
    }

    # --------------------------------------------------------
    # Positivity
    # --------------------------------------------------------
    diagnostics["positivity_checks"] = {}

    for key in [
        "flux_XE",
        "flux_smooth_XE",
        "f_Eh",
        "phi_Eh",
    ]:

        if key in result:

            diagnostics["positivity_checks"][key] = (
                check_positive(
                    result[key],
                    allow_zero=True,
                    name=key,
                    device=device,
                    dtype=dtype,
                )
            )

    # --------------------------------------------------------
    # NaN / Inf checks
    # --------------------------------------------------------
    diagnostics["nan_inf_checks"] = {}

    for key in tensor_keys:

        if key in result:

            diagnostics["nan_inf_checks"][key] = (
                check_no_nan_inf(
                    result[key],
                    name=key,
                    device=device,
                    dtype=dtype,
                )
            )

    # --------------------------------------------------------
    # Profile normalization
    # --------------------------------------------------------
    if (
        "h_grid_km" in result
        and "f_Eh" in result
    ):

        diagnostics["profile_normalization"] = (
            check_profile_normalization(
                h_grid_km=result["h_grid_km"],
                f_Eh=result["f_Eh"],
                atol=profile_atol,
                rtol=profile_rtol,
                device=device,
                dtype=dtype,
            )
        )

    # --------------------------------------------------------
    # flux reconstruction
    # --------------------------------------------------------
    if (
        "h_grid_km" in result
        and "phi_Eh" in result
        and "phi_E_obs" in result
    ):

        diagnostics["flux_reconstruction"] = (
            check_flux_reconstruction(
                h_grid_km=result["h_grid_km"],
                phi_Eh=result["phi_Eh"],
                phi_E_obs=result["phi_E_obs"],
                atol=reconstruction_atol,
                rtol=reconstruction_rtol,
                device=device,
                dtype=dtype,
            )
        )

    return diagnostics