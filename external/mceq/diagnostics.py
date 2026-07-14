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
Diagnostics and validation utilities for Atmosphere height-flux datasets.

This module provides consistency checks for:

    - normalization
    - NaN / Inf detection
    - monotonicity
    - tensor shapes
    - positivity
    - finite values
    - profile reconstruction quality

All checks here are tpeanuts-native tensor arithmetic; nothing in this
module calls MCEq directly. It instead operates on the torch tensors
produced upstream by tpeanuts.external.mceq.profiles (height-dependent
production profile f(h | E, alpha) and flux Phi(E, h, alpha) derived
from an MCEq cascade-equation solve), and is used to sanity-check that
those derived quantities are physically well-behaved: that the
production-height profile f(h | E, alpha) integrates to 1 over altitude
for each energy (it is a normalized probability density in h), and that
integrating the height-differential flux Phi(E, h, alpha) back over h
reproduces the energy-differential flux Phi(E, X_obs, alpha) originally
extracted from MCEq at the observation depth.

Module functions:
    tensor_summary:
        Compute basic descriptive statistics (shape, dtype, device,
        min/max/mean/std, NaN/Inf flags) for a tensor.
    check_no_nan_inf:
        Check that a tensor contains no NaN or Inf values.
    check_monotonic_increasing:
        Check that a 1-D tensor is (strictly or weakly) increasing.
    check_positive:
        Check that a tensor has no negative values.
    compute_profile_normalization:
        Integrate a height-differential profile f(h | E, alpha) over
        altitude h for each energy, which should equal 1 if f is a
        properly normalized probability density.
    check_profile_normalization:
        Check that compute_profile_normalization is close to 1 within
        tolerance for every energy.
    reconstruct_flux_from_profile:
        Integrate a height-differential flux Phi(E, h, alpha) over
        altitude h to reconstruct the energy-differential flux at the
        observation depth.
    check_flux_reconstruction:
        Check that reconstruct_flux_from_profile agrees with the
        originally extracted observation-depth flux within tolerance.
    diagnose_result:
        Run the full battery of checks above on a height-flux result
        dictionary (as produced by
        tpeanuts.external.mceq.profiles.production_profiles_all_energies_from_flux_gradient)
        and return a structured diagnostics report.
"""



from __future__ import annotations

from typing import Dict, Optional, Union

import torch

from tpeanuts.util.type import as_tensor
from tpeanuts.util.torch_util import default_device
from tpeanuts.util.test_utils import (
    check_monotonic_increasing,
    check_no_nan_inf,
    check_positive,
    tensor_summary,
)


TensorLike = Union[float, int, torch.Tensor]


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
    """
    Integrate a height-differential production profile over altitude
    for each energy.

    f(h | E, alpha) is the conditional probability density of particle
    production at altitude h given energy E and zenith angle theta (see
    tpeanuts.external.mceq.profiles); by construction it should
    integrate to 1 over h for every energy. This function computes that
    integral via the trapezoidal rule so callers can verify the
    normalization (see check_profile_normalization).

    Args:
        h_grid_km: Strictly increasing 1-D altitude grid in kilometres
            on which f_Eh is sampled.
        f_Eh: Production-height profile tensor of shape (n_E, n_h),
            dimensionless (probability density in h, units of 1/km).
        device: Working torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation.

    Returns:
        1-D tensor of length n_E with the integral of f_Eh over h_grid_km
        for each energy; should be close to 1 for a properly normalized
        profile.

    Raises:
        ValueError: If f_Eh does not have shape (n_E, n_h) matching
            h_grid_km.
    """
    dev = default_device(device)

    h_t = as_tensor(
        h_grid_km,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    f_t = as_tensor(
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
    """
    Check that the production-height profile f(h | E, alpha) integrates
    to 1 over altitude for every energy, within tolerance.

    Args:
        h_grid_km: Strictly increasing 1-D altitude grid in kilometres.
        f_Eh: Production-height profile tensor of shape (n_E, n_h).
        atol: Absolute tolerance passed to torch.allclose when comparing
            the per-energy integral to 1.
        rtol: Relative tolerance passed to torch.allclose.
        raise_error: If True, raise ValueError when the check fails
            instead of just reporting it in the returned dict.
        device: Working torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation.

    Returns:
        Dict with keys "valid" (bool), "norm_E" (per-energy integral
        tensor) and "max_abs_deviation" (float, largest |integral - 1|
        across energies).

    Raises:
        ValueError: If raise_error is True and the normalization check
            fails.
    """
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
    """
    Integrate a height-differential flux Phi(E, h, alpha) over altitude
    to reconstruct the energy-differential flux at the observation
    depth.

    Since Phi(E, h, alpha) = Phi(E, X_obs, alpha) * f(h | E, alpha) and
    f integrates to 1 over h, integrating Phi(E, h, alpha) over h_grid_km
    should reproduce the original observation-depth flux Phi(E, X_obs,
    theta) extracted from the MCEq solve. This function performs that
    integration via the trapezoidal rule (see check_flux_reconstruction
    for the corresponding consistency check).

    Args:
        h_grid_km: Strictly increasing 1-D altitude grid in kilometres
            on which phi_Eh is sampled.
        phi_Eh: Height-differential flux tensor of shape (n_E, n_h), in
            units of (cm^2 s sr GeV km)^-1.
        device: Working torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation.

    Returns:
        1-D tensor of length n_E with the reconstructed
        energy-differential flux Phi(E, X_obs, alpha), in units of
        (cm^2 s sr GeV)^-1.
    """
    dev = default_device(device)

    h_t = as_tensor(
        h_grid_km,
        device=dev,
        dtype=dtype,
    ).reshape(-1)

    phi_t = as_tensor(
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
    """
    Check that integrating the height-differential flux Phi(E, h,
    theta) over altitude reproduces the energy-differential flux
    Phi(E, X_obs, alpha) originally extracted from the MCEq solve.

    Args:
        h_grid_km: Strictly increasing 1-D altitude grid in kilometres.
        phi_Eh: Height-differential flux tensor of shape (n_E, n_h), in
            units of (cm^2 s sr GeV km)^-1.
        phi_E_obs: Energy-differential flux at the observation depth, in
            units of (cm^2 s sr GeV)^-1, as originally interpolated from
            the MCEq solution (e.g. via
            tpeanuts.medium.atmosphere.depth.interpolate_flux_at_Xobs).
        atol: Absolute tolerance passed to torch.allclose.
        rtol: Relative tolerance passed to torch.allclose.
        raise_error: If True, raise ValueError when the check fails
            instead of just reporting it in the returned dict.
        device: Working torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computation.

    Returns:
        Dict with keys "valid" (bool), "phi_reconstructed" (tensor),
        "residual" (phi_reconstructed - phi_E_obs), "max_abs_error" and
        "max_rel_error" (floats).

    Raises:
        ValueError: If raise_error is True and the reconstruction check
            fails.
    """
    dev = default_device(device)

    phi_obs_t = as_tensor(
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
    """
    Run the full battery of diagnostic checks on an Atmosphere
    height-flux result dictionary.

    Accepts the dict produced by
    tpeanuts.external.mceq.profiles.production_profiles_all_energies_from_flux_gradient
    (or compatible results, e.g. from the Honda backend) and reports,
    for whichever of the recognised keys are present: tensor summaries,
    grid monotonicity, positivity, NaN/Inf detection, production-profile
    normalization, and flux-reconstruction accuracy. Recognised tensor
    keys include "E_grid_GeV" (energy grid, GeV), "X_grid_gcm2"
    (Atmosphere slant-depth grid, g/cm^2), "h_grid_km" (altitude grid,
    km), "flux_XE"/"flux_smooth_XE"/"dPhi_dX_XE" (MCEq flux and its
    smoothed depth-derivative versus depth and energy), "f_Eh"
    (normalized production-height profile), "phi_E_obs"
    (energy-differential flux at the observation depth) and "phi_Eh"
    (height-differential flux).

    Args:
        result: Height-flux result dictionary; only keys that are
            present are checked, all checks are skipped gracefully for
            missing keys.
        profile_atol: Absolute tolerance for the profile-normalization
            check (passed to check_profile_normalization).
        profile_rtol: Relative tolerance for the profile-normalization
            check.
        reconstruction_atol: Absolute tolerance for the
            flux-reconstruction check (passed to
            check_flux_reconstruction).
        reconstruction_rtol: Relative tolerance for the
            flux-reconstruction check.
        device: Working torch device. None selects CUDA when available,
            else CPU.
        dtype: Floating dtype used for the computations.

    Returns:
        Dict with keys "tensor_summaries", "grid_checks",
        "positivity_checks", "nan_inf_checks", and, when the relevant
        input keys are present, "profile_normalization" and
        "flux_reconstruction", each holding the corresponding
        per-quantity diagnostic results.
    """
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
