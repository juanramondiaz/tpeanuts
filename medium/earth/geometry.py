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
Earth geometry utilities.

This module contains geometry-only helper functions for neutrino propagation
through the Earth.

The functions in this file do not compute Hamiltonians, evolution operators,
or probabilities. They only convert detector depth and trajectory angle into
dimensionless path coordinates used by the Earth propagation schemes. Case A
denotes trajectories crossing Earth shells from below the horizon, while case B
denotes shallow detector-near trajectories between the surface and detector.

Module functions:
    detector_radius_fraction(...)
        Convert detector depth into a dimensionless detector radius.
    eta_prime_from_eta(...)
        Map detector nadir angle to the equivalent surface-crossing angle.
    detector_x_coordinate(...)
        Compute the detector coordinate along the trajectory chord.
    chord_length_case_b(...)
        Compute the shallow case-B path length inside Earth.
    build_earth_trajectory(...)
        Build a scalar sampled Earth trajectory for numerical propagation.
    classify_eta_regions(...)
        Split nadir angles into above-horizon, case-A, and case-B masks.
    validate_eta_range(...)
        Validate that nadir angles lie in the physical interval [0, pi].
"""

from __future__ import annotations

import torch
from tpeanuts.medium.atmosphere.geometry import (
    atmosphere_path_grid,
    atmosphere_path_length,
)
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import as_tensor
from tpeanuts.core.numerical.geometry import (
    OdeMethod,
    Trajectory,
    segment_sample_points,
)


def build_atmosphere_trajectories(
    production: dict[str, object],
    *,
    detector_depth_m: float = 0.0,
    trajectory_steps: int = 200,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> dict[str, torch.Tensor]:
    """Build atmosphere-path diagnostics in the detector geometry frame."""
    h = as_tensor(
        production["h_grid_km"], device=context.device, dtype=context.dtype
    )
    theta = as_tensor(
        production["theta_deg"], device=context.device, dtype=context.dtype
    )
    depth_km = torch.as_tensor(
        detector_depth_m / 1.0e3,
        device=context.device,
        dtype=context.dtype,
    )
    length = atmosphere_path_length(
        h_km=h,
        theta_deg=theta,
        depth_km=depth_km,
        device=context.device,
        dtype=context.dtype,
    )
    path, altitude = atmosphere_path_grid(
        h_km=h,
        theta_deg=theta,
        depth_km=depth_km,
        n_steps=trajectory_steps,
        device=context.device,
        dtype=context.dtype,
    )
    return {
        "L_atm_km": length,
        "s_atm_grid_km": path,
        "h_path_grid_km": altitude,
    }


def detector_radius_fraction(
    depth_m: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert detector depth to a dimensionless Earth-radius fraction.

    Args:
        depth_m: Detector depth below the Earth surface in metres.
        device: Device for the returned tensor.
        dtype: Real dtype for the returned tensor.

    Returns:
        Scalar tensor ``r_d = 1 - depth_m / R_E``.
    """
    h = float(depth_m) / float(R_E)
    r_d = 1.0 - h

    return torch.tensor(r_d, device=device, dtype=dtype)


def eta_prime_from_eta(
    eta: torch.Tensor,
    r_d: torch.Tensor,
) -> torch.Tensor:
    """Convert detector nadir angle to the surface-equivalent angle.

    Args:
        eta: Detector nadir angle in radians.
        r_d: Detector radius fraction returned by
            ``detector_radius_fraction``.

    Returns:
        Surface-equivalent nadir angle ``eta_prime`` in radians.
    """
    return torch.asin(r_d * torch.sin(eta))


def detector_x_coordinate(
    eta: torch.Tensor,
    r_d: torch.Tensor,
) -> torch.Tensor:
    """Compute the detector coordinate along the trajectory chord.

    Args:
        eta: Detector nadir angle in radians.
        r_d: Detector radius fraction.

    Returns:
        Dimensionless chord coordinate ``x_d = r_d cos(eta)``.
    """
    return r_d * torch.cos(eta)


def chord_length_case_b(
    eta: torch.Tensor,
    r_d: torch.Tensor,
) -> torch.Tensor:
    """Compute the matter path length for shallow case-B trajectories.

    Args:
        eta: Detector nadir angle in radians.
        r_d: Detector radius fraction.

    Returns:
        Dimensionless distance from the Earth surface to the detector along the
        trajectory.
    """
    return r_d * torch.cos(eta) + torch.sqrt(
        1.0 - r_d**2 * torch.sin(eta) ** 2
    )

def build_earth_trajectory(
    profile_earth,
    eta,
    depth_m: float,
    nsteps: int,
    *,
    method: OdeMethod | None,
    device,
    dtype,
    evolution_scale_m,
):
    """Build a sampled scalar trajectory for numerical Earth propagation.

    Args:
        profile_earth: Earth profile_earth object exposing ``shells_x``.
        eta: Detector nadir angle in radians. Only scalar trajectories are
            supported by this helper.
        depth_m: Detector depth below the Earth surface in metres.
        nsteps: Number of numerical trajectory segments.
        method: Segment sampling rule passed to ``segment_sample_points``.
        device: Device for trajectory tensors.
        dtype: Real dtype for trajectory tensors.
        evolution_scale_m: Positive scale in metres used to convert Earth
            radius units into dimensionless evolution lengths.

    Returns:
        ``Trajectory`` with Earth-radius coordinates ``x``, segment samples,
        dimensionless ``dx_evolution``, and metadata containing the trajectory
        mode, ``eta_prime``, and detector radius.
    """
    if nsteps < 1:
        raise ValueError("nsteps must be at least 1.")

    eta = as_tensor(eta, device=device, dtype=dtype)
    if eta.numel() != 1:
        raise ValueError("build_earth_trajectory only supports scalar eta.")

    evolution_scale = as_tensor(
            evolution_scale_m,
            device=device,
            dtype=dtype,
        )
    if torch.any(evolution_scale <= 0):
        raise ValueError("evolution_scale_m must be positive.")

    r_d = detector_radius_fraction(depth_m, device=device, dtype=dtype)
    x_d = detector_x_coordinate(eta, r_d)
    delta_x = chord_length_case_b(eta, r_d)
    eta_prime = eta_prime_from_eta(eta, r_d)

    eta_float = float(eta.detach().cpu())
    eta_prime_float = float(eta_prime.detach().cpu())

    if 0.0 <= eta_float < torch.pi / 2.0:
        xj_all, crossed, _ = profile_earth.shells_x(eta_prime)
        xj_crossed = torch.where(crossed, xj_all, torch.zeros_like(xj_all))

        x1 = -float(torch.max(xj_crossed).detach().cpu())
        x2 = float(x_d.detach().cpu())
        mode = "earth_crossing"

    else:
        x1 = 0.0
        x2 = float(delta_x.detach().cpu())
        mode = "local_constant"

    x = torch.linspace(
        x1,
        x2,
        nsteps + 1,
        device=device,
        dtype=dtype,
    )
  
    
    dx_evolution = (x[1:] - x[:-1]) * (R_E / evolution_scale)

    sample_x = segment_sample_points(x, method)

    return Trajectory(
        x=x,
        dx_evolution=dx_evolution,
        sample_x=sample_x,
        meta={
            "kind": "earth",
            "mode": mode,
            "eta": eta,
            "eta_prime": eta_prime,
            "eta_prime_float": eta_prime_float,
            "r_d": r_d,
        },
    )

def classify_eta_regions(
    eta: torch.Tensor,
    depth_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify nadir angles into Earth propagation regions.

    Args:
        eta: Detector nadir angles in radians.
        depth_m: Detector depth below the Earth surface in metres.

    Returns:
        Tuple ``(above, mask_a, mask_b)``. ``above`` marks trajectories outside
        Earth matter, ``mask_a`` shell-crossing trajectories, and ``mask_b``
        shallow detector-near trajectories.
    """
    if depth_m == 0.0:
        above = (eta >= torch.pi / 2.0) & (eta <= torch.pi)
    else:
        above = torch.zeros_like(eta, dtype=torch.bool)

    mask_a = (~above) & (eta >= 0.0) & (eta < torch.pi / 2.0)
    mask_b = (~above) & (eta >= torch.pi / 2.0) & (eta <= torch.pi)

    return above, mask_a, mask_b


def validate_eta_range(
    eta: torch.Tensor,
) -> None:
    """Validate detector nadir angles.

    Args:
        eta: Tensor of nadir angles in radians.

    Raises:
        ValueError: If any angle lies outside the interval [0, pi].
    """
    bad = (eta < 0.0) | (eta > torch.pi)

    if bad.any():
        raise ValueError("eta must be between 0 and pi.")
