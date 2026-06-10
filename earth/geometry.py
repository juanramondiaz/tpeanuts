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
earth geometry utilities for peanuts-torch.

This module contains geometry-only helper functions for neutrino propagation
through the earth.

The functions in this file do not compute Hamiltonians, evolution operators,
or probabilities. They only convert detector depth and trajectory angle into
dimensionless path coordinates used by the peanuts earth propagation scheme.
"""



from __future__ import annotations

import torch
import tpeanuts.util.constant as constant


def detector_radius_fraction(
    depth_m: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h = float(depth_m) / float(constant.R_E)
    r_d = 1.0 - h

    return torch.tensor(r_d, device=device, dtype=dtype)


def eta_prime_from_eta(
    eta: torch.Tensor,
    r_d: torch.Tensor,
) -> torch.Tensor:
    return torch.asin(r_d * torch.sin(eta))


def detector_x_coordinate(
    eta: torch.Tensor,
    r_d: torch.Tensor,
) -> torch.Tensor:
    return r_d * torch.cos(eta)


def chord_length_case_b(
    eta: torch.Tensor,
    r_d: torch.Tensor,
) -> torch.Tensor:
    return r_d * torch.cos(eta) + torch.sqrt(
        1.0 - r_d**2 * torch.sin(eta) ** 2
    )


def classify_eta_regions(
    eta: torch.Tensor,
    depth_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    bad = (eta < 0.0) | (eta > torch.pi)

    if bad.any():
        raise ValueError("eta must be between 0 and pi.")