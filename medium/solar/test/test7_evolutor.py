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

"""Pytest-compatible tests for medium.solar.evolutor (numerical propagation)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.solar.evolutor import (
    build_solar_trajectory,
    mass_weights_numerical,
    solar_evolutor_numerical,
    solar_evolutor_numerical_history,
)
from tpeanuts.medium.solar.probability import Tei
from tpeanuts.medium.solar.profile import build_solar_profile
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(*, context: RuntimeContext | None = None) -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO",
        context=context or make_context(),
    )


def make_sterile_oscillation(*, context: RuntimeContext | None = None) -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset(
        "sterile_3p1_bestfit_giunti2017",
        context=context or make_context(),
    )


def make_profile():
    return build_solar_profile(None, context=make_context())


def test_build_solar_trajectory_covers_full_density_range_and_matches_production_points():
    profile = make_profile()

    trajectory = build_solar_trajectory(profile)

    # Every boundary point is unique and sorted.
    assert bool(torch.all(torch.diff(trajectory.x) > 0.0))
    # The merged grid spans (at least) the full density table's own range,
    # not just the narrower production-point range.
    assert float(trajectory.x[0]) <= float(profile.radius[0])
    assert float(trajectory.x[-1]) >= float(profile.radius[-1])
    # Some providers tabulate production only in the core (strictly shorter
    # than the density grid); SF-III tabulates both products through r=1.
    assert float(trajectory.x[-1]) >= float(profile.production_radius[-1])
    # Every production point is an *exact* boundary of the merged grid.
    production_index = trajectory.meta["production_index"]
    torch.testing.assert_close(
        trajectory.x[production_index], profile.production_radius, rtol=0.0, atol=0.0,
    )


def test_build_solar_trajectory_uses_full_radius_and_production_radius():
    profile = make_profile()
    trajectory = build_solar_trajectory(profile)
    assert trajectory.x[0] == profile.radius[0]
    assert trajectory.x[-1] == profile.radius[-1]


def test_solar_evolutor_numerical_history_is_unitary():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    S_history, trajectory = solar_evolutor_numerical_history(oscillation, energy, profile)

    assert S_history.shape[0] == 3
    assert S_history.shape[-2:] == (3, 3)
    assert S_history.shape[-3] == trajectory.x.numel()

    identity = torch.eye(3, device=DEVICE, dtype=torch.complex128)
    unitarity_error = (S_history.conj().transpose(-1, -2) @ S_history - identity).abs().max()
    assert float(unitarity_error) < 1.0e-8

    # S(r_0, r_0) is the identity by construction (zero-length propagation).
    torch.testing.assert_close(
        S_history[:, 0], identity.expand(3, 3, 3), rtol=0.0, atol=1.0e-10,
    )


def test_solar_evolutor_numerical_shape_and_unitarity():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    S = solar_evolutor_numerical(oscillation, energy, profile)

    assert S.shape == (3, profile.production_radius.numel(), 3, 3)
    identity = torch.eye(3, device=DEVICE, dtype=torch.complex128)
    unitarity_error = (S.conj().transpose(-1, -2) @ S - identity).abs().max()
    assert float(unitarity_error) < 1.0e-8


def test_solar_evolutor_numerical_sterile_shape_and_unitarity():
    oscillation = make_sterile_oscillation()
    profile = make_profile()
    energy = torch.tensor([5.0], device=DEVICE, dtype=DTYPE)

    S = solar_evolutor_numerical(oscillation, energy, profile)

    assert S.shape == (1, profile.production_radius.numel(), 4, 4)
    identity = torch.eye(4, device=DEVICE, dtype=torch.complex128)
    unitarity_error = (S.conj().transpose(-1, -2) @ S - identity).abs().max()
    assert float(unitarity_error) < 1.0e-6


def test_solar_evolutor_numerical_include_matter_nc_requires_density_n():
    oscillation = make_sterile_oscillation()
    profile = make_profile()
    profile.density_n = None
    energy = torch.tensor([5.0], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="density_n"):
        solar_evolutor_numerical(oscillation, energy, profile, include_matter_nc=True)


def test_mass_weights_numerical_shape_and_normalization():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    weights = mass_weights_numerical(oscillation, energy, profile)

    assert weights.shape == (3, profile.production_radius.numel(), 3)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones_like(weights[..., 0]), rtol=1.0e-6, atol=1.0e-6,
    )


def test_mass_weights_numerical_matches_tei_at_production_points_in_sm_limit():
    # Cross-validation against the independently-implemented adiabatic path:
    # agreement at the several-percent level is expected for standard LMA
    # parameters, where the adiabatic approximation itself is excellent but
    # not exact, and the numerical path adds its own trajectory-discretization
    # error on top.
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    weights_numerical = mass_weights_numerical(oscillation, energy, profile)
    weights_tei = Tei(
        oscillation,
        energy[:, None],
        profile.electron_density(profile.production_radius)[None, :],
    )

    assert weights_numerical.shape == weights_tei.shape
    torch.testing.assert_close(weights_numerical, weights_tei, rtol=0.0, atol=7.0e-2)
