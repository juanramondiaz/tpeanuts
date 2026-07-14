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
Pytest-compatible tests for tpeanuts.medium.earth.profile.

The diagnostic plots from the historical backup tests live in notebooks; this
file keeps only fast numerical sanity checks that can run automatically.
"""

from __future__ import annotations

import math

import pytest
import torch

from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close, check_no_nan_inf, check_positive


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _profile(
    rj=None,
    coefficients=None,
    *,
    dtype: torch.dtype = DTYPE,
    device: torch.device = DEVICE,
    profile_perturbative_name: str = "even_power",
    profile_perturbative_kwargs: dict | None = None,
    **params_kwargs,
) -> EarthProfile:
    """Build an EarthProfile, optionally from explicit rj/coefficients."""
    kwargs = dict(profile_perturbative_kwargs or {})
    if rj is not None:
        kwargs["rj"] = rj
    if coefficients is not None:
        kwargs["coefficients"] = coefficients

    params = EarthParameters(
        profile_perturbative_name=profile_perturbative_name,
        profile_perturbative_kwargs=kwargs,
        **params_kwargs,
    )
    return EarthProfile(params=params, context=RuntimeContext.resolve(device, dtype))


def _two_shell_profile(**params_kwargs) -> EarthProfile:
    """Synthetic two-shell profile: constant density 2.0 for r<0.5, 1.0 for 0.5<r<=1.0."""
    rj = torch.tensor([0.5, 1.0], device=DEVICE, dtype=DTYPE)
    coefficients = torch.tensor(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    return _profile(rj, coefficients, **params_kwargs)


def _default_profile(**params_kwargs) -> EarthProfile:
    """Profile built from the bundled default even-power density CSV."""
    return EarthProfile(
        params=EarthParameters(**params_kwargs),
        context=RuntimeContext.resolve(DEVICE, DTYPE),
    )


def test_default_profile_loads_and_validates_rj():
    profile = _default_profile()

    assert profile.rj.ndim == 1
    assert profile.rj.numel() > 0
    assert check_no_nan_inf(profile.rj)
    assert torch.all(torch.diff(profile.rj) > 0)
    assert torch.all((profile.rj > 0) & (profile.rj <= 1))


def test_two_shell_synthetic_profile_matches_inputs():
    profile = _two_shell_profile()

    assert_close(profile.rj, torch.tensor([0.5, 1.0], dtype=DTYPE), name="synthetic rj")
    assert profile.device.type == DEVICE.type
    assert profile.dtype == DTYPE


def test_shells_x_nadir_all_shells_crossed():
    profile = _two_shell_profile()
    eta = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

    xj_all, crossed, _ = profile.shells_x(eta)

    assert_close(xj_all, profile.rj, name="nadir xj equals rj")
    assert torch.all(crossed)


def test_shells_x_grazing_incidence_no_shells_crossed():
    profile = _two_shell_profile()
    eta = torch.tensor(math.pi / 2, device=DEVICE, dtype=DTYPE)

    _, crossed, _ = profile.shells_x(eta)

    assert not torch.any(crossed)


def test_shells_x_oblique_partial_crossing():
    profile = _two_shell_profile()
    eta = torch.tensor(math.pi / 3, device=DEVICE, dtype=DTYPE)

    xj_all, crossed, _ = profile.shells_x(eta)

    s = math.sin(math.pi / 3)
    expected_xj = torch.tensor(
        [0.0, math.sqrt(max(1.0 - s * s, 0.0))],
        device=DEVICE,
        dtype=DTYPE,
    )
    assert_close(xj_all, expected_xj, name="oblique xj")
    assert torch.equal(crossed, torch.tensor([False, True], device=DEVICE))


def test_shells_x_batched_eta_shapes():
    profile = _two_shell_profile()
    eta = torch.tensor([0.0, math.pi / 6, math.pi / 3], device=DEVICE, dtype=DTYPE)

    xj_all, crossed, idx0 = profile.shells_x(eta)

    assert xj_all.shape == (3, 2)
    assert crossed.shape == (3, 2)
    assert idx0.shape == (3,)


def test_density_x_eta_layer_selection_nadir():
    profile = _two_shell_profile()
    eta = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

    inner = profile.density_x_eta(torch.tensor(0.3, device=DEVICE, dtype=DTYPE), eta)
    outer = profile.density_x_eta(torch.tensor(0.7, device=DEVICE, dtype=DTYPE), eta)

    assert_close(inner, torch.tensor(2.0, dtype=DTYPE), name="inner-shell density")
    assert_close(outer, torch.tensor(1.0, dtype=DTYPE), name="outer-shell density")


def test_density_x_eta_outside_earth_is_zero():
    profile = _two_shell_profile()
    eta = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
    x = torch.tensor([1.01, 1.2, 5.0], device=DEVICE, dtype=DTYPE)

    n_e = profile.density_x_eta(x, eta)

    assert_close(n_e, torch.zeros_like(n_e), name="density outside Earth is zero")


def test_density_x_eta_oblique_single_crossed_shell():
    profile = _two_shell_profile()
    eta = torch.tensor(math.pi / 3, device=DEVICE, dtype=DTYPE)

    n_e = profile.density_x_eta(torch.tensor(0.0, device=DEVICE, dtype=DTYPE), eta)

    assert_close(n_e, torch.tensor(1.0, dtype=DTYPE), name="oblique closest-approach density")


def test_density_x_eta_symmetric_in_x():
    profile = _two_shell_profile()
    eta = torch.tensor(math.pi / 6, device=DEVICE, dtype=DTYPE)
    x = torch.tensor([0.1, 0.3, 0.6, 0.85], device=DEVICE, dtype=DTYPE)

    n_pos = profile.density_x_eta(x, eta)
    n_neg = profile.density_x_eta(-x, eta)

    assert_close(n_neg, n_pos, name="density is even in x")


def test_density_x_eta_real_profile_center_denser_than_near_surface():
    profile = _default_profile()
    eta = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

    center = profile.density_x_eta(torch.tensor(0.0, device=DEVICE, dtype=DTYPE), eta)
    near_surface = profile.density_x_eta(torch.tensor(0.99, device=DEVICE, dtype=DTYPE), eta)

    assert float(center) >= float(near_surface)


def test_density_x_eta_real_profile_finite_and_nonnegative():
    profile = _default_profile()
    eta = torch.linspace(0.0, math.pi / 2 - 1.0e-3, 9, device=DEVICE, dtype=DTYPE)
    x = torch.linspace(0.0, 0.999, 11, device=DEVICE, dtype=DTYPE)
    x_grid, eta_grid = torch.broadcast_tensors(x[:, None], eta[None, :])

    n_e = profile.density_x_eta(x_grid, eta_grid)

    assert check_no_nan_inf(n_e)
    assert check_positive(n_e)


def test_call_and_dunder_call_match_density_x_eta():
    profile = _default_profile()
    x = torch.tensor([0.0, 0.3, 0.8], device=DEVICE, dtype=DTYPE)
    eta = torch.tensor([0.0, 0.4, 1.0], device=DEVICE, dtype=DTYPE)

    direct = profile.density_x_eta(x, eta)
    via_call = profile.call(x, eta)
    via_dunder = profile(x, eta)

    assert_close(via_call, direct, name="call matches density_x_eta")
    assert_close(via_dunder, direct, name="__call__ matches density_x_eta")


def test_trajectory_profile_consistent_with_shells_x():
    profile = _two_shell_profile()
    eta = torch.tensor(math.pi / 3, device=DEVICE, dtype=DTYPE)

    _, xj_traj, crossed_traj = profile.trajectory_profile(eta)
    xj_direct, crossed_direct, _ = profile.shells_x(eta)

    assert_close(xj_traj, xj_direct, name="trajectory_profile xj matches shells_x")
    assert torch.equal(crossed_traj, crossed_direct)


def test_earth_profile_rejects_non_increasing_rj():
    rj = torch.tensor([1.0, 0.5], device=DEVICE, dtype=DTYPE)
    coefficients = torch.tensor(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        _profile(rj, coefficients)


def test_earth_profile_rejects_rj_out_of_range():
    rj = torch.tensor([0.5, 1.5], device=DEVICE, dtype=DTYPE)
    coefficients = torch.tensor(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    with pytest.raises(ValueError, match=r"0 < rj <= 1"):
        _profile(rj, coefficients)


def test_earth_profile_rejects_empty_rj():
    rj = torch.empty(0, device=DEVICE, dtype=DTYPE)
    coefficients = torch.empty((0, 3), device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError, match="non-empty"):
        _profile(rj, coefficients)


def test_earth_profile_rejects_non_positive_profile_scale_m():
    rj = torch.tensor([0.5, 1.0], device=DEVICE, dtype=DTYPE)
    coefficients = torch.tensor(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    with pytest.raises(ValueError, match="profile_scale_m must be a positive scalar"):
        _profile(rj, coefficients, profile_scale_m=0.0)


def test_earth_profile_accepts_float32_context():
    profile = _two_shell_profile(dtype=torch.float32)

    n_e = profile.density_x_eta(
        torch.tensor(0.3, device=DEVICE, dtype=torch.float32),
        torch.tensor(0.0, device=DEVICE, dtype=torch.float32),
    )

    assert profile.rj.dtype == torch.float32
    assert n_e.dtype == torch.float32
    assert profile.device.type == DEVICE.type


def test_earth_profile_prem_model_also_valid():
    profile = _default_profile(profile_perturbative_name="prem")

    assert profile.rj.ndim == 1
    assert torch.all(torch.diff(profile.rj) > 0)
    assert torch.all((profile.rj > 0) & (profile.rj <= 1))

    n_e = profile.density_x_eta(
        torch.tensor(0.0, device=DEVICE, dtype=DTYPE),
        torch.tensor(0.0, device=DEVICE, dtype=DTYPE),
    )
    assert check_no_nan_inf(n_e)
    assert check_positive(n_e)
