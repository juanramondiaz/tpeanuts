#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytest-compatible tests for tpeanuts.medium.atmosphere.probability."""

from __future__ import annotations

import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.probability import probability_state, probability_transition
from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
from tpeanuts.medium.atmosphere.probability import (
    atmosphere_probability_transition,
    atmosphere_probability_state,
    atmosphere_probability_integrated,
    atmosphere_probability_integrated_angular,
    atmosphere_probability_integrated_height,
)
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
CDTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(*, antinu=False, context: RuntimeContext | None = None) -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO",
        antinu=antinu,
        context=context or make_context(),
    )


def make_sterile_oscillation(*, antinu=False, context: RuntimeContext | None = None) -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset(
        "sterile_3p1_bestfit_giunti2017",
        antinu=antinu,
        context=context or make_context(),
    )


def make_atmosphere(**overrides) -> AtmosphereParameters:
    values = {
        "atmosphere_density_source": "exponential",
        "nsteps": 64,
        "method": "midpoint",
        "matter": False,
    }
    values.update(overrides)
    return AtmosphereParameters(**values)


def assert_probability_matrix(P: torch.Tensor, *, atol: float = 1.0e-11) -> None:
    assert P.shape[-2:] == (3, 3)
    assert torch.isfinite(P).all()
    assert torch.all(P >= -atol)
    torch.testing.assert_close(
        P.sum(dim=-2),
        torch.ones_like(P[..., 0, :]),
        atol=atol,
        rtol=atol,
    )


def test_patmosphere_exposes_analytical_dispatcher():
    context = make_context()
    oscillation = make_oscillation(context=context)
    state = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=CDTYPE)
    atmosphere = make_atmosphere(
        matter=True,
        nsteps=600,
        perturbative_segments=6,
        perturbative_degree=3,
    )
    args = dict(
        nustate=state,
        oscillation=oscillation,
        E_MeV=5000.0,
        h_km=80.0,
        theta_deg=85.0,
        atmosphere=atmosphere,
        context=context,
    )

    analytical = atmosphere_probability_state(**args, method="analytical")
    numerical = atmosphere_probability_state(**args, method="numerical")

    torch.testing.assert_close(analytical, numerical, atol=1.0e-8, rtol=1.0e-8)


def test_atmosphere_probability_matches_evolutor_modulus_squared():
    context = make_context()
    oscillation = make_oscillation(context=context)
    atmosphere = make_atmosphere(nsteps=48, matter=False)

    S, _ = atmosphere_evolutor(
        oscillation,
        E_MeV=1000.0,
        h_km=20.0,
        theta_deg=45.0,
        atmosphere=atmosphere,
        context=context,
    )
    P = atmosphere_probability_transition(
        oscillation,
        E_MeV=1000.0,
        h_km=20.0,
        theta_deg=45.0,
        atmosphere=atmosphere,
        context=context,
    )

    torch.testing.assert_close(P, probability_transition(S, real_dtype=DTYPE), atol=1.0e-13, rtol=1.0e-13)
    assert_probability_matrix(P)


def test_atmosphere_probability_identity_for_zero_height():
    context = make_context()
    P = atmosphere_probability_transition(
        make_oscillation(context=context),
        E_MeV=1000.0,
        h_km=0.0,
        theta_deg=70.0,
        atmosphere=make_atmosphere(nsteps=8),
        context=context,
    )

    expected = torch.eye(3, device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(P, expected, atol=1.0e-14, rtol=1.0e-14)


def test_patmosphere_flavour_state_matches_probability_column():
    context = make_context()
    oscillation = make_oscillation(context=context)
    atmosphere = make_atmosphere(nsteps=56, matter=False)
    state_mu = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=CDTYPE)

    P = atmosphere_probability_transition(
        oscillation,
        E_MeV=1200.0,
        h_km=25.0,
        theta_deg=40.0,
        atmosphere=atmosphere,
        context=context,
    )
    p_mu = atmosphere_probability_state(
        state_mu,
        oscillation,
        E_MeV=1200.0,
        h_km=25.0,
        theta_deg=40.0,
        atmosphere=atmosphere,
        context=context,
    )

    torch.testing.assert_close(p_mu, P[:, 1], atol=1.0e-13, rtol=1.0e-13)
    torch.testing.assert_close(p_mu.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE), atol=1.0e-13, rtol=1.0e-13)


def test_patmosphere_massbasis_matches_core_probability_projection():
    context = make_context()
    oscillation = make_oscillation(context=context)
    atmosphere = make_atmosphere(nsteps=56, matter=False)
    weights = torch.tensor([0.55, 0.30, 0.15], device=DEVICE, dtype=DTYPE)

    S, _ = atmosphere_evolutor(
        oscillation,
        E_MeV=1500.0,
        h_km=35.0,
        theta_deg=55.0,
        atmosphere=atmosphere,
        context=context,
    )
    expected = probability_state(
        S,
        weights,
        pmns=oscillation.pmns,
        massbasis=True,
        antinu=oscillation.antinu,
        real_dtype=DTYPE,
    )
    result = atmosphere_probability_state(
        weights,
        oscillation,
        E_MeV=1500.0,
        h_km=35.0,
        theta_deg=55.0,
        massbasis=True,
        atmosphere=atmosphere,
        context=context,
    )

    torch.testing.assert_close(result, expected, atol=1.0e-13, rtol=1.0e-13)
    torch.testing.assert_close(result.sum(), weights.sum(), atol=1.0e-13, rtol=1.0e-13)


def test_atmosphere_probability_broadcasts_energy_height_angle_grid():
    context = make_context()
    oscillation = make_oscillation(context=context)
    energy = torch.tensor([500.0, 1000.0, 5000.0], device=DEVICE, dtype=DTYPE)[:, None]
    height = torch.tensor([10.0, 50.0], device=DEVICE, dtype=DTYPE)[None, :]
    theta = torch.tensor(65.0, device=DEVICE, dtype=DTYPE)

    P = atmosphere_probability_transition(
        oscillation,
        E_MeV=energy,
        h_km=height,
        theta_deg=theta,
        atmosphere=make_atmosphere(nsteps=16, matter=False),
        context=context,
    )

    assert P.shape == (3, 2, 3, 3)
    assert_probability_matrix(P, atol=2.0e-11)


def test_patmosphere_batched_state_and_fluxlike_weights_are_normalized():
    context = make_context()
    oscillation = make_oscillation(context=context)
    states = torch.eye(3, device=DEVICE, dtype=CDTYPE)

    probs = atmosphere_probability_state(
        states,
        oscillation,
        E_MeV=1200.0,
        h_km=20.0,
        theta_deg=35.0,
        atmosphere=make_atmosphere(nsteps=32, matter=False),
        context=context,
    )

    assert probs.shape == (3, 3)
    assert torch.all(probs >= -1.0e-12)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), atol=1.0e-12, rtol=1.0e-12)


def test_patmosphere_matter_effect_is_small_but_finite_for_long_atmospheric_path():
    context = make_context()
    oscillation = make_oscillation(context=context)
    state_mu = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=CDTYPE)
    args = dict(E_MeV=5000.0, h_km=80.0, theta_deg=85.0, depth_km=1.0)

    p_vac = atmosphere_probability_state(state_mu, oscillation, **args, atmosphere=make_atmosphere(nsteps=96, matter=False), context=context)
    p_mat = atmosphere_probability_state(state_mu, oscillation, **args, atmosphere=make_atmosphere(nsteps=96, matter=True), context=context)

    torch.testing.assert_close(p_mat.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE), atol=1.0e-12, rtol=1.0e-12)
    assert torch.linalg.norm(p_mat - p_vac) < 5.0e-4


def test_atmosphere_probability_integrated_matches_manual_energy_average():
    context = make_context()
    oscillation = make_oscillation(context=context)
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    theta = torch.tensor(65.0, device=DEVICE, dtype=DTYPE)
    h = torch.tensor(20.0, device=DEVICE, dtype=DTYPE)
    E = torch.tensor([500.0, 1000.0, 3000.0], device=DEVICE, dtype=DTYPE)
    spectrum = torch.tensor([1.0, 2.0, 1.0], device=DEVICE, dtype=DTYPE)
    atmosphere = make_atmosphere(nsteps=16, matter=False)

    P = atmosphere_probability_state(
        weights, oscillation, E_MeV=E, h_km=h, theta_deg=theta, massbasis=True,
        atmosphere=atmosphere, context=context, method="analytical",
    )
    expected = torch.trapezoid(P * spectrum[:, None], x=E, dim=-2) / torch.trapezoid(spectrum, x=E)

    result = atmosphere_probability_integrated(
        weights, oscillation, E, h, theta, spectrum, massbasis=True,
        atmosphere=atmosphere, context=context, method="analytical",
    )

    torch.testing.assert_close(result, expected, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(result.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE), atol=1.0e-12, rtol=1.0e-12)


def test_atmosphere_probability_integrated_angular_matches_manual_average():
    context = make_context()
    oscillation = make_oscillation(context=context)
    state_mu = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=CDTYPE)
    theta = torch.tensor([30.0, 60.0, 90.0], device=DEVICE, dtype=DTYPE)
    atmosphere = make_atmosphere(nsteps=16, matter=False)

    P = atmosphere_probability_state(
        state_mu, oscillation, E_MeV=1000.0, h_km=20.0, theta_deg=theta,
        atmosphere=atmosphere, context=context, method="analytical",
    )
    theta_rad = torch.deg2rad(theta)
    sin_theta = torch.sin(theta_rad)
    expected = torch.trapezoid(P * sin_theta[:, None], x=theta_rad, dim=-2) / torch.trapezoid(sin_theta, x=theta_rad)

    result = atmosphere_probability_integrated_angular(
        state_mu, oscillation, E_MeV=1000.0, h_km=20.0, theta_deg=theta,
        atmosphere=atmosphere, context=context, method="analytical",
    )

    torch.testing.assert_close(result, expected, atol=1.0e-12, rtol=1.0e-12)


def test_atmosphere_probability_integrated_height_matches_manual_average():
    context = make_context()
    oscillation = make_oscillation(context=context)
    state_mu = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=CDTYPE)
    h = torch.tensor([5.0, 20.0, 40.0], device=DEVICE, dtype=DTYPE)
    production_flux = torch.tensor([1.0, 3.0, 2.0], device=DEVICE, dtype=DTYPE)
    atmosphere = make_atmosphere(nsteps=16, matter=False)

    P = atmosphere_probability_state(
        state_mu, oscillation, E_MeV=1000.0, h_km=h, theta_deg=45.0,
        atmosphere=atmosphere, context=context, method="analytical",
    )
    expected = torch.trapezoid(P * production_flux[:, None], x=h, dim=-2) / torch.trapezoid(production_flux, x=h)

    result = atmosphere_probability_integrated_height(
        state_mu, oscillation, E_MeV=1000.0, h_km=h, theta_deg=45.0, production_flux=production_flux,
        atmosphere=atmosphere, context=context, method="analytical",
    )

    torch.testing.assert_close(result, expected, atol=1.0e-12, rtol=1.0e-12)


def test_sterile_atmosphere_probability_state_normalized_n4():
    """Regression test: atmosphere_probability_state used to crash for a
    4-flavour (3+1 sterile) oscillation object (broadcast_last3 hardcoded
    the state vector's final dimension to 3)."""
    context = make_context()
    oscillation = make_sterile_oscillation(context=context)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=CDTYPE)
    atmosphere = make_atmosphere(nsteps=64, matter=True)

    P = atmosphere_probability_state(
        state, oscillation, E_MeV=2000.0, h_km=20.0, theta_deg=45.0, depth_km=1.0,
        atmosphere=atmosphere, context=context, method="numerical",
    )

    assert P.shape == (4,)
    assert torch.all(torch.isfinite(P))
    torch.testing.assert_close(P.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE), atol=1.0e-10, rtol=1.0e-10)


def test_sterile_atmosphere_probability_transition_doubly_stochastic_n4():
    context = make_context()
    oscillation = make_sterile_oscillation(context=context)
    atmosphere = make_atmosphere(nsteps=16, matter=False)

    P = atmosphere_probability_transition(
        oscillation, E_MeV=1000.0, h_km=20.0, theta_deg=45.0,
        atmosphere=atmosphere, context=context, method="numerical",
    )

    assert P.shape == (4, 4)
    assert torch.isfinite(P).all()
    assert torch.all(P >= -1.0e-11)
    torch.testing.assert_close(P.sum(dim=-2), torch.ones(4, device=DEVICE, dtype=DTYPE), atol=1.0e-11, rtol=1.0e-11)


def test_atmosphere_probability_state_include_matter_nc_changes_sterile_result():
    context = make_context()
    oscillation = make_sterile_oscillation(context=context)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=CDTYPE)
    args = dict(
        nustate=state, oscillation=oscillation, E_MeV=2000.0, h_km=20.0,
        theta_deg=45.0, depth_km=1.0, context=context, method="numerical",
    )

    P_cc = atmosphere_probability_state(**args, atmosphere=make_atmosphere(nsteps=64, matter=True, include_matter_nc=False))
    P_nc = atmosphere_probability_state(**args, atmosphere=make_atmosphere(nsteps=64, matter=True, include_matter_nc=True))

    assert P_cc.shape == (4,)
    assert P_nc.shape == (4,)
    assert torch.max(torch.abs(P_nc - P_cc)) > 0.0
