#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytest-compatible tests for tpeanuts.medium.atmosphere.probability."""

from __future__ import annotations

import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.probability import probability_from_evolutor, probability_transition
from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
from tpeanuts.medium.atmosphere.probability import atmosphere_probability, patmosphere
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
CDTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(*, antinu=False, context: RuntimeContext | None = None) -> OscillationParameters:
    return OscillationParameters.from_preset(
        "_SM_NUFIT52_NO",
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
    P = atmosphere_probability(
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
    P = atmosphere_probability(
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

    P = atmosphere_probability(
        oscillation,
        E_MeV=1200.0,
        h_km=25.0,
        theta_deg=40.0,
        atmosphere=atmosphere,
        context=context,
    )
    p_mu = patmosphere(
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
    expected = probability_from_evolutor(
        S,
        weights,
        pmns=oscillation.pmns,
        massbasis=True,
        antinu=oscillation.antinu,
        real_dtype=DTYPE,
    )
    result = patmosphere(
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

    P = atmosphere_probability(
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

    probs = patmosphere(
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

    p_vac = patmosphere(state_mu, oscillation, **args, atmosphere=make_atmosphere(nsteps=96, matter=False), context=context)
    p_mat = patmosphere(state_mu, oscillation, **args, atmosphere=make_atmosphere(nsteps=96, matter=True), context=context)

    torch.testing.assert_close(p_mat.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE), atol=1.0e-12, rtol=1.0e-12)
    assert torch.linalg.norm(p_mat - p_vac) < 5.0e-4
