#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytest-compatible tests for tpeanuts.medium.atmosphere.evolutor."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.constant import R_E


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
        "evolution_scale_m": R_E,
    }
    values.update(overrides)
    return AtmosphereParameters(**values)


def assert_unitary(S: torch.Tensor, *, atol: float = 1.0e-11, rtol: float = 1.0e-11) -> None:
    identity = torch.eye(3, device=S.device, dtype=S.dtype)
    torch.testing.assert_close(
        S.conj().transpose(-2, -1) @ S,
        identity.expand(*S.shape[:-2], 3, 3),
        atol=atol,
        rtol=rtol,
    )


def test_atmosphere_evolutor_returns_identity_for_zero_atmosphere_path():
    context = make_context()
    oscillation = make_oscillation(context=context)

    S, x = atmosphere_evolutor(
        oscillation,
        E_MeV=1000.0,
        h_km=0.0,
        theta_deg=45.0,
        atmosphere=make_atmosphere(nsteps=8),
        context=context,
    )

    expected = torch.eye(3, device=DEVICE, dtype=CDTYPE)
    torch.testing.assert_close(S, expected, atol=1.0e-14, rtol=1.0e-14)
    torch.testing.assert_close(x, torch.zeros(9, device=DEVICE, dtype=DTYPE), atol=0.0, rtol=0.0)


def test_atmosphere_evolutor_is_unitary_in_vacuum():
    context = make_context()
    oscillation = make_oscillation(context=context)

    S, x = atmosphere_evolutor(
        oscillation,
        E_MeV=1000.0,
        h_km=20.0,
        theta_deg=45.0,
        depth_km=1.0,
        atmosphere=make_atmosphere(nsteps=80, matter=False),
        context=context,
    )

    assert S.shape == (3, 3)
    assert x.shape == (81,)
    assert torch.all(torch.diff(x) > 0.0)
    assert_unitary(S)


def test_atmosphere_evolutor_preserves_state_norm_in_vacuum():
    context = make_context()
    oscillation = make_oscillation(context=context)
    state = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=CDTYPE)

    S, _ = atmosphere_evolutor(
        oscillation,
        E_MeV=1500.0,
        h_km=30.0,
        theta_deg=60.0,
        atmosphere=make_atmosphere(nsteps=72, matter=False),
        context=context,
    )

    out = S @ state
    torch.testing.assert_close(
        torch.sum(torch.abs(out) ** 2),
        torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_atmosphere_evolutor_broadcasts_energy_height_and_angle_grids():
    context = make_context()
    oscillation = make_oscillation(context=context)
    energy = torch.tensor([500.0, 1000.0, 5000.0], device=DEVICE, dtype=DTYPE)[:, None]
    height = torch.tensor([15.0, 35.0], device=DEVICE, dtype=DTYPE)[None, :]
    theta = torch.tensor(55.0, device=DEVICE, dtype=DTYPE)

    S, x = atmosphere_evolutor(
        oscillation,
        E_MeV=energy,
        h_km=height,
        theta_deg=theta,
        atmosphere=make_atmosphere(nsteps=12, matter=False),
        context=context,
    )

    assert S.shape == (3, 2, 3, 3)
    assert x.shape == (1, 2, 13)
    assert torch.isfinite(S.real).all()
    assert torch.isfinite(S.imag).all()
    assert_unitary(S, atol=2.0e-11, rtol=2.0e-11)


def test_atmosphere_evolutor_matter_and_vacuum_are_close_for_atmospheric_densities():
    context = make_context()
    oscillation = make_oscillation(context=context)
    args = dict(E_MeV=5000.0, h_km=80.0, theta_deg=85.0, depth_km=1.0)

    S_vac, _ = atmosphere_evolutor(
        oscillation,
        **args,
        atmosphere=make_atmosphere(nsteps=96, matter=False),
        context=context,
    )
    S_mat, _ = atmosphere_evolutor(
        oscillation,
        **args,
        atmosphere=make_atmosphere(nsteps=96, matter=True),
        context=context,
    )

    assert_unitary(S_mat, atol=2.0e-11, rtol=2.0e-11)
    assert torch.linalg.norm(S_mat - S_vac) < 5.0e-4


def test_atmosphere_evolutor_antinu_vacuum_is_unitary_and_finite():
    context = make_context()
    osc_nu = make_oscillation(antinu=False, context=context)
    osc_anu = make_oscillation(antinu=True, context=context)

    kwargs = dict(
        E_MeV=2000.0,
        h_km=20.0,
        theta_deg=30.0,
        atmosphere=make_atmosphere(nsteps=48, matter=False),
        context=context,
    )
    S_nu, _ = atmosphere_evolutor(osc_nu, **kwargs)
    S_anu, _ = atmosphere_evolutor(osc_anu, **kwargs)

    assert torch.isfinite(S_anu.real).all()
    assert torch.isfinite(S_anu.imag).all()
    assert_unitary(S_anu)
    assert torch.linalg.norm(S_anu - S_nu) > 0.0


def test_atmosphere_evolutor_rejects_non_positive_nsteps():
    with pytest.raises(ValueError, match="nsteps"):
        atmosphere_evolutor(
            make_oscillation(),
            E_MeV=1000.0,
            h_km=20.0,
            theta_deg=30.0,
            atmosphere=make_atmosphere(nsteps=0),
            context=make_context(),
        )
