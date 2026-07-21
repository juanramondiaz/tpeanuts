#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytest-compatible tests for tpeanuts.medium.atmosphere.evolutor."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.constant import R_E


DTYPE = torch.float64
CDTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(
    *, antinu=False, NSI_extension: str | None = None, context: RuntimeContext | None = None,
) -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO",
        antinu=antinu,
        NSI_extension=NSI_extension,
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
    n = S.shape[-1]
    identity = torch.eye(n, device=S.device, dtype=S.dtype)
    torch.testing.assert_close(
        S.conj().transpose(-2, -1) @ S,
        identity.expand(*S.shape[:-2], n, n),
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


def test_atmosphere_analytical_matches_fine_numerical_exponential_profile():
    context = make_context()
    oscillation = make_oscillation(context=context)
    atmosphere = make_atmosphere(
        nsteps=800,
        matter=True,
        perturbative_segments=6,
        perturbative_degree=3,
    )
    args = dict(
        E_MeV=5000.0,
        h_km=80.0,
        theta_deg=89.0,
        depth_km=1.0,
        atmosphere=atmosphere,
        context=context,
    )
    analytical, x = atmosphere_evolutor(oscillation, **args, method="analytical")
    numerical, _ = atmosphere_evolutor(oscillation, **args, method="numerical")

    assert x.shape == (atmosphere.perturbative_segments + 1,)
    torch.testing.assert_close(analytical, numerical, atol=1.0e-8, rtol=1.0e-8)
    assert_unitary(analytical, atol=2.0e-11, rtol=2.0e-11)


def test_atmosphere_analytical_broadcasts_energy_and_geometry():
    context = make_context()
    oscillation = make_oscillation(context=context)
    energy = torch.tensor([500.0, 1000.0, 5000.0], device=DEVICE, dtype=DTYPE)[:, None]
    height = torch.tensor([15.0, 35.0], device=DEVICE, dtype=DTYPE)[None, :]
    atmosphere = make_atmosphere(matter=True, perturbative_segments=4, perturbative_degree=3)

    S, x = atmosphere_evolutor(
        oscillation,
        energy,
        height,
        55.0,
        atmosphere=atmosphere,
        context=context,
        method="analytical",
    )

    assert S.shape == (3, 2, 3, 3)
    assert x.shape == (1, 2, 5)
    assert torch.isfinite(S).all()


def test_atmosphere_analytical_supports_nsi_and_matches_numerical():
    """Regression test: the analytical atmosphere path now supports NSI
    (previously it raised ValueError). This also exercises a real,
    non-constant density profile combined with a batched (E, theta) grid,
    which used to break ``evolutor_first_order``'s NSI sandwich term (the
    P "direction" matrix leaked the profile's batch shape -- see the
    ``conjugate_right``/``evolutor_first_order`` fixes in
    ``core/perturbative/evolutor.py``).
    """
    ctx = make_context()
    oscillation = make_oscillation(context=ctx, NSI_extension="nsi_globalfit_esteban2018")
    atmosphere = make_atmosphere(matter=True, perturbative_segments=4, perturbative_degree=2)
    E = torch.tensor([800.0, 3000.0], device=DEVICE, dtype=DTYPE)
    theta = torch.tensor([30.0, 150.0], device=DEVICE, dtype=DTYPE)

    S_a, _ = atmosphere_evolutor(
        oscillation, E[:, None], 20.0, theta[None, :], 1.0,
        method="analytical", atmosphere=atmosphere, context=ctx,
    )
    S_n, _ = atmosphere_evolutor(
        oscillation, E[:, None], 20.0, theta[None, :], 1.0,
        method="numerical", atmosphere=make_atmosphere(matter=True, nsteps=400), context=ctx,
    )

    assert S_a.shape == (2, 2, 3, 3)
    assert torch.max(torch.abs(S_a - S_n)) < 5.0e-4

    identity = torch.eye(3, device=DEVICE, dtype=CDTYPE)
    unitarity_error = torch.max(torch.abs(S_a.conj().transpose(-1, -2) @ S_a - identity))
    assert unitarity_error < 1.0e-8


def test_atmosphere_analytical_sterile_handles_zero_path_identity_region():
    """Regression test: ``atmosphere_evolutor_analytical`` used to hardcode
    ``torch.eye(3, ...)`` for the L_atm<=0 (no-atmosphere-path) identity
    region, which crashed with a shape mismatch for a 4-flavour (3+1
    sterile) oscillation object instead of returning the identity. A
    production height at or below the detector depth gives L_atm <= 0 for
    at least one angle in this grid, exercising that branch.
    """
    ctx = make_context()
    oscillation = PropagationConfig.oscillation_parameters_from_preset(
        "sterile_3p1_bestfit_giunti2017", context=ctx,
    )
    atmosphere = make_atmosphere(matter=True)
    E = torch.tensor([1000.0, 5000.0], device=DEVICE, dtype=DTYPE)
    theta = torch.tensor([30.0, 180.0], device=DEVICE, dtype=DTYPE)

    S, _ = atmosphere_evolutor(
        oscillation, E[:, None], 1.0, theta[None, :], 1.0,
        method="analytical", atmosphere=atmosphere, context=ctx,
    )

    assert S.shape == (2, 2, 4, 4)
    assert torch.all(torch.isfinite(S.real)) and torch.all(torch.isfinite(S.imag))
    unitarity_error = torch.max(torch.abs(
        S.conj().transpose(-1, -2) @ S - torch.eye(4, device=DEVICE, dtype=CDTYPE)
    ))
    assert unitarity_error < 1.0e-6


def test_atmosphere_evolutor_numerical_include_matter_nc_changes_sterile_result():
    ctx = make_context()
    oscillation = PropagationConfig.oscillation_parameters_from_preset(
        "sterile_3p1_bestfit_giunti2017", context=ctx,
    )
    args = dict(E_MeV=2000.0, h_km=20.0, theta_deg=45.0, depth_km=1.0, context=ctx)

    S_cc, _ = atmosphere_evolutor(
        oscillation, **args, method="numerical",
        atmosphere=make_atmosphere(nsteps=64, matter=True, include_matter_nc=False),
    )
    S_nc, _ = atmosphere_evolutor(
        oscillation, **args, method="numerical",
        atmosphere=make_atmosphere(nsteps=64, matter=True, include_matter_nc=True),
    )

    assert S_cc.shape == (4, 4)
    assert S_nc.shape == (4, 4)
    assert torch.isfinite(S_nc.real).all() and torch.isfinite(S_nc.imag).all()
    assert_unitary(S_nc, atol=1.0e-9, rtol=1.0e-9)
    assert torch.max(torch.abs(S_nc - S_cc)) > 0.0


def test_atmosphere_evolutor_numerical_include_matter_nc_is_noop_for_three_flavour():
    ctx = make_context()
    oscillation = make_oscillation(context=ctx)
    args = dict(E_MeV=2000.0, h_km=20.0, theta_deg=45.0, depth_km=1.0, context=ctx)

    S_cc, _ = atmosphere_evolutor(
        oscillation, **args, method="numerical",
        atmosphere=make_atmosphere(nsteps=64, matter=True, include_matter_nc=False),
    )
    S_nc, _ = atmosphere_evolutor(
        oscillation, **args, method="numerical",
        atmosphere=make_atmosphere(nsteps=64, matter=True, include_matter_nc=True),
    )

    torch.testing.assert_close(S_nc, S_cc, atol=1.0e-13, rtol=1.0e-13)


def test_atmosphere_evolutor_analytical_include_matter_nc_changes_sterile_result():
    ctx = make_context()
    oscillation = PropagationConfig.oscillation_parameters_from_preset(
        "sterile_3p1_bestfit_giunti2017", context=ctx,
    )
    args = dict(E_MeV=2000.0, h_km=20.0, theta_deg=45.0, depth_km=1.0, context=ctx)

    S_cc, _ = atmosphere_evolutor(
        oscillation, **args, method="analytical",
        atmosphere=make_atmosphere(matter=True, perturbative_segments=4, perturbative_degree=2, include_matter_nc=False),
    )
    S_nc, _ = atmosphere_evolutor(
        oscillation, **args, method="analytical",
        atmosphere=make_atmosphere(matter=True, perturbative_segments=4, perturbative_degree=2, include_matter_nc=True),
    )

    assert S_cc.shape == (4, 4)
    assert S_nc.shape == (4, 4)
    assert torch.isfinite(S_nc.real).all() and torch.isfinite(S_nc.imag).all()
    assert_unitary(S_nc, atol=1.0e-6, rtol=1.0e-6)
    assert torch.max(torch.abs(S_nc - S_cc)) > 0.0


def test_atmosphere_evolutor_analytical_include_matter_nc_is_noop_for_three_flavour():
    ctx = make_context()
    oscillation = make_oscillation(context=ctx)
    args = dict(E_MeV=2000.0, h_km=20.0, theta_deg=45.0, depth_km=1.0, context=ctx)

    S_cc, _ = atmosphere_evolutor(
        oscillation, **args, method="analytical",
        atmosphere=make_atmosphere(matter=True, perturbative_segments=4, perturbative_degree=2, include_matter_nc=False),
    )
    S_nc, _ = atmosphere_evolutor(
        oscillation, **args, method="analytical",
        atmosphere=make_atmosphere(matter=True, perturbative_segments=4, perturbative_degree=2, include_matter_nc=True),
    )

    torch.testing.assert_close(S_nc, S_cc, atol=1.0e-13, rtol=1.0e-13)


def test_atmosphere_analytical_vs_numerical_include_matter_nc_agree():
    ctx = make_context()
    oscillation = PropagationConfig.oscillation_parameters_from_preset(
        "sterile_3p1_bestfit_giunti2017", context=ctx,
    )
    args = dict(E_MeV=5000.0, h_km=80.0, theta_deg=89.0, depth_km=1.0, context=ctx)

    S_a, _ = atmosphere_evolutor(
        oscillation, **args, method="analytical",
        atmosphere=make_atmosphere(matter=True, perturbative_segments=6, perturbative_degree=3, include_matter_nc=True),
    )
    S_n, _ = atmosphere_evolutor(
        oscillation, **args, method="numerical",
        atmosphere=make_atmosphere(matter=True, nsteps=800, include_matter_nc=True),
    )

    assert torch.max(torch.abs(S_a - S_n)) < 1.0e-6
    assert_unitary(S_a, atol=1.0e-6, rtol=1.0e-6)


def test_atmosphere_evolutor_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        atmosphere_evolutor(
            make_oscillation(),
            1000.0,
            20.0,
            30.0,
            method="invalid",
            context=make_context(),
        )
