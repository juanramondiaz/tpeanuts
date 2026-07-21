#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Atmosphere propagation regression checks against legacy peanuts limits."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
from tpeanuts.medium.atmosphere.geometry import theta_to_eta
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.medium.earth.validation import compare_earth_probability_state_with_legacy
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.probability import probability_incoherent
from tpeanuts.pipeline.atmosphere import propagate_atmosphere_to_surface
from tpeanuts.pipeline.atmosphere_earth import propagate_surface_to_detector
from tpeanuts.util.context import RuntimeContext

pytest.importorskip("peanuts", reason="legacy peanuts reference package not available")


DTYPE = torch.float64
CDTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(*, context: RuntimeContext | None = None) -> OscillationParameters:
    ctx = context or make_context()
    pmns = PMNS_SM(PMNSParams(theta12=0.59, theta13=0.15, theta23=0.78, delta=1.20, context=ctx))
    mass_spectrum = MassSpectrum_SM(
        DeltamSq21=torch.as_tensor(7.42e-5, device=ctx.device, dtype=ctx.dtype),
        DeltamSq3l=torch.as_tensor(2.517e-3, device=ctx.device, dtype=ctx.dtype),
    )
    return OscillationParameters(pmns=pmns, mass_spectrum=mass_spectrum, antinu=False)


def make_atmosphere(**overrides) -> AtmosphereParameters:
    values = {
        "atmosphere_density_source": "exponential",
        "nsteps": 16,
        "method": "midpoint",
        "matter": False,
    }
    values.update(overrides)
    return AtmosphereParameters(**values)


def make_production(*, energy_GeV: float, height_km: float, theta_deg: float, particle: str) -> dict:
    return {
        "particle": particle,
        "E_grid_GeV": torch.tensor([energy_GeV], device=DEVICE, dtype=DTYPE),
        "h_grid_km": torch.tensor([height_km], device=DEVICE, dtype=DTYPE),
        "theta_deg": theta_deg,
        "phi_Eh": torch.ones((1, 1), device=DEVICE, dtype=DTYPE),
    }


def make_config(context: RuntimeContext, oscillation: OscillationParameters) -> PropagationConfig:
    return PropagationConfig(
        runtime=context,
        oscillation=oscillation,
        atmosphere=make_atmosphere(),
        detector_depth_m=0.0,
        reunitarize_earth=False,
    )


def test_zero_height_atmosphere_evolutor_is_identity_before_legacy_earth_limit():
    context = make_context()
    S_atm, x = atmosphere_evolutor(
        make_oscillation(context=context),
        E_MeV=1000.0,
        h_km=0.0,
        theta_deg=120.0,
        atmosphere=make_atmosphere(),
        context=context,
    )

    torch.testing.assert_close(S_atm, torch.eye(3, device=DEVICE, dtype=CDTYPE), atol=1.0e-14, rtol=1.0e-14)
    torch.testing.assert_close(x, torch.zeros(17, device=DEVICE, dtype=DTYPE), atol=0.0, rtol=0.0)


def test_pipeline_zero_height_matches_legacy_peanuts_earth_probability_columns():
    context = make_context()
    oscillation = make_oscillation(context=context)
    theta_deg = 120.0
    eta = float(theta_to_eta(theta_deg, device=DEVICE, dtype=DTYPE).detach().cpu())

    config = make_config(context, oscillation)
    columns = []
    for particle in ("nue", "numu", "nutau"):
        surface = propagate_atmosphere_to_surface(
            make_production(
                energy_GeV=1.0,
                height_km=0.0,
                theta_deg=theta_deg,
                particle=particle,
            ),
            config,
            trajectory_steps=2,
        )
        detector = propagate_surface_to_detector(surface, config)
        columns.append(detector.detector_probabilities.squeeze((0, 1)))
    P_pipeline = torch.stack(columns, dim=-1)

    legacy_columns = []
    for alpha in range(3):
        state = torch.zeros(3, device=DEVICE, dtype=CDTYPE)
        state[alpha] = 1.0 + 0.0j
        comparison = compare_earth_probability_state_with_legacy(
            state,
            oscillation,
            E_MeV=1000.0,
            eta=eta,
            depth_m=0.0,
            method="analytical",
            massbasis=False,
            reunitarize=False,
            context=context,
        )
        assert comparison["max_abs"] < 3.0e-6
        legacy_columns.append(torch.as_tensor(comparison["legacy"], device=DEVICE, dtype=DTYPE))

    P_legacy = torch.stack(legacy_columns, dim=-1)
    torch.testing.assert_close(P_pipeline, P_legacy, atol=3.0e-6, rtol=3.0e-5)


def test_pipeline_zero_height_flux_matches_legacy_probability_action():
    context = make_context()
    oscillation = make_oscillation(context=context)
    flux_in = torch.tensor([1.0, 2.0, 0.5], device=DEVICE, dtype=DTYPE)

    config = make_config(context, oscillation)
    columns = []
    for particle in ("nue", "numu", "nutau"):
        surface = propagate_atmosphere_to_surface(
            make_production(
                energy_GeV=1.5,
                height_km=0.0,
                theta_deg=130.0,
                particle=particle,
            ),
            config,
            trajectory_steps=2,
        )
        detector = propagate_surface_to_detector(surface, config)
        columns.append(detector.detector_probabilities.squeeze((0, 1)))
    P = torch.stack(columns, dim=-1)
    flux_out = probability_incoherent(P, flux_in)

    expected = torch.matmul(P, flux_in[..., None]).squeeze(-1)
    torch.testing.assert_close(flux_out, expected, atol=1.0e-13, rtol=1.0e-13)
    assert np.isfinite(flux_out.detach().cpu().numpy()).all()
    torch.testing.assert_close(flux_out.sum(), flux_in.sum(), atol=2.0e-3, rtol=2.0e-3)


def test_theta_to_eta_conversion_used_for_legacy_comparison_has_expected_limits():
    theta = torch.tensor([0.0, 90.0, 180.0], device=DEVICE, dtype=DTYPE)
    eta = theta_to_eta(theta, device=DEVICE, dtype=DTYPE)
    expected = torch.tensor([torch.pi, torch.pi / 2.0, 0.0], device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(eta, expected, atol=1.0e-14, rtol=1.0e-14)
