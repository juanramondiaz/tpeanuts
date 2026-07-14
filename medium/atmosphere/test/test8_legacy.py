#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Atmosphere propagation regression checks against legacy peanuts limits."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
from tpeanuts.medium.atmosphere.geometry import theta_to_eta
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.medium.earth.validation import compare_pearth_with_legacy
from tpeanuts.pipeline.atmosphere_flux import build_probability_matrix, propagate_flux_vector
from tpeanuts.util.context import RuntimeContext

pytest.importorskip("peanuts", reason="legacy peanuts reference package not available")


DTYPE = torch.float64
CDTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(*, context: RuntimeContext | None = None) -> OscillationParameters:
    ctx = context or make_context()
    return OscillationParameters.build(
        theta12=0.59,
        theta13=0.15,
        theta23=0.78,
        delta=1.20,
        DeltamSq21=7.42e-5,
        DeltamSq3l=2.517e-3,
        antinu=False,
        context=ctx,
    )


def make_atmosphere(**overrides) -> AtmosphereParameters:
    values = {
        "atmosphere_density_source": "exponential",
        "nsteps": 16,
        "method": "midpoint",
        "matter": False,
    }
    values.update(overrides)
    return AtmosphereParameters(**values)


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

    P_pipeline, _ = build_probability_matrix(
        oscillation,
        E_MeV=1000.0,
        h_km=0.0,
        theta_deg=theta_deg,
        detector_depth_m=0.0,
        atmosphere=make_atmosphere(),
        reunitarize_earth=False,
        context=context,
    )

    legacy_columns = []
    for alpha in range(3):
        state = torch.zeros(3, device=DEVICE, dtype=CDTYPE)
        state[alpha] = 1.0 + 0.0j
        comparison = compare_pearth_with_legacy(
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

    flux_out, P = propagate_flux_vector(
        flux_in,
        oscillation,
        E_MeV=1500.0,
        h_km=0.0,
        theta_deg=130.0,
        detector_depth_m=0.0,
        atmosphere=make_atmosphere(),
        reunitarize_earth=False,
        context=context,
    )

    expected = torch.matmul(P, flux_in[..., None]).squeeze(-1)
    torch.testing.assert_close(flux_out, expected, atol=1.0e-13, rtol=1.0e-13)
    assert np.isfinite(flux_out.detach().cpu().numpy()).all()
    torch.testing.assert_close(flux_out.sum(), flux_in.sum(), atol=2.0e-3, rtol=2.0e-3)


def test_theta_to_eta_conversion_used_for_legacy_comparison_has_expected_limits():
    theta = torch.tensor([0.0, 90.0, 180.0], device=DEVICE, dtype=DTYPE)
    eta = theta_to_eta(theta, device=DEVICE, dtype=DTYPE)
    expected = torch.tensor([torch.pi, torch.pi / 2.0, 0.0], device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(eta, expected, atol=1.0e-14, rtol=1.0e-14)
