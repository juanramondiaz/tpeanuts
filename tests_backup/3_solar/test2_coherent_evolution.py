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
Spyder-friendly tests for coherent solar propagation.
"""



from __future__ import annotations

from pathlib import Path

import torch

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parents[3]


from tpeanuts.coherent.evolution import (
    solar_radius_fraction_to_core_x,
    solar_surface_evolutor,
    solar_surface_state,
    solar_to_earth_probabilities,
)
from tpeanuts.solar.profiles import load_default_solar_profile
from tpeanuts.util.test_utils import assert_close, assert_true, build_pmns, default_inputs, run_test_suite
from tpeanuts.util.constant import R_E, R_SUN, SUN_EARTH_DISTANCE_KM

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _inputs():
    p = default_inputs()
    return p, build_pmns(), load_default_solar_profile(device=DEVICE, dtype=DTYPE)


def test_solar_core_coordinate_conversion():
    rho = torch.tensor([0.0, 0.5, 1.0], device=DEVICE, dtype=DTYPE)
    x = solar_radius_fraction_to_core_x(rho, device=DEVICE, dtype=DTYPE)
    expected = rho * R_SUN / R_E

    print("\nsolar rho to peanuts core x:")
    print("rho:", rho)
    print("x  :", x)

    assert_close(x, expected, atol=1.0e-12, rtol=1.0e-12, name="rho to core x conversion")


def test_surface_production_is_identity():
    p, pmns, profile = _inputs()

    U = solar_surface_evolutor(
        pmns,
        p["DeltamSq21"],
        p["DeltamSq3l_NO"],
        torch.tensor(10.0, device=DEVICE, dtype=DTYPE),
        torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        profile=profile,
        device=DEVICE,
        dtype=DTYPE,
    )

    I = torch.eye(3, device=DEVICE, dtype=U.dtype)

    print("\nsolar surface evolutor for rho0=1:")
    print(U)

    assert_close(U, I, atol=1.0e-12, rtol=1.0e-12, name="Surface production returns identity")


def test_solar_surface_state_norm_conserved():
    p, pmns, profile = _inputs()

    psi = solar_surface_state(
        "nue",
        pmns,
        p["DeltamSq21"],
        p["DeltamSq3l_NO"],
        torch.tensor(10.0, device=DEVICE, dtype=DTYPE),
        torch.tensor(0.08, device=DEVICE, dtype=DTYPE),
        profile=profile,
        device=DEVICE,
        dtype=DTYPE,
    )

    norm = torch.sum(torch.abs(psi) ** 2)

    print("\ncoherent solar-surface state from rho0=0.08:")
    print(psi)
    print("norm:", norm)

    assert_true(torch.isfinite(psi).all().item(), "solar-surface state must be finite")
    assert_close(norm, torch.tensor(1.0, device=DEVICE, dtype=DTYPE), atol=1.0e-8, rtol=1.0e-8, name="solar-surface norm")


def test_solar_to_earth_probabilities_sum_to_one():
    p, pmns, profile = _inputs()

    P = solar_to_earth_probabilities(
        "nue",
        pmns,
        p["DeltamSq21"],
        p["DeltamSq3l_NO"],
        torch.tensor(10.0, device=DEVICE, dtype=DTYPE),
        torch.tensor(0.08, device=DEVICE, dtype=DTYPE),
        torch.tensor(SUN_EARTH_DISTANCE_KM, device=DEVICE, dtype=DTYPE),
        profile=profile,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("\ncoherent solar-to-earth probabilities:")
    print(P)
    print("sum:", P.sum())

    assert_true(P.shape == (3,), "earth-arrival probabilities must have shape (3,)")
    assert_true(torch.isfinite(P).all().item(), "earth-arrival probabilities must be finite")
    assert_true(torch.all(P >= -1.0e-10).item(), "earth-arrival probabilities must be non-negative within tolerance")
    assert_close(P.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE), atol=1.0e-8, rtol=1.0e-8, name="earth-arrival probability sum")


if __name__ == "__main__":
    tests = [
        test_solar_core_coordinate_conversion,
        test_surface_production_is_identity,
        test_solar_surface_state_norm_conserved,
        test_solar_to_earth_probabilities_sum_to_one,
    ]

    run_test_suite(tests, suite_name="coherent solar EVOLUTION tests", verbose_traceback=True)
