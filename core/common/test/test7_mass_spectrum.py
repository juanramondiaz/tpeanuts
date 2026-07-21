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
Pytest-compatible tests for tpeanuts.core.common.mass_spectrum (MassSpectrum)
and its SM/BSM implementations (MassSpectrum_SM, MassSpectrum_BSM).
"""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.common.mass_spectrum import MassSpectrum
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.core.BSM.bsm_mass_spectrum import MassSpectrum_BSM
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_mass_spectrum_is_abstract():
    with pytest.raises(TypeError):
        MassSpectrum(
            DeltamSq21=torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE),
            DeltamSq3l=torch.tensor(2.517e-3, device=DEVICE, dtype=DTYPE),
        )


def test_difference_vector_base_normal_ordering():
    dm21 = torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE)
    dm3l = torch.tensor(2.517e-3, device=DEVICE, dtype=DTYPE)
    spectrum = MassSpectrum_SM(DeltamSq21=dm21, DeltamSq3l=dm3l)

    mass_vector = spectrum.difference_vector_base()
    expected = torch.stack([torch.zeros_like(dm21), dm21, dm3l])

    assert mass_vector.shape == (3,)
    assert_close(mass_vector, expected, name="normal-ordering mass vector")


def test_difference_vector_base_inverted_ordering():
    dm21 = torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE)
    dm3l = torch.tensor(-2.498e-3, device=DEVICE, dtype=DTYPE)
    spectrum = MassSpectrum_SM(DeltamSq21=dm21, DeltamSq3l=dm3l)

    mass_vector = spectrum.difference_vector_base()
    expected = torch.stack([-dm21, torch.zeros_like(dm21), dm3l])

    assert mass_vector.shape == (3,)
    assert_close(mass_vector, expected, name="inverted-ordering mass vector")


def test_difference_vector_base_batched_ordering():
    dm21 = torch.tensor([7.42e-5, 7.42e-5], device=DEVICE, dtype=DTYPE)
    dm3l = torch.tensor([2.517e-3, -2.498e-3], device=DEVICE, dtype=DTYPE)
    spectrum = MassSpectrum_SM(DeltamSq21=dm21, DeltamSq3l=dm3l)

    mass_vector = spectrum.difference_vector_base()
    expected = torch.tensor(
        [
            [0.0, 7.42e-5, 2.517e-3],
            [-7.42e-5, 0.0, -2.498e-3],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    assert mass_vector.shape == (2, 3)
    assert_close(mass_vector, expected, name="batched ordering mass vector")


def test_mass_spectrum_sm_difference_vector_matches_base():
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    dm21 = torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE)
    dm3l = torch.tensor(2.517e-3, device=DEVICE, dtype=DTYPE)
    spectrum = MassSpectrum_SM(DeltamSq21=dm21, DeltamSq3l=dm3l)

    assert_close(
        spectrum.difference_vector(context=ctx),
        spectrum.difference_vector_base(context=ctx),
        name="SM difference_vector equals difference_vector_base",
    )


def test_mass_spectrum_bsm_appends_deltamsq41():
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    dm21 = torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE)
    dm3l = torch.tensor(2.517e-3, device=DEVICE, dtype=DTYPE)
    dm41 = torch.tensor(1.7, device=DEVICE, dtype=DTYPE)
    spectrum = MassSpectrum_BSM(DeltamSq21=dm21, DeltamSq3l=dm3l, DeltamSq41=dm41)

    mass_vector = spectrum.difference_vector(context=ctx)
    expected = torch.tensor([0.0, 7.42e-5, 2.517e-3, 1.7], device=DEVICE, dtype=DTYPE)

    assert mass_vector.shape == (4,)
    assert_close(mass_vector, expected, name="sterile mass-squared vector")


def test_mass_spectrum_bsm_missing_deltamsq41_raises():
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    dm21 = torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE)
    dm3l = torch.tensor(2.517e-3, device=DEVICE, dtype=DTYPE)
    spectrum = MassSpectrum_BSM(DeltamSq21=dm21, DeltamSq3l=dm3l)

    with pytest.raises(ValueError, match="requires DeltamSq41"):
        spectrum.difference_vector(context=ctx)
