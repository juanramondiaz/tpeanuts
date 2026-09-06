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

"""Pytest-compatible tests for SNODayNightModel's solar-source validation."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.detector.sno.inference_model import SNODayNightModel
from tpeanuts.inference.model_solar import SolarSMOscillationModel
from tpeanuts.medium.solar.profile import build_solar_medium
from tpeanuts.source.solar import build_solar_source
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _base_kwargs(source_name: str) -> dict:
    context = RuntimeContext.resolve(DEVICE, DTYPE)
    oscillation_model, _ = SolarSMOscillationModel.from_preset(
        "_SM_NUFIT52_NO", free=("theta12",), context=context,
    )
    medium = build_solar_medium(None, context=context)
    source = build_solar_source(None, context=context)
    eta = torch.tensor([0.3, 0.6], dtype=DTYPE)
    weight = torch.tensor([0.5, 0.5], dtype=DTYPE)
    return dict(
        oscillation_model=oscillation_model,
        solar_medium=medium,
        solar_source=source,
        earth_profile=None,
        source_names=(source_name,),
        bin_edges_MeV=torch.linspace(4.0, 10.0, 4, dtype=DTYPE),
        eta_day=eta,
        weight_day=weight,
        eta_night=eta,
        weight_night=weight,
        exposure_days_day=100.0,
        exposure_days_night=100.0,
    )


def test_rejects_line_spectrum_source_with_clear_error():
    # SNO's own true-energy grid starts at 1.0 MeV and its real analysis
    # threshold is several MeV higher, above every solar line (7Be/pep) --
    # this model never learned to fold a discrete spectrum (unlike
    # BorexinoEventRateModel), so it must fail fast and explain why rather
    # than crash deep inside SolarNeutrinoSource.spectrum() with an opaque
    # TypeError.
    with pytest.raises(ValueError, match="does not support line-spectrum"):
        SNODayNightModel(**_base_kwargs("7Be"))


def test_accepts_continuous_spectrum_source():
    model = SNODayNightModel(**_base_kwargs("8B"))
    assert model.source_names == ("8B",)
