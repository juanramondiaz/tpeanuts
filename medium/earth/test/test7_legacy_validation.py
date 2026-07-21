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
Validation of tpeanuts.medium.earth against the original NumPy ``peanuts``
implementation (treated as a read-only reference in ``peanuts/``).

Combines the legacy backup scripts ``test8_integration.py`` (earth_probability_exposure
vs legacy) and ``test9_validation_legacy.py`` (earth_probability_state vs legacy) into a single
sanity check, since both require the same expensive legacy-object setup and
report on the same underlying physics (the Earth matter-regeneration
probability). The diagnostic plots from those historical backup tests live in
notebooks; this file keeps only the numerical comparison.

Note on the exposure-integration comparison: legacy ``Pearth_integrated``
(and this file's own manual reference sum) accumulate
``sum_i Pearth(eta_i) * W(eta_i) * deta``. The legacy code hardcodes
``deta = pi / ns``, which does not match the actual point spacing of its own
uniform ``eta`` grid (``pi / (ns - 1)``) -- the same quadrature bug found and
fixed in ``tpeanuts.medium.earth.exposure_integration.earth_probability_exposure``
while building this test suite (see test6_exposure.py). To isolate the
*physics* comparison (does torch ``Pearth`` match legacy ``Pearth``
pointwise?) from that shared quadrature artifact, both sides of the
integrated comparison below use the true grid spacing ``eta[1] - eta[0]``
rather than either legacy convention.

Note on matter-potential precision: legacy peanuts always uses its own
hardcoded charged-current prefactor. ``earth_probability_state``/``earth_probability_state_analytical`` expose
this as ``legacy_precision=True``, which Part 1 below uses for a tight,
apples-to-apples comparison. ``earth_probability_exposure`` has no such passthrough, so
Part 2's tolerance is widened to absorb the resulting ~1e-5 full-precision-vs-
legacy-precision gap rather than pretending it is not there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.medium.earth.exposure_integration import earth_probability_exposure
from tpeanuts.medium.earth.exposure_table import ExposureParameters
from tpeanuts.medium.earth.probability import earth_probability_state
from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile
from tpeanuts.util.context import RuntimeContext

peanuts = pytest.importorskip("peanuts", reason="legacy peanuts reference package not available")
import peanuts.earth as legacy_earth  # noqa: E402
import peanuts.pmns as legacy_pmns_module  # noqa: E402
import peanuts.time_average as legacy_time_average  # noqa: E402


DTYPE = torch.float64
CDTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PACKAGE_DIR = Path(__file__).resolve().parents[3]
LEGACY_DENSITY_FILE = PACKAGE_DIR / "data" / "peanuts" / "Earth_Density.csv"

THETA12, THETA13, THETA23, DELTA_CP = 0.59, 0.15, 0.78, 1.20
DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3
DEPTH_SURFACE_M = 0.0

FLAVOUR_NAMES = ["nu_e", "nu_mu", "nu_tau"]

# Pointwise earth_probability_state-vs-Pearth comparison: Case A (through-Earth) trajectories.
ENERGY_MEV = 1000.0
ETA_VALUES = [0.10, 0.35, 0.60, 0.85, 1.10, 1.35]
ABS_TOL_POINTWISE = 2.0e-6
REL_TOL_POINTWISE = 2.0e-5
SUM_TOL_RAW = 1.0e-3

# Exposure-integrated earth_probability_exposure-vs-manual-legacy-sum comparison.
LATITUDE_RAD = 0.72
D1, D2, NS_INTEGRATED = 0.0, 365.0, 51
INTEGRATED_ENERGIES_MEV = [1000.0, 3000.0]
# earth_probability_exposure has no legacy_precision passthrough (unlike earth_probability_state itself),
# so this comparison inherits the ~1e-5 full-precision-vs-legacy-precision
# matter-potential gap quantified and explained in Part 1 below.
ABS_TOL_INTEGRATED = 3.0e-5


def _flavour_state_numpy(index: int) -> np.ndarray:
    state = np.zeros(3, dtype=np.complex128)
    state[index] = 1.0 + 0.0j
    return state


def _flavour_state_torch(index: int) -> torch.Tensor:
    state = torch.zeros(3, device=DEVICE, dtype=CDTYPE)
    state[index] = 1.0 + 0.0j
    return state


def _build_torch_objects(ctx: RuntimeContext):
    profile = EarthProfile(params=EarthParameters(profile_perturbative_name="even_power"), context=ctx)
    pmns = PMNS_SM(PMNSParams(theta12=THETA12, theta13=THETA13, theta23=THETA23, delta=DELTA_CP, context=ctx))
    mass_spectrum = MassSpectrum_SM(
        DeltamSq21=torch.as_tensor(DM21_EV2, device=ctx.device, dtype=ctx.dtype),
        DeltamSq3l=torch.as_tensor(DM3L_EV2, device=ctx.device, dtype=ctx.dtype),
    )
    oscillation = OscillationParameters(pmns=pmns, mass_spectrum=mass_spectrum, antinu=False)
    return profile, oscillation


def _build_legacy_objects():
    density = legacy_earth.earthdensity(str(LEGACY_DENSITY_FILE))
    pmns = legacy_pmns_module.PMNS(THETA12, THETA13, THETA23, DELTA_CP)
    return density, pmns


def test_earth_probability_state_and_earth_probability_exposure_match_legacy_peanuts():
    assert LEGACY_DENSITY_FILE.is_file(), f"legacy density file not found: {LEGACY_DENSITY_FILE}"

    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    torch_profile, torch_oscillation = _build_torch_objects(ctx)
    legacy_density, legacy_pmns = _build_legacy_objects()

    # ------------------------------------------------------------------
    # Part 1: pointwise earth_probability_state (flavour basis, no reunitarize) vs legacy Pearth.
    # ------------------------------------------------------------------
    torch_probs = np.zeros((len(FLAVOUR_NAMES), len(ETA_VALUES), 3), dtype=float)
    legacy_probs = np.zeros_like(torch_probs)

    for flavour_index in range(len(FLAVOUR_NAMES)):
        state_np = _flavour_state_numpy(flavour_index)
        state_t = _flavour_state_torch(flavour_index)

        for eta_index, eta_value in enumerate(ETA_VALUES):
            legacy_p = legacy_earth.Pearth(
                state_np, legacy_density, legacy_pmns, DM21_EV2, DM3L_EV2,
                ENERGY_MEV, float(eta_value), DEPTH_SURFACE_M,
                mode="analytical", massbasis=False, antinu=False,
            )
            torch_p = earth_probability_state(
                state_t, torch_profile, torch_oscillation,
                torch.tensor(ENERGY_MEV, device=DEVICE, dtype=DTYPE),
                torch.tensor(float(eta_value), device=DEVICE, dtype=DTYPE),
                DEPTH_SURFACE_M, method="analytical", massbasis=False, reunitarize=False,
                legacy_precision=True,
            )

            legacy_probs[flavour_index, eta_index] = np.asarray(legacy_p, dtype=float)
            torch_probs[flavour_index, eta_index] = torch_p.detach().cpu().numpy()

    pointwise_abs_diff = np.abs(torch_probs - legacy_probs)
    pointwise_rel_diff = pointwise_abs_diff / np.maximum(np.abs(legacy_probs), 1.0e-15)
    torch_sum_error = np.max(np.abs(np.sum(torch_probs, axis=-1) - 1.0))
    legacy_sum_error = np.max(np.abs(np.sum(legacy_probs, axis=-1) - 1.0))

    assert np.all(np.isfinite(torch_probs)), "torch pointwise probabilities must be finite"
    assert np.all(np.isfinite(legacy_probs)), "legacy pointwise probabilities must be finite"
    assert np.all(torch_probs >= -1.0e-12), "torch pointwise probabilities must be non-negative"
    assert np.all(legacy_probs >= -1.0e-12), "legacy pointwise probabilities must be non-negative"
    assert np.max(pointwise_abs_diff) < ABS_TOL_POINTWISE, (
        f"pointwise earth_probability_state vs legacy Pearth exceeds absolute tolerance: {np.max(pointwise_abs_diff):.3e}"
    )
    assert np.max(pointwise_rel_diff) < REL_TOL_POINTWISE, (
        f"pointwise earth_probability_state vs legacy Pearth exceeds relative tolerance: {np.max(pointwise_rel_diff):.3e}"
    )
    assert torch_sum_error < SUM_TOL_RAW, "raw (non-reunitarized) torch normalization drift must stay small"
    assert legacy_sum_error < SUM_TOL_RAW, "raw legacy normalization drift must stay small"
    assert abs(torch_sum_error - legacy_sum_error) < 1.0e-10, (
        "torch and legacy raw normalization drift must match (same underlying perturbative algorithm)"
    )

    # ------------------------------------------------------------------
    # Part 2: exposure-integrated earth_probability_exposure vs a manual legacy sum,
    # both using the true (non-buggy) grid spacing -- see module docstring.
    # ------------------------------------------------------------------
    exposure_np = legacy_time_average.NadirExposure(
        lam=LATITUDE_RAD, d1=D1, d2=D2, ns=NS_INTEGRATED,
        normalized=False, from_file=None, angle="Nadir", daynight=None,
    )
    eta_np = np.asarray(exposure_np)[:, 0]
    w_np = np.asarray(exposure_np)[:, 1]
    deta_true = float(eta_np[1] - eta_np[0])

    legacy_exposure = ExposureParameters(
        exposure_source="legacy", detector_latitude_rad=LATITUDE_RAD,
        exposure_d1=D1, exposure_d2=D2, exposure_ns=NS_INTEGRATED, exposure_use_cache=False,
    )

    integrated_abs_diff = []

    for flavour_index in range(2):
        state_np = _flavour_state_numpy(flavour_index)
        state_t = _flavour_state_torch(flavour_index)

        for energy_mev in INTEGRATED_ENERGIES_MEV:
            legacy_integrated = np.zeros(3, dtype=float)
            for eta_value, weight in zip(eta_np, w_np):
                legacy_integrated += np.asarray(
                    legacy_earth.Pearth(
                        state_np, legacy_density, legacy_pmns, DM21_EV2, DM3L_EV2,
                        float(energy_mev), float(eta_value), DEPTH_SURFACE_M,
                        mode="analytical", massbasis=False, antinu=False,
                    ),
                    dtype=float,
                ) * float(weight) * deta_true

            torch_integrated = earth_probability_exposure(
                state_t, torch_profile, torch_oscillation,
                torch.tensor(energy_mev, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M,
                method="analytical", massbasis=False, exposure=legacy_exposure,
                context=ctx, reunitarize=False, normalized_exposure=False,
            ).detach().cpu().numpy()

            diff = np.max(np.abs(torch_integrated - legacy_integrated))
            integrated_abs_diff.append(diff)

            assert np.all(np.isfinite(torch_integrated)), "integrated torch probabilities must be finite"
            assert diff < ABS_TOL_INTEGRATED, (
                f"exposure-integrated earth_probability_exposure vs legacy exceeds tolerance "
                f"(flavour={FLAVOUR_NAMES[flavour_index]}, E={energy_mev} MeV): {diff:.3e}"
            )

    assert max(integrated_abs_diff) < ABS_TOL_INTEGRATED
