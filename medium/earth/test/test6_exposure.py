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
Pytest-compatible tests for the tpeanuts.medium.earth ``exposure_*`` modules:
exposure_math, exposure_table, exposure_io, and exposure_integration.

The diagnostic plots from the historical backup tests live in notebooks; this
file keeps only fast numerical sanity checks that can run automatically.
"""

from __future__ import annotations

import math

import pytest
import torch

from tpeanuts.medium.earth.exposure_io import (
    _convert_csv_angle_mode,
    nadir_exposure_from_cache,
    nadir_exposure_from_csv,
    save_nadir_exposure_to_cache,
)
from tpeanuts.medium.earth.exposure_math import (
    IndefiniteIntegralDay,
    IntegralAngle,
    IntegralDay,
    csqrt,
    make_eta_grid,
)
from tpeanuts.medium.earth.exposure_table import (
    ExposureParameters,
    NadirExposureTable,
    build_nadir_exposure,
    integrate_exposure,
    nadir_exposure_from_math,
    prepare_nadir_exposure,
)
from tpeanuts.medium.earth.exposure_integration import _prepare_energy_grid, pearth_integrated
from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close, build_pmns


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATITUDE_RAD = 0.72
DEPTH_SURFACE_M = 0.0


def _oscillation() -> OscillationParameters:
    return OscillationParameters(
        pmns=build_pmns(),
        DeltamSq21=torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE),
        DeltamSq3l=torch.tensor(2.517e-3, device=DEVICE, dtype=DTYPE),
        antinu=False,
    )


def _two_shell_profile() -> EarthProfile:
    """Synthetic two-shell profile: constant density 2.0 for r<0.5, 1.0 for 0.5<r<=1.0."""
    rj = torch.tensor([0.5, 1.0], device=DEVICE, dtype=DTYPE)
    coefficients = torch.tensor(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    params = EarthParameters(
        profile_perturbative_name="even_power",
        profile_perturbative_kwargs={"rj": rj, "coefficients": coefficients},
    )
    return EarthProfile(params=params, context=RuntimeContext.resolve(DEVICE, DTYPE))


def _synthetic_table() -> NadirExposureTable:
    eta = torch.linspace(0.0, math.pi, 21, device=DEVICE, dtype=DTYPE)
    exposure = 0.25 + torch.sin(eta) ** 2
    return NadirExposureTable(eta=eta, exposure=exposure)


# ============================================================
# exposure_math
# ============================================================


def test_make_eta_grid_full_and_daynight_slices():
    eta_full = make_eta_grid(9, daynight=None, device=DEVICE, dtype=DTYPE)
    eta_day = make_eta_grid(9, daynight="day", device=DEVICE, dtype=DTYPE)
    eta_night = make_eta_grid(9, daynight="night", device=DEVICE, dtype=DTYPE)

    assert eta_full.shape == (9,)
    assert eta_day.shape == (4,)
    assert eta_night.shape == (4,)


def test_make_eta_grid_endpoints_and_sorted():
    eta_full = make_eta_grid(11, daynight=None, device=DEVICE, dtype=DTYPE)

    assert_close(eta_full[0], torch.tensor(0.0, dtype=DTYPE), atol=1.0e-14, rtol=1.0e-14, name="eta grid starts at 0")
    assert_close(eta_full[-1], torch.tensor(math.pi, dtype=DTYPE), name="eta grid ends at pi")
    assert torch.all(torch.diff(eta_full) > 0)


def test_csqrt_real_and_negative_branch():
    positive = csqrt(torch.tensor(4.0, device=DEVICE, dtype=DTYPE))
    negative = csqrt(torch.tensor(-4.0, device=DEVICE, dtype=DTYPE))

    assert torch.is_complex(positive) and torch.is_complex(negative)
    assert_close(positive.real, torch.tensor(2.0, dtype=DTYPE), name="sqrt(4) real part")
    assert_close(positive.imag, torch.tensor(0.0, dtype=DTYPE), atol=1.0e-12, rtol=1.0e-12, name="sqrt(4) imag part")
    assert_close(negative.real, torch.tensor(0.0, dtype=DTYPE), atol=1.0e-12, rtol=1.0e-12, name="sqrt(-4) real part")
    assert_close(negative.imag, torch.tensor(2.0, dtype=DTYPE), name="sqrt(-4) imag part (principal branch)")


def test_indefinite_integral_day_is_finite_complex():
    T = torch.tensor(0.15, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(1.00, device=DEVICE, dtype=DTYPE)
    lam = torch.tensor(LATITUDE_RAD, device=DEVICE, dtype=DTYPE)

    value = IndefiniteIntegralDay(T, eta, lam, device=DEVICE, dtype=DTYPE)

    assert torch.is_complex(value)
    assert torch.isfinite(value.real) and torch.isfinite(value.imag)


def test_integral_angle_scalar_shape_and_finite():
    eta = torch.tensor(1.10, device=DEVICE, dtype=DTYPE)
    lam = torch.tensor(LATITUDE_RAD, device=DEVICE, dtype=DTYPE)

    weight = IntegralAngle(eta, lam, a1=0.20, a2=2.80, device=DEVICE, dtype=DTYPE)

    assert weight.shape == ()
    assert torch.isfinite(weight)


def test_integral_angle_invalid_bounds_raises():
    eta = torch.tensor(1.10, device=DEVICE, dtype=DTYPE)
    lam = torch.tensor(LATITUDE_RAD, device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError):
        IntegralAngle(eta, lam, a1=2.0, a2=1.0, device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError):
        IntegralAngle(eta, lam, a1=-0.1, a2=1.0, device=DEVICE, dtype=DTYPE)


def test_integral_day_invalid_d1_d2_raises():
    eta = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)
    lam = torch.tensor(LATITUDE_RAD, device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError):
        IntegralDay(eta, lam, d1=300.0, d2=20.0, device=DEVICE, dtype=DTYPE)


def test_integral_day_full_year_at_least_subwindow():
    eta = torch.tensor(1.10, device=DEVICE, dtype=DTYPE)
    lam = torch.tensor(LATITUDE_RAD, device=DEVICE, dtype=DTYPE)

    full_year = IntegralDay(eta, lam, d1=0.0, d2=365.0, device=DEVICE, dtype=DTYPE)
    sub_window = IntegralDay(eta, lam, d1=100.0, d2=200.0, device=DEVICE, dtype=DTYPE)

    assert torch.isfinite(full_year) and torch.isfinite(sub_window)
    assert float(full_year) >= float(sub_window) - 1.0e-12


# ============================================================
# exposure_table
# ============================================================


def test_nadir_exposure_table_device_dtype_properties():
    table = _synthetic_table()

    assert table.device.type == DEVICE.type
    assert table.dtype == DTYPE


def test_nadir_exposure_table_normalize_integrates_to_one():
    table = _synthetic_table()
    table.normalize_()

    integral = torch.trapezoid(table.exposure, x=table.eta)
    assert_close(integral, torch.tensor(1.0, dtype=DTYPE), atol=1.0e-10, rtol=1.0e-10, name="normalized exposure integrates to one")


def test_nadir_exposure_table_interp_matches_nodes():
    table = _synthetic_table()
    interpolated = table.interp(table.eta)

    assert_close(interpolated, table.exposure, name="interpolation at stored nodes reproduces stored values")


def test_nadir_exposure_table_interp_clamps_outside_range():
    table = _synthetic_table()
    query = torch.tensor([-1.0, math.pi + 1.0], device=DEVICE, dtype=DTYPE)

    interpolated = table.interp(query)

    assert_close(interpolated[0], table.exposure[0], name="below-range query clamps to left endpoint")
    assert_close(interpolated[1], table.exposure[-1], name="above-range query clamps to right endpoint")


def test_integrate_exposure_matches_manual_trapz():
    table = _synthetic_table()
    table.normalize_()

    probabilities_eta = torch.full((table.eta.numel(), 3), 0.3, device=DEVICE, dtype=DTYPE)
    result = integrate_exposure(probabilities_eta, table.eta, table.exposure)
    expected = torch.trapezoid(probabilities_eta * table.exposure[:, None], x=table.eta, dim=-2)

    assert_close(result, expected, name="integrate_exposure matches manual trapezoid formula")

    probabilities_batched = probabilities_eta[None, :, :].expand(2, -1, -1)
    result_batched = integrate_exposure(probabilities_batched, table.eta, table.exposure)
    assert result_batched.shape == (2, 3)
    assert_close(result_batched[0], expected, name="batched leading dim preserved")


def test_integrate_exposure_rejects_mismatched_shapes():
    table = _synthetic_table()
    probabilities_eta = torch.full((table.eta.numel(), 3), 0.3, device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError):
        integrate_exposure(probabilities_eta, table.eta, table.exposure[:-1])

    with pytest.raises(ValueError):
        integrate_exposure(probabilities_eta[:-1], table.eta, table.exposure)


def test_nadir_exposure_from_math_shapes_and_finiteness():
    exposure = ExposureParameters(detector_latitude_rad=LATITUDE_RAD, exposure_ns=7, exposure_d1=0.0, exposure_d2=365.0)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)

    eta, values = nadir_exposure_from_math(exposure=exposure, context=ctx)

    assert eta.shape == (7,)
    assert values.shape == (7,)
    assert torch.all(torch.isfinite(values))


def test_build_nadir_exposure_math_without_cache():
    exposure = ExposureParameters(detector_latitude_rad=LATITUDE_RAD, exposure_ns=7, exposure_use_cache=False)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)

    table = build_nadir_exposure(exposure=exposure, context=ctx, normalized=False)

    assert isinstance(table, NadirExposureTable)
    assert table.eta.shape == (7,)
    assert table.exposure.shape == (7,)
    assert torch.all(torch.isfinite(table.exposure))


def test_build_nadir_exposure_requires_latitude_for_math():
    exposure = ExposureParameters(detector_latitude_rad=None, exposure_source="math", exposure_use_cache=False)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)

    with pytest.raises(ValueError, match="detector_latitude_rad"):
        build_nadir_exposure(exposure=exposure, context=ctx)


def test_build_nadir_exposure_rejects_invalid_source():
    exposure = ExposureParameters(exposure_source="bogus", exposure_use_cache=False)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)

    with pytest.raises(ValueError, match="exposure_source"):
        build_nadir_exposure(exposure=exposure, context=ctx)


def test_prepare_nadir_exposure_explicit_eta_uniform():
    eta = torch.linspace(0.0, math.pi, 15, device=DEVICE, dtype=DTYPE)
    exposure = ExposureParameters()
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)

    eta_grid, exposure_weights, meta = prepare_nadir_exposure(eta, exposure=exposure, context=ctx)

    assert_close(eta_grid, eta, atol=0.0, rtol=0.0, name="explicit eta grid passed through unchanged")
    integral = torch.trapezoid(exposure_weights, x=eta_grid)
    assert_close(integral, torch.tensor(1.0, dtype=DTYPE), atol=1.0e-10, rtol=1.0e-10, name="uniform exposure normalized")
    assert meta["source"] == "user_eta_uniform"


# ============================================================
# exposure_io
# ============================================================


def test_cache_roundtrip_save_and_load(tmp_path):
    eta = torch.linspace(0.0, math.pi, 9, device=DEVICE, dtype=DTYPE)
    exposure = 0.25 + torch.sin(eta) ** 2
    cache_dir = str(tmp_path)

    save_nadir_exposure_to_cache(eta, exposure, lam_rad=LATITUDE_RAD, d1=0.0, d2=365.0, ns=9, daynight=None, cache_dir=cache_dir)
    eta_loaded, exposure_loaded = nadir_exposure_from_cache(
        lam_rad=LATITUDE_RAD, d1=0.0, d2=365.0, ns=9, daynight=None,
        cache_dir=cache_dir, device=DEVICE, dtype=DTYPE,
    )

    assert_close(eta_loaded, eta, name="cached eta round-trips")
    assert_close(exposure_loaded, exposure, name="cached exposure round-trips")


def test_nadir_exposure_from_cache_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        nadir_exposure_from_cache(
            lam_rad=LATITUDE_RAD, d1=0.0, d2=365.0, ns=9, daynight=None,
            cache_dir=str(tmp_path), device=DEVICE, dtype=DTYPE,
        )


def test_convert_csv_angle_mode_nadir_zenith_coszenith():
    eta = make_eta_grid(5, daynight=None, device=DEVICE, dtype=DTYPE)
    raw = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE, dtype=DTYPE)

    nadir = _convert_csv_angle_mode(raw, eta, angle="Nadir", dtype=DTYPE)
    zenith = _convert_csv_angle_mode(raw, eta, angle="Zenith", dtype=DTYPE)
    cos_zenith = _convert_csv_angle_mode(raw, eta, angle="CosZenith", dtype=DTYPE)

    assert_close(nadir, raw, name="Nadir mode is a passthrough")
    assert_close(zenith, torch.flip(raw, dims=(0,)), name="Zenith mode reverses the exposure array")
    assert cos_zenith.shape == raw.shape
    assert torch.all(torch.isfinite(cos_zenith))
    assert torch.all(cos_zenith >= 0.0)


def test_nadir_exposure_from_csv_daynight_slicing(tmp_path):
    csv_path = tmp_path / "exposure.csv"
    values = torch.linspace(1.0, 2.0, 9, dtype=DTYPE)
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("Exposure\n")
        for value in values.tolist():
            handle.write(f"{value}\n")

    eta_full, exposure_full = nadir_exposure_from_csv(str(csv_path), angle="Nadir", daynight=None, device=DEVICE, dtype=DTYPE)
    eta_night, exposure_night = nadir_exposure_from_csv(str(csv_path), angle="Nadir", daynight="night", device=DEVICE, dtype=DTYPE)

    assert eta_full.shape == (9,)
    assert exposure_full.shape == (9,)
    assert eta_night.shape == (4,)
    assert exposure_night.shape == (4,)


def test_nadir_exposure_from_csv_missing_column_raises(tmp_path):
    csv_path = tmp_path / "bad_exposure.csv"
    csv_path.write_text("NotExposure\n1.0\n2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Exposure"):
        nadir_exposure_from_csv(str(csv_path), angle="Nadir", daynight=None, device=DEVICE, dtype=DTYPE)


# ============================================================
# exposure_integration
# ============================================================


def test_prepare_energy_grid_scalar_and_vector():
    E_scalar, squeeze_scalar = _prepare_energy_grid(1000.0, device=DEVICE, dtype=DTYPE)
    E_vector, squeeze_vector = _prepare_energy_grid(
        torch.tensor([800.0, 2000.0], device=DEVICE, dtype=DTYPE), device=DEVICE, dtype=DTYPE
    )

    assert E_scalar.shape == (1,)
    assert squeeze_scalar is True
    assert E_vector.shape == (2,)
    assert squeeze_vector is False


def test_pearth_integrated_scalar_energy_sums_to_one():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    exposure = ExposureParameters(detector_latitude_rad=0.5, exposure_ns=9, exposure_use_cache=False)

    P_int = pearth_integrated(
        weights, profile, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M,
        method="analytical", massbasis=True, exposure=exposure, context=ctx, normalized_exposure=True,
    )

    assert P_int.shape == (3,)
    assert_close(torch.sum(P_int), torch.tensor(1.0, dtype=DTYPE), atol=1.0e-10, rtol=1.0e-10,
                 name="normalized-exposure-averaged probabilities sum to one")


def test_pearth_integrated_vector_energy_preserves_dimension():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([800.0, 2000.0], device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    exposure = ExposureParameters(detector_latitude_rad=0.5, exposure_ns=9, exposure_use_cache=False)

    P_int = pearth_integrated(
        weights, profile, oscillation, E, DEPTH_SURFACE_M,
        method="analytical", massbasis=True, exposure=exposure, context=ctx,
    )

    assert P_int.shape == (2, 3)
    assert torch.all(torch.isfinite(P_int))


def test_pearth_integrated_chunk_eta_matches_full_batch():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    exposure = ExposureParameters(detector_latitude_rad=0.5, exposure_ns=9, exposure_use_cache=False)

    P_full = pearth_integrated(
        weights, profile, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M,
        method="analytical", massbasis=True, exposure=exposure, context=ctx, chunk_eta=None,
    )
    P_chunked = pearth_integrated(
        weights, profile, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M,
        method="analytical", massbasis=True, exposure=exposure, context=ctx, chunk_eta=3,
    )

    assert_close(P_chunked, P_full, atol=1.0e-10, rtol=1.0e-10, name="chunked eta integration matches full-batch result")


def test_pearth_integrated_rejects_invalid_method():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    exposure = ExposureParameters(detector_latitude_rad=0.5, exposure_ns=9, exposure_use_cache=False)

    with pytest.raises(ValueError, match="method must be either"):
        pearth_integrated(
            weights, profile, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M,
            method="bogus", massbasis=True, exposure=exposure, context=ctx,
        )
