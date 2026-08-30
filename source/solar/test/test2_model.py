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

"""Pytest-compatible tests for tpeanuts.source.solar.model (SolarNeutrinoSource).

Density/composition (medium) tests live in
``medium.solar.test.test2_profile`` -- see that module.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

import tpeanuts.config.default as default
from tpeanuts.source.solar.io import load_solar_production
from tpeanuts.source.solar.model import (
    ContinuousSolarSpectrum,
    SolarLineSpectrum,
    SolarNeutrinoSource,
    SolarSourceParameters,
    build_solar_source,
)
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_same_device(actual: torch.device, expected: torch.device) -> None:
    assert actual.type == torch.device(expected).type


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_source(*, dtype: torch.dtype = DTYPE, production_measure: str = "radial_pdf") -> SolarNeutrinoSource:
    device = DEVICE
    radius = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device, dtype=dtype)
    fractions = {
        "pp": torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0], device=device, dtype=dtype),
        "8B": torch.tensor([4.0, 2.0, 0.5, 0.0, 0.0], device=device, dtype=dtype),
        "hep": torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0], device=device, dtype=dtype),
    }
    fluxes = {
        "pp": torch.tensor(6.0e10, device=device, dtype=dtype),
        "8B": torch.tensor(4.5e6, device=device, dtype=dtype),
        "hep": torch.tensor(8.0e3, device=device, dtype=dtype),
    }
    return SolarNeutrinoSource(
        production_radius=radius,
        fractions=fractions, fluxes=fluxes,
        production_measure=production_measure,
    )


def test_solar_source_device_dtype_properties_and_string_summary():
    source = make_source()
    summary = str(source)

    assert_same_device(source.device, DEVICE)
    assert source.dtype == DTYPE
    assert "SolarNeutrinoSource" in summary
    assert "sources=" in summary


def test_production_distribution_returns_grid_values_and_interpolates_query_points():
    source = make_source()
    query = torch.tensor([0.125, 0.375], device=DEVICE, dtype=DTYPE)

    grid_fraction = source.production_distribution("pp")
    query_fraction = source.production_distribution("pp", query)

    assert_close(grid_fraction, source.fractions["pp"], name="production fraction on grid")
    assert_close(query_fraction, torch.tensor([0.5, 1.5], device=DEVICE, dtype=DTYPE), name="production fraction interpolation")


def test_production_distribution_unknown_source_raises():
    source = make_source()

    with pytest.raises(KeyError, match="Unknown solar source"):
        source.production_distribution("unknown")


def test_production_distribution_single_and_multiple_sources():
    source = make_source()

    single = source.production_distribution("pp")
    multiple = source.production_distribution(["pp", "8B"])

    assert single.shape == source.production_radius.shape
    assert multiple.shape == (2, source.production_radius.numel())
    assert_close(multiple[0], source.fractions["pp"], name="stacked pp")
    assert_close(multiple[1], source.fractions["8B"], name="stacked 8B")


def test_radial_pdf_is_normalized_and_nonnegative_on_construction():
    source = make_source()

    normalized = source.production_distribution("pp")
    area = torch.trapz(normalized, x=source.production_radius)

    assert bool(torch.all(normalized >= 0.0))
    assert_close(area, torch.tensor(1.0, device=DEVICE, dtype=DTYPE), name="normalized production fraction area")


def test_shell_fraction_is_normalized_by_sum_on_construction():
    # A "shell_fraction" source's stored distribution should sum (plain
    # sum) to 1 rather than trapezoidal-integrate to 1 -- treating discrete
    # per-shell weights as a continuous density would reintroduce the same
    # shell-vs-density mismatch the production_measure branch exists to
    # avoid (see SolarNeutrinoSource.mass_weights_integrate).
    source = make_source(production_measure="shell_fraction")
    normalized = source.production_distribution("pp")

    assert bool(torch.all(normalized >= 0.0))
    assert_close(
        normalized.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        name="shell_fraction normalized sum",
    )


def test_production_distribution_clamps_roundoff_negative_before_normalizing():
    source = make_source()
    source.fractions["pp"] = torch.tensor(
        [-1.0e-12, 1.0, 2.0, 1.0, 0.0], device=DEVICE, dtype=DTYPE
    )
    source.__post_init__()
    distribution = source.production_distribution("pp")
    assert bool(torch.all(distribution >= 0.0))
    assert_close(
        torch.trapezoid(distribution, x=source.production_radius),
        torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        name="sanitized radial PDF normalization",
    )


def test_construction_rejects_unknown_production_measure():
    with pytest.raises(ValueError, match="production_measure"):
        make_source(production_measure="bogus")


def _make_two_shell_source(*, production_measure: str) -> SolarNeutrinoSource:
    # Two production shells at very different radial spacing, so
    # shell_fraction (discrete sum) and radial_pdf (trapz) genuinely
    # disagree on this source.
    radius = torch.tensor([0.0, 0.001, 1.0], device=DEVICE, dtype=DTYPE)
    fractions = torch.tensor([0.0, 0.5, 0.5], device=DEVICE, dtype=DTYPE)
    return SolarNeutrinoSource(
        production_radius=radius,
        fractions={"pp": fractions}, fluxes={"pp": torch.tensor(1.0, device=DEVICE, dtype=DTYPE)},
        production_measure=production_measure,
    )


def test_mass_weights_integrate_shell_fraction_ignores_grid_spacing():
    # For "shell_fraction" data (e.g. Bahcall) the reduction must not be
    # reweighted by the (arbitrary) local grid spacing: each tabulated
    # fraction already carries its own shell's full share of production, so
    # the result must equal the plain fraction-weighted average regardless
    # of how close together the shells are sampled.
    source = _make_two_shell_source(production_measure="shell_fraction")
    fractions = source.production_distribution("pp")
    weights_r = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], device=DEVICE, dtype=DTYPE,
    )

    result = source.mass_weights_integrate(weights_r, fractions, energy_ndim=0)

    expected = torch.tensor([0.5, 0.5], device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(result, expected, rtol=1e-14, atol=1e-14)


def test_mass_weights_integrate_radial_pdf_matches_manual_trapz():
    source = _make_two_shell_source(production_measure="radial_pdf")
    fractions = source.production_distribution("pp")
    weights_r = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], device=DEVICE, dtype=DTYPE,
    )

    result = source.mass_weights_integrate(weights_r, fractions, energy_ndim=0)

    weighted = weights_r * fractions[:, None]
    expected = torch.trapz(weighted, x=source.production_radius, dim=0)
    torch.testing.assert_close(result, expected, rtol=1e-14, atol=1e-14)

    # Sanity check that the two measures genuinely disagree on this
    # deliberately non-uniform grid (otherwise the two tests above would
    # not actually be distinguishing anything).
    shell_source = _make_two_shell_source(production_measure="shell_fraction")
    shell_result = shell_source.mass_weights_integrate(
        weights_r, shell_source.production_distribution("pp"), energy_ndim=0,
    )
    assert not torch.allclose(result, shell_result)


def test_production_distribution_rejects_significant_negative_and_nonfinite():
    radius = torch.tensor([0.0, 0.5, 1.0], device=DEVICE, dtype=DTYPE)
    fluxes = {"pp": torch.tensor(1.0, device=DEVICE, dtype=DTYPE)}
    with pytest.raises(ValueError, match="significant negative"):
        SolarNeutrinoSource(
            radius,
            {"pp": torch.tensor([0.0, -1.0e-3, 1.0], device=DEVICE, dtype=DTYPE)}, fluxes
        )
    with pytest.raises(ValueError, match="non-finite"):
        SolarNeutrinoSource(
            radius,
            {"pp": torch.tensor([0.0, float("nan"), 1.0], device=DEVICE, dtype=DTYPE)}, fluxes
        )


def test_flux_returns_scalar_and_unknown_source_raises():
    source = make_source()

    flux = source.total_flux("8B")

    assert flux.shape == ()
    assert_close(flux, torch.tensor(4.5e6, device=DEVICE, dtype=DTYPE), name="8B flux")
    with pytest.raises(KeyError, match="Unknown solar flux source"):
        source.total_flux("unknown")


def test_build_solar_source_returns_existing_source_when_context_matches():
    source = make_source()

    out = build_solar_source(source, context=RuntimeContext.resolve(source.device, source.dtype))

    assert out is source


def test_build_solar_source_casts_existing_source_to_requested_dtype():
    source = make_source(dtype=torch.float64)
    ctx = make_context(dtype=torch.float32)

    out = build_solar_source(source, context=ctx)

    assert out is not source
    assert out.production_radius.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in out.fractions.values())
    assert all(value.dtype == torch.float32 for value in out.fluxes.values())


def test_solar_source_default_loads_from_explicit_synthetic_paths(tmp_path):
    production_path = tmp_path / "production.csv"
    flux_path = tmp_path / "fluxes.csv"
    pd.DataFrame(
        {
            "radius": [0.0, 0.5, 1.0],
            "pp fraction": [0.0, 1.0, 0.0],
            "8B fraction": [1.0, 0.0, 0.0],
        }
    ).to_csv(production_path, index=False)
    pd.DataFrame(
        {
            "fraction": ["pp", "8B"],
            "flux": [6.0e10, 4.5e6],
        }
    ).to_csv(flux_path, index=False)

    params = SolarSourceParameters(
        production_path=str(production_path),
        fluxes_path=str(flux_path),
        spectrum_provider=None,
    )
    source = SolarNeutrinoSource.default(params=params, context=make_context())

    assert sorted(source.fractions) == ["8B", "pp"]
    assert sorted(source.fluxes) == ["8B", "pp"]
    # Explicit path override, no provider given: defaults to the historical
    # "radial_pdf" (trapz) convention (see load_solar_production).
    assert source.production_measure == "radial_pdf"
    # An explicit path always means the loaded table is not actually the
    # named provider's canonical file, so provenance must read "custom".
    assert source.production_provider == "custom"
    assert source.flux_provider == "custom"


def test_solar_source_default_records_custom_provenance_even_with_explicit_provider_name(tmp_path):
    # Regression test: passing both an explicit path AND a provider name
    # must still record "custom" provenance -- the provider name alone does
    # not make the loaded table the named provider's actual canonical file,
    # and the pre-fix code recorded the (misleading) named provider instead.
    production_path = tmp_path / "production.csv"
    flux_path = tmp_path / "fluxes.csv"
    pd.DataFrame(
        {"radius": [0.0, 1.0], "pp fraction": [1.0, 1.0]}
    ).to_csv(production_path, index=False)
    pd.DataFrame(
        {"fraction": ["pp"], "flux": [1.0]}
    ).to_csv(flux_path, index=False)

    source = SolarNeutrinoSource.default(
        params=SolarSourceParameters(
            production_provider="bahcall",
            flux_provider="bahcall",
            production_path=str(production_path),
            fluxes_path=str(flux_path),
            spectrum_provider=None,
        ),
        context=make_context(),
    )

    assert source.production_provider == "custom"
    assert source.flux_provider == "custom"


def test_solar_source_default_rejects_production_flux_source_mismatch(tmp_path):
    production_path = tmp_path / "production.csv"
    flux_path = tmp_path / "flux.csv"
    pd.DataFrame(
        {"radius": [0.0, 1.0], "pp fraction": [1.0, 1.0]}
    ).to_csv(production_path, index=False)
    pd.DataFrame(
        {"fraction": ["8B"], "flux": [1.0]}
    ).to_csv(flux_path, index=False)
    with pytest.raises(ValueError, match="production/flux source mismatch"):
        SolarNeutrinoSource.default(
            params=SolarSourceParameters(
                production_path=str(production_path),
                fluxes_path=str(flux_path),
                spectrum_provider=None,
            ),
            context=make_context(),
        )


def test_build_solar_source_loads_default_when_source_is_none():
    source = build_solar_source(None, context=make_context())

    assert {"pp", "8B", "7Be", "hep"}.issubset(source.fractions)
    assert {"pp", "8B", "7Be", "hep"}.issubset(source.fluxes)
    assert source.spectrum_provider == "legacy"
    assert source.has_spectrum("8B")
    assert source.has_spectrum("7Be")
    assert source.has_spectrum("pep")
    assert source.spectrum_table("pep").weights.sum() == 1.0
    assert source.spectrum_table("7Be").weights.sum() == 1.0
    # Must match whatever tpeanuts.config.default.solar_provider resolves to
    # (kept provider-agnostic so this test does not go stale if the
    # configured default provider changes).
    expected_measure = load_solar_production(
        provider=default.solar_provider, device=DEVICE, dtype=DTYPE,
    )["production_measure"]
    assert source.production_measure == expected_measure
    for distribution in source.fractions.values():
        assert bool(torch.all(distribution >= 0.0))
        normalization = (
            distribution.sum()
            if source.production_measure == "shell_fraction"
            else torch.trapezoid(distribution, x=source.production_radius)
        )
        assert_close(
            normalization,
            torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
            name="default production-distribution normalization",
        )


def test_continuous_spectrum_validates_and_normalizes_once():
    energy = torch.tensor([0.0, 1.0, 2.0], device=DEVICE, dtype=DTYPE)
    spectrum = ContinuousSolarSpectrum(
        energy, torch.tensor([0.0, 2.0, 0.0], device=DEVICE, dtype=DTYPE),
    )
    assert_close(
        torch.trapezoid(spectrum.density_MeV_inverse, x=energy),
        torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        name="continuous spectrum normalization",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        ContinuousSolarSpectrum(
            torch.tensor([0.0, 0.0], device=DEVICE, dtype=DTYPE),
            torch.ones(2, device=DEVICE, dtype=DTYPE),
        )


def test_line_spectrum_validates_and_normalizes_without_interpolation():
    spectrum = SolarLineSpectrum(
        torch.tensor([0.3843, 0.8618], device=DEVICE, dtype=DTYPE),
        torch.tensor([10.3, 89.7], device=DEVICE, dtype=DTYPE),
    )
    assert_close(
        spectrum.weights.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        name="line spectrum normalization",
    )
    source = build_solar_source(None, context=make_context())
    with pytest.raises(TypeError, match="discrete line spectrum"):
        source.spectrum("7Be", torch.linspace(0.0, 1.0, 10, device=DEVICE, dtype=DTYPE))


def test_default_source_records_effective_provenance_and_flux_reference():
    source = build_solar_source(None, context=make_context())
    assert source.production_provider == default.solar_provider
    assert source.flux_provider == default.solar_provider
    assert source.spectrum_provider == default.solar_spectrum_provider
    assert source.flux_reference_distance_au == pytest.approx(1.0)
    assert source.flux_unit == "cm^-2 s^-1"
    assert set(source.spectrum_variants) == set(source.spectra)
