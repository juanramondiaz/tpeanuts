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

"""Pytest-compatible tests for tpeanuts.medium.solar.profile."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

import tpeanuts.config.default as default
from tpeanuts.medium.solar.io import load_solar_production
from tpeanuts.medium.solar.profile import (
    SolarParameters,
    SolarProfile,
    build_solar_profile,
)
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_same_device(actual: torch.device, expected: torch.device) -> None:
    assert actual.type == torch.device(expected).type


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_profile(*, dtype: torch.dtype = DTYPE, production_measure: str = "radial_pdf") -> SolarProfile:
    device = DEVICE
    radius = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device, dtype=dtype)
    density = torch.tensor([100.0, 50.0, 10.0, 2.0, 1.0], device=device, dtype=dtype)
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
    return SolarProfile(
        radius=radius, density=density, production_radius=radius,
        fractions=fractions, fluxes=fluxes,
        production_measure=production_measure,
    )


def test_solar_profile_device_dtype_properties_and_string_summary():
    profile = make_profile()
    summary = str(profile)

    assert_same_device(profile.device, DEVICE)
    assert profile.dtype == DTYPE
    assert "SolarProfile" in summary
    assert "sources=" in summary
    assert "use_LZ=False" in summary


def test_electron_density_interpolates_linearly_and_clamps_edges():
    profile = make_profile()
    query = torch.tensor([-0.2, 0.0, 0.125, 0.375, 1.0, 1.2], device=DEVICE, dtype=DTYPE)

    density = profile.electron_density(query)
    expected = torch.tensor([100.0, 100.0, 75.0, 30.0, 1.0, 1.0], device=DEVICE, dtype=DTYPE)

    assert_close(density, expected, name="solar density interpolation")


def test_density_n_defaults_to_none_on_manually_constructed_profile():
    profile = make_profile()

    assert profile.density_n is None


def test_neutron_density_interpolates_linearly_and_clamps_edges():
    profile = make_profile()
    profile.density_n = torch.tensor([50.0, 25.0, 5.0, 1.0, 0.5], device=DEVICE, dtype=DTYPE)
    query = torch.tensor([-0.2, 0.0, 0.125, 0.375, 1.0, 1.2], device=DEVICE, dtype=DTYPE)

    neutron_density = profile.neutron_density(query)
    expected = torch.tensor([50.0, 50.0, 37.5, 15.0, 0.5, 0.5], device=DEVICE, dtype=DTYPE)

    assert_close(neutron_density, expected, name="solar neutron-density interpolation")


def test_neutron_density_raises_when_density_n_is_not_set():
    profile = make_profile()

    with pytest.raises(ValueError, match="density_n is not set"):
        profile.neutron_density(profile.radius)


def test_production_distribution_returns_grid_values_and_interpolates_query_points():
    profile = make_profile()
    query = torch.tensor([0.125, 0.375], device=DEVICE, dtype=DTYPE)

    grid_fraction = profile.production_distribution("pp")
    query_fraction = profile.production_distribution("pp", query)

    assert_close(grid_fraction, profile.fractions["pp"], name="production fraction on grid")
    assert_close(query_fraction, torch.tensor([0.5, 1.5], device=DEVICE, dtype=DTYPE), name="production fraction interpolation")


def test_production_distribution_unknown_source_raises():
    profile = make_profile()

    with pytest.raises(KeyError, match="Unknown solar source"):
        profile.production_distribution("unknown")


def test_production_distribution_single_and_multiple_sources():
    profile = make_profile()

    single = profile.production_distribution("pp")
    multiple = profile.production_distribution(["pp", "8B"])

    assert single.shape == profile.production_radius.shape
    assert multiple.shape == (2, profile.production_radius.numel())
    assert_close(multiple[0], profile.fractions["pp"], name="stacked pp")
    assert_close(multiple[1], profile.fractions["8B"], name="stacked 8B")


def test_radial_pdf_is_normalized_and_nonnegative_on_construction():
    profile = make_profile()

    normalized = profile.production_distribution("pp")
    area = torch.trapz(normalized, x=profile.production_radius)

    assert bool(torch.all(normalized >= 0.0))
    assert_close(area, torch.tensor(1.0, device=DEVICE, dtype=DTYPE), name="normalized production fraction area")


def test_shell_fraction_is_normalized_by_sum_on_construction():
    # A "shell_fraction" profile's stored distribution should sum (plain
    # sum) to 1 rather than trapezoidal-integrate to 1 -- treating discrete
    # per-shell weights as a continuous density would reintroduce the same
    # shell-vs-density mismatch the production_measure branch exists to
    # avoid (see SolarProfile.mass_weights_integrate).
    profile = make_profile(production_measure="shell_fraction")
    normalized = profile.production_distribution("pp")

    assert bool(torch.all(normalized >= 0.0))
    assert_close(
        normalized.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        name="shell_fraction normalized sum",
    )


def test_production_distribution_clamps_roundoff_negative_before_normalizing():
    profile = make_profile()
    profile.fractions["pp"] = torch.tensor(
        [-1.0e-12, 1.0, 2.0, 1.0, 0.0], device=DEVICE, dtype=DTYPE
    )
    profile.__post_init__()
    distribution = profile.production_distribution("pp")
    assert bool(torch.all(distribution >= 0.0))
    assert_close(
        torch.trapezoid(distribution, x=profile.production_radius),
        torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        name="sanitized radial PDF normalization",
    )


def test_construction_rejects_unknown_production_measure():
    with pytest.raises(ValueError, match="production_measure"):
        make_profile(production_measure="bogus")


def _make_two_shell_profile(*, production_measure: str) -> SolarProfile:
    # Two production shells at very different radial spacing, so
    # shell_fraction (discrete sum) and radial_pdf (trapz) genuinely
    # disagree on this profile.
    radius = torch.tensor([0.0, 0.001, 1.0], device=DEVICE, dtype=DTYPE)
    fractions = torch.tensor([0.0, 0.5, 0.5], device=DEVICE, dtype=DTYPE)
    return SolarProfile(
        radius=radius, density=torch.ones_like(radius), production_radius=radius,
        fractions={"pp": fractions}, fluxes={"pp": torch.tensor(1.0, device=DEVICE, dtype=DTYPE)},
        production_measure=production_measure,
    )


def test_mass_weights_integrate_shell_fraction_ignores_grid_spacing():
    # For "shell_fraction" data (e.g. Bahcall) the reduction must not be
    # reweighted by the (arbitrary) local grid spacing: each tabulated
    # fraction already carries its own shell's full share of production, so
    # the result must equal the plain fraction-weighted average regardless
    # of how close together the shells are sampled.
    profile = _make_two_shell_profile(production_measure="shell_fraction")
    fractions = profile.production_distribution("pp")
    weights_r = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], device=DEVICE, dtype=DTYPE,
    )

    result = profile.mass_weights_integrate(weights_r, fractions, energy_ndim=0)

    expected = torch.tensor([0.5, 0.5], device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(result, expected, rtol=1e-14, atol=1e-14)


def test_mass_weights_integrate_radial_pdf_matches_manual_trapz():
    profile = _make_two_shell_profile(production_measure="radial_pdf")
    fractions = profile.production_distribution("pp")
    weights_r = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], device=DEVICE, dtype=DTYPE,
    )

    result = profile.mass_weights_integrate(weights_r, fractions, energy_ndim=0)

    weighted = weights_r * fractions[:, None]
    expected = torch.trapz(weighted, x=profile.production_radius, dim=0)
    torch.testing.assert_close(result, expected, rtol=1e-14, atol=1e-14)

    # Sanity check that the two measures genuinely disagree on this
    # deliberately non-uniform grid (otherwise the two tests above would
    # not actually be distinguishing anything).
    shell_profile = _make_two_shell_profile(production_measure="shell_fraction")
    shell_result = shell_profile.mass_weights_integrate(
        weights_r, shell_profile.production_distribution("pp"), energy_ndim=0,
    )
    assert not torch.allclose(result, shell_result)


def test_production_distribution_rejects_significant_negative_and_nonfinite():
    radius = torch.tensor([0.0, 0.5, 1.0], device=DEVICE, dtype=DTYPE)
    density = torch.ones_like(radius)
    fluxes = {"pp": torch.tensor(1.0, device=DEVICE, dtype=DTYPE)}
    with pytest.raises(ValueError, match="significant negative"):
        SolarProfile(
            radius, density, radius,
            {"pp": torch.tensor([0.0, -1.0e-3, 1.0], device=DEVICE, dtype=DTYPE)}, fluxes
        )
    with pytest.raises(ValueError, match="non-finite"):
        SolarProfile(
            radius, density, radius,
            {"pp": torch.tensor([0.0, float("nan"), 1.0], device=DEVICE, dtype=DTYPE)}, fluxes
        )


def test_flux_returns_scalar_and_unknown_source_raises():
    profile = make_profile()

    flux = profile.flux("8B")

    assert flux.shape == ()
    assert_close(flux, torch.tensor(4.5e6, device=DEVICE, dtype=DTYPE), name="8B flux")
    with pytest.raises(KeyError, match="Unknown solar flux source"):
        profile.flux("unknown")


def test_build_solar_profile_returns_existing_profile_when_context_matches():
    profile = make_profile()

    out = build_solar_profile(profile, context=RuntimeContext.resolve(profile.device, profile.dtype))

    assert out is profile


def test_build_solar_profile_casts_existing_profile_to_requested_dtype():
    profile = make_profile(dtype=torch.float64)
    ctx = make_context(dtype=torch.float32)

    out = build_solar_profile(profile, context=ctx)

    assert out is not profile
    assert out.radius.dtype == torch.float32
    assert out.production_radius.dtype == torch.float32
    assert out.density.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in out.fractions.values())
    assert all(value.dtype == torch.float32 for value in out.fluxes.values())
    assert out.density_n is None


def test_build_solar_profile_casts_density_n_when_present():
    profile = make_profile(dtype=torch.float64)
    profile.density_n = torch.tensor(
        [50.0, 25.0, 5.0, 1.0, 0.5], device=DEVICE, dtype=torch.float64,
    )
    ctx = make_context(dtype=torch.float32)

    out = build_solar_profile(profile, context=ctx)

    assert out.density_n is not None
    assert out.density_n.dtype == torch.float32
    assert_close(
        out.density_n,
        profile.density_n.to(dtype=torch.float32),
        name="cast density_n",
    )


def test_solar_profile_default_loads_from_explicit_synthetic_paths(tmp_path):
    density_path = tmp_path / "density.csv"
    production_path = tmp_path / "production.csv"
    flux_path = tmp_path / "fluxes.csv"
    pd.DataFrame(
        {
            "radius": [0.0, 0.5, 1.0],
            "electron_density_mol_cm3": [100.0, 10.0, 1.0],
            "neutron_density_mol_cm3": [80.0, 8.0, 0.8],
        }
    ).to_csv(density_path, index=False)
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

    params = SolarParameters(
        density_path=str(density_path),
        production_path=str(production_path),
        fluxes_path=str(flux_path),
    )
    profile = SolarProfile.default(params=params, context=make_context())

    assert_close(profile.radius, torch.tensor([0.0, 0.5, 1.0], device=DEVICE, dtype=DTYPE), name="default synthetic radius")
    assert_close(profile.density, torch.tensor([100.0, 10.0, 1.0], device=DEVICE, dtype=DTYPE), name="default synthetic density")
    assert sorted(profile.fractions) == ["8B", "pp"]
    assert sorted(profile.fluxes) == ["8B", "pp"]
    # Explicit path override, no provider given: defaults to the historical
    # "radial_pdf" (trapz) convention (see load_solar_production).
    assert profile.production_measure == "radial_pdf"


def test_solar_profile_default_rejects_production_flux_source_mismatch(tmp_path):
    density_path = tmp_path / "density.csv"
    production_path = tmp_path / "production.csv"
    flux_path = tmp_path / "flux.csv"
    pd.DataFrame(
        {"radius": [0.0, 1.0], "electron_density_mol_cm3": [10.0, 0.0]}
    ).to_csv(density_path, index=False)
    pd.DataFrame(
        {"radius": [0.0, 1.0], "pp fraction": [1.0, 1.0]}
    ).to_csv(production_path, index=False)
    pd.DataFrame(
        {"fraction": ["8B"], "flux": [1.0]}
    ).to_csv(flux_path, index=False)
    with pytest.raises(ValueError, match="production/flux source mismatch"):
        SolarProfile.default(
            params=SolarParameters(
                density_path=str(density_path),
                production_path=str(production_path),
                fluxes_path=str(flux_path),
                spectrum_provider=None,
            ),
            context=make_context(),
        )


def test_build_solar_profile_loads_default_when_profile_is_none():
    profile = build_solar_profile(None, context=make_context())

    assert profile.radius.ndim == 1
    assert profile.density.shape == profile.radius.shape
    assert bool(torch.all(torch.diff(profile.radius) > 0.0))
    assert bool(torch.all(profile.radius >= 0.0))
    assert bool(torch.all(profile.radius <= 1.0))
    assert bool(torch.all(profile.density > 0.0))
    assert {"pp", "8B", "7Be", "hep"}.issubset(profile.fractions)
    assert {"pp", "8B", "7Be", "hep"}.issubset(profile.fluxes)
    assert profile.spectrum_provider == "legacy"
    assert profile.has_spectrum("8B")
    assert profile.has_spectrum("7Be")
    assert not profile.has_spectrum("pep")
    gap_energy = torch.tensor(0.6, device=DEVICE, dtype=DTYPE)
    assert profile.spectrum("7Be", gap_energy) == 0.0
    # Must match whatever tpeanuts.config.default.solar_provider resolves to
    # (kept provider-agnostic so this test does not go stale if the
    # configured default provider changes).
    expected_measure = load_solar_production(
        provider=default.solar_provider, device=DEVICE, dtype=DTYPE,
    )["production_measure"]
    assert profile.production_measure == expected_measure
    for distribution in profile.fractions.values():
        assert bool(torch.all(distribution >= 0.0))
        normalization = (
            distribution.sum()
            if profile.production_measure == "shell_fraction"
            else torch.trapezoid(distribution, x=profile.production_radius)
        )
        assert_close(
            normalization,
            torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
            name="default production-distribution normalization",
        )
    # The default Zenodo provider supplies density_n directly. Near the
    # photosphere n_n can exceed the free-electron density because the plasma
    # is no longer fully ionized, so no global n_n <= n_e inequality applies.
    assert profile.density_n is not None
    assert profile.density_n.shape == profile.radius.shape
    assert torch.isfinite(profile.density_n).all()
    assert bool(torch.all(profile.density_n > 0.0))
