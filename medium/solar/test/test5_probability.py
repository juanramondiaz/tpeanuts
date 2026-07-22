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

"""Pytest-compatible tests for solar adiabatic probabilities."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tpeanuts.core.BSM.bsm_nsi import NSIConfig
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.medium.solar.profile import build_solar_profile, SolarParameters
from tpeanuts.medium.solar.probability import (
    Tei,
    solar_probability_state,
    solar_probability_mass,
)
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(*, context: RuntimeContext | None = None) -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO",
        context=context or make_context(),
    )


def make_sterile_oscillation(
    preset: str = "sterile_3p1_bestfit_giunti2017",
    *,
    context: RuntimeContext | None = None,
) -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset(
        preset,
        context=context or make_context(),
    )


def make_nsi_oscillation(
    preset: str = "nsi_ee_only",
    *,
    context: RuntimeContext | None = None,
) -> OscillationParameters:
    ctx = context or make_context()
    base = make_oscillation(context=ctx)
    nsi = NSIConfig.from_preset(preset, device=ctx.device, real_dtype=ctx.dtype)
    return dataclasses.replace(base, nsi=nsi)


def make_nsi_sterile_oscillation(
    *,
    sterile_preset: str = "sterile_3p1_bestfit_giunti2017",
    nsi_preset: str = "nsi_ee_only",
    context: RuntimeContext | None = None,
) -> OscillationParameters:
    ctx = context or make_context()
    base = make_sterile_oscillation(sterile_preset, context=ctx)
    nsi = NSIConfig.from_preset(nsi_preset, device=ctx.device, real_dtype=ctx.dtype)
    return dataclasses.replace(base, nsi=nsi)


def make_inverted_ordering(oscillation: OscillationParameters) -> OscillationParameters:
    """Flip DeltamSq3l's sign to switch a preset from normal to inverted ordering."""
    io_mass_spectrum = dataclasses.replace(
        oscillation.mass_spectrum,
        DeltamSq3l=-oscillation.mass_spectrum.DeltamSq3l.abs(),
    )
    return dataclasses.replace(oscillation, mass_spectrum=io_mass_spectrum)


def make_zero_nsi(oscillation: OscillationParameters) -> OscillationParameters:
    """Attach an exactly-zero NSIConfig, forcing Tei's numerical (eigh) path
    while remaining physically identical to the plain analytic SM path."""
    zero_nsi = NSIConfig()
    zero_nsi = dataclasses.replace(zero_nsi, epsilon=zero_nsi.epsilon_tensor_base())
    return dataclasses.replace(oscillation, nsi=zero_nsi)


def make_profile(*, use_lz: bool = False):
    context = make_context()
    profile = build_solar_profile(None, context=context)
    profile.use_LZ = use_lz
    return profile


def test_tei_returns_normalized_finite_weights_for_energy_density_grid():
    oscillation = make_oscillation()
    energy = torch.tensor([0.1, 1.0, 10.0], device=DEVICE, dtype=DTYPE)[:, None]
    density = torch.tensor([0.0, 1.0, 100.0], device=DEVICE, dtype=DTYPE)[None, :]

    weights = Tei(oscillation, energy, density)

    assert weights.shape == (3, 3, 3)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0.0)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), rtol=1.0e-14, atol=1.0e-14)


def test_solar_probability_mass_bahcall_profile_uses_shell_fraction():
    # End-to-end: a Bahcall-provider profile must route through the
    # discrete-sum branch, not silently fall back to trapz -- regardless of
    # which provider is currently configured as the package-wide default.
    oscillation = make_oscillation()
    context = make_context()
    profile = build_solar_profile(
        None, params=SolarParameters(provider="bahcall"), context=context,
    )
    assert profile.production_measure == "shell_fraction"

    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)
    weights = solar_probability_mass(oscillation, energy, profile, "8B")

    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0.0)
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1e-13, atol=1e-13,
    )


def test_solar_probability_mass_single_source_shape_and_normalization():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    weights = solar_probability_mass(oscillation, energy, profile, "8B")

    assert weights.shape == (3, 3)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0.0)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-13, atol=1.0e-13)


def test_solar_probability_mass_multiple_sources_preserves_source_order():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 10.0], device=DEVICE, dtype=DTYPE)
    sources = ("pp", "8B", "hep")

    multi = solar_probability_mass(oscillation, energy, profile, sources)
    stacked = torch.stack(
        [solar_probability_mass(oscillation, energy, profile, source) for source in sources],
        dim=0,
    )

    assert multi.shape == (3, 2, 3)
    torch.testing.assert_close(multi, stacked, rtol=1.0e-14, atol=1.0e-14)


def test_psolar_probabilities_are_normalized_and_match_mass_projection():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    weights = solar_probability_mass(oscillation, energy, profile, "8B")
    probabilities = solar_probability_state(oscillation, energy, profile, "8B")
    pmns_projection = oscillation.pmns.pmns_matrix().abs() ** 2
    expected = torch.einsum("ei,ni->ne", pmns_projection, weights)

    assert probabilities.shape == (3, 3)
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-13, atol=1.0e-13)
    torch.testing.assert_close(probabilities, expected, rtol=1.0e-13, atol=1.0e-13)


def test_psolar_multiple_sources_matches_single_source_stack():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([0.5, 5.0], device=DEVICE, dtype=DTYPE)
    sources = ("pp", "7Be", "8B")

    multi = solar_probability_state(oscillation, energy, profile, sources)
    stacked = torch.stack(
        [solar_probability_state(oscillation, energy, profile, source) for source in sources],
        dim=0,
    )

    assert multi.shape == (3, 2, 3)
    torch.testing.assert_close(multi, stacked, rtol=1.0e-14, atol=1.0e-14)


def test_electron_survival_decreases_from_low_to_high_energy_for_8b():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([0.1, 10.0], device=DEVICE, dtype=DTYPE)

    pee = solar_probability_state(oscillation, energy, profile, "8B")[:, 0]

    assert pee[0] > pee[1]
    assert 0.45 < float(pee[0]) < 0.65
    assert 0.20 < float(pee[1]) < 0.40


def test_lz_enabled_standard_lma_matches_adiabatic_result_to_float_precision():
    oscillation = make_oscillation()
    profile_ad = make_profile(use_lz=False)
    profile_lz = make_profile(use_lz=True)
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    p_ad = solar_probability_state(oscillation, energy, profile_ad, "8B")
    p_lz = solar_probability_state(oscillation, energy, profile_lz, "8B")

    torch.testing.assert_close(p_lz, p_ad, rtol=0.0, atol=0.0)


# -----------------------------------------------------------------------
# 3+1 sterile extension
# -----------------------------------------------------------------------

def test_tei_numerical_path_triggered_by_sterile_alone_returns_four_weights():
    oscillation = make_sterile_oscillation()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)[:, None]
    density = torch.tensor([0.0, 1.0, 100.0], device=DEVICE, dtype=DTYPE)[None, :]

    weights = Tei(oscillation, energy, density)

    assert weights.shape == (3, 3, 4)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0.0)
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones_like(weights[..., 0]), rtol=1.0e-12, atol=1.0e-12,
    )


def test_solar_probability_state_sterile_does_not_crash_and_is_normalized():
    # Regression test: solar_probability_state used to hardcode torch.eye(3)
    # and crash with a RuntimeError (3x3 @ 4x4 shape mismatch) for any
    # 4-flavour oscillation.pmns.
    oscillation = make_sterile_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    probabilities = solar_probability_state(oscillation, energy, profile, "8B")

    assert probabilities.shape == (3, 4)
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-12, atol=1.0e-12,
    )


def test_solar_probability_state_sterile_null_mixing_reduces_to_sm_active_sector():
    # theta14 = theta24 = theta34 = 0 must reproduce the plain 3-flavour SM
    # result in the active sector, and populate zero sterile flavour --
    # the same "exact SM-limit embedding" property already validated for
    # the sterile extension elsewhere (core/BSM/test/test3_bsm_sterile.py).
    # Numerical-diagonalization vs. closed-form-analytic agreement is
    # expected only to numerical precision, not bit-for-bit -- both are
    # independently correct implementations of the same physics.
    context = make_context()
    oscillation_sm = make_oscillation(context=context)
    oscillation_st = make_sterile_oscillation("sterile_3p1_null_mixing", context=context)
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    p_sm = solar_probability_state(oscillation_sm, energy, profile, "8B")
    p_st = solar_probability_state(oscillation_st, energy, profile, "8B")

    assert p_st.shape == (3, 4)
    torch.testing.assert_close(p_st[..., 3], torch.zeros(3, device=DEVICE, dtype=DTYPE), rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(p_st[..., :3], p_sm, rtol=0.0, atol=1.0e-4)


def test_tei_numerical_path_io_zero_nsi_matches_analytic_path():
    # Regression test for the eigh mass-index mislabelling bug (see Tei's
    # numerical-path vacuum-permutation logic). With epsilon exactly zero,
    # Tei's numerical (eigh) path and its analytic path describe the same
    # physics via two
    # independent algorithms (closed-form arccos formulas vs. eigh matrix
    # diagonalisation), so they agree only to numerical precision, not
    # bit-for-bit -- the same "independently correct implementations"
    # tolerance already used below for the sterile-null-mixing check.
    # Before the fix, this disagreed by O(1) under inverted ordering
    # (a wrong-index swap), not at this ~1e-4 numerical-precision level.
    oscillation_io = make_inverted_ordering(make_oscillation())
    oscillation_io_zero_nsi = make_zero_nsi(oscillation_io)

    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)[:, None]
    density = torch.tensor([0.0, 1.0, 50.0, 100.0], device=DEVICE, dtype=DTYPE)[None, :]

    w_analytic = Tei(oscillation_io, energy, density)
    w_numerical = Tei(oscillation_io_zero_nsi, energy, density)

    torch.testing.assert_close(w_numerical, w_analytic, rtol=0.0, atol=1.0e-4)


def test_solar_probability_state_sterile_null_mixing_reduces_to_sm_active_sector_io():
    # Same "sterile null-mixing reduces to plain SM" check as above, but
    # under inverted ordering -- the scenario that used to be silently
    # wrong (the sterile path always takes Tei's numerical/eigh branch
    # regardless of theta14/24/34).
    context = make_context()
    oscillation_sm = make_inverted_ordering(make_oscillation(context=context))
    oscillation_st = make_inverted_ordering(
        make_sterile_oscillation("sterile_3p1_null_mixing", context=context)
    )
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    p_sm = solar_probability_state(oscillation_sm, energy, profile, "8B")
    p_st = solar_probability_state(oscillation_st, energy, profile, "8B")

    assert p_st.shape == (3, 4)
    torch.testing.assert_close(p_st[..., 3], torch.zeros(3, device=DEVICE, dtype=DTYPE), rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(p_st[..., :3], p_sm, rtol=0.0, atol=1.0e-4)


def test_solar_probability_state_io_sterile_adiabatic_matches_numerical_method():
    # Cross-validates the (now fixed) eigh-based numerical sub-path of
    # method="adiabatic" against the independent method="numerical"
    # coherent-evolutor path, which never had this labelling issue (it
    # projects directly with the real PMNS matrix, not eigh's sorted
    # columns). For weak sterile mixing and standard LMA-like splittings
    # the propagation stays close to adiabatic, so the two independent
    # methods should agree -- before the fix they disagreed well beyond
    # this tolerance under inverted ordering.
    context = make_context()
    oscillation = make_inverted_ordering(
        make_sterile_oscillation("sterile_3p1_null_mixing", context=context)
    )
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    p_adiabatic = solar_probability_state(oscillation, energy, profile, "8B", method="adiabatic")
    p_numerical = solar_probability_state(oscillation, energy, profile, "8B", method="numerical")

    torch.testing.assert_close(p_adiabatic, p_numerical, rtol=0.0, atol=5.0e-3)


def test_solar_probability_mass_include_matter_nc_changes_result_for_sterile():
    oscillation = make_sterile_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    weights_cc_only = solar_probability_mass(oscillation, energy, profile, "8B", include_matter_nc=False)
    weights_with_nc = solar_probability_mass(
        oscillation, energy, profile, "8B", include_matter_nc=True,
    )
    # include_matter_nc=None (the default) auto-resolves to True here since
    # the profile carries neutron-density data and sterile is active.
    weights_default = solar_probability_mass(oscillation, energy, profile, "8B")

    assert weights_cc_only.shape == weights_with_nc.shape == weights_default.shape == (3, 4)
    for weights in (weights_cc_only, weights_with_nc, weights_default):
        assert torch.isfinite(weights).all()
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-12, atol=1.0e-12,
        )
    assert not torch.allclose(weights_cc_only, weights_with_nc)
    torch.testing.assert_close(weights_default, weights_with_nc, rtol=1.0e-12, atol=1.0e-12)


def test_solar_probability_mass_include_matter_nc_ignored_for_three_flavour():
    # Mirrors hamiltonian_matter_reduced's own convention: V_NC is an
    # unobservable common phase in the plain 3-flavour case, so requesting
    # it there is mathematically inert, not an error.
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    weights_default = solar_probability_mass(oscillation, energy, profile, "8B")
    weights_with_nc = solar_probability_mass(
        oscillation, energy, profile, "8B", include_matter_nc=True,
    )

    torch.testing.assert_close(weights_default, weights_with_nc, rtol=0.0, atol=0.0)


def test_solar_probability_mass_include_matter_nc_requires_density_n():
    oscillation = make_sterile_oscillation()
    profile = make_profile()
    profile.density_n = None
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="density_n"):
        solar_probability_mass(oscillation, energy, profile, "8B", include_matter_nc=True)


def test_tei_warns_when_p_lz_supplied_on_sterile_diagonalization_path():
    oscillation = make_sterile_oscillation()
    energy = torch.tensor([5.0], device=DEVICE, dtype=DTYPE)
    density = torch.tensor([10.0], device=DEVICE, dtype=DTYPE)
    p_lz = torch.tensor([0.1], device=DEVICE, dtype=DTYPE)

    with pytest.warns(RuntimeWarning, match="p_lz is ignored"):
        Tei(oscillation, energy, density, p_lz=p_lz)


def test_solar_probability_mass_rejects_lz_with_sterile():
    oscillation = make_sterile_oscillation()
    profile_lz = make_profile(use_lz=True)
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="use_LZ=True is not supported"):
        solar_probability_mass(oscillation, energy, profile_lz, "8B")


def test_solar_probability_mass_rejects_lz_with_numerical_method():
    oscillation = make_oscillation()
    profile_lz = make_profile(use_lz=True)
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="use_LZ=True has no effect"):
        solar_probability_mass(oscillation, energy, profile_lz, "8B", method="numerical")


def test_solar_probability_mass_rejects_lz_with_multidim_energy():
    oscillation = make_oscillation()
    profile_lz = make_profile(use_lz=True)
    energy = torch.tensor([[1.0, 5.0], [8.0, 10.0]], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="scalar or 1-D energy grid"):
        solar_probability_mass(oscillation, energy, profile_lz, "8B")


# -----------------------------------------------------------------------
# NSI, and NSI + 3+1 sterile combined
# -----------------------------------------------------------------------

def test_solar_probability_state_nsi_alone_changes_result_and_stays_three_flavour():
    context = make_context()
    oscillation_sm = make_oscillation(context=context)
    oscillation_nsi = make_nsi_oscillation(context=context)
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    p_sm = solar_probability_state(oscillation_sm, energy, profile, "8B")
    p_nsi = solar_probability_state(oscillation_nsi, energy, profile, "8B")

    assert p_nsi.shape == (3, 3)
    assert torch.isfinite(p_nsi).all()
    torch.testing.assert_close(
        p_nsi.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-12, atol=1.0e-12,
    )
    assert not torch.allclose(p_sm, p_nsi, atol=1.0e-6)


def test_solar_probability_state_nsi_and_sterile_combined_does_not_crash():
    # The genuinely new, previously entirely unsupported combination: NSI
    # and the 3+1 sterile extension active simultaneously in the solar
    # pipeline.
    oscillation = make_nsi_sterile_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    probabilities = solar_probability_state(oscillation, energy, profile, "8B")

    assert probabilities.shape == (3, 4)
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-12, atol=1.0e-12,
    )


def test_solar_probability_state_nsi_and_sterile_with_matter_nc_does_not_crash():
    oscillation = make_nsi_sterile_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    probabilities = solar_probability_state(
        oscillation, energy, profile, "8B", include_matter_nc=True,
    )

    assert probabilities.shape == (3, 4)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-12, atol=1.0e-12,
    )


# -----------------------------------------------------------------------
# method="numerical" dispatch (medium.solar.evolutor)
# -----------------------------------------------------------------------

def test_solar_probability_mass_rejects_unknown_method():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="method must be"):
        solar_probability_mass(oscillation, energy, profile, "8B", method="bogus")


def test_solar_probability_state_numerical_matches_adiabatic_in_sm_limit():
    # method="numerical" makes no adiabatic assumption at all, so it is not
    # expected to agree with method="adiabatic" to floating-point precision
    # -- only to the level that the adiabatic approximation itself is good,
    # which is excellent for standard LMA parameters.
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    p_adiabatic = solar_probability_state(oscillation, energy, profile, "8B", method="adiabatic")
    p_numerical = solar_probability_state(oscillation, energy, profile, "8B", method="numerical")

    assert p_numerical.shape == p_adiabatic.shape == (3, 3)
    assert torch.isfinite(p_numerical).all()
    torch.testing.assert_close(
        p_numerical.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-6, atol=1.0e-6,
    )
    torch.testing.assert_close(p_numerical, p_adiabatic, rtol=0.0, atol=5.0e-3)


def test_solar_probability_state_numerical_sterile_does_not_crash_and_is_normalized():
    oscillation = make_sterile_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    probabilities = solar_probability_state(oscillation, energy, profile, "8B", method="numerical")

    assert probabilities.shape == (3, 4)
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-6, atol=1.0e-6,
    )


def test_solar_probability_state_numerical_nsi_and_sterile_combined_does_not_crash():
    oscillation = make_nsi_sterile_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    probabilities = solar_probability_state(oscillation, energy, profile, "8B", method="numerical")

    assert probabilities.shape == (3, 4)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-6, atol=1.0e-6,
    )


def test_solar_probability_state_numerical_include_matter_nc_changes_result():
    oscillation = make_sterile_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    p_cc_only = solar_probability_state(
        oscillation, energy, profile, "8B", method="numerical", include_matter_nc=False,
    )
    p_with_nc = solar_probability_state(
        oscillation, energy, profile, "8B", method="numerical", include_matter_nc=True,
    )
    # include_matter_nc=None (the default) auto-resolves to True here since
    # the profile carries neutron-density data and sterile is active.
    p_default = solar_probability_state(oscillation, energy, profile, "8B", method="numerical")

    assert not torch.allclose(p_cc_only, p_with_nc)
    torch.testing.assert_close(p_default, p_with_nc, rtol=1.0e-12, atol=1.0e-12)
    for p in (p_cc_only, p_with_nc, p_default):
        torch.testing.assert_close(
            p.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-6, atol=1.0e-6,
        )
