"""Regression tests for the real SNO salt-phase (Phase II) data loaders."""

import torch

from tpeanuts.detector.sno_ii.io import (
    load_cc_spectral_systematics,
    load_cc_spectrum_day_night,
    load_integrated_fluxes_day_night,
    load_statistical_correlation,
    load_zenith_exposure,
)
from tpeanuts.detector.sno_ii.parameters import CC_BIN_EDGES_MEV, CHANNEL_ORDER, N_CC_BINS, N_CHANNELS


def test_cc_spectrum_bin_sum_matches_published_integral_flux():
    """Table XXX's 17 bins must sum to Table XXIV's own published CC integral flux."""
    day, night = load_cc_spectrum_day_night()
    assert day.value.shape == (N_CC_BINS,)
    assert night.value.shape == (N_CC_BINS,)
    assert torch.allclose(day.bin_edges_MeV, CC_BIN_EDGES_MEV, atol=1.0e-9)

    fluxes = load_integrated_fluxes_day_night()
    assert abs(float(day.value.sum()) - float(fluxes["CC_integral"].day)) < 5.0e-3
    assert abs(float(night.value.sum()) - float(fluxes["CC_integral"].night)) < 5.0e-3


def test_integrated_fluxes_have_nc_es_and_cc_cross_check():
    fluxes = load_integrated_fluxes_day_night()
    assert set(fluxes) == {"NC", "ES", "CC_integral"}
    for flux in fluxes.values():
        assert float(flux.day_sigma_stat) > 0
        assert float(flux.night_sigma_stat) > 0


def test_statistical_correlation_matrices_are_valid_correlation_matrices():
    """Both matrices must be exactly (19, 19), symmetric, unit-diagonal, and PSD --
    load_statistical_correlation itself already enforces this on load, so a
    passing call here is the regression test."""
    for period in ("day", "night"):
        m = load_statistical_correlation(period)
        assert m.shape == (N_CHANNELS, N_CHANNELS)
        assert torch.allclose(m, m.T)
        assert torch.allclose(torch.diag(m), torch.ones(N_CHANNELS, dtype=m.dtype))
        eigvals = torch.linalg.eigvalsh(m)
        assert eigvals.min() > -1.0e-6


def test_zenith_exposure_spans_full_nadir_range():
    eta, exposure = load_zenith_exposure()
    assert eta.shape == (60,)
    assert exposure.shape == (60,)
    assert torch.all(exposure >= 0)
    assert float(eta.min()) == 0.0
    assert abs(float(eta.max()) - torch.pi) < 1.0e-9


def test_cc_spectral_systematics_covers_every_channel():
    systematics = load_cc_spectral_systematics()
    assert set(systematics.channels) == set(CHANNEL_ORDER)
    assert systematics.plus.shape == (len(systematics.names), len(systematics.channels))
    assert systematics.plus.shape == systematics.minus.shape
