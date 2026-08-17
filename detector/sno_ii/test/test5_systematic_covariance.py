"""Tests for the systematic covariance V_syst (Eq. 21) built from Table XXXIV."""

import functools

import torch

from tpeanuts.detector.sno_ii.inference_model import SNOPhaseIIObservableModel
from tpeanuts.detector.sno_ii.io import (
    build_systematic_covariance,
    load_cc_spectral_systematics,
    load_observed_vector_and_covariance,
    load_observed_vector_and_total_covariance,
)
from tpeanuts.detector.sno_ii.parameters import CHANNEL_ORDER, N_OBSERVABLES_TOTAL
from tpeanuts.inference.fit import fit_lbfgs
from tpeanuts.inference.likelihood import cholesky_from_covariance, correlated_gaussian_nll
from tpeanuts.inference.solar_model import SolarSMOscillationModel
from tpeanuts.medium.earth.profile import EarthParameters, build_earth_profile
from tpeanuts.medium.solar.profile import build_solar_medium
from tpeanuts.source.solar import build_solar_source
from tpeanuts.util.context import RuntimeContext


def test_v_syst_is_symmetric_and_psd():
    value, _ = load_observed_vector_and_covariance()
    V_syst = build_systematic_covariance(value)
    assert V_syst.shape == (N_OBSERVABLES_TOTAL, N_OBSERVABLES_TOTAL)
    assert torch.allclose(V_syst, V_syst.T)
    eigvals = torch.linalg.eigvalsh(V_syst)
    assert eigvals.min() > -1.0e-8  # a Gram matrix (dY^T @ dY) is PSD by construction.


def test_v_syst_diagonal_matches_the_quadrature_sum_of_table_xxxiv():
    """Manually reconstruct CC1_day's own sigma_syst from the raw table
    (bypassing build_systematic_covariance) and confirm it matches the
    function's own diagonal entry -- checks the percent -> absolute
    conversion and the CHANNEL_ORDER <-> SYSTEMATICS_CHANNEL_ORDER remap."""
    value, _ = load_observed_vector_and_covariance()
    V_syst = build_systematic_covariance(value)
    systematics = load_cc_spectral_systematics()

    cc1_idx_in_systematics = systematics.channels.index("CC1")
    fractional = 0.5 * (
        systematics.plus[:, cc1_idx_in_systematics].abs() + systematics.minus[:, cc1_idx_in_systematics].abs()
    ) / 100.0
    cc1_value_day = value[CHANNEL_ORDER.index("CC1")]
    expected_sigma = torch.sqrt(torch.sum((fractional * cc1_value_day) ** 2))

    cc1_idx_in_value = CHANNEL_ORDER.index("CC1")
    actual_sigma = torch.sqrt(V_syst[cc1_idx_in_value, cc1_idx_in_value])
    assert abs(float(actual_sigma) - float(expected_sigma)) < 1.0e-12


def test_v_syst_gives_exact_unit_day_night_correlation_by_construction():
    """Table XXXIV provides only one (not day/night-separated) sensitivity
    per systematic, applied to both periods' own central value -- so any
    two observables of the *same channel* are exactly (not just highly)
    correlated in V_syst alone: one is a positive-scalar multiple of the
    other in every systematic's contribution. This is a direct, documented
    consequence of the data's own structure, not a bug -- V_stat (which
    IS independently day/night by construction) is what actually
    decorrelates the two periods in the total covariance."""
    value, _ = load_observed_vector_and_covariance()
    V_syst = build_systematic_covariance(value)
    nc_day, nc_night = 0, CHANNEL_ORDER.index("NC") + 19
    corr = V_syst[nc_day, nc_night] / torch.sqrt(V_syst[nc_day, nc_day] * V_syst[nc_night, nc_night])
    assert abs(float(corr) - 1.0) < 1.0e-6


def test_v_total_is_v_stat_plus_v_syst_and_positive_definite():
    value, V_stat = load_observed_vector_and_covariance()
    V_syst = build_systematic_covariance(value)
    value2, V_total = load_observed_vector_and_total_covariance()

    assert torch.allclose(value, value2)
    assert torch.allclose(V_total, V_stat + V_syst)
    eigvals = torch.linalg.eigvalsh(V_total)
    assert eigvals.min() > 0.0  # V_stat alone is already PD; adding a PSD V_syst keeps it PD.
    cholesky_from_covariance(V_total)  # must not raise.


def test_fit_with_v_total_likelihood_converges():
    """A lighter sanity check than test4_fit.py's full closure test: just
    confirms the V_stat+V_syst correlated-Gaussian likelihood is usable
    end-to-end with fit_lbfgs and improves on the starting loss."""
    context = RuntimeContext.resolve("cpu", torch.float64)
    osc_model, theta0_osc = SolarSMOscillationModel.from_preset(
        "_SM_NUFIT61_NO", free=("theta12", "DeltamSq21"), context=context,
    )
    solar_medium = build_solar_medium(None, context=context)
    solar_source = build_solar_source(None, context=context)
    earth_profile = build_earth_profile(None, params=EarthParameters(), context=context)
    model = SNOPhaseIIObservableModel.from_real_exposure(
        osc_model, solar_medium, solar_source, earth_profile,
        device=context.device, dtype=context.dtype,
    )
    theta_start = torch.cat(
        [theta0_osc.detach(), torch.tensor([0.0], dtype=context.dtype)],
    ).requires_grad_(True)

    value, V_total = load_observed_vector_and_total_covariance(device=context.device, dtype=context.dtype)
    L = cholesky_from_covariance(V_total)
    likelihood_fn = functools.partial(correlated_gaussian_nll, cholesky_L=L)

    result = fit_lbfgs(model, theta_start, value, likelihood=likelihood_fn, max_iter=40)
    assert torch.isfinite(result.theta_hat).all()
    assert result.chi2_history[-1] < result.chi2_history[0]
