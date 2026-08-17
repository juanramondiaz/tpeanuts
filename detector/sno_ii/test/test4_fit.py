"""Phase 7: first real V_stat-only fit against the real published SNO salt-phase
data, plus a synthetic closure test using the same statistical covariance."""

import functools
import math

import pytest
import torch

from tpeanuts.detector.sno_ii.inference_model import SNOPhaseIIObservableModel
from tpeanuts.detector.sno_ii.io import load_observed_vector_and_covariance
from tpeanuts.inference.fit import fit_lbfgs
from tpeanuts.inference.likelihood import cholesky_from_covariance, correlated_gaussian_nll
from tpeanuts.inference.solar_model import SolarSMOscillationModel
from tpeanuts.medium.earth.profile import EarthParameters, build_earth_profile
from tpeanuts.medium.solar.profile import build_solar_medium
from tpeanuts.source.solar import build_solar_source
from tpeanuts.util.context import RuntimeContext


@pytest.fixture(scope="module")
def context():
    return RuntimeContext.resolve("cpu", torch.float64)


@pytest.fixture(scope="module")
def model(context):
    osc_model, _ = SolarSMOscillationModel.from_preset(
        "_SM_NUFIT61_NO", free=("theta12", "DeltamSq21"), context=context,
    )
    solar_medium = build_solar_medium(None, context=context)
    solar_source = build_solar_source(None, context=context)
    earth_profile = build_earth_profile(None, params=EarthParameters(), context=context)
    return SNOPhaseIIObservableModel.from_real_exposure(
        osc_model, solar_medium, solar_source, earth_profile,
        device=context.device, dtype=context.dtype,
    )


@pytest.fixture(scope="module")
def theta_start(context):
    _, theta0_osc = SolarSMOscillationModel.from_preset(
        "_SM_NUFIT61_NO", free=("theta12", "DeltamSq21"), context=context,
    )
    return torch.cat(
        [theta0_osc.detach(), torch.tensor([0.0], dtype=context.dtype, device=context.device)],
    ).requires_grad_(True)


@pytest.fixture(scope="module")
def stat_likelihood(context):
    _, V_stat = load_observed_vector_and_covariance(device=context.device, dtype=context.dtype)
    L = cholesky_from_covariance(V_stat)
    return functools.partial(correlated_gaussian_nll, cholesky_L=L)


def test_real_data_fit_converges_to_a_physically_sensible_point(model, theta_start, stat_likelihood):
    """V_stat-only fit against the real 38 published observables. Not
    expected to precisely reproduce the paper's own combined (V_stat+V_syst,
    plus KamLAND) best fit -- DeltamSq21 is known to be only weakly
    constrained by solar data alone in this project's own earlier Borexino/
    SNO work -- so this checks convergence and physical sanity (positive
    flux, theta12/DeltamSq21 in a broad plausible range), not precision
    recovery of a specific published number."""
    value, _ = load_observed_vector_and_covariance()
    result = fit_lbfgs(model, theta_start, value, likelihood=stat_likelihood, max_iter=80)

    theta12_hat, dm21_hat, log_phi8b_hat = result.theta_hat
    assert torch.isfinite(result.theta_hat).all()
    assert 0.0 < float(theta12_hat) < math.pi / 2
    assert float(dm21_hat) > 0.0
    assert torch.isfinite(result.covariance).all()

    final_loss = result.chi2_history[-1]
    assert math.isfinite(final_loss)
    assert final_loss < result.chi2_history[0]  # the fit must have improved on the starting point.


def test_synthetic_closure_recovers_injected_truth(model, theta_start, context):
    """Generate synthetic data at a known truth using V_stat's own real
    correlation structure (not an idealized diagonal one), refit from the
    NuFit starting point, and confirm recovery -- the same closure pattern
    used for detector.icecube's own synthetic-recovery test this session."""
    _, V_stat = load_observed_vector_and_covariance(device=context.device, dtype=context.dtype)
    L = cholesky_from_covariance(V_stat)
    likelihood_fn = functools.partial(correlated_gaussian_nll, cholesky_L=L)

    theta_truth = torch.tensor(
        [math.radians(35.0), 6.0e-5, 0.05], dtype=context.dtype, device=context.device,
    )
    prediction_truth = model.predict(theta_truth).detach()

    generator = torch.Generator().manual_seed(0)
    z = torch.randn(38, generator=generator, dtype=context.dtype)
    synthetic_value = prediction_truth + L @ z

    result = fit_lbfgs(model, theta_start, synthetic_value, likelihood=likelihood_fn, max_iter=80)
    theta12_hat, dm21_hat, log_phi8b_hat = result.theta_hat

    assert abs(math.degrees(float(theta12_hat)) - 35.0) < 3.0
    # DeltamSq21's tolerance is deliberately loose: the primary source's own
    # SNO-only fit (Table XXVIII) quotes Delta m^2 = 5.0 (+4.4/-1.8) e-5 eV^2
    # -- a -36%/+88% *published* uncertainty -- so SNO alone (no KamLAND) is
    # known to only weakly constrain this direction; a tighter tolerance
    # here would be stricter than SNO's own official fit.
    assert abs(float(dm21_hat) / 6.0e-5 - 1.0) < 0.6
    # log_phi_8B is tightly tied to NC alone (Eq. 10, oscillation-
    # independent), whose own real statistical uncertainty is
    # sigma_NC/NC ~= 0.31/4.81 ~= 6% (day) -- a single synthetic noise
    # draw can land ~1.5 sigma away in log-flux space, so a tolerance
    # tighter than ~3 sigma (~0.15) would make this test seed-dependent
    # rather than a genuine closure check.
    assert abs(float(log_phi8b_hat) - 0.05) < 0.15
