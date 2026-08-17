"""Tests for SNOPhaseIIObservableModel: shape, gradients, and real-data-scale sanity checks."""

import pytest
import torch

from tpeanuts.detector.sno_ii.inference_model import SNOPhaseIIObservableModel
from tpeanuts.detector.sno_ii.io import load_integrated_fluxes_day_night
from tpeanuts.detector.sno_ii.parameters import N_OBSERVABLES_TOTAL
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
def theta0(model, context):
    _, theta0_osc = SolarSMOscillationModel.from_preset(
        "_SM_NUFIT61_NO", free=("theta12", "DeltamSq21"), context=context,
    )
    return torch.cat(
        [theta0_osc.detach(), torch.tensor([0.0], dtype=context.dtype, device=context.device)],
    ).requires_grad_(True)


def test_free_params_are_oscillation_params_plus_log_phi_8b(model):
    assert model.free == ("theta12", "DeltamSq21", "log_phi_8B")


def test_predict_shape_and_positivity(model, theta0):
    pred = model.predict(theta0)
    assert pred.shape == (N_OBSERVABLES_TOTAL,)
    assert torch.all(pred > 0)
    assert torch.isfinite(pred).all()


def test_predict_is_same_order_of_magnitude_as_published_data(model, theta0):
    """Not a fit -- just confirms the units/normalization are right (a units
    bug would be off by orders of magnitude, not a plausible physics
    mismatch): at NuFit-default (not SNO-bestfit) parameters, every one of
    the three real published NC/CC1/ES values should be within a factor of
    2 of this model's prediction."""
    pred = model.predict(theta0)
    fluxes = load_integrated_fluxes_day_night()

    assert 0.5 < float(pred[0]) / float(fluxes["NC"].day) < 2.0
    assert 0.5 < float(pred[18]) / float(fluxes["ES"].day) < 2.0
    cc_day_sum = float(pred[1:18].sum())
    assert 0.5 < cc_day_sum / float(fluxes["CC_integral"].day) < 2.0


def test_nc_shows_no_day_night_asymmetry_in_the_standard_model(model, theta0):
    """Eq. 10: NC measures the total active flux, day and night alike --
    the two must agree far more tightly than CC/ES (which do carry a real,
    if small, Earth-regeneration-driven day/night difference)."""
    pred = model.predict(theta0)
    nc_day, nc_night = float(pred[0]), float(pred[19])
    assert abs(nc_day - nc_night) / nc_day < 1.0e-3


def test_gradient_flows_from_every_free_parameter(model, theta0):
    pred = model.predict(theta0)
    grad = torch.autograd.grad(pred.sum(), theta0)[0]
    assert torch.isfinite(grad).all()
    assert torch.all(grad.abs() > 0)


def test_log_phi_8b_zero_reproduces_the_ssm_table_value(model, theta0, context):
    """log_phi_8B=0 must reproduce solar_source.total_flux('8B') up to the
    hep addition and E_NU_GRID_MEV's own quadrature coverage.

    Not exact to machine precision: source.spectrum('8B', E_NU_GRID_MEV)
    integrates to 0.9953 (not 1.0) over this project's shared 400-point,
    1-20 MeV grid (a real, pre-existing quadrature/range-truncation
    characteristic of E_NU_GRID_MEV itself -- ``tpeanuts.detector.sno
    .parameters.E_NU_GRID_MEV`` uses the identical grid, so this is not
    specific to sno_ii), hence the ~0.5% tolerance rather than 1e-9."""
    from tpeanuts.detector.sno_ii.inference_model import FLUX_UNIT_CM2S
    from tpeanuts.detector.sno_ii.parameters import HEP_FLUX_CM2S

    pred = model.predict(theta0)
    total_predicted = float(pred[0])  # NC == total active flux exactly, per Eq. 10.
    expected = (float(model.solar_source.total_flux("8B")) + HEP_FLUX_CM2S) / FLUX_UNIT_CM2S
    assert abs(total_predicted - expected) / expected < 0.01
