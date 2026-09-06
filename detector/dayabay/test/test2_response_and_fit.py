"""Regression tests for the real IAV/LSNL response order, the IBD cross section, the
sin^2(2 theta13)/Delta m^2_ee reparametrization, LSNL pull nuisances, and rate-only
parameter recovery."""

import math

import torch

from tpeanuts.detector.common.response import gaussian_response_matrix
from tpeanuts.detector.dayabay.inference_model import (
    DayaBayDetectorModel,
    DayaBayExperimentalModel,
    NearFarRatioDayaBayModel,
)
from tpeanuts.detector.dayabay.io import load_survival_probability_truth
from tpeanuts.detector.dayabay.parameters import (
    DETECTORS,
    ERES_A,
    ERES_B,
    ERES_C,
    FAR_DETECTORS,
    NEAR_DETECTORS,
    T_GRID_MEV,
    TPRIME_GRID_MEV,
)
from tpeanuts.detector.dayabay.response import _iav_mass_matrix, _lsnl_warp_matrix, response_matrix, sigma_MeV
from tpeanuts.detector.interaction.inverse_beta_decay import (
    ibd_cross_section_grid_precise,
    sigma_ibd,
)
from tpeanuts.inference.fit import fit_lbfgs
from tpeanuts.inference.model_vacuum import VacuumOscillationModel
from tpeanuts.util.context import RuntimeContext


def _build_oscillation_model(context, *, free=("theta13", "DeltamSq3l")):
    truth = load_survival_probability_truth()
    theta12 = 0.5 * math.asin(math.sqrt(truth["SinSq2Theta12"]))
    base, _ = VacuumOscillationModel.from_preset("_SM_NUFIT61_NO", free=free, context=context)
    fixed = {
        "theta12": torch.tensor(theta12, dtype=context.dtype),
        "DeltamSq21": torch.tensor(truth["DeltaMSq21"], dtype=context.dtype),
    }
    if "DeltamSq3l" not in free:
        fixed["DeltamSq3l"] = torch.tensor(
            truth["DeltaMSq32"] + truth["DeltaMSq21"], dtype=context.dtype,
        )
    return VacuumOscillationModel(
        context=context, free=free, fixed=fixed, theta23=base.theta23, delta13=base.delta13,
    ), truth


def test_response_matrix_applies_iav_before_lsnl():
    """Regression test for the real Daya Bay IAV-then-LSNL response order.

    The official Daya Bay dagflow/GNA analysis pipeline's own node order is
    raw -> IAV -> LSNL (evis) -> resolution (erec) -- confirmed directly
    from the Collaboration's own CHEP 2026 presentation. Re-derives both the
    correct (IAV first) and the previously-buggy (LSNL first) compositions
    independently here and asserts ``response_matrix`` matches the correct
    one, so an accidental re-swap of the two matrices is caught.
    """
    dT = T_GRID_MEV[1] - T_GRID_MEV[0]
    gaussian_mass = gaussian_response_matrix(
        T_GRID_MEV, TPRIME_GRID_MEV, sigma_MeV(T_GRID_MEV, a=ERES_A, b=ERES_B, c=ERES_C),
    ) * dT
    lsnl_mass = _lsnl_warp_matrix(T_GRID_MEV)
    iav_mass = _iav_mass_matrix(T_GRID_MEV)

    correct_order = (gaussian_mass @ lsnl_mass @ iav_mass) / dT
    buggy_order = (gaussian_mass @ iav_mass @ lsnl_mass) / dT

    actual = response_matrix()

    assert torch.allclose(actual, correct_order)
    assert not torch.allclose(actual, buggy_order, atol=1.0e-6)


def test_response_matrix_conserves_mass_away_from_grid_edges():
    """Trapezoidal-integrating an interior column over T' should recover ~1 (see module docstring)."""
    R = response_matrix()
    for j in (50, 100, 150, 200):
        integral = torch.trapezoid(R[:, j], x=TPRIME_GRID_MEV)
        assert abs(float(integral) - 1.0) < 0.02


def test_precise_cross_section_is_a_small_real_reduction_from_zeroth_order():
    """The order-1/M correction should reduce sigma_tot by a few percent at reactor energies,
    growing with E_nu (Vogel & Beacom's own O(E_nu/M) recoil scaling), never flipping sign."""
    E_nu = torch.linspace(2.0, 8.0, 13, dtype=torch.float64)
    T_grid = torch.linspace(0.0, 12.0, 241, dtype=torch.float64)

    sigma_zeroth = sigma_ibd(E_nu)
    grid_precise = ibd_cross_section_grid_precise(E_nu, T_grid)
    sigma_precise = torch.trapezoid(grid_precise, x=T_grid, dim=-1)

    ratio = sigma_precise / sigma_zeroth
    assert torch.all(ratio > 0.85) and torch.all(ratio < 1.0)
    # The correction grows (ratio decreases) with E_nu.
    assert torch.all(torch.diff(ratio) < 0.0)


def test_experimental_parametrization_round_trips_to_native_theta13_dm31():
    """sin^2(2 theta13)/Delta m^2_ee must be a pure, exact change of variables (see class docstring)."""
    context = RuntimeContext.resolve("cpu", torch.float64)
    osc_model, truth = _build_oscillation_model(context)

    theta13_truth = 0.5 * math.asin(math.sqrt(truth["SinSq2Theta13"]))
    theta12_truth = 0.5 * math.asin(math.sqrt(truth["SinSq2Theta12"]))
    dm31_truth = truth["DeltaMSq32"] + truth["DeltaMSq21"]
    dm2_ee_truth = dm31_truth - math.sin(theta12_truth) ** 2 * truth["DeltaMSq21"]

    model = DayaBayDetectorModel(osc_model, "AD11", normalization_free=True)
    exp_model = DayaBayExperimentalModel(
        model=model,
        theta12=torch.tensor(theta12_truth, dtype=context.dtype),
        dm21=torch.tensor(truth["DeltaMSq21"], dtype=context.dtype),
    )
    assert exp_model.free == ("SinSq2Theta13", "DeltamSqEE", "global_normalization")

    theta_exp = torch.tensor(
        [truth["SinSq2Theta13"], dm2_ee_truth, 1.0], dtype=context.dtype, requires_grad=True,
    )
    theta_native = torch.tensor([theta13_truth, dm31_truth, 1.0], dtype=context.dtype)

    pred_exp = exp_model.predict(theta_exp)
    pred_native = model.predict(theta_native)
    assert torch.allclose(pred_exp, pred_native)

    grad = torch.autograd.grad(pred_exp.sum(), theta_exp)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_rate_only_fit_recovers_a_synthetic_theta13_and_normalization():
    """Closure test: fit synthetic (not real) Poisson counts generated at a known theta13/nu_norm."""
    context = RuntimeContext.resolve("cpu", torch.float64)
    osc_model, truth = _build_oscillation_model(context, free=("theta13",))

    models = tuple(
        DayaBayDetectorModel(osc_model, det, normalization_free=True) for det in DETECTORS
    )

    class RateOnlyModel:
        def __init__(self, models):
            self.models = models

        @property
        def free(self):
            return self.models[0].free

        def predict(self, theta):
            return torch.stack([m.predict(theta).sum() for m in self.models])

    rate_model = RateOnlyModel(models)

    theta13_injected = math.radians(10.0)
    norm_injected = 0.9
    theta_injected = torch.tensor([theta13_injected, norm_injected], dtype=context.dtype)

    generator = torch.Generator().manual_seed(0)
    synthetic_mean = rate_model.predict(theta_injected).detach()
    synthetic_counts = torch.poisson(synthetic_mean, generator=generator)

    theta_start = torch.tensor(
        [theta13_injected + math.radians(2.0), 1.0], dtype=context.dtype, requires_grad=True,
    )
    result = fit_lbfgs(rate_model, theta_start, synthetic_counts, likelihood="poisson", max_iter=200)

    assert abs(math.degrees(float(result.theta_hat[0])) - math.degrees(theta13_injected)) < 0.5
    assert abs(float(result.theta_hat[1]) - norm_injected) < 0.02


def test_lsnl_pulls_at_zero_reproduce_the_nominal_response_matrix():
    """lsnl_pulls=zeros must be identical to the nominal (pull-free) response (see class docstring)."""
    nominal = response_matrix()
    zero_pulls = torch.zeros(4, dtype=T_GRID_MEV.dtype)
    with_zero_pulls = response_matrix(lsnl_pulls=zero_pulls)
    assert torch.allclose(nominal, with_zero_pulls)


def test_lsnl_pulls_gradient_is_finite_and_nonzero():
    """Confirms gradient flows from each of the 4 real LSNL pull nuisances into predicted counts."""
    context = RuntimeContext.resolve("cpu", torch.float64)
    osc_model, truth = _build_oscillation_model(context)
    theta13_truth = 0.5 * math.asin(math.sqrt(truth["SinSq2Theta13"]))
    dm31_truth = truth["DeltaMSq32"] + truth["DeltaMSq21"]

    model = DayaBayDetectorModel(
        osc_model, "AD11", normalization_free=True, lsnl_free=True,
    )
    assert model.free[-4:] == ("lsnl_pull_0", "lsnl_pull_1", "lsnl_pull_2", "lsnl_pull_3")

    theta = torch.tensor(
        [theta13_truth, dm31_truth, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=context.dtype, requires_grad=True,
    )
    predicted = model.predict(theta)
    grad = torch.autograd.grad(predicted.sum(), theta)[0]
    assert torch.isfinite(grad).all()
    assert grad[-4:].abs().sum() > 0


def _build_near_far_model(context, *, normalization_free=False):
    osc_model, truth = _build_oscillation_model(context)
    near_models = tuple(
        DayaBayDetectorModel(osc_model, det, normalization_free=normalization_free) for det in NEAR_DETECTORS
    )
    far_models = tuple(
        DayaBayDetectorModel(osc_model, det, normalization_free=normalization_free) for det in FAR_DETECTORS
    )
    return NearFarRatioDayaBayModel(near_models, far_models), truth


def test_near_far_ratio_is_almost_insensitive_to_global_normalization():
    """A shared global_normalization must cancel (near-)exactly in the far/near ratio (class docstring)."""
    context = RuntimeContext.resolve("cpu", torch.float64)
    model, truth = _build_near_far_model(context, normalization_free=True)
    theta13_truth = 0.5 * math.asin(math.sqrt(truth["SinSq2Theta13"]))
    dm31_truth = truth["DeltaMSq32"] + truth["DeltaMSq21"]

    ratio_norm_1 = model.predict(torch.tensor([theta13_truth, dm31_truth, 1.0], dtype=context.dtype))
    ratio_norm_half = model.predict(torch.tensor([theta13_truth, dm31_truth, 0.5], dtype=context.dtype))

    relative_change = (ratio_norm_1 - ratio_norm_half).abs() / ratio_norm_1
    assert torch.all(relative_change < 0.05)


def test_near_far_ratio_fit_recovers_both_theta13_and_dm31_jointly():
    """Real closure test: unlike the full-shape fit, the near/far ratio fit should recover BOTH
    theta13 and Delta m^2_31 jointly, close to Daya Bay's own published values."""
    context = RuntimeContext.resolve("cpu", torch.float64)
    model, truth = _build_near_far_model(context, normalization_free=False)
    theta13_truth = 0.5 * math.asin(math.sqrt(truth["SinSq2Theta13"]))
    dm31_truth = truth["DeltaMSq32"] + truth["DeltaMSq21"]

    ratio_obs, sigma_obs = NearFarRatioDayaBayModel.real_observed_ratio(dtype=context.dtype)

    theta_start = torch.tensor(
        [theta13_truth + math.radians(1.0), dm31_truth * 1.05], dtype=context.dtype, requires_grad=True,
    )
    result = fit_lbfgs(
        model, theta_start, ratio_obs, sigma_obs, sigma_obs, likelihood="chi2_asymmetric", max_iter=150,
    )

    theta13_hat_deg = math.degrees(float(result.theta_hat[0]))
    dm31_hat = float(result.theta_hat[1])

    assert abs(theta13_hat_deg - math.degrees(theta13_truth)) < 0.5
    assert abs(dm31_hat / dm31_truth - 1.0) < 0.05
