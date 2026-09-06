"""Tests for tpeanuts.inference.model_solar.

SolarNSIOscillationModel used to build its NSIConfig via
NSIConfig.from_raw_epsilon (see core.BSM.bsm_nsi._hermitian_3x3's rewrite to
differentiable torch ops); this file is the first automated coverage of
that model, added alongside the migration off from_raw_epsilon.
"""

import torch

from tpeanuts.inference.model_solar import (
    NSI_FREE_PARAM_KEYS,
    SolarNSIOscillationModel,
    SolarPointModel,
    SolarSMOscillationModel,
)
from tpeanuts.medium.solar.profile import build_solar_medium
from tpeanuts.source.solar import build_solar_source
from tpeanuts.util.context import RuntimeContext

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context() -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, DTYPE)


def make_medium_source():
    ctx = make_context()
    medium = build_solar_medium(None, context=ctx)
    source = build_solar_source(None, context=ctx)
    return medium, source


def test_solar_nsi_oscillation_model_predicts_finite_probabilities():
    ctx = make_context()
    model, theta0 = SolarNSIOscillationModel.from_preset(
        "_SM_NUFIT52_NO", free=("theta12", "eps_ee"), eps_ee0=0.1, context=ctx,
    )
    medium, source = make_medium_source()

    p_ee = model.predict_pee(theta0, medium, source, ["8B"], [8.0])

    assert p_ee.shape == (1,)
    assert torch.isfinite(p_ee).all()
    assert 0.0 <= p_ee.item() <= 1.0


def test_solar_nsi_oscillation_model_gradient_flows_to_eps_ee():
    """The whole point of building NSIConfig from the eps_ee scalar field
    directly (instead of from_raw_epsilon) is that this gradient exists and
    is finite/non-trivial -- see core.BSM.bsm_nsi._hermitian_3x3."""
    ctx = make_context()
    model, theta0 = SolarNSIOscillationModel.from_preset(
        "_SM_NUFIT52_NO", free=("eps_ee",), eps_ee0=0.15, context=ctx,
    )
    medium, source = make_medium_source()

    p_ee = model.predict_pee(theta0, medium, source, ["8B", "8B"], [3.0, 9.0])
    loss = p_ee.sum()
    loss.backward()

    assert theta0.grad is not None
    assert torch.isfinite(theta0.grad).all()
    assert torch.any(theta0.grad != 0.0), "d(P_ee)/d(eps_ee) must be non-zero for a generic point"


def test_solar_nsi_oscillation_model_eps_ee_zero_matches_sm_model():
    """eps_ee=0 (NSI SM limit) must reproduce the plain SM model's P_ee
    (both go through hamiltonian_flavour/adiabatic_exact identically once
    epsilon is exactly zero)."""
    ctx = make_context()
    sm_model, sm_theta0 = SolarSMOscillationModel.from_preset(
        "_SM_NUFIT52_NO", free=("theta12", "DeltamSq21"), context=ctx,
    )
    nsi_model, _ = SolarNSIOscillationModel.from_preset(
        "_SM_NUFIT52_NO", free=("theta12", "DeltamSq21"), eps_ee0=0.0, context=ctx,
    )
    nsi_theta0 = sm_theta0.detach().clone().requires_grad_(True)
    medium, source = make_medium_source()

    p_sm = sm_model.predict_pee(sm_theta0, medium, source, ["8B"], [8.0])
    p_nsi = nsi_model.predict_pee(nsi_theta0, medium, source, ["8B"], [8.0])

    torch.testing.assert_close(p_sm, p_nsi, atol=5.0e-3, rtol=5.0e-3)


def test_solar_point_model_wraps_nsi_model_predict():
    ctx = make_context()
    model, theta0 = SolarNSIOscillationModel.from_preset(
        "_SM_NUFIT52_NO", free=NSI_FREE_PARAM_KEYS, eps_ee0=0.05, context=ctx,
    )
    medium, source = make_medium_source()
    point_model = SolarPointModel(
        oscillation_model=model, medium=medium, source=source,
        sources=("8B",), energies_MeV=(8.0,),
    )

    assert point_model.free == model.free
    prediction = point_model.predict(theta0)
    direct = model.predict_pee(theta0, medium, source, ("8B",), (8.0,))
    torch.testing.assert_close(prediction, direct)
