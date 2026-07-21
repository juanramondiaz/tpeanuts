"""Tests for the structured solar, Earth, and solar-Earth pipelines."""

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.pipeline.solar import SolarSurfaceResult, propagate_solar_to_surface
from tpeanuts.pipeline.solar_earth import (
    SolarEarthDetectorResult,
    propagate_solar_to_earth_detector,
)
from tpeanuts.util.context import RuntimeContext


def _config() -> PropagationConfig:
    context = RuntimeContext.resolve("cpu", torch.float64)
    oscillation = PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO",
        context=context,
    )
    return PropagationConfig(runtime=context, oscillation=oscillation)


def test_solar_pipeline_returns_one_normalized_mass_mixture_per_energy():
    result = propagate_solar_to_surface(
        E_MeV=[2.0, 5.0],
        config=_config(),
        source="8B",
    )

    assert isinstance(result, SolarSurfaceResult)
    assert result.mass_weights.shape == (2, 3)
    torch.testing.assert_close(
        result.mass_weights.sum(dim=-1),
        torch.ones(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.flavour_probabilities.sum(dim=-1),
        torch.ones(2, dtype=torch.float64),
    )


def test_solar_earth_pipeline_composes_surface_weights_without_duplicate_state():
    result = propagate_solar_to_earth_detector(
        E_MeV=[5.0],
        config=_config(),
        source="8B",
        eta=[0.4, 1.0],
        integrate_exposure=False,
    )

    assert isinstance(result, SolarEarthDetectorResult)
    assert result.earth_detector.incident_state is result.solar_surface.mass_weights
    assert result.earth_detector.probabilities_eta.shape == (1, 2, 3)
    assert result.detector_flux.shape == (1, 2, 3)
    assert result.detector_probabilities is result.earth_detector.probabilities_eta


def test_solar_earth_pipeline_optionally_integrates_energy_observables():
    spectrum = torch.tensor([0.4, 0.6], dtype=torch.float64)
    result = propagate_solar_to_earth_detector(
        E_MeV=[4.0, 8.0],
        config=_config(),
        source="8B",
        eta=[0.4, 1.0],
        source_spectrum=spectrum,
        integrate_exposure=False,
        integrate_energy=True,
    )
    assert result.probabilities_energy_averaged.shape == (2, 3)
    assert result.detector_flux_energy_integrated.shape == (2, 3)
