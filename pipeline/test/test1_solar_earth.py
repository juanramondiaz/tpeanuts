"""Tests for the structured solar, Earth, and solar-Earth pipelines."""

import pytest
import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.medium.earth.exposure_table import ExposureParameters
from tpeanuts.medium.solar.geometry import (
    sun_earth_distance_factor,
    sun_earth_distance_factor_averaged,
)
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


def _config_with_exposure() -> PropagationConfig:
    context = RuntimeContext.resolve("cpu", torch.float64)
    oscillation = PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO",
        context=context,
    )
    exposure = ExposureParameters(
        detector_latitude_rad=0.5, exposure_ns=9, exposure_use_cache=False,
    )
    return PropagationConfig(runtime=context, oscillation=oscillation, exposure=exposure)


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


def test_solar_earth_pipeline_integrates_with_profile_spectrum_by_default():
    result = propagate_solar_to_earth_detector(
        E_MeV=[4.0, 8.0],
        config=_config(),
        source="8B",
        eta=[0.4, 1.0],
        integrate_exposure=False,
        integrate_energy=True,
    )
    assert result.probabilities_energy_averaged.shape == (2, 3)
    assert result.detector_flux_energy_integrated.shape == (2, 3)


def test_solar_earth_pipeline_date_applies_sun_earth_distance_factor():
    reference = propagate_solar_to_earth_detector(
        E_MeV=[5.0],
        config=_config(),
        source="8B",
        eta=[0.4, 1.0],
        integrate_exposure=False,
    )
    on_date = propagate_solar_to_earth_detector(
        E_MeV=[5.0],
        config=_config(),
        source="8B",
        eta=[0.4, 1.0],
        integrate_exposure=False,
        date="2026-01-04",
    )
    factor = sun_earth_distance_factor("2026-01-04", dtype=torch.float64)

    torch.testing.assert_close(
        on_date.detector_flux, reference.detector_flux * factor, rtol=1.0e-13, atol=1.0e-13,
    )


def test_solar_earth_pipeline_date_raises_when_integrating_exposure():
    with pytest.raises(ValueError, match="only meaningful"):
        propagate_solar_to_earth_detector(
            E_MeV=[5.0],
            config=_config_with_exposure(),
            source="8B",
            integrate_exposure=True,
            date="2026-01-04",
        )


def test_solar_earth_pipeline_average_sun_earth_distance_requires_exposure():
    with pytest.raises(ValueError, match="requires an exposure-averaged"):
        propagate_solar_to_earth_detector(
            E_MeV=[5.0],
            config=_config(),
            source="8B",
            eta=[0.4, 1.0],
            integrate_exposure=False,
            average_sun_earth_distance=True,
        )


def test_solar_earth_pipeline_average_sun_earth_distance_applies_factor():
    config = _config_with_exposure()
    reference = propagate_solar_to_earth_detector(
        E_MeV=[5.0],
        config=config,
        source="8B",
        integrate_exposure=True,
    )
    averaged = propagate_solar_to_earth_detector(
        E_MeV=[5.0],
        config=config,
        source="8B",
        integrate_exposure=True,
        average_sun_earth_distance=True,
    )
    factor = sun_earth_distance_factor_averaged(
        config.exposure.exposure_d1, config.exposure.exposure_d2, dtype=torch.float64,
    )

    torch.testing.assert_close(
        averaged.detector_flux, reference.detector_flux * factor, rtol=1.0e-13, atol=1.0e-13,
    )
