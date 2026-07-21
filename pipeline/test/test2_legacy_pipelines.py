"""Tests for the structured legacy validation pipelines."""

import pytest
import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.pipeline.legacy import (
    LegacySolarEarthDetectorResult,
    propagate_legacy_solar_to_earth_detector,
    propagate_legacy_solar_to_surface,
)
from tpeanuts.util.context import RuntimeContext


def _config(preset: str = "_SM_NUFIT52_NO") -> PropagationConfig:
    context = RuntimeContext.resolve("cpu", torch.float64)
    oscillation = PropagationConfig.oscillation_parameters_from_preset(
        preset, context=context
    )
    return PropagationConfig(runtime=context, oscillation=oscillation)


def test_legacy_solar_earth_pipeline_preserves_structured_boundaries():
    result = propagate_legacy_solar_to_earth_detector(
        E_MeV=[5.0],
        config=_config(),
        source="8B",
        eta=[0.4, 1.0],
        integrate_exposure=False,
        validate_psolar=True,
    )

    assert isinstance(result, LegacySolarEarthDetectorResult)
    assert result.solar_surface.mass_weights.shape == (1, 3)
    assert result.solar_surface.flavour_probabilities.shape == (1, 3)
    assert result.earth_detector.incident_state is result.solar_surface.mass_weights
    assert result.earth_detector.probabilities_eta.shape == (1, 2, 3)
    assert result.detector_flux.shape == (1, 2, 3)


def test_legacy_pipeline_rejects_sterile_configuration_explicitly():
    with pytest.raises(ValueError, match="three-flavour"):
        propagate_legacy_solar_to_surface(
            E_MeV=[5.0],
            config=_config("sterile_3p1_null_mixing"),
            source="8B",
        )
