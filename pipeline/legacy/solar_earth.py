"""Composed legacy solar-to-Earth detector validation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.flux import flux_state
from tpeanuts.pipeline.legacy.earth import (
    LegacyEarthDetectorResult,
    propagate_legacy_earth_to_detector,
    propagate_legacy_earth_to_detector_integrated,
)
from tpeanuts.pipeline.legacy.solar import (
    LegacySolarSurfaceResult,
    propagate_legacy_solar_to_surface,
)
from tpeanuts.util.type import TensorLike


@dataclass(frozen=True)
class LegacySolarEarthDetectorResult:
    """Legacy solar and Earth results joined without duplicated states."""

    solar_surface: LegacySolarSurfaceResult
    earth_detector: LegacyEarthDetectorResult
    detector_flux: torch.Tensor


@torch.no_grad()
def propagate_legacy_solar_to_earth_detector(
    *,
    E_MeV: TensorLike,
    config: PropagationConfig,
    source: str,
    solar_model: Optional[object] = None,
    earth_density: Optional[object] = None,
    eta: Optional[TensorLike] = None,
    source_spectrum: Optional[torch.Tensor] = None,
    integrate_exposure: Optional[bool] = None,
    exposure_normalized: bool = True,
    integration_method: Literal[
        "peanuts_integrated", "legacy_rectangle", "trapezoid"
    ] = "peanuts_integrated",
    validate_psolar: bool = False,
) -> LegacySolarEarthDetectorResult:
    """Compose legacy solar mass production and Earth regeneration."""
    solar = propagate_legacy_solar_to_surface(
        E_MeV=E_MeV,
        config=config,
        source=source,
        solar_model=solar_model,
        validate_psolar=validate_psolar,
    )
    if integrate_exposure is None:
        integrate_exposure = config.exposure.integrate_exposure
    if integrate_exposure:
        earth = propagate_legacy_earth_to_detector_integrated(
            solar.mass_weights,
            E_MeV=solar.E_MeV,
            config=config,
            incident_basis="mass",
            earth_density=earth_density,
            eta=eta,
            exposure_normalized=exposure_normalized,
            integration_method=integration_method,
        )
        probabilities = earth.probabilities_integrated
    else:
        earth = propagate_legacy_earth_to_detector(
            solar.mass_weights,
            E_MeV=solar.E_MeV,
            config=config,
            incident_basis="mass",
            earth_density=earth_density,
            eta=eta,
            exposure_normalized=exposure_normalized,
        )
        probabilities = earth.probabilities_eta
    detector_flux = flux_state(
        probabilities,
        float(solar.legacy_model.flux(source)),
        source_spectrum,
    )
    return LegacySolarEarthDetectorResult(
        solar_surface=solar,
        earth_detector=earth,
        detector_flux=detector_flux,
    )
