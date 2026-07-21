"""Composed incoherent solar-production to Earth-detector workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.flux import flux_integrated, flux_state
from tpeanuts.core.common.probability import probability_integrated
from tpeanuts.medium.earth.profile import EarthProfile
from tpeanuts.medium.solar.profile import SolarProfile
from tpeanuts.pipeline.earth import (
    EarthDetectorResult,
    propagate_earth_to_detector,
    propagate_earth_to_detector_exposure,
)
from tpeanuts.pipeline.solar import SolarSurfaceResult, propagate_solar_to_surface
from tpeanuts.util.type import TensorLike


@dataclass(frozen=True)
class SolarEarthDetectorResult:
    """Solar-surface and Earth-detector results without duplicated states."""

    solar_surface: SolarSurfaceResult
    earth_detector: EarthDetectorResult
    detector_flux: Optional[torch.Tensor]
    detector_probabilities: Optional[torch.Tensor] = None
    probabilities_energy_averaged: Optional[torch.Tensor] = None
    detector_flux_energy_integrated: Optional[torch.Tensor] = None


@torch.no_grad()
def propagate_solar_to_earth_detector(
    *,
    E_MeV: TensorLike,
    config: PropagationConfig,
    source: str,
    solar_profile: Optional[SolarProfile] = None,
    earth_profile: Optional[EarthProfile] = None,
    eta: Optional[TensorLike] = None,
    source_spectrum: Optional[torch.Tensor] = None,
    integrate_exposure: Optional[bool] = None,
    integrate_energy: bool = False,
    legacy_precision: bool = False,
) -> SolarEarthDetectorResult:
    """Compose solar production and incoherent Earth regeneration."""
    solar = propagate_solar_to_surface(
        E_MeV=E_MeV,
        config=config,
        source=source,
        solar_profile=solar_profile,
        legacy_precision=legacy_precision,
    )
    if integrate_exposure is None:
        integrate_exposure = config.exposure.integrate_exposure
    if integrate_exposure:
        earth = propagate_earth_to_detector_exposure(
            solar.mass_weights,
            E_MeV=solar.E_MeV,
            config=config,
            incident_basis="mass",
            earth_profile=earth_profile,
        )
        probabilities = earth.probabilities_exposure
    else:
        earth = propagate_earth_to_detector(
            solar.mass_weights,
            E_MeV=solar.E_MeV,
            config=config,
            incident_basis="mass",
            earth_profile=earth_profile,
            eta=eta,
        )
        probabilities = earth.probabilities_eta
    detector_flux = flux_state(
        probabilities,
        solar.profile.flux(source),
        source_spectrum,
    )
    probability_energy = None
    detector_rate = None
    if integrate_energy:
        if source_spectrum is None:
            raise ValueError(
                "source_spectrum is required when integrate_energy=True."
            )
        probability_energy = probability_integrated(
            probabilities,
            solar.E_MeV,
            source_spectrum,
            energy_dim=0,
        )
        detector_rate = flux_integrated(
            detector_flux,
            solar.E_MeV,
            energy_dim=0,
        )
    return SolarEarthDetectorResult(
        solar_surface=solar,
        earth_detector=earth,
        detector_flux=detector_flux,
        detector_probabilities=probabilities,
        probabilities_energy_averaged=probability_energy,
        detector_flux_energy_integrated=detector_rate,
    )
