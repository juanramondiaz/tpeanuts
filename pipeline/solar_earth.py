"""Composed incoherent solar-production to Earth-detector workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.flux import flux_integrated, flux_state
from tpeanuts.core.common.probability import probability_integrated
from tpeanuts.medium.earth.profile import EarthProfile
from tpeanuts.medium.solar.geometry import (
    sun_earth_distance_factor,
    sun_earth_distance_factor_averaged,
)
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
    solar_method: str = "adiabatic",
    legacy_precision: bool = False,
    include_matter_nc: Optional[bool] = None,
    date: Optional[str] = None,
    average_sun_earth_distance: bool = False,
) -> SolarEarthDetectorResult:
    """Compose solar production and incoherent Earth regeneration.

    ``solar_method`` and ``include_matter_nc`` (both new) are forwarded to
    ``pipeline.solar.propagate_solar_to_surface`` for the solar leg only --
    ``include_matter_nc: Optional[bool] = None`` auto-resolves per-call (see
    ``core.common.oscillation.resolve_include_matter_nc``): the 3+1 sterile
    extension's neutral-current matter term is included whenever
    ``config.oscillation`` is sterile and the solar profile has
    neutron-density data available, and omitted otherwise (with a
    ``RuntimeWarning`` if sterile was requested but the data is missing).
    Always omitted for the plain 3-flavour case. The Earth leg
    (``propagate_earth_to_detector*``) is unaffected by this argument -- it
    has no analogous parameter exposed here yet.

    ``date``/``average_sun_earth_distance`` apply the Sun-Earth distance
    modulation ``(1 AU / R)^2`` to ``detector_flux``/
    ``detector_flux_energy_integrated`` (see ``medium.solar.geometry``):
    solar-model flux tables are normalized to 1 AU, but Earth's elliptical
    orbit makes the physically received flux vary by about +-3.4% over the
    year. Exactly one of the two is meaningful for a given call, matching
    whether ``integrate_exposure`` resolves to a single propagation or an
    exposure-averaged one:

        date
            An ISO ``"YYYY-MM-DD"`` calendar date giving the instantaneous
            factor for a single, non-exposure-averaged propagation. Raises
            if ``integrate_exposure`` resolves to True (a single date is
            not meaningful once already averaging over a day-of-year
            window; use ``average_sun_earth_distance`` instead).
        average_sun_earth_distance
            If True, averages the factor uniformly over the *same*
            day-of-year window already used for the nadir-angle exposure
            average, ``config.exposure.exposure_d1``/``exposure_d2`` (see
            ``medium.solar.geometry.sun_earth_distance_factor_averaged``),
            so a single exposure window consistently accounts for both the
            detector's day/night geometry and the Sun-Earth distance over
            the same period. Raises if ``integrate_exposure`` resolves to
            False (there is no day-of-year window to average over).
    """
    solar = propagate_solar_to_surface(
        E_MeV=E_MeV,
        config=config,
        source=source,
        solar_profile=solar_profile,
        method=solar_method,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
    )
    if integrate_exposure is None:
        integrate_exposure = config.exposure.integrate_exposure
    if date is not None and integrate_exposure:
        raise ValueError(
            "date is only meaningful for a single (non-exposure-averaged) "
            "propagation; for an exposure-averaged period, pass "
            "average_sun_earth_distance=True instead, which reuses "
            "config.exposure.exposure_d1/exposure_d2."
        )
    if average_sun_earth_distance and not integrate_exposure:
        raise ValueError(
            "average_sun_earth_distance=True requires an exposure-averaged "
            "propagation (integrate_exposure resolving to True); pass an "
            "explicit date= instead for a single propagation."
        )
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
    resolved_spectrum = (
        solar.profile.spectrum(source, solar.E_MeV)
        if source_spectrum is None else source_spectrum
    )
    detector_flux = flux_state(
        probabilities,
        solar.profile.flux(source),
        resolved_spectrum,
    )
    if date is not None:
        detector_flux = detector_flux * sun_earth_distance_factor(
            date, device=detector_flux.device, dtype=detector_flux.dtype,
        )
    elif average_sun_earth_distance:
        detector_flux = detector_flux * sun_earth_distance_factor_averaged(
            config.exposure.exposure_d1,
            config.exposure.exposure_d2,
            device=detector_flux.device,
            dtype=detector_flux.dtype,
        )
    probability_energy = None
    detector_rate = None
    if integrate_energy:
        probability_energy = probability_integrated(
            probabilities,
            solar.E_MeV,
            resolved_spectrum,
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
