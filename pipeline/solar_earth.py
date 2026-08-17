"""Composed incoherent solar-production to Earth-detector workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.flux import flux_integrated, flux_state
from tpeanuts.core.common.probability import probability_integrated
from tpeanuts.medium.earth.profile import EarthProfile
from tpeanuts.medium.earth.evolutor import EarthPerturbativeDiagnostics
from tpeanuts.medium.vacuum.solar_geometry import (
    sun_earth_distance_factor,
    sun_earth_distance_factor_averaged,
)
from tpeanuts.medium.solar.profile import SolarMediumProfile
from tpeanuts.source.solar import SolarNeutrinoSource, build_solar_source
from tpeanuts.source.solar import ContinuousSolarSpectrum, SolarLineSpectrum
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
    perturbative_diagnostics: Optional[EarthPerturbativeDiagnostics] = None


@torch.no_grad()
def propagate_solar_to_earth_detector(
    *,
    E_MeV: Optional[TensorLike] = None,
    config: PropagationConfig,
    source: str,
    solar_medium: Optional[SolarMediumProfile] = None,
    solar_source: Optional[SolarNeutrinoSource] = None,
    earth_profile: Optional[EarthProfile] = None,
    eta: Optional[TensorLike] = None,
    source_spectrum: Optional[torch.Tensor] = None,
    integrate_exposure: Optional[bool] = None,
    integrate_energy: bool = False,
    solar_method: str = "adiabatic_approximated",
    legacy_precision: bool = False,
    include_matter_nc: Optional[bool] = None,
    date: Optional[str] = None,
    average_sun_earth_distance: bool = False,
    return_diagnostics: bool = False,
) -> SolarEarthDetectorResult:
    """Compose solar production and incoherent Earth regeneration.

    ``return_diagnostics=True`` propagates the analytical Earth first-order
    diagnostics into both ``earth_detector.perturbative_diagnostics`` and the
    convenience field ``perturbative_diagnostics`` of this result.

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
    ``detector_flux_energy_integrated`` (see ``medium.vacuum.solar_geometry``):
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
            ``medium.vacuum.solar_geometry.sun_earth_distance_factor_averaged``),
            so a single exposure window consistently accounts for both the
            detector's day/night geometry and the Sun-Earth distance over
            the same period. Raises if ``integrate_exposure`` resolves to
            False (there is no day-of-year window to average over).
    """
    solar_source = build_solar_source(
        solar_source, params=config.solar.source, context=config.runtime,
    )
    if E_MeV is None:
        spectrum = solar_source.spectrum_table(source)
        if not isinstance(spectrum, SolarLineSpectrum):
            raise ValueError(
                "E_MeV may be omitted only for a discrete solar line source; "
                f"{source!r} has a continuous spectrum."
            )
        E_MeV = spectrum.energy_MeV

    solar = propagate_solar_to_surface(
        E_MeV=E_MeV,
        config=config,
        source=source,
        solar_medium=solar_medium,
        solar_source=solar_source,
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
            return_diagnostics=return_diagnostics,
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
            return_diagnostics=return_diagnostics,
        )
        probabilities = earth.probabilities_eta
    spectrum_model = solar.source.spectrum_table(source) if source_spectrum is None else None
    if isinstance(spectrum_model, SolarLineSpectrum):
        if solar.E_MeV.shape != spectrum_model.energy_MeV.shape or not torch.allclose(
            solar.E_MeV, spectrum_model.energy_MeV,
        ):
            raise ValueError(
                "A solar line source must be propagated at its exact line energies; "
                "pass E_MeV=solar_source.spectrum_table(source).energy_MeV."
            )
        resolved_spectrum = spectrum_model.weights
        # flux_state's internal broadcasting lifts resolved_spectrum (rank 1)
        # against probabilities' actual rank (2 without an eta axis, 3 with
        # one) by appending trailing singleton axes -- a plain
        # resolved_spectrum[..., None] only appends one and silently
        # misaligns the line-weight axis against the eta axis whenever
        # probabilities is rank 3 (see propagate_earth_to_detector's
        # multi-eta probabilities_eta).
        weighted_probabilities = flux_state(probabilities, 1.0, resolved_spectrum)
        detector_flux = weighted_probabilities * solar.source.total_flux(source)
    else:
        resolved_spectrum = (
            spectrum_model.evaluate(solar.E_MeV)
            if isinstance(spectrum_model, ContinuousSolarSpectrum)
            else source_spectrum
        )
        detector_flux = flux_state(
            probabilities, solar.source.total_flux(source), resolved_spectrum,
        )
    if date is not None:
        detector_flux = detector_flux * solar.source.flux_reference_distance_au ** 2 * sun_earth_distance_factor(
            date, device=detector_flux.device, dtype=detector_flux.dtype,
        )
    elif average_sun_earth_distance:
        detector_flux = detector_flux * solar.source.flux_reference_distance_au ** 2 * sun_earth_distance_factor_averaged(
            config.exposure.exposure_d1,
            config.exposure.exposure_d2,
            device=detector_flux.device,
            dtype=detector_flux.dtype,
        )
    probability_energy = None
    detector_rate = None
    if integrate_energy:
        if isinstance(spectrum_model, SolarLineSpectrum):
            probability_energy = weighted_probabilities.sum(dim=0)
            detector_rate = detector_flux.sum(dim=0)
        else:
            probability_energy = probability_integrated(
                probabilities, solar.E_MeV, resolved_spectrum, energy_dim=0,
            )
            detector_rate = flux_integrated(detector_flux, solar.E_MeV, energy_dim=0)
    return SolarEarthDetectorResult(
        solar_surface=solar,
        earth_detector=earth,
        detector_flux=detector_flux,
        detector_probabilities=probabilities,
        probabilities_energy_averaged=probability_energy,
        detector_flux_energy_integrated=detector_rate,
        perturbative_diagnostics=earth.perturbative_diagnostics,
    )
