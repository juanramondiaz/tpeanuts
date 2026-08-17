#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#  This module is part of the Master's Thesis (MSc Dissertation):
#  - Fast Simulation of Neutrino Oscillations in Matter
#
#  Author:
#      Juan Ramon Diaz Santos <diazjuan@alumni.uv.es>
#
#  Supervisors:
#      Roberto Ruiz de Austri Bazan <rruiz@ific.uv.es>
#      Michele Lucente <michele.lucente@unibo.it>
#
#  Date:
#      June 2026
# =============================================================================

"""
Differentiable SNO day/night event-count model: oscillation x Earth x detector.

``SNODayNightModel`` is the composition layer for SNO, mirroring
``tpeanuts.detector.borexino.inference_model.BorexinoEventRateModel`` but
adding a stage neither Borexino nor any earlier notebook in this project's
inference series exercises: Earth matter regeneration. It calls
``tpeanuts.medium.solar.probability.solar_probability_mass`` (production)
and ``tpeanuts.medium.earth.probability.earth_probability_state`` (Earth
crossing) directly rather than going through
``tpeanuts.pipeline.solar_earth`` (whose ``propagate_solar_to_earth_detector``
is wrapped in ``@torch.no_grad()``) -- the same bypass reason as every other
composition class in this package. Reaching gradients through the Earth leg
specifically required its own ``@torch.no_grad()`` audit
(``medium.earth.probability``, ``medium.earth.evolutor``,
``core.perturbative.evolutor.evolutor_perturbative_segment`` -- 10
decorators across 3 files), on top of the ``core.SM.sm_pmns``/
``core.common.pmns`` fixes the solar-only notebooks already needed.

**Predicts CC + ES + NC, summed, matching SNO's own raw candidate counts.**
SNO Phase I did not tag events by reaction: ``day_counts``/``night_counts``
in ``data/detector/sno/observation/day_night_spectrum.csv`` are raw candidates
surviving the electron-energy/fiducial-volume cuts (CC and ES recoil
electrons, plus NC neutron-capture-gamma Compton electrons above
threshold), and the real SNO analysis separated the three rates
*statistically*, fitting a combined probability-density function in
reconstructed energy, the electron's angle to the Sun, and volume-weighted
radius -- an event-by-event PDF decomposition this project does not
reproduce. Summing ``detector.sno.event_rate``'s three forward-modeled
channels (``cc_event_rate`` + ``es_event_rate`` + ``nc_event_rate``) is the
closest match to that combined candidate count without redoing SNO's own
statistical extraction.

Module contents:
    SNODayNightModel
        Wraps a SolarSMOscillationModel, a solar profile/source, an Earth
        profile, and day/night nadir-angle exposure weights, predicting
        concatenated (day, night) CC+ES+NC counts per bin from a
        free-parameter vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from tpeanuts.detector.sno.io import load_coszenith_exposure, total_livetime_days
import tpeanuts.detector.sno.parameters as sno_parameters
from tpeanuts.detector.sno.event_rate import cc_event_rate, es_event_rate, nc_event_rate
from tpeanuts.medium.earth.probability import earth_probability_state
from tpeanuts.inference.solar_model import SolarSMOscillationModel
from tpeanuts.medium.solar.probability import solar_probability_mass
from tpeanuts.medium.solar.profile import SolarMediumProfile
from tpeanuts.source.solar import SolarLineSpectrum, SolarNeutrinoSource


@dataclass(frozen=True)
class SNODayNightModel:
    """Predicted (day, night) SNO CC+ES+NC counts per bin, as a function of oscillation parameters.

    Parameters
    ----------
    oscillation_model:
        A ``SolarSMOscillationModel`` supplying ``free``/``oscillation(theta)``.
    solar_medium:
        Solar density profile (``medium.solar.profile.SolarMediumProfile``).
    solar_source:
        Solar production source (production/flux/spectrum tables,
        ``source.solar.SolarNeutrinoSource``).
    earth_profile:
        Earth density profile (``tpeanuts.medium.earth.profile
        .build_earth_profile``'s return value).
    source_names:
        Continuous solar source keys whose CC, ES, and NC contributions are
        summed (normally ``("8B", "hep")``).
    bin_edges_MeV:
        Observed electron-energy bin edges, shared by day and night.
    eta_day, weight_day, eta_night, weight_night:
        Nadir-angle grids and *normalized* (each summing to 1) averaging
        weights for the day and night periods, e.g. from
        ``SNODayNightModel.from_real_exposure``.
    exposure_days_day, exposure_days_night:
        Live time in days for each period.
    detector_depth_m:
        Detector depth below the Earth's surface, in metres (SNO's
        approximate depth; see ``from_real_exposure`` for the exact value
        used and its caveat).
    E_nu_grid_MeV:
        True neutrino energy grid P_ee is evaluated on before folding.
        Defaults to ``detector.sno.parameters.E_NU_GRID_MEV``.
    background_day, background_night:
        Optional real *instrumental* background counts per bin (the
        "neutron"/"low_energy" columns of ``detector.sno.backgrounds`` --
        contamination, not the NC neutrino signal, which is its own
        channel, see module docstring), added once to the summed CC+ES+NC
        signal.
    """

    oscillation_model: SolarSMOscillationModel
    solar_medium: SolarMediumProfile
    solar_source: SolarNeutrinoSource
    earth_profile: object
    source_names: tuple[str, ...]
    bin_edges_MeV: torch.Tensor
    eta_day: torch.Tensor
    weight_day: torch.Tensor
    eta_night: torch.Tensor
    weight_night: torch.Tensor
    exposure_days_day: float
    exposure_days_night: float
    detector_depth_m: float = 2039.0
    E_nu_grid_MeV: torch.Tensor = field(
        default_factory=lambda: sno_parameters.E_NU_GRID_MEV
    )
    background_day: Optional[torch.Tensor] = None
    background_night: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        """Reject solar sources tabulated as discrete lines.

        SNO's own true-neutrino-energy grid
        (``detector.sno.parameters.E_NU_GRID_MEV``) starts at 1.0 MeV, and
        its real CC/ES analysis threshold sits several MeV higher still --
        above both 7Be lines (0.384/0.862 MeV) and even pep's single line
        (1.442 MeV). Unlike Borexino, no solar source SNO actually measures
        is a line spectrum, so this model's ``cc_event_rate``/
        ``es_event_rate``/``nc_event_rate`` folds were never extended to
        integrate one (compare
        ``detector.borexino.inference_model.BorexinoEventRateModel``, which
        does support them via ``detector.borexino.event_rate
        .line_event_rate``). Failing fast here, with an explanation, is
        preferable to the opaque ``TypeError`` that
        ``SolarNeutrinoSource.spectrum()`` would otherwise raise deep inside
        ``predict()``.

        Raises:
            ValueError: If ``source_name`` has a tabulated spectrum and it
                is a ``SolarLineSpectrum``.
        """
        if not self.source_names:
            raise ValueError("source_names must contain at least one solar source")
        line_sources = [name for name in self.source_names if self.solar_source.has_spectrum(name) and isinstance(
            self.solar_source.spectrum_table(name), SolarLineSpectrum,
        )]
        if line_sources:
            raise ValueError(
                f"SNODayNightModel does not support line-spectrum solar "
                f"sources ({line_sources!r} are tabulated as "
                "discrete lines). SNO's real analysis threshold lies above "
                "every solar line (7Be/pep), so only continuous sources "
                "(e.g. '8B', 'hep') are physically meaningful here; see "
                "detector.borexino.inference_model.BorexinoEventRateModel "
                "for line-spectrum support."
            )

    @property
    def free(self) -> tuple[str, ...]:
        """Free oscillation-parameter names, forwarded from ``oscillation_model``."""
        return self.oscillation_model.free

    def day_night_probabilities(
        self, theta: torch.Tensor, source_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exposure-averaged flavour probabilities for the day and night periods.

        Args:
            theta: 1-D free-parameter tensor, see
                ``self.oscillation_model.oscillation``.

        Returns:
            ``(p_day, p_night)``, each shaped ``(n_E, n_flavour)`` and
            differentiable with respect to ``theta``.
        """
        oscillation = self.oscillation_model.oscillation(theta)
        mass_weights = solar_probability_mass(
            oscillation, self.E_nu_grid_MeV, self.solar_medium, self.solar_source, source_name,
        )  # (n_E, n_mass)

        p_day_eta = earth_probability_state(
            mass_weights, self.earth_profile, oscillation,
            self.E_nu_grid_MeV, self.eta_day, self.detector_depth_m,
            massbasis=True, method="analytical",
        )  # (n_E, n_eta_day, n_flavour)
        p_night_eta = earth_probability_state(
            mass_weights, self.earth_profile, oscillation,
            self.E_nu_grid_MeV, self.eta_night, self.detector_depth_m,
            massbasis=True, method="analytical",
        )  # (n_E, n_eta_night, n_flavour)

        p_day = (p_day_eta * self.weight_day[None, :, None]).sum(dim=-2)
        p_night = (p_night_eta * self.weight_night[None, :, None]).sum(dim=-2)
        return p_day, p_night

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """Predict concatenated (day, night) SNO CC+ES+NC counts per bin from ``theta``.

        Sums all three of ``detector.sno.event_rate``'s composable channels
        (see module docstring for why: SNO's own raw candidate counts are
        not separated by reaction either) before adding the instrumental
        background once.

        Args:
            theta: 1-D free-parameter tensor.

        Returns:
            Predicted counts, shape ``(2 * n_bins,)``: day bins followed by
            night bins, matching how real data should be concatenated for
            comparison (``torch.cat([day_counts, night_counts])``).
        """
        n_bins = self.bin_edges_MeV.numel() - 1
        counts_day = torch.zeros(n_bins, device=theta.device, dtype=theta.dtype)
        counts_night = torch.zeros_like(counts_day)
        for source_name in self.source_names:
            p_day, p_night = self.day_night_probabilities(theta, source_name)
            flux_tot_MeV = self.solar_source.total_flux(source_name) * self.solar_source.spectrum(
                source_name, self.E_nu_grid_MeV,
            )
            counts_day = counts_day + (
                cc_event_rate(
                    p_day, flux_tot_MeV, self.bin_edges_MeV,
                    E_nu_grid_MeV=self.E_nu_grid_MeV, exposure_days=self.exposure_days_day,
                )
                + es_event_rate(
                    p_day, flux_tot_MeV, self.bin_edges_MeV,
                    E_nu_grid_MeV=self.E_nu_grid_MeV, exposure_days=self.exposure_days_day,
                )
                + nc_event_rate(
                    flux_tot_MeV, self.bin_edges_MeV, probabilities=p_day,
                    E_nu_grid_MeV=self.E_nu_grid_MeV, exposure_days=self.exposure_days_day,
                )
            )
            counts_night = counts_night + (
                cc_event_rate(
                    p_night, flux_tot_MeV, self.bin_edges_MeV,
                    E_nu_grid_MeV=self.E_nu_grid_MeV, exposure_days=self.exposure_days_night,
                )
                + es_event_rate(
                    p_night, flux_tot_MeV, self.bin_edges_MeV,
                    E_nu_grid_MeV=self.E_nu_grid_MeV, exposure_days=self.exposure_days_night,
                )
                + nc_event_rate(
                    flux_tot_MeV, self.bin_edges_MeV, probabilities=p_night,
                    E_nu_grid_MeV=self.E_nu_grid_MeV, exposure_days=self.exposure_days_night,
                )
            )
        if self.background_day is not None:
            counts_day = counts_day + self.background_day
        if self.background_night is not None:
            counts_night = counts_night + self.background_night

        return torch.cat([counts_day, counts_night])

    @classmethod
    def from_real_exposure(
        cls,
        oscillation_model: SolarSMOscillationModel,
        solar_medium: SolarMediumProfile,
        solar_source: SolarNeutrinoSource,
        earth_profile: object,
        source_names: tuple[str, ...],
        bin_edges_MeV: torch.Tensor,
        *,
        stride: int = 12,
        detector_depth_m: float = 2039.0,
        background_day: Optional[torch.Tensor] = None,
        background_night: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> "SNODayNightModel":
        """Build a model from the real SNO cos(zenith) exposure table.

        Downsamples the real 480-point nadir-angle exposure
        (``detector.sno.io.load_coszenith_exposure``) by a factor
        ``stride`` for computational tractability (each retained eta point
        costs one Earth-crossing evolutor evaluation per energy), and
        renormalizes the day/night weight halves to each sum to 1 (a plain
        weighted average over the retained points, not a trapezoidal
        integral -- adequate given the exposure varies smoothly over
        cos(zenith), not exercised at full resolution here). Real
        (day, night) live times come from
        ``detector.sno.io.total_livetime_days`` (exact, independent of the
        downsampling above).

        Args:
            oscillation_model: A ``SolarSMOscillationModel``.
            solar_medium: Solar density profile.
            solar_source: Solar production source.
            earth_profile: Earth density profile.
            source_name: Solar source key.
            bin_edges_MeV: Observed electron-energy bin edges.
            stride: Keep every ``stride``-th point of the real 480-point
                exposure grid.
            detector_depth_m: Detector depth below the Earth's surface, in
                metres. 2039 m is SNO's commonly quoted approximate depth
                (Sudbury mine); not verified here against a primary SNO
                paper -- the physical effect of this value on Earth-crossing
                geometry is tiny relative to the Earth's ~6371 km radius.
            background_day, background_night: Optional real background
                counts per bin (see ``detector.sno.backgrounds``).
            device: Target torch device.
            dtype: Target real dtype.

        Returns:
            A configured ``SNODayNightModel``.
        """
        
        eta, exposure = load_coszenith_exposure(device=device, dtype=dtype)
        eta = eta[::stride]
        exposure = exposure[::stride].clamp_min(0.0)

        night_mask = eta < (torch.pi / 2)
        day_mask = ~night_mask

        eta_night = eta[night_mask]
        weight_night = exposure[night_mask]
        weight_night = weight_night / weight_night.sum().clamp_min(torch.finfo(dtype).tiny)

        eta_day = eta[day_mask]
        weight_day = exposure[day_mask]
        weight_day = weight_day / weight_day.sum().clamp_min(torch.finfo(dtype).tiny)

        exposure_days_day, exposure_days_night = total_livetime_days()

        return cls(
            oscillation_model=oscillation_model,
            solar_medium=solar_medium,
            solar_source=solar_source,
            earth_profile=earth_profile,
            source_names=source_names,
            bin_edges_MeV=bin_edges_MeV,
            eta_day=eta_day, weight_day=weight_day,
            eta_night=eta_night, weight_night=weight_night,
            exposure_days_day=exposure_days_day,
            exposure_days_night=exposure_days_night,
            detector_depth_m=detector_depth_m,
            background_day=background_day,
            background_night=background_night,
        )
