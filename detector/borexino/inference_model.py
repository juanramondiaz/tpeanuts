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
Differentiable Borexino event-count model: composes oscillation x detector.

``BorexinoEventRateModel`` wraps a ``SolarSMOscillationModel``'s P_ee(E)
curve with ``tpeanuts.detector.borexino.event_rate.event_rate``, giving a
``fit_lbfgs``-compatible ``predict(theta)`` method that actually returns
per-bin event counts (not P_ee) -- so it can be fit directly with
``likelihood="poisson"``.

Module contents:
    BorexinoEventRateModel
        Wraps a SolarSMOscillationModel and a solar profile/source,
        predicting Borexino counts per bin from a free-parameter vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

import tpeanuts.detector.borexino.parameters as borexino_parameters
from tpeanuts.detector.borexino.backgrounds import backgrounds_MeV
from tpeanuts.detector.borexino.event_rate import event_rate, line_event_rate
from tpeanuts.inference.solar_model import SolarSMOscillationModel
from tpeanuts.medium.solar.probability import solar_probability_state
from tpeanuts.medium.solar.profile import SolarMediumProfile
from tpeanuts.source.solar import SolarNeutrinoSource
from tpeanuts.source.solar import SolarLineSpectrum


@dataclass(frozen=True)
class BorexinoEventRateModel:
    """Predicted Borexino counts per bin, as a function of oscillation parameters.

    Parameters
    ----------
    oscillation_model:
        A ``SolarSMOscillationModel`` supplying ``free``/``oscillation(theta)``
        (theta12/theta13/DeltamSq21/DeltamSq3l -- the plain SM path; NSI is
        not wired in here yet, see ``tpeanuts.inference.solar_model
        .SolarNSIOscillationModel`` for the differentiable NSI oscillation
        machinery this could be extended to reuse).
    medium:
        Solar density profile (``medium.solar.profile.SolarMediumProfile``).
    source:
        Solar production source (radius/flux/spectrum tables,
        ``source.solar.SolarNeutrinoSource``).
    source_names:
        Solar source keys whose signal contributions are summed before the
        detector background is added once.
    bin_edges_MeV:
        Observed-spectrum bin edges passed to
        ``detector.borexino.event_rate.event_rate``.
    exposure_days:
        Exposure in days, forwarded to ``event_rate``. Defaults to the 1-day
        reference normalization
        (``detector.borexino.parameters.REFERENCE_EXPOSURE_DAYS``); pass a
        realistic multi-year value (e.g. 1000) for anything resembling
        actual Borexino statistics -- 1 day of a single source's ES rate is
        typically well under 1 event.
    E_nu_grid_MeV:
        True neutrino energy grid P_ee is evaluated on before folding.
        Defaults to ``detector.borexino.parameters.E_NU_GRID_MEV``.
    """

    oscillation_model: SolarSMOscillationModel
    medium: SolarMediumProfile
    source: SolarNeutrinoSource
    source_names: tuple[str, ...]
    bin_edges_MeV: torch.Tensor
    exposure_days: float = borexino_parameters.REFERENCE_EXPOSURE_DAYS
    E_nu_grid_MeV: torch.Tensor = field(
        default_factory=lambda: borexino_parameters.E_NU_GRID_MEV
    )
    background_counts: Optional[torch.Tensor] = None

    @property
    def free(self) -> tuple[str, ...]:
        """Free oscillation-parameter names, forwarded from ``oscillation_model``."""
        return self.oscillation_model.free

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """Predict Borexino counts per bin from ``theta``.

        Args:
            theta: 1-D free-parameter tensor, see
                ``self.oscillation_model.oscillation``.

        Returns:
            Predicted counts per bin, shape ``(n_bins,)``.
        """
        oscillation = self.oscillation_model.oscillation(theta)
        if not self.source_names:
            raise ValueError("source_names must contain at least one solar source")
        counts = torch.zeros(
            self.bin_edges_MeV.numel() - 1, device=theta.device, dtype=theta.dtype,
        )
        zero_background = torch.zeros_like(counts)
        for source_name in self.source_names:
            spectrum = self.source.spectrum_table(source_name)
            if isinstance(spectrum, SolarLineSpectrum):
                probabilities = solar_probability_state(
                    oscillation, spectrum.energy_MeV, self.medium, self.source, source_name,
                )
                contribution = line_event_rate(
                    probabilities, self.source.total_flux(source_name), spectrum.weights,
                    spectrum.energy_MeV, self.bin_edges_MeV,
                    exposure_days=self.exposure_days, background_counts=zero_background,
                )
            else:
                probabilities = solar_probability_state(
                    oscillation, self.E_nu_grid_MeV, self.medium, self.source, source_name,
                )
                flux_tot_MeV = self.source.total_flux(source_name) * spectrum.evaluate(
                    self.E_nu_grid_MeV,
                )
                contribution = event_rate(
                    probabilities, flux_tot_MeV, self.bin_edges_MeV,
                    E_nu_grid_MeV=self.E_nu_grid_MeV, exposure_days=self.exposure_days,
                    background_counts=zero_background,
                )
            counts = counts + contribution
        background = (
            backgrounds_MeV(counts.numel(), device=theta.device, dtype=theta.dtype)
            if self.background_counts is None
            else self.background_counts.to(device=theta.device, dtype=theta.dtype)
        )
        counts = counts + background
        return counts
