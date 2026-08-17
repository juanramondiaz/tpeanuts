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
#      August 2026
# =============================================================================

"""
Differentiable SNO salt-phase (Phase II) 38-observable model.

``SNOPhaseIIObservableModel`` composes ``tpeanuts.detector.sno_ii
.observable``'s three equivalent-flux functions with Earth-matter
regeneration (day/night), mirroring
``tpeanuts.detector.sno.inference_model.SNODayNightModel``'s own
production -> Earth-crossing -> detector-fold pipeline, but predicting
SNO's own 38 published pseudo-observables (day+night x [NC, 17 CC bins,
ES]) instead of Phase I's raw candidate counts.

**Free normalization.** Unlike Phase I (no free flux normalization in
``SNODayNightModel``), this model adds a free ``log_phi_8B``: ``Phi_8B =
solar_source.total_flux("8B") * exp(log_phi_8B)``, so ``log_phi_8B = 0``
reproduces the SSM table value exactly and the fitted flux stays positive
by construction (mirrors SNO's own free ``f_B`` in the primary source's
chi-square minimization, see
``data/detector/sno_ii/metadata/source.json``'s ``phi_b8_free_parameter``
note) -- ``.free`` is therefore ``oscillation_model.free + ("log_phi_8B",)``,
not just a pass-through of the oscillation model's own free tuple.

**hep uses the 8B production profile.** ``solar_probability_mass`` takes a
production-radius profile that depends on ``source_name`` (8B and hep are
produced at very slightly different radii in the Sun). Since hep
contributes only ~0.15% of the total 8B+hep flux (``HEP_FLUX_CM2S`` =
9.3e3 vs. 8B's ~5e6 cm^-2 s^-1) and the two sources' production profiles
are both strongly core-concentrated for such high-energy neutrinos, this
model evaluates the day/night survival probabilities once using 8B's own
production profile and applies that same Pee(E) to the combined
8B+hep flux -- an explicit, quantified-as-negligible approximation, not a
silent one. Combining the two sources' *fluxes* (rather than computing two
separate equivalent-flux predictions and adding them) is required for
correctness: Eq. A1's construction is a ratio, and (A1+A2)/(B1+B2) is not
generally equal to A1/B1 + A2/B2 -- see
``tpeanuts.detector.sno_ii.observable``'s module docstring.

Module contents:
    SNOPhaseIIObservableModel
        The composition model described above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from tpeanuts.detector.sno_ii.io import load_zenith_exposure
from tpeanuts.detector.sno_ii.observable import (
    cc_equivalent_flux_spectrum,
    es_equivalent_flux,
    nc_equivalent_flux,
)
from tpeanuts.detector.sno_ii.parameters import (
    CC_BIN_EDGES_MEV,
    DETECTOR_DEPTH_M,
    E_NU_GRID_MEV,
    HEP_FLUX_CM2S,
)
from tpeanuts.inference.solar_model import SolarSMOscillationModel
from tpeanuts.medium.earth.probability import earth_probability_state
from tpeanuts.medium.solar.probability import solar_probability_mass
from tpeanuts.medium.solar.profile import SolarMediumProfile
from tpeanuts.source.solar import SolarNeutrinoSource

__all__ = ["SNOPhaseIIObservableModel"]

# data/detector/sno_ii/ tabulates every flux in units of 1e6 cm^-2 s^-1
# (see metadata/source.json's "units" entry), while
# SolarNeutrinoSource.total_flux returns the real flux in raw cm^-2 s^-1
# (e.g. "8B" ~= 4.135e6). predict() divides by this constant so its output
# is directly comparable to the tabulated data without an external rescale.
FLUX_UNIT_CM2S: float = 1.0e6


@dataclass(frozen=True)
class SNOPhaseIIObservableModel:
    """Predicted (day, night) SNO salt-phase 38-observable vector, as a function of theta.

    Parameters
    ----------
    oscillation_model:
        A ``SolarSMOscillationModel`` supplying ``free``/``oscillation(theta)``.
    solar_medium:
        Solar density profile (``medium.solar.profile.SolarMediumProfile``).
    solar_source:
        Solar production source, must have both "8B" and "hep" tabulated
        (``source.solar.SolarNeutrinoSource``).
    earth_profile:
        Earth density profile (``tpeanuts.medium.earth.profile
        .build_earth_profile``'s return value).
    eta_day, weight_day, eta_night, weight_night:
        Nadir-angle grids and *normalized* (each summing to 1) averaging
        weights for the day and night periods, e.g. from
        ``SNOPhaseIIObservableModel.from_real_exposure``.
    detector_depth_m:
        Detector depth below the Earth's surface, in metres.
    E_nu_grid_MeV:
        True neutrino energy grid, shared by both the 8B and hep
        contributions and by ``tpeanuts.detector.sno_ii.observable``'s
        folds. Defaults to ``tpeanuts.detector.sno_ii.parameters
        .E_NU_GRID_MEV``.
    hep_flux_cm2s:
        Fixed hep flux (not a free parameter), see module docstring.
    """

    oscillation_model: SolarSMOscillationModel
    solar_medium: SolarMediumProfile
    solar_source: SolarNeutrinoSource
    earth_profile: object
    eta_day: torch.Tensor
    weight_day: torch.Tensor
    eta_night: torch.Tensor
    weight_night: torch.Tensor
    detector_depth_m: float = DETECTOR_DEPTH_M
    E_nu_grid_MeV: torch.Tensor = field(default_factory=lambda: E_NU_GRID_MEV)
    hep_flux_cm2s: float = HEP_FLUX_CM2S

    @property
    def free(self) -> tuple[str, ...]:
        """Free-parameter names: the oscillation model's own, plus ``log_phi_8B``."""
        return self.oscillation_model.free + ("log_phi_8B",)

    def day_night_probabilities(self, theta_osc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exposure-averaged flavour probabilities for the day and night periods.

        Uses 8B's own production profile for both the 8B and hep flux
        contributions (see module docstring).

        Args:
            theta_osc: The oscillation model's own free-parameter
                sub-vector (``self.free[:-1]``, i.e. ``theta`` without
                ``log_phi_8B``).

        Returns:
            ``(p_day, p_night)``, each shaped ``(n_E, n_flavour)`` and
            differentiable with respect to ``theta_osc``.
        """
        oscillation = self.oscillation_model.oscillation(theta_osc)
        mass_weights = solar_probability_mass(
            oscillation, self.E_nu_grid_MeV, self.solar_medium, self.solar_source, "8B",
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
        """Predict the concatenated (day, night) 38-observable vector from ``theta``.

        Args:
            theta: 1-D free-parameter tensor, ``self.free``'s order (the
                oscillation model's own parameters, then ``log_phi_8B``).

        Returns:
            Predicted equivalent-flux vector, shape ``(38,)``, in units of
            ``FLUX_UNIT_CM2S`` = 1e6 cm^-2 s^-1 (matching
            ``data/detector/sno_ii/``'s own convention): day's
            ``[NC, CC1..CC17, ES]`` (``tpeanuts.detector.sno_ii.parameters
            .CHANNEL_ORDER`` order) followed by night's.
        """
        theta_osc = theta[:-1]
        log_phi_8b = theta[-1]

        phi_8b = (self.solar_source.total_flux("8B") / FLUX_UNIT_CM2S) * torch.exp(log_phi_8b)
        phi_hep = torch.as_tensor(self.hep_flux_cm2s / FLUX_UNIT_CM2S, dtype=theta.dtype, device=theta.device)
        flux_tot_MeV = (
            phi_8b * self.solar_source.spectrum("8B", self.E_nu_grid_MeV)
            + phi_hep * self.solar_source.spectrum("hep", self.E_nu_grid_MeV)
        )

        p_day, p_night = self.day_night_probabilities(theta_osc)

        def _period_vector(probabilities: torch.Tensor) -> torch.Tensor:
            cc = cc_equivalent_flux_spectrum(
                probabilities, flux_tot_MeV, CC_BIN_EDGES_MEV, E_nu_grid_MeV=self.E_nu_grid_MeV,
            )
            es = es_equivalent_flux(probabilities, flux_tot_MeV, E_nu_grid_MeV=self.E_nu_grid_MeV)
            nc = nc_equivalent_flux(probabilities, flux_tot_MeV, E_nu_grid_MeV=self.E_nu_grid_MeV)
            return torch.cat([nc.reshape(1), cc, es.reshape(1)])  # NC, CC1..CC17, ES

        return torch.cat([_period_vector(p_day), _period_vector(p_night)])

    @classmethod
    def from_real_exposure(
        cls,
        oscillation_model: SolarSMOscillationModel,
        solar_medium: SolarMediumProfile,
        solar_source: SolarNeutrinoSource,
        earth_profile: object,
        *,
        detector_depth_m: float = DETECTOR_DEPTH_M,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> "SNOPhaseIIObservableModel":
        """Build a model from the real SNO salt-phase cos(zenith) exposure table (Table XXXI).

        Unlike Phase I's ``SNODayNightModel.from_real_exposure`` (480
        points, downsampled by a ``stride``), Table XXXI has only 60
        points already, used in full here.

        Args:
            oscillation_model: A ``SolarSMOscillationModel``.
            solar_medium: Solar density profile.
            solar_source: Solar production source (must have "8B" and
                "hep" tabulated).
            earth_profile: Earth density profile.
            detector_depth_m: Detector depth below the Earth's surface, m.
            device: Target torch device.
            dtype: Target real dtype.

        Returns:
            A configured ``SNOPhaseIIObservableModel``.
        """
        eta, exposure = load_zenith_exposure(device=device, dtype=dtype)

        night_mask = eta < (torch.pi / 2)
        day_mask = ~night_mask

        eta_night = eta[night_mask]
        weight_night = exposure[night_mask]
        weight_night = weight_night / weight_night.sum().clamp_min(torch.finfo(dtype).tiny)

        eta_day = eta[day_mask]
        weight_day = exposure[day_mask]
        weight_day = weight_day / weight_day.sum().clamp_min(torch.finfo(dtype).tiny)

        return cls(
            oscillation_model=oscillation_model,
            solar_medium=solar_medium,
            solar_source=solar_source,
            earth_profile=earth_profile,
            eta_day=eta_day, weight_day=weight_day,
            eta_night=eta_night, weight_night=weight_night,
            detector_depth_m=detector_depth_m,
        )
