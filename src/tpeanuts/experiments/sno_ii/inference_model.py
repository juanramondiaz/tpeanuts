#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Differentiable SNO salt-phase (Phase II) 38-observable model.

Ported from v1_0's ``experiments/sno_ii/inference_model.py``.
``SNOPhaseIIModel`` composes
``tpeanuts.experiments.sno_ii.observable``'s three equivalent-flux
functions with Earth-matter regeneration (day/night), mirroring
``tpeanuts.experiments.sno.inference_model.SNOModel``'s own
production -> Earth-crossing -> detector-fold pipeline, but predicting
SNO's own 38 published pseudo-observables (day+night x [NC, 17 CC
bins, ES]) instead of Phase I's raw candidate counts. Reuses
``tpeanuts.pipeline.solar.solar_flux`` +
``tpeanuts.pipeline._legs.EarthLeg`` (a
``tpeanuts.medium.earth.exposure.table.NadirExposureTable``) for the
day/night average -- the same v2 pipeline SNO Phase I's own
``SNOModel`` already uses, in place of v1_0's own manual
per-eta_rad loop.

Following ``tpeanuts.experiments.borexino.inference_model
.BorexinoModel``/``tpeanuts.experiments.sno.inference_model
.SNOModel``'s own precedent, there is no separate
oscillation-model class here either: ``_build_model`` constructs its
own ``PropagationModel`` inline.

**Free normalization.** Unlike Phase I (no free flux normalization in
``SNOModel``), this model adds a free ``log_phi_8B``:
``Phi_8B = solar_source.total_flux("8B") * exp(log_phi_8B)``, so
``log_phi_8B = 0`` reproduces the SSM table value exactly and the
fitted flux stays positive by construction (mirrors SNO's own free
``f_B`` in the primary source's chi-square minimization) --
``.free`` is therefore ``oscillation_free + ("log_phi_8B",)``, always
the last entry, not just the oscillation parameters alone.

**hep uses the 8B production profile.** ``solar_flux``
takes a source name whose own production-radius profile determines the
Earth-crossing probability (8B and hep are produced at very slightly
different radii in the Sun). Since hep contributes only ~0.15% of the
total 8B+hep flux (``HEP_FLUX_CM2S`` = 9.3e3 vs. 8B's ~4.1e6 cm^-2 s^-1)
and the two sources' production profiles are both strongly
core-concentrated for such high-energy neutrinos, this model evaluates
the day/night survival probabilities once using 8B's own production
profile and applies that same Pee(E) to the combined 8B+hep flux -- an
explicit, quantified-as-negligible approximation, not a silent one.
Combining the two sources' *fluxes* (rather than computing two separate
equivalent-flux predictions and adding them) is required for
correctness: Eq. A1's construction is a ratio, and (A1+A2)/(B1+B2) is
not generally equal to A1/B1 + A2/B2 -- see
``tpeanuts.experiments.sno_ii.observable``'s own module docstring.

Module contents:
    SNOPhaseIIModel
        The composition model described above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch

from tpeanuts.experiments.sno_ii.io import load_zenith_exposure
from tpeanuts.experiments.sno_ii.observable import (
    cc_equivalent_flux_spectrum,
    es_equivalent_flux,
    nc_equivalent_flux,
)
from tpeanuts.experiments.sno_ii.parameters import DETECTOR_DEPTH_M, HEP_FLUX_CM2S, SNOPhaseIIParameters
from tpeanuts.medium.earth.exposure.table import NadirExposureTable
from tpeanuts.medium.solar.solar import SolarMedium
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix, StandardMixingParameters
from tpeanuts.physics.common.nature import Dirac
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.model import PropagationModel
from tpeanuts.pipeline._legs import EarthLeg, SolarLeg
from tpeanuts.pipeline.solar import solar_flux
from tpeanuts.source.solar.solar import SolarSource
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike

Tensor = torch.Tensor

_OSC_PARAM_KEYS: Tuple[str, ...] = ("theta12", "theta13", "DeltamSq21", "DeltamSq3l")

# data/experiments/sno_ii/ tabulates every flux in units of 1e6 cm^-2 s^-1
# (see metadata/source.json's "units" entry), while
# SolarSource.total_flux returns the real flux in raw cm^-2 s^-1
# (e.g. "8B" ~= 4.135e6). predict() divides by this constant so its output
# is directly comparable to the tabulated data without an external rescale.
FLUX_UNIT_CM2S: float = 1.0e6


@dataclass(frozen=True)
class SNOPhaseIIModel:
    """Predicted (day, night) SNO salt-phase 38-observable vector, as a function of theta.

    Attributes:
        oscillation_free: Names of the oscillation parameters to fit,
            a subset of ``_OSC_PARAM_KEYS`` (``"theta12"``,
            ``"theta13"``, ``"DeltamSq21"``, ``"DeltamSq3l"``), in the
            order ``predict``'s ``theta`` argument unpacks them (before
            ``log_phi_8B``, always the last entry -- see ``.free``).
        oscillation_fixed: Values (radians for angles, eV^2 for the
            splittings) for every key of ``_OSC_PARAM_KEYS`` not in
            ``oscillation_free``.
        theta23, delta13: Fixed atmospheric angle and CP phase, in
            radians -- do not affect P_ee, only needed to build a
            well-formed PMNS matrix.
        solar_medium: Solar density profile (independent of ``theta``).
        solar_source: Solar production source, must have both "8B" and
            "hep" tabulated.
        day_exposure_table, night_exposure_table: Nadir-angle exposure
            tables for the day (``eta_rad >= pi/2``) and night
            (``eta_rad < pi/2``) periods -- see ``from_real_exposure``.
        context: RuntimeContext every tensor this model builds is on.
        detector_depth_m: Detector depth below the Earth's surface,
            metres.
        earth_method: Which ``PropagationMethod`` builds each nadir
            angle's own Earth evolutor. Defaults to
            ``PropagationMethod.ANALYTIC`` (exact closed-form; see
            ``tpeanuts.experiments.sno.inference_model.SNOModel``'s
            own docstring for why this, not v1_0's "analytical" =
            ``ANALYTIC_PERTURBATIVE``, is the right default).
        earth_nsteps: Chord discretization forwarded to
            ``EarthLeg`` (ignored by
            ``ANALYTIC``).
        hep_flux_cm2s: Fixed hep flux (not a free parameter), see
            module docstring.
        extensions: Optional propagation extensions (for example
            ``tpeanuts.physics.extensions.nsi.NSITerm``), forwarded to the
            ``PropagationModel`` this model builds, so they act in the Sun
            (through ``SolarLeg(method="adiabatic_exact")``, the only solar
            method that supports an extension) and in the Earth. Empty (the
            default) is the Standard Model and changes nothing.

    Raises:
        ValueError: If ``oscillation_free``/``oscillation_fixed`` do
            not together cover exactly ``_OSC_PARAM_KEYS`` with no
            overlap and no gap.
    """

    oscillation_free: Tuple[str, ...]
    oscillation_fixed: Mapping[str, TensorLike]
    theta23: TensorLike
    delta13: TensorLike
    solar_medium: SolarMedium
    solar_source: SolarSource
    day_exposure_table: NadirExposureTable
    night_exposure_table: NadirExposureTable
    context: RuntimeContext
    detector_depth_m: float = DETECTOR_DEPTH_M
    earth_method: PropagationMethod = PropagationMethod.ANALYTIC
    earth_nsteps: int = 100
    hep_flux_cm2s: float = HEP_FLUX_CM2S
    extensions: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        free_set = set(self.oscillation_free)
        fixed_set = set(self.oscillation_fixed)
        if len(free_set) != len(self.oscillation_free):
            raise ValueError(f"oscillation_free contains a duplicate parameter name: {self.oscillation_free!r}.")
        if free_set & fixed_set:
            raise ValueError(
                f"oscillation_free and oscillation_fixed both name {sorted(free_set & fixed_set)} -- "
                "each of _OSC_PARAM_KEYS must be in exactly one of them."
            )
        covered = free_set | fixed_set
        if covered != set(_OSC_PARAM_KEYS):
            missing = set(_OSC_PARAM_KEYS) - covered
            unknown = covered - set(_OSC_PARAM_KEYS)
            raise ValueError(
                f"oscillation_free/oscillation_fixed together must cover exactly {_OSC_PARAM_KEYS}; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )

    @property
    def free(self) -> Tuple[str, ...]:
        """Free parameter names: the oscillation parameters, then ``log_phi_8B``."""
        return self.oscillation_free + ("log_phi_8B",)

    def _build_model(self, theta_osc: Tensor) -> PropagationModel:
        values = dict(self.oscillation_fixed)
        values.update(zip(self.oscillation_free, theta_osc.unbind(), strict=True))

        space = FlavourSpace()
        mixing = MixingMatrix(
            space=space,
            standard=StandardMixingParameters(
                theta12=values["theta12"], theta13=values["theta13"],
                theta23=self.theta23, delta13=self.delta13,
                context=self.context,
            ),
            nature=Dirac(),
            context=self.context,
        )
        mass_spectrum = MassSpectrum(
            space=space, DeltamSq21=values["DeltamSq21"], DeltamSq3l=values["DeltamSq3l"], context=self.context,
        )
        return PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, extensions=self.extensions)

    def _solar_options(self) -> Dict[str, Any]:
        """Keyword arguments of ``solar_flux``: the exact adiabatic leg if an extension is active, none otherwise."""
        return {"solar_leg": SolarLeg(method="adiabatic_exact")} if self.extensions else {}

    def day_night_probabilities(self, theta_osc: Tensor) -> Tuple[Tensor, Tensor]:
        """Exposure-averaged flavour probabilities for the day and night periods.

        Uses 8B's own production profile for both the 8B and hep flux
        contributions (see module docstring).

        Args:
            theta_osc: The oscillation sub-vector (``self.oscillation_free``'s
                own order, i.e. ``theta`` without ``log_phi_8B``).

        Returns:
            ``(p_day, p_night)``, each shaped ``(n_E, n_flavour)`` and
            differentiable with respect to ``theta_osc``.
        """
        model = self._build_model(theta_osc)
        e_nu_grid_MeV = SNOPhaseIIParameters.default(context=self.context).e_nu_grid_MeV
        at_surface = solar_flux(
            model, self.solar_source, self.solar_medium, "8B", e_nu_grid_MeV, intensity=False, **self._solar_options(),
        )["8B"]   # the Sun once, then the Earth for each period
        p_day, p_night = (
            EarthLeg(
                depth_m=self.detector_depth_m, method=self.earth_method, nsteps=self.earth_nsteps, exposure=table,
            ).propagate(at_surface, model).state.probabilities()
            for table in (self.day_exposure_table, self.night_exposure_table)
        )
        return p_day, p_night

    def predict(self, theta: Tensor) -> Tensor:
        """Predict the concatenated (day, night) 38-observable vector from ``theta``.

        Args:
            theta: 1-D free-parameter tensor, ``self.free``'s order
                (the oscillation parameters, then ``log_phi_8B``).

        Returns:
            Predicted equivalent-flux vector, shape ``(38,)``, in units
            of ``FLUX_UNIT_CM2S`` = 1e6 cm^-2 s^-1 (matching
            ``data/experiments/sno_ii/``'s own convention): day's
            ``[NC, CC1..CC17, ES]``
            (``tpeanuts.experiments.sno_ii.parameters.CHANNEL_ORDER``
            order) followed by night's.
        """
        theta_osc = theta[:-1]
        log_phi_8b = theta[-1]

        grids = SNOPhaseIIParameters.default(context=RuntimeContext(device=theta.device, dtype=theta.dtype))
        e_nu_grid_MeV = grids.e_nu_grid_MeV
        T_grid_MeV = grids.T_grid_MeV
        Tprime_grid_MeV = grids.Tprime_grid_MeV
        cc_bin_edges_MeV = grids.cc_bin_edges_MeV

        phi_8b = (self.solar_source.total_flux("8B") / FLUX_UNIT_CM2S) * torch.exp(log_phi_8b)
        phi_hep = torch.as_tensor(self.hep_flux_cm2s / FLUX_UNIT_CM2S, dtype=theta.dtype, device=theta.device)
        flux_tot_MeV = (
            phi_8b * self.solar_source.spectrum("8B", e_nu_grid_MeV)
            + phi_hep * self.solar_source.spectrum("hep", e_nu_grid_MeV)
        )

        p_day, p_night = self.day_night_probabilities(theta_osc)

        def _period_vector(probabilities: Tensor) -> Tensor:
            cc = cc_equivalent_flux_spectrum(
                probabilities, flux_tot_MeV, cc_bin_edges_MeV, e_nu_grid_MeV,
                T_grid_MeV, Tprime_grid_MeV, context=self.context,
            )
            es = es_equivalent_flux(probabilities, flux_tot_MeV, e_nu_grid_MeV, T_grid_MeV, Tprime_grid_MeV)
            nc = nc_equivalent_flux(probabilities, flux_tot_MeV, e_nu_grid_MeV)
            return torch.cat([nc.reshape(1), cc, es.reshape(1)])  # NC, CC1..CC17, ES

        return torch.cat([_period_vector(p_day), _period_vector(p_night)])

    @classmethod
    def from_real_exposure(
        cls,
        oscillation_free: Tuple[str, ...],
        oscillation_fixed: Mapping[str, TensorLike],
        theta23: TensorLike,
        delta13: TensorLike,
        solar_medium: SolarMedium,
        solar_source: SolarSource,
        *,
        context: RuntimeContext,
        detector_depth_m: float = DETECTOR_DEPTH_M,
        earth_method: PropagationMethod = PropagationMethod.ANALYTIC,
        earth_nsteps: int = 100,
        extensions: Tuple[Any, ...] = (),
    ) -> "SNOPhaseIIModel":
        """Build a model from the real SNO salt-phase cos(zenith) exposure table (Table XXXI).

        Unlike Phase I's ``SNOModel.from_real_exposure`` (480
        points, downsampled by a ``stride``), Table XXXI has only 60
        points already, used in full here.

        Args:
            oscillation_free, oscillation_fixed, theta23, delta13: See
                the class docstring.
            solar_medium: Solar density profile.
            solar_source: Solar production source (must have "8B" and
                "hep" tabulated).
            context: RuntimeContext every tensor this model builds is
                on.
            detector_depth_m: Detector depth below the Earth's
                surface, metres.
            earth_method, earth_nsteps, extensions: See the class docstring.

        Returns:
            A configured ``SNOPhaseIIModel``.
        """
        eta_rad, exposure = load_zenith_exposure(context=context)

        night_mask = eta_rad < (torch.pi / 2)
        day_mask = ~night_mask

        day_table = NadirExposureTable(eta_rad=eta_rad[day_mask], exposure=exposure[day_mask])
        night_table = NadirExposureTable(eta_rad=eta_rad[night_mask], exposure=exposure[night_mask])

        return cls(
            oscillation_free=oscillation_free, oscillation_fixed=oscillation_fixed,
            theta23=theta23, delta13=delta13,
            solar_medium=solar_medium, solar_source=solar_source,
            day_exposure_table=day_table, night_exposure_table=night_table,
            context=context, detector_depth_m=detector_depth_m,
            earth_method=earth_method, earth_nsteps=earth_nsteps, extensions=extensions,
        )
