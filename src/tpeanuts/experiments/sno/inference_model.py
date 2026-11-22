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

"""Differentiable SNO day/night event-count model: oscillation x Sun x Earth x detector.

Ported from v1_0's ``experiments/sno/inference_model.py``
(``SNOModel``). v1_0 computes the day/night exposure-averaged
flavour probability *by hand*
(``day_night_probabilities``: a Python loop over nadir angle, calling
``medium.solar.probability.solar_probability_mass`` once and
``medium.earth.probability.earth_probability_state`` once per angle,
then a manual weighted sum) and needed a dedicated
``@torch.no_grad()`` audit across 3 files (10 decorators) to make the
Earth leg differentiable at all.

None of that is needed here:
``tpeanuts.pipeline.solar.solar_flux`` (production through the Sun ->
``Flux``) followed by ``tpeanuts.pipeline._legs.EarthLeg`` with an
exposure table (a nadir-angle exposure-weighted Earth chord ->
flavour-basis probabilities) composes exactly that chain, and carries
no ``@torch.no_grad()`` anywhere in its own call chain
(``tpeanuts.medium.earth``/``tpeanuts.pipeline``, confirmed directly)
-- v2 was designed autodifferentiable from the start. This is the first concrete detector
in this rewrite to consume that pipeline. There is also no opaque
pre-built ``earth_profile`` object to carry around (v1_0's own
``SNOModel.earth_profile`` field): ``EarthMedium`` is built
internally, per nadir angle, from ``eta_rad``/``depth_m``/``context``
alone.

Following ``tpeanuts.experiments.borexino.inference_model
.BorexinoModel``'s own precedent (builds its own
``PropagationModel`` inline rather than depending on another model's
private method), ``SNOModel`` does the same here.

**Why the default ``earth_method`` is ``PropagationMethod.ANALYTIC``,
not a v1_0-matching choice.** v1_0's own Earth leg hardcodes
``method="analytical"``, which is *not* the same algorithm as v2's
``PropagationMethod.ANALYTIC`` (an exact closed-form Cardano/Ferrari
diagonalization) -- v1_0's own docstring names it "the analytical
*perturbative* evolutor", i.e. v2's ``ANALYTIC_PERTURBATIVE``
(zero-order exact plus a first-order correction). Confirmed directly
that ``ANALYTIC_PERTURBATIVE`` cannot be used here at all with this
model's batched (400-point) energy grid: it raises a shape-mismatch
error inside ``tpeanuts.physics.evolution.perturbative.profile
.EvenPowerProfile.residual_integral`` (energy axis colliding with the
shell-segment axis), the same class of shared-code batching
limitation already documented for ``kinetic_potential`` (see
[[reference_batching_axis_collision]]) -- out of scope to fix here.
``PropagationMethod.NUMERICAL`` (v2's own default elsewhere) *is*
usable, and was confirmed directly to converge to the same physics as
``ANALYTIC`` (agreement improves from ~0.3% at the shared default
``nsteps=100`` to <0.1% by ``nsteps=8000``, at a representative
eta_rad/energy point) -- the discretization error at the shared default
resolution is a genuine, expected artifact of coarse equal-angle
segmentation across the Earth's steep core/mantle density gradient,
not a bug in either codebase (the solar-production leg alone was
independently confirmed exact against v1_0 to 1e-16). ``ANALYTIC`` is
therefore both the more accurate and the faster choice, and is
deliberately not tuned to reproduce v1_0's own approximate default.

**Predicts CC + ES + NC, summed, matching SNO's own raw candidate
counts.** SNO Phase I did not tag events by reaction:
``day_counts``/``night_counts`` in
``data/experiments/sno/observation/day_night_spectrum.csv`` are raw
candidates surviving the electron-energy/fiducial-volume cuts (CC and
ES recoil electrons, plus NC neutron-capture-gamma Compton electrons
above threshold), and the real SNO analysis separated the three rates
*statistically*, fitting a combined probability-density function in
reconstructed energy, the electron's angle to the Sun, and
volume-weighted radius -- an event-by-event PDF decomposition this
project does not reproduce. Summing
``tpeanuts.experiments.sno.event_rate``'s three forward-modeled channels
(``cc_event_rate`` + ``es_event_rate`` + ``nc_event_rate``) is the
closest match to that combined candidate count without redoing SNO's
own statistical extraction.

Module contents:
    SNOModel
        Wraps oscillation parameters, a solar profile/source, and
        day/night nadir-angle exposure tables, predicting concatenated
        (day, night) CC+ES+NC counts per bin from a free-parameter
        vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from tpeanuts.config import default
from tpeanuts.experiments.sno.event_rate import cc_detector, es_detector, nc_detector
from tpeanuts.experiments.sno.io import load_nadir_exposure_raw, total_livetime_days
from tpeanuts.experiments.sno.parameters import SNOParameters
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
from tpeanuts.source.solar.solar import SolarLineSpectrum, SolarSource
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike

Tensor = torch.Tensor

_OSC_PARAM_KEYS: Tuple[str, ...] = ("theta12", "theta13", "DeltamSq21", "DeltamSq3l")

# SNO's commonly quoted approximate depth (Sudbury mine), metres; not
# verified here against a primary SNO paper -- the physical effect of this
# value on Earth-crossing geometry is tiny relative to the Earth's ~6371 km
# radius (same caveat v1_0's own default carries).
_DETECTOR_DEPTH_M = 2039.0


@dataclass(frozen=True)
class SNOModel:
    """Predicted (day, night) SNO CC+ES+NC counts per bin, as a function of oscillation parameters.

    Attributes:
        params: Loaded SNO parameters.
        free: Names of the parameters to fit, a subset of
            ``_OSC_PARAM_KEYS`` (``"theta12"``, ``"theta13"``,
            ``"DeltamSq21"``, ``"DeltamSq3l"``), in the order
            ``predict``'s ``theta`` argument is unpacked.
        fixed: Values (radians for angles, eV^2 for the splittings) for
            every key of ``_OSC_PARAM_KEYS`` not in ``free``.
        theta23, delta13: Fixed atmospheric angle and CP phase, in
            radians -- do not affect P_ee, only needed to build a
            well-formed PMNS matrix.
        solar_medium: Solar density profile (independent of ``theta``).
        solar_source: Solar production source (independent of
            ``theta``).
        source_names: Continuous solar source keys whose CC, ES, and
            NC contributions are summed (normally ``("8B", "hep")``).
            Line-spectrum sources (7Be/pep) are rejected -- see
            ``__post_init__``.
        bin_edges_MeV: Observed electron-energy bin edges, shared by
            day and night.
        day_exposure_table, night_exposure_table: Nadir-angle exposure
            tables for the day (``eta_rad < pi/2``) and night
            (``eta_rad >= pi/2``) periods -- see ``from_real_exposure``.
        exposure_days_day, exposure_days_night: Live time in days for
            each period.
        context: RuntimeContext every tensor this model builds is on.
        detector_depth_m: Detector depth below the Earth's surface, in
            metres.
        earth_method: Which ``PropagationMethod`` builds each nadir
            angle's own Earth evolutor. Defaults to
            ``PropagationMethod.ANALYTIC`` (v1_0's own
            ``method="analytical"`` choice, for speed -- exact
            closed-form, not an approximation).
        earth_nsteps: Chord discretization forwarded to
            ``EarthLeg`` (ignored by
            ``ANALYTIC``).
        background_day, background_night: Optional real *instrumental*
            background counts per bin (the "neutron"/"low_energy"
            columns of ``tpeanuts.experiments.sno.io.load_backgrounds`` --
            contamination, not the NC neutrino signal, which is its own
            channel, see module docstring), added once to the summed
            CC+ES+NC signal.
        extensions: Optional propagation extensions (for example
            ``tpeanuts.physics.extensions.nsi.NSITerm``), forwarded to the
            ``PropagationModel`` this model builds, so they act in the Sun
            (through ``SolarLeg(method="adiabatic_exact")``, the only solar
            method that supports an extension) and in the Earth. Empty (the
            default) is the Standard Model and changes nothing.

    Raises:
        ValueError: If ``free``/``fixed`` do not together cover exactly
            ``_OSC_PARAM_KEYS`` with no overlap and no gap, if
            ``source_names`` is empty, or if any ``source_names`` entry
            is a line-spectrum solar source.
    """

    params: SNOParameters
    free: Tuple[str, ...]
    fixed: Mapping[str, TensorLike]
    theta23: TensorLike
    delta13: TensorLike
    solar_medium: SolarMedium
    solar_source: SolarSource
    source_names: Tuple[str, ...]
    bin_edges_MeV: Tensor
    day_exposure_table: NadirExposureTable
    night_exposure_table: NadirExposureTable
    exposure_days_day: float
    exposure_days_night: float
    context: RuntimeContext
    detector_depth_m: float = _DETECTOR_DEPTH_M
    earth_method: PropagationMethod = PropagationMethod.ANALYTIC
    earth_nsteps: int = default.EARTH_NSTEPS
    background_day: Optional[Tensor] = None
    background_night: Optional[Tensor] = None
    extensions: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        free_set = set(self.free)
        fixed_set = set(self.fixed)
        if len(free_set) != len(self.free):
            raise ValueError(f"free contains a duplicate parameter name: {self.free!r}.")
        if free_set & fixed_set:
            raise ValueError(
                f"free and fixed both name {sorted(free_set & fixed_set)} -- "
                "each parameter must be in exactly one of them."
            )
        covered = free_set | fixed_set
        if covered != set(_OSC_PARAM_KEYS):
            missing = set(_OSC_PARAM_KEYS) - covered
            unknown = covered - set(_OSC_PARAM_KEYS)
            raise ValueError(
                f"free/fixed together must cover exactly {_OSC_PARAM_KEYS}; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        if not self.source_names:
            raise ValueError("source_names must contain at least one solar source")

        line_sources = [
            name
            for name in self.source_names
            if self.solar_source.has_spectrum(name)
            and isinstance(self.solar_source.spectrum_table(name), SolarLineSpectrum)
        ]
        if line_sources:
            raise ValueError(
                f"SNOModel does not support line-spectrum solar sources "
                f"({line_sources!r} are tabulated as discrete lines). SNO's real "
                "analysis threshold lies above every solar line (7Be/pep), so "
                "only continuous sources (e.g. '8B', 'hep') are physically "
                "meaningful here; see "
                "tpeanuts.experiments.borexino.inference_model.BorexinoModel "
                "for line-spectrum support."
            )

    def _build_model(self, theta: Tensor) -> PropagationModel:
        params = dict(self.fixed)
        params.update(zip(self.free, theta.unbind(), strict=True))

        space = FlavourSpace()
        mixing = MixingMatrix(
            space=space,
            standard=StandardMixingParameters(
                theta12=params["theta12"], theta13=params["theta13"],
                theta23=self.theta23, delta13=self.delta13,
                context=self.context,
            ),
            nature=Dirac(),
            context=self.context,
        )
        mass_spectrum = MassSpectrum(
            space=space, DeltamSq21=params["DeltamSq21"], DeltamSq3l=params["DeltamSq3l"], context=self.context,
        )
        return PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, extensions=self.extensions)

    def _solar_options(self) -> Dict[str, Any]:
        """Keyword arguments of ``solar_flux``: the exact adiabatic leg if an extension is active, none otherwise."""
        return {"solar_leg": SolarLeg(method="adiabatic_exact")} if self.extensions else {}

    def predict(self, theta: Tensor) -> Tensor:
        """Predict concatenated (day, night) SNO CC+ES+NC counts per bin from ``theta``.

        Sums all three of ``tpeanuts.experiments.sno.event_rate``'s
        composable channels (see module docstring for why: SNO's own
        raw candidate counts are not separated by reaction either)
        before adding the instrumental background once.

        Args:
            theta: 1-D free-parameter tensor, see ``self.free``.

        Returns:
            Predicted counts, shape ``(2 * n_bins,)``: day bins
            followed by night bins, matching how real data should be
            concatenated for comparison
            (``torch.cat([day_counts, night_counts])``).
        """
        model = self._build_model(theta)
        n_bins = self.bin_edges_MeV.shape[0] - 1
        counts_day = torch.zeros(n_bins, device=theta.device, dtype=theta.dtype)
        counts_night = torch.zeros_like(counts_day)
        detectors = {
            period: tuple(
                build(self.params, self.bin_edges_MeV, exposure_days=days)
                for build in (cc_detector, es_detector, nc_detector)
            )
            for period, days in (("day", self.exposure_days_day), ("night", self.exposure_days_night))
        }
        tables = {"day": self.day_exposure_table, "night": self.night_exposure_table}

        # the fold is linear in the flavour probabilities, so averaging them over the nadir-angle exposure is the
        # same as averaging the counts; the Earth leg carries the average as the flux's state
        earth = {
            period: EarthLeg(
                depth_m=self.detector_depth_m, method=self.earth_method, nsteps=self.earth_nsteps, exposure=table,
            )
            for period, table in tables.items()
        }

        for source_name in self.source_names:
            at_surface = solar_flux(
                model, self.solar_source, self.solar_medium, source_name, self.params.e_nu_grid_MeV,
                **self._solar_options(),
            )[source_name]
            for period, leg in earth.items():
                arrived = leg.propagate(at_surface, model)
                total = sum(folded.observe(arrived).counts for folded in detectors[period])
                if period == "day":
                    counts_day = counts_day + total
                else:
                    counts_night = counts_night + total

        if self.background_day is not None:
            counts_day = counts_day + self.background_day.to(device=theta.device, dtype=theta.dtype)
        if self.background_night is not None:
            counts_night = counts_night + self.background_night.to(device=theta.device, dtype=theta.dtype)

        return torch.cat([counts_day, counts_night])

    @classmethod
    def from_real_exposure(
        cls,
        params: SNOParameters,
        free: Tuple[str, ...],
        fixed: Mapping[str, TensorLike],
        theta23: TensorLike,
        delta13: TensorLike,
        solar_medium: SolarMedium,
        solar_source: SolarSource,
        source_names: Tuple[str, ...],
        bin_edges_MeV: Tensor,
        *,
        stride: int = 12,
        detector_depth_m: float = _DETECTOR_DEPTH_M,
        earth_method: PropagationMethod = PropagationMethod.ANALYTIC,
        earth_nsteps: int = default.EARTH_NSTEPS,
        background_day: Optional[Tensor] = None,
        background_night: Optional[Tensor] = None,
        extensions: Tuple[Any, ...] = (),
    ) -> "SNOModel":
        """Build a model from the real SNO cos(zenith) exposure table.

        Downsamples the real 480-point nadir-angle exposure
        (``tpeanuts.experiments.sno.io.load_nadir_exposure_raw``) by a
        factor ``stride`` for computational tractability (each retained
        eta_rad point costs one Earth-crossing evolutor evaluation per
        energy), and splits it into a day (``eta_rad < pi/2``) and a night
        (``eta_rad >= pi/2``) ``NadirExposureTable``. Neither table is
        explicitly renormalized here: ``EarthLeg`` ->
        ``tpeanuts.pipeline._legs.exposure_average`` ->
        ``tpeanuts.medium.earth.exposure.table.integrate_exposure``
        already renormalizes by the table's own trapezoidal integral
        every time it is used, regardless of whether it was
        pre-normalized (see that function's own docstring), so a
        redundant division here would only add float noise). Real
        (day, night) live times come from
        ``tpeanuts.experiments.sno.io.total_livetime_days`` (exact,
        independent of the downsampling above).

        Args:
            params: Loaded SNO parameters.
            free, fixed, theta23, delta13: See the class docstring.
            solar_medium: Solar density profile.
            solar_source: Solar production source.
            source_names: Continuous solar source keys.
            bin_edges_MeV: Observed electron-energy bin edges.
            stride: Keep every ``stride``-th point of the real
                480-point exposure grid.
            detector_depth_m: Detector depth below the Earth's
                surface, in metres.
            earth_method: Which ``PropagationMethod`` builds each nadir
                angle's own Earth evolutor.
            earth_nsteps: Chord discretization forwarded to
                ``EarthLeg``.
            background_day, background_night: Optional real background
                counts per bin (see
                ``tpeanuts.experiments.sno.io.load_backgrounds``).
            extensions: See the class docstring.

        Returns:
            A configured ``SNOModel``.
        """
        eta_rad, exposure = load_nadir_exposure_raw(context=params.context)
        eta_rad = eta_rad[::stride]
        exposure = exposure[::stride].clamp_min(0.0)

        night_mask = eta_rad < (torch.pi / 2)
        day_mask = ~night_mask

        day_table = NadirExposureTable(eta_rad=eta_rad[day_mask], exposure=exposure[day_mask])
        night_table = NadirExposureTable(eta_rad=eta_rad[night_mask], exposure=exposure[night_mask])

        exposure_days_day, exposure_days_night = total_livetime_days()

        return cls(
            params=params,
            free=free, fixed=fixed, theta23=theta23, delta13=delta13,
            solar_medium=solar_medium, solar_source=solar_source,
            source_names=source_names, bin_edges_MeV=bin_edges_MeV,
            day_exposure_table=day_table, night_exposure_table=night_table,
            exposure_days_day=exposure_days_day, exposure_days_night=exposure_days_night,
            context=params.context, detector_depth_m=detector_depth_m,
            earth_method=earth_method, earth_nsteps=earth_nsteps,
            background_day=background_day, background_night=background_night, extensions=extensions,
        )
