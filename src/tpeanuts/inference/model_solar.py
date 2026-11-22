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

"""Differentiable solar-medium P_ee(E) oscillation models, at a fixed set of (source, energy) points.

Phase 9 (the last phase) of the solar migration. Wraps
``tpeanuts.pipeline.solar.solar_flux`` behind
``tpeanuts.inference.model.DifferentiableModel``'s ``free``/
``predict(theta)`` surface, mirroring
``tpeanuts.inference.model_reactor.ReactorEventRateModel`` exactly: no
new physics, only a differentiable re-composition of already-tested
pieces (``tpeanuts.source.solar``, ``tpeanuts.medium.solar``,
``tpeanuts.pipeline.solar``).

Only theta12, theta13, DeltamSq21, and DeltamSq3l affect P_ee; theta23
and the CP phase drop out of the survival probability and are held fixed
(matching v1_0's own documented reasoning) rather than exposed as free
parameters.

Ported from v1_0's ``inference/model_solar.py``
(``SolarSMOscillationModel``/``SolarNSIOscillationModel``/
``SolarPointModel``), collapsed from that three-class split (an
oscillation-parameter model plus a separate point-binding wrapper) into
one frozen dataclass per variant that both builds the propagation model
*and* exposes ``predict(theta)`` directly -- matching
``ReactorEventRateModel``'s own single-class shape, not v1_0's. v1_0's
``from_preset`` classmethods (building starting values from
``config.presets.OSCILLATION_PRESETS``) are not ported: that named
-preset registry does not exist anywhere in this rewrite yet (see
``tpeanuts.physics.model``'s own module docstring -- "a later, separate
piece: ``tpeanuts.config``, not yet built") and ``ReactorEventRateModel``
itself has no such classmethod either, for the same reason -- callers
supply starting values directly, as every other differentiable model in
this rewrite already does.

Each ``predict(theta)`` call evaluates one ``solar_flux`` (without intensity: only the
probabilities)
per bound ``(source, energy)`` point (not a spectrum-averaged rate) --
matching how e.g. Borexino's Nature 2018 P_ee(E) measurement itself
reports one representative energy per source.

Module contents:
    PARAM_KEYS
        The four parameter names ``SolarPointModel`` can free.
    SolarPointModel
        Differentiable plain-Standard-Model P_ee(E) at bound (source,
        energy) points.
    NSI_PARAM_KEYS
        ``PARAM_KEYS`` plus ``"eps_ee"``.
    SolarNSIPointModel
        Like ``SolarPointModel``, with an additional diagonal NSI
        coupling -- always evaluated via ``method="adiabatic_exact"``,
        since the closed-form adiabatic-approximated path has no NSI
        generalisation (see ``tpeanuts.physics.evolution.adiabatic.adiabatic``'s
        own module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import torch

from tpeanuts.medium.solar.solar import SolarMedium
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix, StandardMixingParameters
from tpeanuts.physics.common.nature import Dirac
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.extensions.nsi import NSIConfig, NSITerm
from tpeanuts.physics.model import PropagationModel
from tpeanuts.pipeline._legs import SolarLeg
from tpeanuts.pipeline.solar import solar_flux
from tpeanuts.source.solar.solar import SolarSource
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike

PARAM_KEYS = ("theta12", "theta13", "DeltamSq21", "DeltamSq3l")
NSI_PARAM_KEYS = PARAM_KEYS + ("eps_ee",)


def _validate_free_fixed(free: Tuple[str, ...], fixed: Mapping[str, TensorLike], keys: Tuple[str, ...]) -> None:
    free_set = set(free)
    fixed_set = set(fixed)
    if len(free_set) != len(free):
        raise ValueError(f"free contains a duplicate parameter name: {free!r}.")
    if free_set & fixed_set:
        raise ValueError(
            f"free and fixed both name {sorted(free_set & fixed_set)} -- "
            "each parameter must be in exactly one of them."
        )
    covered = free_set | fixed_set
    if covered != set(keys):
        missing = set(keys) - covered
        unknown = covered - set(keys)
        raise ValueError(
            f"free/fixed together must cover exactly {keys}; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )


def _validate_points(sources: Tuple[str, ...], energies_MeV: Tuple[TensorLike, ...]) -> None:
    if len(sources) != len(energies_MeV):
        raise ValueError(
            f"sources and energies_MeV must have the same length, got {len(sources)} and {len(energies_MeV)}."
        )
    if len(sources) == 0:
        raise ValueError("sources/energies_MeV must bind at least one (source, energy) point.")


@dataclass(frozen=True)
class SolarPointModel:
    """Differentiable plain-SM solar P_ee(E), at a fixed set of (source, energy) points.

    Attributes:
        free: Names of the parameters to fit, a subset of ``PARAM_KEYS``
            (``"theta12"``, ``"theta13"``, ``"DeltamSq21"``,
            ``"DeltamSq3l"``), in the order ``predict``'s ``theta``
            argument is unpacked.
        fixed: Values (radians for angles, eV^2 for the splittings) for
            every key in ``PARAM_KEYS`` not in ``free``.
        theta23, delta13: Fixed atmospheric angle and CP phase, in
            radians -- do not affect P_ee, only needed to build a
            well-formed PMNS matrix.
        medium: Solar density profile; independent of ``theta``.
        source: Solar production source (radius/flux/spectrum tables);
            independent of ``theta``.
        sources: Solar source key per data point (e.g. ``"pp"``,
            ``"8B"``), same length and order as ``energies_MeV``.
        energies_MeV: Scalar neutrino energy in MeV per data point.
        context: RuntimeContext every tensor this model builds is on.

    Raises:
        ValueError: If ``free``/``fixed`` do not together cover exactly
            ``PARAM_KEYS`` with no overlap and no gap, or if
            ``sources``/``energies_MeV`` are empty or mismatched in
            length.
    """

    free: Tuple[str, ...]
    fixed: Mapping[str, TensorLike]
    theta23: TensorLike
    delta13: TensorLike
    medium: SolarMedium
    source: SolarSource
    sources: Tuple[str, ...]
    energies_MeV: Tuple[TensorLike, ...]
    context: RuntimeContext

    def __post_init__(self) -> None:
        _validate_free_fixed(self.free, self.fixed, PARAM_KEYS)
        _validate_points(self.sources, self.energies_MeV)

    def _build_model(self, theta: torch.Tensor) -> PropagationModel:
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
        return PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum)

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """Predict P_ee independently at every bound ``(source, energy)`` point.

        Args:
            theta: Free-parameter vector, shaped ``(len(self.free),)``,
                same order as ``self.free``.

        Returns:
            1-D tensor of predicted P_ee, one entry per bound point,
            same order as ``self.sources``/``self.energies_MeV``.
        """
        model = self._build_model(theta)
        predictions = [
            solar_flux(model, self.source, self.medium, source_key, energy, intensity=False)[source_key]
            .state.probabilities()[..., 0]
            for source_key, energy in zip(self.sources, self.energies_MeV, strict=True)
        ]
        return torch.stack(predictions)


@dataclass(frozen=True)
class SolarNSIPointModel:
    """Like ``SolarPointModel``, with an additional diagonal NSI coupling ``eps_ee``.

    Always evaluated via ``method="adiabatic_exact"``: the closed-form
    ``adiabatic_approximated`` path has no NSI generalisation (see
    ``tpeanuts.physics.evolution.adiabatic.adiabatic``'s own module
    docstring, and its ``mass_weights``'s own rejection of
    ``variant="approximated"`` with an active extension).

    Attributes:
        free: Names of the parameters to fit, a subset of
            ``NSI_PARAM_KEYS`` (``PARAM_KEYS`` plus ``"eps_ee"``).
        fixed: Values for every key in ``NSI_PARAM_KEYS`` not in
            ``free``.
        theta23, delta13, medium, source, sources, energies_MeV,
        context: Same meaning as ``SolarPointModel``.

    Raises:
        ValueError: Same conditions as ``SolarPointModel``, against
            ``NSI_PARAM_KEYS``.
    """

    free: Tuple[str, ...]
    fixed: Mapping[str, TensorLike]
    theta23: TensorLike
    delta13: TensorLike
    medium: SolarMedium
    source: SolarSource
    sources: Tuple[str, ...]
    energies_MeV: Tuple[TensorLike, ...]
    context: RuntimeContext

    def __post_init__(self) -> None:
        _validate_free_fixed(self.free, self.fixed, NSI_PARAM_KEYS)
        _validate_points(self.sources, self.energies_MeV)

    def _build_model(self, theta: torch.Tensor) -> PropagationModel:
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
        nsi_term = NSITerm(NSIConfig(context=self.context, eps_ee=params["eps_ee"]))
        return PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, extensions=(nsi_term,))

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """Predict P_ee independently at every bound ``(source, energy)`` point.

        Args:
            theta: Free-parameter vector, shaped ``(len(self.free),)``,
                same order as ``self.free``.

        Returns:
            1-D tensor of predicted P_ee, one entry per bound point.
        """
        model = self._build_model(theta)
        predictions = [
            solar_flux(
                model, self.source, self.medium, source_key, energy,
                solar_leg=SolarLeg(method="adiabatic_exact"), intensity=False,
            )[source_key].state.probabilities()[..., 0]
            for source_key, energy in zip(self.sources, self.energies_MeV, strict=True)
        ]
        return torch.stack(predictions)
