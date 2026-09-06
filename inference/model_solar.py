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
Differentiable solar-medium P_ee(E) oscillation models.

Medium physics, not detector physics: any solar-fed detector wraps one of
these models in its own detector-composition layer instead of duplicating
the solar-production/MSW-conversion path. Each model builds a fresh
``OscillationParameters`` from a 1-D free-parameter tensor on every call, so
autograd can differentiate the predicted P_ee with respect to it.

Notes:
    - Only theta12, theta13, DeltamSq21 and DeltamSq3l affect P_ee; theta23
      and the CP phase drop out of the survival probability and are held
      fixed at their preset value.
    - ``SolarNSIOscillationModel`` adds a single diagonal NSI coupling,
      ``eps_ee``, on top of the same four parameters, and always evaluates
      the exact (pointwise Hamiltonian-diagonalisation) propagation method,
      since the fast adiabatic approximation has no NSI generalisation.
    - ``SolarPointModel`` adapts either model to a plain, single-argument
      ``predict(theta)``, for fits against a fixed set of (source, energy)
      data points rather than through a detector's binned response.

Module contents:
    FREE_PARAM_KEYS
        The four parameter names ``SolarSMOscillationModel`` can free.
    SolarSMOscillationModel
        Holds the fixed parameters and builds OscillationParameters/predicts
        P_ee from a free-parameter tensor.
    NSI_FREE_PARAM_KEYS
        ``FREE_PARAM_KEYS`` plus ``"eps_ee"``.
    SolarNSIOscillationModel
        Like ``SolarSMOscillationModel``, with an additional diagonal NSI
        coupling and the ``adiabatic_exact`` evaluation path.
    SolarPointModel
        Adapts either model to the single-argument ``predict(theta)``
        contract for a fixed set of (source, energy) points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from tpeanuts.config.presets import OSCILLATION_PRESETS, get_preset
from tpeanuts.core.BSM.bsm_nsi import NSIConfig
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.medium.solar.probability import solar_probability_state
from tpeanuts.medium.solar.profile import SolarMediumProfile
from tpeanuts.source.solar import SolarNeutrinoSource
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike

FREE_PARAM_KEYS = ("theta12", "theta13", "DeltamSq21", "DeltamSq3l")
NSI_FREE_PARAM_KEYS = FREE_PARAM_KEYS + ("eps_ee",)


@dataclass(frozen=True)
class SolarSMOscillationModel:
    """Differentiable SM solar P_ee(E) model, parametrised by ``free``.

    Parameters
    ----------
    context:
        Runtime device/dtype shared by every tensor the model builds.
    free:
        Names (subset of ``FREE_PARAM_KEYS``, in order) of the parameters
        read from the ``theta`` vector passed to ``oscillation``/
        ``predict_pee``. Every other entry of ``FREE_PARAM_KEYS`` is held
        fixed at ``fixed[name]``.
    fixed:
        Values (real tensors on ``context``) for every ``FREE_PARAM_KEYS``
        entry not in ``free``.
    theta23:
        Fixed atmospheric angle in radians (does not affect P_ee; only
        needed to build a well-formed 3x3 PMNS matrix).
    delta13:
        Fixed CP phase in radians (does not affect P_ee; see module
        docstring).
    """

    context: RuntimeContext
    free: tuple[str, ...]
    fixed: dict[str, torch.Tensor]
    theta23: torch.Tensor
    delta13: torch.Tensor

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        *,
        free: Sequence[str] = ("theta12", "DeltamSq21"),
        context: Optional[RuntimeContext] = None,
    ) -> tuple["SolarSMOscillationModel", torch.Tensor]:
        """Build a model and its initial ``theta`` vector from a named preset.

        Args:
            preset_name: Name in ``tpeanuts.config.presets.OSCILLATION_PRESETS``
                (e.g. "_SM_NUFIT61_NO") supplying every starting value.
            free: Parameter names (subset of ``FREE_PARAM_KEYS``) to expose
                as fit parameters. The rest stay fixed at the preset value.
            context: Runtime device/dtype. None resolves the default
                device with float64.

        Returns:
            ``(model, theta0)``: the built model and a 1-D leaf tensor of
            length ``len(free)`` (in ``free`` order), with
            ``requires_grad=True``, holding the preset's starting values.

        Raises:
            ValueError: If ``free`` contains a name outside
                ``FREE_PARAM_KEYS`` or a duplicate.
        """
        free = tuple(free)
        if len(set(free)) != len(free) or any(name not in FREE_PARAM_KEYS for name in free):
            raise ValueError(
                f"free must be distinct names from {FREE_PARAM_KEYS}, got {free!r}."
            )

        context = context or RuntimeContext.resolve(None, torch.float64)
        preset = get_preset(OSCILLATION_PRESETS, preset_name, kind="oscillation preset")

        values = {
            "theta12": math.radians(float(preset["theta12_deg"])),
            "theta13": math.radians(float(preset["theta13_deg"])),
            "DeltamSq21": float(preset["DeltamSq21"]),
            "DeltamSq3l": float(preset["DeltamSq3l"]),
        }

        fixed = {
            name: torch.tensor(value, dtype=context.dtype, device=context.device)
            for name, value in values.items()
            if name not in free
        }
        theta23 = torch.tensor(
            math.radians(float(preset["theta23_deg"])), dtype=context.dtype, device=context.device,
        )
        delta13 = torch.tensor(
            math.radians(float(preset["delta13_deg"])), dtype=context.dtype, device=context.device,
        )

        model = cls(context=context, free=free, fixed=fixed, theta23=theta23, delta13=delta13)
        theta0 = torch.tensor(
            [values[name] for name in free],
            dtype=context.dtype,
            device=context.device,
            requires_grad=True,
        )
        return model, theta0

    def oscillation(self, theta: torch.Tensor) -> OscillationParameters:
        """Build a fresh, differentiable ``OscillationParameters`` from ``theta``.

        Args:
            theta: 1-D tensor of length ``len(self.free)``, values in
                ``self.free`` order. Leaf or non-leaf; gradients w.r.t. any
                ancestor tensor of ``theta`` propagate through the returned
                object's ``pmns``/``mass_spectrum``.

        Returns:
            OscillationParameters built with a plain 3-flavour PMNS_SM and
            MassSpectrum_SM (no NSI, no sterile extension, antinu=False).
        """
        values = dict(self.fixed)
        values.update(zip(self.free, theta.unbind()))

        pmns_params = PMNSParams(
            theta12=values["theta12"],
            theta13=values["theta13"],
            theta23=self.theta23,
            delta=self.delta13,
            context=self.context,
        )
        pmns = PMNS_SM(pmns_params)
        mass_spectrum = MassSpectrum_SM(
            DeltamSq21=values["DeltamSq21"], DeltamSq3l=values["DeltamSq3l"],
        )
        return OscillationParameters(pmns=pmns, mass_spectrum=mass_spectrum)

    def predict_pee(
        self,
        theta: torch.Tensor,
        medium: SolarMediumProfile,
        source: SolarNeutrinoSource,
        sources: Sequence[str],
        energies_MeV: Sequence[TensorLike],
    ) -> torch.Tensor:
        """Predict P_ee independently for each ``(source, energy)`` pair.

        Each pair is a separate ``solar_probability_state`` call (a
        single-energy point evaluation, not a spectrum-averaged rate): this
        matches how the Borexino Nature 2018 P_ee(E) points are themselves
        reported, one representative energy per source.

        Args:
            theta: 1-D free-parameter tensor, see ``oscillation``.
            medium: Solar density profile; independent of ``theta``.
            source: Solar production source (radius/flux/spectrum tables);
                independent of ``theta``.
            sources: Solar source key per data point (e.g. "pp", "7Be").
            energies_MeV: Scalar neutrino energy in MeV per data point,
                same length and order as ``sources``.

        Returns:
            1-D tensor of predicted P_ee, one entry per ``(source, energy)``
            pair, same length as ``sources``.
        """
        oscillation = self.oscillation(theta)
        predictions = [
            solar_probability_state(oscillation, energy, medium, source, source_key)[..., 0]
            for source_key, energy in zip(sources, energies_MeV)
        ]
        return torch.stack(predictions)


@dataclass(frozen=True)
class SolarNSIOscillationModel:
    """Differentiable solar P_ee(E) model with a diagonal NSI coupling.

    Same construction as ``SolarSMOscillationModel`` (``theta12``, ``theta13``,
    ``DeltamSq21``, ``DeltamSq3l``), plus ``eps_ee``. Always evaluates via
    ``method="adiabatic_exact"`` (see module docstring for why).

    Parameters
    ----------
    context, theta23, delta13:
        Same meaning as ``SolarSMOscillationModel``.
    free:
        Names (subset of ``NSI_FREE_PARAM_KEYS``, in order) read from the
        ``theta`` vector.
    fixed:
        Values for every ``NSI_FREE_PARAM_KEYS`` entry not in ``free``.
    """

    context: RuntimeContext
    free: tuple[str, ...]
    fixed: dict[str, torch.Tensor]
    theta23: torch.Tensor
    delta13: torch.Tensor

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        *,
        free: Sequence[str] = ("theta12", "eps_ee"),
        eps_ee0: float = 0.0,
        context: Optional[RuntimeContext] = None,
    ) -> tuple["SolarNSIOscillationModel", torch.Tensor]:
        """Build a model and its initial ``theta`` vector from a named preset.

        Args:
            preset_name: Name in ``tpeanuts.config.presets.OSCILLATION_PRESETS``
                supplying every SM starting value (``eps_ee`` has no entry
                there -- an oscillation preset carries no NSI information,
                see ``config.presets``' module docstring -- so its starting
                value is given directly via ``eps_ee0``).
            free: Parameter names (subset of ``NSI_FREE_PARAM_KEYS``) to
                expose as fit parameters.
            eps_ee0: Starting value for ``eps_ee`` (whether free or fixed).
            context: Runtime device/dtype. None resolves the default
                device with float64.

        Returns:
            ``(model, theta0)``: the built model and a 1-D leaf tensor of
            length ``len(free)`` (in ``free`` order), with
            ``requires_grad=True``.

        Raises:
            ValueError: If ``free`` contains a name outside
                ``NSI_FREE_PARAM_KEYS`` or a duplicate.
        """
        free = tuple(free)
        if len(set(free)) != len(free) or any(name not in NSI_FREE_PARAM_KEYS for name in free):
            raise ValueError(
                f"free must be distinct names from {NSI_FREE_PARAM_KEYS}, got {free!r}."
            )

        context = context or RuntimeContext.resolve(None, torch.float64)
        preset = get_preset(OSCILLATION_PRESETS, preset_name, kind="oscillation preset")

        values = {
            "theta12": math.radians(float(preset["theta12_deg"])),
            "theta13": math.radians(float(preset["theta13_deg"])),
            "DeltamSq21": float(preset["DeltamSq21"]),
            "DeltamSq3l": float(preset["DeltamSq3l"]),
            "eps_ee": float(eps_ee0),
        }

        fixed = {
            name: torch.tensor(value, dtype=context.dtype, device=context.device)
            for name, value in values.items()
            if name not in free
        }
        theta23 = torch.tensor(
            math.radians(float(preset["theta23_deg"])), dtype=context.dtype, device=context.device,
        )
        delta13 = torch.tensor(
            math.radians(float(preset["delta13_deg"])), dtype=context.dtype, device=context.device,
        )

        model = cls(context=context, free=free, fixed=fixed, theta23=theta23, delta13=delta13)
        theta0 = torch.tensor(
            [values[name] for name in free],
            dtype=context.dtype,
            device=context.device,
            requires_grad=True,
        )
        return model, theta0

    def oscillation(self, theta: torch.Tensor) -> OscillationParameters:
        """Build a fresh, differentiable ``OscillationParameters`` from ``theta``.

        Args:
            theta: 1-D tensor of length ``len(self.free)``, values in
                ``self.free`` order.

        Returns:
            OscillationParameters built with a plain 3-flavour PMNS_SM,
            MassSpectrum_SM, and a diagonal-only NSIConfig (``eps_ee`` on
            the (e, e) entry, every other coupling exactly zero).
        """
        values = dict(self.fixed)
        values.update(zip(self.free, theta.unbind()))

        pmns_params = PMNSParams(
            theta12=values["theta12"],
            theta13=values["theta13"],
            theta23=self.theta23,
            delta=self.delta13,
            context=self.context,
        )
        pmns = PMNS_SM(pmns_params)
        mass_spectrum = MassSpectrum_SM(
            DeltamSq21=values["DeltamSq21"], DeltamSq3l=values["DeltamSq3l"],
        )

        nsi = NSIConfig(
            eps_ee=values["eps_ee"],
            device=self.context.device,
            real_dtype=self.context.dtype,
        )

        return OscillationParameters(pmns=pmns, mass_spectrum=mass_spectrum, nsi=nsi)

    def predict_pee(
        self,
        theta: torch.Tensor,
        medium: SolarMediumProfile,
        source: SolarNeutrinoSource,
        sources: Sequence[str],
        energies_MeV: Sequence[TensorLike],
    ) -> torch.Tensor:
        """Predict P_ee independently for each ``(source, energy)`` pair.

        Same pointwise-energy convention as ``SolarSMOscillationModel
        .predict_pee``, but always via ``method="adiabatic_exact"`` (see
        module docstring).

        Args:
            theta: 1-D free-parameter tensor, see ``oscillation``.
            medium: Solar density profile; independent of ``theta``.
            source: Solar production source; independent of ``theta``.
            sources: Solar source key per data point.
            energies_MeV: Scalar neutrino energy in MeV per data point,
                same length and order as ``sources``.

        Returns:
            1-D tensor of predicted P_ee, one entry per ``(source, energy)``
            pair.
        """
        oscillation = self.oscillation(theta)
        predictions = [
            solar_probability_state(
                oscillation, energy, medium, source, source_key, method="adiabatic_exact",
            )[..., 0]
            for source_key, energy in zip(sources, energies_MeV)
        ]
        return torch.stack(predictions)


@dataclass(frozen=True)
class SolarPointModel:
    """Adapts a pointwise solar oscillation model to the ``predict(theta)`` contract.

    Binds the (fixed, per-analysis) evaluation points -- profile, sources,
    energies -- at construction time, so the wrapped
    ``SolarSMOscillationModel``/``SolarNSIOscillationModel`` only needs to
    expose its richer ``predict_pee(theta, profile, sources, energies_MeV)``
    method (see module docstring).

    Parameters
    ----------
    oscillation_model:
        A ``SolarSMOscillationModel`` or ``SolarNSIOscillationModel``.
    medium:
        Solar density profile.
    source:
        Solar production source (radius/flux/spectrum tables).
    sources:
        Solar source key per data point.
    energies_MeV:
        Scalar neutrino energy per data point, same length/order as
        ``sources``.
    """

    oscillation_model: Union[SolarSMOscillationModel, SolarNSIOscillationModel]
    medium: SolarMediumProfile
    source: SolarNeutrinoSource
    sources: tuple[str, ...]
    energies_MeV: tuple[TensorLike, ...]

    @property
    def free(self) -> tuple[str, ...]:
        """Free oscillation-parameter names, forwarded from ``oscillation_model``."""
        return self.oscillation_model.free

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """Predict P_ee at every bound ``(source, energy)`` point from ``theta``."""
        return self.oscillation_model.predict_pee(
            theta, self.medium, self.source, self.sources, self.energies_MeV,
        )
