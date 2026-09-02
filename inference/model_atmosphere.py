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
Differentiable atmospheric-medium oscillation model: theta23/DeltamSq3l, Earth matter.

``AtmosphericOscillationModel`` is the mirror image of
``tpeanuts.medium.vacuum.oscillation_model.VacuumOscillationModel``: same SM
PMNS_SM/MassSpectrum_SM construction, but frees the *other* two oscillation
parameters -- theta23 and DeltamSq3l are what an atmospheric-neutrino
measurement (e.g. IceCube DeepCore) actually constrains, while theta12/
theta13/DeltamSq21 are fixed at their real (NuFit) values, the reverse of
``tpeanuts.inference.solar_model.SolarSMOscillationModel``/
``VacuumOscillationModel``'s theta12/theta13/DeltamSq21-free,
theta23-fixed split. Both splits follow from the same |U_ei|^2 product
structure (``core.SM.sm_pmns.PMNS_SM.pmns_matrix``): P_ee depends only on
theta12/theta13/DeltamSq21 (theta23/delta drop out), while atmospheric
nu_mu disappearance/nu_tau appearance depend on all six parameters in
general, but are overwhelmingly dominated by theta23/DeltamSq3l/DeltamSq21's
already-tiny splitting relative to DeltamSq3l -- theta12/theta13/DeltamSq21
are held fixed here at their solar/reactor-measured values rather than
re-fit from atmospheric data alone, matching how every published
atmospheric-neutrino oscillation analysis treats them.

Unlike ``SolarSMOscillationModel``/``VacuumOscillationModel`` (whose
``antinu`` flag is fixed once at construction, since a solar analysis only
ever sees neutrinos and a reactor analysis only ever sees antineutrinos),
``AtmosphericOscillationModel.oscillation`` takes ``antinu`` as a call-time
argument: a real atmospheric-neutrino sample (e.g. IceCube's event-by-event
Monte Carlo, tagged by PDG code sign) genuinely mixes neutrinos and
antineutrinos within the same fit, and ``core.common.oscillation
.OscillationParameters.antinu``/``core.common.pmns.PMNS.select_antinu``
already support a per-event boolean tensor mask for exactly this reason.

Does not itself compute a transition probability (unlike
``SolarSMOscillationModel.predict_pee``/``VacuumOscillationModel
.predict_pee``, which have one natural scalar observable, P_ee): an
atmospheric detector needs the full flavour-transition matrix through
Earth matter at each event's own (energy, nadir angle), which
``tpeanuts.medium.earth.probability.earth_probability_transition`` already
provides directly from an ``OscillationParameters`` object -- see
``tpeanuts.detector.icecube.event_rate`` for that composition.

Module contents:
    FREE_PARAM_KEYS
        The two parameter names ``AtmosphericOscillationModel`` can free.
    AtmosphericOscillationModel
        Holds the fixed parameters and builds OscillationParameters (given
        an explicit antinu mask) from a free-parameter tensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from tpeanuts.config.presets import OSCILLATION_PRESETS, get_preset
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.util.context import RuntimeContext

FREE_PARAM_KEYS = ("theta23", "DeltamSq3l")


@dataclass(frozen=True)
class AtmosphericOscillationModel:
    """Differentiable SM atmospheric oscillation model, parametrised by ``free``.

    Parameters
    ----------
    context:
        Runtime device/dtype shared by every tensor the model builds.
    free:
        Names (subset of ``FREE_PARAM_KEYS``, in order) of the parameters
        read from the ``theta`` vector passed to ``oscillation``. Every
        other entry of ``FREE_PARAM_KEYS`` is held fixed at
        ``fixed[name]``.
    fixed:
        Values (real tensors on ``context``) for every ``FREE_PARAM_KEYS``
        entry not in ``free``.
    theta12, theta13, delta13:
        Fixed solar/reactor angle and CP phase in radians (see module
        docstring for why these are not re-fit from atmospheric data).
    DeltamSq21:
        Fixed solar mass splitting in eV^2.
    """

    context: RuntimeContext
    free: tuple[str, ...]
    fixed: dict[str, torch.Tensor]
    theta12: torch.Tensor
    theta13: torch.Tensor
    delta13: torch.Tensor
    DeltamSq21: torch.Tensor

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        *,
        free: Sequence[str] = FREE_PARAM_KEYS,
        context: Optional[RuntimeContext] = None,
    ) -> tuple["AtmosphericOscillationModel", torch.Tensor]:
        """Build a model and its initial ``theta`` vector from a named preset.

        Args:
            preset_name: Name in ``tpeanuts.config.presets.OSCILLATION_PRESETS``
                supplying every starting value.
            free: Parameter names (subset of ``FREE_PARAM_KEYS``) to expose
                as fit parameters.
            context: Runtime device/dtype. None resolves the default device
                with float64.

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
            "theta23": math.radians(float(preset["theta23_deg"])),
            "DeltamSq3l": float(preset["DeltamSq3l"]),
        }

        fixed = {
            name: torch.tensor(value, dtype=context.dtype, device=context.device)
            for name, value in values.items()
            if name not in free
        }
        theta12 = torch.tensor(
            math.radians(float(preset["theta12_deg"])), dtype=context.dtype, device=context.device,
        )
        theta13 = torch.tensor(
            math.radians(float(preset["theta13_deg"])), dtype=context.dtype, device=context.device,
        )
        delta13 = torch.tensor(
            math.radians(float(preset["delta13_deg"])), dtype=context.dtype, device=context.device,
        )
        DeltamSq21 = torch.tensor(
            float(preset["DeltamSq21"]), dtype=context.dtype, device=context.device,
        )

        model = cls(
            context=context, free=free, fixed=fixed,
            theta12=theta12, theta13=theta13, delta13=delta13, DeltamSq21=DeltamSq21,
        )
        theta0 = torch.tensor(
            [values[name] for name in free],
            dtype=context.dtype,
            device=context.device,
            requires_grad=True,
        )
        return model, theta0

    def oscillation(
        self, theta: torch.Tensor, antinu: Union[bool, torch.Tensor] = False,
    ) -> OscillationParameters:
        """Build a fresh, differentiable ``OscillationParameters`` from ``theta``.

        Args:
            theta: 1-D tensor of length ``len(self.free)``, values in
                ``self.free`` order. Leaf or non-leaf; gradients w.r.t. any
                ancestor tensor of ``theta`` propagate through the returned
                object's ``pmns``/``mass_spectrum``.
            antinu: Bool or boolean tensor selecting antineutrino
                propagation, forwarded unchanged to ``OscillationParameters``
                (see module docstring -- unlike the solar/vacuum models,
                this is a call-time argument, not fixed at construction).

        Returns:
            OscillationParameters built with a plain 3-flavour PMNS_SM and
            MassSpectrum_SM (no NSI, no sterile extension).
        """
        values = dict(self.fixed)
        values.update(zip(self.free, theta.unbind()))

        pmns_params = PMNSParams(
            theta12=self.theta12,
            theta13=self.theta13,
            theta23=values["theta23"],
            delta=self.delta13,
            context=self.context,
        )
        pmns = PMNS_SM(pmns_params)
        mass_spectrum = MassSpectrum_SM(
            DeltamSq21=self.DeltamSq21, DeltamSq3l=values["DeltamSq3l"],
        )
        return OscillationParameters(pmns=pmns, mass_spectrum=mass_spectrum, antinu=antinu)
