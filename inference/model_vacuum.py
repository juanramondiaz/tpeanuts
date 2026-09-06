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
Differentiable vacuum-medium P_ee(E, L) reactor antineutrino oscillation model.

Medium physics, not detector physics: any reactor experiment wraps this in
its own detector-composition layer instead of duplicating the vacuum-
propagation path. Builds
``OscillationParameters`` with ``antinu=True`` (reactors emit nu_e_bar) and
predicts P_ee via vacuum propagation directly, since reactor baselines are
traversed in vacuum and need no medium profile.

Notes:
    - Only theta12, theta13, DeltamSq21 and DeltamSq3l affect P_ee; theta23
      and the CP phase drop out of the survival probability and are held
      fixed, matching ``model_solar.FREE_PARAM_KEYS`` (kept as its own copy
      here, since the two media sharing the same four free names is not a
      reason to import across packages).

Module contents:
    FREE_PARAM_KEYS
        The four parameter names ``VacuumOscillationModel`` can free.
    VacuumOscillationModel
        Holds the fixed parameters and predicts P_ee(E, L) from a
        free-parameter tensor, antinu=True vacuum propagation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from tpeanuts.config.presets import OSCILLATION_PRESETS, get_preset
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.medium.vacuum.probability import vacuum_probability_state
from tpeanuts.util.context import RuntimeContext

FREE_PARAM_KEYS = ("theta12", "theta13", "DeltamSq21", "DeltamSq3l")

# Pure electron-antineutrino flavour state, the source term for reactor IBD.
_NUE_STATE = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128)


@dataclass(frozen=True)
class VacuumOscillationModel:
    """Differentiable SM reactor nu_e_bar survival-probability model.

    Same free/fixed structure as
    ``tpeanuts.inference.model_solar.SolarSMOscillationModel``, but
    builds ``OscillationParameters(antinu=True)`` and evaluates ``P_ee(E, L)``
    in vacuum (see module docstring).

    Parameters
    ----------
    context:
        Runtime device/dtype shared by every tensor the model builds.
    free:
        Names (subset of ``FREE_PARAM_KEYS``, in order) read from the
        ``theta`` vector.
    fixed:
        Values for every ``FREE_PARAM_KEYS`` entry not in ``free``.
    theta23, delta13:
        Fixed angle/phase in radians (do not affect P_ee; see module
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
        free: Sequence[str] = ("theta13", "DeltamSq21"),
        context: Optional[RuntimeContext] = None,
    ) -> tuple["VacuumOscillationModel", torch.Tensor]:
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
                ``self.free`` order.

        Returns:
            OscillationParameters built with a plain 3-flavour PMNS_SM and
            MassSpectrum_SM, ``antinu=True``.
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
        return OscillationParameters(pmns=pmns, mass_spectrum=mass_spectrum, antinu=True)

    def predict_pee(
        self,
        theta: torch.Tensor,
        L_km: float,
        E_nu_grid_MeV: torch.Tensor,
    ) -> torch.Tensor:
        """Vacuum nu_e_bar survival probability on ``E_nu_grid_MeV`` at baseline ``L_km``.

        Args:
            theta: 1-D free-parameter tensor, see ``oscillation``.
            L_km: Baseline, kilometres.
            E_nu_grid_MeV: True antineutrino energy grid.

        Returns:
            Real tensor shaped ``(n_E,)``, differentiable w.r.t. ``theta``.
        """
        oscillation = self.oscillation(theta)
        state = _NUE_STATE.to(device=self.context.device)
        P = vacuum_probability_state(
            state, oscillation, E_nu_grid_MeV, L_km,
            massbasis=False, context=self.context,
        )
        return P[..., 0]
