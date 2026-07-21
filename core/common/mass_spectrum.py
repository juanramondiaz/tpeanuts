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
Model-independent mass-squared-splitting container.

MassSpectrum bundles the two active-sector mass-squared splittings
(DeltamSq21, DeltamSq3l) shared by every oscillation scenario, plus the
model-dependent recipe for turning them into the mass-squared vector a
Hamiltonian builder actually consumes. The active-sector vector construction
(``difference_vector_base``) is identical for the Standard Model and every
BSM extension, so it lives here as a concrete method; sizing that vector to
a specific ``pmns`` flavour count is model-dependent (plain 3-flavour vs.
3+1 sterile with ``DeltamSq41`` appended), so ``difference_vector`` is an
abstract method, implemented by
``tpeanuts.core.SM.sm_mass_spectrum.MassSpectrum_SM`` and
``tpeanuts.core.BSM.bsm_mass_spectrum.MassSpectrum_BSM``.

Module contents:
    MassSpectrum
        Abstract base storing DeltamSq21/DeltamSq3l. ``difference_vector_base``
        builds the reduced three-component active vector; ``difference_vector``
        is the abstract model-specific full vector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.torch_util import infer_device_dtype
from tpeanuts.util.type import as_tensor


@dataclass(frozen=True)
class MassSpectrum(ABC):
    """
    Active-sector mass-squared splittings shared by every oscillation model.

    Parameters
    ----------
    DeltamSq21:
        Solar mass-squared splitting Delta m^2_21 in eV^2. Sets the
        oscillation frequency of the "slow", solar-driven term.

    DeltamSq3l:
        Atmospheric mass-squared splitting Delta m^2_3l in eV^2
        (Delta m^2_31 for normal ordering, Delta m^2_32 for inverted
        ordering, by sign convention). Sets the oscillation frequency of the
        "fast", atmospheric-driven term.
    """

    DeltamSq21: torch.Tensor
    DeltamSq3l: torch.Tensor

    def difference_vector_base(
        self,
        *,
        context: Optional[RuntimeContext] = None,
    ) -> torch.Tensor:
        """Build the reduced three-component active-sector mass-squared vector.

        Args:
            context: Optional runtime device/dtype. When omitted, both are
                inferred from ``DeltamSq21``/``DeltamSq3l``.

        Returns:
            Tensor shaped ``(..., 3)`` with the common phase removed.
        """
        if context is not None:
            device, dtype = context.device, context.dtype
        else:
            device, dtype = infer_device_dtype(self.DeltamSq21, self.DeltamSq3l)
        dm21 = as_tensor(self.DeltamSq21, device=device, dtype=dtype)
        dm3l = as_tensor(self.DeltamSq3l, device=device, dtype=dtype)
        zeros = torch.zeros_like(dm21)

        if dm3l.ndim == 0 and dm3l.device.type == "cpu":
            if dm3l.item() > 0:
                return torch.stack([zeros, dm21, dm3l], dim=-1)
            return torch.stack([-dm21, zeros, dm3l], dim=-1)

        normal = torch.stack([zeros, dm21, dm3l], dim=-1)
        inverted = torch.stack([-dm21, zeros, dm3l], dim=-1)
        return torch.where((dm3l > 0)[..., None], normal, inverted)

    @abstractmethod
    def difference_vector(
        self,
        *,
        context: Optional[RuntimeContext] = None,
    ) -> torch.Tensor:
        """Build the mass-squared vector sized to this spectrum's flavour count.

        Args:
            context: Optional runtime device/dtype. When omitted, both are
                inferred from ``DeltamSq21``/``DeltamSq3l``.

        Returns:
            Tensor shaped ``(..., 3)`` for a 3-flavour spectrum, or
            ``(..., 4)`` for a 3+1 sterile spectrum.
        """
        raise NotImplementedError
