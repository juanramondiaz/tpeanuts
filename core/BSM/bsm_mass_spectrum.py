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
3+1 sterile mass spectrum, for BSM neutrino propagation.

Module contents
---------------
MassSpectrum_BSM
    ``tpeanuts.core.common.mass_spectrum.MassSpectrum`` implementation for a
    3+1 sterile ``PMNS_sterile`` object: ``difference_vector`` appends
    ``DeltamSq41`` to the reduced active-sector vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.core.common.mass_spectrum import MassSpectrum
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.torch_util import infer_device_dtype
from tpeanuts.util.type import as_tensor


# ---------------------------------------------------------------------------
# MassSpectrum_BSM: 3+1 sterile mass spectrum
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MassSpectrum_BSM(MassSpectrum):
    """3+1 sterile mass spectrum: active vector plus ``DeltamSq41``.

    Parameters
    ----------
    DeltamSq41:
        Sterile mass-squared splitting Delta m^2_41 in eV^2. Like
        ``DeltamSq21``/``DeltamSq3l``, it does not enter any mixing-matrix
        rotation (only the active-sterile mixing angles do). ``None`` is
        accepted at construction time but raises in ``difference_vector``,
        since a 3+1 sterile Hamiltonian genuinely requires it.
    """

    DeltamSq41: Optional[torch.Tensor] = None

    def difference_vector(
        self,
        *,
        context: Optional[RuntimeContext] = None,
    ) -> torch.Tensor:
        """Append ``DeltamSq41`` to the reduced active-sector vector.

        Args:
            context: Optional runtime device/dtype. When omitted, both are
                inferred from ``DeltamSq21``/``DeltamSq3l``.

        Returns:
            Tensor shaped ``(..., 4)``.

        Raises:
            ValueError: If ``DeltamSq41`` is ``None``.
        """
        if context is not None:
            device, dtype = context.device, context.dtype
        else:
            device, dtype = infer_device_dtype(self.DeltamSq21, self.DeltamSq3l)
        base = self.difference_vector_base(context=RuntimeContext(device=device, dtype=dtype))

        if self.DeltamSq41 is None:
            raise ValueError(
                "BSM Hamiltonian construction for a 3+1 sterile mass spectrum "
                "requires DeltamSq41."
            )
        DeltamSq41_t = as_tensor(self.DeltamSq41, device=device, dtype=dtype)
        return torch.cat(
            [base, DeltamSq41_t.expand(base.shape[:-1]).unsqueeze(-1)],
            dim=-1,
        )
