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
Standard Model (3-flavour, active-sector-only) mass spectrum.

Module contents:
    MassSpectrum_SM
        ``MassSpectrum`` implementation for a plain 3-flavour ``PMNS_SM``
        object: ``difference_vector`` is exactly ``difference_vector_base``,
        with no sterile extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.core.common.mass_spectrum import MassSpectrum
from tpeanuts.util.context import RuntimeContext


@dataclass(frozen=True)
class MassSpectrum_SM(MassSpectrum):
    """3-flavour active-sector mass spectrum (no sterile extension)."""

    def difference_vector(
        self,
        *,
        context: Optional[RuntimeContext] = None,
    ) -> torch.Tensor:
        """Return the reduced three-component active-sector vector unchanged.

        Args:
            context: Optional runtime device/dtype. When omitted, both are
                inferred from ``DeltamSq21``/``DeltamSq3l``.

        Returns:
            Tensor shaped ``(..., 3)``.
        """
        return self.difference_vector_base(context=context)
