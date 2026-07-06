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
Shared torch execution context for tpeanuts computations.

RuntimeContext bundles the device and dtype pair that almost every function in
core, medium, coherent, and pipeline currently receives as two separate
keyword arguments. Passing a single RuntimeContext instead removes that
duplication and gives every layer a single, consistent place to resolve the
default device.

Module contents:
    RuntimeContext
        Frozen container for (device, dtype) used throughout a computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from tpeanuts.util.torch_util import resolve_device


@dataclass(frozen=True)
class RuntimeContext:
    """
    Torch execution context shared by every numerical computation.

    Parameters
    ----------
    device:
        Torch device (CPU or CUDA) where new tensors created during the
        computation are allocated. Determines where the physics calculation
        actually runs; it has no effect on the physical result.

    dtype:
        Real floating dtype used for angles, energies, mass splittings, and
        other real physical quantities. The corresponding complex dtype (used
        for oscillation amplitudes and PMNS matrices) is derived from it via
        ``cdtype_from_real``.
    """

    device: torch.device
    dtype: torch.dtype

    @classmethod
    def resolve(
        cls,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
    ) -> "RuntimeContext":
        """Build a RuntimeContext, resolving ``None`` to the default device.

        Args:
            device: Device specification accepted by ``torch.device``, a
                deferred device factory, or None to select CUDA when
                available and CPU otherwise.
            dtype: Real floating dtype for the resolved context.

        Returns:
            RuntimeContext with a concrete ``torch.device``.
        """
        return cls(device=resolve_device(device), dtype=dtype)
