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

"""Shared torch execution context threaded through every tpeanuts routine.

RuntimeContext bundles the device and real floating dtype pair that almost
every physics routine needs. It is deliberately the *only* place in the
project that resolves a default device or fixes a floating precision: every
``physics``/``source``/``medium``/``detector`` routine receives an
already-built RuntimeContext rather than a loose ``device``/``dtype`` pair.

Just as deliberately, no such routine ever wraps itself in
``torch.no_grad()``: whether gradients track is a decision made by the
caller, at the point where a computation starts (``pipeline`` for a plain
simulation, ``inference`` for a fit that needs gradients), never something
baked into a routine's own implementation. This is what lets the exact same
physics code serve both simulation and inference without a separate
"inference" copy of anything.

Module contents:
    RuntimeContext
        Immutable (device, dtype) pair used throughout a computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from tpeanuts.util.tensor import resolve_device


@dataclass(frozen=True)
class RuntimeContext:
    """Torch execution context shared by every numerical computation.

    Attributes:
        device: Torch device (CPU or CUDA) where tensors created during the
            computation are allocated. Determines where the computation
            physically runs; it has no effect on the physical result.
        dtype: Real floating dtype used for angles, energies, densities, and
            other real physical quantities. The matching complex dtype (used
            for mixing matrices and density matrices) is derived from it via
            ``tpeanuts.util.tensor.cdtype_from_real``.

    Examples:
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> context.device.type, context.dtype
        ('cpu', torch.float64)
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
                zero-argument deferred device factory, or None to select
                CUDA when available and CPU otherwise.
            dtype: Real floating dtype for the resolved context.

        Returns:
            RuntimeContext holding a concrete ``torch.device``.
        """
        return cls(device=resolve_device(device), dtype=dtype)
