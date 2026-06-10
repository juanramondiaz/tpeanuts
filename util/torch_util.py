

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

from __future__ import annotations

from typing import Union, Optional

import torch

# ============================================================
# Internal helpers
# ============================================================

def _default_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve a concrete torch device from a value or a deferred device factory.

    Args:
        device: Device specification accepted by torch.device, None for the
            default CUDA/CPU selection, or a callable returning either form.

    Returns:
        Resolved torch.device instance.
    """
    if callable(device):
        return device()
    return _default_device(device)


def _resolve_dtype(dtype: Optional[torch.dtype], *values) -> torch.dtype:
    """
    Resolve a real floating torch dtype from an explicit dtype or input values.

    Args:
        dtype: Optional dtype requested by the caller. When provided, it is
            returned unchanged.
        *values: Candidate values inspected in order. The first tensor with
            dtype torch.float32 or torch.float64 determines the result.

    Returns:
        Explicit dtype, the first floating tensor dtype found in values, or
        torch.float64 when no suitable tensor dtype is available.
    """
    if dtype is not None:
        return dtype

    for value in values:
        if torch.is_tensor(value) and value.dtype in (torch.float32, torch.float64):
            return value.dtype

    return torch.float64


