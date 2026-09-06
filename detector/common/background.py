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

"""Background-count utilities.

No experiment-specific background model is defined in this module. It only
provides a zero-valued vector for calculations without a background model.

Module functions:
    zero_background(...)
        Return one zero background count per observed bin.
"""

from __future__ import annotations

from typing import Optional

import torch


def zero_background(
    n_bins: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Zero background counts, one entry per bin.

    Args:
        n_bins: Number of observed bins.
        device: Target torch device.
        dtype: Target real dtype.

    Returns:
        Real tensor shaped ``(n_bins,)``, all zeros.
    """
    return torch.zeros(n_bins, device=device, dtype=dtype)
