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
Background-rate placeholder interface.

No real per-experiment background model is implemented yet: Borexino's own
backgrounds alone (14C, 210Po, 85Kr, 210Bi, cosmogenic isotopes, external
gammas, ...) are a dedicated analysis in the source papers, not something to
approximate here without a citable model. ``zero_background`` is the
explicit placeholder every detector's ``event_rate.py`` uses until a real
one is added -- explicit and zero, not silently omitted, so a future
Poisson fit against real spectral data knows exactly what it is (and is
not) accounting for.

Module contents:
    zero_background(...)
        A zero background-count vector, shaped for
        ``tpeanuts.detector.common.event_rate.predicted_counts``'s
        ``background_counts`` argument.
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
