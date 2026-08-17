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
Borexino background placeholder.

No real background model (14C, 210Po, 85Kr, 210Bi, cosmogenic isotopes,
external gammas, ...) is implemented: each is its own dedicated analysis in
the Borexino papers, not something to approximate here without a citable
model. See ``tpeanuts.detector.common.background``'s module docstring for
why this is explicit and zero rather than silently omitted.

Module contents:
    backgrounds_MeV(...)
        Zero background per bin, at Borexino's bin count.
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.detector.common.background import zero_background


def backgrounds_MeV(
    n_bins: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Borexino background counts per bin -- currently always zero, see module docstring."""
    return zero_background(n_bins, device=device, dtype=dtype)
