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
SNO background-count access point.

Unlike ``detector.borexino.backgrounds`` (a zero placeholder), SNO's real
published day/night background counts
(``data/detector/sno/observation/backgrounds.csv``) are available and used
directly -- this module is a thin, named access point to
``detector.sno.io.load_backgrounds`` matching the other detector packages'
layout, not a model.

Module contents:
    day_night_backgrounds(...)
        (day, night) background counts per bin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from tpeanuts.detector.sno.io import load_backgrounds


def day_night_backgrounds(
    path: Optional[str | Path] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real (day, night) background counts per bin -- see ``detector.sno.io.load_backgrounds``."""
    return load_backgrounds(path, device=device, dtype=dtype)
