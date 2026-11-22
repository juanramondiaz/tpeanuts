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

"""``BorexinoParameters``: the loaded-once bundle of Borexino target/grid constants.

Ported from v1_0's ``experiments/borexino/parameters.py``, which instead
loaded ``N_TARGET_ELECTRONS``/``E_NU_GRID_MEV``/``T_GRID_MEV``/
``TPRIME_GRID_MEV`` as bare module-level constants at import time,
hardcoded to ``dtype=torch.float64`` with no ``device`` at all --
incompatible with this project's own ``RuntimeContext`` convention.
``BorexinoParameters.default(context=...)`` is the replacement: a
frozen dataclass built lazily, on request, on whatever device/dtype the
caller needs, mirroring
``tpeanuts.experiments.dayabay.parameters.DayaBayParameters.default``'s
own "call the loader(s) once, bundle the results" shape.

``PSEUDOCUMENE_COMPOSITION``/``PSEUDOCUMENE_MOLAR_MASS_G_MOL``/
``REFERENCE_TARGET_MASS_TON``/``REFERENCE_EXPOSURE_DAYS``/
``ENERGY_RESOLUTION_A`` stay plain module-level constants (fixed
physical/reference values, not loaded data) -- same treatment
``DayaBayParameters``'s own ``DETECTORS``/``REACTORS`` module
constants get.

Module contents:
    PSEUDOCUMENE_COMPOSITION, PSEUDOCUMENE_MOLAR_MASS_G_MOL
        Pseudocumene (C9H12) composition and molar mass.
    REFERENCE_TARGET_MASS_TON, REFERENCE_EXPOSURE_DAYS
        Reference normalization (100 t, 1 day).
    ENERGY_RESOLUTION_A
        Illustrative sqrt(T) energy-resolution coefficient -- NOT
        verified against the Nature 2018 paper (same caveat v1_0
        itself carries).
    BorexinoParameters
        Frozen dataclass bundling the target electron count and the
        true/reconstructed energy grids, with a
        ``.default(context=...)`` classmethod.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from tpeanuts.detector.common.target import n_electrons
from tpeanuts.util.context import RuntimeContext

Tensor = torch.Tensor

PSEUDOCUMENE_COMPOSITION: Dict[str, int] = {"C": 9, "H": 12}
PSEUDOCUMENE_MOLAR_MASS_G_MOL: float = 120.19
REFERENCE_TARGET_MASS_TON: float = 100.0
REFERENCE_EXPOSURE_DAYS: float = 1.0
ENERGY_RESOLUTION_A: float = 0.05  # illustrative, NOT verified against Nature 2018 paper

# True neutrino energy grid: covers the low-energy pp/7Be/pep range up
# to the 8B endpoint.
_E_NU_GRID_MIN_MEV = 0.15
_E_NU_GRID_MAX_MEV = 16.0
_E_NU_GRID_N = 400

# True/reconstructed electron-recoil grid (Tprime_grid_MeV == T_grid_MeV,
# see BorexinoParameters.default).
_T_GRID_MIN_MEV = 0.05
_T_GRID_MAX_MEV = 14.0
_T_GRID_N = 400


@dataclass(frozen=True)
class BorexinoParameters:
    """Every Borexino target/grid constant, bundled on one device/dtype.

    Attributes:
        context: RuntimeContext every tensor here is built on.
        n_target_electrons: Total target electron count for
            ``REFERENCE_TARGET_MASS_TON`` tons of pseudocumene.
        e_nu_grid_MeV: True neutrino energy grid, shape ``(400,)``.
        T_grid_MeV: True electron-recoil grid, shape ``(400,)``.
        Tprime_grid_MeV: Reconstructed electron-recoil grid -- equal to
            ``T_grid_MeV`` (see ``tpeanuts.detector.common.event_rate
            .predicted_counts``'s own same-grid requirement).
        energy_resolution_a: Illustrative sqrt(T) resolution
            coefficient, see ``ENERGY_RESOLUTION_A``.
        exposure_days: Reference exposure, see
            ``REFERENCE_EXPOSURE_DAYS``.
    """

    context: RuntimeContext
    n_target_electrons: Tensor
    e_nu_grid_MeV: Tensor
    T_grid_MeV: Tensor
    Tprime_grid_MeV: Tensor
    energy_resolution_a: float = ENERGY_RESOLUTION_A
    exposure_days: float = REFERENCE_EXPOSURE_DAYS

    @classmethod
    def default(cls, *, context: RuntimeContext) -> "BorexinoParameters":
        """Build the default Borexino parameter bundle on ``context``.

        Args:
            context: RuntimeContext every returned tensor is built on.

        Returns:
            BorexinoParameters with the reference (100 t, 1 day)
            normalization and the default 400-point energy grids.
        """
        T_grid_MeV = torch.linspace(
            _T_GRID_MIN_MEV, _T_GRID_MAX_MEV, _T_GRID_N, device=context.device, dtype=context.dtype,
        )
        return cls(
            context=context,
            n_target_electrons=n_electrons(
                PSEUDOCUMENE_COMPOSITION, PSEUDOCUMENE_MOLAR_MASS_G_MOL, REFERENCE_TARGET_MASS_TON, context=context,
            ),
            e_nu_grid_MeV=torch.linspace(
                _E_NU_GRID_MIN_MEV, _E_NU_GRID_MAX_MEV, _E_NU_GRID_N, device=context.device, dtype=context.dtype,
            ),
            T_grid_MeV=T_grid_MeV,
            Tprime_grid_MeV=T_grid_MeV,
        )
