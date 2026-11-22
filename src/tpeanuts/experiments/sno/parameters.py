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

"""``SNOParameters``: the loaded-once bundle of SNO heavy-water target/grid/response constants.

Ported from v1_0's ``experiments/sno/parameters.py``, which instead
loaded ``N_TARGET_DEUTERONS``/``N_TARGET_ELECTRONS``/``E_NU_GRID_MEV``/
``T_GRID_MEV``/``TPRIME_GRID_MEV`` as bare module-level constants at
import time, hardcoded to ``dtype=torch.float64`` with no ``device`` --
incompatible with this project's own ``RuntimeContext`` convention.
``SNOParameters.default(context=...)`` is the replacement, mirroring
``tpeanuts.experiments.borexino.parameters.BorexinoParameters.default``'s
own "call the loader(s) once, bundle the results" shape. Unlike
Borexino, the response-formula coefficients and NC capture Q-value/
efficiency are real (not illustrative) physical constants, but the
bundling choice is the same: plain module-level constants (fixed
values, not loaded data) *and* dataclass fields defaulting to them
(so a caller can override per-instance without touching the module
constants), matching ``BorexinoParameters.energy_resolution_a``'s own
precedent.

Module contents:
    D2O_COMPOSITION, D2O_MOLAR_MASS_G_MOL, TARGET_MASS_TON
        Heavy-water stoichiometry inputs.
    ENERGY_RESOLUTION_C0, ENERGY_RESOLUTION_C1, ENERGY_RESOLUTION_C2
        Real SNO Phase-I kinetic-energy resolution parametrization
        coefficients -- see ``experiments.sno.response`` for the formula.
    ENERGY_RESOLUTION_FLOOR_MEV
        Numerical floor on sigma(T), not a physical parameter -- see
        ``experiments.sno.response``.
    NC_CAPTURE_ENERGY_MEV
        Real Q-value of n + d -> t + gamma, SNO Phase-I's NC-channel
        capture-gamma energy.
    NC_CAPTURE_EFFICIENCY
        Phase-I neutron detection efficiency after the fiducial-volume
        and 5-MeV kinetic-energy cuts.
    SNOParameters
        Frozen dataclass bundling the target deuteron/electron counts,
        the true/reconstructed energy grids, and the response/capture
        constants above, with a ``.default(context=...)`` classmethod.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from tpeanuts.detector.common.target import n_electrons
from tpeanuts.util import constant
from tpeanuts.util.context import RuntimeContext

Tensor = torch.Tensor

# D2O (heavy water) molar mass, g/mol: 2 * deuterium (2.0141) + oxygen (15.9994).
D2O_MOLAR_MASS_G_MOL: float = 2.0 * 2.0141 + 15.9994

# Deuterium's nucleus (a single proton) has the same electron structure as
# ordinary hydrogen (Z=1); detector.common.target.ATOMIC_NUMBER counts
# electrons per neutral atom, which does not depend on the isotope, so "H"
# is reused rather than adding a "D" entry that would carry an identical value.
D2O_COMPOSITION: Dict[str, int] = {"H": 2, "O": 1}

# SNO's approximate heavy-water target mass (order-of-magnitude figure widely
# quoted for the experiment; not verified here against a primary SNO paper).
TARGET_MASS_TON: float = 1000.0

# Real SNO Phase-I kinetic-energy resolution parametrization,
# sigma_T(Te) = C0 + C1*sqrt(Te) + C2*Te [MeV, Te in MeV] -- the widely-quoted
# SNO Phase-I resolution function used in solar-neutrino oscillation-fit
# literature (see experiments.sno.response for the full formula and its
# numerical floor). Unlike experiments.borexino.parameters.ENERGY_RESOLUTION_A,
# this is real, not illustrative.
ENERGY_RESOLUTION_C0: float = -0.0684
ENERGY_RESOLUTION_C1: float = 0.331
ENERGY_RESOLUTION_C2: float = 0.0425

# Numerical floor on sigma(T), MeV -- the formula above goes non-positive
# for Te below ~0.04 MeV (well under SNO's actual analysis threshold), so
# this only guards T_GRID_MEV's low-T tail against feeding a non-positive
# sigma into gaussian_response_matrix; it carries no physical meaning.
ENERGY_RESOLUTION_FLOOR_MEV: float = 1.0e-3

# True neutrino energy grid, and true/reconstructed electron-energy grid
# (Tprime_grid_MeV == T_grid_MeV, see SNOParameters.default), spanning the
# day/night spectrum's electron-energy range (5-20 MeV).
_E_NU_GRID_MIN_MEV = 1.0
_E_NU_GRID_MAX_MEV = 20.0
_E_NU_GRID_N = 400

_T_GRID_MIN_MEV = 0.0
_T_GRID_MAX_MEV = 20.0
_T_GRID_N = 500

# Q-value of n + d -> t + gamma, MeV -- the real (well-established nuclear
# data, not illustrative) energy of the single gamma SNO Phase-I's pure-D2O
# NC channel is detected through (neutron thermalizes and is radiatively
# captured on a second deuteron; the gamma Compton-scatters an electron,
# detected the same way as CC/ES Cherenkov light).
NC_CAPTURE_ENERGY_MEV: float = 6.25

# SNO Phase-I neutron detection efficiency within the fiducial volume and
# above the 5-MeV kinetic-energy threshold: 14.4%. This is the effective
# analysis efficiency, not merely the larger probability (~29.9%) that a
# neutron captures on deuterium. SNO Collaboration, Phys. Rev. Lett. 89,
# 011301 (2002), DOI 10.1103/PhysRevLett.89.011301.
NC_CAPTURE_EFFICIENCY: float = 0.144


@dataclass(frozen=True)
class SNOParameters:
    """Every SNO heavy-water target/grid/response constant, bundled on one device/dtype.

    Attributes:
        context: RuntimeContext every tensor here is built on.
        n_target_deuterons: Target deuteron count at
            ``TARGET_MASS_TON`` tons of D2O (2 deuterons/molecule).
        n_target_electrons: Target electron count at the same mass,
            from D2O stoichiometry (10 electrons/molecule).
        e_nu_grid_MeV: True neutrino energy grid, shape ``(400,)``.
        T_grid_MeV: True electron-energy grid, shape ``(500,)``.
        Tprime_grid_MeV: Reconstructed electron-energy grid -- equal to
            ``T_grid_MeV`` (see ``detector.common.event_rate
            .predicted_counts``'s own same-grid requirement).
        energy_resolution_c0, energy_resolution_c1, energy_resolution_c2:
            Real resolution-formula coefficients, see
            ``ENERGY_RESOLUTION_C0``/``C1``/``C2``.
        energy_resolution_floor_MeV: See ``ENERGY_RESOLUTION_FLOOR_MEV``.
        nc_capture_energy_MeV: See ``NC_CAPTURE_ENERGY_MEV``.
        nc_capture_efficiency: See ``NC_CAPTURE_EFFICIENCY``.
    """

    context: RuntimeContext
    n_target_deuterons: Tensor
    n_target_electrons: Tensor
    e_nu_grid_MeV: Tensor
    T_grid_MeV: Tensor
    Tprime_grid_MeV: Tensor
    energy_resolution_c0: float = ENERGY_RESOLUTION_C0
    energy_resolution_c1: float = ENERGY_RESOLUTION_C1
    energy_resolution_c2: float = ENERGY_RESOLUTION_C2
    energy_resolution_floor_MeV: float = ENERGY_RESOLUTION_FLOOR_MEV
    nc_capture_energy_MeV: float = NC_CAPTURE_ENERGY_MEV
    nc_capture_efficiency: float = NC_CAPTURE_EFFICIENCY

    @classmethod
    def default(cls, *, context: RuntimeContext) -> "SNOParameters":
        """Build the default SNO parameter bundle on ``context``.

        Args:
            context: RuntimeContext every returned tensor is built on.

        Returns:
            SNOParameters with the 1000 t D2O reference target and the
            default 400/500-point energy grids.
        """
        n_d2o_molecules = (TARGET_MASS_TON * 1.0e6 / D2O_MOLAR_MASS_G_MOL) * constant.N_A
        T_grid_MeV = torch.linspace(
            _T_GRID_MIN_MEV, _T_GRID_MAX_MEV, _T_GRID_N, device=context.device, dtype=context.dtype,
        )
        return cls(
            context=context,
            n_target_deuterons=torch.tensor(
                2.0 * n_d2o_molecules, device=context.device, dtype=context.dtype,
            ),
            n_target_electrons=n_electrons(
                D2O_COMPOSITION, D2O_MOLAR_MASS_G_MOL, TARGET_MASS_TON, context=context,
            ),
            e_nu_grid_MeV=torch.linspace(
                _E_NU_GRID_MIN_MEV, _E_NU_GRID_MAX_MEV, _E_NU_GRID_N, device=context.device, dtype=context.dtype,
            ),
            T_grid_MeV=T_grid_MeV,
            Tprime_grid_MeV=T_grid_MeV,
        )
