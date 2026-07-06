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

# peanuts_torch/potentials.py
# -*- coding: utf-8 -*-
"""
Potentials (matter + kinetic) in pure PyTorch (GPU-first).

The physical potentials are multiplied by a configurable evolution length
scale. This makes the Hamiltonian dimensionless when it is integrated over
the matching coordinate:

    x = L / evolution_scale_m. 
    
The Earth radius remains the default scale for backwards compatibility.

Units:
- n: mol / cm^3
- mSq: eV^2
- E: MeV
- evolution_scale_m: m

Module functions:
    
    matter_potential(...)
        Computes the dimensionless charged-current matter potential from
        electron density, the neutrino/antineutrino sign, and a configurable
        evolution scale. Tensor conversion and device/dtype inference are
        handled internally.
    
    kinetic_potential(...)
        Computes the dimensionless kinetic term
        evolution_scale_m Delta m^2/(2E hbar c), accepting scalar or tensor
        inputs and normalizing them internally.
"""



from __future__ import annotations
from typing import Optional, Union
import math

import torch
import tpeanuts.util.constant as constant
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor
from tpeanuts.util.torch_util import infer_device_dtype

# ---------------------------------------------------------------------------
# Precomputed physical constant: sqrt(2) * G_F * N_A * 1e6 * (hbar*c)^2
# Units: (MeV * m^2) so that multiplied by n [mol/cm^3] and L_scale [m]
# yields a dimensionless potential.
# Computed once at import time to avoid per-call tensor allocation.
# ---------------------------------------------------------------------------
_MATTER_FACTOR: float = (
    math.sqrt(2.0)
    * constant.G_F_MEV_M2
    * constant.N_A
    * 1.0e6
    * constant.HBARC_MeV_m ** 2
)

# ---------------------------------------------------------------------------
# Legacy peanuts precomputed matter-potential prefactor (peanuts/potentials.py,
# Eq. 4.17 in 1802.05781). Hardcoded there with 4 significant digits instead of
# being derived from full-precision G_F, N_A, and hbar*c. 
# 
# Differs from _MATTER_FACTOR by a relative ~1.85e-5. 
# Kept only for bit-comparable validation against the legacy implementation.
# ---------------------------------------------------------------------------
_MATTER_FACTOR_LEGACY: float = 3.868e-7


@torch.no_grad()
def matter_potential(
    n_mol_cm3: TensorLike,
    antinu: Union[bool, torch.Tensor],
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Convert electron density into a dimensionless matter potential.

    The implemented normalization is

        V = s evolution_scale_m sqrt(2) G_F n_e,

    where s is +1 for neutrinos and -1 for antineutrinos. The result must be
    combined with coordinates normalized by the same evolution scale.

    Args:
        n_mol_cm3: Electron density in mol/cm^3; scalar or tensor.
        antinu: Bool or boolean tensor; True selects the antineutrino sign.
        evolution_scale_m: Positive length scale in metres used to
            nondimensionalize the Hamiltonian. Defaults to the Earth radius.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.
        legacy_precision: If True, use the legacy peanuts prefactor
            ``_MATTER_FACTOR_LEGACY`` (hardcoded to 4 significant digits in
            ``peanuts/potentials.py``) instead of the full-precision
            ``_MATTER_FACTOR`` derived from G_F, N_A, and hbar*c. Intended
            only for bit-comparable validation against legacy peanuts; the
            two prefactors differ by a relative ~1.85e-5.

    Returns:
        Dimensionless matter potential, with sign flipped for antineutrinos.
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(n_mol_cm3, evolution_scale_m)
    n_mol_cm3 = as_tensor(n_mol_cm3, device=device, dtype=dtype)

    if isinstance(antinu, bool):
        sign = -1.0 if antinu else 1.0
    else:
        antinu = antinu.to(device=n_mol_cm3.device, dtype=torch.bool)
        while antinu.ndim < n_mol_cm3.ndim:
            antinu = antinu.unsqueeze(-1)
        sign = torch.where(
            antinu,
            torch.full_like(n_mol_cm3, -1.0),
            torch.ones_like(n_mol_cm3),
        )

    scale_dtype = n_mol_cm3.real.dtype if n_mol_cm3.is_complex() else n_mol_cm3.dtype
    scale = torch.as_tensor(
        evolution_scale_m,
        device=n_mol_cm3.device,
        dtype=scale_dtype,
    )
    # Validate only when scale is a non-trivial tensor (skip for fixed scalars
    # such as the default R_E to avoid per-call overhead in hot loops).
    if scale.ndim > 0 or bool(scale <= 0):
        if torch.any(scale <= 0):
            raise ValueError("evolution_scale_m must be positive.")

    factor = _MATTER_FACTOR_LEGACY if legacy_precision else _MATTER_FACTOR

    return (
        sign
        * scale
        * factor
        * n_mol_cm3
    )


@torch.no_grad()
def kinetic_potential(
    mSq_eV2: TensorLike,
    E_MeV: TensorLike,
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Convert mass-squared values and energy into dimensionless kinetic terms.

    Uses k = evolution_scale_m m^2 / (2 E hbar c), with E in MeV and
    m^2 in eV^2.

    Args:
        mSq_eV2: Mass-squared value or vector in eV^2.
        E_MeV: Neutrino energy in MeV; scalar or tensor broadcastable with other inputs.
        evolution_scale_m: Positive length scale in metres used to
            nondimensionalize the Hamiltonian. Defaults to the Earth radius.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.
        legacy_precision: Accepted for API consistency with
            ``matter_potential``. The kinetic prefactor has no legacy-rounded
            alternative, so this flag does not change the calculation.

    Returns:
        Dimensionless kinetic phase tensor.
    """
    _ = legacy_precision
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(mSq_eV2, E_MeV, evolution_scale_m)
    mSq_eV2 = as_tensor(mSq_eV2, device=device, dtype=dtype)
    E = as_tensor(E_MeV, device=device, dtype=dtype)

    # If E is scalar -> ok
    if E.ndim == 0:
        E_unsqueeze = E
    else:
        # If mSq is (3,) and E is (Ne,), make denom (Ne,1) so division broadcasts to (Ne,3)
        # If mSq is (...,3) and E is (...,), make denom (...,1)
        E_unsqueeze = E.unsqueeze(-1)

    scale = torch.as_tensor(
        evolution_scale_m,
        device=mSq_eV2.device,
        dtype=mSq_eV2.dtype,
    )
    # Validate only when scale is a non-trivial tensor (skip for fixed scalars
    # such as the default R_E to avoid per-call overhead in hot loops).
    if scale.ndim > 0 or bool(scale <= 0):
        if torch.any(scale <= 0):
            raise ValueError("evolution_scale_m must be positive.")

    while scale.ndim < mSq_eV2.ndim:
        scale = scale.unsqueeze(-1)

    return scale * 0.5 * 1e-12 * mSq_eV2 / E_unsqueeze / constant.HBARC_MeV_m

