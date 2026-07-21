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
Elementary matter and kinetic potentials in pure PyTorch (GPU-first).

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

    matter_potential_cc(...)
        Computes the dimensionless charged-current matter potential from
        electron density, the neutrino/antineutrino sign, and a configurable
        evolution scale. Tensor conversion and device/dtype inference are
        handled internally.

    matter_potential_nc(...)
        Computes the dimensionless neutral-current matter potential from
        neutron density, the neutrino/antineutrino sign, and a configurable
        evolution scale. Only physically relevant for the 3+1 sterile
        extension (see ``core.common.hamiltonian.hamiltonian_matter_reduced``):
        in the 3-flavour Standard Model it is a pure common phase across the
        three active flavours and has no effect on oscillation probabilities.

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

# ---------------------------------------------------------------------------
# Neutral-current prefactor: G_F * N_A * 1e6 * (hbar*c)^2 / sqrt(2).
# The NC potential is V_NC = -(1/sqrt(2)) G_F n_n, versus the CC potential's
# V_CC = +sqrt(2) G_F n_e -- the two prefactors differ by exactly a factor 2
# (ratio (1/sqrt(2))/sqrt(2) = 1/2), so this is derived directly from
# _MATTER_FACTOR rather than recomputed from the elementary constants.
# There is no legacy-peanuts counterpart (the original code never
# implemented NC or the sterile extension), so unlike matter_potential_cc,
# matter_potential_nc has no ``legacy_precision`` option.
# ---------------------------------------------------------------------------
_MATTER_FACTOR_NC: float = _MATTER_FACTOR / 2.0


@torch.no_grad()
def matter_potential_cc(
    n_mol_cm3: TensorLike,
    antinu: Union[bool, torch.Tensor],
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Convert electron density into a dimensionless charged-current matter potential.

    The implemented normalization is

        V_CC = s evolution_scale_m sqrt(2) G_F n_e,

    where s is +1 for neutrinos and -1 for antineutrinos. The result must be
    combined with coordinates normalized by the same evolution scale.

    This is the only matter potential every 3-flavour Standard Model
    calculation in this project needs: the neutral-current potential common
    to all three active flavours is a pure phase there and is intentionally
    never added (see ``matter_potential_nc``). It only becomes physically
    relevant once a fourth, NC-blind sterile state is present.

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
        Dimensionless CC matter potential, with sign flipped for antineutrinos.
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(n_mol_cm3, evolution_scale_m)
    n_mol_cm3 = as_tensor(n_mol_cm3, device=device, dtype=dtype)

    sign = _antinu_sign(antinu, n_mol_cm3)
    scale = _positive_scale(evolution_scale_m, n_mol_cm3)
    factor = _MATTER_FACTOR_LEGACY if legacy_precision else _MATTER_FACTOR

    return (
        sign
        * scale
        * factor
        * n_mol_cm3
    )


@torch.no_grad()
def matter_potential_nc(
    n_n_mol_cm3: TensorLike,
    antinu: Union[bool, torch.Tensor],
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """
    Convert neutron density into a dimensionless neutral-current matter potential.

    The implemented normalization is

        V_NC = -s evolution_scale_m (1/sqrt(2)) G_F n_n,

    where s is +1 for neutrinos and -1 for antineutrinos -- the same sign
    flip as ``matter_potential_cc``, since both potentials come from the same
    coherent-forward-scattering mechanism. For neutral matter (n_p = n_e),
    the proton and electron NC contributions cancel, leaving only the
    neutron term.

    This potential is common to all three active flavours (lepton
    universality of the NC coupling) and is therefore a pure overall phase
    in the 3-flavour Standard Model -- it never needs to be added there, and
    ``core.common.hamiltonian.hamiltonian_matter_reduced`` never calls this
    function for a 3-flavour ``pmns``. It becomes physically relevant only
    for the 3+1 sterile extension, where the sterile state is an SM gauge
    singlet and therefore does not receive this term: once the common phase
    is removed, a genuine relative potential ``-V_NC`` remains on the
    sterile diagonal entry (see ``hamiltonian_matter_reduced``).

    Args:
        n_n_mol_cm3: Neutron density in mol/cm^3; scalar or tensor.
        antinu: Bool or boolean tensor; True selects the antineutrino sign.
        evolution_scale_m: Positive length scale in metres used to
            nondimensionalize the Hamiltonian. Defaults to the Earth radius.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        Dimensionless NC matter potential (negative for neutrinos, positive
        neutron density), with sign flipped for antineutrinos.
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(n_n_mol_cm3, evolution_scale_m)
    n_n_mol_cm3 = as_tensor(n_n_mol_cm3, device=device, dtype=dtype)

    sign = _antinu_sign(antinu, n_n_mol_cm3)
    scale = _positive_scale(evolution_scale_m, n_n_mol_cm3)

    return (
        -sign
        * scale
        * _MATTER_FACTOR_NC
        * n_n_mol_cm3
    )


def _antinu_sign(
    antinu: Union[bool, torch.Tensor],
    reference: torch.Tensor,
) -> Union[float, torch.Tensor]:
    """Return +1 for neutrinos and -1 for antineutrinos, broadcast to ``reference``."""
    if isinstance(antinu, bool):
        return -1.0 if antinu else 1.0

    antinu = antinu.to(device=reference.device, dtype=torch.bool)
    while antinu.ndim < reference.ndim:
        antinu = antinu.unsqueeze(-1)
    return torch.where(
        antinu,
        torch.full_like(reference, -1.0),
        torch.ones_like(reference),
    )


def _positive_scale(
    evolution_scale_m: TensorLike,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Validate and return ``evolution_scale_m`` as a tensor matching ``reference``."""
    scale_dtype = reference.real.dtype if reference.is_complex() else reference.dtype
    scale = torch.as_tensor(
        evolution_scale_m,
        device=reference.device,
        dtype=scale_dtype,
    )
    # Validate only when scale is a non-trivial tensor (skip for fixed scalars
    # such as the default R_E to avoid per-call overhead in hot loops).
    if scale.ndim > 0 or bool(scale <= 0):
        if torch.any(scale <= 0):
            raise ValueError("evolution_scale_m must be positive.")
    return scale


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
            ``matter_potential_cc``. The kinetic prefactor has no
            legacy-rounded alternative, so this flag does not change the
            calculation.

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

