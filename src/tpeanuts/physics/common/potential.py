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

"""Elementary matter and kinetic potentials.

The physical potentials are multiplied by a configurable evolution length
scale, so that the generator built from them is dimensionless when
integrated over the matching coordinate ``x = L / evolution_scale_m``. The
Earth radius is the default scale.

Units: densities in mol/cm^3, mass-squared in eV^2, energy in MeV,
evolution_scale_m in m.

Module contents:
    matter_potential_cc(...)
        Dimensionless charged-current matter potential from electron
        density.
    matter_potential_nc(...)
        Dimensionless neutral-current matter potential from neutron
        density. Only physically relevant once sterile flavours are
        present.
    kinetic_potential(...)
        Dimensionless kinetic term from a mass-squared value and energy.
    magnetic_moment_scale(...)
        Dimensionless scale factor from a magnetic-field magnitude, meant to
        multiply a transition-moment matrix expressed in units of the Bohr
        magneton. Only physically relevant once a magnetic-moment extension
        is active.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

import tpeanuts.util.constant as constant
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, antinu_sign, as_tensor, infer_device_dtype

# Precomputed physical constant: sqrt(2) * G_F * N_A * 1e6 * (hbar*c)^2.
# Units: (MeV * m^2), so that multiplied by n [mol/cm^3] and L_scale [m]
# yields a dimensionless potential. Computed once at import time to avoid
# per-call tensor allocation.
_MATTER_FACTOR_CC: float = (
    math.sqrt(2.0) * constant.G_F_MEV_M2 * constant.N_A * 1.0e6 * constant.HBARC_MeV_m**2
)

# Neutral-current prefactor: G_F * N_A * 1e6 * (hbar*c)^2 / sqrt(2). The NC
# potential is V_NC = -(1/sqrt(2)) G_F n_n, versus the CC potential's
# V_CC = +sqrt(2) G_F n_e -- the two prefactors differ by exactly a factor 2
# (ratio (1/sqrt(2))/sqrt(2) = 1/2), so this is derived directly from
# _MATTER_FACTOR_CC rather than recomputed from the elementary constants.
_MATTER_FACTOR_NC: float = _MATTER_FACTOR_CC / 2.0


def _positive_scale(evolution_scale_m: TensorLike, reference: torch.Tensor) -> torch.Tensor:
    """Validate and return ``evolution_scale_m`` as a tensor matching ``reference``.

    Args:
        evolution_scale_m: Positive length scale in metres.
        reference: Tensor whose device/dtype the returned scale is cast to.

    Returns:
        ``evolution_scale_m`` as a real tensor on ``reference``'s
        device/dtype.

    Raises:
        ValueError: If any entry of ``evolution_scale_m`` is not positive.
    """
    scale_dtype = reference.real.dtype if reference.is_complex() else reference.dtype
    scale = torch.as_tensor(evolution_scale_m, device=reference.device, dtype=scale_dtype)
    if torch.any(scale <= 0):
        raise ValueError("evolution_scale_m must be positive.")
    return scale


def matter_potential_cc(
    n_mol_cm3: TensorLike,
    antinu: Union[bool, torch.Tensor],
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Convert electron density into a dimensionless charged-current matter
      potential.

    The implemented normalization is:

       ``V_CC = s * evolution_scale_m * sqrt(2) * G_F * n_e``

    where ``s`` is +1 for neutrinos and -1 for antineutrinos.

    Args:
        n_mol_cm3: Electron density in mol/cm^3; scalar or tensor.
        antinu: Bool or boolean tensor; True selects the antineutrino sign
            (see ``tpeanuts.util.tensor.antinu_sign``).
        evolution_scale_m: Positive length scale in metres used to
            non-dimensionalize the generator. Defaults to the Earth radius.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        Dimensionless CC matter potential, with sign flipped for
        antineutrinos.
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(n_mol_cm3, evolution_scale_m)
    n_mol_cm3 = as_tensor(n_mol_cm3, device=device, dtype=dtype)

    sign = antinu_sign(antinu, n_mol_cm3)
    scale = _positive_scale(evolution_scale_m, n_mol_cm3)

    return sign * scale * _MATTER_FACTOR_CC * n_mol_cm3


def matter_potential_nc(
    n_n_mol_cm3: TensorLike,
    antinu: Union[bool, torch.Tensor],
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Convert neutron density into a dimensionless neutral-current matter
      potential.

    The implemented normalization is:

       ``V_NC = -s * evolution_scale_m * (1/sqrt(2)) * G_F * n_n``,

    where ``s`` is +1 for neutrinos and -1 for antineutrinos since
    both potentials come from the same coherent-forward-scattering
    mechanism.

    For neutral matter (``n_p == n_e``), the proton and electron
    NC contributions cancel, leaving only the neutron term.

    This potential is common to every standard flavour (lepton universality
    of the NC coupling), so it is a pure overall phase whenever every
    flavour in play is a standard (non-sterile) one.

    It becomes physically relevant only once sterile flavours are present: a
    sterile state is an SM gauge singlet and therefore does not receive this
    term, so once the common phase shared by the standard flavours is
    removed, a relative potential survives on the sterile diagonal entries.

    Args:
        n_n_mol_cm3: Neutron density in mol/cm^3; scalar or tensor.
        antinu: Bool or boolean tensor; True selects the antineutrino sign.
        evolution_scale_m: Positive length scale in metres used to
            non-dimensionalize the generator. Defaults to the Earth radius.
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

    sign = antinu_sign(antinu, n_n_mol_cm3)
    scale = _positive_scale(evolution_scale_m, n_n_mol_cm3)

    return -sign * scale * _MATTER_FACTOR_NC * n_n_mol_cm3


def kinetic_potential(
    mSq_eV2: TensorLike,
    E_MeV: TensorLike,
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Convert mass-squared values and energy into dimensionless kinetic terms.

    Uses ``k = evolution_scale_m * m^2 / (2 * E * hbar * c)``, with ``E`` in
    MeV and ``m^2`` in eV^2.

    Args:
        mSq_eV2: Mass-squared value or vector in eV^2.
        E_MeV: Neutrino energy in MeV; scalar or tensor broadcastable with
            ``mSq_eV2`` (a trailing dimension is inserted automatically when
            ``mSq_eV2`` carries an extra flavour axis).
        evolution_scale_m: Positive length scale in metres used to
            non-dimensionalize the generator. Defaults to the Earth radius. A tensor batches like ``E_MeV``
            (one value per scenario, e.g. a column of baselines against an energy grid), not along the
            flavour axis of ``mSq_eV2``.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        Dimensionless kinetic phase tensor.
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(mSq_eV2, E_MeV, evolution_scale_m)
    mSq_eV2 = as_tensor(mSq_eV2, device=device, dtype=dtype)
    E = as_tensor(E_MeV, device=device, dtype=dtype)

    E_unsqueezed = E if E.ndim == 0 else E.unsqueeze(-1)
    scale = _positive_scale(evolution_scale_m, mSq_eV2)
    if scale.ndim > 0:
        scale = scale.unsqueeze(-1)  # the scale batches like E: one value per scenario, a trailing flavour axis
    while scale.ndim < mSq_eV2.ndim:
        scale = scale.unsqueeze(-1)

    return scale * 0.5 * 1e-12 * mSq_eV2 / E_unsqueezed / constant.HBARC_MeV_m


def magnetic_moment_scale(
    B_field_T: TensorLike,
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Convert a magnetic-field magnitude into a dimensionless magnetic-moment scale factor.

    The implemented normalization is:

       ``s = evolution_scale_m * mu_B * B``

    where ``mu_B`` is the Bohr magneton
    (``tpeanuts.util.constant.MU_B_MEV_PER_T``). A transition moment is thus
    expressed as a dimensionless multiple of ``mu_B`` (matching the
    convention used throughout the neutrino magnetic-moment literature,
    e.g. ``mu_nu < 2.9e-11 mu_B``): the returned scale multiplies that
    dimensionless matrix to give the term's own contribution to the
    dimensionless generator, exactly as ``matter_potential_cc``'s ``V_CC``
    multiplies ``diag(1, 0, ..., 0)``.

    Args:
        B_field_T: Magnetic field magnitude in tesla; scalar or tensor.
        evolution_scale_m: Positive length scale in metres used to
            non-dimensionalize the generator. Defaults to the Earth radius.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        Dimensionless real scale factor, broadcastable against a transition
        moment matrix.
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(B_field_T, evolution_scale_m)
    B_field_T = as_tensor(B_field_T, device=device, dtype=dtype)

    scale = _positive_scale(evolution_scale_m, B_field_T)
    return scale * constant.MU_B_MEV_PER_T * B_field_T / constant.HBARC_MeV_m
