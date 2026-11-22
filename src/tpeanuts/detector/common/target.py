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

"""Target-particle counts derived from molecular composition and mass.

Unlike ``tpeanuts.detector.common.response``/``.event_rate`` (which
always have an already-placed sibling tensor to infer device/dtype from),
these two functions take only plain Python scalars, so they accept an
explicit ``context: RuntimeContext`` instead -- the same reasoning
``tpeanuts.medium.earth.earth.EarthMedium`` already applies to its own
geometry helpers.

Module contents:
    ATOMIC_NUMBER
        Atomic numbers used to count electrons in neutral matter.
    n_electrons(...)
        Calculate the number of electrons in a molecular target.
    n_atoms(...)
        Calculate the number of atoms of one element in a molecular target.
    reduced_target_count(...)
        Target count times ``SIGMA_UNIT_CM2``: the factor that pairs with
        cross sections expressed in units of ``SIGMA_UNIT_CM2``.
"""

from __future__ import annotations

from typing import Mapping

import torch

from tpeanuts.util import constant
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor

# Electrons per neutral atom (= atomic number Z), for the light elements
# that make up common liquid-scintillator/water targets. Textbook values,
# not detector-specific.
ATOMIC_NUMBER: Mapping[str, int] = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
}


def n_electrons(
    composition: Mapping[str, int],
    molar_mass_g_mol: float,
    mass_ton: float,
    *,
    context: RuntimeContext,
) -> torch.Tensor:
    """Total number of target electrons in ``mass_ton`` tons of a molecular target.

    ``N_e = (mass_ton * 1e6 g/ton / molar_mass_g_mol) * N_A * Z_molecule``,
    with ``Z_molecule = sum(ATOMIC_NUMBER[element] * count for element,
    count in composition.items())`` the total electron count of one
    neutral molecule.

    Args:
        composition: Element symbol -> atom count in one target molecule
            (e.g. ``{"C": 9, "H": 12}`` for pseudocumene, C9H12). Every
            key must be in ``ATOMIC_NUMBER``.
        molar_mass_g_mol: Molar mass of the target molecule, g/mol.
        mass_ton: Target mass, metric tons.
        context: RuntimeContext the returned tensor is built on.

    Returns:
        Scalar tensor, total number of target electrons (dimensionless
        count).

    Raises:
        ValueError: If ``composition`` contains an element not in
            ``ATOMIC_NUMBER``, or if ``molar_mass_g_mol``/``mass_ton`` is
            not positive.
    """
    if molar_mass_g_mol <= 0:
        raise ValueError(f"molar_mass_g_mol must be positive, got {molar_mass_g_mol}.")
    if mass_ton <= 0:
        raise ValueError(f"mass_ton must be positive, got {mass_ton}.")
    unknown = set(composition) - set(ATOMIC_NUMBER)
    if unknown:
        raise ValueError(f"composition contains elements not in ATOMIC_NUMBER: {sorted(unknown)}.")

    z_molecule = sum(ATOMIC_NUMBER[element] * count for element, count in composition.items())
    mass_g = mass_ton * 1.0e6
    n_molecules = (mass_g / molar_mass_g_mol) * constant.N_A

    return torch.tensor(n_molecules * z_molecule, device=context.device, dtype=context.dtype)


def n_atoms(
    composition: Mapping[str, int],
    molar_mass_g_mol: float,
    mass_ton: float,
    element: str,
    *,
    context: RuntimeContext,
) -> torch.Tensor:
    """Total number of one element's atoms in ``mass_ton`` tons of a molecular target.

    ``N = (mass_ton * 1e6 g/ton / molar_mass_g_mol) * N_A * composition[element]``.
    Unlike ``n_electrons`` (which sums ``Z`` over every element to count
    electrons), this isolates a single element's atom count -- the
    relevant quantity for a hydrogen (free-proton) inverse-beta-decay
    target, where each hydrogen nucleus, not each electron, is the
    reaction target.

    Args:
        composition: Element symbol -> atom count in one target molecule
            (e.g. ``{"C": 9, "H": 12}`` for pseudocumene, C9H12).
        molar_mass_g_mol: Molar mass of the target molecule, g/mol.
        mass_ton: Target mass, metric tons.
        element: Which element's atom count to return (must be a key of
            ``composition``).
        context: RuntimeContext the returned tensor is built on.

    Returns:
        Scalar tensor, total atom count of ``element`` (dimensionless
        count).

    Raises:
        ValueError: If ``element`` is not a key of ``composition``, or if
            ``molar_mass_g_mol``/``mass_ton`` is not positive.
    """
    if molar_mass_g_mol <= 0:
        raise ValueError(f"molar_mass_g_mol must be positive, got {molar_mass_g_mol}.")
    if mass_ton <= 0:
        raise ValueError(f"mass_ton must be positive, got {mass_ton}.")
    if element not in composition:
        raise ValueError(f"element {element!r} is not a key of composition {sorted(composition)}.")

    mass_g = mass_ton * 1.0e6
    n_molecules = (mass_g / molar_mass_g_mol) * constant.N_A

    return torch.tensor(n_molecules * composition[element], device=context.device, dtype=context.dtype)


def reduced_target_count(
    n_target: TensorLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Target count expressed for cross sections stored in units of ``SIGMA_UNIT_CM2``.

    Every ``tpeanuts.detector`` cross section is ``sigma / SIGMA_UNIT_CM2``
    (see ``tpeanuts.util.constant.SIGMA_UNIT_CM2``), so the physical product
    ``n_target * sigma`` is ``(n_target * SIGMA_UNIT_CM2) * sigma_reduced``.
    This returns the first factor, formed in float64 (``n_target`` ~1e30 and
    the unit 1e-42 are each fine there, and the product, ~1e-12, is a normal
    float32) and then cast to ``dtype``. Use it wherever a physical target
    count meets a detector cross section; ratios that cancel ``n_target``
    (e.g. SNO-II equivalent fluxes) simply pass ``1.0`` and need no unit.

    Args:
        n_target: Physical number of targets, scalar or batched.
        device: Device of the returned tensor.
        dtype: Real dtype of the returned tensor.

    Returns:
        ``n_target * SIGMA_UNIT_CM2`` as a ``dtype`` tensor on ``device``;
        differentiable w.r.t. ``n_target`` if it requires grad.
    """
    n = as_tensor(n_target, device=device, dtype=torch.float64)
    return (n * constant.SIGMA_UNIT_CM2).to(dtype)
