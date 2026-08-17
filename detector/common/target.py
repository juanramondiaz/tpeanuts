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
Number of target electrons/nucleons from material stoichiometry.

Derives the target particle count from chemistry (molecular composition,
molar mass, target mass) rather than quoting a detector's published target
count directly: a stoichiometric derivation is independently checkable
against textbook atomic numbers and molar masses, the same preference this
project already applies to ``medium.solar.io.load_solar_composition``'s
neutron-density derivation (composition-derived, not a hardcoded ratio).

Module contents:
    ATOMIC_NUMBER
        Electrons per neutral atom, for the light elements relevant to
        liquid-scintillator/water targets.
    n_electrons(...)
        Total target electron count from a molecular formula, molar mass,
        and target mass.
    n_atoms(...)
        Total count of one element's atoms in a molecular target -- e.g.
        hydrogen atoms, whose nuclei are the free protons
        ``tpeanuts.detector.dayabay`` folds inverse beta decay against.
"""

from __future__ import annotations

from typing import Mapping

import torch

import tpeanuts.util.constant as constant

# Electrons per neutral atom (= atomic number Z), for the light elements
# that make up common liquid-scintillator/water targets. Textbook values,
# not detector-specific.
ATOMIC_NUMBER: dict[str, int] = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
}


def n_electrons(
    composition: Mapping[str, int],
    molar_mass_g_mol: float,
    mass_ton: float,
) -> torch.Tensor:
    """Total number of target electrons in ``mass_ton`` tons of a molecular target.

    N_e = (mass_ton * 1e6 g/ton / molar_mass_g_mol) * N_A * Z_molecule,
    with ``Z_molecule = sum(ATOMIC_NUMBER[element] * count for element, count
    in composition.items())`` the total electron count of one neutral
    molecule.

    Args:
        composition: Element symbol -> atom count in one target molecule
            (e.g. ``{"C": 9, "H": 12}`` for pseudocumene, C9H12). Every key
            must be in ``ATOMIC_NUMBER``.
        molar_mass_g_mol: Molar mass of the target molecule, g/mol.
        mass_ton: Target mass, metric tons.

    Returns:
        Scalar tensor, total number of target electrons (dimensionless
        count, float64).

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
        raise ValueError(
            f"composition contains elements not in ATOMIC_NUMBER: {sorted(unknown)}."
        )

    z_molecule = sum(ATOMIC_NUMBER[element] * count for element, count in composition.items())
    mass_g = mass_ton * 1.0e6
    n_molecules = (mass_g / molar_mass_g_mol) * constant.N_A

    return torch.tensor(n_molecules * z_molecule, dtype=torch.float64)


def n_atoms(
    composition: Mapping[str, int],
    molar_mass_g_mol: float,
    mass_ton: float,
    element: str,
) -> torch.Tensor:
    """Total number of one element's atoms in ``mass_ton`` tons of a molecular target.

    N = (mass_ton * 1e6 g/ton / molar_mass_g_mol) * N_A * composition[element].
    Unlike ``n_electrons`` (which sums ``Z`` over every element to count
    electrons), this isolates a single element's atom count -- the relevant
    quantity for a hydrogen (free-proton) inverse-beta-decay target, where
    each hydrogen nucleus, not each electron, is the reaction target.

    Args:
        composition: Element symbol -> atom count in one target molecule
            (e.g. ``{"C": 9, "H": 12}`` for pseudocumene, C9H12).
        molar_mass_g_mol: Molar mass of the target molecule, g/mol.
        mass_ton: Target mass, metric tons.
        element: Which element's atom count to return (must be a key of
            ``composition``).

    Returns:
        Scalar tensor, total atom count of ``element`` (dimensionless count,
        float64).

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

    return torch.tensor(n_molecules * composition[element], dtype=torch.float64)
