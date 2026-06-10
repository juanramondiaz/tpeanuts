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

This ports peanuts potentials.py exactly:
- earth radius scaling R_E makes the Hamiltonian dimensionless when integrating over r = R/R_E.
- MatterPotential(n, antinu) returns dimensionless V_mat.
- k(mSq, E) returns dimensionless kinetic potential.

Units:
- n: mol / cm^3
- mSq: eV^2
- E: MeV

Module functions:
    
    matter_potential(...)
        Computes the dimensionless charged-current matter potential from
        electron density n_e in mol/cm^3 and the neutrino/antineutrino sign.
    
    kinetic_potential(...)
        Computes the dimensionless kinetic term R_E Delta m^2/(2E hbar c) for
        torch mass-splitting and energy tensors.
"""



from __future__ import annotations
from typing import Union

import torch
import tpeanuts.util.constant as constant


@torch.no_grad()
def matter_potential(
    n_mol_cm3: torch.Tensor,
    antinu: Union[bool, torch.Tensor],
) -> torch.Tensor:
    """
    Convert electron density into the charged-current matter potential in km^-1.
    
    Formula: Uses V = sqrt(2) G_F n_e with the project unit conversion to km^-1.
    
    Args:
        n_mol_cm3: Electron density in mol/cm^3; scalar or tensor.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
    
    Returns:
        Matter potential V in km^-1, with sign flipped for antineutrinos.
    """
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

    return (sign * constant.R_E * 3.868e-7) * n_mol_cm3


@torch.no_grad()
def kinetic_potential(mSq_eV2: torch.Tensor, E_MeV: torch.Tensor) -> torch.Tensor:
    """
    Convert mass-squared values and neutrino energy into kinetic phases in km^-1.
    
    Formula: Uses k = m^2 / (2 E) with E provided in MeV and m^2 in eV^2.
    
    Args:
        mSq_eV2: Mass-squared value or vector in eV^2.
        E_MeV: Neutrino energy in MeV; scalar or tensor broadcastable with other inputs.
    
    Returns:
        Kinetic phase tensor m^2/(2E) in km^-1.
    """
    if not torch.is_tensor(mSq_eV2):
        raise TypeError("mSq_eV2 must be a torch.Tensor")
    if not torch.is_tensor(E_MeV):
        raise TypeError("E_MeV must be a torch.Tensor")

    # Ensure same device/dtype
    E = E_MeV.to(device=mSq_eV2.device, dtype=mSq_eV2.dtype)

    # If E is scalar -> ok
    if E.ndim == 0:
        E_unsqueeze = E
    else:
        # If mSq is (3,) and E is (Ne,), make denom (Ne,1) so division broadcasts to (Ne,3)
        # If mSq is (...,3) and E is (...,), make denom (...,1)
        E_unsqueeze = E.unsqueeze(-1)

    return constant.R_E * 0.5 * 1e-12 * mSq_eV2 / E_unsqueeze / constant.HBARC_MeV_m

