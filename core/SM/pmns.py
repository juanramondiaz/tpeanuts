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
STANDARD MODEL (SM) 3-flavour PMNS mixing matrix.

``PMNS_SM`` is the concrete 3-flavour implementation of the abstract base class

    ``tpeanuts.core.common.pmns.PMNS``

The shared rotation/phase builders 

    ``R12``, ``R13``, ``R23``, ``Delta``

 and the flavour-count-agnostic public interface 
 ``H_flavour_basis``, ``H_mass_basis``, ``refresh``, 
 ``select_antinu``, ``dagger``, ...

 are inherited unchanged from the base;

this module only supplies the STANDARD MODEL product structure for the full
and reduced mixing matrices, and the corresponding basis transformation.

Module contents:
----------------
    PMNS_SM
        Builds the full PMNS matrix U = R23 @ Delta @ R13 @ Delta.conj() @ R12
        and the reduced matrix U_red = R13 @ R12.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from tpeanuts.core.common.pmns import PMNS, PMNSParams


class PMNS_SM(PMNS):
    """
    Standard Model 3-flavour PMNS mixing matrix generator in pure PyTorch.

    Builds both the full PMNS matrix and the reduced peanuts mixing matrix
    for the 3 active flavours. All shared machinery (rotation/phase
    builders, basis transformations, caching) is inherited from

        ``tpeanuts.core.common.pmns.PMNS``
        
    this class only supplies the Standard Model product structure.

        The full Standard Model PMNS matrix is constructed using the peanuts
    convention:

        U_PMNS = R23 @ Delta @ R13 @ Delta.conj() @ R12

    where

        Delta = diag(1, 1, exp(i delta)).

    The reduced mixing matrix frequently used in peanuts is

        U_red = R13 @ R12.


    Parameters
    ----------
    params:
        PMNSParams bundling the solar (theta12), reactor (theta13), and
        atmospheric (theta23) mixing angles and the CP-violating phase
        (delta), all in radians, plus a RuntimeContext.

    Examples
    --------
    pmns = PMNS_SM(
        PMNSParams(
            theta12,
            theta13,
            theta23,
            delta,
            context=RuntimeContext.resolve("cuda", torch.float64),
        )
    )

    U_full = pmns.pmns_matrix()
    U_red = pmns.reduced()

    The reduced matrix is also available as:

    U_red = pmns.U
    """

    n_flavours: int = 3
    n_active: int = 3
    n_sterile: int = 0

    @torch.no_grad()
    def reduced(
        self,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """
        Build the reduced neutrino or antineutrino mixing matrix.

        The neutrino matrix is U_red = R13 R12. For antineutrinos, the
        complex conjugate U_red* is selected. A boolean tensor may select the
        convention independently over a broadcast batch.

        Args:
            antinu: Boolean scalar or tensor mask. True selects U_red*.

        Returns:
            Complex reduced mixing matrix shaped (..., 3, 3), conjugated for
            entries selected by ``antinu``.
        """
        r13 = self.R13()
        r12 = self.R12()

        Ured = r13 @ r12

        return self.select_antinu(Ured, antinu)

    @torch.no_grad()
    def pmns_matrix(
        self,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """
        Build the full neutrino or antineutrino PMNS matrix.

        Formula: Uses U = R23 Delta R13 Delta^dagger R12.

        Args:
            antinu: Boolean scalar or tensor mask. True selects U*.

        Returns:
            Complex full PMNS matrix shaped (..., 3, 3).
        """
        r23 = self.R23()
        r13 = self.R13()
        r12 = self.R12()
        delt = self.Delta()

        U = r23 @ delt @ r13 @ delt.conj() @ r12

        return self.select_antinu(U, antinu)

    @torch.no_grad()
    def operator_flavour_basis(
        self,
        operator_reduced: torch.Tensor,
        antinu: Union[bool, torch.Tensor] = False,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Transform an operator from the reduced to the flavour basis.

        Applies O_flavour = R23 Delta O_reduced Delta^dagger R23^T using the
        neutrino or antineutrino convention over broadcast batches.

        Args:
            operator_reduced: Reduced-basis operator shaped (..., 3, 3).
            antinu: Boolean scalar or tensor mask selecting antineutrinos.
            device: Optional output device; defaults to the operator device.
            dtype: Optional output dtype; defaults to the operator dtype.

        Returns:
            Operator represented in the full flavour basis.
        """
        output_device = (
            operator_reduced.device if device is None else torch.device(device)
        )
        output_dtype = operator_reduced.dtype if dtype is None else dtype
        operator_reduced = operator_reduced.to(
            device=output_device,
            dtype=output_dtype,
        )

        r23 = self.select_antinu(self.R23(), antinu).to(
            device=output_device,
            dtype=output_dtype,
        )
        delta = self.select_antinu(self.Delta(), antinu).to(
            device=output_device,
            dtype=output_dtype,
        )

        return (
            r23
            @ delta
            @ operator_reduced
            @ torch.conj(delta).transpose(-1, -2)
            @ r23.transpose(-1, -2)
        )
