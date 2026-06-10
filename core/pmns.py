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
PMNS mixing matrix utilities for peanuts-torch.

This module defines a PyTorch implementation of the PMNS mixing matrix used in
neutrino oscillation calculations.

The implementation is GPU-compatible, fully based on torch, and supports both
scalar and batched oscillation parameters.

Module functions:

    PMNSParams
        Stores detached PMNS parameter tensors and exposes device/dtype
        convenience properties.

    PMNS
        Builds R12, R13, R23, CP phase, reduced, and full PMNS matrices as
        torch tensors.

The full PMNS matrix is constructed using the peanuts convention:

    U_PMNS = R23 @ Delta @ R13 @ Delta.conj() @ R12

where

    Delta = diag(1, 1, exp(i delta)).

The reduced mixing matrix frequently used in peanuts is

    U_red = R13 @ R12.

This reduced matrix is stored as

    pmns.U

so that other modules can directly use it when building the reduced
Hamiltonian:

    H = U_red diag(k_i) U_red^T + diag(V, 0, 0).

The functions and classes are organized as follows:

    PMNSParams
        Immutable container storing the mixing angles and CP phase.

    PMNS
        Torch module that builds R12, R13, R23, Delta, the full PMNS matrix,
        and the reduced peanuts matrix.

Main attributes after initialization:

    pmns.pmns
        Full PMNS matrix.

    pmns.U
        Reduced matrix U_red = R13 @ R12.

This module does not compute Hamiltonians, evolutors, probabilities, or matter
effects. It only provides mixing matrices.

"""



from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from tpeanuts.util.type import (
    _cdtype_from_real,
    _as_tensor,
)
from tpeanuts.util.torch_util import _default_device

TensorLike = Union[float, int, torch.Tensor]


@dataclass(frozen=True)
class PMNSParams:
    """
    Immutable container for PMNS mixing parameters.

    Parameters
    ----------
    theta12:
        solar mixing angle theta_12 in radians.

    theta13:
        Reactor mixing angle theta_13 in radians.

    theta23:
        Atmospheric mixing angle theta_23 in radians.

    delta:
        CP-violating phase delta in radians.

    Notes
    -----
    All parameters are stored as real torch tensors.
    """

    theta12: torch.Tensor
    theta13: torch.Tensor
    theta23: torch.Tensor
    delta: torch.Tensor

    @property
    def device(self) -> torch.device:
        """
        Return the torch device that stores the PMNS parameter tensors.
        
        Args:
            None.
        
        Returns:
            torch.device associated with the stored PMNS parameters.
        """
        return self.theta12.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Return the real torch dtype used by the PMNS parameter tensors.
        
        Args:
            None.
        
        Returns:
            torch.dtype associated with the stored real PMNS parameters.
        """
        return self.theta12.dtype


class PMNS(torch.nn.Module):
    """
    PMNS mixing matrix generator in pure PyTorch.

    This class builds both the full PMNS matrix and the reduced peanuts mixing
    matrix.

    Parameters
    ----------
    theta12:
        solar mixing angle theta_12 in radians.

    theta13:
        Reactor mixing angle theta_13 in radians.

    theta23:
        Atmospheric mixing angle theta_23 in radians.

    delta:
        CP-violating phase delta in radians.

    device:
        Torch device where the matrices will be allocated.
        If None, CUDA is used when available.

    real_dtype:
        Real torch dtype used for the input parameters.
        The corresponding complex dtype is inferred automatically.

    Examples
    --------
    pmns = PMNS(
        theta12,
        theta13,
        theta23,
        delta,
        device="cuda",
        real_dtype=torch.float64,
    )

    U_full = pmns.pmns_matrix()
    U_red = pmns.reduced()

    The reduced matrix is also available as:

    U_red = pmns.U
    """

    def __init__(
        self,
        theta12: TensorLike,
        theta13: TensorLike,
        theta23: TensorLike,
        delta: TensorLike,
        *,
        device: Optional[torch.device | str] = None,
        real_dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Initialize PMNS angles, CP phase, full matrix, and reduced matrix as torch tensors.
        
        Args:
            theta12: Solar mixing angle theta12 in radians.
            theta13: Reactor mixing angle theta13 in radians.
            theta23: Atmospheric mixing angle theta23 in radians.
            delta: CP-violating phase delta in radians.
            device: Optional torch device for newly created tensors.
            real_dtype: Real floating dtype used to store the PMNS parameters.
        
        Returns:
            None.
        """
        super().__init__()

        device = _default_device(device)

        th12 = _as_tensor(theta12, device=device, dtype=real_dtype)
        th13 = _as_tensor(theta13, device=device, dtype=real_dtype)
        th23 = _as_tensor(theta23, device=device, dtype=real_dtype)
        delt = _as_tensor(delta, device=device, dtype=real_dtype)

        # Store oscillation parameters as buffers.
        # They are not trainable by default.
        # If parameter fitting is needed, these buffers can be replaced by
        # torch.nn.Parameter objects.
        self.register_buffer("theta12", th12)
        self.register_buffer("theta13", th13)
        self.register_buffer("theta23", th23)
        self.register_buffer("delta", delt)

        self.register_buffer("pmns", self.pmns_matrix())
        self.register_buffer("U", self.reduced())

    def params(self) -> PMNSParams:
        """
        Pack the current PMNS angles and CP phase into a PMNSParams dataclass.
        
        Args:
            None.
        
        Returns:
            PMNSParams containing theta12, theta13, theta23, and delta tensors.
        """
        return PMNSParams(
            self.theta12,
            self.theta13,
            self.theta23,
            self.delta,
        )

    def R12(self) -> torch.Tensor:
        """
        Build the real rotation matrix in the 1-2 plane.
        
        Args:
            None.
        
        Returns:
            Complex 3x3 or batched rotation matrix.
        """
        p = self.params()

        c = torch.cos(p.theta12)
        s = torch.sin(p.theta12)

        cdtype = _cdtype_from_real(p.dtype)

        out = torch.zeros(
            (*c.shape, 3, 3),
            device=p.device,
            dtype=cdtype,
        )

        out[..., 0, 0] = c
        out[..., 0, 1] = s
        out[..., 1, 0] = -s
        out[..., 1, 1] = c
        out[..., 2, 2] = 1.0

        return out

    def R13(self) -> torch.Tensor:
        """
        Build the real rotation matrix in the 1-3 plane.
        
        Args:
            None.
        
        Returns:
            Complex 3x3 or batched rotation matrix.
        """
        p = self.params()

        c = torch.cos(p.theta13)
        s = torch.sin(p.theta13)

        cdtype = _cdtype_from_real(p.dtype)

        out = torch.zeros(
            (*c.shape, 3, 3),
            device=p.device,
            dtype=cdtype,
        )

        out[..., 0, 0] = c
        out[..., 0, 2] = s
        out[..., 1, 1] = 1.0
        out[..., 2, 0] = -s
        out[..., 2, 2] = c

        return out

    def R23(self) -> torch.Tensor:
        """
        Build the real rotation matrix in the 2-3 plane.
        
        Args:
            None.
        
        Returns:
            Complex 3x3 or batched rotation matrix.
        """
        p = self.params()

        c = torch.cos(p.theta23)
        s = torch.sin(p.theta23)

        cdtype = _cdtype_from_real(p.dtype)

        out = torch.zeros(
            (*c.shape, 3, 3),
            device=p.device,
            dtype=cdtype,
        )

        out[..., 0, 0] = 1.0
        out[..., 1, 1] = c
        out[..., 1, 2] = s
        out[..., 2, 1] = -s
        out[..., 2, 2] = c

        return out

    def Delta(self) -> torch.Tensor:
        """
        Build the diagonal CP-phase matrix Delta=diag(1, 1, exp(i delta)).
        
        Formula: Uses Delta = diag(1, 1, exp(i delta)).
        
        Args:
            None.
        
        Returns:
            Complex diagonal CP-phase matrix shaped (..., 3, 3).
        """
        p = self.params()

        cdtype = _cdtype_from_real(p.dtype)

        phase = torch.exp(
            1j * p.delta.to(dtype=cdtype)
        )

        out = torch.zeros(
            (*phase.shape, 3, 3),
            device=p.device,
            dtype=cdtype,
        )

        out[..., 0, 0] = 1.0
        out[..., 1, 1] = 1.0
        out[..., 2, 2] = phase

        return out

    @torch.no_grad()
    def reduced(self) -> torch.Tensor:
        """
        Build the reduced mixing matrix U_red = R13 R12.
        
        Formula: Uses U_red = R13 R12.
        
        Args:
            None.
        
        Returns:
            Complex reduced mixing matrix U_red shaped (..., 3, 3).
        """
        r13 = self.R13()
        r12 = self.R12()

        return r13 @ r12

    @torch.no_grad()
    def pmns_matrix(self) -> torch.Tensor:
        """
        Build the full PMNS matrix U = R23 Delta R13 Delta^dagger R12.
        
        Formula: Uses U = R23 Delta R13 Delta^dagger R12.
        
        Args:
            None.
        
        Returns:
            Complex full PMNS matrix shaped (..., 3, 3).
        """
        r23 = self.R23()
        r13 = self.R13()
        r12 = self.R12()
        delt = self.Delta()

        return r23 @ delt @ r13 @ delt.conj() @ r12

    @torch.no_grad()
    def refresh(self) -> None:
        """
        Recompute cached full and reduced PMNS matrices after parameter changes.
        
        Args:
            None.
        
        Returns:
            None; cached attributes self.pmns and self.U are updated in place.
        """
        self.pmns = self.pmns_matrix()
        self.U = self.reduced()

    @torch.no_grad()
    def __getitem__(
        self,
        idx,
    ) -> torch.Tensor:
        """
        Index the cached full PMNS matrix.
        
        Args:
            idx: Index, slice, or tuple applied to the cached PMNS matrix.
        
        Returns:
            Selected entry, slice, or batch of the cached full PMNS matrix.
        """
        return self.pmns_matrix()[idx]

    @torch.no_grad()
    def conjugate(self) -> torch.Tensor:
        """
        Return the complex conjugate of the cached full PMNS matrix.
        
        Args:
            None.
        
        Returns:
            Complex conjugated full PMNS matrix.
        """
        return self.pmns_matrix().conj()

    @torch.no_grad()
    def transpose(self) -> torch.Tensor:
        """
        Return the transpose of the cached full PMNS matrix.
        
        Args:
            None.
        
        Returns:
            Transposed full PMNS matrix with the last two axes swapped.
        """
        return self.pmns_matrix().transpose(-2, -1)

    @torch.no_grad()
    def dagger(self) -> torch.Tensor:
        """
        Return the Hermitian conjugate of the cached full PMNS matrix.
        
        Args:
            None.
        
        Returns:
            Hermitian-conjugated full PMNS matrix.
        """
        U = self.pmns_matrix()

        return U.conj().transpose(-2, -1)

    @torch.no_grad()
    def reduced_conjugate(self) -> torch.Tensor:
        """
        Return the complex conjugate of the reduced PMNS matrix.
        
        Args:
            None.
        
        Returns:
            Complex conjugated reduced PMNS matrix.
        """
        return self.reduced().conj()

    @torch.no_grad()
    def reduced_transpose(self) -> torch.Tensor:
        """
        Return the transpose of the reduced PMNS matrix.
        
        Args:
            None.
        
        Returns:
            Transposed reduced PMNS matrix with the last two axes swapped.
        """
        return self.reduced().transpose(-2, -1)

    @torch.no_grad()
    def reduced_dagger(self) -> torch.Tensor:
        """
        Return the Hermitian conjugate of the reduced PMNS matrix.
        
        Args:
            None.
        
        Returns:
            Hermitian-conjugated reduced PMNS matrix.
        """
        Ured = self.reduced()

        return Ured.conj().transpose(-2, -1)

