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
Shared PMNS mixing-matrix infrastructure for peanuts-torch.

This module defines the parts of the PMNS mixing-matrix machinery that are
common to every flavour scenario:
    1) the Standard Model 3-flavour case (``tpeanuts.core.SM.pmns.PMNS_SM``)
    2) BSM extensions such as the 3+1 sterile-neutrino case
        (``tpeanuts.core.BSM.PMNS_sterile.PMNS_sterile``).

Both concrete classes subclass the abstract ``PMNS`` defined here.

Notes:
------
The implementation is GPU-compatible, fully based on torch, and supports both
scalar and batched oscillation parameters.

Module contents:
----------------
    PMNSParams
        Immutable container for the 3 standard mixing angles, the CP phase,
        and the RuntimeContext used to build the Standard Model mixing
        matrix. Pure mixing-matrix inputs; mass-squared splittings and
        named-preset construction live in
        ``core.common.oscillation.OscillationParameters``.

    PMNS
        Abstract base class. Builds the shared rotation/phase matrices
        (``R12``, ``R13``, ``R23``, ``Delta``), sized generically by
        ``self.n_flavours`` so the same formulas serve both the 3x3 SM case
        and the 4x4 (or larger) BSM cases,
        
        and implements the flavour-count-agnostic parts of the public interface:
          (``refresh``, ``H_flavour_basis``, ``H_mass_basis``, 
          ``select_antinu``, ``__getitem__``, ``conjugate``, ``transpose``, 
          ``dagger``, ``reduced_conjugate``, ``reduced_transpose``, 
          ``reduced_dagger``).
        
        Concrete subclasses must implement:
          ``pmns_matrix``, ``reduced``, and ``operator_flavour_basis``,
        since the product structure of the full  mixing matrix is genuinely
        different per scenario (extra active-sterile rotations are inserted
        at specific points that follow from a scenario-specific commutation
        argument, not from matrix size alone).

The neutrino reduced matrix is available through

    pmns.reduced(antinu=False),

while ``antinu=True`` returns its complex conjugate. A neutrino-only cached
copy remains stored as ``pmns.U`` for backwards compatibility.

    H = U_red diag(k_i) U_red^T + diag(V, 0, 0).

Main attributes after initialization:

    pmns.pmns
        Full PMNS matrix.

    pmns.U
        Reduced matrix U_red = R13 @ R12.

    pmns.operator_flavour_basis(...)
        Transforms any reduced-basis operator to the flavour basis.

    pmns.H_flavour_basis(...)
        Transforms a reduced-basis Hamiltonian to the flavour basis.

    pmns.H_mass_basis(...)
        Transforms a flavour-basis Hamiltonian to the mass basis using the
        reduced mixing matrix.

This module does not compute Hamiltonians, evolutors, probabilities, or matter
effects. It only provides mixing matrices and transformation basis utilities.

"""


from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Union, cast

import torch

from tpeanuts.util.type import (
    cdtype_from_real,
    as_tensor,
    TensorLike,
)
from tpeanuts.util.context import RuntimeContext


@dataclass(frozen=True)
class PMNSParams:
    """
    Immutable container for Standard Model PMNS mixing parameters.

    Parameters
    ----------
    theta12:
        Solar mixing angle theta_12 in radians.

    theta13:
        Reactor mixing angle theta_13 in radians.

    theta23:
        Atmospheric mixing angle theta_23 in radians.

    delta:
        CP-violating phase delta in radians.

    context:
        Runtime device/dtype used to store tensor parameters.

    Notes
    -----
    Angle parameters are stored as real torch tensors on the ``context``
    device and dtype. Mass-squared splittings and named-preset construction
    live in ``core.common.oscillation.OscillationParameters``, since the
    PMNS mixing matrix itself does not depend on them.
    """

    theta12: TensorLike
    theta13: TensorLike
    theta23: TensorLike
    delta: TensorLike
    context: RuntimeContext

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "theta12",
            as_tensor(self.theta12, device=self.context.device, dtype=self.context.dtype),
        )
        object.__setattr__(
            self,
            "theta13",
            as_tensor(self.theta13, device=self.context.device, dtype=self.context.dtype),
        )
        object.__setattr__(
            self,
            "theta23",
            as_tensor(self.theta23, device=self.context.device, dtype=self.context.dtype),
        )
        object.__setattr__(
            self,
            "delta",
            as_tensor(self.delta, device=self.context.device, dtype=self.context.dtype),
        )

    @property
    def device(self) -> torch.device:
        """
        Return the torch device that stores the PMNS parameter tensors.

        Args:
            None.

        Returns:
            torch.device associated with the stored PMNS parameters.
        """
        return self.context.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Return the real torch dtype used by the PMNS parameter tensors.

        Args:
            None.

        Returns:
            torch.dtype associated with the stored real PMNS parameters.
        """
        return self.context.dtype


class PMNS(torch.nn.Module, abc.ABC):
    """
    Abstract base class for PMNS-compatible mixing-matrix generators.

    Shared, flavour-count-agnostic machinery for every PMNS scenario in this
    project. Concrete subclasses (``tpeanuts.core.SM.pmns.PMNS_SM`` for the
    3-flavour Standard Model, ``tpeanuts.core.BSM.PMNS_sterile.PMNS_sterile``
    for the 3+1 sterile-neutrino extension) only need to supply
    ``n_flavours``/``n_active``/``n_sterile`` and implement ``pmns_matrix``,
    ``reduced``, and ``operator_flavour_basis``.

    Parameters
    ----------
    params:
        Object bundling the standard mixing angles ``theta12``, ``theta13``,
        ``theta23``, the CP-violating phase ``delta`` (all in radians), and
        a ``device``/``dtype`` pair. ``PMNSParams`` for the Standard Model;
        BSM subclasses may bundle additional scenario-specific parameters
        (e.g. ``PMNSSterileParams``) as long as those four angle attributes
        and the device/dtype pair are present.

    Notes
    -----
    Functions

        ``R12``, ``R13``, ``R23``, and ``Delta`` 
    
    are implemented here, sized by  ``self.n_flavours``: the rotation in 
    the (i, j) subspace and the  CP-phase placement at index 2 are the same 
    formula regardless of how many flavours the full matrix has 
    
    the rotation is embedded in a larger identity for BSM cases).
     
    What is genuinely scenario-specific is the *product structure* combining 
    these blocks into the full and reduced mixing matrices (which extra 
    rotations are inserted, and where) -- that is why
     
       ``pmns_matrix``, ``reduced``, ``operator_flavour_basis``
    
    remain abstract here.
    """

    n_flavours: int = 3
    n_active: int = 3
    n_sterile: int = 0

    def __init__(
        self,
        params: PMNSParams,
    ) -> None:
        """
        Store the mixing parameters and cache the full and reduced matrices.

        Args:
            params: Object bundling theta12, theta13, theta23, delta, and a
                device/dtype pair (see class docstring).

        Returns:
            None.
        """
        super().__init__()

        self.params = params

        self.register_buffer("pmns", self.pmns_matrix())
        self.register_buffer("U", self.reduced())

    @property
    def device(self) -> torch.device:
        """Return the torch device used by the PMNS parameter tensors."""
        return self.params.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the real torch dtype used by the PMNS parameter tensors."""
        return self.params.dtype

    def _rot(
        self,
        i: int,
        j: int,
        theta: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build an ``n_flavours x n_flavours`` complex rotation matrix.

        The rotation acts on axes ``i`` and ``j`` (0-indexed):

            R[i, i] = cos theta
            R[j, j] = cos theta
            R[i, j] = +sin theta * e^{-i phase}
            R[j, i] = -sin theta * e^{+i phase}
            R[k, k] = 1   for k not in {i, j}

        Notes:
        ------    
        When ``phase`` is None the rotation is real (phase treated as zero).
        This single formula, sized by ``self.n_flavours``, 
        
            builds R12/R13/R23  for the Standard Model 
            (n_flavours=3, phase=None) 
            
            builds R12/R13/R23 plus  the active-sterile rotations 
            R14/R24/R34 for the 3+1 case
            (n_flavours=4, phase set for the active-sterile rotations).

        Args:
            i: Row/column index of the first axis (0-indexed).
            j: Row/column index of the second axis (0-indexed).
            theta: Mixing angle tensor with batch shape (...).
            phase: Optional CP phase tensor with the same batch shape as
                theta.

        Returns:
            Complex rotation tensor shaped (..., n_flavours, n_flavours).
        """
        cdtype = cdtype_from_real(theta.dtype)
        c = torch.cos(theta)
        s = torch.sin(theta)

        n = self.n_flavours
        eye = torch.eye(n, device=theta.device, dtype=cdtype)
        out = eye.expand(*c.shape, n, n).clone()

        if phase is not None:
            exp_p = torch.exp(1j * phase.to(dtype=cdtype))
        else:
            exp_p = torch.ones_like(c, dtype=cdtype)

        out[..., i, i] = c.to(dtype=cdtype)
        out[..., j, j] = c.to(dtype=cdtype)
        out[..., i, j] = s.to(dtype=cdtype) * exp_p.conj()
        out[..., j, i] = -s.to(dtype=cdtype) * exp_p

        return out

    def _phase_diag(
        self,
        index: int,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """Build an ``n_flavours x n_flavours`` diagonal CP-phase matrix.

        Identity everywhere except a single ``exp(i phase)`` entry at
        ``(index, index)``. 
        
        Sized by ``self.n_flavours``, this builds

            ``Delta = diag(1, 1, exp(i delta))`` for the Standard Model
            (n_flavours=3) 
             
            ``diag(1, 1, exp(i delta), 1)`` for the 3+1 case
            (n_flavours=4) with the same formula.

        Args:
            index: Diagonal position carrying the CP phase (0-indexed).
            phase: CP-phase tensor with batch shape (...).

        Returns:
            Complex diagonal tensor shaped (..., n_flavours, n_flavours).
        """
        cdtype = cdtype_from_real(phase.dtype)
        exp_p = torch.exp(1j * phase.to(dtype=cdtype))

        n = self.n_flavours
        eye = torch.eye(n, device=phase.device, dtype=cdtype)
        out = eye.expand(*exp_p.shape, n, n).clone()
        out[..., index, index] = exp_p

        return out

    def R12(self) -> torch.Tensor:
        """
        Build the rotation matrix in the (e, mu) = (0, 1) subspace.

        Returns:
            Complex rotation matrix shaped (..., n_flavours, n_flavours).
        """
        return self._rot(0, 1, cast(torch.Tensor, self.params.theta12))

    def R13(self) -> torch.Tensor:
        """
        Build the rotation matrix in the (e, tau) = (0, 2) subspace.

        Returns:
            Complex rotation matrix shaped (..., n_flavours, n_flavours).
        """
        return self._rot(0, 2, cast(torch.Tensor, self.params.theta13))

    def R23(self) -> torch.Tensor:
        """
        Build the rotation matrix in the (mu, tau) = (1, 2) subspace.

        Returns:
            Complex rotation matrix shaped (..., n_flavours, n_flavours).
        """
        return self._rot(1, 2, cast(torch.Tensor, self.params.theta23))

    def Delta(self) -> torch.Tensor:
        """
        Build the diagonal CP-phase matrix, phase carried at index 2.

        Formula: diag(1, 1, exp(i delta), ...).

        Returns:
            Complex diagonal matrix shaped (..., n_flavours, n_flavours).
        """
        return self._phase_diag(2, cast(torch.Tensor, self.params.delta))

    @torch.no_grad()
    def select_antinu(
        self,
        U: torch.Tensor,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """
        Select the neutrino or antineutrino convention for a mixing matrix.

        Args:
            U: Full or reduced mixing matrix shaped (..., n, n).
            antinu: Boolean scalar or tensor mask. True selects U*.

        Returns:
            U for neutrinos and its complex conjugate for antineutrinos,
            preserving and broadcasting the matrix batch dimensions.
        """
        if isinstance(antinu, bool):
            return U.conj() if antinu else U

        antinu = antinu.to(device=U.device, dtype=torch.bool)
        while antinu.ndim < U.ndim - 2:
            antinu = antinu.unsqueeze(-1)

        return torch.where(
            antinu[..., None, None],
            U.conj(),
            U,
        )

    @abc.abstractmethod
    def reduced(
        self,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """
        Build the reduced neutrino or antineutrino mixing matrix.

        Args:
            antinu: Boolean scalar or tensor mask. True selects the
                complex-conjugated convention.

        Returns:
            Complex reduced mixing matrix shaped (..., n_flavours, n_flavours).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pmns_matrix(
        self,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """
        Build the full neutrino or antineutrino PMNS matrix.

        Args:
            antinu: Boolean scalar or tensor mask. True selects the
                complex-conjugated convention.

        Returns:
            Complex full PMNS matrix shaped (..., n_flavours, n_flavours).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def operator_flavour_basis(
        self,
        operator_reduced: torch.Tensor,
        antinu: Union[bool, torch.Tensor] = False,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Transform an operator from the reduced to the flavour basis.

        Args:
            operator_reduced: Reduced-basis operator shaped
                (..., n_flavours, n_flavours).
            antinu: Boolean scalar or tensor mask selecting antineutrinos.
            device: Optional output device; defaults to the operator device.
            dtype: Optional output dtype; defaults to the operator dtype.

        Returns:
            Operator represented in the full flavour basis.
        """
        raise NotImplementedError

    @torch.no_grad()
    def H_flavour_basis(
        self,
        H_reduced: torch.Tensor,
        antinu: Union[bool, torch.Tensor] = False,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Transform a reduced-basis Hamiltonian to the flavour basis.

        Args:
            H_reduced: Reduced-basis Hamiltonian shaped
                (..., n_flavours, n_flavours).
            antinu: Boolean scalar or tensor mask selecting antineutrinos.
            device: Optional output device; defaults to H_reduced.device.
            dtype: Optional output dtype; defaults to H_reduced.dtype.

        Returns:
            Hamiltonian represented in the full flavour basis.
        """
        return self.operator_flavour_basis(
            H_reduced,
            antinu=antinu,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def H_mass_basis(
        self,
        H_flavour: torch.Tensor,
        antinu: Union[bool, torch.Tensor] = False,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Transform a flavour-basis Hamiltonian to the reduced mass basis.

        Applies ``H_mass = Ured^dagger H_flavour Ured`` with
        ``Ured = reduced(antinu=antinu)``.

        Args:
            H_flavour: Flavour-basis Hamiltonian shaped
                (..., n_flavours, n_flavours).
            antinu: Boolean scalar or tensor mask selecting antineutrinos.
            device: Optional output device; defaults to H_flavour.device.
            dtype: Optional output dtype; defaults to H_flavour.dtype.

        Returns:
            Hamiltonian represented in the reduced mass basis.
        """
        n = self.n_flavours
        if H_flavour.shape[-2:] != (n, n):
            raise ValueError(f"H_flavour must have final dimensions ({n}, {n}).")

        output_device = (
            H_flavour.device if device is None else torch.device(device)
        )
        output_dtype = H_flavour.dtype if dtype is None else dtype
        H_flavour = H_flavour.to(
            device=output_device,
            dtype=output_dtype,
        )

        Ured = self.reduced(antinu=antinu).to(
            device=output_device,
            dtype=output_dtype,
        )

        return Ured.conj().transpose(-2, -1) @ H_flavour @ Ured

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
    def jarlskog_invariant(self) -> torch.Tensor:
        """
        Compute the Jarlskog CP-violation invariant.

        Uses the rephasing-invariant definition (PDG convention):

            J = Im[ U_{e1} U_{μ2} U*_{e2} U*_{μ1} ]

        which for the standard PMNS parametrisation reduces to

            J = (1/8) sin 2θ₁₂ sin 2θ₁₃ sin 2θ₂₃ cos θ₁₃ sin δ_CP.

        The computation uses the 3×3 active subblock of the full PMNS matrix,
        so the result is valid for both the SM (n_flavours=3) and BSM
        extensions (n_flavours=4), where the invariant is extracted from
        the active-flavour rows and the first three mass-eigenstate columns.

        Args:
            None.

        Returns:
            Real scalar tensor (shape () for unbatched parameters).
        """
        U = self.pmns_matrix()
        Ua = U[..., :3, :3]
        return torch.imag(
            Ua[..., 0, 0] * Ua[..., 1, 1]
            * Ua[..., 0, 1].conj() * Ua[..., 1, 0].conj()
        )

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
        return self.reduced(antinu=True)

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

    @torch.no_grad()
    def vacuum_flavour_projector(
        self,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """
        Return the decohered vacuum flavour projector |U_{alpha i}|^2.

        Args:
            antinu: Boolean scalar or tensor mask. True selects the
                antineutrino PMNS convention through ``pmns_matrix``.

        Returns:
            Real tensor shaped (..., n_flavours, n_flavours) with flavour
            rows and mass-eigenstate columns.
        """
        U = self.pmns_matrix(antinu=antinu)
        return U.abs() ** 2
