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

"""Density-matrix representation of a (possibly mixed) neutrino state.

Every propagated neutrino state in tpeanuts is a density matrix, never a bare
state vector: a pure coherent flavour superposition is simply the special
case where rho has rank 1 (``pure``/``basis_state`` below), and an
incoherent classical mixture (e.g. mass eigenstates leaving the Sun with no
relative phase) is the special case where rho is diagonal
(``from_weights``).

Genuine decoherence during propagation and a
magnetic-moment-induced neutrino/antineutrino conversion both need
everything in between -- a general Hermitian, positive semi-definite rho --
which a bare state vector cannot represent at all. Choosing rho as the one
state representation from the start means no separate "pure state" code path
has to be generalized later.

This module only defines the state object itself and the basis-independent
operations every consumer needs (probabilities, both a "sandwiching"
transform law and a genuinely general Liouville-space one, and validation of
the Hermitian/positive-semi-definite invariants). It builds no Hamiltonians,
knows no mixing angles, and runs no propagation -- that is
``tpeanuts.physics.common.mixing``, ``tpeanuts.physics.common.generator``,
and ``tpeanuts.physics.evolution``, respectively. Only the open-system/
decoherence evolver within the last of those is not yet built (see
``apply_superoperator``'s own docstring below for the mechanism it will
use).

Module contents:
    DensityMatrix
        Batched Hermitian density matrix over a FlavourSpace, with
        constructors for pure states, incoherent mixtures, and computational
        basis eigenstates, plus probability extraction, a generic
        "sandwiching" transformation law (``conjugate_by``), a genuinely
        general Liouville-space one (``vectorize``/``from_vector``/
        ``apply_superoperator``, for a future open-system/decoherence
        evolver), and Hermiticity/positivity diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import eigvalsh_chunked
from tpeanuts.util.tensor import TensorLike, as_tensor, cdtype_from_real


@dataclass(frozen=True)
class DensityMatrix:
    """Batched density matrix rho over a given FlavourSpace.

    Attributes:
        tensor: Complex tensor shaped ``(..., dim, dim)``, ``dim =
            space.dim``. Every constructor below builds a Hermitian
            ``tensor``, but nothing in this class assumes ``trace(rho) ==
            1``: it can equally represent a normalized probability state or
            an unnormalized sub-flux weight -- each constructor's docstring
            states which convention it follows.
        space: FlavourSpace giving the meaning of the ``dim`` computational
            basis indices (flavour, and helicity when
            ``space.is_helicity_extended``).
        context: RuntimeContext (device, real dtype) the state is stored on.
            ``tensor``'s actual dtype is the complex counterpart,
            ``cdtype_from_real(context.dtype)``.

    Raises:
        ValueError: If ``tensor``'s final two dimensions do not both equal
            ``space.dim``.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.physics import DensityMatrix
        >>> rho = DensityMatrix.basis_state(1, tp.FlavourSpace(), context)   # a pure muon neutrino
        >>> rho.probabilities().tolist()
        [0.0, 1.0, 0.0]
    """

    tensor: torch.Tensor
    space: FlavourSpace
    context: RuntimeContext

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            tensor: TensorLike,
            space: FlavourSpace,
            context: RuntimeContext,
        ) -> None: ...

    def __post_init__(self) -> None:
        cdtype = cdtype_from_real(self.context.dtype)
        tensor = as_tensor(self.tensor, device=self.context.device, dtype=cdtype)

        dim = self.space.dim
        if tensor.ndim < 2 or tensor.shape[-2:] != (dim, dim):
            raise ValueError(
                f"tensor must have final dimensions ({dim}, {dim}) to match "
                f"space.dim, got {tuple(tensor.shape[-2:])}."
            )

        object.__setattr__(self, "tensor", tensor)

    @classmethod
    def pure(
        cls,
        amplitudes: TensorLike,
        space: FlavourSpace,
        context: RuntimeContext,
    ) -> "DensityMatrix":
        """Build ``rho = |psi><psi|`` from a coherent flavour-amplitude state.

        Args:
            amplitudes: Complex state amplitudes shaped ``(..., dim)``,
                ``dim = space.dim``. This constructor does not renormalize:
                a unit-norm input yields a trace-1 (probability-normalized)
                state, while an unnormalized input yields ``trace(rho) =
                sum(|amplitudes|**2)``.
            space: FlavourSpace the amplitudes are expressed in.
            context: RuntimeContext for the returned density matrix.

        Returns:
            DensityMatrix with ``tensor = amplitudes ⊗ amplitudes.conj()``
            (outer product on the last axis).

        Raises:
            ValueError: If ``amplitudes``'s last dimension is not
                ``space.dim``.
        """
        cdtype = cdtype_from_real(context.dtype)
        psi = as_tensor(amplitudes, device=context.device, dtype=cdtype)
        if psi.shape[-1] != space.dim:
            raise ValueError(
                f"amplitudes must have last dimension {space.dim} (space.dim), "
                f"got {psi.shape[-1]}."
            )

        rho = psi[..., :, None] * psi.conj()[..., None, :]
        return cls(tensor=rho, space=space, context=context)

    @classmethod
    def from_weights(
        cls,
        weights: TensorLike,
        space: FlavourSpace,
        context: RuntimeContext,
    ) -> "DensityMatrix":
        """Build a fully incoherent mixture ``rho = diag(weights)``.

        Every off-diagonal coherence is exactly zero from construction,
        rather than damped away during propagation -- this is the
        already-decohered limit needed e.g. for an incoherent mass-eigenstate
        mixture leaving the Sun, or as the target that a decoherence-capable
        open-system evolver (``tpeanuts.physics.evolution.liouville.open_system``, not
        yet built -- ``tpeanuts.physics.evolution.numerical.numerical`` handles only
        unitary evolution so far) should relax towards as its dissipation
        strength grows.

        Args:
            weights: Real, non-negative populations shaped ``(..., dim)``,
                ``dim = space.dim``. Not required to sum to 1 -- an
                unnormalized input yields ``trace(rho) = sum(weights)``,
                exactly like ``pure`` above.
            space: FlavourSpace the weights are expressed in.
            context: RuntimeContext for the returned density matrix.

        Returns:
            DensityMatrix with ``tensor = diag(weights)`` (batched).

        Raises:
            ValueError: If ``weights``'s last dimension is not ``space.dim``.
        """
        w = as_tensor(weights, device=context.device, dtype=context.dtype)
        if w.shape[-1] != space.dim:
            raise ValueError(
                f"weights must have last dimension {space.dim} (space.dim), "
                f"got {w.shape[-1]}."
            )

        cdtype = cdtype_from_real(context.dtype)
        rho = torch.diag_embed(w.to(dtype=cdtype))
        return cls(tensor=rho, space=space, context=context)

    @classmethod
    def basis_state(
        cls,
        index: int,
        space: FlavourSpace,
        context: RuntimeContext,
    ) -> "DensityMatrix":
        """Build the pure state at one computational basis index.

        A convenience wrapper around ``pure`` for the common case of
        starting in a single basis state (e.g. "start as a pure electron
        neutrino"). Which physical label attaches to each index (electron
        flavour, sterile flavour i, a helicity sign, ...) is a convention
        owned by ``tpeanuts.physics.common.mixing``, not by
        this class -- this constructor only knows about the bare
        computational basis index.

        Args:
            index: Basis index in ``range(space.dim)``.
            space: FlavourSpace the index is defined in.
            context: RuntimeContext for the returned density matrix.

        Returns:
            DensityMatrix equal to the outer product of the index-th
            standard basis vector with itself (``trace`` exactly 1).

        Raises:
            ValueError: If ``index`` is out of range for ``space.dim``.
        """
        if not 0 <= index < space.dim:
            raise ValueError(f"index must be in range(0, {space.dim}), got {index}.")

        cdtype = cdtype_from_real(context.dtype)
        psi = torch.zeros(space.dim, device=context.device, dtype=cdtype)
        psi[index] = 1.0
        return cls.pure(psi, space, context)

    @property
    def dim(self) -> int:
        """Hilbert-space dimension.

        Returns:
            ``space.dim``.
        """
        return self.space.dim

    @property
    def batch_shape(self) -> torch.Size:
        """Leading batch shape shared by every batched quantity this state produces.

        Returns:
            ``tensor.shape`` without its trailing ``(dim, dim)``.
        """
        return self.tensor.shape[:-2]

    def trace(self) -> torch.Tensor:
        """Compute the trace of rho, batched.

        Returns:
            Real tensor shaped ``batch_shape``. Exactly 1 for a state built
            by ``pure``/``basis_state``/``from_weights`` from a unit-norm
            (respectively unit-sum) input.
        """
        return torch.diagonal(self.tensor, dim1=-2, dim2=-1).sum(dim=-1).real

    def probabilities(self) -> torch.Tensor:
        """Return the diagonal populations of rho in its current basis.

        Returns:
            Real tensor shaped ``(..., dim)``: the diagonal of ``tensor``,
            i.e. the physically measurable population in each computational
            basis state -- the density-matrix generalization of
            ``|psi_i|**2`` for a pure state.
        """
        return torch.diagonal(self.tensor, dim1=-2, dim2=-1).real

    def conjugate_by(self, operator: torch.Tensor) -> "DensityMatrix":
        """Return ``operator @ rho @ operator^dagger`` as a new DensityMatrix.

        This is the transformation law for a density matrix under a single
        "sandwiching" operator -- a basis change (e.g. flavour <-> mass) and
        a *unitary* propagation step are both instances of it. For a unitary
        ``operator`` it exactly preserves ``trace``, Hermiticity, and
        positive semi-definiteness (see this class's tests for a direct
        check); for a general non-unitary ``operator`` none of those three
        are guaranteed to survive. Note that this is *not* general enough to
        express genuine dissipation (a Lindblad-type decoherence channel
        has no single ``operator``/``operator^dagger`` factorization) --
        that needs ``apply_superoperator`` below, acting in the doubled,
        vectorized Liouville space instead.

        Args:
            operator: Complex tensor shaped ``(..., dim, dim)``,
                broadcastable against ``tensor``.

        Returns:
            New DensityMatrix sharing this instance's ``space`` and
            ``context``.

        Raises:
            ValueError: If ``operator``'s final two dimensions do not both
                equal ``dim``.
        """
        dim = self.space.dim
        if operator.shape[-2:] != (dim, dim):
            raise ValueError(
                f"operator must have final dimensions ({dim}, {dim}), got "
                f"{tuple(operator.shape[-2:])}."
            )

        operator = operator.to(dtype=self.tensor.dtype, device=self.tensor.device)
        rho = operator @ self.tensor @ operator.conj().transpose(-2, -1)
        return replace(self, tensor=rho)

    def vectorize(self) -> torch.Tensor:
        """Flatten ``rho`` into its Liouville-space vector representation.

        Uses ``torch``'s own native flattening order -- row-major
        (``tensor.reshape(..., dim*dim)``, no transpose) -- rather than the
        column-major ``vec`` convention most linear-algebra textbooks use.
        This is a deliberate, documented choice, not an oversight: with
        this convention, ``vectorize``/``from_vector`` are exactly
        ``reshape`` (cheap, no copy when already contiguous), but it changes
        which Kronecker-product identity is correct for building a
        superoperator elsewhere in the project. For row-major flattening,
        ``vec_r(A X B) = (A kron B^T) vec_r(X)`` (*not* the textbook
        ``vec(A X B) = (B^T kron A) vec(X)``, which assumes column-major
        flattening) -- every Liouville-space superoperator this project
        builds (e.g. a future
        ``tpeanuts.physics.evolution.liouville.open_system.commutator_superoperator``)
        must use this row-major form. Verified directly in this class's own
        tests: ``apply_superoperator(U kron U.conj())`` reproduces
        ``conjugate_by(U)`` exactly for a unitary ``U``, which is only true
        under the row-major identity above.

        Returns:
            Complex tensor shaped ``(..., dim*dim)``.
        """
        return self.tensor.reshape(*self.batch_shape, self.dim * self.dim)

    @classmethod
    def from_vector(
        cls,
        vector: torch.Tensor,
        space: FlavourSpace,
        context: RuntimeContext,
    ) -> "DensityMatrix":
        """Build a DensityMatrix from its Liouville-space vector representation.

        The exact inverse of ``vectorize``: ``rho.from_vector(rho.vectorize(),
        space, context)`` round-trips to ``rho`` unchanged (same row-major
        ``reshape`` convention, no transpose).

        Args:
            vector: Complex tensor shaped ``(..., dim*dim)``, ``dim =
                space.dim``.
            space: FlavourSpace the vector is expressed in.
            context: RuntimeContext for the returned density matrix.

        Returns:
            DensityMatrix with ``tensor = vector.reshape(..., dim, dim)``.

        Raises:
            ValueError: If ``vector``'s last dimension is not
                ``space.dim**2``.
        """
        dim = space.dim
        if vector.shape[-1] != dim * dim:
            raise ValueError(
                f"vector must have last dimension {dim * dim} (space.dim**2), "
                f"got {vector.shape[-1]}."
            )
        tensor = vector.reshape(*vector.shape[:-1], dim, dim)
        return cls(tensor=tensor, space=space, context=context)

    def apply_superoperator(self, superoperator: torch.Tensor) -> "DensityMatrix":
        """Return the state obtained by applying a Liouville-space superoperator to ``rho``.

        The genuinely general transformation law -- ``vec_r(rho') =
        superoperator @ vec_r(rho)`` -- of which ``conjugate_by`` is only
        the special "sandwiching" case: any ``operator @ rho @
        operator^dagger`` is also expressible this way (via
        ``operator kron operator.conj()``, see ``vectorize``'s own
        docstring for why that specific Kronecker form is correct under
        this project's row-major convention), but a genuine dissipative
        channel (a Lindblad decoherence term, no
        ``operator``/``operator^dagger`` factorization) is only expressible
        through this method, not ``conjugate_by``.

        Args:
            superoperator: Complex tensor shaped ``(..., dim*dim, dim*dim)``,
                broadcastable against this state's vectorized tensor -- the
                ``...`` batch shape need not match ``self.batch_shape``
                exactly, only broadcast against it (e.g. one unbatched
                state evolved by one superoperator per energy bin, the
                common case for a real sweep): the *result*'s batch shape
                is whatever the broadcasted matmul actually produces, not
                necessarily this instance's own.

        Returns:
            New DensityMatrix sharing this instance's ``space`` and
            ``context``, with ``batch_shape`` given by broadcasting this
            instance's own batch shape against ``superoperator``'s leading
            dimensions (not necessarily equal to either one alone).

        Raises:
            ValueError: If ``superoperator``'s final two dimensions do not
                both equal ``dim*dim``.
        """
        size = self.dim * self.dim
        if superoperator.shape[-2:] != (size, size):
            raise ValueError(
                f"superoperator must have final dimensions ({size}, {size}), got "
                f"{tuple(superoperator.shape[-2:])}."
            )

        vector = self.vectorize()
        superoperator = superoperator.to(dtype=vector.dtype, device=vector.device)
        new_vector = (superoperator @ vector[..., None])[..., 0]
        # new_vector's own batch shape (from the broadcasted matmul above)
        # is what must drive the reshape below, not self.batch_shape: those
        # two differ exactly when superoperator carries batch dimensions
        # this (possibly unbatched) state does not -- e.g. propagating one
        # initial state through one Liouvillian per energy bin, the normal
        # case for a real energy sweep. from_vector already reshapes from
        # the vector's own shape for exactly this reason.
        return DensityMatrix.from_vector(new_vector, self.space, self.context)

    def is_hermitian(self, atol: float = 1e-8) -> torch.Tensor:
        """Check ``rho == rho^dagger`` elementwise, batched.

        Args:
            atol: Absolute tolerance for the elementwise comparison.

        Returns:
            Boolean tensor shaped ``batch_shape``, True where every entry of
            ``rho`` matches its Hermitian conjugate within ``atol``.
        """
        dagger = self.tensor.conj().transpose(-2, -1)
        diff = (self.tensor - dagger).abs()
        max_diff = diff.reshape(*diff.shape[:-2], -1).amax(dim=-1)
        return max_diff <= atol

    def is_positive_semidefinite(self, atol: float = 1e-8) -> torch.Tensor:
        """Check that every eigenvalue of rho is at least ``-atol``, batched.

        Diagonalizes ``rho`` (assumed Hermitian; call ``is_hermitian`` first
        if that is not already guaranteed), so this is an ``O(dim**3))``
        diagnostic meant for validation and tests, not a hot-path check.

        Args:
            atol: Eigenvalues below ``-atol`` count as a violation, allowing
                for floating-point noise around an exact zero eigenvalue.

        Returns:
            Boolean tensor shaped ``batch_shape``.
        """
        eigenvalues = eigvalsh_chunked(self.tensor)
        return torch.all(eigenvalues >= -atol, dim=-1)
