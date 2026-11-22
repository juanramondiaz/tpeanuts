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

"""Generalized N-flavour neutrino mixing matrix.

- A single ``MixingMatrix`` class serves the plain Standard Model, the 3+1
    sterile extension, and the general 3+N extension alike -- genuinely
    general, not merely "any N": for ``n_sterile <= 1`` every physical
    mixing angle a general U(3+n_sterile) parametrization needs is
    reachable (3 standard + 3 per sterile flavour); for ``n_sterile >= 2``,
    the sterile-sterile rotations below close what would otherwise be an
    ``n_sterile*(n_sterile-1)/2``-dimensional gap (e.g. the single
    ``theta_45`` a 3+2 scenario needs on top of the 3+1-style angles) --
    without them, some physically distinct "which flavour combination has
    which mass" assignments are unreachable even for purely active-flavour
    oscillation observables, not just once some other sterile-distinguishing
    physics (a magnetic moment, sterile NSI, ...) is also active.
- Every rotation is built by the same generic Givens-rotation formula, sized
    by ``space.n_flavour``.
- The standard-sterile and sterile-sterile rotations are simply an empty
    product (the identity) when ``space.n_sterile == 0`` (respectively
    ``< 2``), so every method below reproduces the plain 3-flavour result
    exactly in that case, and the 3+1 formulas exactly when
    ``space.n_sterile == 1`` (3+1 needs no sterile-sterile rotation at all).

Physics background
-------------------
- The mixing matrix factorizes as:

 ``U = O @ (R13 @ Delta^dagger @ R12 @ [sterile e rot] @ [Majorana phase])``

  where the "outer" block

``O = R23 @ Delta @ [sterile muon/tau rotations] @ [sterile-sterile rotations]``

commutes with the charged-current matter potential ``diag(V_CC, 0, ..., 0)``
(none of these ever touch the electron index alone -- the sterile-sterile
rotations do not touch *any* standard-flavour index), while everything on
the other side does not.

- This is what lets a propagation Hamiltonian be assembled directly in the
"reduced" basis and dressed with ``O`` only once, rather than in the full
flavour basis at every step.

 ``reduced()``/``pmns_matrix()``/``outer_block()`` below
compute these three matrices directly (not ``pmns_matrix = outer_block @
reduced``: the two sides carry ``Delta``/``Delta.conj()`` in different
places, exactly as in the plain 3-flavour case).

- Which optional pieces (sterile rotations, Majorana phase factor) are
included is read entirely from the ``space``/``sterile``/``nature``
arguments.

- ``space.n_sterile == 0`` and ``sterile is None`` recovers the plain Standard
Model; ``nature`` is a ``tpeanuts.physics.common.nature.Dirac`` or
``Majorana`` strategy.

Module contents:
    StandardMixingParameters
        The 3-flavour standard-sector mixing angles and CP phase.
    SterileMixingParameters
        Standard-sterile mixing angles/phases, generalized to N sterile
        flavours, plus the sterile-sterile mixing angles/phases needed for
        genuine U(3+N) generality once N >= 2.
    MixingMatrix
        Builds the full, reduced, and outer-block mixing matrices for any
        FlavourSpace and any Nature, plus one member at a time of the
        ``J_{alpha,beta}^{i,j}`` family of rephasing-invariant CP
        invariants (a single Jarlskog invariant only fully characterizes
        CP violation for the plain 3-flavour case; 3+N has several
        algebraically independent ones), and a helicity-aware
        ``pmns_matrix_total`` for a doubled (``n_helicity == 2``) space.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from tpeanuts.physics.common.nature import Nature
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import tree_reduce_matmul
from tpeanuts.util.tensor import TensorLike, as_tensor, cdtype_from_real, select_for_antinu


@dataclass(frozen=True)
class StandardMixingParameters:
    """3-flavour standard-sector mixing angles and CP-violating phase.

    Attributes:
        theta12: Solar mixing angle, in radians.
        theta13: Reactor mixing angle, in radians.
        theta23: Atmospheric mixing angle, in radians.
        delta13: Dirac CP-violating phase, in radians.
        context: RuntimeContext the angles are stored on.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> params = tp.StandardMixingParameters(
        ...     theta12=math.radians(33.4), theta13=math.radians(8.6), theta23=math.radians(42.0),
        ...     delta13=math.radians(230.0), context=context)
        >>> params.theta13.dtype   # plain numbers are stored as tensors on the context
        torch.float64
        >>> round(float(params.theta12), 4)
        0.5829
    """

    theta12: torch.Tensor
    theta13: torch.Tensor
    theta23: torch.Tensor
    delta13: torch.Tensor
    context: RuntimeContext

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            theta12: TensorLike,
            theta13: TensorLike,
            theta23: TensorLike,
            delta13: TensorLike,
            context: RuntimeContext,
        ) -> None: ...

    def __post_init__(self) -> None:
        for name in ("theta12", "theta13", "theta23", "delta13"):
            object.__setattr__(
                self,
                name,
                as_tensor(
                    getattr(self, name),
                    device=self.context.device,
                    dtype=self.context.dtype
                ),
            )


@dataclass(frozen=True)
class SterileMixingParameters:
    """Standard-sterile and sterile-sterile mixing angles/phases, generalized to N sterile flavours.

    Attributes:
        theta_e: Electron-sterile mixing angles, in radians, shaped
            ``(..., n_sterile)``. Entry ``k`` is the angle of the rotation
            mixing the electron flavour with sterile flavour ``k`` (matrix
            index ``3 + k``). ``n_sterile == 1`` recovers the
            scalar ``theta14``.
        theta_mu: Muon-sterile mixing angles, same shape and convention
            (recovers ``theta24`` for ``n_sterile == 1``).
        theta_tau: Tau-sterile mixing angles, same shape and convention
            (recovers ``theta34`` for ``n_sterile == 1``).
        delta_e: CP phases matching ``theta_e``, same shape.
        delta_mu: CP phases matching ``theta_mu``, same shape.
        delta_tau: CP phases matching ``theta_tau``, same shape.
        context: RuntimeContext the angles/phases are stored on.
        theta_ss: Sterile-sterile mixing angles, in radians, shaped
            ``(..., n_sterile*(n_sterile-1)//2)`` -- one per unordered pair
            of sterile flavours, ordered exactly as ``sterile_pairs`` lists
            them. ``None`` (the default) means zero sterile-sterile mixing:
            every method on ``MixingMatrix`` reproduces exactly what it
            built before this field existed. Together with
            ``theta_e``/``theta_mu``/``theta_tau``, this is what makes the
            mixing matrix a genuinely general parametrization of
            U(3+n_sterile) -- without it, ``n_sterile*(n_sterile-1)/2`` real
            mixing angles are unreachable for ``n_sterile >= 2`` (see this
            module's own docstring).
        delta_ss: CP phases matching ``theta_ss``, same shape and default.
            Given the same treatment as ``delta_e``/``delta_mu``/
            ``delta_tau`` (one phase per Givens rotation) -- this project
            does not attempt to identify which linear combinations of
            these phases are independently physical (rephasing-invariant)
            for either sector, so ``delta_ss`` is not held to a stricter
            standard than ``delta_e``/``delta_mu``/``delta_tau`` already
            are.

    Raises:
        ValueError: If ``theta_e``/``theta_mu``/``theta_tau``/``delta_e``/
            ``delta_mu``/``delta_tau`` do not all share the same last
            dimension, or if ``theta_ss``/``delta_ss`` is given with a last
            dimension other than ``n_sterile*(n_sterile-1)//2``.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.physics import SterileMixingParameters
        >>> sterile = SterileMixingParameters(
        ...     theta_e=[0.1], theta_mu=[0.05], theta_tau=[0.02], delta_e=[0.0], delta_mu=[0.0], delta_tau=[0.0],
        ...     context=context)
        >>> sterile.n_sterile, tuple(sterile.theta_mu.shape)
        (1, (1,))
    """

    theta_e: torch.Tensor
    theta_mu: torch.Tensor
    theta_tau: torch.Tensor
    delta_e: torch.Tensor
    delta_mu: torch.Tensor
    delta_tau: torch.Tensor
    context: RuntimeContext
    theta_ss: Optional[torch.Tensor] = None
    delta_ss: Optional[torch.Tensor] = None

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            theta_e: TensorLike,
            theta_mu: TensorLike,
            theta_tau: TensorLike,
            delta_e: TensorLike,
            delta_mu: TensorLike,
            delta_tau: TensorLike,
            context: RuntimeContext,
            theta_ss: Optional[TensorLike] = ...,
            delta_ss: Optional[TensorLike] = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        names = ("theta_e", "theta_mu", "theta_tau",
                 "delta_e", "delta_mu", "delta_tau")
        tensors = {
            name: as_tensor(
                getattr(self, name),
                device=self.context.device,
                dtype=self.context.dtype
            )
            for name in names
        }

        last_dims = {tensor.shape[-1] for tensor in tensors.values()}
        if len(last_dims) != 1:
            raise ValueError(
                "theta_e/theta_mu/theta_tau/delta_e/delta_mu/delta_tau "
                f"must all share the same last dimension (n_sterile), "
                f"got {last_dims}."
            )

        for name, tensor in tensors.items():
            object.__setattr__(self, name, tensor)

        n_sterile = self.theta_e.shape[-1]
        n_pairs = n_sterile * (n_sterile - 1) // 2
        for name in ("theta_ss", "delta_ss"):
            value = getattr(self, name)
            if value is None:
                tensor = torch.zeros(n_pairs, device=self.context.device, dtype=self.context.dtype)
            else:
                tensor = as_tensor(value, device=self.context.device, dtype=self.context.dtype)
                if tensor.shape[-1] != n_pairs:
                    raise ValueError(
                        f"{name}'s last dimension must be n_sterile*(n_sterile-1)//2 "
                        f"({n_pairs} for n_sterile={n_sterile}), got {tensor.shape[-1]}."
                    )
            object.__setattr__(self, name, tensor)

    @property
    def n_sterile(self) -> int:
        """Number of sterile flavours these parameters cover."""
        return self.theta_e.shape[-1]

    @property
    def sterile_pairs(self) -> List[Tuple[int, int]]:
        """Ordered list of sterile-flavour index pairs ``theta_ss``/``delta_ss`` are indexed by.

        Returns:
            ``[(i, j), ...]`` for every ``0 <= i < j < n_sterile``, in
            ``itertools.combinations`` order -- the same order
            ``theta_ss[..., k]``/``delta_ss[..., k]``'s ``k``-th entry
            corresponds to.
        """
        return list(itertools.combinations(range(self.n_sterile), 2))


def _givens_rotation(
    i: int,
    j: int,
    n: int,
    theta: torch.Tensor,
    phase: Optional[torch.Tensor],
    *,
    cdtype: torch.dtype,
) -> torch.Tensor:
    """Build an ``n x n`` complex Givens rotation acting on axes ``(i, j)``.

    Args:
        i: First 0-indexed axis the rotation acts on.
        j: Second 0-indexed axis the rotation acts on.
        n: Total matrix dimension.
        theta: Mixing angle, batch shape ``(...)``.
        phase: Optional CP phase, same batch shape as ``theta``. A real
            (phase-free) rotation when None.
        cdtype: Complex dtype for the returned tensor.

    Returns:
        Complex tensor shaped ``(..., n, n)``: identity outside the
        ``(i, j)`` subspace, with ``R[i,i] = R[j,j] = cos(theta)``,
        ``R[i,j] = sin(theta) exp(-i phase)``,
        ``R[j,i] = -sin(theta) exp(i phase)``.
    """
    c = torch.cos(theta).to(dtype=cdtype)
    s = torch.sin(theta).to(dtype=cdtype)
    exp_p = torch.ones_like(c) if phase is None else torch.exp(1j * phase.to(dtype=cdtype))

    eye = torch.eye(n, device=theta.device, dtype=cdtype)
    out = eye.expand(*theta.shape, n, n).clone()
    out[..., i, i] = c
    out[..., j, j] = c
    out[..., i, j] = s * exp_p.conj()
    out[..., j, i] = -s * exp_p
    return out


def _phase_diag(index: int, n: int, phase: torch.Tensor, *, cdtype: torch.dtype) -> torch.Tensor:
    """Build an ``n x n`` diagonal CP-phase matrix, ``exp(i phase)`` at ``index``.

    Args:
        index: 0-indexed diagonal position carrying the phase.
        n: Total matrix dimension.
        phase: CP phase, batch shape ``(...)``.
        cdtype: Complex dtype for the returned tensor.

    Returns:
        Complex tensor shaped ``(..., n, n)``: identity except a single
        ``exp(i phase)`` entry at ``(index, index)``.
    """
    exp_p = torch.exp(1j * phase.to(dtype=cdtype))
    eye = torch.eye(n, device=phase.device, dtype=cdtype)
    out = eye.expand(*phase.shape, n, n).clone()
    out[..., index, index] = exp_p
    return out


@dataclass(frozen=True)
class MixingMatrix:
    """Full, reduced, and outer-block mixing matrices for any FlavourSpace.

    Attributes:
        space: FlavourSpace this mixing matrix is built for.
        standard: 3-flavour standard-sector angles and CP phase.
        nature: Dirac or Majorana strategy
            (``tpeanuts.physics.common.nature``).
        context: RuntimeContext for the returned matrices.
        sterile: Standard-sterile angles/phases. Required iff
            ``space.n_sterile > 0``; must be omitted (None) iff
            ``space.n_sterile == 0``.

    Raises:
        ValueError: If ``sterile`` is missing when ``space.n_sterile > 0``,
            given when ``space.n_sterile == 0``, or its ``n_sterile`` does
            not match ``space.n_sterile``.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> params = tp.StandardMixingParameters(
        ...     theta12=math.radians(33.4), theta13=math.radians(8.6), theta23=math.radians(42.0),
        ...     delta13=math.radians(230.0), context=context)
        >>> mixing = tp.MixingMatrix(space=tp.FlavourSpace(), standard=params, nature=tp.Dirac(), context=context)
        >>> U = mixing.pmns_matrix()
        >>> U.shape
        torch.Size([3, 3])
        >>> bool(torch.allclose(U @ U.conj().T, torch.eye(3, dtype=U.dtype)))   # unitary
        True
        >>> [round(float(x), 3) for x in U[0].abs() ** 2]   # |U_ei|^2: the mass content of an electron neutrino
        [0.681, 0.296, 0.022]
    """

    space: FlavourSpace
    standard: StandardMixingParameters
    nature: Nature
    context: RuntimeContext
    sterile: Optional[SterileMixingParameters] = None

    def __post_init__(self) -> None:
        if self.space.n_sterile > 0:
            if self.sterile is None:
                raise ValueError(
                    f"space has {self.space.n_sterile} sterile flavour(s); "
                    "sterile mixing parameters are required."
                )
            if self.sterile.n_sterile != self.space.n_sterile:
                raise ValueError(
                    f"sterile.n_sterile ({self.sterile.n_sterile}) does not "
                    f"match space.n_sterile ({self.space.n_sterile})."
                )
        elif self.sterile is not None:
            raise ValueError(
                "space has no sterile flavours (n_sterile == 0); sterile "
                "mixing parameters must be omitted."
            )

    @property
    def _cdtype(self) -> torch.dtype:
        """Complex dtype matching context.dtype."""
        return cdtype_from_real(self.context.dtype)

    @property
    def _n(self) -> int:
        """Flavour-space dimension the mixing matrix itself acts on (excludes helicity)."""
        return self.space.n_flavour

    def R12(self) -> torch.Tensor:
        """Standard rotation in the (e, mu) = (0, 1) subspace."""
        return _givens_rotation(0, 1, self._n, self.standard.theta12, None, cdtype=self._cdtype)

    def R13(self) -> torch.Tensor:
        """Standard rotation in the (e, tau) = (0, 2) subspace."""
        return _givens_rotation(0, 2, self._n, self.standard.theta13, None, cdtype=self._cdtype)

    def R23(self) -> torch.Tensor:
        """Standard rotation in the (mu, tau) = (1, 2) subspace."""
        return _givens_rotation(1, 2, self._n, self.standard.theta23, None, cdtype=self._cdtype)

    def Delta(self) -> torch.Tensor:
        """Diagonal CP-phase matrix, phase carried at the tau index (2)."""
        return _phase_diag(2, self._n, self.standard.delta13, cdtype=self._cdtype)

    def _sterile_params(self) -> SterileMixingParameters:
        """The sterile mixing parameters; only called on code paths reached when ``space.n_sterile > 0``."""
        if self.sterile is None:
            raise ValueError("This MixingMatrix has no sterile mixing parameters (space.n_sterile == 0).")
        return self.sterile

    def _sterile_rotation(self, standard_index: int, sterile_index: int) -> torch.Tensor:
        """Rotation mixing one standard flavour with one sterile flavour.

        Args:
            standard_index: 0 (electron), 1 (muon), or 2 (tau).
            sterile_index: Sterile flavour index ``k``,
                ``0 <= k < space.n_sterile``.

        Returns:
            Complex tensor shaped ``(..., n_flavour, n_flavour)``: the
            rotation in the ``(standard_index, 3 + sterile_index)`` subspace.
        """
        theta = {
            0: self._sterile_params().theta_e,
            1: self._sterile_params().theta_mu,
            2: self._sterile_params().theta_tau,
        }[standard_index][..., sterile_index]
        phase = {
            0: self._sterile_params().delta_e,
            1: self._sterile_params().delta_mu,
            2: self._sterile_params().delta_tau,
        }[standard_index][..., sterile_index]
        return _givens_rotation(
            standard_index, 3 + sterile_index, self._n, theta, phase, cdtype=self._cdtype
        )

    def _outer_sterile_block(self) -> Optional[torch.Tensor]:
        """Product of every muon/tau-sterile rotation, or None if space.n_sterile == 0.

        These rotations never touch the electron index (0), so -- exactly
        like R23/Delta -- they commute with the charged-current matter
        potential ``diag(V_CC, 0, ..., 0)`` and belong on the "outer" side
        of the mixing-matrix factorization.

        Composed via ``tree_reduce_matmul`` (O(log N) batched matmul levels)
        rather than a sequential Python loop, so this stays cheap even for a
        large sterile count -- ``space.n_sterile`` is not hardcoded to 3+1
        anywhere in this class, and neither is how these rotations combine.
        """
        if self.space.n_sterile == 0:
            return None

        terms = torch.stack(
            [self._sterile_rotation(1, k) @ self._sterile_rotation(2, k) for k in range(self.space.n_sterile)],
            dim=-3,
        )
        return tree_reduce_matmul(terms)

    def _reduced_sterile_block(self) -> Optional[torch.Tensor]:
        """Product of every electron-sterile rotation, or None if space.n_sterile == 0.

        These rotations touch the electron index (0), so they do not
        commute with the charged-current matter potential and belong on the
        "reduced" side of the factorization instead. See
        ``_outer_sterile_block`` for why ``tree_reduce_matmul`` is used here
        too.
        """
        if self.space.n_sterile == 0:
            return None

        terms = torch.stack(
            [self._sterile_rotation(0, k) for k in range(self.space.n_sterile)], dim=-3
        )
        return tree_reduce_matmul(terms)

    def _sterile_sterile_rotation(self, i: int, j: int, pair_index: int) -> torch.Tensor:
        """Rotation mixing two sterile flavours directly.

        Args:
            i: First sterile flavour index, ``i < j``.
            j: Second sterile flavour index.
            pair_index: This pair's position in ``self.sterile.
                sterile_pairs`` -- indexes ``theta_ss``/``delta_ss``'s own
                last dimension.

        Returns:
            Complex tensor shaped ``(..., n_flavour, n_flavour)``: the
            rotation in the ``(3 + i, 3 + j)`` subspace.
        """
        sterile = self._sterile_params()
        assert sterile.theta_ss is not None and sterile.delta_ss is not None  # n_sterile >= 2 guarantees them
        theta = sterile.theta_ss[..., pair_index]
        phase = sterile.delta_ss[..., pair_index]
        return _givens_rotation(3 + i, 3 + j, self._n, theta, phase, cdtype=self._cdtype)

    def _sterile_sterile_block(self) -> Optional[torch.Tensor]:
        """Product of every sterile-sterile rotation, or None if fewer than 2 sterile flavours.

        These rotations never touch the electron index (0) -- nor, in
        fact, any standard-flavour index at all -- so, exactly like
        R23/Delta/the muon/tau-sterile rotations, they commute with the
        charged-current matter potential ``diag(V_CC, 0, ..., 0)`` and
        belong on the "outer" side of the mixing-matrix factorization.
        Without this block, only ``3 + 3*n_sterile`` of the
        ``(3+n_sterile)(2+n_sterile)/2`` mixing angles a genuinely general
        U(3+n_sterile) parametrization needs are reachable -- exactly
        ``n_sterile*(n_sterile-1)/2`` short, zero only for
        ``n_sterile <= 1`` (see this module's own docstring). See
        ``_outer_sterile_block`` for why ``tree_reduce_matmul`` is used
        here too.
        """
        if self.space.n_sterile < 2:
            return None

        pairs = self._sterile_params().sterile_pairs
        terms = torch.stack(
            [self._sterile_sterile_rotation(i, j, k) for k, (i, j) in enumerate(pairs)],
            dim=-3,
        )
        return tree_reduce_matmul(terms)

    def outer_block(self, antinu: Union[bool, torch.Tensor] = False) -> torch.Tensor:
        """Build the outer mixing block, unaffected by nature.

        ``O = R23 @ Delta @ [muon/tau-sterile rotations] @
        [sterile-sterile rotations]``. Identical for Dirac and Majorana
        neutrinos: the Majorana phase factor lives entirely on the reduced
        side (see ``reduced`` below).

        Args:
            antinu: Bool or boolean tensor selecting the antineutrino
                (complex-conjugated) convention.

        Returns:
            Complex tensor shaped ``(..., n_flavour, n_flavour)``.
        """
        O = self.R23() @ self.Delta()
        sterile_block = self._outer_sterile_block()
        if sterile_block is not None:
            O = O @ sterile_block
        sterile_sterile_block = self._sterile_sterile_block()
        if sterile_sterile_block is not None:
            O = O @ sterile_sterile_block
        return self._select_antinu(O, antinu)

    def reduced(self, antinu: Union[bool, torch.Tensor] = False) -> torch.Tensor:
        """Build the reduced mixing matrix.

        ``U_red = R13 @ R12 @ [electron-sterile rotations] @ [Majorana
        phase factor, if any]``.

        Args:
            antinu: Bool or boolean tensor selecting the antineutrino
                (complex-conjugated) convention.

        Returns:
            Complex tensor shaped ``(..., n_flavour, n_flavour)``.
        """
        Ured = self.R13() @ self.R12()

        sterile_block = self._reduced_sterile_block()
        if sterile_block is not None:
            Ured = Ured @ sterile_block

        phase_matrix = self.nature.phase_matrix(self.space, self.context)
        if phase_matrix is not None:
            Ured = Ured @ phase_matrix

        return self._select_antinu(Ured, antinu)

    def pmns_matrix(self, antinu: Union[bool, torch.Tensor] = False) -> torch.Tensor:
        """Build the full mixing matrix.

        ``U = R23 @ Delta @ [muon/tau-sterile rotations] @
        [sterile-sterile rotations] @ R13 @ Delta^dagger @ R12 @
        [electron-sterile rotations] @ [Majorana phase factor, if any]``.

        Args:
            antinu: Bool or boolean tensor selecting the antineutrino
                (complex-conjugated) convention.

        Returns:
            Complex tensor shaped ``(..., n_flavour, n_flavour)``, unitary.
        """
        delta = self.Delta()
        U = self.R23() @ delta

        outer_sterile = self._outer_sterile_block()
        if outer_sterile is not None:
            U = U @ outer_sterile

        sterile_sterile = self._sterile_sterile_block()
        if sterile_sterile is not None:
            U = U @ sterile_sterile

        U = U @ self.R13() @ delta.conj() @ self.R12()

        reduced_sterile = self._reduced_sterile_block()
        if reduced_sterile is not None:
            U = U @ reduced_sterile

        phase_matrix = self.nature.phase_matrix(self.space, self.context)
        if phase_matrix is not None:
            U = U @ phase_matrix

        return self._select_antinu(U, antinu)

    def pmns_matrix_total(self, antinu: Union[bool, torch.Tensor] = False) -> torch.Tensor:
        """Build the full ``(dim, dim)`` PMNS-like matrix, block-diagonal when helicity-doubled.

        For ``space.n_helicity == 1`` this is exactly ``pmns_matrix(antinu)``
        (``dim == n_flavour``, no doubling). For ``space.n_helicity == 2``,
        mirrors ``tpeanuts.physics.model.PropagationModel._outer_block_total``:
        the primary and companion blocks are each built at their own antinu
        convention (identical for Dirac, opposite for Majorana -- see
        ``Nature.companion_helicity_terms``), combined block-diagonally,
        since a plain ``MixingMatrix`` (with no magnetic-moment extension in
        the picture) has no off-diagonal coupling between the two blocks to
        represent.

        Args:
            antinu: Bool or boolean tensor selecting the primary block's
                antinu convention.

        Returns:
            Complex tensor shaped ``(..., space.dim, space.dim)``, unitary.
        """
        U_primary = self.pmns_matrix(antinu=antinu)
        if not self.space.is_helicity_extended:
            return U_primary

        companion_antinu, _ = self.nature.companion_helicity_terms(antinu)
        U_companion = self.pmns_matrix(antinu=companion_antinu)

        n = self.space.n_flavour
        dim = self.space.dim
        out = torch.zeros(*U_primary.shape[:-2], dim, dim, device=U_primary.device, dtype=U_primary.dtype)
        out[..., :n, :n] = U_primary
        out[..., n:, n:] = U_companion
        return out

    def _select_antinu(
        self,
        U: torch.Tensor,
        antinu: Union[bool, torch.Tensor],
    ) -> torch.Tensor:
        """Select the neutrino or antineutrino convention for a mixing matrix.

        A thin wrapper around the project-wide
        ``tpeanuts.util.tensor.select_for_antinu`` primitive.

        Args:
            U: Full or reduced mixing matrix, shaped ``(..., n, n)``.
            antinu: Boolean scalar or tensor mask. True selects ``U.conj()``.

        Returns:
            ``U`` for neutrinos, its complex conjugate for antineutrinos,
            broadcasting over any batch dimensions ``antinu`` carries.
        """
        return select_for_antinu(antinu, U, U.conj())

    def jarlskog_invariant(
        self,
        alpha: int = 0,
        beta: int = 1,
        i: int = 0,
        j: int = 1,
    ) -> torch.Tensor:
        """Compute one rephasing-invariant CP invariant of the ``J_{alpha,beta}^{i,j}`` family.

        Uses the general definition

        ``J_{alpha,beta}^{i,j} = Im[U_{alpha,i}  U_{beta,j} U*_{alpha,j} U*_{beta,i}]``

        exactly invariant under any independent rephasing of each flavour row and each
        mass-eigenstate column
        (every phase such a rephasing introduces cancels pairwise in this combination,
        for any ``alpha != beta``, ``i != j``, and any ``space.n_flavour``).

        For the plain 3-flavour case (``space.n_sterile == 0``), unitarity
        ties every choice of ``(alpha, beta, i, j)`` (with ``alpha != beta``,
        ``i != j``) to the same value up to an overall sign -- there is
        essentially one Jarlskog invariant, and the default arguments
        (``alpha=e``, ``beta=mu``, ``i=1``, ``j=2``, 0-indexed here as
        ``0, 1, 0, 1``) reproduce the textbook one exactly.

        This stops being true once ``space.n_sterile > 0``: the number of
        physical CP-violating phases grows to ``(n_flavour-1)(n_flavour-2)/2``
        (1 for ``n_flavour=3``, but already 3 for ``n_flavour=4``), so a
        single invariant can no longer capture all of it -- several
        algebraically *independent* members of this family coexist. This
        method computes exactly the one selected by ``(alpha, beta, i, j)``;
        it does not attempt to enumerate or return the complete independent
        set.

        Unaffected by nature for any choice of indices: a Majorana phase
        factor is diagonal, so it cancels in this combination exactly as
        the sign convention of any other rephasing does.

        Args:
            alpha: First flavour-row index (0-indexed into the full
                ``n_flavour x n_flavour`` matrix). Defaults to the electron
                row.
            beta: Second flavour-row index, ``beta != alpha``. Defaults to
                the muon row.
            i: First mass-eigenstate-column index. Defaults to 0.
            j: Second mass-eigenstate-column index, ``j != i``. Defaults to
                1.

        Returns:
            Real tensor, the batch shape of the underlying mixing
            parameters.
        """
        U = self.pmns_matrix()
        return torch.imag(
            U[..., alpha, i] * U[..., beta, j] * U[..., alpha, j].conj() * U[..., beta, i].conj()
        )
