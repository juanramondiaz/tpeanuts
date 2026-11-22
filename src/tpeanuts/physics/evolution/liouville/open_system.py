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

"""Liouville-space (open-system) evolution: the pieces built so far.

Unitary evolution (``tpeanuts.physics.evolution.numerical.numerical``) propagates
``rho`` via ``rho' = U rho U^dagger`` -- a single "sandwiching" operator,
exactly what ``tpeanuts.physics.common.state.DensityMatrix.conjugate_by``
computes. Genuine decoherence (a Lindblad/GKLS dissipator) has no such
factorization: the general master equation

    drho/dt = -i[H, rho] + sum_k ( L_k rho L_k^dagger
                                    - (1/2){L_k^dagger L_k, rho} )

only becomes an ordinary matrix acting by left-multiplication once ``rho``
is *vectorized* (``DensityMatrix.vectorize``) into a ``dim*dim``-dimensional
vector -- every term above then becomes a ``(dim*dim, dim*dim)``
"superoperator" built from Kronecker products, and the whole right-hand side
is one combined superoperator (the "Liouvillian") whose ``matrix_exp`` plays
exactly the role ``tpeanuts.physics.evolution.numerical.numerical.segment_evolutor_numerical``
already plays for the unitary-only case -- see
``tpeanuts.physics.common.state.DensityMatrix.apply_superoperator``.

Convention note (repeated here because getting it backwards is a genuine,
easy-to-miss physics bug, not a style choice): every superoperator in this
module is built for ``DensityMatrix.vectorize``'s *row-major* convention
(``vec_r(A X B) = (A kron B^T) vec_r(X)``), not the column-major "vec"
convention most linear-algebra references use. See ``vectorize``'s own
docstring for the derivation and the test that independently verifies it
(``tpeanuts.physics.common.test.test_state.
test_apply_superoperator_matches_conjugate_by_for_unitary_case``).

A dissipator is specified in terms of Lindblad (jump) operators ``L_k``, not
a raw superoperator: that is the natural physical object a decoherence
model is written down in terms of (see any concrete
``DissipatorTerm``-conforming class, e.g. a future mass-basis wave-packet
decoherence model), and it lets ``dissipator_superoperator`` below -- the
Kronecker-product conversion from ``L_k`` to its own
``(dim*dim, dim*dim)`` contribution -- live in exactly one place rather
than being reimplemented by every concrete dissipator, mirroring why
``commutator_superoperator`` itself exists as a shared primitive rather
than something every consumer builds by hand.

Liouvillian assembly itself lives on
``tpeanuts.physics.model.PropagationModel`` (``liouvillian_reduced``/
``liouvillian_flavour``), mirroring how ``generator_reduced``/
``generator_flavour`` assemble the unitary-only generator -- not here, since
assembly needs ``MixingMatrix``/``MassSpectrum``/``extensions`` together,
which this module deliberately does not depend on. What *does* live here is
the Liouville-space evolutor (``segment_evolutor_liouville``/
``evolve_path_liouville``), the open-system counterpart of
``tpeanuts.physics.evolution.numerical.numerical.segment_evolutor_numerical``/``evolve_path_numerical``.

Module contents:
    commutator_superoperator(...)
        The Liouville-space superoperator representing ``-i[H, .]``, built
        from any Hermitian generator already produced by
        ``tpeanuts.physics``'s unitary machinery (``PropagationModel.
        generator_reduced``/``generator_flavour``, or the bare
        ``tpeanuts.physics.common.generator`` functions).
    dissipator_superoperator(...)
        The Liouville-space superoperator representing ``L rho L^dagger -
        (1/2){L^dagger L, rho}`` for a single Lindblad operator ``L``.
    DissipatorTerm
        The structural Protocol every pluggable Lindblad dissipator
        follows: ``validate(...)``, ``lindblad_operators(...)``. Deliberately
        simpler than ``tpeanuts.physics.common.terms.GeneratorTerm`` -- no
        ``kind``/``REPLACES`` distinction, since every dissipator is purely
        additive (there is no "base dissipator" to ever supersede; the
        no-decoherence limit is simply zero active dissipators, not a
        default one being replaced).
    segment_evolutor_liouville(...)
        Build ``matrix_exp(liouvillian * dx)`` for one segment (or an
        entire single-segment path) -- the open-system counterpart of
        ``tpeanuts.physics.evolution.numerical.numerical.segment_evolutor_numerical``.
    evolve_path_liouville(...)
        Build and compose the per-segment Liouville-space evolutors for a
        whole path into one total evolution superoperator, via
        ``tpeanuts.physics.common.composition.compose_segment_evolutors``
        unchanged -- it is dimension-agnostic by design (see its own module
        docstring), so no Liouville-specific composition logic is needed.
"""

from __future__ import annotations

from typing import Protocol, Tuple, Union, runtime_checkable

import torch

from tpeanuts.physics.common.composition import compose_segment_evolutors
from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.util.tensor import TensorLike, as_tensor, kron_batched


def commutator_superoperator(H: torch.Tensor) -> torch.Tensor:
    """Build the Liouville-space superoperator representing ``-i[H, .]``.

    Under ``DensityMatrix.vectorize``'s row-major convention,
    ``vec_r(-i[H, rho]) = -i(H kron I - I kron H^T) vec_r(rho)`` -- derived
    from ``vec_r(A X B) = (A kron B^T) vec_r(X)`` applied to
    ``H rho I`` and ``I rho H`` separately (see this module's docstring).
    This is *not* the Kronecker order a naive column-major "vec" adaptation
    would give (``H^T kron I - I kron H``); using the wrong one here would
    silently mismatch every other superoperator built from ``vectorize``.

    Uses ``kron_batched`` rather than ``torch.kron`` directly:
    ``torch.kron`` has no real notion of a leading batch dimension (see
    ``kron_batched``'s own docstring), which would only matter here if
    ``H``'s own batch size happened to collide with the multiplication --
    it does not for this function specifically (``identity`` is always
    unbatched), but using the correctly-batched helper throughout is what
    keeps that true by construction rather than by accident.

    Args:
        H: Complex Hamiltonian-like generator, shaped ``(..., dim, dim)``
            -- any already-built unitary generator (does not need to be
            Hermitian for this function itself to work, though every
            physical generator in this project is).

    Returns:
        Complex tensor shaped ``(..., dim*dim, dim*dim)``.

    Raises:
        ValueError: If ``H``'s final two dimensions are not equal (square).
    """
    if H.shape[-1] != H.shape[-2]:
        raise ValueError(
            f"H must be square in its final two dimensions, got {tuple(H.shape[-2:])}."
        )

    dim = H.shape[-1]
    identity = torch.eye(dim, device=H.device, dtype=H.dtype)
    phase = torch.tensor(-1j, device=H.device, dtype=H.dtype)
    H_T = H.transpose(-2, -1)
    return phase * (kron_batched(H, identity) - kron_batched(identity, H_T))


def dissipator_superoperator(L: torch.Tensor) -> torch.Tensor:
    """Build the Liouville-space superoperator for one Lindblad (jump) operator ``L``.

    Represents ``D[L](rho) = L rho L^dagger - (1/2){L^dagger L, rho}``, the
    per-channel piece of the Lindblad/GKLS master equation (see this
    module's docstring). Under ``DensityMatrix.vectorize``'s row-major
    convention (derived the same way ``commutator_superoperator`` derives
    its own Kronecker order):

        vec_r(D[L](rho)) = (L kron L* - (1/2)(L^dagger L kron I)
                             - (1/2)(I kron (L^dagger L)^T)) vec_r(rho)

    Args:
        L: Complex Lindblad (jump) operator, shaped ``(..., dim, dim)``.

    Returns:
        Complex tensor shaped ``(..., dim*dim, dim*dim)``.

    Raises:
        ValueError: If ``L``'s final two dimensions are not equal (square).
    """
    if L.shape[-1] != L.shape[-2]:
        raise ValueError(
            f"L must be square in its final two dimensions, got {tuple(L.shape[-2:])}."
        )

    dim = L.shape[-1]
    identity = torch.eye(dim, device=L.device, dtype=L.dtype)
    L_dagger_L = L.conj().transpose(-2, -1) @ L
    L_dagger_L_T = L_dagger_L.transpose(-2, -1)
    half = torch.tensor(0.5, device=L.device, dtype=L.dtype)

    # kron_batched, not torch.kron directly: L kron L.conj() below has both
    # operands carrying L's own batch dimension, exactly the case
    # torch.kron itself gets wrong (see kron_batched's own docstring) --
    # this is the case that actually surfaces the bug, not a theoretical one.
    return (
        kron_batched(L, L.conj())
        - half * kron_batched(L_dagger_L, identity)
        - half * kron_batched(identity, L_dagger_L_T)
    )


@runtime_checkable
class DissipatorTerm(Protocol):
    """Structural contract every pluggable Lindblad dissipator follows.

    No inheritance required -- any object exposing these members satisfies
    it (``isinstance(obj, DissipatorTerm)`` works, via ``runtime_checkable``,
    though it only checks that the members exist, not their exact
    signatures). Deliberately simpler than
    ``tpeanuts.physics.common.terms.GeneratorTerm``: no ``kind``/``replaces``
    distinction, since a dissipator is always purely additive -- there is no
    "base dissipator" one could ever need to supersede, unlike the
    charged-current term NSI replaces. The no-decoherence limit is simply
    zero active dissipators, not a default one being replaced.
    """

    def validate(self, mixing: MixingMatrix, environment: PropagationEnvironment) -> None:
        """Raise ValueError if this term cannot be evaluated as configured.

        Called once per Liouvillian assembly, before any term's
        ``lindblad_operators`` runs, so a missing environment field (or an
        incompatible ``mixing``) fails fast rather than partway through
        assembling the Liouvillian -- mirrors
        ``GeneratorTerm.validate``'s own role exactly.

        Args:
            mixing: The MixingMatrix the Liouvillian is being built for.
            environment: The PropagationEnvironment this call was given.

        Returns:
            None.

        Raises:
            ValueError: If this term's requirements are not met.
        """
        ...

    def lindblad_operators(
        self,
        mixing: MixingMatrix,
        environment: PropagationEnvironment,
        *,
        antinu: Union[bool, torch.Tensor],
        evolution_scale_m: TensorLike,
    ) -> Tuple[torch.Tensor, ...]:
        """Return this term's Lindblad (jump) operators, in the reduced basis.

        Each operator is later converted to its own Liouville-space
        contribution via ``dissipator_superoperator`` and summed into the
        total Liouvillian -- mirroring how ``GeneratorTerm.
        contribute_reduced`` returns a reduced-basis Hamiltonian
        contribution summed into the total generator. Returning the raw
        operators rather than each term's own already-converted
        superoperator keeps the Kronecker-product conversion itself in
        exactly one place (``dissipator_superoperator``), not reimplemented
        per dissipator.

        Args:
            mixing: The MixingMatrix the Liouvillian is being built for.
            environment: Neutrino/medium data (only the fields this term
                actually needs are read).
            antinu: Bool or boolean tensor selecting the antineutrino
                convention.
            evolution_scale_m: Positive length scale in metres, matching
                whatever every other active term (Hamiltonian or
                dissipator) uses.

        Returns:
            Tuple of complex tensors, each shaped
            ``(..., mixing.space.dim, mixing.space.dim)`` -- following
            ``GeneratorTerm.contribute_reduced``'s own convention, a term
            built from ordinary flavour physics calls
            ``tpeanuts.physics.common.terms.embed_primary_block`` on each
            operator as its own last step.
        """
        ...


def segment_evolutor_liouville(liouvillian: torch.Tensor, dx: TensorLike) -> torch.Tensor:
    """Build the exact Liouville-space evolution superoperator for one segment.

    The open-system counterpart of
    ``tpeanuts.physics.evolution.numerical.numerical.segment_evolutor_numerical``:
    ``matrix_exp(liouvillian * dx)`` plays exactly the same role for
    ``vec_r(rho)`` that ``matrix_exp(-i H dx)`` plays for ``rho`` itself in
    the purely unitary case -- the ``-i`` factor is already baked into
    ``commutator_superoperator``, so this function does not add one of its
    own (unlike ``segment_evolutor_numerical``, which does).

    Args:
        liouvillian: Complex Liouvillian superoperator, shaped
            ``(..., dim*dim, dim*dim)`` -- e.g.
            ``tpeanuts.physics.model.PropagationModel.liouvillian_reduced``/
            ``liouvillian_flavour``'s output.
        dx: Dimensionless path length(s), broadcastable against
            ``liouvillian``'s batch shape -- see ``segment_evolutor_numerical``'s own
            docstring for what "dimensionless" requires.

    Returns:
        Complex tensor shaped ``(..., dim*dim, dim*dim)``:
        ``matrix_exp(liouvillian * dx)``. Unlike the purely unitary
        ``segment_evolutor_numerical``'s output, this is generally *not* unitary once
        any dissipator is active (a genuine CPTP map instead); it reduces
        to exactly ``U kron U.conj()`` (a unitary-conjugation superoperator,
        ``U = segment_evolutor_numerical(H, dx)``) whenever ``liouvillian`` has no
        dissipative contribution -- see
        ``test_commutator_superoperator_exponentiates_to_unitary_conjugation``.
    """
    dx_t = as_tensor(dx, device=liouvillian.device, dtype=liouvillian.real.dtype)
    phase = dx_t.to(dtype=liouvillian.dtype)[..., None, None] * liouvillian
    return torch.linalg.matrix_exp(phase)


def evolve_path_liouville(
    liouvillian_segments: torch.Tensor,
    dx_segments: TensorLike,
    *,
    reverse: bool = False,
) -> torch.Tensor:
    """Build and compose the per-segment Liouville-space evolutors for a whole path.

    The open-system counterpart of
    ``tpeanuts.physics.evolution.numerical.numerical.evolve_path_numerical`` -- composes via
    ``tpeanuts.physics.common.composition.compose_segment_evolutors``
    unchanged, since it is dimension-agnostic by design (see its own module
    docstring: exactly the same tree-reduction whether the operators being
    composed are ``(dim, dim)`` unitary evolutors or ``(dim*dim, dim*dim)``
    Liouville-space ones).

    Args:
        liouvillian_segments: Liouvillian superoperators, one per segment,
            shaped ``(..., N, dim*dim, dim*dim)``.
        dx_segments: Dimensionless segment lengths, shaped ``(..., N)`` (or
            broadcastable against it) -- see ``segment_evolutor_liouville``
            for what "dimensionless" requires.
        reverse: Forwarded to ``compose_segment_evolutors``: if False
            (default), segment 0 is the first one a state passes through.

    Returns:
        Complex tensor shaped ``(..., dim*dim, dim*dim)``, the total
        evolution superoperator for the whole path.
    """
    S_segments = segment_evolutor_liouville(liouvillian_segments, dx_segments)
    return compose_segment_evolutors(S_segments, reverse=reverse)
