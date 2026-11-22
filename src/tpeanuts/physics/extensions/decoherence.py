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

"""Mass-basis pure dephasing: the simplest wave-packet-style decoherence model.

Wave-packet separation (different mass eigenstates travel at slightly
different group velocities, so their wave packets stop overlapping over a
long enough baseline) damps *coherences* between mass eigenstates without
directly changing their *populations* -- "pure dephasing". The simplest
Lindblad realization of this is one jump operator per mass eigenstate,
proportional to the projector onto it:

    L_k = sqrt(gamma_k) |k><k|

Summing ``dissipator_superoperator(L_k)`` (see
``tpeanuts.physics.evolution.liouville.open_system``) over every ``k`` gives, in the
mass basis, exactly

    d(rho_ii)/dt = 0
    d(rho_ij)/dt = -(gamma_i + gamma_j)/2 * rho_ij   (i != j)

-- populations exactly conserved, every coherence damped at the *average* of
its two eigenstates' own rates (derived directly from the projector algebra,
not asserted; see this module's tests for the closed-form verification). A
single scalar ``gamma`` (every eigenstate the same rate) recovers the
textbook uniform-dephasing model exactly.

This is deliberately the simplest physically consistent decoherence model,
not the most realistic one (a genuine wave-packet-overlap model has
``gamma`` depend on energy, baseline, and each ``Delta m^2_ij`` through the
wave-packet width) -- it exists to validate the ``DissipatorTerm``
machinery end to end with a concrete, exactly-solvable example, the same
role ``tpeanuts.physics.extensions.magnetic_moment``'s first concrete
matrix played for the magnetic-moment work.

Scope note (helicity doubling): ``DecoherenceTerm`` builds up to *two* independent
sets of Lindblad operators, one per helicity block -- ``gamma`` for the
primary block (via ``tpeanuts.physics.common.terms.embed_primary_block``,
exactly like an ordinary ``GeneratorTerm``'s own Hamiltonian contribution),
and, only when explicitly supplied, ``gamma_companion`` for the companion
block (via ``embed_companion_block``, at the companion's own antinu
convention from ``Nature.companion_helicity_terms`` -- the same rotation
``tpeanuts.physics.model.PropagationModel._companion_generator_reduced``
already uses for the companion block's own Hamiltonian). The two sets are
built completely independently: neither block's rate affects the other's
operators, and there is deliberately no jump operator mixing the two blocks
-- this is "independent decoherence per helicity sector", one of several
equally defensible physical models (the others being primary-only,
correlated cross-sector decoherence, or helicity-mixing jump operators);
this project picked this one as the natural generalization to implement
first because it reuses every piece the magnetic-moment work already built,
introduces no new speculative physics, and reduces to today's already-
validated primary-only behaviour exactly when ``gamma_companion`` is left
at its default.

``gamma_companion=None`` (the default) means "the companion block's
decoherence has not been decided" -- for a helicity-doubled space
(``space.n_helicity == 2``), that is rejected outright by ``validate``
rather than silently reproducing the primary-only behaviour (which is
itself internally consistent -- a coherence touching the primary block
still decays even though the companion-companion block is individually
untouched, a direct algebraic consequence of the Lindblad structure, not a
bug -- but is exactly the sort of unstated choice this project does not
make silently; see ``tpeanuts.physics.extensions.sterile.SterileNCTerm.
validate`` for the same "fail fast rather than silently reproduce an
unstated default" discipline applied to its own analogous scope
limitation, ``space.n_sterile == 0``). Passing ``gamma_companion``
explicitly -- even ``0.0``, to deliberately keep the companion block
undecohered -- is what unlocks a helicity-doubled space.

Module contents:
    DecoherenceConfig
        Frozen dataclass: per-mass-eigenstate rates ``gamma`` (primary block) and ``gamma_companion`` (companion
        block, optional), on a ``RuntimeContext`` -- the data, like ``NSIConfig`` and ``MagneticMomentConfig``.
    DecoherenceTerm
        A ``tpeanuts.physics.evolution.liouville.open_system.DissipatorTerm``-conforming dissipator built from a
        ``DecoherenceConfig`` (``validate`` and ``lindblad_operators``) -- the term, like ``NSITerm`` and
        ``MagneticMomentTerm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.common.terms import (
    PropagationEnvironment,
    embed_companion_block,
    embed_primary_block,
)
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor, cdtype_from_real


@dataclass(frozen=True)
class DecoherenceConfig:
    """Rates of pure dephasing of mass-eigenstate coherences, one per mass eigenstate per helicity block.

    Attributes:
        gamma: Per-mass-eigenstate decoherence rate for the primary
            helicity block, in units of inverse metres, shaped
            ``(..., n_flavour)`` (or broadcastable to it, so a single
            scalar reproduces the uniform-rate limit). The damping rate for
            a given pair of mass eigenstates ``(i, j)`` is
            ``(gamma_i + gamma_j) / 2`` -- see this module's own docstring.
        context: RuntimeContext ``gamma``/``gamma_companion`` are stored/cast
            on.
        gamma_companion: The same, for the companion helicity block. ``None``
            (the default) means the companion block is left completely
            undecohered by this term -- valid on its own for a plain
            (``space.n_helicity == 1``) space, but rejected by ``validate``
            on a helicity-doubled one (see this module's "Scope note").

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.physics.extensions import DecoherenceConfig
        >>> gamma = torch.tensor(1e-12, dtype=torch.float64)   # one rate for every mass state, in 1/m
        >>> config = DecoherenceConfig(gamma=gamma, context=context)
        >>> float(config.gamma), config.gamma_companion is None
        (1e-12, True)
    """

    gamma: torch.Tensor
    context: RuntimeContext
    gamma_companion: Optional[torch.Tensor] = None

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            gamma: TensorLike,
            context: RuntimeContext,
            gamma_companion: Optional[TensorLike] = ...,
        ) -> None: ...


@dataclass(frozen=True)
class DecoherenceTerm:
    """Mass-basis pure-dephasing Lindblad dissipator, built from a ``DecoherenceConfig``.

    A ``tpeanuts.physics.evolution.liouville.open_system.DissipatorTerm``: ``validate`` and ``lindblad_operators``
    (one operator ``sqrt(rate_k) |k><k|`` per mass eigenstate and active helicity block); pass it in
    ``PropagationModel(dissipators=(...))``.

    Attributes:
        config: The DecoherenceConfig carrying ``gamma``/``gamma_companion`` and the RuntimeContext they live on.

    Examples:
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.physics.extensions import DecoherenceConfig, DecoherenceTerm
        >>> gamma = torch.tensor(1e-12, dtype=torch.float64)   # one rate for every mass state, in 1/m
        >>> term = DecoherenceTerm(DecoherenceConfig(gamma=gamma, context=context))
        >>> float(term.config.gamma)
        1e-12
    """

    config: DecoherenceConfig

    @staticmethod
    def _validate_rate(rate: torch.Tensor, *, n: int, name: str) -> None:
        """Shared shape/finite/non-negative validation for ``gamma``/``gamma_companion``.

        Args:
            rate: Already-cast rate tensor to validate.
            n: ``mixing.space.n_flavour``, the expected trailing dimension.
            name: Field name, used only to point error messages correctly.

        Raises:
            ValueError: If ``rate``'s last dimension is neither absent, 1,
                nor ``n``, if it contains non-finite (NaN/Inf) entries, or
                if any entry is negative -- the GKLS/Lindblad CPTP guarantee
                (trace, Hermiticity, and positive semi-definiteness
                preserved for any evolution time) only holds for
                non-negative decoherence rates; a negative one is not a
                milder decoherence, it is a different, invalid operator
                entirely (``sqrt(rate)`` would silently produce NaN
                Lindblad operators otherwise, rather than rejecting the
                invalid configuration up front).
        """
        if rate.shape[-1:] not in ((), (1,), (n,)):
            raise ValueError(
                f"{name}'s last dimension must be absent, 1, or {n} "
                f"(mixing.space.n_flavour), got shape {tuple(rate.shape)}."
            )
        if not torch.isfinite(rate).all():
            raise ValueError(f"{name} contains non-finite (NaN/Inf) entries.")
        if torch.any(rate < 0):
            raise ValueError(
                f"{name} must be non-negative everywhere: a negative "
                "decoherence rate is not a physical Lindblad channel (the "
                "GKLS CPTP guarantee requires it >= 0)."
            )

    def validate(self, mixing: MixingMatrix, environment: PropagationEnvironment) -> None:
        """Check both rates' shape/physicality, and that a helicity-doubled space has decided its companion rate.

        Args:
            mixing: The MixingMatrix the Liouvillian is being built for.
            environment: Unused -- this dissipator needs no medium data.

        Raises:
            ValueError: If ``gamma`` (see ``_validate_rate``) is invalid; if
                ``gamma_companion is None`` and ``mixing.space.
                n_helicity == 2`` (an undecided companion rate on a
                helicity-doubled space, see this module's "Scope note"); if
                ``gamma_companion`` is given but
                ``mixing.space.n_helicity == 1`` (no companion block
                exists to decohere); or if ``gamma_companion`` itself is
                invalid.
        """
        n = mixing.space.n_flavour
        gamma = as_tensor(self.config.gamma, device=self.config.context.device, dtype=self.config.context.dtype)
        self._validate_rate(gamma, n=n, name="gamma")

        if self.config.gamma_companion is None:
            if mixing.space.is_helicity_extended:
                raise ValueError(
                    "DecoherenceTerm's companion helicity block would "
                    "otherwise receive no decoherence of its own while its "
                    "cross-coherences with the primary block are still "
                    "damped (see this module's 'Scope note' docstring) -- "
                    "an unstated asymmetric default. Pass gamma_companion "
                    "explicitly (even 0.0, to deliberately keep the "
                    "companion block undecohered) to use this term on a "
                    "helicity-doubled space."
                )
            return

        if not mixing.space.is_helicity_extended:
            raise ValueError(
                "gamma_companion was given but mixing.space.n_helicity == 1: "
                "there is no companion helicity block to decohere."
            )
        gamma_companion = as_tensor(
            self.config.gamma_companion, device=self.config.context.device, dtype=self.config.context.dtype
        )
        self._validate_rate(gamma_companion, n=n, name="gamma_companion")

    def lindblad_operators(
        self,
        mixing: MixingMatrix,
        environment: PropagationEnvironment,
        *,
        antinu: Union[bool, torch.Tensor] = False,
        evolution_scale_m: TensorLike,
    ) -> Tuple[torch.Tensor, ...]:
        """Build one Lindblad operator ``sqrt(rate_k) |k><k|`` per mass eigenstate, per active helicity block.

        The mass-eigenstate projector ``|k><k|``, expressed in a given
        block's own reduced basis, is the outer product of
        ``mixing.reduced(block_antinu)``'s own ``k``-th column with itself
        -- exactly the same rotation ``kinetic_term_reduced`` uses to
        diagonalize that block's own kinetic term, since "mass basis" and
        "the basis the kinetic term is diagonal in" are the same thing by
        definition. The primary block uses ``antinu`` directly; the
        companion block (only built when ``self.config.gamma_companion is not
        None``) uses ``mixing.nature.companion_helicity_terms(antinu)``'s
        own antinu convention instead -- identical to ``antinu`` for Dirac,
        the opposite for Majorana, exactly mirroring
        ``PropagationModel._companion_generator_reduced``'s own logic for
        the companion block's Hamiltonian.

        Args:
            mixing: Built MixingMatrix.
            environment: Unused -- this dissipator needs no medium data.
            antinu: Bool or boolean tensor selecting the primary block's
                antinu convention.
            evolution_scale_m: Positive length scale in metres, matching
                whatever every other active term (Hamiltonian or
                dissipator) uses -- each rate (in 1/metres) is made
                dimensionless by multiplying it, the same
                non-dimensionalization convention every physical rate in
                this project follows.

        Returns:
            Tuple of complex tensors, each shaped
            ``(..., mixing.space.dim, mixing.space.dim)`` -- ``n_flavour``
            of them if ``gamma_companion is None``, ``2 * n_flavour`` if a
            companion rate was also given.
        """
        self.validate(mixing, environment)

        context = mixing.context
        n = mixing.space.n_flavour
        cdtype = cdtype_from_real(context.dtype)
        scale = as_tensor(evolution_scale_m, device=context.device, dtype=context.dtype)

        def _block_operators(rate_value: TensorLike, block_antinu, embed) -> List[torch.Tensor]:
            rate = as_tensor(rate_value, device=context.device, dtype=context.dtype)
            rate_dimensionless = rate * scale
            sqrt_rate = torch.sqrt(rate_dimensionless).to(dtype=cdtype)
            # Broadcast a scalar or single-entry rate up to exactly n
            # entries via ordinary tensor broadcasting, rather than
            # branching on sqrt_rate's own shape by hand (which mishandles
            # the plain-scalar case: a 0-dim tensor has no last dimension
            # to inspect at all).
            sqrt_rate = sqrt_rate + torch.zeros(n, device=context.device, dtype=cdtype)

            U_red = mixing.reduced(antinu=block_antinu)
            block_ops = []
            for k in range(n):
                column = U_red[..., :, k : k + 1]
                projector = column @ column.conj().transpose(-2, -1)
                L_k = sqrt_rate[..., k][..., None, None] * projector
                block_ops.append(embed(L_k, mixing.space))
            return block_ops

        operators = _block_operators(self.config.gamma, antinu, embed_primary_block)
        if self.config.gamma_companion is not None:
            companion_antinu, _ = mixing.nature.companion_helicity_terms(antinu)
            operators += _block_operators(self.config.gamma_companion, companion_antinu, embed_companion_block)
        return tuple(operators)
