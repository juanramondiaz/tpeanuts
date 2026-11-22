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

"""Transition magnetic moment: the off-diagonal helicity coupling.

Only meaningful for a helicity-extended (``space.n_helicity == 2``) model.

While ``tpeanuts.physics.model.PropagationModel``'s own
``_companion_generator_reduced`` builds each helicity block's own *diagonal*
physics (kinetic, matter potential, NSI, sterile NC -- everything that does
not flip helicity), this term builds the *coupling* between the two blocks:
the off-diagonal piece that genuinely requires a doubled state space in the
first place.

In block form,

    H_full = [[H_primary,        mu^dagger * s],
              [mu * s,           H_companion  ]]

where ``s`` is the dimensionless magnetic-moment scale factor built from
the magnetic field:

``tpeanuts.physics.common.potential.magnetic_moment_scale``

and ``mu`` is the transition moment matrix, physically defined in the
flavour basis (mapping a primary-block flavour state to a companion-block
flavour state) exactly like NSI's ``epsilon`` or the sterile term's projector.

Basis note: unlike every other extension in this project, the two sides of
``mu``'s coupling live in *different* reduced bases
-- the primary block's own (built from ``mixing.outer_block(antinu=antinu)``)
-- the companion block's own (built from
    ``mixing.outer_block(antinu=companion_antinu)``, which differs from the
    primary one whenever nature is Majorana, since
    ``Nature.companion_helicity_terms`` then flips ``antinu``).

So this term cannot reuse

``tpeanuts.physics.common.generator.rotate_to_reduced``

(built for a single shared ``O``) as every other extension does -- it rotates
each side of ``mu`` by its own block's outer mixing matrix directly. For Dirac,
where the companion's antinu equals the primary's, the two outer blocks
coincide and this collapses to exactly the single-``O`` case.

Module contents:
    MagneticMomentConfig
        Frozen dataclass storing the transition moment matrix ``mu``
        (dimensionless, in units of the Bohr magneton), plus nature-aware
        validation (Majorana requires ``mu`` antisymmetric with zero
        diagonal -- CPT forbids a Majorana state coupling to itself).
    MagneticMomentTerm
        ``tpeanuts.physics.common.terms.GeneratorTerm`` built from a
        ``MagneticMomentConfig`` -- ``kind = TermKind.ADDITIVE``,
        ``replaces = None`` (it only ever couples the primary/companion
        diagonal blocks, never overlaps their own contributions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch

from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.common.potential import magnetic_moment_scale
from tpeanuts.physics.common.terms import PropagationEnvironment, TermKind
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import (
    TensorLike,
    as_tensor,
    cdtype_from_real,
    embed_square_block,
    transpose_symmetry_residual,
)


@dataclass(frozen=True)
class MagneticMomentConfig:
    """Transition magnetic-moment matrix ``mu``, in the flavour basis and Bohr-magneton units.

    Attributes:
        mu: Complex tensor, final two dimensions either
            ``(mixing.space.n_flavour, mixing.space.n_flavour)`` or a
            smaller top-left standard/sterile block (embedded with zeros on
            the rest, mirroring ``tpeanuts.physics.extensions.nsi.
            NSIConfig.epsilon_tensor``), expressed as a dimensionless
            multiple of the Bohr magneton -- see ``tpeanuts.physics.common.
            potential.magnetic_moment_scale``. ``mu[i, j]`` couples the
            companion-block flavour state ``i`` to the primary-block
            flavour state ``j``.
        context: RuntimeContext ``mu`` is stored/cast on.

    Raises:
        ValueError: If ``mu`` is not square, or contains non-finite entries.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.physics.extensions import MagneticMomentConfig
        >>> mu = torch.zeros(3, 3, dtype=torch.complex128)
        >>> mu[0, 1] = 1e-11   # transition moment, in Bohr magnetons
        >>> config = MagneticMomentConfig(mu=mu, context=context)
        >>> config.mu_tensor(n_flavour=3).shape
        torch.Size([3, 3])
    """

    mu: torch.Tensor
    context: RuntimeContext

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            mu: TensorLike,
            context: RuntimeContext,
        ) -> None: ...

    def __post_init__(self) -> None:
        cdtype = cdtype_from_real(self.context.dtype)
        mu = as_tensor(self.mu, device=self.context.device, dtype=cdtype)
        if mu.shape[-1] != mu.shape[-2]:
            raise ValueError(f"mu must be square, got final dimensions {tuple(mu.shape[-2:])}.")
        if not torch.isfinite(mu).all():
            raise ValueError("mu contains non-finite (NaN/Inf) entries.")
        object.__setattr__(self, "mu", mu)

    def mu_tensor(self, *, n_flavour: int) -> torch.Tensor:
        """Embed ``self.mu`` for a mixing matrix with ``n_flavour`` flavours.

        Args:
            n_flavour: Total number of flavours of the target generator
                (``mixing.space.n_flavour``).

        Returns:
            Complex tensor shaped ``(..., n_flavour, n_flavour)``: unchanged
            if already that shape, or embedded in the top-left corner of an
            ``(n_flavour, n_flavour)`` zero matrix otherwise -- via
            ``tpeanuts.util.tensor.embed_square_block``.

        Raises:
            ValueError: If ``self.mu``'s shape is incompatible with
                ``n_flavour`` (larger than it, and not already matching).
        """
        try:
            return embed_square_block(self.mu, n_flavour)
        except ValueError as exc:
            raise ValueError(
                f"mu must have final dimensions ({n_flavour}, {n_flavour}) or a "
                f"smaller top-left block; got {tuple(self.mu.shape[-2:])}."
            ) from exc

    def validate_for_nature(self, *, is_majorana: bool, n_flavour: int) -> None:
        """Check ``mu``'s nature-dependent structure.

        Dirac leaves ``mu`` fully unconstrained (``nu_L``/``nu_R`` are
        independent fields, so nothing relates ``mu_ij`` to ``mu_ji`` -- see
        ``tpeanuts.physics.common.nature``'s module docstring). Majorana
        requires ``mu`` antisymmetric (``mu_ij = -mu_ji``), which forces a
        zero diagonal automatically (``mu_ii = -mu_ii`` implies
        ``mu_ii = 0``) -- CPT forbids a Majorana state coupling to itself,
        only transition moments between distinct massive states survive.

        Args:
            is_majorana: Whether the owning model's nature is Majorana.
            n_flavour: Target flavour count (see ``mu_tensor``).

        Raises:
            ValueError: If ``is_majorana`` and ``mu`` is not antisymmetric
                within tolerance.
        """
        if not is_majorana:
            return

        mu = self.mu_tensor(n_flavour=n_flavour)
        residual, tolerance = transpose_symmetry_residual(mu, sign=-1.0, conjugate=False)
        if residual > tolerance:
            raise ValueError(
                "Majorana nature requires mu antisymmetric (mu_ij = -mu_ji, "
                "zero diagonal -- CPT forbids a Majorana state coupling to "
                f"itself): max|mu + mu^T| = {float(residual):.3e} exceeds "
                f"tolerance {float(tolerance):.3e}."
            )


class MagneticMomentTerm:
    """Off-diagonal generator contribution coupling the primary and companion helicity blocks.

    A ``tpeanuts.physics.common.terms.GeneratorTerm``: ``kind =
    TermKind.ADDITIVE``, ``replaces = None`` -- it only ever adds the
    off-diagonal coupling, never overlapping the primary or companion
    block's own diagonal contribution (built entirely separately by
    ``tpeanuts.physics.model.PropagationModel``).

    Attributes:
        config: The MagneticMomentConfig carrying ``mu``.
        kind: Always ``TermKind.ADDITIVE``.
        replaces: Always ``None``.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.physics.extensions import MagneticMomentConfig, MagneticMomentTerm
        >>> mu = torch.zeros(3, 3, dtype=torch.complex128)
        >>> term = MagneticMomentTerm(MagneticMomentConfig(mu=mu, context=context))
        >>> term.kind.name, term.replaces   # it only adds the coupling between the two helicity blocks
        ('ADDITIVE', None)
    """

    kind = TermKind.ADDITIVE
    replaces: Optional[str] = None

    def __init__(self, config: MagneticMomentConfig) -> None:
        self.config = config

    def validate(self, mixing: MixingMatrix, environment: PropagationEnvironment) -> None:
        """Check that ``mixing`` is helicity-extended, ``environment`` carries a field, and ``mu`` fits ``nature``.

        Args:
            mixing: The MixingMatrix the generator is being built for.
            environment: The PropagationEnvironment this call was given.

        Raises:
            ValueError: If ``mixing.space.n_helicity != 2`` -- this term has
                no meaning otherwise, so using it is almost certainly a
                caller bug rather than something to silently no-op. Or if
                ``environment.B_field_T`` is None, or (Majorana nature)
                ``config.mu`` is not antisymmetric.
        """
        if not mixing.space.is_helicity_extended:
            raise ValueError(
                "MagneticMomentTerm requires a helicity-extended FlavourSpace "
                "(space.n_helicity == 2); it has no physical effect otherwise "
                "(there is no companion block to couple to), so using it is "
                "almost certainly a caller bug rather than something to "
                "silently no-op."
            )
        if environment.B_field_T is None:
            raise ValueError("MagneticMomentTerm requires environment.B_field_T.")
        self.config.validate_for_nature(
            is_majorana=mixing.nature.is_majorana, n_flavour=mixing.space.n_flavour
        )

    def contribute_reduced(
        self,
        mixing: MixingMatrix,
        environment: PropagationEnvironment,
        *,
        antinu: Union[bool, torch.Tensor] = False,
        evolution_scale_m: TensorLike = R_E,
    ) -> torch.Tensor:
        """Build the ``(dim, dim)`` off-diagonal magnetic-moment contribution.

        Zero on both diagonal (``n_flavour, n_flavour``) blocks; the
        bottom-left block is ``O_companion^dagger @ mu @ O_primary`` (``mu``
        rotated from the flavour basis into the two blocks' own reduced
        bases -- see this module's docstring for why two different outer
        blocks are needed, and the derivation is direct: a flavour-basis
        state factors as ``|flavour> = O(antinu) |reduced>``, so a matrix
        connecting two different antinu conventions' flavour states rotates
        with one side's ``O`` on the left and the other's on the right, not
        a single shared ``O`` the way ``rotate_to_reduced`` assumes); the
        top-right block is its Hermitian conjugate, guaranteeing the
        returned tensor is Hermitian regardless of ``mu``'s own structure.

        Args:
            mixing: Built MixingMatrix; must be helicity-extended (checked
                by ``validate``).
            environment: Neutrino/medium data; only ``B_field_T`` is read.
            antinu: Bool or boolean tensor selecting the primary block's
                antinu convention (the companion's own is derived via
                ``mixing.nature.companion_helicity_terms``).
            evolution_scale_m: Positive length scale in metres, matching
                whatever every other active term uses.

        Returns:
            Complex Hermitian tensor shaped
            ``(..., mixing.space.dim, mixing.space.dim)``.
        """
        self.validate(mixing, environment)

        context = mixing.context
        n = mixing.space.n_flavour
        dim = mixing.space.dim
        cdtype = cdtype_from_real(context.dtype)

        companion_antinu, _ = mixing.nature.companion_helicity_terms(antinu)
        O_primary = mixing.outer_block(antinu=antinu)
        O_companion = mixing.outer_block(antinu=companion_antinu)

        mu = self.config.mu_tensor(n_flavour=n).to(device=context.device, dtype=cdtype)
        assert environment.B_field_T is not None  # checked in validate()
        scale = magnetic_moment_scale(
            environment.B_field_T, evolution_scale_m=evolution_scale_m, context=context
        ).to(dtype=cdtype)

        bottom_left = scale[..., None, None] * (
            O_companion.conj().transpose(-2, -1) @ mu @ O_primary
        )

        out = torch.zeros(*bottom_left.shape[:-2], dim, dim, device=context.device, dtype=cdtype)
        out[..., n:, :n] = bottom_left
        out[..., :n, n:] = bottom_left.conj().transpose(-2, -1)
        return out
