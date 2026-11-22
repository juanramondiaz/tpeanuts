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

"""Non-Standard Interactions (NSI), including optional sterile couplings.

NSI are effective four-fermion operators that modify the coherent forward
scattering of neutrinos in matter beyond the MSW potential. The matter
generator contribution in the flavour basis becomes

    H_mat^NSI = V_cc * (diag(1, 0, ..., 0) + epsilon)

where ``epsilon`` is a Hermitian matrix of dimensionless couplings. There is
no physical reason this operator must be restricted to the three *standard*
flavours -- a four-fermion vertex coupling an active or sterile neutrino
index to a matter fermion is equally well-defined either way, only the
*naming convention* for the well-studied 3x3 active-sector case is
historically fixed. By default (``NSIConfig.n_sterile == 0``) ``epsilon`` is
exactly the familiar 3x3 active-sector matrix, diagonal entries real,
off-diagonal entries generally complex, Standard Model limit ``epsilon = 0``
-- every existing caller is unaffected. Setting ``n_sterile > 0`` explicitly
opts into a larger ``(3+n_sterile, 3+n_sterile)`` Hermitian matrix that also
carries active-sterile (``eps_active_sterile_*``) and sterile-sterile
(``eps_sterile_*``) couplings -- see ``NSIConfig``'s own docstring. This is
independent of, and additive with,
``tpeanuts.physics.extensions.sterile.SterileNCTerm``'s own ``-V_NC``
contribution: that term is the *Standard Model* neutral-current effect
surviving once a sterile state (which does not feel V_NC at all) is added,
while a non-zero ``eps_sterile`` diagonal is a genuinely non-standard
addition on top of it -- different physical origin, different
normalization (``sqrt(2) G_F`` for NSI's ``epsilon``/``epsilon_n``
throughout, vs. the ``V_NC``-specific ``G_F/sqrt(2)`` prefactor
``matter_potential_nc`` uses), so the two never overlap or double-count.

Composition dependence (electron/proton vs neutron)
-----------------------------------------------------
NSI couple to individual matter fermions (e, u, d), not to "matter" as a
single substance. Since matter is electrically neutral (n_p = n_e
everywhere), the general three-fermion coupling collapses to exactly two
*effective* pieces: ``epsilon`` (the electron/proton-normalized piece,
``eps_* `` below) and ``epsilon_n`` (the neutron-normalized piece,
``eps_*_n`` below), combining as

    epsilon_eff(x) = epsilon + (n_n(x) / n_e(x)) * epsilon_n

-- this reduction depends only on which Standard Model matter fermions
mediate the interaction, not on which neutrino indices (active or sterile)
the operator connects, so it applies unchanged to the sterile-extended
matrix too. ``epsilon_n`` defaults to the zero matrix, so every existing
caller reproduces the electron/proton-only ``epsilon`` exactly. A non-zero
``eps_*_n`` is treated as a firm request, not an optional refinement: the
consumer (``NSITerm.contribute_reduced`` below) raises rather than silently
reproducing the ``epsilon_n = 0`` result when local neutron-density data is
unavailable.

Module contents:
    NSIConfig
        Frozen dataclass storing the nine ``eps_*`` (electron/proton) and
        nine ``eps_*_n`` (neutron) real scalar couplings for the fixed
        3-flavour active sector, plus -- when ``n_sterile > 0`` -- the
        active-sterile and sterile-sterile coupling blocks, and the derived
        Hermitian ``epsilon``/``epsilon_n`` tensors.
    NSITerm
        ``tpeanuts.physics.common.terms.GeneratorTerm`` built from an
        ``NSIConfig`` -- ``kind = TermKind.REPLACES``, ``replaces =
        "matter_cc"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from tpeanuts.physics.common.generator import electron_projector, rotate_to_reduced
from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.common.potential import matter_potential_cc
from tpeanuts.physics.common.terms import PropagationEnvironment, TermKind, embed_primary_block
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import (
    TensorLike,
    as_tensor,
    cdtype_from_real,
    embed_square_block,
    select_for_antinu,
    transpose_symmetry_residual,
)


def _hermitian_3x3(
    ee: TensorLike,
    mumu: TensorLike,
    tautau: TensorLike,
    emu_re: TensorLike,
    emu_im: TensorLike,
    etau_re: TensorLike,
    etau_im: TensorLike,
    mutau_re: TensorLike,
    mutau_im: TensorLike,
    *,
    context: RuntimeContext,
) -> torch.Tensor:
    """Assemble a Hermitian 3x3 complex tensor from 9 real scalar couplings.

    Built entirely with differentiable ``torch`` ops (``as_tensor``,
    ``torch.complex``, ``torch.stack``), never Python's ``complex()``
    builtin: a ``torch.Tensor`` argument with ``requires_grad=True`` stays
    connected to the returned matrix, so gradients can flow through NSI
    parameters exactly as through mixing angles -- no separate "inference
    mode" of this function exists (see
    ``tpeanuts.util.context.RuntimeContext``'s module docstring).

    Args:
        ee, mumu, tautau: Real diagonal entries.
        emu_re, emu_im, etau_re, etau_im, mutau_re, mutau_im: Real/imaginary
            parts of the three off-diagonal entries.
        context: RuntimeContext for the returned tensor.

    Returns:
        Complex tensor shaped ``(3, 3)``.
    """
    cdtype = cdtype_from_real(context.dtype)

    def _rt(x: TensorLike) -> torch.Tensor:
        return as_tensor(x, dtype=context.dtype, device=context.device)

    zero = torch.zeros((), dtype=context.dtype, device=context.device)
    ee_c = torch.complex(_rt(ee), zero)
    mumu_c = torch.complex(_rt(mumu), zero)
    tautau_c = torch.complex(_rt(tautau), zero)
    emu_c = torch.complex(_rt(emu_re), _rt(emu_im))
    etau_c = torch.complex(_rt(etau_re), _rt(etau_im))
    mutau_c = torch.complex(_rt(mutau_re), _rt(mutau_im))

    row0 = torch.stack((ee_c, emu_c, etau_c))
    row1 = torch.stack((emu_c.conj(), mumu_c, mutau_c))
    row2 = torch.stack((etau_c.conj(), mutau_c.conj(), tautau_c))
    return torch.stack((row0, row1, row2)).to(dtype=cdtype)


def _hermitian_active_sterile_block(
    re: TensorLike, im: TensorLike, *, n_sterile: int, context: RuntimeContext
) -> torch.Tensor:
    """Build the ``(3, n_sterile)`` active-sterile NSI coupling block from real/imaginary parts.

    Only this "upper" half of the full Hermitian matrix's off-diagonal block
    is ever specified -- the Hermitian-conjugate placement in the
    sterile-active corner is built automatically by whichever caller
    assembles the full matrix (see ``NSIConfig._assemble_extended``), the
    same "never ask for the redundant conjugate entry" discipline
    ``_hermitian_3x3`` already follows for the fixed 3x3 active block.

    Args:
        re, im: Real/imaginary parts, broadcastable to ``(3, n_sterile)`` --
            a scalar ``0.0`` (the default) broadcasts to an all-zero block.
        n_sterile: Number of sterile flavours.
        context: RuntimeContext for the returned tensor.

    Returns:
        Complex tensor shaped ``(3, n_sterile)``.
    """
    cdtype = cdtype_from_real(context.dtype)
    re_t = as_tensor(re, device=context.device, dtype=context.dtype)
    im_t = as_tensor(im, device=context.device, dtype=context.dtype)
    block = torch.complex(re_t, im_t).to(dtype=cdtype)
    zeros = torch.zeros(3, n_sterile, device=context.device, dtype=cdtype)
    return block + zeros


def _hermitian_sterile_block(
    re: TensorLike, im: TensorLike, *, n_sterile: int, context: RuntimeContext, name: str
) -> torch.Tensor:
    """Build the ``(n_sterile, n_sterile)`` sterile-sterile NSI coupling block, Hermiticity-checked.

    Unlike the fixed-size 3x3 active block (built by construction, always
    Hermitian -- see ``_hermitian_3x3``), ``n_sterile`` is variable, so the
    caller supplies the *full* block's real/imaginary parts directly and
    this function validates Hermiticity explicitly (reusing
    ``tpeanuts.util.tensor.transpose_symmetry_residual``, the same helper
    ``NSIConfig.from_raw_epsilon``/``tpeanuts.physics.extensions.
    magnetic_moment.MagneticMomentConfig`` already use) rather than silently
    symmetrizing a caller's mistake.

    Args:
        re, im: Real/imaginary parts, broadcastable to
            ``(n_sterile, n_sterile)`` -- a scalar ``0.0`` (the default)
            broadcasts to an all-zero block.
        n_sterile: Number of sterile flavours.
        context: RuntimeContext for the returned tensor.
        name: Field name, used only to point the error message correctly.

    Returns:
        Complex tensor shaped ``(n_sterile, n_sterile)``.

    Raises:
        ValueError: If the assembled block is not Hermitian within
            tolerance.
    """
    cdtype = cdtype_from_real(context.dtype)
    re_t = as_tensor(re, device=context.device, dtype=context.dtype)
    im_t = as_tensor(im, device=context.device, dtype=context.dtype)
    block = torch.complex(re_t, im_t).to(dtype=cdtype)
    zeros = torch.zeros(n_sterile, n_sterile, device=context.device, dtype=cdtype)
    block = block + zeros

    residual, tolerance = transpose_symmetry_residual(block, sign=1.0, conjugate=True)
    if residual > tolerance:
        raise ValueError(
            f"{name} must be Hermitian ({name} == {name}.conj().transpose(-2, -1)): "
            f"max|{name} - {name}^dagger| = {float(residual):.3e} exceeds tolerance "
            f"{float(tolerance):.3e}."
        )
    return block


_ZERO: Any = 0.0  # default of every epsilon field; __post_init__ turns it into a tensor on the context


@dataclass(frozen=True, eq=False)
class NSIConfig:
    """Complete NSI parameter set for neutrino propagation in matter.

    Equality and hashing are deliberately identity-based (``eq=False``):
    every ``eps_*``/``eps_*_n`` field accepts a differentiable
    ``torch.Tensor``, which makes a field-tuple-based structural ``__eq__``
    unsound (comparing tensors with ``==``/hashing them can raise). Use
    ``torch.equal(a.epsilon, b.epsilon)`` for value comparison instead.

    Attributes:
        context: RuntimeContext the derived ``epsilon``/``epsilon_n``
            tensors are built on.
        n_sterile: Number of sterile flavours the active-sterile/
            sterile-sterile coupling fields below are sized for. ``0``
            (the default) means "no sterile NSI coupling configured" --
            ``epsilon``/``epsilon_n`` stay exactly the familiar ``(3, 3)``
            active-sector matrix, and every ``eps_*_sterile*`` field below
            is ignored. A positive value opts into the larger
            ``(3+n_sterile, 3+n_sterile)`` matrix; see
            ``tpeanuts.physics.extensions.nsi.NSITerm.validate`` for why
            this must then match ``mixing.space.n_sterile`` exactly rather
            than merely being compatible with it.
        eps_ee, eps_mumu, eps_tautau: Real diagonal entries of ``epsilon``.
            Only the traceless part of ``epsilon`` is physically observable,
            so ``eps_mumu = 0`` is a valid convention for the diagonal
            sector, not a restriction.
        eps_emu_re, eps_emu_im: Real/imaginary parts of ``epsilon``'s
            e-mu entry.
        eps_etau_re, eps_etau_im: Real/imaginary parts of ``epsilon``'s
            e-tau entry.
        eps_mutau_re, eps_mutau_im: Real/imaginary parts of ``epsilon``'s
            mu-tau entry.
        eps_ee_n, eps_mumu_n, eps_tautau_n, eps_emu_n_re, eps_emu_n_im,
            eps_etau_n_re, eps_etau_n_im, eps_mutau_n_re, eps_mutau_n_im:
            Neutron-normalized counterparts of the nine fields above (see
            the module docstring's "Composition dependence" section). All
            default to 0, reproducing the electron/proton-only ``epsilon``
            exactly.
        eps_active_sterile_re, eps_active_sterile_im: Real/imaginary parts
            of the electron/proton-normalized active-sterile coupling
            block, shaped ``(..., 3, n_sterile)`` (or broadcastable to it,
            so the ``0.0`` default gives an all-zero block). Entry
            ``[alpha, k]`` is the coupling between standard flavour
            ``alpha`` and sterile flavour ``k`` -- only this "upper half"
            of the full Hermitian matrix's off-diagonal block is given; its
            conjugate-transpose placement is built automatically (see
            ``_hermitian_active_sterile_block``). Ignored when
            ``n_sterile == 0``.
        eps_sterile_re, eps_sterile_im: Real/imaginary parts of the
            electron/proton-normalized sterile-sterile coupling block,
            shaped ``(..., n_sterile, n_sterile)``. Unlike the fixed-size
            active block, ``n_sterile`` is variable, so this is given as
            the *full* block directly (not split into per-pair named
            scalars) and validated Hermitian at construction time (see
            ``_hermitian_sterile_block``) rather than silently symmetrized.
            Ignored when ``n_sterile == 0``.
        eps_active_sterile_n_re, eps_active_sterile_n_im,
            eps_sterile_n_re, eps_sterile_n_im: Neutron-normalized
            counterparts of the four fields above, entering ``epsilon_n``
            the same way ``eps_*_n`` does for the active sector.
        label: Short identifier string.
        description: Human-readable description.
        epsilon: Derived complex Hermitian tensor, recomputed in
            ``__post_init__`` from the ``eps_*``/``eps_*_sterile*`` fields
            on every construction (including every ``dataclasses.replace``
            call), so those fields are always the single source of truth.
            Shaped ``(3, 3)`` when ``n_sterile == 0``, otherwise
            ``(3+n_sterile, 3+n_sterile)``. Any value passed explicitly to
            the constructor is discarded; use ``from_raw_epsilon`` for an
            arbitrary tensor that does not decompose into the scalar
            parametrization.
        epsilon_n: Mirrors ``epsilon``, built from the ``eps_*_n``/
            ``eps_*_sterile_n*`` fields.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.physics.extensions import NSIConfig
        >>> config = NSIConfig(context=context, eps_ee=0.05, eps_emu_re=0.02)
        >>> eps = config.epsilon_tensor(n_flavour=3)   # the 3x3 coupling matrix in the flavour basis
        >>> eps.shape
        torch.Size([3, 3])
        >>> round(float(eps[0, 0].real), 3), round(float(eps[0, 1].real), 3)
        (0.05, 0.02)
        >>> bool(torch.allclose(eps, eps.conj().T))   # Hermitian
        True
    """

    context: RuntimeContext
    n_sterile: int = 0

    eps_ee: torch.Tensor = _ZERO
    eps_mumu: torch.Tensor = _ZERO
    eps_tautau: torch.Tensor = _ZERO
    eps_emu_re: torch.Tensor = _ZERO
    eps_emu_im: torch.Tensor = _ZERO
    eps_etau_re: torch.Tensor = _ZERO
    eps_etau_im: torch.Tensor = _ZERO
    eps_mutau_re: torch.Tensor = _ZERO
    eps_mutau_im: torch.Tensor = _ZERO

    eps_ee_n: torch.Tensor = _ZERO
    eps_mumu_n: torch.Tensor = _ZERO
    eps_tautau_n: torch.Tensor = _ZERO
    eps_emu_n_re: torch.Tensor = _ZERO
    eps_emu_n_im: torch.Tensor = _ZERO
    eps_etau_n_re: torch.Tensor = _ZERO
    eps_etau_n_im: torch.Tensor = _ZERO
    eps_mutau_n_re: torch.Tensor = _ZERO
    eps_mutau_n_im: torch.Tensor = _ZERO

    eps_active_sterile_re: torch.Tensor = _ZERO
    eps_active_sterile_im: torch.Tensor = _ZERO
    eps_sterile_re: torch.Tensor = _ZERO
    eps_sterile_im: torch.Tensor = _ZERO

    eps_active_sterile_n_re: torch.Tensor = _ZERO
    eps_active_sterile_n_im: torch.Tensor = _ZERO
    eps_sterile_n_re: torch.Tensor = _ZERO
    eps_sterile_n_im: torch.Tensor = _ZERO

    label: str = ""
    description: str = ""

    epsilon: Optional[torch.Tensor] = field(default=None, compare=False)
    epsilon_n: Optional[torch.Tensor] = field(default=None, compare=False)

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            context: RuntimeContext,
            n_sterile: int = ...,
            eps_ee: TensorLike = ...,
            eps_mumu: TensorLike = ...,
            eps_tautau: TensorLike = ...,
            eps_emu_re: TensorLike = ...,
            eps_emu_im: TensorLike = ...,
            eps_etau_re: TensorLike = ...,
            eps_etau_im: TensorLike = ...,
            eps_mutau_re: TensorLike = ...,
            eps_mutau_im: TensorLike = ...,
            eps_ee_n: TensorLike = ...,
            eps_mumu_n: TensorLike = ...,
            eps_tautau_n: TensorLike = ...,
            eps_emu_n_re: TensorLike = ...,
            eps_emu_n_im: TensorLike = ...,
            eps_etau_n_re: TensorLike = ...,
            eps_etau_n_im: TensorLike = ...,
            eps_mutau_n_re: TensorLike = ...,
            eps_mutau_n_im: TensorLike = ...,
            eps_active_sterile_re: TensorLike = ...,
            eps_active_sterile_im: TensorLike = ...,
            eps_sterile_re: TensorLike = ...,
            eps_sterile_im: TensorLike = ...,
            eps_active_sterile_n_re: TensorLike = ...,
            eps_active_sterile_n_im: TensorLike = ...,
            eps_sterile_n_re: TensorLike = ...,
            eps_sterile_n_im: TensorLike = ...,
            label: str = ...,
            description: str = ...,
            epsilon: Optional[torch.Tensor] = ...,
            epsilon_n: Optional[torch.Tensor] = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        """Recompute ``epsilon``/``epsilon_n`` from the ``eps_*``/``eps_*_n`` fields."""
        object.__setattr__(self, "epsilon", self.epsilon_tensor_base())
        object.__setattr__(self, "epsilon_n", self.epsilon_n_tensor_base())

    @classmethod
    def from_raw_epsilon(
        cls,
        epsilon: torch.Tensor,
        context: RuntimeContext,
        *,
        epsilon_n: Optional[torch.Tensor] = None,
        label: str = "",
        description: str = "",
    ) -> "NSIConfig":
        """Build an ``NSIConfig`` from an arbitrary epsilon tensor.

        Bypasses the ``eps_*``/``eps_*_n`` scalar parametrization entirely:
        those fields are left at their 0.0 defaults and do not reflect
        ``epsilon``/``epsilon_n``'s actual entries. Use this only when
        ``epsilon`` does not decompose into the standard 3x3 Hermitian
        parametrization; prefer the ``eps_*``/``eps_*_n`` fields otherwise.
        The returned config's ``n_sterile`` stays 0 regardless of
        ``epsilon``'s actual size, so ``NSITerm.validate``'s
        ``n_sterile``-matching check never triggers for a config built this
        way -- if a validated active-sterile/sterile-sterile coupling is
        wanted, use ``n_sterile``/``eps_active_sterile_*``/``eps_sterile_*``
        instead of embedding those entries into a raw ``epsilon`` by hand.

        Do not call ``dataclasses.replace()`` on a config built this way:
        ``__post_init__`` always recomputes ``epsilon``/``epsilon_n`` from
        the (here, still-zero) ``eps_*``/``eps_*_n`` fields on every
        construction, so ``replace()`` would silently discard the raw
        tensor. Build a fresh config with ``from_raw_epsilon`` again
        instead.

        Args:
            epsilon: Tensor stored as-is on the returned config. Must be
                square, finite, and Hermitian.
            context: RuntimeContext for the returned config.
            epsilon_n: Optional neutron-normalized epsilon tensor, stored
                as-is when given, subject to the same validation. Left at
                the ``__post_init__``-built zero matrix when omitted.
            label: Optional short identifier string.
            description: Optional human-readable description.

        Returns:
            ``NSIConfig`` with ``epsilon`` (and, if given, ``epsilon_n``)
            set to exactly the given tensor(s).

        Raises:
            ValueError: If ``epsilon``/``epsilon_n`` is not square, contains
                non-finite entries, is not Hermitian within tolerance, or
                the two have different standard-flavour block sizes.
        """
        cls._validate_raw_epsilon(epsilon, name="epsilon")
        if epsilon_n is not None:
            cls._validate_raw_epsilon(epsilon_n, name="epsilon_n")
            if tuple(epsilon.shape[-2:]) != tuple(epsilon_n.shape[-2:]):
                raise ValueError(
                    "epsilon and epsilon_n must have the same standard-flavour "
                    f"block size (final two dimensions): got {tuple(epsilon.shape[-2:])} "
                    f"and {tuple(epsilon_n.shape[-2:])}."
                )

        cfg = cls(context=context, label=label, description=description)
        object.__setattr__(cfg, "epsilon", epsilon)
        if epsilon_n is not None:
            object.__setattr__(cfg, "epsilon_n", epsilon_n)
        return cfg

    def _assemble_extended(
        self,
        base_3x3: torch.Tensor,
        active_sterile_re: TensorLike,
        active_sterile_im: TensorLike,
        sterile_re: TensorLike,
        sterile_im: TensorLike,
        *,
        name: str,
    ) -> torch.Tensor:
        """Assemble the full ``(3+n_sterile, 3+n_sterile)`` Hermitian matrix from its three blocks.

        Returns ``base_3x3`` unchanged when ``self.n_sterile == 0`` (the
        default) -- zero extra cost, and ``epsilon_tensor``/
        ``epsilon_n_tensor``'s own ``embed_square_block``-based zero-padding
        against a sterile-extended ``mixing.space.n_flavour`` continues to
        work exactly as it did before this extension existed.

        Args:
            base_3x3: The fixed 3x3 active-sector Hermitian block.
            active_sterile_re, active_sterile_im: See
                ``_hermitian_active_sterile_block``.
            sterile_re, sterile_im: See ``_hermitian_sterile_block``.
            name: Field name (``"epsilon"`` or ``"epsilon_n"``), used only
                to point the sterile-sterile Hermiticity error message
                correctly.

        Returns:
            Complex tensor shaped ``(3, 3)`` if ``self.n_sterile == 0``,
            otherwise ``(3 + self.n_sterile, 3 + self.n_sterile)``.
        """
        if self.n_sterile == 0:
            return base_3x3

        cdtype = cdtype_from_real(self.context.dtype)
        active_sterile = _hermitian_active_sterile_block(
            active_sterile_re, active_sterile_im, n_sterile=self.n_sterile, context=self.context
        )
        sterile = _hermitian_sterile_block(
            sterile_re, sterile_im, n_sterile=self.n_sterile, context=self.context, name=f"{name}_sterile"
        )

        n = 3 + self.n_sterile
        out = torch.zeros(n, n, device=self.context.device, dtype=cdtype)
        out[:3, :3] = base_3x3.to(dtype=cdtype)
        out[:3, 3:] = active_sterile
        out[3:, :3] = active_sterile.conj().transpose(-2, -1)
        out[3:, 3:] = sterile
        return out

    def epsilon_tensor_base(self) -> torch.Tensor:
        """Build the full Hermitian epsilon matrix: the 3x3 active block, extended when ``n_sterile > 0``.

        Returns:
            Complex tensor shaped ``(3, 3)`` when ``self.n_sterile == 0``,
            otherwise ``(3+n_sterile, 3+n_sterile)``.
        """
        base = _hermitian_3x3(
            self.eps_ee, self.eps_mumu, self.eps_tautau,
            self.eps_emu_re, self.eps_emu_im,
            self.eps_etau_re, self.eps_etau_im,
            self.eps_mutau_re, self.eps_mutau_im,
            context=self.context,
        )
        return self._assemble_extended(
            base, self.eps_active_sterile_re, self.eps_active_sterile_im,
            self.eps_sterile_re, self.eps_sterile_im, name="epsilon",
        )

    def epsilon_n_tensor_base(self) -> torch.Tensor:
        """Build the full Hermitian neutron-normalized epsilon_n matrix, extended when ``n_sterile > 0``.

        Returns:
            Complex tensor shaped ``(3, 3)`` when ``self.n_sterile == 0``,
            otherwise ``(3+n_sterile, 3+n_sterile)``.
        """
        base = _hermitian_3x3(
            self.eps_ee_n, self.eps_mumu_n, self.eps_tautau_n,
            self.eps_emu_n_re, self.eps_emu_n_im,
            self.eps_etau_n_re, self.eps_etau_n_im,
            self.eps_mutau_n_re, self.eps_mutau_n_im,
            context=self.context,
        )
        return self._assemble_extended(
            base, self.eps_active_sterile_n_re, self.eps_active_sterile_n_im,
            self.eps_sterile_n_re, self.eps_sterile_n_im, name="epsilon_n",
        )

    def epsilon_tensor(self, *, n_flavour: int) -> torch.Tensor:
        """Embed ``self.epsilon`` for a mixing matrix with ``n_flavour`` flavours.

        Args:
            n_flavour: Total number of flavours of the target generator
                (``mixing.space.n_flavour``).

        Returns:
            Complex tensor shaped ``(..., n_flavour, n_flavour)``: unchanged
            if already that shape, or embedded in the top-left corner of an
            ``(n_flavour, n_flavour)`` zero matrix otherwise (the sterile
            rows/columns beyond ``self.epsilon``'s own size carry no NSI
            coupling -- see ``NSITerm.validate`` for why a non-zero
            ``self.n_sterile`` requires this padding to never actually
            trigger).

        Raises:
            ValueError: If ``self.epsilon``'s shape is incompatible with
                ``n_flavour``.
        """
        return self._embed_standard(self.epsilon, n_flavour=n_flavour, name="epsilon")

    def epsilon_n_tensor(self, *, n_flavour: int) -> torch.Tensor:
        """Embed ``self.epsilon_n`` for a mixing matrix with ``n_flavour`` flavours.

        Mirrors ``epsilon_tensor`` exactly, embedding ``self.epsilon_n``.

        Args:
            n_flavour: Total number of flavours of the target generator.

        Returns:
            Complex tensor shaped ``(..., n_flavour, n_flavour)``.

        Raises:
            ValueError: If ``self.epsilon_n``'s shape is incompatible with
                ``n_flavour``.
        """
        return self._embed_standard(self.epsilon_n, n_flavour=n_flavour, name="epsilon_n")

    @property
    def is_sm_limit(self) -> bool:
        """True when both epsilon and epsilon_n are exactly zero (Standard Model limit)."""
        if self.epsilon is not None and bool(torch.any(self.epsilon != 0)):
            return False
        if self.epsilon_n is not None and bool(torch.any(self.epsilon_n != 0)):
            return False
        return True

    @property
    def has_neutron_coupling(self) -> bool:
        """True when epsilon_n is not exactly the zero matrix.

        Gates the composition-dependent contribution
        ``V_cc(n_n) * epsilon_n`` in ``NSITerm.contribute_reduced``: the
        default (all-zero) ``epsilon_n`` costs nothing extra there.
        """
        if self.epsilon_n is None:
            return False
        return bool(torch.any(self.epsilon_n != 0))

    @property
    def has_cp_violation(self) -> bool:
        """True when any off-diagonal entry of epsilon or epsilon_n has a non-zero imaginary part."""
        return self._has_offdiag_imag(self.epsilon) or self._has_offdiag_imag(self.epsilon_n)

    @staticmethod
    def _has_offdiag_imag(eps: Optional[torch.Tensor]) -> bool:
        """True when ``eps`` has any non-zero off-diagonal imaginary part.

        A staticmethod (not a module function): this is specifically about
        the shape of an NSIConfig-owned epsilon/epsilon_n tensor, not a
        generic linear-algebra utility -- unlike ``_hermitian_3x3`` above,
        which builds one from nine independent scalars and is exactly as
        applicable to any Hermitian-matrix use case, not just this class's.

        Args:
            eps: Complex tensor, or None.

        Returns:
            False if ``eps`` is None; otherwise whether any off-diagonal
            entry has a non-zero imaginary part.
        """
        if eps is None:
            return False
        n = eps.shape[-1]
        eye = torch.eye(n, dtype=torch.bool, device=eps.device)
        off_diagonal_imag = eps.imag.masked_fill(eye, 0.0)
        return bool(torch.any(off_diagonal_imag != 0))

    @staticmethod
    def _validate_raw_epsilon(epsilon: torch.Tensor, *, name: str) -> None:
        """Shared square/finite/Hermitian validation for ``from_raw_epsilon``.

        Args:
            epsilon: Tensor to validate.
            name: Field name, used only to point error messages at
                ``epsilon`` or ``epsilon_n`` correctly.

        Raises:
            ValueError: If ``epsilon`` has fewer than 2 dimensions, is not
                square, contains non-finite entries, or is not Hermitian
                within tolerance.
        """
        if epsilon.ndim < 2:
            raise ValueError(
                f"{name} must have at least 2 dimensions (a square matrix in "
                f"the final two), got shape {tuple(epsilon.shape)}."
            )
        if epsilon.shape[-1] != epsilon.shape[-2]:
            raise ValueError(
                f"{name} must be square, got final dimensions {tuple(epsilon.shape[-2:])}."
            )
        if not torch.isfinite(epsilon).all():
            raise ValueError(f"{name} contains non-finite (NaN/Inf) entries.")

        residual, tolerance = transpose_symmetry_residual(epsilon, sign=1.0, conjugate=True)
        if residual > tolerance:
            raise ValueError(
                f"{name} must be Hermitian ({name} == {name}.conj().transpose(-2, -1)): "
                f"max|{name} - {name}^dagger| = {float(residual):.3e} exceeds tolerance "
                f"{float(tolerance):.3e}. The propagation matter potential must be "
                "Hermitian for the resulting generator -- and hence the evolution -- "
                "to be unitary."
            )

    def _embed_standard(self, eps: Optional[torch.Tensor], *, n_flavour: int, name: str) -> torch.Tensor:
        """Shared embedding logic for ``epsilon_tensor``/``epsilon_n_tensor``.

        An instance method (not a staticmethod): unlike the two helpers
        above, this reads ``self.context`` -- there is no independent
        caller-supplied context, since embedding is always relative to this
        config's own device/dtype. The actual zero-padding is
        ``tpeanuts.util.tensor.embed_square_block`` -- this wrapper only
        adds the None check and an NSI-specific error message.

        Args:
            eps: The tensor to embed (``self.epsilon`` or ``self.epsilon_n``).
            n_flavour: Total number of flavours of the target generator.
            name: Field name, used only to point error messages correctly.

        Returns:
            Complex tensor shaped ``(..., n_flavour, n_flavour)``.

        Raises:
            ValueError: If ``eps`` is None, or has an incompatible shape.
        """
        if eps is None:
            raise ValueError(f"No {name} matrix available: build an NSIConfig with {name} populated.")

        cdtype = cdtype_from_real(self.context.dtype)
        eps = eps.to(device=self.context.device, dtype=cdtype)

        try:
            return embed_square_block(eps, n_flavour)
        except ValueError as exc:
            raise ValueError(
                f"{name} must have final dimensions ({n_flavour}, {n_flavour}) or a "
                f"smaller square standard-flavour block; got {tuple(eps.shape[-2:])}."
            ) from exc


class NSITerm:
    """Reduced-basis generator contribution from Non-Standard Interactions.

    A ``tpeanuts.physics.common.terms.GeneratorTerm``: ``kind =
    TermKind.REPLACES``, ``replaces = "matter_cc"``, since this term's own
    contribution already includes the plain charged-current piece
    (``V_cc * diag(1,0,...,0)``).

    Attributes:
        config: The NSIConfig carrying epsilon/epsilon_n.
        kind: Always ``TermKind.REPLACES``.
        replaces: Always ``"matter_cc"``.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.physics.extensions import NSIConfig, NSITerm
        >>> term = NSITerm(NSIConfig(context=context, eps_ee=0.05))
        >>> term.kind.name, term.replaces   # it takes over the plain charged-current matter term
        ('REPLACES', 'matter_cc')
    """

    kind = TermKind.REPLACES
    replaces = "matter_cc"

    def __init__(self, config: NSIConfig) -> None:
        self.config = config

    def validate(self, mixing: MixingMatrix, environment: PropagationEnvironment) -> None:
        """Check ``config.n_sterile`` against ``mixing`` and that ``environment`` carries neutron density whenever
        needed.

        Args:
            mixing: The MixingMatrix the generator is being built for.
            environment: The PropagationEnvironment this call was given.

        Raises:
            ValueError: If ``config.n_sterile > 0`` and it does not exactly
                match ``mixing.space.n_sterile`` -- a non-zero
                ``config.n_sterile`` means an active-sterile/sterile-sterile
                NSI coupling was deliberately configured for a specific
                sterile count, so silently zero-padding a mismatched extra
                sterile flavour (which ``epsilon_tensor``'s own
                ``embed_square_block`` call would otherwise do without
                complaint, since it only requires the source block to fit
                inside the target, not match it exactly) would leave that
                flavour's coupling undefined rather than reflecting an
                explicit choice. ``config.n_sterile == 0`` (no sterile NSI
                coupling intended) is unaffected and keeps working against
                any ``mixing.space.n_sterile``, as before this extension
                existed. Or if ``config.has_neutron_coupling`` is True and
                ``environment.n_n_mol_cm3`` is None -- see the module
                docstring's "Composition dependence" section: a non-zero
                ``eps_*_n`` is a firm request, not an optional refinement
                that can silently fall back to the epsilon-only result.
        """
        if self.config.n_sterile > 0 and mixing.space.n_sterile != self.config.n_sterile:
            raise ValueError(
                f"config.n_sterile ({self.config.n_sterile}) does not match "
                f"mixing.space.n_sterile ({mixing.space.n_sterile}). Use "
                "config.n_sterile == 0 if no active-sterile/sterile-sterile "
                "NSI coupling is intended (works against any "
                "mixing.space.n_sterile); otherwise the two must agree exactly."
            )
        if self.config.has_neutron_coupling and environment.n_n_mol_cm3 is None:
            raise ValueError(
                "config.has_neutron_coupling is True (a non-zero eps_*_n was "
                "set) but environment.n_n_mol_cm3 was not supplied. This is "
                "not an optional refinement: eps_*_n was set explicitly, so "
                "silently reproducing the epsilon-only result would be a "
                "silent physical mismatch with the requested NSI "
                "configuration. Pass the medium's neutron density, or set "
                "eps_*_n back to zero."
            )

    def contribute_reduced(
        self,
        mixing: MixingMatrix,
        environment: PropagationEnvironment,
        *,
        antinu: Union[bool, torch.Tensor] = False,
        evolution_scale_m: TensorLike = R_E,
    ) -> torch.Tensor:
        """Build the reduced-basis NSI term.

        ``H_mat^NSI = V_cc * (diag(1,0,...,0) + epsilon) [+ V_cc(n_n) *
        epsilon_n, when ``config.has_neutron_coupling``]``, built in the
        flavour basis (where ``epsilon`` is physically defined) and rotated
        down via ``tpeanuts.physics.common.generator.rotate_to_reduced``.

        Args:
            mixing: Built MixingMatrix.
            environment: Neutrino/medium data; ``n_e_mol_cm3`` is always
                read, ``n_n_mol_cm3`` only when ``config.has_neutron_coupling``
                (checked by ``validate``).
            antinu: Bool or boolean tensor selecting the antineutrino
                convention (flips ``V_cc``'s sign and complex-conjugates
                ``epsilon``/``epsilon_n`` and the outer block).
            evolution_scale_m: Positive length scale in metres, matching
                whatever every other active term uses.

        Returns:
            Complex Hermitian tensor shaped
            ``(..., mixing.space.n_flavour, mixing.space.n_flavour)``.
        """
        self.validate(mixing, environment)

        context = mixing.context
        n = mixing.space.n_flavour
        cdtype = cdtype_from_real(context.dtype)
        O = mixing.outer_block(antinu=antinu)

        V_cc = matter_potential_cc(
            environment.n_e_mol_cm3, antinu, evolution_scale_m=evolution_scale_m, context=context
        )
        eps = self.config.epsilon_tensor(n_flavour=n)
        eps = select_for_antinu(antinu, eps, eps.conj())
        D_flavour = electron_projector(n, device=context.device, cdtype=cdtype) + eps

        contribution = V_cc.to(dtype=cdtype)[..., None, None] * rotate_to_reduced(D_flavour, O)

        if self.config.has_neutron_coupling:
            if environment.n_n_mol_cm3 is None:
                raise ValueError("NSITerm with neutron couplings needs environment.n_n_mol_cm3.")
            V_cc_n = matter_potential_cc(
                environment.n_n_mol_cm3, antinu, evolution_scale_m=evolution_scale_m, context=context
            )
            eps_n = self.config.epsilon_n_tensor(n_flavour=n)
            eps_n = select_for_antinu(antinu, eps_n, eps_n.conj())
            contribution = contribution + V_cc_n.to(dtype=cdtype)[..., None, None] * rotate_to_reduced(
                eps_n, O
            )

        return embed_primary_block(contribution, mixing.space)

    def direction_reduced(
        self,
        mixing: MixingMatrix,
        environment: PropagationEnvironment,
        *,
        antinu: Union[bool, torch.Tensor] = False,
    ) -> torch.Tensor:
        """Return this term's V-independent direction matrix ``diag(1,0,...,0) + epsilon``, rotated and embedded.

        Optional ``GeneratorTerm`` extension consumed by
        ``tpeanuts.physics.evolution.perturbative.perturbative``'s first-order perturbative
        correction (not part of the required ``GeneratorTerm`` protocol
        surface). Pairs with ``tpeanuts.physics.common.potential.
        matter_potential_cc(environment.n_e_mol_cm3, antinu,
        evolution_scale_m=...)``: multiplying the two together reproduces
        ``contribute_reduced`` exactly -- but only when
        ``config.has_neutron_coupling`` is False. When a non-zero
        ``eps_*_n`` is set, this term's own contribution has a *second*,
        independent V*P piece (``V_cc(n_n) * epsilon_n``, paired with a
        *different* potential) that a single direction matrix cannot
        represent -- rather than silently dropping that physics, this
        raises instead of returning a partial answer.

        Args:
            mixing: Built MixingMatrix.
            environment: Neutrino/medium data; only used by ``validate``.
            antinu: Bool or boolean tensor selecting the antineutrino
                convention (complex-conjugates ``epsilon`` and the outer
                block), matching ``contribute_reduced``'s own convention.

        Returns:
            Complex tensor shaped ``(..., mixing.space.dim,
            mixing.space.dim)``.

        Raises:
            ValueError: If ``config.has_neutron_coupling`` is True.
        """
        if self.config.has_neutron_coupling:
            raise ValueError(
                "NSITerm.direction_reduced does not support "
                "config.has_neutron_coupling == True: this term's own "
                "contribution then has a second, independent V*P piece "
                "(V_cc(n_n) * epsilon_n) that a single direction matrix "
                "cannot represent. Use PropagationMethod.NUMERICAL for "
                "this configuration, or set eps_*_n back to zero."
            )
        self.validate(mixing, environment)

        context = mixing.context
        n = mixing.space.n_flavour
        cdtype = cdtype_from_real(context.dtype)
        O = mixing.outer_block(antinu=antinu)

        eps = self.config.epsilon_tensor(n_flavour=n)
        eps = select_for_antinu(antinu, eps, eps.conj())
        D_flavour = electron_projector(n, device=context.device, cdtype=cdtype) + eps
        return embed_primary_block(rotate_to_reduced(D_flavour, O), mixing.space)
