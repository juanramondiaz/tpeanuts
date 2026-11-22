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

"""Generic tensor, dtype, and device helpers with no physics content.

Every routine in tpeanuts works with a real floating dtype (angles, energies,
densities) and its complex counterpart (mixing matrices, density matrices).
This module centralizes the conversion between the two, plain value-to-tensor
conversion, and CUDA/CPU device resolution, so no other module in the project
needs to reimplement them.

Module contents:
    TensorLike
        Type alias for values accepted wherever a scalar or tensor is valid.
    cdtype_from_real(...)
        Map a real floating dtype to its matching complex dtype.
    real_dtype_from_tensor(...)
        Infer the real floating dtype matching a tensor's precision.
    as_tensor(...)
        Convert a value to a torch tensor with optional dtype/device/grad.
    default_device(...)
        Resolve None to the project default CUDA/CPU device.
    resolve_device(...)
        Resolve an explicit device, a deferred device factory, or the
        default.
    infer_device_dtype(...)
        Infer a (device, dtype) pair from explicit overrides or tensor
        inputs.
    select_for_antinu(...)
        Pick between a "neutrino" and an "antineutrino" value from a plain
        bool or a boolean/numeric tensor mask.
    antinu_sign(...)
        Convenience over select_for_antinu returning +1/-1 rather than two
        arbitrary values.
    negate_antinu(...)
        Flip an antinu selector (bool or tensor) to its opposite convention.
    embed_square_block(...)
        Embed a smaller square matrix in the top-left corner of a larger
        zero matrix, unchanged if already the target size.
    transpose_symmetry_residual(...)
        The two numbers (residual, tolerance) behind every "is this matrix
        Hermitian/antisymmetric within floating-point tolerance" check in
        the project, so each call site only has to compare and phrase its
        own error message.
    kron_batched(...)
        A Kronecker product that treats leading dimensions as an ordinary
        batch -- unlike ``torch.kron`` itself, which has no real notion of
        a batch dimension at all.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch

TensorLike = Union[float, int, torch.Tensor, np.ndarray, Sequence[float]]


def cdtype_from_real(dtype: torch.dtype) -> torch.dtype:
    """Map a real floating torch dtype to its matching complex dtype.

    Args:
        dtype: Real floating dtype, ``torch.float32`` or ``torch.float64``.

    Returns:
        ``torch.complex64`` for ``float32``, or ``torch.complex128`` for
        ``float64``.

    Raises:
        TypeError: If ``dtype`` is not ``torch.float32`` or ``torch.float64``.
    """
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    raise TypeError(
        f"Expected a real floating dtype (float32 or float64), got {dtype}."
    )


def real_dtype_from_tensor(tensor: torch.Tensor) -> torch.dtype:
    """Infer the real floating dtype matching a tensor's precision.

    Args:
        tensor: Real or complex tensor whose precision determines the
            result.

    Returns:
        ``torch.float64`` if ``tensor`` is ``complex128`` or ``float64``;
        ``torch.float32`` otherwise.
    """
    if tensor.dtype in (torch.complex128, torch.float64):
        return torch.float64
    return torch.float32


def as_tensor(
    x: TensorLike,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    requires_grad: Optional[bool] = None,
) -> torch.Tensor:
    """Convert a value to a torch tensor, applying optional dtype/device/grad.

    An existing tensor is cast/moved in place rather than rebuilt from
    scratch, so it keeps its identity (and its autograd history) whenever no
    dtype/device change is actually requested.

    Args:
        x: Scalar, array-like value, or existing torch tensor.
        dtype: Optional dtype applied to the result.
        device: Optional device applied to the result.
        requires_grad: Optional autograd flag explicitly set on the result.
            Left untouched (whatever ``x`` already had) when omitted -- this
            module never decides on its own whether gradients should track;
            see ``tpeanuts.util.context.RuntimeContext``.

    Returns:
        Torch tensor representation of ``x``.
    """
    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None:
            t = t.to(dtype=dtype)
        if device is not None:
            t = t.to(device=device)
    else:
        if isinstance(x, np.ndarray) and not x.flags.writeable:
            # torch.as_tensor shares memory with a writable NumPy array for
            # free; a non-writable one (e.g. a read-only view) can't be
            # shared that way without torch warning that in-place writes
            # would be undefined behaviour, so copy defensively only then.
            x = x.copy()
        t = torch.as_tensor(x, dtype=dtype, device=device)

    if requires_grad is not None:
        t.requires_grad_(requires_grad)

    return t


def default_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve an explicit device, or the project default CUDA/CPU device.

    Args:
        device: Device specification accepted by ``torch.device``, or None
            to select CUDA when available and CPU otherwise.

    Returns:
        Resolved ``torch.device``. A bare ``"cuda"`` gets the index of the current CUDA device, because the tensors
        report ``cuda:0`` and ``torch.device("cuda") != torch.device("cuda:0")``: a context built from the bare name
        would be rejected as "another device" by every check that compares it with a tensor's device.
    """
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a concrete torch device from a value or a deferred factory.

    Args:
        device: Device specification accepted by ``torch.device``, None for
            the default CUDA/CPU selection, or a zero-argument callable
            returning either form (useful for configuration objects that
            must not query ``torch.cuda.is_available()`` at import time).

    Returns:
        Resolved ``torch.device``.
    """
    if callable(device):
        return default_device(device())
    return default_device(device)


def infer_device_dtype(
    *values: TensorLike,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.device, torch.dtype]:
    """Infer a (device, dtype) pair from explicit overrides or tensor inputs.

    The first tensor found in ``values`` supplies whichever of ``device``/
    ``dtype`` was not given explicitly. When no tensor is present, the
    project default device and ``torch.float64`` are used.

    Args:
        *values: Candidate scalar or tensor-like values, inspected in order.
        device: Optional explicit device override; bypasses tensor
            inference when given.
        dtype: Optional explicit dtype override; bypasses tensor inference
            when given.

    Returns:
        ``(device, dtype)`` resolved from the explicit overrides, the first
        tensor found in ``values``, or the project defaults (CUDA/CPU,
        ``torch.float64``) when neither is available.
    """
    for value in values:
        if torch.is_tensor(value):
            return (
                value.device if device is None else torch.device(device),
                value.dtype if dtype is None else dtype,
            )

    return (
        default_device(device),
        torch.float64 if dtype is None else dtype,
    )


def select_for_antinu(
    antinu: Union[bool, torch.Tensor],
    value_neutrino: torch.Tensor,
    value_antineutrino: torch.Tensor,
) -> torch.Tensor:
    """Pick between a "neutrino" and an "antineutrino" value.

    The single shared primitive behind every neutrino/antineutrino
    convention switch in the project (a mixing matrix vs. its complex
    conjugate, a matter potential's sign, ...): everywhere something needs
    to pick one of two values based on a per-batch-entry bool, it goes
    through this function rather than reimplementing the same
    cast-broadcast-``torch.where`` sequence locally.

    Args:
        antinu: Plain Python bool, or a boolean/numeric tensor mask selecting
            the antineutrino convention per batch entry. A tensor mask is
            padded with trailing singleton dimensions (on its own copy, not
            in place) until it matches ``value_neutrino``'s number of
            dimensions, so it broadcasts against values of any rank (a
            scalar potential, a matrix, ...) without the caller having to
            reshape it first.
        value_neutrino: Value(s) used where ``antinu`` is False.
        value_antineutrino: Value(s) used where ``antinu`` is True,
            broadcastable with ``value_neutrino``.

    Returns:
        ``value_antineutrino`` where ``antinu`` is True, ``value_neutrino``
        elsewhere. A plain bool ``antinu`` returns one of the two inputs
        directly, with no tensor operation at all.
    """
    if isinstance(antinu, bool):
        return value_antineutrino if antinu else value_neutrino

    mask = antinu.to(device=value_neutrino.device, dtype=torch.bool)
    while mask.ndim < value_neutrino.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, value_antineutrino, value_neutrino)


def antinu_sign(
    antinu: Union[bool, torch.Tensor],
    reference: torch.Tensor,
) -> Union[float, torch.Tensor]:
    """Return +1 for neutrinos and -1 for antineutrinos, broadcast against ``reference``.

    A thin convenience over ``select_for_antinu`` for the sign-flip pattern
    shared by every matter potential (charged-current, neutral-current, ...):
    the potential's magnitude is identical for neutrinos and antineutrinos,
    only its sign differs.

    Args:
        antinu: Plain Python bool, or a boolean/numeric tensor mask; see
            ``select_for_antinu``.
        reference: Tensor whose shape/device the returned sign is broadcast
            against.

    Returns:
        A plain Python ``float`` (+1.0 or -1.0) when ``antinu`` is a plain
        bool; otherwise a real tensor matching ``reference``'s shape and
        device, +1 where ``antinu`` is False and -1 where it is True.
    """
    if isinstance(antinu, bool):
        return -1.0 if antinu else 1.0

    return select_for_antinu(antinu, torch.ones_like(reference), -torch.ones_like(reference))


def negate_antinu(antinu: Union[bool, torch.Tensor]) -> Union[bool, torch.Tensor]:
    """Flip an antinu selector: neutrino becomes antineutrino and vice versa.

    Needed wherever one calculation's "companion" is the opposite
    convention from a given primary one -- e.g.
    ``tpeanuts.physics.common.nature.Majorana.companion_helicity_terms``,
    where the companion helicity block *is* the antiparticle sector.

    Args:
        antinu: Plain Python bool, or a boolean/numeric tensor mask; see
            ``select_for_antinu``.

    Returns:
        ``not antinu`` for a plain bool; a boolean tensor with every entry
        logically negated otherwise.
    """
    if isinstance(antinu, bool):
        return not antinu

    return ~antinu.to(dtype=torch.bool)


def embed_square_block(matrix: torch.Tensor, size: int) -> torch.Tensor:
    """Embed a smaller square matrix in the top-left corner of a larger ``(size, size)`` zero matrix.

    The shared primitive behind every "a config's own matrix may be given at
    a smaller block size than the target space and gets zero-padded" pattern
    in the project (e.g. ``tpeanuts.physics.extensions.nsi.NSIConfig.
    epsilon_tensor``, ``tpeanuts.physics.extensions.magnetic_moment.
    MagneticMomentConfig.mu_tensor``) -- everywhere that pattern is needed,
    it goes through this function rather than reimplementing the same
    shape-check-and-zero-pad locally.

    Args:
        matrix: Tensor whose final two dimensions are square,
            ``(..., n0, n0)``, ``n0 <= size``.
        size: Target final-dimension size.

    Returns:
        ``matrix`` unchanged (same object) if already ``(size, size)``;
        otherwise a tensor shaped ``(..., size, size)``, zero outside the
        top-left ``(n0, n0)`` block.

    Raises:
        ValueError: If ``matrix``'s final two dimensions are not square, or
            ``n0 > size``.
    """
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"matrix must be square in its final two dimensions, got {tuple(matrix.shape[-2:])}."
        )
    n0 = matrix.shape[-1]
    if n0 == size:
        return matrix
    if n0 > size:
        raise ValueError(
            f"matrix's own square block ({n0}) is larger than the target size ({size})."
        )

    out = torch.zeros(*matrix.shape[:-2], size, size, device=matrix.device, dtype=matrix.dtype)
    out[..., :n0, :n0] = matrix
    return out


def transpose_symmetry_residual(
    matrix: torch.Tensor,
    *,
    sign: float = 1.0,
    conjugate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(residual, tolerance)`` for ``matrix == sign * transpose(matrix)``.

    The shared numerical core behind every "is this matrix Hermitian (or
    antisymmetric) within floating-point tolerance" check in the project
    (e.g. ``tpeanuts.physics.extensions.nsi.NSIConfig._validate_raw_epsilon``'s
    Hermiticity check, ``tpeanuts.physics.extensions.magnetic_moment.
    MagneticMomentConfig.validate_for_nature``'s antisymmetry check) --
    only the numerical comparison is shared; each call site still decides
    whether to raise and phrases its own domain-specific error message,
    since "epsilon must be Hermitian" and "mu must be antisymmetric (CPT)"
    mean very different things physically.

    Args:
        matrix: Complex or real tensor, square in its final two dimensions.
        sign: ``+1.0`` for a symmetric/Hermitian check (``matrix ==
            transpose(matrix)``), ``-1.0`` for an antisymmetric check
            (``matrix == -transpose(matrix)``).
        conjugate: ``True`` (default) compares against the conjugate
            transpose (Hermitian-family checks); ``False`` compares against
            the plain transpose (e.g. CPT antisymmetry, which constrains
            ``mu_ij = -mu_ji`` directly, not its complex conjugate).

    Returns:
        ``(residual, tolerance)``: ``residual`` is
        ``max|matrix - sign * transpose(matrix)|``; ``tolerance`` is
        ``100 * torch.finfo(matrix's real dtype).eps *
        max(1.0, max|matrix|)``, i.e. a fixed multiplier of machine epsilon
        scaled by the matrix's own magnitude. The caller raises iff
        ``residual > tolerance``.
    """
    transposed = matrix.conj().transpose(-2, -1) if conjugate else matrix.transpose(-2, -1)
    real_dtype = matrix.real.dtype if matrix.is_complex() else matrix.dtype
    eps_mach = torch.finfo(real_dtype).eps
    scale = matrix.abs().max().clamp_min(1.0)
    residual = (matrix - sign * transposed).abs().max()
    tolerance = 1.0e2 * eps_mach * scale
    return residual, tolerance


def kron_batched(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Kronecker product treating leading dimensions as an ordinary batch.

    ``torch.kron`` itself does *not* do this: per its own definition it
    treats every dimension of both inputs multiplicatively (matching ranks
    by prepending size-1 dimensions), so it only happens to reproduce a
    per-batch Kronecker product when one operand has no batch dimension at
    all (broadcast from an implicit leading 1) -- confirmed directly to
    give the *wrong* answer (a literal batch-times-batch contraction, not a
    per-slice one) once both operands carry the same batch size, e.g. two
    ``(3, d, d)`` tensors give a ``(9, d**2, d**2)`` result from
    ``torch.kron`` directly, not the ``(3, d**2, d**2)`` per-slice one.
    Needed by every Liouville-space superoperator in
    ``tpeanuts.physics.evolution.liouville.open_system`` (a Hamiltonian or a Lindblad
    operator both routinely carry a batch dimension, e.g. one entry per
    energy bin).

    Args:
        A: Complex or real tensor shaped ``(..., m, n)``.
        B: Tensor shaped ``(..., p, q)``, broadcastable against ``A``'s
            leading batch dimensions.

    Returns:
        Tensor shaped ``(..., m*p, n*q)``.
    """
    m, n = A.shape[-2:]
    p, q = B.shape[-2:]
    expanded = A[..., :, None, :, None] * B[..., None, :, None, :]
    return expanded.reshape(*expanded.shape[:-4], m * p, n * q)
