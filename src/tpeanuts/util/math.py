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

"""Generic, domain-neutral mathematical helpers.

Populated incrementally as new modules need new helpers (matching
``tpeanuts.util.constant``'s convention) -- ``tree_reduce_matmul`` for
``tpeanuts.physics.common.composition.compose_segment_evolutors``, and
``sec``/``csc``/``intersection``/``carlson_rf``/``ellipf_incomplete``/
``complex_sqrt`` (ported from v1_0's own ``util/math.py``) for
``tpeanuts.medium.earth.exposure.time_average``'s elliptic-integral nadir-
exposure geometry, the first consumer needing them. ``csin``/``ccos``/
``ctan``/``casin`` (v1_0's own thin ``torch.sin``/``cos``/``tan``/``asin``
wrappers) are deliberately not ported -- they added no behaviour over
calling ``torch`` directly, real or complex dtype alike.

Module contents:
    eigh_chunked(...), eigvalsh_chunked(...)
        ``torch.linalg.eigh``/``eigvalsh`` that split a large batch into chunks on CUDA, where cusolver rejects
        batches of about 65 000 matrices or more.
    searchsorted_contiguous(...)
        ``torch.searchsorted`` on contiguous operands: torch silently copies a non-contiguous
        query and warns about it (once per process), so every call in tpeanuts goes through this.
    tree_reduce_matmul(...)
        Reduce a stack of square matrices to a single product via a binary
        tree, in O(log N) batched matmul levels instead of O(N) sequential
        ones.
    interp1d_linear(...)
        Piecewise-linear interpolation, clamped to given left/right values
        outside the tabulated domain. Promoted from
        ``tpeanuts.source.solar.solar``'s own private copy once a third
        independent use (``tpeanuts.source.reactor.reactor``'s Huber-Mueller
        spectrum, and ``tpeanuts.experiments.dayabay``'s non-equilibrium/SNF/
        LSNL curve corrections) crossed the "still local to one module"
        threshold that helper had been kept under.
    sec(...), csc(...)
        Secant and cosecant, ``1/cos``/``1/sin``.
    intersection(...)
        Common overlap of one or more closed 1D intervals, empty when
        disjoint.
    complex_sqrt(...)
        Square root, promoting a real tensor to its matching complex dtype
        first.
    carlson_rf(...)
        Carlson's symmetric elliptic integral RF(x, y, z), via the
        duplication algorithm.
    ellipf_incomplete(...)
        Incomplete elliptic integral of the first kind F(phi | m), via
        ``carlson_rf``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from tpeanuts.util.tensor import cdtype_from_real

# torch.linalg.eigh/eigvalsh of a batch of small matrices fails on CUDA with CUSOLVER_STATUS_INVALID_VALUE from a batch
# of about 65 000 matrices (measured: 30 000 works, 70 000 fails), so a larger one is cut into chunks of this size.
CUDA_EIGH_MAX_BATCH = 32768


def _chunk_size(matrix: torch.Tensor, max_batch: Optional[int]) -> Optional[int]:
    """Largest batch to hand to ``eigh`` in one call, or None when the whole batch can go at once."""
    if max_batch is not None:
        return max_batch
    return CUDA_EIGH_MAX_BATCH if matrix.device.type == "cuda" else None


def eigh_chunked(matrix: torch.Tensor, *, max_batch: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """``torch.linalg.eigh`` that cuts a large batch into chunks, differentiable like the original.

    On CUDA cusolver rejects a batch of about 65 000 matrices or more, so the batch is processed in chunks of
    ``CUDA_EIGH_MAX_BATCH`` there; elsewhere it is one call. The result is the same either way.

    Args:
        matrix: Hermitian matrices shaped ``(..., n, n)``.
        max_batch: Chunk size; ``None`` uses the default of the device (no chunking off CUDA), an integer forces it
            on any device (for tests).

    Returns:
        ``(eigenvalues, eigenvectors)`` shaped ``(..., n)`` and ``(..., n, n)``, as ``torch.linalg.eigh``.
    """
    chunk = _chunk_size(matrix, max_batch)
    batch_shape = matrix.shape[:-2]
    count = math.prod(batch_shape)
    if chunk is None or count <= chunk:
        return torch.linalg.eigh(matrix)
    flat = matrix.reshape(count, *matrix.shape[-2:])
    parts = [torch.linalg.eigh(flat[start:start + chunk]) for start in range(0, count, chunk)]
    eigenvalues = torch.cat([part[0] for part in parts], dim=0).reshape(*batch_shape, matrix.shape[-1])
    eigenvectors = torch.cat([part[1] for part in parts], dim=0).reshape(*batch_shape, *matrix.shape[-2:])
    return eigenvalues, eigenvectors


def eigvalsh_chunked(matrix: torch.Tensor, *, max_batch: Optional[int] = None) -> torch.Tensor:
    """``torch.linalg.eigvalsh`` that cuts a large batch into chunks, see ``eigh_chunked``.

    Args:
        matrix: Hermitian matrices shaped ``(..., n, n)``.
        max_batch: Chunk size, as in ``eigh_chunked``.

    Returns:
        Eigenvalues shaped ``(..., n)``, ascending.
    """
    chunk = _chunk_size(matrix, max_batch)
    batch_shape = matrix.shape[:-2]
    count = math.prod(batch_shape)
    if chunk is None or count <= chunk:
        return torch.linalg.eigvalsh(matrix)
    flat = matrix.reshape(count, *matrix.shape[-2:])
    parts = [torch.linalg.eigvalsh(flat[start:start + chunk]) for start in range(0, count, chunk)]
    return torch.cat(parts, dim=0).reshape(*batch_shape, matrix.shape[-1])


def searchsorted_contiguous(
    sorted_sequence: torch.Tensor,
    values: torch.Tensor,
    *,
    right: bool = False,
) -> torch.Tensor:
    """``torch.searchsorted`` that never triggers torch's non-contiguous-operand warning.

    Args:
        sorted_sequence: Ascending 1-D boundaries.
        values: Query values, any shape (may be a non-contiguous view, e.g. after ``expand``/``transpose``).
        right: If True, return the last suitable index (``side='right'``), otherwise the first.

    Returns:
        Integer indices shaped like ``values``, identical to ``torch.searchsorted``.
    """
    return torch.searchsorted(sorted_sequence.contiguous(), values.contiguous(), right=right)


def interp1d_linear(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, *, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Piecewise-linear interpolation, clamped to ``left``/``right`` outside ``xp``'s domain.

    Args:
        x: Query points, any shape.
        xp: Strictly increasing 1-D grid the table is defined on.
        fp: Table values at ``xp``, shape ``xp.shape``.
        left: Value returned where ``x < xp[0]``, broadcastable against
            ``x``.
        right: Value returned where ``x > xp[-1]``, broadcastable against
            ``x``.

    Returns:
        Interpolated values, shaped like ``x``.
    """
    n = xp.shape[0]
    idx_right = searchsorted_contiguous(xp, x, right=False).clamp(1, n - 1)
    idx_left = idx_right - 1

    x0, x1 = xp[idx_left], xp[idx_right]
    f0, f1 = fp[idx_left], fp[idx_right]
    frac = (x - x0) / (x1 - x0).clamp_min(torch.finfo(xp.dtype).eps)
    interpolated = f0 + frac * (f1 - f0)

    return torch.where(x < xp[0], left, torch.where(x > xp[-1], right, interpolated))


def tree_reduce_matmul(mats: torch.Tensor, *, left: bool = True) -> torch.Tensor:
    """Reduce a stack of square matrices to a single product using a binary tree.

    The reduction proceeds by pairwise matmul at each level, halving the
    stack depth per level. The total number of batched matmul launches is
    O(log N) instead of the O(N) sequential launches of a plain loop; on GPU
    the paired operations at each level are independent and dispatch in
    parallel.

    Args:
        mats: Tensor shaped ``(..., N, d, d)``, where ``N`` is the number of
            matrices to reduce and ``d`` is the matrix size. The ``N``
            dimension is axis ``-3``.
        left: When True (default), the product is evaluated left-to-right:
            ``mats[...,0,:,:] @ mats[...,1,:,:] @ ... @ mats[...,N-1,:,:]``.
            When False, the order is reversed.

    Returns:
        Reduced matrix shaped ``(..., d, d)``.

    Raises:
        ValueError: If ``mats`` has fewer than 3 dimensions, or its last two
            dimensions are not square.
    """
    if mats.ndim < 3:
        raise ValueError("mats must have at least 3 dimensions (..., N, d, d).")
    d1, d2 = mats.shape[-2], mats.shape[-1]
    if d1 != d2:
        raise ValueError(
            f"mats must have square matrices in the last two dimensions, got ({d1}, {d2})."
        )

    if mats.shape[-3] == 1:
        return mats[..., 0, :, :]

    while mats.shape[-3] > 1:
        n = mats.shape[-3]
        if n % 2 == 1:
            # Odd length: carry the last matrix forward unpaired.
            tail = mats[..., -1:, :, :]
            even = mats[..., 0:-1:2, :, :]
            odd = mats[..., 1:-1:2, :, :]
            paired = even @ odd if left else odd @ even
            mats = torch.cat([paired, tail], dim=-3)
        else:
            even = mats[..., 0::2, :, :]
            odd = mats[..., 1::2, :, :]
            mats = even @ odd if left else odd @ even

    return mats[..., 0, :, :]


def sec(x: torch.Tensor) -> torch.Tensor:
    """Secant, ``1 / cos(x)``.

    Args:
        x: Real or complex tensor of angles in radians.

    Returns:
        Tensor containing ``sec(x)``.
    """
    return 1.0 / torch.cos(x)


def csc(x: torch.Tensor) -> torch.Tensor:
    """Cosecant, ``1 / sin(x)``.

    Args:
        x: Real or complex tensor of angles in radians.

    Returns:
        Tensor containing ``csc(x)``.
    """
    return 1.0 / torch.sin(x)


def intersection(i1: torch.Tensor, *intervals: torch.Tensor) -> torch.Tensor:
    """Common overlap of one or more closed 1D intervals.

    Args:
        i1: First interval, shaped ``(2,)`` and ordered ``[lower, upper]``.
            An empty tensor (``numel() == 0``) represents an empty interval.
        *intervals: Additional intervals with the same shape convention.

    Returns:
        Tensor shaped ``(2,)`` with the common overlap, or an empty tensor
        (same device/dtype as ``i1``) if any two intervals are disjoint.

    Raises:
        ValueError: If any non-empty interval does not have exactly two
            endpoints.
    """
    if i1.numel() == 0:
        return i1.new_empty((0,))
    if i1.numel() != 2:
        raise ValueError("i1 must have shape (2,) or be empty.")

    current = i1
    for interval in intervals:
        if interval.numel() == 0:
            return current.new_empty((0,))
        if interval.numel() != 2:
            raise ValueError("Each interval must have shape (2,) or be empty.")

        minimum = torch.maximum(current[0], interval[0])
        maximum = torch.minimum(current[1], interval[1])
        if minimum > maximum:
            return current.new_empty((0,))

        current = torch.stack([minimum, maximum])

    return current


def complex_sqrt(z: torch.Tensor) -> torch.Tensor:
    """Square root, promoting a real tensor to its matching complex dtype first.

    Args:
        z: Real or complex tensor.

    Returns:
        Complex tensor holding ``sqrt(z)``.
    """
    if not torch.is_complex(z):
        z = z.to(dtype=cdtype_from_real(z.dtype))
    return torch.sqrt(z)


def carlson_rf(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    tol: float = 1.0e-12,
    max_iter: int = 50,
) -> torch.Tensor:
    """Carlson's symmetric elliptic integral RF(x, y, z), via the duplication algorithm.

    Inputs are promoted to a complex dtype so the same code path supports
    both real and complex analytic expressions (needed by
    ``ellipf_incomplete`` when its own ``m``/``phi`` arguments are already
    complex).

    Args:
        x: First RF argument.
        y: Second RF argument.
        z: Third RF argument.
        tol: Convergence tolerance for the duplication iterations.
        max_iter: Maximum number of duplication iterations.

    Returns:
        Tensor containing RF(x, y, z), broadcast over the input shapes.
    """
    if not torch.is_complex(x):
        x = x.to(dtype=cdtype_from_real(x.dtype))
    if not torch.is_complex(y):
        y = y.to(dtype=x.dtype)
    if not torch.is_complex(z):
        z = z.to(dtype=x.dtype)

    xn, yn, zn = x, y, z

    for _ in range(max_iter):
        mu = (xn + yn + zn) / 3.0
        X = 1.0 - xn / mu
        Y = 1.0 - yn / mu
        Z = 1.0 - zn / mu

        err = torch.max(torch.stack([X.abs(), Y.abs(), Z.abs()], dim=0), dim=0).values
        if torch.all(err < tol):
            break

        sx, sy, sz = complex_sqrt(xn), complex_sqrt(yn), complex_sqrt(zn)
        lam = sx * sy + sy * sz + sz * sx

        xn = (xn + lam) / 4.0
        yn = (yn + lam) / 4.0
        zn = (zn + lam) / 4.0

    mu = (xn + yn + zn) / 3.0
    X = 1.0 - xn / mu
    Y = 1.0 - yn / mu
    Z = 1.0 - zn / mu

    E2 = X * Y + Y * Z + Z * X
    E3 = X * Y * Z

    series = 1.0 - E2 / 10.0 + E3 / 14.0 + (E2 * E2) / 24.0 - (3.0 * E2 * E3) / 44.0
    return series / complex_sqrt(mu)


def ellipf_incomplete(phi: torch.Tensor, m: torch.Tensor, tol: float = 1.0e-12) -> torch.Tensor:
    """Incomplete elliptic integral of the first kind, ``F(phi | m)``.

    Computed via Carlson's RF: ``F(phi | m) = sin(phi) * RF(cos(phi)**2, 1 -
    m*sin(phi)**2, 1)``.

    Args:
        phi: Amplitude angle in radians (real or complex).
        m: Elliptic parameter (real or complex).
        tol: Convergence tolerance passed to ``carlson_rf``.

    Returns:
        Tensor containing ``F(phi | m)``, broadcast over the input shapes.
    """
    if not torch.is_complex(phi):
        phi = phi.to(dtype=cdtype_from_real(phi.dtype))
    if not torch.is_complex(m):
        m = m.to(dtype=phi.dtype)

    s = torch.sin(phi)
    c = torch.cos(phi)

    x = c * c
    y = 1.0 - m * (s * s)
    z = torch.ones_like(x)

    return s * carlson_rf(x, y, z, tol=tol)
