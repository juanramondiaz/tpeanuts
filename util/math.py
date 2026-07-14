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
Generic mathematical helpers for TPeanuts.

This module collects small tensor-oriented utilities shared by several physics
domains. The functions here are intentionally domain-neutral: interval
operations, trapezoidal normalization, interpolation, relative-error summaries,
complex trigonometric wrappers, elliptic-integral helpers, and basic linear
algebra projections.

Functions
---------
intersection(...)
    Return the common overlap of one or more closed intervals.
_trapz(...)
    Compute trapezoidal integration along the last tensor axis.
normalize_trapz(...)
    Normalize a tensor so its trapezoidal integral is one along a chosen axis.
trapz_weights(...)
    Build one-dimensional trapezoidal quadrature weights for a coordinate grid.
relative_error_summary(...)
    Summarize maximum per-flavour and total relative errors.
clamp_positive(...)
    Clamp tensor-like values to a strictly positive lower bound.
nearest_index(...)
    Return the index of the grid point closest to a target value.
selected_indices(...)
    Select approximately evenly spaced indices from a range.
relative_error(...)
    Compute element-wise relative error against a reference value.
csin(...), ccos(...), ctan(...), casin(...)
    Thin torch trigonometric wrappers used by analytic formulae.
csqrt(...)
    Convert real tensors to complex tensors before taking a square root.
sec(...), csc(...)
    Compute secant and cosecant.
carlson_rf(...)
    Evaluate Carlson's symmetric elliptic integral RF(x, y, z).
ellipf_incomplete(...)
    Evaluate the incomplete elliptic integral of the first kind F(phi | m).
project_to_unitary(...)
    Project a square matrix onto the nearest unitary matrix via SVD.
interp1d_linear(...)
    Perform one-dimensional linear interpolation with explicit edge values.
interp_logx(...)
    Perform one-dimensional interpolation in log10(x), useful for log-spaced
    positive grids.
numpy_trapezoid(...)
    NumPy trapezoidal integration wrapper compatible with both old and new
    NumPy versions.
tree_reduce_matmul(...)
    Reduce a stack of square matrices to a single product via a binary tree.
binom(...)
    Return a binomial coefficient as a torch scalar.
"""



from __future__ import annotations

import torch
import numpy as np

from typing import Any, Dict, Union, Optional

from tpeanuts.util.type import TensorLike, as_tensor, to_numpy


def intersection(
    i1: torch.Tensor,
    *intervals: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the intersection of closed 1D intervals.

    Args:
        i1: First interval as a tensor with shape (2,), ordered as
            [lower, upper]. An empty tensor represents an empty interval.
        *intervals: Additional intervals with the same shape convention.

    Returns:
        Tensor with shape (2,) containing the common overlap. If the overlap is
        empty, returns an empty tensor on the same device and dtype as i1.

    Raises:
        ValueError: If any non-empty interval does not contain exactly two
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


def interp_logx(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate values in log10(x).

    Args:
        x: Query points. Values must be positive.
        xp: Sample x-coordinates. Values must be positive and sorted as
            expected by ``numpy.interp`` after log10 transformation.
        fp: Sample values at xp.

    Returns:
        Interpolated values evaluated at x.
    """
    return np.interp(np.log10(x), np.log10(xp), fp)


def numpy_trapezoid(
    y,
    x=None,
    dx: float = 1.0,
    axis: int = -1,
):
    """
    Integrate with NumPy's trapezoidal rule across NumPy versions.

    NumPy 2.x exposes ``numpy.trapezoid`` while older environments commonly
    use ``numpy.trapz``. This wrapper keeps project code independent of the
    installed NumPy spelling.

    Args:
        y: Values to integrate.
        x: Optional coordinates.
        dx: Sample spacing used when x is not provided.
        axis: Axis along which to integrate.

    Returns:
        Integral computed by ``numpy.trapezoid`` when available, otherwise
        by ``numpy.trapz``.
    """
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = getattr(np, "trapz")
    return trapezoid(y, x=x, dx=dx, axis=axis)


def _trapz(
    y: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Integrate samples with the trapezoidal rule along the last axis.

    Args:
        y: Sample values. The last dimension is interpreted as the integration
            axis.
        x: Coordinates associated with y along the last dimension. Must be
            broadcast-compatible with y over the integration axis.

    Returns:
        Tensor with the last dimension reduced by trapezoidal integration.
    """
    return torch.sum(
        0.5
        * (y[..., 1:] + y[..., :-1])
        * (x[..., 1:] - x[..., :-1]),
        dim=-1,
    )


@torch.no_grad()
def normalize_trapz(
    y: torch.Tensor,
    x: Optional[torch.Tensor] = None,
    *,
    dim: int = -1,
    clamp_min: Optional[float] = 0.0,
    eps: float = 1.0e-30,
    inplace: bool = False,
    return_norm: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize samples so their trapezoidal integral is one.

    Args:
        y: Values to normalize.
        x: Optional coordinate tensor passed to torch.trapz. If omitted, unit
            spacing is assumed.
        dim: Axis along which the integral is computed.
        clamp_min: Optional lower bound applied to y before normalization. Set
            to None to skip clamping.
        eps: Minimum normalization denominator used to avoid division by zero.
        inplace: If True, modify y in place after optional clamping.
        return_norm: If True, also return the unclamped integral denominator
            after broadcasting dimensions are removed.

    Returns:
        Normalized tensor. If return_norm is True, returns
        (normalized_tensor, norm).
    """
    if clamp_min is not None:

        if inplace:
            y.clamp_(min=clamp_min)
        else:
            y = torch.clamp(y, min=clamp_min)

    elif not inplace:
        y = y.clone()

    # --------------------------------------------------------
    # Integral
    # --------------------------------------------------------

    if x is None:
        norm = torch.trapz(y, dim=dim)
    else:
        norm = torch.trapz(y, x=x, dim=dim)

    norm = torch.clamp(norm, min=eps)

    # --------------------------------------------------------
    # Broadcast norm
    # --------------------------------------------------------

    while norm.ndim < y.ndim:
        norm = norm.unsqueeze(dim)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    if inplace:
        y /= norm
        out = y
    else:
        out = y / norm

    if return_norm:
        return out, norm.squeeze()

    return out


def trapz_weights(grid: torch.Tensor) -> torch.Tensor:
    """Build trapezoidal quadrature weights for a one-dimensional grid.

    Args:
        grid: One-dimensional coordinate tensor.

    Returns:
        Tensor with the same shape as ``grid`` containing the trapezoidal
        integration weights. A single-point grid receives weight one.

    Raises:
        ValueError: If ``grid`` is not one-dimensional.
    """
    if grid.ndim != 1:
        raise ValueError("Trapz weights require a one-dimensional grid.")
    if grid.numel() == 1:
        return torch.ones_like(grid)

    weights = torch.empty_like(grid)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    if grid.numel() > 2:
        weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


def relative_error_summary(
    value: Any,
    reference: Any,
    *,
    flavour_order: Optional[list[str]] = None,
    eps: float = 1.0e-12,
) -> Dict[str, Any]:
    """
    Compute relative-error diagnostics for flavour-resolved arrays.

    Args:
        value: Candidate values. The final dimension must have length three.
        reference: Reference values with the same shape as value.
        flavour_order: Optional names assigned to the three final-dimension
            entries. Defaults to ["nue", "numu", "nutau"].
        eps: Minimum reference magnitude used in denominators.

    Returns:
        Dictionary with "by_flavour" maximum relative error per flavour and
        "total" aggregate L1 relative error.

    Raises:
        ValueError: If value and reference shapes differ or the final dimension
            is not length three.
    """
    value_t = torch.as_tensor(value, dtype=torch.float64)
    reference_t = torch.as_tensor(reference, dtype=torch.float64)

    if value_t.shape != reference_t.shape:
        raise ValueError(
            "value and reference must have the same shape. "
            f"Got {tuple(value_t.shape)} and {tuple(reference_t.shape)}."
        )

    if value_t.shape[-1] != 3:
        raise ValueError("Expected final dimension to contain three flavours.")

    if flavour_order is None:
        flavour_order = ["nue", "numu", "nutau"]

    rel = torch.abs(value_t - reference_t) / torch.clamp(
        torch.abs(reference_t),
        min=eps,
    )

    rel_flat = rel.reshape(-1, 3)

    by_flavour = {
        flavour_order[i]: float(torch.max(rel_flat[:, i]).item())
        for i in range(3)
    }

    total = float(
        (
            torch.sum(torch.abs(value_t - reference_t))
            / torch.clamp(torch.sum(torch.abs(reference_t)), min=eps)
        ).item()
    )

    return {
        "by_flavour": by_flavour,
        "total": total,
    }


def clamp_positive(x: Any, eps: float = 1.0e-30) -> Any:
    """
    Clamp tensor-like values to a strictly positive lower bound.

    Args:
        x: Torch tensor or array-like values to clamp.
        eps: Minimum returned value.

    Returns:
        Clamped values. Torch tensor inputs return tensors; other array-like
        inputs return NumPy arrays.
    """
    if torch.is_tensor(x):
        return torch.clamp(x, min=eps)

    return np.maximum(np.asarray(x), eps)


def nearest_index(grid: Any, value: float) -> int:
    """
    Return the index of the grid point closest to value.

    Args:
        grid: One-dimensional tensor-like coordinate grid.
        value: Coordinate value to locate.

    Returns:
        Integer index of the nearest grid entry.
    """
    grid_t = torch.as_tensor(grid)
    return int(torch.argmin(torch.abs(grid_t - float(value))).item())


def selected_indices(n: int, max_count: int) -> torch.Tensor:
    """
    Select approximately evenly spaced indices from a range.

    Args:
        n: Total number of available entries.
        max_count: Maximum number of indices to return.

    Returns:
        Sorted tensor of unique integer indices.
    """
    if int(n) <= 0 or int(max_count) <= 0:
        return torch.empty(0, dtype=torch.long)

    return torch.linspace(
        0,
        int(n) - 1,
        min(int(max_count), int(n)),
    ).round().long().unique()


def relative_error(value: Any, reference: Any, eps: float = 1.0e-15) -> np.ndarray:
    """
    Compute element-wise relative error against a reference value.

    Args:
        value: Candidate tensor-like values.
        reference: Reference tensor-like values.
        eps: Minimum denominator magnitude.

    Returns:
        NumPy array containing abs(value - reference) /
        max(abs(reference), eps).
    """
    value_np = to_numpy(value, dtype=float)
    reference_np = to_numpy(reference, dtype=float)
    return np.abs(value_np - reference_np) / np.maximum(np.abs(reference_np), eps)

def csin(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute sine using torch.sin.

    Args:
        x: Real or complex tensor of angles in radians.

    Returns:
        Tensor with sin(x), preserving torch broadcasting and dtype behavior.
    """
    return torch.sin(x)


def ccos(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine using torch.cos.

    Args:
        x: Real or complex tensor of angles in radians.

    Returns:
        Tensor with cos(x), preserving torch broadcasting and dtype behavior.
    """
    return torch.cos(x)


def ctan(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute tangent using torch.tan.

    Args:
        x: Real or complex tensor of angles in radians.

    Returns:
        Tensor with tan(x), preserving torch broadcasting and dtype behavior.
    """
    return torch.tan(x)


def casin(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute inverse sine using torch.asin.

    Args:
        x: Real or complex tensor.

    Returns:
        Tensor with asin(x), preserving torch broadcasting and dtype behavior.
    """
    return torch.asin(x)


def csqrt(
    z: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """
    Compute a complex-valued square root.

    Args:
        z: Input tensor. Real tensors are promoted to complex64 or complex128
            depending on their floating dtype.
        eps: Optional imaginary offset added before taking the square root.
            This can be used to choose a branch consistently near the real
            axis.

    Returns:
        Complex tensor containing sqrt(z + i eps).
    """
    if not torch.is_complex(z):
        z = z.to(
            torch.complex128
            if z.dtype == torch.float64
            else torch.complex64
        )

    if eps != 0.0:
        z = z + 1j * z.new_tensor(eps)

    return torch.sqrt(z)


def sec(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute secant, 1 / cos(x).

    Args:
        x: Real or complex tensor of angles in radians.

    Returns:
        Tensor containing sec(x).
    """
    return 1.0 / torch.cos(x)


def csc(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosecant, 1 / sin(x).

    Args:
        x: Real or complex tensor of angles in radians.

    Returns:
        Tensor containing csc(x).
    """
    return 1.0 / torch.sin(x)


@torch.no_grad()
def carlson_rf(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    tol: float = 1.0e-12,
    max_iter: int = 50,
) -> torch.Tensor:
    """
    Evaluate Carlson's symmetric elliptic integral RF(x, y, z).

    The implementation uses the duplication algorithm and a final series
    correction. Inputs are promoted to a complex dtype so the same code path can
    support real and complex analytical expressions.

    Args:
        x: First RF argument.
        y: Second RF argument.
        z: Third RF argument.
        tol: Convergence tolerance for the duplication iterations.
        max_iter: Maximum number of duplication iterations.

    Returns:
        Tensor containing RF(x, y, z), broadcast over input shapes.
    """
    if not torch.is_complex(x):
        x = x.to(
            torch.complex128
            if x.dtype == torch.float64
            else torch.complex64
        )

    if not torch.is_complex(y):
        y = y.to(dtype=x.dtype)

    if not torch.is_complex(z):
        z = z.to(dtype=x.dtype)

    xn = x
    yn = y
    zn = z

    for _ in range(max_iter):

        mu = (xn + yn + zn) / 3.0

        X = 1.0 - xn / mu
        Y = 1.0 - yn / mu
        Z = 1.0 - zn / mu

        err = torch.max(
            torch.stack(
                [
                    torch.abs(X),
                    torch.abs(Y),
                    torch.abs(Z),
                ],
                dim=0,
            ),
            dim=0,
        ).values

        if torch.all(err < tol):
            break

        sx = csqrt(xn)
        sy = csqrt(yn)
        sz = csqrt(zn)

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

    series = (
        1.0
        - E2 / 10.0
        + E3 / 14.0
        + (E2 * E2) / 24.0
        - (3.0 * E2 * E3) / 44.0
    )

    return series / csqrt(mu)


@torch.no_grad()
def ellipf_incomplete(
    phi: torch.Tensor,
    m: torch.Tensor,
    tol: float = 1.0e-12,
) -> torch.Tensor:
    """
    Evaluate the incomplete elliptic integral of the first kind.

    Computes F(phi | m) through Carlson RF using
    F(phi | m) = sin(phi) RF(cos(phi)^2, 1 - m sin(phi)^2, 1).

    Args:
        phi: Amplitude angle in radians.
        m: Elliptic parameter.
        tol: Convergence tolerance passed to carlson_rf.

    Returns:
        Tensor containing F(phi | m), broadcast over input shapes.
    """
    if not torch.is_complex(phi):
        phi = phi.to(
            torch.complex128
            if phi.dtype == torch.float64
            else torch.complex64
        )

    if not torch.is_complex(m):
        m = m.to(dtype=phi.dtype)

    s = torch.sin(phi)
    c = torch.cos(phi)

    x = c * c
    y = 1.0 - m * (s * s)
    z = torch.ones_like(x)

    return s * carlson_rf(
        x,
        y,
        z,
        tol=tol,
    )


def project_to_unitary(
    S: torch.Tensor,
) -> torch.Tensor:
    """
    Project a matrix onto the closest unitary matrix in Frobenius norm.

    Args:
        S: Square matrix or batch of square matrices.

    Returns:
        Unitary factor U @ Vh from the singular value decomposition of S.
    """
    U, _, Vh = torch.linalg.svd(S)

    return U @ Vh

@torch.no_grad()
def interp1d_linear(
    x: TensorLike,
    xp: torch.Tensor,
    fp: torch.Tensor,
    *,
    left: Optional[TensorLike] = None,
    right: Optional[TensorLike] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Linearly interpolate one-dimensional tabulated data.

    Args:
        x: Query points. Scalar, array-like, or tensor.
        xp: 1D tensor of tabulated coordinates. Values are assumed to be sorted
            in ascending order.
        fp: 1D tensor of tabulated values with the same length as xp.
        left: Optional fill value for x below xp[0]. Defaults to fp[0].
        right: Optional fill value for x above xp[-1]. Defaults to fp[-1].
        device: Optional device for the computation. Defaults to xp.device.
        dtype: Floating dtype used for interpolation.

    Returns:
        Tensor with interpolated values matching the shape of x.

    Raises:
        ValueError: If xp or fp are not 1D, have different lengths, or contain
            fewer than two points.
    """
    x = as_tensor(x, device=device if device is not None else xp.device, dtype=dtype)

    xp = xp.to(device=x.device, dtype=dtype)
    fp = fp.to(device=x.device, dtype=dtype)

    if xp.ndim != 1 or fp.ndim != 1:
        raise ValueError("xp and fp must be 1D tensors.")

    if xp.numel() != fp.numel():
        raise ValueError("xp and fp must have the same length.")

    if xp.numel() < 2:
        raise ValueError("At least two interpolation points are required.")

    left_val = fp[0] if left is None else as_tensor(left, device=x.device, dtype=dtype)
    right_val = fp[-1] if right is None else as_tensor(right, device=x.device, dtype=dtype)

    idx = torch.searchsorted(xp, x, right=False)

    below = idx <= 0
    above = idx >= xp.numel()

    idx0 = torch.clamp(idx - 1, 0, xp.numel() - 2)
    idx1 = idx0 + 1

    x0 = xp[idx0]
    x1 = xp[idx1]
    y0 = fp[idx0]
    y1 = fp[idx1]

    denom = torch.clamp(x1 - x0, min=torch.finfo(dtype).eps)
    t = (x - x0) / denom

    y = y0 + t * (y1 - y0)

    y = torch.where(below, torch.zeros_like(y) + left_val, y)
    y = torch.where(above, torch.zeros_like(y) + right_val, y)

    return y

def tree_reduce_matmul(mats: torch.Tensor, *, left: bool = True) -> torch.Tensor:
    """Reduce a stack of square matrices to a single product using a binary tree.

    The reduction proceeds by pairwise matmul at each level, halving the stack
    depth per level.  The total number of batched matmul launches is O(N log N)
    instead of the O(N) sequential launches of a plain loop.  On GPU the paired
    operations at each level are independent and can be dispatched in parallel.

    Args:
        mats: Tensor shaped ``(..., N, d, d)`` where ``N`` is the number of
            matrices to reduce and ``d`` is the matrix size.  The ``N``
            dimension is the penultimate-2 axis (index ``-3`` for square
            matrices).
        left: When ``True`` (default), the product is evaluated left-to-right:
            ``mats[..., 0, :, :] @ mats[..., 1, :, :] @ ... @ mats[..., N-1, :, :]``.
            When ``False``, the order is reversed.

    Returns:
        Reduced matrix shaped ``(..., d, d)``.

    Raises:
        ValueError: If ``mats`` has fewer than three dimensions or the last two
            dimensions are not square.
    """
    if mats.ndim < 3:
        raise ValueError("mats must have at least 3 dimensions (..., N, d, d).")
    d1, d2 = mats.shape[-2], mats.shape[-1]
    if d1 != d2:
        raise ValueError(
            f"mats must have square matrices in the last two dimensions, got ({d1}, {d2})."
        )

    # Single-matrix shortcut.
    if mats.shape[-3] == 1:
        return mats[..., 0, :, :]

    while mats.shape[-3] > 1:
        n = mats.shape[-3]
        if n % 2 == 1:
            # Odd length: carry the last matrix forward unpaired.
            tail = mats[..., -1:, :, :]
            even = mats[..., 0:-1:2, :, :]   # indices 0, 2, 4, …
            odd  = mats[..., 1:-1:2, :, :]   # indices 1, 3, 5, …
            paired = even @ odd if left else odd @ even
            mats = torch.cat([paired, tail], dim=-3)
        else:
            even = mats[..., 0::2, :, :]
            odd  = mats[..., 1::2, :, :]
            mats = even @ odd if left else odd @ even

    return mats[..., 0, :, :]


def binom(
    n: int,
    k: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return a binomial coefficient as a torch scalar.

    Args:
        n: Upper integer in "n choose k".
        k: Lower integer in "n choose k".
        device: Device for the returned tensor.
        dtype: Dtype for the returned tensor.

    Returns:
        Scalar tensor containing C(n, k).
    """
    if k == 0 or k == n:
        return torch.tensor(1.0, device=device, dtype=dtype)

    numerator = 1.0
    denominator = 1.0

    for i in range(1, k + 1):
        numerator *= n + 1 - i
        denominator *= i

    return torch.tensor(
        numerator / denominator,
        device=device,
        dtype=dtype,
    )
