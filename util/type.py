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
Shared tensor/array type conversion and validation helpers.

This module collects small, domain-neutral helpers for converting between
Python scalars, NumPy arrays, and torch tensors, for resolving the
real/complex dtype pairing used throughout tpeanuts, and for validating the
shape of flavour-vector-like arrays (last dimension of length 3).

Module functions:
    cdtype_from_real(...): Map a real floating dtype to its complex
        counterpart.
    real_dtype_from_tensor(...): Infer the real floating dtype matching a
        tensor's precision.
    as_tensor(...): Convert a value to a torch tensor with optional
        dtype/device/grad settings.
    as_tensor_like(...): Convert a value to a tensor matching an optional
        reference tensor.
    first_tensor(...): Return the first tensor found among several values.
    as_complex_tensor(...): Convert a value to a complex tensor derived from
        a real dtype.
    state_tensor(...): Convert and validate a 3-flavour state vector tensor.
    broadcast_last3(...): Broadcast a 3-component vector to a batch shape.
    to_numpy(...): Convert tensors and array-like values to a NumPy array.
    get_entry_tensor(...): Fetch and convert a dictionary entry to a tensor.
    as_datetime64(...): Convert a date-like value to a NumPy datetime64
        scalar.
    constant_float_row(...): Build a one-row NumPy array of a repeated
        value.
    ensure_1d(...): Validate and return a 1D NumPy array.
"""

import torch
import numpy as np
from typing import Any, Union

# Type alias for values accepted as scalar-or-tensor inputs throughout
# tpeanuts.
TensorLike = Union[float, int, torch.Tensor]


def cdtype_from_real(dtype: torch.dtype) -> torch.dtype:
    """
    Map a real floating torch dtype to its matching complex dtype.

    Used to derive the complex dtype for oscillation amplitudes and PMNS
    matrices from the real dtype used for angles and energies.

    Args:
        dtype: Real floating dtype, torch.float32 or torch.float64.

    Returns:
        torch.complex64 for float32, or torch.complex128 for float64.

    Raises:
        TypeError: If dtype is not torch.float32 or torch.float64.
    """
    if dtype == torch.float32:
        return torch.complex64
    elif dtype == torch.float64:
        return torch.complex128
    else:
        raise TypeError(
            f"Expected a real floating dtype (float32 or float64), got {dtype}"
        )

def real_dtype_from_tensor(tensor: torch.Tensor) -> torch.dtype:
    """
    Infer the real floating dtype matching a tensor's precision.

    Args:
        tensor: Tensor whose dtype (real or complex) determines the result.

    Returns:
        torch.float64 if tensor is complex128 or float64; torch.float32
        otherwise.
    """
    if tensor.dtype in (torch.complex128, torch.float64):
        return torch.float64
    return torch.float32

def as_tensor(
    x,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool | None = None,
) -> torch.Tensor:
    """
    Convert a value to a torch tensor, applying optional dtype/device/grad.

    Existing tensors are cast/moved in place rather than recreated from
    scratch, preserving the original tensor identity when no
    dtype/device change is requested.

    Args:
        x: Scalar, array-like value, or existing torch tensor.
        dtype: Optional dtype applied to the result.
        device: Optional device applied to the result.
        requires_grad: Optional autograd flag explicitly set on the result.

    Returns:
        Torch tensor representation of x.
    """
    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None:
            t = t.to(dtype=dtype)
        if device is not None:
            t = t.to(device=device)
    else:
        t = torch.as_tensor(x, dtype=dtype, device=device)

    if requires_grad is not None:
        t.requires_grad_(requires_grad)

    return t


def as_tensor_like(
    x: TensorLike,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert a value to a tensor matching an optional reference tensor.

    Args:
        x: Scalar or tensor-like value.
        reference: Optional tensor providing device and dtype.

    Returns:
        Tensor representation of ``x`` on the reference device/dtype, or a
        float64 tensor when no reference tensor is available.
    """
    if reference is not None:
        return torch.as_tensor(x, device=reference.device, dtype=reference.dtype)

    if torch.is_tensor(x):
        return x

    return torch.tensor(x, dtype=torch.float64)


def first_tensor(*values: TensorLike) -> torch.Tensor | None:
    """Return the first tensor found in a sequence of values.

    Args:
        *values: Candidate scalar or tensor-like values.

    Returns:
        First torch.Tensor in ``values``, or None when all values are scalars.
    """
    for value in values:
        if torch.is_tensor(value):
            return value

    return None


def as_complex_tensor(
    x,
    *,
    device=None,
    real_dtype=torch.float64,
):
    """
    Convert a value to a complex tensor derived from a real dtype.

    Args:
        x: Scalar, array-like value, or torch tensor.
        device: Optional device for the returned tensor.
        real_dtype: Real floating dtype (float32 or float64) whose matching
            complex dtype (complex64 or complex128) is used for the result.

    Returns:
        Complex torch tensor representation of x.

    Raises:
        TypeError: If real_dtype is not torch.float32 or torch.float64.
    """
    return as_tensor(
        x,
        device=device,
        dtype=cdtype_from_real(real_dtype),
    )

def state_tensor(
    state: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert and validate a 3-flavour neutrino state vector tensor.

    Args:
        state: Tensor-like state vector whose last dimension indexes the
            three flavour (or mass) amplitudes.
        device: Device for the returned tensor.
        dtype: Dtype for the returned tensor.

    Returns:
        Tensor representation of state on the requested device/dtype.

    Raises:
        ValueError: If the last dimension of state is not length 3.
    """
    state = as_tensor(state, device=device, dtype=dtype)

    if state.shape[-1] != 3:
        raise ValueError("state must have last dimension equal to 3.")

    return state


def broadcast_last3(vector: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
    """
    Broadcast a vector whose last dimension contains three components.

    Args:
        vector: Tensor with final dimension equal to three.
        batch_shape: Desired leading broadcast shape.

    Returns:
        Tensor with shape (*batch_shape, 3).

    Raises:
        ValueError: If vector does not have final dimension equal to three.
    """
    if vector.shape[-1] != 3:
        raise ValueError("vector must have last dimension equal to 3.")

    if len(batch_shape) == 0:
        return vector

    if vector.ndim == 1:
        return vector.expand(*batch_shape, 3)

    return torch.broadcast_to(vector, (*batch_shape, 3))

def _as_numpy(x, dtype=np.float64, ndim=None):
    """Convert a tensor or array-like value to a NumPy array (internal helper).

    Args:
        x: Torch tensor, NumPy array, scalar, or array-like value.
        dtype: NumPy dtype applied during conversion.
        ndim: Unused except to request a reshape check for ndim == 1; note
            that the reshape result is currently not assigned back, so this
            argument has no effect on the returned array's shape.

    Returns:
        NumPy array detached from autograd and moved to CPU when x is a
        torch tensor.
    """
    if torch.is_tensor(x):
        x = x.detach()
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    x = np.asarray(x, dtype=dtype)
    if ndim is not None:
        if ndim==1:
            x.reshape(-1)
    return x


def to_numpy(x: Any, dtype: Any | None = None) -> np.ndarray:
    """
    Convert tensors and array-like values to a NumPy array.

    Args:
        x: Torch tensor, NumPy array, scalar, or array-like object.
        dtype: Optional NumPy dtype applied during conversion.

    Returns:
        NumPy array detached from autograd and moved to CPU when x is a torch
        tensor.
    """
    if torch.is_tensor(x):
        x = x.detach()
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()

    return np.asarray(x, dtype=dtype)


def get_entry_tensor(
    entry: dict[str, Any],
    key: str,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Fetch a dictionary entry and convert it to a torch tensor.

    Args:
        entry: Mapping containing tensor-like values.
        key: Required entry key.
        dtype: Optional torch dtype for the returned tensor.
        device: Optional torch device for the returned tensor.

    Returns:
        Tensor representation of entry[key].

    Raises:
        KeyError: If key is not present in entry.
    """
    if key not in entry:
        raise KeyError(
            f"Entry does not contain {key!r}. Available keys: {sorted(entry.keys())}"
        )

    return torch.as_tensor(entry[key], device=device, dtype=dtype)

def as_datetime64(value, name="date") -> np.datetime64:
    """
    Convert a date-like value into a NumPy datetime64 scalar.

    Args:
        value: String accepted by np.datetime64 or an existing np.datetime64
            scalar. Typical values are ISO-8601 strings such as
            "2026-05-10T12:00".
        name: Name used in the error message if conversion fails.

    Returns:
        np.datetime64 scalar preserving the supplied date/time precision.
    """
    if isinstance(value, np.datetime64):
        return value

    try:
        return np.datetime64(value)
    except Exception as exc:
        raise ValueError(f"{name} must be convertible to np.datetime64.") from exc

def constant_float_row(value, length: int, name="value") -> np.ndarray:
    """
    Build a one-row NumPy array filled with a repeated floating value.

    Args:
        value: Scalar value convertible to float.
        length: Number of columns in the output row. Must be positive.
        name: Name used in validation error messages.

    Returns:
        NumPy array with shape (1, length) and dtype float.
    """
    if int(length) <= 0:
        raise ValueError("length must be a positive integer.")

    try:
        value_f = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be convertible to float.") from exc

    return np.full((1, int(length)), value_f, dtype=float)

def ensure_1d(x, name="x"):
    """
    Validate and return a one-dimensional NumPy array.

    Args:
        x: Tensor or array-like value. Scalars are promoted to length one.
        name: Name used in validation error messages.

    Returns:
        NumPy array with exactly one dimension.

    Raises:
        ValueError: If x is None, or has more than one dimension.
    """
    x = _as_numpy(x)
    if x is None:
        raise ValueError(f"{name} is required.")
    x = np.asarray(x)
    if x.ndim == 0:
        x = x[None]
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D. Got shape {x.shape}")
    return x

def _ensure_2d(x, name="x"):
    """
    Validate and return a (N, 3) flavour-vector-like NumPy array.

    A 1D input of length 3 (a single flavour vector) is promoted to shape
    (1, 3).

    Args:
        x: Tensor or array-like value.
        name: Name used in validation error messages.

    Returns:
        NumPy array with shape (N, 3).

    Raises:
        ValueError: If x is None, or cannot be interpreted as shape (N, 3).
    """
    x = _as_numpy(x)
    if x is None:
        raise ValueError(f"{name} is required.")
    x = np.asarray(x)
    if x.ndim == 1:
        # allow (3,) -> (1,3)
        if x.shape[0] == 3:
            x = x[None, :]
        else:
            raise ValueError(f"{name} 1D must be length 3. Got {x.shape}")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got shape {x.shape}")
    if x.shape[1] != 3:
        raise ValueError(f"{name} must have last dim=3. Got shape {x.shape}")
    return x

def _ensure_3d(x, name="x"):
    """
    Validate and return an (Ne, Neta, 3) flavour-vector-grid NumPy array.

    Args:
        x: Tensor or array-like value representing a grid of flavour
            vectors over an energy axis (Ne) and an angular axis (Neta).
        name: Name used in validation error messages.

    Returns:
        NumPy array with shape (Ne, Neta, 3).

    Raises:
        ValueError: If x is None, not 3D, or its last dimension is not 3.
    """
    x = _as_numpy(x)
    if x is None:
        raise ValueError(f"{name} is required.")
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"{name} must be (Ne,Neta,3). Got shape {x.shape}")
    return x
