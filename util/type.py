

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

import torch
import numpy as np
from typing import Any, Union

TensorLike = Union[float, int, torch.Tensor]


def _cdtype_from_real(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
        return torch.complex64
    elif dtype == torch.float64:
        return torch.complex128
    else:
        raise TypeError(
            f"Expected a real floating dtype (float32 or float64), got {dtype}"
        )

def _as_tensor(
    x,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool | None = None,
) -> torch.Tensor:
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

def _as_complex_tensor(
    x,
    *,
    device=None,
    real_dtype=torch.float64,
):
    return _as_tensor(
        x,
        device=device,
        dtype=_cdtype_from_real(real_dtype),
    )

def _state_tensor(
    state: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    state = _as_tensor(state, device=device, dtype=dtype)
    
    if state.shape[-1] != 3:
        raise ValueError("state must have last dimension equal to 3.")

    return state

def _as_numpy(x, dtype=np.float64, ndim=None):
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

def _as_datetime64(value, name="date") -> np.datetime64:
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

def _constant_float_row(value, length: int, name="value") -> np.ndarray:
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

def _ensure_1d(x, name="x"):
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
    x = _as_numpy(x)
    if x is None:
        raise ValueError(f"{name} is required.")
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"{name} must be (Ne,Neta,3). Got shape {x.shape}")
    return x
