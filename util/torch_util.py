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
Shared PyTorch device, dtype, broadcasting, and flattening utilities.

Module functions:
    default_device(...): Resolve None to the project default CUDA/CPU device.
    resolve_device(...): Resolve the configured torch device.
    infer_device_dtype(...): Infer device and dtype from tensor inputs.
    resolve_dtype(...): Resolve the real floating-point dtype.
    scalar_float(...): Convert a scalar tensor-like value to a Python float.
    as_1d_tensor(...): Convert scalar or one-dimensional input to a 1D tensor.
    cast_tensor_tree(...): Recursively detach, move, and cast tensor leaves.
    broadcast_tensor(...): Convert and broadcast generic tensor-like values.
    flat_tensor(...): Flatten a tensor and optionally select indexed entries.
    identity_evolutor_like(...): Build scalar or batched NxN identities.
    enforce_identity_for_zero_length(...): Replace zero-length evolutors with
        the identity.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from tpeanuts.util.type import TensorLike

# ============================================================
# Tensor helpers
# ============================================================


def identity_evolutor_like(
    L: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n: int = 3,
) -> torch.Tensor:
    """Build an nxn identity evolutor matching a path-length batch.

    Args:
        L: Dimensionless scalar or batched segment length.
        device: Device used for the identity tensor.
        dtype: Real or complex dtype used for the identity tensor.
        n: Evolutor dimension (3 for the SM, 4 for the 3+1 sterile
            extension).

    Returns:
        Identity evolutor shaped L.shape + (n, n).
    """
    identity = torch.eye(n, device=device, dtype=dtype)
    if L.ndim == 0:
        return identity

    return identity.expand(*L.shape, n, n)


def enforce_identity_for_zero_length(
    U: torch.Tensor,
    L: torch.Tensor,
    zero_mask: torch.Tensor,
) -> torch.Tensor:
    """Replace evolution operators on zero-length segments with the identity.

    Args:
        U: Evolution operators shaped (..., N, N), N in {3, 4}.
        L: Segment lengths. Retained alongside the mask for API clarity.
        zero_mask: Boolean tensor selecting zero-length segments.

    Returns:
        Evolution operators with identities at the selected positions.
    """
    del L
    identity = identity_evolutor_like(
        zero_mask,
        device=U.device,
        dtype=U.dtype,
        n=U.shape[-1],
    )
    return torch.where(zero_mask[..., None, None], identity, U)


def default_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve the explicit device or the project default CUDA/CPU device.

    Args:
        device: Device specification accepted by torch.device, or None to use
            CUDA when available and CPU otherwise.

    Returns:
        Resolved torch.device instance.
    """
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve a concrete torch device from a value or a deferred device factory.

    Args:
        device: Device specification accepted by torch.device, None for the
            default CUDA/CPU selection, or a callable returning either form.

    Returns:
        Resolved torch.device instance.
    """
    if callable(device):
        return device()
    return default_device(device)


def infer_device_dtype(
    *values: TensorLike,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.device, torch.dtype]:
    """
    Infer a device and dtype from explicit overrides or tensor inputs.

    The first tensor supplies any value not explicitly provided. When no
    tensor is available, the project default device and float64 are used.

    Args:
        *values: Candidate scalar or tensor-like values inspected in order.
        device: Optional explicit device override; bypasses tensor
            inference when provided.
        dtype: Optional explicit dtype override; bypasses tensor inference
            when provided.

    Returns:
        Tuple (device, dtype) resolved from the explicit overrides, the
        first tensor found in values, or the project defaults (CUDA/CPU,
        float64) when neither is available.
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


def resolve_dtype(dtype: Optional[torch.dtype], *values) -> torch.dtype:
    """
    Resolve a real floating torch dtype from an explicit dtype or input values.

    Args:
        dtype: Optional dtype requested by the caller. When provided, it is
            returned unchanged.
        *values: Candidate values inspected in order. The first tensor with
            dtype torch.float32 or torch.float64 determines the result.

    Returns:
        Explicit dtype, the first floating tensor dtype found in values, or
        torch.float64 when no suitable tensor dtype is available.
    """
    if dtype is not None:
        return dtype

    for value in values:
        if torch.is_tensor(value) and value.dtype in (torch.float32, torch.float64):
            return value.dtype

    return torch.float64


def scalar_float(value: TensorLike) -> float:
    """
    Convert a tensor-like scalar to a plain Python float.

    Args:
        value: Scalar or tensor-like value; only the first element is used if
            more than one is given.

    Returns:
        The value as a Python float.
    """
    value_t = torch.as_tensor(value, device="cpu", dtype=torch.float64)
    return float(value_t.detach().cpu().reshape(-1)[0].item())


def as_1d_tensor(
    value: TensorLike,
    *,
    name: str = "value",
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Convert a scalar or one-dimensional value into a 1D tensor.

    Args:
        value: Scalar, array-like value, or tensor.
        name: Name used in validation error messages.
        device: Optional device for the returned tensor.
        dtype: Optional dtype for the returned tensor.

    Returns:
        Tensor with one dimension. Scalar inputs are promoted to length one.

    Raises:
        ValueError: If the converted value has more than one dimension.
    """
    out = torch.as_tensor(value, device=device, dtype=dtype)
    if out.ndim == 0:
        out = out[None]
    if out.ndim != 1:
        raise ValueError(f"{name} must be scalar or one-dimensional.")
    return out


def cast_tensor_tree(
    obj,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    """Recursively detach, move, and cast tensor leaves in a Python tree.

    Args:
        obj: Tensor, mapping, list, tuple, or any non-tensor object.
        dtype: Optional floating dtype applied to floating-point tensor leaves.
        device: Optional device applied to all tensor leaves.

    Returns:
        Object with the same container structure. Tensor leaves are detached,
        optionally moved to ``device``, and optionally cast to ``dtype`` when
        they are floating-point tensors. Non-tensor leaves are returned
        unchanged.
    """
    if isinstance(obj, torch.Tensor):
        x = obj.detach()

        if device is not None:
            x = x.to(device=device)

        if dtype is not None and torch.is_floating_point(x):
            x = x.to(dtype=dtype)

        return x

    if isinstance(obj, dict):
        return {
            key: cast_tensor_tree(
                value,
                dtype=dtype,
                device=device,
            )
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            cast_tensor_tree(
                value,
                dtype=dtype,
                device=device,
            )
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            cast_tensor_tree(
                value,
                dtype=dtype,
                device=device,
            )
            for value in obj
        )

    return obj


def broadcast_tensor(
    *values: TensorLike,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[torch.dtype, Sequence[Optional[torch.dtype]]]] = None,
    independent_1d: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Convert values to tensors and broadcast them to a common shape.

    Args:
        *values: Tensor-like values to convert and broadcast.
        device: Optional common output device.
        dtype: Common dtype or one optional dtype per input value.
        independent_1d: Treat two unequal 1D inputs as independent grid axes.

    Returns:
        Tuple of tensors with a common broadcast shape.

    Raises:
        ValueError: If values is empty or the dtype sequence length is invalid.
    """
    if not values:
        raise ValueError("broadcast_tensor requires at least one value.")

    tensor_value = next((v for v in values if torch.is_tensor(v)), None)
    output_device = (
        tensor_value.device if device is None and tensor_value is not None
        else default_device() if device is None
        else torch.device(device)
    )

    if isinstance(dtype, Sequence):
        dtypes = tuple(dtype)
        if len(dtypes) != len(values):
            raise ValueError("dtype must contain one entry per input value.")
    else:
        dtypes = (dtype,) * len(values)

    tensors = tuple(
        torch.as_tensor(value, device=output_device, dtype=value_dtype)
        for value, value_dtype in zip(values, dtypes)
    )
    if (
        independent_1d
        and len(tensors) == 2
        and tensors[0].ndim == tensors[1].ndim == 1
        and tensors[0].numel() != tensors[1].numel()
    ):
        tensors = (tensors[0][:, None], tensors[1][None, :])

    return tuple(torch.broadcast_tensors(*tensors))


def flat_tensor(
    value: TensorLike,
    indices: Optional[torch.Tensor] = None,
) -> TensorLike:
    """Flatten a tensor and optionally select entries by flat index.

    Args:
        value: Tensor to flatten; non-tensor scalars remain unchanged.
        indices: Optional integer tensor selecting flattened entries.

    Returns:
        Flattened tensor, selected entries, or the unchanged scalar.
    """
    if not torch.is_tensor(value):
        return value
    flattened = value.reshape(-1)
    return flattened if indices is None else flattened[indices]

