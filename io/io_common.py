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
Common I/O helpers shared by several TPeanuts domains.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def safe_filename_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("+", "plus")
        .replace("-", "minus")
        .replace(":", "_")
        .replace(";", "_")
        .replace(",", "_")
    )


def angle_to_filename(angle_deg: float) -> str:
    return f"{angle_deg:.3f}".replace(".", "p").replace("-", "m")


def ensure_torch_extension(filename: str) -> str:
    base_name, ext = os.path.splitext(filename)

    if ext == "":
        return base_name + ".pt"

    if ext.lower() in {".pt", ".pth"}:
        return filename

    return base_name + ".pt"


def build_output_path(output_dir: str, filename: str) -> str:
    return os.path.join(output_dir, filename)


def torch_load_file(
    input_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    try:
        return torch.load(
            input_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            input_path,
            map_location=map_location,
        )


def tensor_shape_dict(result: Dict[str, Any]) -> Dict[str, tuple]:
    shapes = {}

    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            shapes[key] = tuple(value.shape)

    return shapes


def scalar_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())

    return float(value)


def cast_tensor_tree(
    obj: Any,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
) -> Any:
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


def ensure_output_directory(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def list_torch_files(directory: str) -> list[str]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith((".pt", ".pth"))
    ]
    files.sort()

    return files
