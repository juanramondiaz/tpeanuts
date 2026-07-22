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

Module functions:
    package_dir(...): Return the TPeanuts package/repository root.
    load_datafile_2column(...): Load, sort, and return the first two numeric
        columns of a whitespace/comma-delimited text file.
    safe_filename_name(...): Sanitize a string into a filesystem-safe name.
    angle_to_filename(...): Format an angle in degrees as a filename-safe
        token.
    ensure_torch_extension(...): Ensure a filename carries a recognized torch
        tensor file extension.
    build_output_path(...): Join an output directory and filename.
    torch_load_file(...): Load a torch file, falling back when weights_only
        is unsupported.
    tensor_shape_dict(...): Collect the shapes of tensor values in a dict.
    scalar_float_or_none(...): Convert a scalar-like value to a Python float
        or None.
    ensure_output_directory(...): Create an output directory if missing.
    list_torch_files(...): List and sort torch tensor files in a directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

import tpeanuts.config.default as default
from tpeanuts.util.torch_util import default_device


def package_dir() -> Path:
    """Return the TPeanuts package/repository root containing ``data/``.

    Returns:
        Directory one level above ``util/``.
    """
    return Path(__file__).resolve().parents[1]


@torch.no_grad()
def load_datafile_2column(
    file: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load, sort, and return the first two numeric columns of a text file.

    Lines that are blank or start with "#" are skipped. Each remaining line
    is split on whitespace or commas, and the first two numeric fields are
    read. Rows are returned sorted by ascending value of the first column,
    which is the convention expected by interpolation helpers such as
    interp1d_linear.

    Args:
        file: Path to the text file to load.
        device: Optional device for the returned tensors. Defaults to the
            project default CUDA/CPU device.
        dtype: Floating dtype used for the returned tensors.

    Returns:
        Tuple (values_1, values_2) of 1D tensors sorted by ascending
        values_1.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If no valid two-column numeric rows are found.
    """
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File not found: {file}")

    column_1 = []
    column_2 = []

    with open(file, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.replace(",", " ").split()
            if len(fields) < 2:
                continue

            column_1.append(float(fields[0]))
            column_2.append(float(fields[1]))

    if not column_1:
        raise ValueError(
            "file does not contain valid numeric rows with at least two columns."
        )

    dev = default_device(device)
    values_1 = torch.tensor(column_1, device=dev, dtype=dtype)
    values_2 = torch.tensor(column_2, device=dev, dtype=dtype)
    order = torch.argsort(values_1)

    return values_1[order], values_2[order]


def safe_filename_name(name: str) -> str:
    """
    Sanitize a string so it can be safely used as a filename component.

    Replaces spaces, path separators, and common punctuation characters with
    filesystem-friendly substitutes (e.g. "+" becomes "plus").

    Args:
        name: String to sanitize.

    Returns:
        Sanitized string with unsafe characters replaced.

    Raises:
        TypeError: If name is not a string.
    """
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
    """
    Format an angle in degrees as a filename-safe token.

    Args:
        angle_deg: Angle value in degrees, formatted with three decimal
            places.

    Returns:
        String with "." replaced by "p" and "-" replaced by "m" (e.g.
        -12.5 becomes "m12p500").
    """
    return f"{angle_deg:.3f}".replace(".", "p").replace("-", "m")


def ensure_torch_extension(filename: str) -> str:
    """
    Ensure a filename carries a recognized torch tensor file extension.

    Args:
        filename: Candidate filename, with or without an extension.

    Returns:
        filename unchanged if it already has a recognized extension (see
        default.torch_file_extensions); otherwise filename with the default
        extension (default.torch_default_extension) appended, replacing any
        unrecognized extension.
    """
    base_name, ext = os.path.splitext(filename)

    if ext == "":
        return base_name + default.torch_default_extension

    if ext.lower() in default.torch_file_extensions:
        return filename

    return base_name + default.torch_default_extension


def build_output_path(output_dir: str, filename: str) -> str:
    """
    Join an output directory and filename into a single path.

    Args:
        output_dir: Directory portion of the path.
        filename: Filename portion of the path.

    Returns:
        Combined path via os.path.join.
    """
    return os.path.join(output_dir, filename)


def torch_load_file(
    input_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """
    Load a torch file, tolerating older torch versions without weights_only.

    Args:
        input_path: Path to the serialized torch file (.pt/.pth).
        map_location: Device or device string the loaded tensors are mapped
            to.

    Returns:
        Deserialized object stored in the torch file (commonly a tensor or a
        dict of tensors).
    """
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
    """
    Collect the shapes of tensor values in a dictionary.

    Args:
        result: Mapping that may contain torch tensors among its values.

    Returns:
        Dictionary mapping each key whose value is a torch.Tensor to that
        tensor's shape as a tuple. Non-tensor entries are omitted.
    """
    shapes = {}

    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            shapes[key] = tuple(value.shape)

    return shapes


def scalar_float_or_none(value: Any) -> Optional[float]:
    """
    Convert a scalar-like value to a Python float, or None.

    Args:
        value: None, a torch tensor (any shape; the first flattened element
            is used), or any value convertible via float().

    Returns:
        None if value is None; otherwise the value as a Python float.
    """
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())

    return float(value)


def ensure_output_directory(output_dir: str) -> None:
    """
    Create an output directory if it does not already exist.

    Args:
        output_dir: Directory path to create, including any missing parents.
    """
    os.makedirs(output_dir, exist_ok=True)


def list_torch_files(directory: str) -> list[str]:
    """
    List and sort torch tensor files within a directory.

    Args:
        directory: Directory to scan for files ending in a recognized torch
            tensor extension (see default.torch_file_extensions).

    Returns:
        Sorted list of full paths to matching files.

    Raises:
        FileNotFoundError: If directory does not exist.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(default.torch_file_extensions)
    ]
    files.sort()

    return files
