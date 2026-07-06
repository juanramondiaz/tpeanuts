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
Input/output utilities for Atmosphere height-flux datasets.

This module owns the canonical loader for Atmosphere MCEq/Honda torch
datasets. The saved files are expected to contain energy grids, height grids,
height-differential fluxes, particle/flavour labels, theta values, and metadata
produced by the Atmosphere generation pipeline.

Public API
----------
OutputConfig
    Dataclass with the output directory, base filename, dtype, and save flags
    used by Atmosphere generators.
build_angle_output_path(...)
    Builds the complete output path for one particle/flavour and Atmosphere
    angle. It handles theta-only and alpha+theta filenames with one rule.
build_result_metadata(...)
    Creates the metadata dictionary saved together with each tensor payload.
save_phi_Eh_theta_result(...)
    Saves one Atmosphere height-differential flux result to a torch file.
load_phi_Eh_theta_result(...)
    Loads one saved torch file and returns its data dictionary.
load_phi_Eh_alpha_theta_from_config(...)
    Convenience loader that reconstructs the expected path from OutputConfig,
    particle, alpha, and theta before calling load_phi_Eh_theta_result.
load_directory(...)
    Loads all torch files in a directory and stacks them by particle or
    flavour, producing batched theta-dependent tensors.
load_phi_E_h_flavours_for_theta(...)
    Thin compatibility wrapper over load_directory for callers that need the
    old tuple format at one selected theta.

Private helpers
---------------
_metadata_group_key(...)
    Resolves the grouping key used by load_directory.
_required_tensor(...)
    Fetches mandatory tensor fields while accepting legacy aliases.
_optional_scalar_float(...)
    Reads optional scalar metadata from either data or metadata.
_assert_same_grid(...)
    Checks that loaded files share the same energy or height grid.
_angle_sort_key(...)
    Provides a stable sorting key for theta-only and alpha+theta datasets.
_theta_index(...)
    Selects the nearest theta entry within a tolerance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from os import PathLike
from typing import Any, Dict, Optional, Union

import torch

import tpeanuts.util.default as default
from tpeanuts.util.io import (
    angle_to_filename,
    build_output_path,
    ensure_output_directory,
    ensure_torch_extension,
    list_torch_files,
    safe_filename_name,
    scalar_float_or_none,
    tensor_shape_dict,
    torch_load_file,
)
from tpeanuts.util.torch_util import cast_tensor_tree

@dataclass
class OutputConfig:
    """
    Output configuration for Atmosphere height-flux torch files.

    This object is shared by MCEq and Honda generators. It describes where
    files are written, which base filename is used before angle suffixes are
    added, which dtype is stored on disk, and whether existing files may be
    overwritten.
    """

    output_dir: str | PathLike[str] = default.atmosphere_height_flux_output_dir
    filename: str = default.atmosphere_height_flux_filename
    dtype: torch.dtype = torch.float32
    compressed: bool = True
    overwrite: bool = False
    save_intermediate: bool = False

    def validate(self) -> None:
        """
        Validate field types before a save operation.

        The save code only needs a valid directory path, filename string,
        torch dtype, and boolean flags. This method keeps those assumptions
        explicit at the edge of the I/O layer while accepting pathlib paths.
        """
        if not isinstance(self.dtype, torch.dtype):
            raise ValueError(f"dtype must be a torch.dtype, got {type(self.dtype)}")

        if not isinstance(self.filename, str) or self.filename.strip() == "":
            raise ValueError("filename must be a non-empty string.")

        try:
            output_dir = os.fspath(self.output_dir)
        except TypeError as exc:
            raise ValueError(
                "output_dir must be a non-empty path-like value."
            ) from exc

        if not isinstance(output_dir, str) or output_dir.strip() == "":
            raise ValueError("output_dir must be a non-empty path-like value.")

        for attr in ("compressed", "overwrite", "save_intermediate"):
            if not isinstance(getattr(self, attr), bool):
                raise ValueError(f"{attr} must be a boolean.")


# ============================================================
# Filename utilities
# ============================================================









def build_angle_output_path(
    output_config: OutputConfig,
    theta_deg: float,
    *,
    particle: Optional[str] = None,
    flavour_name: Optional[str] = None,
    alpha_deg: Optional[float] = None,
) -> str:
    """
    Build the full output path for one Atmosphere flux file.

    This is the single filename/path builder for Atmosphere datasets. It
    replaces the older theta-only and alpha+theta helpers by treating alpha as
    optional. The generated filename always contains the base filename, a
    flavour or particle label, an optional particle label when it differs from
    the flavour, an optional alpha angle, and the required theta angle.

    Args:
        output_config: Output settings containing directory and base filename.
        theta_deg: Atmosphere surface-intersection angle in degrees.
        particle: Physical particle label stored in the file, if known.
        flavour_name: Grouping flavour label. If omitted, particle is used.
        alpha_deg: Optional detector angle in degrees.

    Returns:
        Complete filesystem path for the output torch file.
    """
    base_filename = ensure_torch_extension(output_config.filename)
    base_name, ext = os.path.splitext(base_filename)

    flavour = flavour_name if flavour_name is not None else particle
    if flavour is None:
        flavour = "unknown"

    parts = [
        base_name,
        safe_filename_name(str(flavour)),
    ]

    if particle is not None and str(particle) != str(flavour):
        parts.append(safe_filename_name(str(particle)))

    if alpha_deg is not None:
        alpha_safe = angle_to_filename(float(alpha_deg))
        parts.append(f"alpha_{alpha_safe}deg")

    theta_safe = angle_to_filename(float(theta_deg))
    parts.append(f"theta_{theta_safe}deg")

    filename = "_".join(parts) + ext

    return build_output_path(
        output_dir=output_config.output_dir,
        filename=filename,
    )




# ============================================================
# Metadata
# ============================================================





def build_result_metadata(
    result: Dict[str, Any],
    *,
    flavour_name: Optional[str] = None,
    particle: Optional[str] = None,
    alpha_deg: Optional[float] = None,
    theta_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build metadata saved next to one Atmosphere flux payload.

    The metadata is intentionally source-neutral: it describes the common
    Atmosphere Phi(E,h) file contract used by both MCEq-derived and
    Honda-derived datasets. Source-specific details can be supplied through
    result["metadata_extra"].

    Args:
        result: Result dictionary that will be saved.
        flavour_name: Optional flavour grouping label.
        particle: Optional physical particle label.
        alpha_deg: Optional detector angle in degrees.
        theta_deg: Optional Atmosphere theta angle in degrees.

    Returns:
        JSON-serializable metadata dictionary.
    """
    particle_value = particle if particle is not None else result.get("particle")
    alpha_value = (
        scalar_float_or_none(alpha_deg)
        if alpha_deg is not None
        else scalar_float_or_none(result.get("alpha_deg", None))
    )
    theta_value = (
        scalar_float_or_none(theta_deg)
        if theta_deg is not None
        else scalar_float_or_none(result.get("theta_deg", None))
    )

    metadata = {
        "description": (
            "Atmosphere height-differential flux dataset."
        ),
        "relation": (
            "Phi(E,h) = Phi(E;X_obs) * f(h|E,theta)"
        ),
        "normalization": (
            "Integral_h f(h|E,theta) dh = 1"
        ),
        "particle": particle_value,
        "flavour_name": flavour_name,
        "alpha_deg": alpha_value,
        "theta_deg": theta_value,
        "angle_units": "deg",
        "angle_convention": {
            "alpha_deg": "detector angle, when provided by the caller",
            "theta_deg": "surface-intersection Atmosphere angle",
        },
        "tensor_shapes": tensor_shape_dict(result),
        "format": "torch",
        "extension": default.torch_default_extension,
    }

    metadata_extra = result.get("metadata_extra", None)

    if metadata_extra is not None:
        if not isinstance(metadata_extra, dict):
            raise TypeError("result['metadata_extra'] must be a dictionary.")

        metadata.update(metadata_extra)

    return metadata


# ============================================================
# Tensor preparation
# ============================================================



# ============================================================
# Save
# ============================================================



def save_phi_Eh_theta_result(
    result: Dict[str, Any],
    output_config: OutputConfig,
    *,
    flavour_name: Optional[str] = None,
    alpha_deg: Optional[float] = None,
    theta_deg: Optional[float] = None,
    particle: Optional[str] = None,
) -> str:
    """
    Save one Atmosphere height-differential flux result.

    This is the canonical writer for both theta-only and alpha+theta datasets.
    Passing alpha_deg simply adds alpha information to the filename, payload,
    and metadata; no separate save function is needed.

    Args:
        result: Data dictionary containing at least theta_deg unless theta_deg
            is supplied explicitly. Typical tensors include E_grid_GeV,
            h_grid_km, phi_Eh, phi_E_obs, and f_Eh.
        output_config: Output directory, base filename, dtype, and save flags.
        flavour_name: Optional flavour label used for grouping/loading.
        alpha_deg: Optional detector angle in degrees.
        theta_deg: Optional Atmosphere theta angle in degrees.
        particle: Optional physical particle label.

    Returns:
        Path of the written file, or the existing file when overwrite is False.
    """
    output_config.validate()

    ensure_output_directory(output_config.output_dir)

    if theta_deg is None and "theta_deg" not in result:
        raise KeyError("result must contain key 'theta_deg'.")

    theta_value = (
        scalar_float_or_none(theta_deg)
        if theta_deg is not None
        else scalar_float_or_none(result["theta_deg"])
    )

    alpha_value = (
        scalar_float_or_none(alpha_deg)
        if alpha_deg is not None
        else scalar_float_or_none(result.get("alpha_deg", None))
    )

    particle_value = (
        particle
        if particle is not None
        else result.get("particle", "unknown")
    )

    flavour = (
        flavour_name
        if flavour_name is not None
        else particle_value
    )

    output_path = build_angle_output_path(
        output_config=output_config,
        theta_deg=theta_value,
        flavour_name=str(flavour),
        alpha_deg=alpha_value,
        particle=str(particle_value),
    )

    if os.path.exists(output_path) and not output_config.overwrite:
        return output_path

    metadata = build_result_metadata(
        result=result,
        flavour_name=flavour_name,
        particle=str(particle_value),
        alpha_deg=alpha_value,
        theta_deg=theta_value,
    )

    data = dict(result)
    data["particle"] = str(particle_value)
    data["theta_deg"] = torch.as_tensor(theta_value)

    if alpha_value is not None:
        data["alpha_deg"] = torch.as_tensor(alpha_value)

    save_dict = {
        "metadata": metadata,
        "metadata_json": json.dumps(metadata, indent=2),
        "data": cast_tensor_tree(
            data,
            dtype=output_config.dtype,
            device="cpu",
        ),
    }

    torch.save(
        save_dict,
        output_path,
    )

    return output_path


def load_phi_Eh_alpha_theta_from_config(
    output_config: OutputConfig,
    particle: str,
    alpha_deg: float,
    theta_deg: float,
    *,
    flavour_name: Optional[str] = None,
    map_location: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = torch.float64,
    device: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    """
    Load one alpha+theta file by reconstructing its configured path.

    This is a convenience function for callers that know the OutputConfig and
    angle labels but do not want to manually build the filename. It differs
    from load_phi_Eh_theta_result only in how the input path is obtained.

    Args:
        output_config: Output settings used when the file was saved.
        particle: Physical particle label embedded in the filename.
        alpha_deg: Detector angle in degrees embedded in the filename.
        theta_deg: Atmosphere theta angle in degrees embedded in the filename.
        flavour_name: Optional flavour label embedded in the filename.
        map_location: Torch map_location used while reading.
        dtype: Optional dtype applied to floating tensors after loading.
        device: Optional final device for tensors. Defaults to map_location.

    Returns:
        Loaded data dictionary, including metadata when available.
    """
    input_path = build_angle_output_path(
        output_config=output_config,
        theta_deg=theta_deg,
        particle=particle,
        flavour_name=flavour_name,
        alpha_deg=alpha_deg,
    )

    return load_phi_Eh_theta_result(
        input_path=input_path,
        map_location=map_location,
        dtype=dtype,
        device=device,
    )


def load_phi_Eh_theta_result(
    input_path: str,
    *,
    map_location: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = torch.float64,
    device: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    """
    Load one Atmosphere height-differential flux torch file.

    This is the canonical single-file reader. It expects the format written by
    save_phi_Eh_theta_result, but it also accepts older files where the payload
    was saved directly rather than under a "data" key.

    Args:
        input_path: Path to one .pt or .pth file.
        map_location: Torch map_location used while reading.
        dtype: Optional dtype applied to floating tensors after loading.
        device: Optional final device for tensors. Defaults to map_location.

    Returns:
        Loaded data dictionary with tensors cast as requested and metadata
        attached when present.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"File not found: {input_path}"
        )

    loaded = torch_load_file(
        map_location=map_location,
        input_path=input_path,
    )

    if not isinstance(loaded, dict):
        raise TypeError(
            "Loaded object must be a dictionary."
        )

    if "data" in loaded:
        data = loaded["data"]
    else:
        data = loaded

    if device is None:
        device = map_location

    data = cast_tensor_tree(
        data,
        dtype=dtype,
        device=device,
    )

    if "metadata" in loaded:
        data["metadata"] = loaded["metadata"]

    elif "metadata_json" in loaded:
        data["metadata"] = json.loads(loaded["metadata_json"])

    if "metadata_json" in loaded:
        data["metadata_json"] = loaded["metadata_json"]

    return data


def _metadata_group_key(
    data: Dict[str, Any],
    *,
    group_by: str,
) -> str:
    """
    Resolve the particle/flavour key used to group a loaded file.

    Args:
        data: Loaded single-file data dictionary.
        group_by: Either "particle" or "flavour_name".

    Returns:
        String key used by load_directory.
    """
    metadata = data.get("metadata", {})

    if group_by == "particle":
        value = metadata.get("particle", data.get("particle", None))
    elif group_by == "flavour_name":
        value = metadata.get("flavour_name", data.get("flavour_name", None))
    else:
        raise ValueError("group_by must be 'particle' or 'flavour_name'.")

    if value is None and group_by == "particle":
        value = metadata.get("flavour_name", data.get("flavour_name", None))

    if value is None and group_by == "flavour_name":
        value = metadata.get("particle", data.get("particle", None))

    if value is None:
        raise KeyError("Could not resolve particle/flavour name from metadata.")

    return str(value)


def _required_tensor(
    data: Dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
) -> torch.Tensor:
    """
    Return a mandatory tensor field, accepting legacy aliases.

    Args:
        data: Loaded data dictionary.
        key: Canonical tensor key.
        aliases: Alternative key names accepted for backwards compatibility.

    Returns:
        Tensor stored under key or one of its aliases.
    """
    candidate_keys = (key, *aliases)

    for candidate_key in candidate_keys:
        if candidate_key in data:
            value = data[candidate_key]

            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{candidate_key} must be a torch.Tensor.")

            return value

    raise KeyError(
        f"Missing required tensor '{key}'. "
        f"Tried aliases: {candidate_keys}."
    )


def _optional_scalar_float(
    data: Dict[str, Any],
    key: str,
) -> Optional[float]:
    """
    Read an optional scalar value from data or metadata.

    Args:
        data: Loaded data dictionary.
        key: Scalar field name.

    Returns:
        Float value when present, otherwise None.
    """
    metadata = data.get("metadata", {})

    if key in data:
        return scalar_float_or_none(data[key])

    if key in metadata:
        return scalar_float_or_none(metadata[key])

    return None


def _assert_same_grid(
    reference: torch.Tensor,
    value: torch.Tensor,
    *,
    group_name: str,
    grid_name: str,
) -> None:
    """
    Ensure a loaded grid matches the reference grid.

    Args:
        reference: Grid already accepted for the group.
        value: Candidate grid from another file.
        group_name: Particle/flavour group being validated.
        grid_name: Human-readable grid name for error messages.
    """
    if reference.shape != value.shape or not torch.allclose(reference, value):
        raise ValueError(
            f"Inconsistent {grid_name} found for {group_name}."
        )


def _angle_sort_key(entry: Dict[str, Any]) -> tuple[float, float]:
    """
    Sort entries by the angle that identifies the scan.

    Alpha+theta datasets are primarily ordered by alpha and secondarily by
    theta. Theta-only datasets are ordered by theta.

    Args:
        entry: Internal load_directory entry dictionary.

    Returns:
        Pair of floats suitable for sorted(..., key=...).
    """
    alpha_deg = entry.get("alpha_deg", None)
    theta_deg = entry.get("theta_deg", None)

    primary = theta_deg if alpha_deg is None else alpha_deg
    secondary = theta_deg if alpha_deg is not None else alpha_deg

    if primary is None:
        primary = float("inf")
    if secondary is None:
        secondary = float("inf")

    return float(primary), float(secondary)


def load_directory(
    directory: str,
    *,
    map_location: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = torch.float64,
    device: Optional[str | torch.device] = None,
    group_by: str = "particle",
    verbose: bool = False,
):
    """
    Load and stack all Atmosphere flux torch files in a directory.

    This is the canonical dataset loader for MCEq and Honda Atmosphere
    outputs. It groups files by particle or flavour, checks that each group
    shares the same energy and height grids, sorts entries by angle, and stacks
    tensors along the angle axis.

    Args:
        directory: Directory containing .pt or .pth files.
        map_location: Torch map_location used while reading.
        dtype: Optional dtype applied to floating tensors after loading.
        device: Optional final tensor device. Defaults to map_location.
        group_by: Group key, either "particle" or "flavour_name".
        verbose: If True, print one summary line per group.

    Returns:
        Dictionary keyed by particle/flavour. Each value contains paths,
        metadata, entries, E_grid_GeV, h_grid_km, theta_grid_deg,
        optional alpha_grid_deg, phi_E_theta_h, phi_E_theta, and f_theta_E_h.
    """
    files = list_torch_files(directory)

    if len(files) == 0:
        raise FileNotFoundError(
            f"No torch files with extensions {default.torch_file_extensions} "
            f"found in: {directory}"
        )

    grouped: Dict[str, list[Dict[str, Any]]] = {}

    for path in files:
        loaded = load_phi_Eh_theta_result(
            input_path=path,
            map_location=map_location,
            dtype=dtype,
            device=device,
        )

        group_name = _metadata_group_key(
            loaded,
            group_by=group_by,
        )

        theta_deg = _optional_scalar_float(loaded, "theta_deg")

        if theta_deg is None:
            raise KeyError(
                f"Missing theta_deg in loaded file: {path}"
            )

        grouped.setdefault(group_name, []).append(
            {
                "path": path,
                "metadata": loaded.get("metadata", {}),
                "theta_deg": theta_deg,
                "alpha_deg": _optional_scalar_float(loaded, "alpha_deg"),
                "E_grid_GeV": _required_tensor(
                    loaded,
                    "E_grid_GeV",
                    aliases=("E_grid",),
                ),
                "h_grid_km": _required_tensor(loaded, "h_grid_km"),
                "phi_Eh": _required_tensor(
                    loaded,
                    "phi_Eh",
                    aliases=("phi_E_h",),
                ),
                "phi_E_obs": _required_tensor(
                    loaded,
                    "phi_E_obs",
                    aliases=("phi_E",),
                ),
                "f_Eh": _required_tensor(
                    loaded,
                    "f_Eh",
                    aliases=("f_E_h",),
                ),
                "data": loaded,
            }
        )

    data: Dict[str, Dict[str, Any]] = {}

    for group_name, entries in grouped.items():
        entries = sorted(entries, key=_angle_sort_key)

        E_ref = entries[0]["E_grid_GeV"]
        h_ref = entries[0]["h_grid_km"]

        for entry in entries[1:]:
            _assert_same_grid(
                E_ref,
                entry["E_grid_GeV"],
                group_name=group_name,
                grid_name="E_grid_GeV",
            )

            _assert_same_grid(
                h_ref,
                entry["h_grid_km"],
                group_name=group_name,
                grid_name="h_grid_km",
            )

        theta_grid_deg = torch.tensor(
            [entry["theta_deg"] for entry in entries],
            device=E_ref.device,
            dtype=E_ref.dtype if torch.is_floating_point(E_ref) else torch.float64,
        )

        alpha_values = [entry["alpha_deg"] for entry in entries]
        alpha_grid_deg = None

        if all(alpha_deg is not None for alpha_deg in alpha_values):
            alpha_grid_deg = torch.tensor(
                alpha_values,
                device=E_ref.device,
                dtype=E_ref.dtype
                if torch.is_floating_point(E_ref)
                else torch.float64,
            )

        phi_E_theta_h = torch.stack(
            [entry["phi_Eh"] for entry in entries],
            dim=0,
        )

        phi_E_theta = torch.stack(
            [entry["phi_E_obs"] for entry in entries],
            dim=0,
        )

        f_theta_E_h = torch.stack(
            [entry["f_Eh"] for entry in entries],
            dim=0,
        )

        group_data = {
            "paths": [entry["path"] for entry in entries],
            "metadata": [entry["metadata"] for entry in entries],
            "entries": [entry["data"] for entry in entries],
            "E_grid_GeV": E_ref,
            "E_grid": E_ref,
            "theta_grid_deg": theta_grid_deg,
            "h_grid_km": h_ref,
            "phi_E_theta_h": phi_E_theta_h,
            "phi_E_theta": phi_E_theta,
            "f_theta_E_h": f_theta_E_h,
        }

        if alpha_grid_deg is not None:
            group_data["alpha_grid_deg"] = alpha_grid_deg

        data[group_name] = group_data

        if verbose:
            print(
                f"Loaded {group_name:12s} | "
                f"n_theta = {len(theta_grid_deg):3d} | "
                f"phi(E,theta,h) shape = {tuple(phi_E_theta_h.shape)}"
            )

    return data


def _theta_index(
    theta_grid_deg: torch.Tensor,
    theta_deg: float,
    theta_tolerance_deg: float,
    *,
    group_name: str,
) -> int:
    """
    Find the theta entry closest to a requested angle.

    Args:
        theta_grid_deg: 1D tensor of available theta angles.
        theta_deg: Requested theta angle in degrees.
        theta_tolerance_deg: Maximum accepted absolute mismatch.
        group_name: Group name used in error messages.

    Returns:
        Integer index of the selected theta entry.
    """
    theta_target = torch.as_tensor(
        theta_deg,
        device=theta_grid_deg.device,
        dtype=theta_grid_deg.dtype,
    )
    delta = torch.abs(theta_grid_deg - theta_target)
    index = int(torch.argmin(delta).item())

    if float(delta[index].detach().cpu()) > theta_tolerance_deg:
        raise FileNotFoundError(
            f"No entry for {group_name} at theta={theta_deg} deg "
            f"within tolerance {theta_tolerance_deg} deg."
        )

    return index


@torch.no_grad()
def load_phi_E_h_flavours_for_theta(
    data_dir: str,
    theta_deg: float,
    required_flavours=("nue", "numu", "nutau"),
    theta_tolerance_deg: float = 1e-6,
    verbose: bool = True,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
):
    """
    Load flavour-resolved height-differential flux tensors at one theta.

    This compatibility wrapper delegates to load_directory and converts the
    stacked dictionary format into the older tuple format expected by some
    Atmosphere propagation code. Use load_directory for new code that needs
    the full theta grid.

    Args:
        data_dir: Directory containing Atmosphere torch files.
        theta_deg: Requested theta angle in degrees.
        required_flavours: Ordered flavour names that must be present.
        theta_tolerance_deg: Maximum accepted absolute theta mismatch.
        verbose: If True, print one line per selected flavour.
        device: Optional final tensor device.
        dtype: Optional dtype applied to floating tensors.

    Returns:
        Tuple (E_grid, h_grid_km, phi_E_h_flavours, metadata_dict), where
        phi_E_h_flavours maps each required flavour to one Phi(E,h) tensor.
    """
    dataset = load_directory(
        data_dir,
        map_location="cpu",
        dtype=dtype,
        device=device,
        group_by="particle",
        verbose=False,
    )

    phi_E_h_flavours = {}
    metadata_dict = {}

    E_ref = None
    h_ref = None
    missing = []

    for flavour in required_flavours:
        if flavour not in dataset:
            missing.append(flavour)
            continue

        group = dataset[flavour]
        index = _theta_index(
            group["theta_grid_deg"],
            theta_deg,
            theta_tolerance_deg,
            group_name=str(flavour),
        )

        E_grid = group["E_grid_GeV"]
        h_grid_km = group["h_grid_km"]
        phi_E_h = group["phi_E_theta_h"][index]

        if E_ref is None:
            E_ref = E_grid
        else:
            _assert_same_grid(
                E_ref,
                E_grid,
                group_name=str(flavour),
                grid_name="E_grid_GeV",
            )

        if h_ref is None:
            h_ref = h_grid_km
        else:
            _assert_same_grid(
                h_ref,
                h_grid_km,
                group_name=str(flavour),
                grid_name="h_grid_km",
            )

        phi_E_h_flavours[flavour] = phi_E_h
        metadata_dict[flavour] = group["metadata"][index]

        if verbose:
            theta_file = float(group["theta_grid_deg"][index].detach().cpu())
            print(
                f"Loaded {flavour:8s} | "
                f"theta = {theta_file:7.3f} deg | "
                f"shape = {tuple(phi_E_h.shape)} | "
                f"device = {phi_E_h.device} | "
                f"{group['paths'][index]}"
            )

    if len(missing) > 0:
        raise FileNotFoundError(
            f"Missing flavours for theta={theta_deg} deg: {missing}"
        )

    return (
        E_ref,
        h_ref,
        phi_E_h_flavours,
        metadata_dict,
    )
