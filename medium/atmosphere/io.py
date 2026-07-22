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
    angle. Filenames always use an alpha angle label and never a theta label.
build_result_metadata(...)
    Creates the metadata dictionary saved together with each tensor payload.
save_phi_Eh_theta_result(...)
    Saves one Atmosphere height-differential flux result to a torch file.
load_phi_Eh_theta_result(...)
    Loads one saved torch file and returns its data dictionary.
load_phi_Eh_from_config(...)
    Convenience loader that reconstructs the expected path from OutputConfig,
    particle, alpha, and theta before calling load_phi_Eh_theta_result.
load_directory(...)
    Loads all torch files in a directory and stacks them by particle or
    flavour, producing batched theta-dependent tensors.

Private helpers
---------------
_metadata_group_key(...)
    Resolves the grouping key used by load_directory.
_required_tensor(...)
    Fetches mandatory tensor fields.
_optional_scalar_float(...)
    Reads optional scalar metadata from either data or metadata.
_assert_same_grid(...)
    Checks that loaded files share the same energy or height grid.
_angle_sort_key(...)
    Provides a stable sorting key for theta-only and alpha+theta payloads.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch

import tpeanuts.config.default as default
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
    package_dir,
)
from tpeanuts.util.torch_util import cast_tensor_tree
from tpeanuts.util.type import as_tensor


@dataclass(frozen=True)
class AtmosphericFluxTable:
    """Canonical long-form atmospheric-neutrino flux table."""

    energy_GeV: torch.Tensor
    cos_zenith: torch.Tensor
    flux: torch.Tensor
    particle: tuple[str, ...]
    azimuth_deg: torch.Tensor | None = None
    altitude_km: torch.Tensor | None = None


def load_atmospheric_flux(
    path: str | PathLike[str] | None = None,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> AtmosphericFluxTable:
    """Load a canonical provider-neutral atmospheric flux CSV.

    Required columns are ``energy_GeV``, ``cos_zenith``, ``particle`` and
    ``flux``. Optional azimuth and production-altitude axes are preserved.
    The differential-flux unit belongs in the provider metadata because some
    original tables publish ``dN/dE`` while Bartol publishes ``dN/dlnE``.
    """
    if path is None:
        path = package_dir() / default.atmosphere_flux_dir / default.atmosphere_flux_filename
    table = pd.read_csv(path)
    required = {"energy_GeV", "cos_zenith", "particle", "flux"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "Atmospheric flux table is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if not table["cos_zenith"].between(-1.0, 1.0).all():
        raise ValueError("cos_zenith values must lie in [-1, 1].")
    if (table["energy_GeV"] <= 0).any() or (table["flux"] < 0).any():
        raise ValueError("Atmospheric energies must be positive and fluxes non-negative.")

    def optional(name: str) -> torch.Tensor | None:
        if name not in table:
            return None
        return as_tensor(table[name].to_numpy(), device=device, dtype=dtype)

    return AtmosphericFluxTable(
        energy_GeV=as_tensor(table["energy_GeV"].to_numpy(), device=device, dtype=dtype),
        cos_zenith=as_tensor(table["cos_zenith"].to_numpy(), device=device, dtype=dtype),
        flux=as_tensor(table["flux"].to_numpy(), device=device, dtype=dtype),
        particle=tuple(table["particle"].astype(str)),
        azimuth_deg=optional("azimuth_deg"),
        altitude_km=optional("altitude_km"),
    )

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
    *,
    alpha_deg: float,
    particle: Optional[str] = None,
    flavour_name: Optional[str] = None,
) -> str:
    """
    Build the full output path for one Atmosphere flux file.

    This is the single filename/path builder for Atmosphere datasets. 
    The generated filename always contains the base filename, a
    flavour or particle label, an optional particle label when it differs from
    the flavour, and one mandatory alpha angle label. The filename angle
    label is always alpha, never theta.

    Args:
        output_config: Output settings containing directory and base filename.
        alpha_deg: Surface/source zenith angle in degrees. This value is
            always embedded in the filename as alpha_<value>deg.
        particle: Physical particle label stored in the file, if known.
        flavour_name: Grouping flavour label. If omitted, particle is used.

    Returns:
        Complete filesystem path for the output torch file.
    """
    base_filename = ensure_torch_extension(output_config.filename)
    base_name, ext = os.path.splitext(base_filename)
    base_name = (
        base_name
        .replace("THETA", "ALPHA")
        .replace("Theta", "Alpha")
        .replace("theta", "alpha")
    )

    flavour = flavour_name if flavour_name is not None else particle
    if flavour is None:
        flavour = "unknown"

    parts = [
        base_name,
        safe_filename_name(str(flavour)),
    ]

    if particle is not None and str(particle) != str(flavour):
        parts.append(safe_filename_name(str(particle)))

    alpha_safe = angle_to_filename(float(alpha_deg))
    parts.append(f"alpha_{alpha_safe}deg")

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
        alpha_deg: Optional surface zenith angle in degrees.
        theta_deg: Optional detector theta angle in degrees.

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
            "theta_deg": "detector zenith angle used by atmosphere geometry",
            "alpha_deg": "surface zenith angle used by source generators",
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

    This is the canonical writer for both detector-theta-only and
    alpha+theta datasets. Passing alpha_deg adds the associated surface
    angle to the filename, payload, and metadata; no separate save function
    is needed.

    Args:
        result: Data dictionary containing at least theta_deg unless theta_deg
            is supplied explicitly. Typical tensors include E_grid_GeV,
            h_grid_km, phi_Eh, phi_E_obs, and f_Eh.
        output_config: Output directory, base filename, dtype, and save flags.
        flavour_name: Optional flavour label used for grouping/loading.
        alpha_deg: Optional surface zenith angle in degrees.
        theta_deg: Optional detector theta angle in degrees.
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


def load_phi_Eh_from_config(
    output_config: OutputConfig,
    particle: str,
    alpha_deg: float,
    *,
    flavour_name: Optional[str] = None,
    map_location: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = torch.float64,
    device: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    """
    Load one alpha-named file by reconstructing its configured path.

    This is a convenience function for callers that know the OutputConfig and
    angle labels but do not want to manually build the filename. It differs
    from load_phi_Eh_theta_result only in how the input path is obtained.

    Args:
        output_config: Output settings used when the file was saved.
        particle: Physical particle label embedded in the filename.
        alpha_deg: Surface zenith angle in degrees embedded in the filename.
        flavour_name: Optional flavour label embedded in the filename.
        map_location: Torch map_location used while reading.
        dtype: Optional dtype applied to floating tensors after loading.
        device: Optional final device for tensors. Defaults to map_location.

    Returns:
        Loaded data dictionary, including metadata when available.
    """
    input_path = build_angle_output_path(
        output_config=output_config,
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

    This reader accepts only the canonical payload written by
    ``save_phi_Eh_theta_result``.

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

    required = {"data", "metadata"}
    missing = required.difference(loaded)
    if missing:
        raise ValueError(
            "Atmosphere tensor payload is not canonical; missing keys: "
            + ", ".join(sorted(missing))
        )
    data = loaded["data"]

    if device is None:
        device = map_location

    data = cast_tensor_tree(
        data,
        dtype=dtype,
        device=device,
    )

    data["metadata"] = loaded["metadata"]

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
) -> torch.Tensor:
    """
    Return a mandatory tensor field.

    Args:
        data: Loaded data dictionary.
        key: Canonical tensor key.

    Returns:
        Tensor stored under key.
    """
    if key in data:
        value = data[key]

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor.")

        return value

    raise KeyError(f"Missing required tensor '{key}'.")


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

    Datasets are primarily ordered by detector theta and secondarily by the
    associated surface alpha when present.

    Args:
        entry: Internal load_directory entry dictionary.

    Returns:
        Pair of floats suitable for sorted(..., key=...).
    """
    alpha_deg = entry.get("alpha_deg", None)
    theta_deg = entry.get("theta_deg", None)

    primary = theta_deg
    secondary = alpha_deg

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
        Here theta_grid_deg is the detector-angle grid, while alpha_grid_deg
        is the associated surface/source-angle grid when present.
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
                ),
                "h_grid_km": _required_tensor(loaded, "h_grid_km"),
                "phi_Eh": _required_tensor(
                    loaded,
                    "phi_Eh",
                ),
                "phi_E_obs": _required_tensor(
                    loaded,
                    "phi_E_obs",
                ),
                "f_Eh": _required_tensor(
                    loaded,
                    "f_Eh",
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


