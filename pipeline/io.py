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
I/O helpers for detector-level fluxes after full propagation.

These helpers serialize and reassemble the per-particle, per-angle outputs of
the atmosphere pipeline (see ``pipeline.pipeline_atmosphere`` /
``pipeline.atmosphere_flux``): one torch file holds the production, surface,
and detector fluxes for a single produced particle (e.g. ``"numu"`` or
``"anti_nue"``) evaluated at a single detector angle (alpha, the detector
zenith convention, or theta, the atmosphere zenith convention). The
aggregation helpers below stack many such single-particle/single-angle files
back into angle-grid tensors, summing contributions from different produced
particles/flavours that feed the same final neutrino/antineutrino mode so
that flux and oscillation-probability spectra can be plotted or compared
across the full angular range.

The saved files contain one produced particle and one detector alpha/theta.
Main tensor convention:

    detector_flux_Ei: (n_E, 3)

where the last index is final flavour [nue, numu, nutau].

Module functions:
    build_detector_flux_filename(...)
        Build a filename encoding particle and detector angle.
    build_detector_flux_path(...)
        Join an output directory with a built detector-flux filename.
    detector_particle_mode(...)
        Classify a particle key as neutrino ("nu") or antineutrino ("antinu").
    detector_initial_flavour(...)
        Strip the antineutrino prefix and validate a particle's flavour key.
    aggregate_detector_flux_by_mode(...)
        Stack per-angle detector-flux files into angle-grid tensors, summing
        contributions per neutrino/antineutrino mode.
    aggregate_detector_conversion_by_mode(...)
        Stack per-angle detector conversion-probability files into
        angle/initial-flavour-grid tensors per mode.
    build_detector_flux_metadata(...)
        Build the JSON-serializable metadata block stored alongside a saved
        detector-flux result.
    save_detector_flux_result(...)
        Save one particle/angle detector-flux result to a torch file.
    load_detector_flux_result(...)
        Load one detector-flux torch file back into a tensor dictionary.
    list_detector_flux_files(...)
        List torch detector-flux files in a directory.
    load_detector_flux_directory(...)
        Load and group every detector-flux file in a directory.
"""



from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch

import tpeanuts.util.default as default
from tpeanuts.util.io import (
    angle_to_filename,
    ensure_output_directory,
    safe_filename_name,
    scalar_float_or_none,
    tensor_shape_dict,
    torch_load_file,
)
from tpeanuts.util.torch_util import cast_tensor_tree








def build_detector_flux_filename(
    particle: str,
    *,
    alpha_deg: Optional[float] = None,
    theta_deg: Optional[float] = None,
    base_filename: str = default.detector_flux_filename,
) -> str:
    """
    Build a detector-flux filename encoding particle and detector angle.

    Args:
        particle: Produced-particle key (e.g. "numu", "anti_nue") whose
            flux this file stores.
        alpha_deg: Detector zenith angle in degrees, if the file was produced
            on a detector-alpha grid.
        theta_deg: Atmosphere zenith angle in degrees, if the file was
            produced on a propagation-theta grid.
        base_filename: Base filename (and extension) to derive the torch
            file extension from; defaults to ``default.detector_flux_filename``.

    Returns:
        Filename string of the form
        ``"<base>_<particle>[_alpha_<a>deg][_theta_<t>deg]<ext>"``. When
        neither angle is given, an ``"angle_unknown"`` suffix is used instead.
    """
    base, ext = os.path.splitext(base_filename)
    if ext.lower() not in default.torch_file_extensions:
        ext = default.torch_default_extension

    parts = [
        base,
        safe_filename_name(str(particle)),
    ]

    if alpha_deg is not None:
        parts.append(f"alpha_{angle_to_filename(float(alpha_deg))}deg")

    if theta_deg is not None:
        parts.append(f"theta_{angle_to_filename(float(theta_deg))}deg")

    if alpha_deg is None and theta_deg is None:
        parts.append("angle_unknown")

    return "_".join(parts) + ext


def build_detector_flux_path(
    output_dir: str,
    particle: str,
    *,
    alpha_deg: Optional[float] = None,
    theta_deg: Optional[float] = None,
    base_filename: str = default.detector_flux_filename,
) -> str:
    """
    Join an output directory with a built detector-flux filename.

    Args:
        output_dir: Directory the detector-flux file will live in.
        particle: Produced-particle key (e.g. "numu", "anti_nue").
        alpha_deg: Detector zenith angle in degrees, if applicable.
        theta_deg: Atmosphere zenith angle in degrees, if applicable.
        base_filename: Base filename used to derive the torch file
            extension; defaults to ``default.detector_flux_filename``.

    Returns:
        Full path string combining ``output_dir`` with the filename built by
        ``build_detector_flux_filename``.
    """
    return os.path.join(
        output_dir,
        build_detector_flux_filename(
            particle,
            alpha_deg=alpha_deg,
            theta_deg=theta_deg,
            base_filename=base_filename,
        ),
    )






def detector_particle_mode(particle: str) -> str:
    """
    Classify a produced-particle key as neutrino or antineutrino mode.

    Args:
        particle: Produced-particle key, e.g. "numu" or "anti_nue".

    Returns:
        ``"antinu"`` if ``particle`` starts with "anti", otherwise ``"nu"``.
        Used to bucket aggregated fluxes/probabilities separately for
        neutrinos and antineutrinos.
    """
    return "antinu" if str(particle).startswith("anti") else "nu"


def detector_initial_flavour(particle: str) -> str:
    """
    Resolve the flavour key of a produced particle, stripping any antineutrino prefix.

    Args:
        particle: Produced-particle key, e.g. "numu", "anti_nue", "NUTAU".

    Returns:
        One of "nue", "numu", "nutau" — the produced flavour regardless of
        whether the particle is a neutrino or antineutrino.

    Raises:
        ValueError: If the (antineutrino-stripped) key is not a recognised
            flavour.
    """
    key = str(particle).lower()
    if key.startswith("anti"):
        key = key[4:]
    if key in {"nue", "numu", "nutau"}:
        return key
    raise ValueError(f"Unknown detector-flux particle flavour: {particle}")


def _detector_scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def _detectorscalar_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    return _detector_scalar_float(value)


def _detector_entry_angle(entry: Dict[str, Any]) -> tuple[float, Optional[float], Optional[float], str]:
    """
    Resolve the angle used to stack detector-flux entries.

    Detector files may be produced either on a detector-alpha grid or directly
    on a propagation-theta grid. Older theta-mode files store alpha_deg=None,
    so aggregation must fall back to theta_deg instead of assuming alpha_deg is
    always present.
    """
    alpha = _detectorscalar_float_or_none(entry.get("alpha_deg", None))
    theta = _detectorscalar_float_or_none(entry.get("theta_deg", None))

    if alpha is not None:
        return alpha, alpha, theta, "alpha"

    if theta is not None:
        return theta, alpha, theta, "theta"

    particle = entry.get("particle", "unknown")
    raise ValueError(
        "Detector-flux entries must contain at least one valid angle "
        f"('alpha_deg' or 'theta_deg'). Missing both for particle={particle!r}."
    )


def _detector_angle_sort_key(entry: Dict[str, Any]) -> tuple[float, float]:
    angle, alpha, theta, _ = _detector_entry_angle(entry)
    secondary = theta if alpha is not None else alpha
    if secondary is None:
        secondary = float("inf")
    return float(angle), float(secondary)


def aggregate_detector_flux_by_mode(
    data: Dict[str, list[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Stack per-particle, per-angle detector-flux files into angle-grid tensors.

    Groups the loaded entries (as returned by ``load_detector_flux_directory``
    with ``group_by="particle"``) by neutrino/antineutrino mode
    (``detector_particle_mode``), sums the initial/surface/detector fluxes of
    every produced particle that lands in the same (mode, angle) bucket, and
    stacks the result along a new leading angle axis sorted by increasing
    angle. This turns many single-angle files (one physical detector
    pointing direction each) into one flux-vs-angle dataset per mode, ready
    for plotting an angular distribution.

    Args:
        data: Mapping from produced-particle key to a list of per-angle
            result dictionaries (each holding ``E_grid_GeV``,
            ``initial_flux_Ei``, ``surface_flux_Ei``, ``detector_flux_Ei``,
            ``source_flux_E``, and angle metadata).

    Returns:
        Dictionary keyed by mode ("nu"/"antinu"), each value holding the
        common energy grid, the angle grid (``angle_grid_deg`` plus separate
        ``alpha_grid_deg``/``theta_grid_deg``), and flux tensors stacked as
        ``(n_angle, n_E, 3)`` (last index is final flavour
        [nue, numu, nutau]), along with the list of particles contributing
        to each angle bin.

    Raises:
        ValueError: If entries being summed into the same bucket do not
            share the same energy grid.
    """
    aggregated: Dict[str, Dict[tuple[str, float], Dict[str, Any]]] = {}

    for particle, entries in data.items():
        mode = detector_particle_mode(particle)
        mode_data = aggregated.setdefault(mode, {})

        for entry in entries:
            angle, alpha_deg, theta_deg, angle_kind = _detector_entry_angle(entry)
            angle_key = (angle_kind, round(angle, 9))
            bucket = mode_data.setdefault(
                angle_key,
                {
                    "angle_deg": angle,
                    "angle_kind": angle_kind,
                    "alpha_deg": alpha_deg,
                    "theta_deg": theta_deg,
                    "E_grid_GeV": entry["E_grid_GeV"],
                    "initial_flux_Ei": torch.zeros_like(entry["initial_flux_Ei"]),
                    "surface_flux_Ei": torch.zeros_like(entry["surface_flux_Ei"]),
                    "detector_flux_Ei": torch.zeros_like(entry["detector_flux_Ei"]),
                    "source_flux_E": torch.zeros_like(entry["source_flux_E"]),
                    "particles": [],
                },
            )

            if not torch.allclose(bucket["E_grid_GeV"], entry["E_grid_GeV"]):
                raise ValueError(f"Energy grid mismatch while aggregating {particle}.")

            bucket["initial_flux_Ei"] += entry["initial_flux_Ei"]
            bucket["surface_flux_Ei"] += entry["surface_flux_Ei"]
            bucket["detector_flux_Ei"] += entry["detector_flux_Ei"]
            bucket["source_flux_E"] += entry["source_flux_E"]
            bucket["particles"].append(str(particle))

    output: Dict[str, Dict[str, Any]] = {}
    for mode, buckets in aggregated.items():
        ordered = sorted(buckets.values(), key=_detector_angle_sort_key)
        if not ordered:
            continue

        dtype = ordered[0]["E_grid_GeV"].dtype
        angle_values = [item["angle_deg"] for item in ordered]
        alpha_values = [
            item["alpha_deg"] if item["alpha_deg"] is not None else item["angle_deg"]
            for item in ordered
        ]
        theta_values = [
            item["theta_deg"] if item["theta_deg"] is not None else float("nan")
            for item in ordered
        ]
        angle_kinds = sorted({item["angle_kind"] for item in ordered})

        output[mode] = {
            "angle_grid_deg": torch.as_tensor(angle_values, dtype=dtype),
            "angle_grid_kind": angle_kinds[0] if len(angle_kinds) == 1 else "mixed",
            "alpha_grid_deg": torch.as_tensor(alpha_values, dtype=dtype),
            "theta_grid_deg": torch.as_tensor(theta_values, dtype=dtype),
            "E_grid_GeV": ordered[0]["E_grid_GeV"],
            "initial_flux_alpha_Ei": torch.stack([item["initial_flux_Ei"] for item in ordered], dim=0),
            "surface_flux_alpha_Ei": torch.stack([item["surface_flux_Ei"] for item in ordered], dim=0),
            "detector_flux_alpha_Ei": torch.stack([item["detector_flux_Ei"] for item in ordered], dim=0),
            "source_flux_alpha_E": torch.stack([item["source_flux_E"] for item in ordered], dim=0),
            "contributors_by_angle": [sorted(set(item["particles"])) for item in ordered],
            "contributors_by_alpha": [sorted(set(item["particles"])) for item in ordered],
            "particles": sorted({particle for item in ordered for particle in item["particles"]}),
        }

    return output


def aggregate_detector_conversion_by_mode(
    data: Dict[str, list[Dict[str, Any]]],
    *,
    flavour_order: Optional[list[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Stack per-particle, per-angle detector conversion-probability files.

    Similar to ``aggregate_detector_flux_by_mode`` but for oscillation
    *probabilities* (initial flavour beta -> final flavour i) rather than
    fluxes: each input entry is placed at its initial-flavour row of a
    ``(n_flavour, n_E)`` probability block (other rows are left as NaN if no
    entry was provided for that initial flavour at that angle), then the
    per-mode blocks are stacked along a new leading angle axis sorted by
    increasing angle.

    Args:
        data: Mapping from produced-particle key to a list of per-angle
            result dictionaries, each holding ``E_grid_GeV``,
            ``detector_probability_Ei`` (and optionally
            ``surface_probability_Ei``) plus angle metadata.
        flavour_order: Optional ordering of initial flavours used to index
            the stacked probability tensor. Defaults to
            ``["nue", "numu", "nutau"]``.

    Returns:
        Dictionary keyed by mode ("nu"/"antinu"), each value holding the
        flavour order, angle grids, common energy grid, and probability
        tensors stacked as ``(n_angle, n_flavour, n_E)`` under
        ``detector_probability_alpha_beta_Ei`` (and
        ``surface_probability_alpha_beta_Ei`` when present in every entry),
        plus the contributing particles per angle bin.

    Raises:
        ValueError: If entries being merged into the same bucket do not
            share the same energy grid.
    """
    if flavour_order is None:
        flavour_order = ["nue", "numu", "nutau"]

    flavour_to_index = {flavour: i for i, flavour in enumerate(flavour_order)}
    aggregated: Dict[str, Dict[tuple[str, float], Dict[str, Any]]] = {}

    for particle, entries in data.items():
        mode = detector_particle_mode(particle)
        initial_flavour = detector_initial_flavour(particle)
        beta_index = flavour_to_index[initial_flavour]
        mode_data = aggregated.setdefault(mode, {})

        for entry in entries:
            angle, alpha_deg, theta_deg, angle_kind = _detector_entry_angle(entry)
            angle_key = (angle_kind, round(angle, 9))
            probability = entry["detector_probability_Ei"]
            surface_probability = entry.get("surface_probability_Ei")

            bucket = mode_data.setdefault(
                angle_key,
                {
                    "angle_deg": angle,
                    "angle_kind": angle_kind,
                    "alpha_deg": alpha_deg,
                    "theta_deg": theta_deg,
                    "E_grid_GeV": entry["E_grid_GeV"],
                    "detector_probability_beta_Ei": torch.full(
                        (len(flavour_order), *probability.shape),
                        float("nan"),
                        dtype=probability.dtype,
                        device=probability.device,
                    ),
                    "surface_probability_beta_Ei": None
                    if surface_probability is None
                    else torch.full(
                        (len(flavour_order), *surface_probability.shape),
                        float("nan"),
                        dtype=surface_probability.dtype,
                        device=surface_probability.device,
                    ),
                    "particles": [],
                },
            )

            if not torch.allclose(bucket["E_grid_GeV"], entry["E_grid_GeV"]):
                raise ValueError(f"Energy grid mismatch while aggregating {particle}.")

            bucket["detector_probability_beta_Ei"][beta_index] = probability
            if surface_probability is not None and bucket["surface_probability_beta_Ei"] is not None:
                bucket["surface_probability_beta_Ei"][beta_index] = surface_probability
            bucket["particles"].append(str(particle))

    output: Dict[str, Dict[str, Any]] = {}
    for mode, buckets in aggregated.items():
        ordered = sorted(buckets.values(), key=_detector_angle_sort_key)
        if not ordered:
            continue

        dtype = ordered[0]["E_grid_GeV"].dtype
        angle_values = [item["angle_deg"] for item in ordered]
        alpha_values = [
            item["alpha_deg"] if item["alpha_deg"] is not None else item["angle_deg"]
            for item in ordered
        ]
        theta_values = [
            item["theta_deg"] if item["theta_deg"] is not None else float("nan")
            for item in ordered
        ]
        angle_kinds = sorted({item["angle_kind"] for item in ordered})

        mode_output = {
            "flavour_order": list(flavour_order),
            "angle_grid_deg": torch.as_tensor(angle_values, dtype=dtype),
            "angle_grid_kind": angle_kinds[0] if len(angle_kinds) == 1 else "mixed",
            "alpha_grid_deg": torch.as_tensor(alpha_values, dtype=dtype),
            "theta_grid_deg": torch.as_tensor(theta_values, dtype=dtype),
            "E_grid_GeV": ordered[0]["E_grid_GeV"],
            "detector_probability_alpha_beta_Ei": torch.stack([item["detector_probability_beta_Ei"] for item in ordered], dim=0),
            "contributors_by_angle": [sorted(set(item["particles"])) for item in ordered],
            "contributors_by_alpha": [sorted(set(item["particles"])) for item in ordered],
            "particles": sorted({particle for item in ordered for particle in item["particles"]}),
        }

        if ordered[0]["surface_probability_beta_Ei"] is not None:
            mode_output["surface_probability_alpha_beta_Ei"] = torch.stack(
                [item["surface_probability_beta_Ei"] for item in ordered],
                dim=0,
            )

        output[mode] = mode_output

    return output


def build_detector_flux_metadata(
    result: Dict[str, Any],
    *,
    particle: str,
    alpha_deg: Optional[float],
    theta_deg: Optional[float],
) -> Dict[str, Any]:
    """
    Build the JSON-serializable metadata block stored with a detector-flux result.

    Args:
        result: Detector-flux result dictionary (e.g. produced by the
            atmosphere pipeline) used to derive tensor shapes; may contain an
            optional ``"metadata_extra"`` dict whose entries override/extend
            the defaults below.
        particle: Produced-particle key the result belongs to.
        alpha_deg: Detector zenith angle in degrees, or None.
        theta_deg: Atmosphere zenith angle in degrees, or None.

    Returns:
        Dictionary with a human-readable description of the physical
        scenario (mceq production, coherent atmosphere propagation, coherent
        Earth propagation, height integration), the particle and angle
        values, the final-flavour order, per-tensor shapes, and the storage
        format/extension.

    Raises:
        TypeError: If ``result["metadata_extra"]`` is present but not a dict.
    """
    metadata = {
        "description": (
            "Detector-level Atmosphere neutrino flux after mceq production, "
            "coherent Atmosphere propagation, coherent earth propagation, "
            "and integration over production height."
        ),
        "particle": str(particle),
        "alpha_deg": scalar_float_or_none(alpha_deg),
        "theta_deg": scalar_float_or_none(theta_deg),
        "angle_units": "deg",
        "final_flavour_order": ["nue", "numu", "nutau"],
        "tensor_shapes": tensor_shape_dict(result),
        "format": "torch",
        "extension": default.torch_default_extension,
    }

    extra = result.get("metadata_extra", None)
    if extra is not None:
        if not isinstance(extra, dict):
            raise TypeError("result['metadata_extra'] must be a dictionary.")
        metadata.update(extra)

    return metadata


def save_detector_flux_result(
    result: Dict[str, Any],
    output_dir: str,
    *,
    particle: Optional[str] = None,
    alpha_deg: Optional[float] = None,
    theta_deg: Optional[float] = None,
    base_filename: str = default.detector_flux_filename,
    dtype: Optional[torch.dtype] = torch.float32,
    overwrite: bool = False,
) -> str:
    """
    Save one produced-particle, one-angle detector-flux result to a torch file.

    Args:
        result: Detector-flux result dictionary to persist (initial/surface/
            detector fluxes and grids); stored as-is alongside built
            metadata.
        output_dir: Directory the file is written into (created if needed).
        particle: Produced-particle key. Defaults to ``result["particle"]``.
        alpha_deg: Detector zenith angle in degrees. Defaults to
            ``result.get("alpha_deg")``.
        theta_deg: Atmosphere zenith angle in degrees. Defaults to
            ``result.get("theta_deg")``.
        base_filename: Base filename used to derive the output filename and
            torch file extension.
        dtype: Floating-point dtype tensors are cast to before saving (real
            tensors only; complex/int tensors are left as-is by
            ``cast_tensor_tree``). None keeps the original dtype.
        overwrite: If False and the target file already exists, the existing
            path is returned without rewriting it.

    Returns:
        Path of the saved (or pre-existing) torch file.
    """
    particle_value = str(particle if particle is not None else result["particle"])
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

    ensure_output_directory(output_dir)

    output_path = build_detector_flux_path(
        output_dir,
        particle_value,
        alpha_deg=alpha_value,
        theta_deg=theta_value,
        base_filename=base_filename,
    )

    if os.path.exists(output_path) and not overwrite:
        return output_path

    data = dict(result)
    data["particle"] = particle_value
    if alpha_value is not None:
        data["alpha_deg"] = torch.as_tensor(alpha_value)
    if theta_value is not None:
        data["theta_deg"] = torch.as_tensor(theta_value)

    metadata = build_detector_flux_metadata(
        data,
        particle=particle_value,
        alpha_deg=alpha_value,
        theta_deg=theta_value,
    )

    payload = {
        "metadata": metadata,
        "metadata_json": json.dumps(metadata, indent=2),
        "data": cast_tensor_tree(
            data,
            dtype=dtype,
            device="cpu",
        ),
    }

    torch.save(payload, output_path)

    return output_path


def load_detector_flux_result(
    input_path: str,
    *,
    map_location: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = torch.float64,
    device: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a single detector-flux torch file saved by ``save_detector_flux_result``.

    Args:
        input_path: Path to the saved torch file.
        map_location: torch.load map_location used while reading the file.
        dtype: Floating-point dtype tensors are cast to after loading. None
            keeps the dtype stored on disk.
        device: Device tensors are moved to. Defaults to ``map_location``.

    Returns:
        Dictionary with the saved result fields (fluxes, grids, ...) plus
        ``"metadata"`` (parsed dict) and ``"metadata_json"`` (raw string)
        when present in the file.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        TypeError: If the loaded object is not a dictionary.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    loaded = torch_load_file(input_path, map_location=map_location)
    if not isinstance(loaded, dict):
        raise TypeError("Loaded object must be a dictionary.")

    data = loaded.get("data", loaded)

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


def list_detector_flux_files(directory: str) -> list[str]:
    """
    List detector-flux torch files contained in a directory.

    Args:
        directory: Directory to scan for saved detector-flux files.

    Returns:
        Sorted list of full paths to files whose extension matches
        ``default.torch_file_extensions``.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
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


def load_detector_flux_directory(
    directory: str,
    *,
    map_location: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = torch.float64,
    device: Optional[str | torch.device] = None,
    group_by: str = "particle",
) -> Dict[str, list[Dict[str, Any]]]:
    """
    Load every detector-flux file in a directory and group the results.

    Args:
        directory: Directory containing saved detector-flux torch files.
        map_location: torch.load map_location used while reading each file.
        dtype: Floating-point dtype tensors are cast to after loading. None
            keeps the dtype stored on disk.
        device: Device tensors are moved to. Defaults to ``map_location``.
        group_by: Key used to group loaded entries: "particle" (produced
            particle key), "alpha" (detector zenith angle), or "theta"
            (atmosphere zenith angle).

    Returns:
        Dictionary mapping each group key (as a string) to the list of
        loaded result dictionaries belonging to that group, sorted by
        increasing angle (see ``_detector_angle_sort_key``). This is the
        input format expected by ``aggregate_detector_flux_by_mode`` and
        ``aggregate_detector_conversion_by_mode`` when ``group_by="particle"``.

    Raises:
        ValueError: If ``group_by`` is not one of "particle", "alpha", or
            "theta".
        FileNotFoundError: If ``directory`` does not exist.
    """
    if group_by not in {"particle", "alpha", "theta"}:
        raise ValueError("group_by must be 'particle', 'alpha', or 'theta'.")

    grouped: Dict[str, list[Dict[str, Any]]] = {}

    for path in list_detector_flux_files(directory):
        data = load_detector_flux_result(
            path,
            map_location=map_location,
            dtype=dtype,
            device=device,
        )
        metadata = data.get("metadata", {})

        if group_by == "particle":
            key = str(metadata.get("particle", data.get("particle", "unknown")))
        elif group_by == "alpha":
            key = str(metadata.get("alpha_deg", data.get("alpha_deg", "unknown")))
        else:
            key = str(metadata.get("theta_deg", data.get("theta_deg", "unknown")))

        grouped.setdefault(key, []).append(data)

    for entries in grouped.values():
        entries.sort(key=_detector_angle_sort_key)

    return grouped
