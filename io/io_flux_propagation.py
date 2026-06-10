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

The saved files contain one produced particle and one detector alpha/theta.
Main tensor convention:

    detector_flux_Ei: (n_E, 3)

where the last index is final flavour [nue, numu, nutau].
"""



from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch

from tpeanuts.io.io_common import (
    angle_to_filename,
    cast_tensor_tree,
    ensure_output_directory,
    safe_filename_name,
    scalar_float_or_none,
    tensor_shape_dict,
    torch_load_file,
)








def build_detector_flux_filename(
    particle: str,
    *,
    alpha_deg: Optional[float] = None,
    theta_deg: Optional[float] = None,
    base_filename: str = "Detectorflux.pt",
) -> str:
    base, ext = os.path.splitext(base_filename)
    if ext.lower() not in {".pt", ".pth"}:
        ext = ".pt"

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
    base_filename: str = "Detectorflux.pt",
) -> str:
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
    return "antinu" if str(particle).startswith("anti") else "nu"


def detector_initial_flavour(particle: str) -> str:
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
    metadata = {
        "description": (
            "Detector-level atmospheric neutrino flux after mceq production, "
            "coherent atmospheric propagation, coherent earth propagation, "
            "and integration over production height."
        ),
        "particle": str(particle),
        "alpha_deg": scalar_float_or_none(alpha_deg),
        "theta_deg": scalar_float_or_none(theta_deg),
        "angle_units": "deg",
        "final_flavour_order": ["nue", "numu", "nutau"],
        "tensor_shapes": tensor_shape_dict(result),
        "format": "torch",
        "extension": ".pt",
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
    base_filename: str = "Detectorflux.pt",
    dtype: Optional[torch.dtype] = torch.float32,
    overwrite: bool = False,
) -> str:
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
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith((".pt", ".pth"))
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
