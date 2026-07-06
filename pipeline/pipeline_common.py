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
Shared helpers for high-level propagation pipelines.

This module contains preparation utilities used by coherent, incoherent, and
legacy pipeline entry points. It keeps reusable orchestration concerns out of
the physics-specific pipeline modules: Earth profile construction and solar
distance selection.

Module functions:
    package_root(...)
        Return the project package root.
    default_earth_density_path(...)
        Return the configured default Earth-density CSV path.
    prepare_initial_state(...)
        Convert a non-string initial state into a complex state tensor.
    prepare_earth_profile(...)
        Use an existing EarthProfile-like object or build the default one.
    prepare_earth_distance(...)
        Select the Sun-Earth distance from user input, table data, or default.
"""


from __future__ import annotations

from pathlib import Path
from typing import Any, Union, Optional

import torch

import tpeanuts.util.default as default
from tpeanuts.medium.solar.io import load_sun_earth_distance
from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile
from tpeanuts.util.constant import SUN_EARTH_DISTANCE_KM
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor, cdtype_from_real, state_tensor

StateLike = Union[str, list[complex], tuple[complex, ...], torch.Tensor]

FLAVOUR_ORDER = ["nue", "numu", "nutau"]


def package_root() -> Path:
    """Return the tpeanuts package root directory.

    Returns:
        Path pointing to the root directory that contains the package modules
        and bundled data.
    """
    return Path(__file__).resolve().parents[1]


def default_earth_density_path() -> str:
    """Return the configured default Earth-density CSV path.

    Returns:
        String path built from ``util.default.earth_density_dir`` and
        ``util.default.earth_density_filename``.
    """
    return str(
        package_root()
        / default.earth_density_dir
        / default.earth_density_filename
    )


def prepare_initial_state(
    initial_state: StateLike,
    *,
    context: RuntimeContext,
) -> StateLike:
    """Prepare an initial flavour state for coherent propagation.

    Args:
        initial_state: Flavour name or explicit state amplitudes.
        context: Runtime device/dtype used to select the complex state dtype.

    Returns:
        The original string state or a complex tensor with final dimension 3.
    """
    if isinstance(initial_state, str):
        return initial_state
    return state_tensor(
        initial_state,
        device=context.device,
        dtype=cdtype_from_real(context.dtype),
    )


def prepare_earth_profile(
    earth_profile: Optional[object],
    *,
    earth: EarthParameters,
    context: RuntimeContext,
) -> tuple[object, Optional[str]]:
    """Return an EarthProfile-compatible object and its density-file metadata.

    Args:
        earth_profile: Existing EarthProfile-compatible object.
        earth: Earth electron-density profile construction settings. When
            ``earth.profile_perturbative_kwargs`` has no ``"density_file"``
            entry, the bundled default Earth-density CSV is used.
        context: Runtime device/dtype for a newly constructed profile.

    Returns:
        Tuple ``(profile, density_path)``. ``density_path`` is None when an
        external profile object is supplied and no file path is known.
    """
    if earth_profile is not None:
        density_path = (
            (earth.profile_perturbative_kwargs or {}).get("density_file")
        )
        return earth_profile, density_path

    kwargs = dict(earth.profile_perturbative_kwargs or {})
    density_path = kwargs.get("density_file") or default_earth_density_path()
    kwargs["density_file"] = density_path
    kwargs.setdefault("tabulated_density", False)

    return (
        EarthProfile(
            params=EarthParameters(
                profile_perturbative_name=earth.profile_perturbative_name,
                profile_perturbative_kwargs=kwargs,
                profile_scale_m=earth.profile_scale_m,
                evolution_scale_m=earth.evolution_scale_m,
            ),
            context=context,
        ),
        density_path,
    )


def prepare_earth_distance(
    earth_distance_km: Optional[TensorLike],
    *,
    sun_earth_distance_path: Optional[str],
    use_sun_earth_distance_table: bool,
    context: RuntimeContext,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Prepare the Sun-Earth distance and metadata used by solar pipelines.

    Args:
        earth_distance_km: Explicit user-provided distance in km.
        sun_earth_distance_path: Optional path to the tabulated distance file.
        use_sun_earth_distance_table: If True, use the table mean when no
            explicit distance is provided.
        context: Runtime device/dtype.

    Returns:
        Tuple ``(distance, metadata)``.
    """
    device, dtype = context.device, context.dtype

    if earth_distance_km is not None:
        distance = as_tensor(earth_distance_km, device=device, dtype=dtype)
        return distance, {"source": "user", "distance_km": float(distance.reshape(-1)[0].detach().cpu().item())}

    if use_sun_earth_distance_table:
        table = load_sun_earth_distance(
            sun_earth_distance_path,
            device=device,
            dtype=dtype,
        )
        distance = torch.mean(table["distance_km"])
        return distance, {
            "source": "sun_earth_distance_table_mean",
            "path": sun_earth_distance_path,
            "n_dates": len(table["date"]),
            "distance_km": float(distance.detach().cpu().item()),
        }

    distance = torch.tensor(SUN_EARTH_DISTANCE_KM, device=device, dtype=dtype)
    return distance, {
        "source": "SUN_EARTH_DISTANCE_KM",
        "distance_km": float(distance.detach().cpu().item()),
    }
