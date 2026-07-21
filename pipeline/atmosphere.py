"""Atmospheric-neutrino workflow from production to the Earth surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.neutrino import flavour_index, flavour_state
from tpeanuts.core.common.flux import flux_integrated_coordinate, flux_state
from tpeanuts.core.common.probability import (
    probability_coherent_state,
    probability_weighted_average,
)
from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
from tpeanuts.medium.atmosphere.geometry import _angle_distance
from tpeanuts.medium.earth.geometry import build_atmosphere_trajectories
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import as_tensor


def _canonical_flavour_name(name: Union[str, int]) -> Union[str, int]:
    if isinstance(name, int):
        return name
    key = str(name).lower().replace("total_", "")
    return key.replace("anti", "").replace("bar", "").strip("_")


@dataclass(frozen=True)
class AtmosphereSurfaceResult:
    """Coherent atmospheric propagation result at the Earth surface."""

    production: Dict[str, object]
    initial_state: torch.Tensor
    surface_states: torch.Tensor
    S_atmosphere: torch.Tensor
    surface_probabilities: torch.Tensor
    trajectories: Dict[str, torch.Tensor]
    surface_probability_height_averaged: Optional[torch.Tensor] = None
    surface_flux_Eh: Optional[torch.Tensor] = None
    surface_flux_height_integrated_E: Optional[torch.Tensor] = None
    surface_flux_energy_height_integrated: Optional[torch.Tensor] = None


def select_production_flux(
    flux_data: Dict[str, Dict[str, object]],
    particle: str,
    *,
    alpha_deg: Optional[float] = None,
    theta_deg: Optional[float] = None,
    angle_index: Optional[int] = None,
    angle_mode: str = "theta",
    angle_tolerance_deg: Optional[float] = None,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> Dict[str, object]:
    """Select one produced particle and angular bin from a flux dataset."""
    if particle not in flux_data:
        raise KeyError(
            f"Particle {particle!r} not found. Available: {sorted(flux_data)}"
        )
    group = flux_data[particle]
    if angle_mode == "alpha" and "alpha_grid_deg" not in group:
        angle_mode = "theta"
    angle_key = "alpha_grid_deg" if angle_mode == "alpha" else "theta_grid_deg"
    angle_grid = as_tensor(
        group[angle_key], device=context.device, dtype=context.dtype
    )
    if angle_index is None:
        distance = _angle_distance(
            angle_grid,
            alpha_deg=alpha_deg,
            theta_deg=theta_deg,
            angle_mode=angle_mode,
        )
        angle_index = int(torch.argmin(distance).item())
        if (
            angle_tolerance_deg is not None
            and float(distance[angle_index].item()) > angle_tolerance_deg
        ):
            raise ValueError("No angular bin lies within angle_tolerance_deg.")
    angle_index = int(angle_index)
    if angle_index < 0 or angle_index >= angle_grid.numel():
        raise IndexError("angle_index is outside the available angle grid.")

    E = as_tensor(group["E_grid_GeV"], device=context.device, dtype=context.dtype)
    h = as_tensor(group["h_grid_km"], device=context.device, dtype=context.dtype)
    theta_grid = as_tensor(
        group["theta_grid_deg"], device=context.device, dtype=context.dtype
    )
    alpha_value = None
    if "alpha_grid_deg" in group:
        alpha_value = float(as_tensor(group["alpha_grid_deg"])[angle_index].item())
    return {
        "particle": particle,
        "flavour_index": flavour_index(_canonical_flavour_name(particle)),
        "angle_index": angle_index,
        "angle_mode": angle_mode,
        "alpha_deg": alpha_value,
        "theta_deg": float(theta_grid[angle_index].item()),
        "E_grid_GeV": E,
        "h_grid_km": h,
        "phi_Eh": as_tensor(
            group["phi_E_theta_h"][angle_index],
            device=context.device,
            dtype=context.dtype,
        ),
        "phi_E_theta": as_tensor(
            group["phi_E_theta"][angle_index],
            device=context.device,
            dtype=context.dtype,
        ),
        "f_Eh": as_tensor(
            group["f_theta_E_h"][angle_index],
            device=context.device,
            dtype=context.dtype,
        ),
        "metadata": group["metadata"][angle_index],
        "entry": group["entries"][angle_index],
        "path": group["paths"][angle_index],
    }


@torch.no_grad()
def propagate_atmosphere_to_surface(
    production: Dict[str, object],
    config: PropagationConfig,
    *,
    initial_flavour: Optional[Union[str, int]] = None,
    trajectory_steps: int = 200,
    integrate_energy: bool = False,
) -> AtmosphereSurfaceResult:
    """Propagate one produced flavour coherently to the Earth surface."""
    context = config.runtime
    E = as_tensor(production["E_grid_GeV"], device=context.device, dtype=context.dtype)
    h = as_tensor(production["h_grid_km"], device=context.device, dtype=context.dtype)
    theta = as_tensor(production["theta_deg"], device=context.device, dtype=context.dtype)
    E_grid, h_grid = torch.meshgrid(E, h, indexing="ij")
    initial_state = flavour_state(
        _canonical_flavour_name(
            initial_flavour if initial_flavour is not None else production["particle"]
        ),
        device=context.device,
        dtype=context.dtype,
    )
    S_atmosphere, _ = atmosphere_evolutor(
        config.oscillation,
        1.0e3 * E_grid,
        h_grid,
        theta,
        torch.as_tensor(
            config.detector_depth_m / 1.0e3,
            device=context.device,
            dtype=context.dtype,
        ),
        atmosphere=config.atmosphere,
        context=context,
    )
    surface_states = torch.einsum("...ab,b->...a", S_atmosphere, initial_state)
    surface_probabilities = probability_coherent_state(
        surface_states, real_dtype=context.dtype
    )
    production_flux = production.get("phi_Eh")
    surface_flux_Eh = None
    probability_height = None
    flux_height = None
    flux_energy_height = None
    if production_flux is not None:
        production_flux = as_tensor(
            production_flux, device=context.device, dtype=context.dtype
        )
        surface_flux_Eh = flux_state(surface_probabilities, production_flux)
        probability_height = probability_weighted_average(
            surface_probabilities,
            h,
            production_flux,
            dim=-2,
        )
        flux_height = flux_integrated_coordinate(
            surface_flux_Eh, h, dim=-2
        )
        if integrate_energy:
            flux_energy_height = flux_integrated_coordinate(
                flux_height, E, dim=-2
            )

    return AtmosphereSurfaceResult(
        production=production,
        initial_state=initial_state,
        surface_states=surface_states,
        S_atmosphere=S_atmosphere,
        surface_probabilities=surface_probabilities,
        trajectories=build_atmosphere_trajectories(
            production,
            detector_depth_m=config.detector_depth_m,
            trajectory_steps=trajectory_steps,
            context=context,
        ),
        surface_probability_height_averaged=probability_height,
        surface_flux_Eh=surface_flux_Eh,
        surface_flux_height_integrated_E=flux_height,
        surface_flux_energy_height_integrated=flux_energy_height,
    )
