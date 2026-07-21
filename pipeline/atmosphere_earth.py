"""Atmospheric-neutrino workflow from the Earth surface to a detector.

Detector-composition helpers live here rather than in
``medium.atmosphere.flux``: the atmosphere medium only propagates
production -> surface, while combining the resulting per-flavour detector
probabilities with production-flux tables, integrating over production
height, and summing incoherent produced-flavour contributions are all
pipeline-level concerns tied to this surface-to-detector composition step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.flux import (
    flux_integrated_angular,
    flux_integrated_coordinate,
    flux_state,
    flux_transition,
)
from tpeanuts.core.common.probability import (
    probability_coherent_state,
    probability_transition,
    probability_weighted_average,
)
from tpeanuts.medium.earth.evolutor import earth_evolutor_from_zenith
from tpeanuts.medium.earth.profile import EarthProfile, build_earth_profile
from tpeanuts.pipeline.atmosphere import (
    AtmosphereSurfaceResult,
    propagate_atmosphere_to_surface,
)
from tpeanuts.util.type import as_tensor, cdtype_from_real


@dataclass(frozen=True)
class AtmosphereEarthDetectorResult:
    """Coherent surface-to-detector propagation for one produced flavour."""

    surface: AtmosphereSurfaceResult
    S_earth: torch.Tensor
    detector_states: torch.Tensor
    detector_probabilities: torch.Tensor
    earth_profile: Optional[EarthProfile] = None
    S_total: Optional[torch.Tensor] = None
    transition_probabilities: Optional[torch.Tensor] = None
    detector_probability_height_averaged: Optional[torch.Tensor] = None
    detector_flux_Eh: Optional[torch.Tensor] = None
    detector_flux_height_integrated_E: Optional[torch.Tensor] = None
    detector_flux_energy_height_integrated: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class AtmosphereDetectorGridResult:
    """Multi-angle, full-transition atmospheric detector observables."""

    theta_deg: torch.Tensor
    E_grid_GeV: torch.Tensor
    h_grid_km: torch.Tensor
    flavour_order: tuple[str, ...]
    transition_probability_theta_Eh_beta_alpha: torch.Tensor
    probability_height_theta_E_beta_alpha: torch.Tensor
    production_flux_theta_Eh_alpha: torch.Tensor
    detector_flux_theta_Eh_beta: torch.Tensor
    detector_flux_height_theta_E_beta: torch.Tensor
    detector_flux_angular_E_beta: Optional[torch.Tensor]
    detector_rate: Optional[torch.Tensor]
    detector_results: tuple[AtmosphereEarthDetectorResult, ...]


@torch.no_grad()
def propagate_surface_to_detector(
    surface: AtmosphereSurfaceResult,
    config: PropagationConfig,
    *,
    profile_earth: Optional[EarthProfile] = None,
    integrate_energy: bool = False,
) -> AtmosphereEarthDetectorResult:
    """Continue a coherent atmospheric surface state through the Earth."""
    context = config.runtime
    production = surface.production
    energy_GeV = as_tensor(
        production["E_grid_GeV"],
        device=context.device,
        dtype=context.dtype,
    )
    theta_deg = as_tensor(
        production["theta_deg"],
        device=context.device,
        dtype=context.dtype,
    )
    resolved_profile = build_earth_profile(
        profile_earth,
        params=config.earth,
        context=context,
    )
    S_earth = earth_evolutor_from_zenith(
        profile_earth=resolved_profile,
        oscillation=config.oscillation,
        E_MeV=1.0e3 * energy_GeV,
        theta_deg=theta_deg,
        depth_m=float(config.detector_depth_m),
        reunitarize=config.reunitarize_earth,
    )
    surface_states = as_tensor(
        surface.surface_states,
        device=context.device,
        dtype=cdtype_from_real(context.dtype),
    )
    S_total = torch.matmul(S_earth.unsqueeze(-3), surface.S_atmosphere)
    detector_states = torch.einsum("...ab,b->...a", S_total, surface.initial_state)
    detector_probabilities = probability_coherent_state(
        detector_states,
        real_dtype=context.dtype,
    )
    transition_probabilities = probability_transition(
        S_total, real_dtype=context.dtype
    )
    production_flux = production.get("phi_Eh")
    probability_height = None
    detector_flux_Eh = None
    detector_flux_height = None
    detector_flux_energy_height = None
    if production_flux is not None:
        h = as_tensor(
            production["h_grid_km"], device=context.device, dtype=context.dtype
        )
        production_flux = as_tensor(
            production_flux, device=context.device, dtype=context.dtype
        )
        probability_height = probability_weighted_average(
            detector_probabilities, h, production_flux, dim=-2
        )
        detector_flux_Eh = flux_state(detector_probabilities, production_flux)
        detector_flux_height = flux_integrated_coordinate(
            detector_flux_Eh, h, dim=-2
        )
        if integrate_energy:
            detector_flux_energy_height = flux_integrated_coordinate(
                detector_flux_height, energy_GeV, dim=-2
            )
    return AtmosphereEarthDetectorResult(
        surface=surface,
        S_earth=S_earth,
        detector_states=detector_states,
        detector_probabilities=detector_probabilities,
        earth_profile=resolved_profile,
        S_total=S_total,
        transition_probabilities=transition_probabilities,
        detector_probability_height_averaged=probability_height,
        detector_flux_Eh=detector_flux_Eh,
        detector_flux_height_integrated_E=detector_flux_height,
        detector_flux_energy_height_integrated=detector_flux_energy_height,
    )


@torch.no_grad()
def propagate_atmosphere_grid_to_detector(
    production_by_flavour: dict[str, Sequence[dict[str, object]]],
    config: PropagationConfig,
    *,
    flavour_order: Optional[Sequence[str]] = None,
    profile_earth: Optional[EarthProfile] = None,
    trajectory_steps: int = 200,
    integrate_angular: bool = False,
    integrate_energy: bool = False,
) -> AtmosphereDetectorGridResult:
    """Propagate aligned flavour-production tables over a zenith grid.

    Each mapping value is a sequence with one production dictionary per
    zenith angle. Energy, height and angle grids must agree across flavours.
    Missing flavours are represented by zero production flux.
    """
    if not production_by_flavour:
        raise ValueError("production_by_flavour cannot be empty.")
    if flavour_order is None:
        n_flavours = int(config.oscillation.pmns.n_flavours)
        flavour_order = ("nue", "numu", "nutau") + (
            ("nusterile",) if n_flavours == 4 else ()
        )
    flavour_order = tuple(flavour_order)
    sequences = [list(values) for values in production_by_flavour.values()]
    n_theta = len(sequences[0])
    if n_theta == 0 or any(len(values) != n_theta for values in sequences):
        raise ValueError("All production sequences must share a non-zero angle count.")

    reference_sequence = sequences[0]
    detector_results = []
    transition_by_theta = []
    production_flux_by_theta = []
    theta_values = []
    reference_E = None
    reference_h = None
    resolved_profile = profile_earth

    for angle_index, reference in enumerate(reference_sequence):
        surface = propagate_atmosphere_to_surface(
            reference,
            config,
            initial_flavour=0,
            trajectory_steps=trajectory_steps,
        )
        detector = propagate_surface_to_detector(
            surface,
            config,
            profile_earth=resolved_profile,
        )
        resolved_profile = detector.earth_profile
        detector_results.append(detector)
        transition_by_theta.append(detector.transition_probabilities)

        E = as_tensor(reference["E_grid_GeV"], device=config.runtime.device, dtype=config.runtime.dtype)
        h = as_tensor(reference["h_grid_km"], device=config.runtime.device, dtype=config.runtime.dtype)
        if reference_E is None:
            reference_E, reference_h = E, h
        elif not torch.equal(E, reference_E) or not torch.equal(h, reference_h):
            raise ValueError("All angle entries must share energy and height grids.")
        theta_values.append(float(reference["theta_deg"]))

        flux_components = []
        for flavour in flavour_order:
            entries = production_by_flavour.get(flavour)
            if entries is None:
                flux_components.append(torch.zeros((E.numel(), h.numel()), device=E.device, dtype=E.dtype))
                continue
            entry = entries[angle_index]
            if abs(float(entry["theta_deg"]) - theta_values[-1]) > 1.0e-9:
                raise ValueError("Production flavours must use aligned theta grids.")
            entry_E = as_tensor(entry["E_grid_GeV"], device=E.device, dtype=E.dtype)
            entry_h = as_tensor(entry["h_grid_km"], device=E.device, dtype=E.dtype)
            if not torch.equal(entry_E, E) or not torch.equal(entry_h, h):
                raise ValueError("Production flavours must share energy and height grids.")
            flux_components.append(as_tensor(entry["phi_Eh"], device=E.device, dtype=E.dtype))
        production_flux_by_theta.append(torch.stack(flux_components, dim=-1))

    transition = torch.stack(transition_by_theta, dim=0)
    production_flux = torch.stack(production_flux_by_theta, dim=0)
    if transition.shape[-1] != len(flavour_order):
        raise ValueError("flavour_order length must match the oscillation model.")
    detector_flux = flux_transition(transition, production_flux)
    detector_flux_height = flux_integrated_coordinate(
        detector_flux, reference_h, dim=2
    )

    weights = production_flux.unsqueeze(-2)
    numerator = torch.trapezoid(
        transition * weights, x=reference_h, dim=2
    )
    denominator = torch.trapezoid(weights, x=reference_h, dim=2)
    probability_height = numerator / denominator.clamp_min(
        torch.finfo(transition.dtype).tiny
    )

    theta = torch.as_tensor(theta_values, device=reference_E.device, dtype=reference_E.dtype)
    detector_flux_angular = None
    if integrate_angular:
        detector_flux_angular = flux_integrated_angular(
            detector_flux_height, theta, angular_dim=0
        )
    detector_rate = None
    if integrate_energy:
        rate_input = detector_flux_angular if detector_flux_angular is not None else detector_flux_height
        detector_rate = flux_integrated_coordinate(
            rate_input, reference_E, dim=-2
        )

    return AtmosphereDetectorGridResult(
        theta_deg=theta,
        E_grid_GeV=reference_E,
        h_grid_km=reference_h,
        flavour_order=flavour_order,
        transition_probability_theta_Eh_beta_alpha=transition,
        probability_height_theta_E_beta_alpha=probability_height,
        production_flux_theta_Eh_alpha=production_flux,
        detector_flux_theta_Eh_beta=detector_flux,
        detector_flux_height_theta_E_beta=detector_flux_height,
        detector_flux_angular_E_beta=detector_flux_angular,
        detector_rate=detector_rate,
        detector_results=tuple(detector_results),
    )


def detector_flux_from_production(
    production_flux: torch.Tensor,
    detector_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Build the detector flux for one coherently propagated source flavour."""
    return flux_state(detector_probabilities, production_flux)


def integrate_detector_flux_over_height(
    h_grid_km: torch.Tensor,
    detector_flux: torch.Tensor,
) -> torch.Tensor:
    """Integrate a height-resolved detector flux over production altitude."""
    h = torch.as_tensor(
        h_grid_km,
        device=detector_flux.device,
        dtype=detector_flux.dtype,
    )
    if detector_flux.shape[-2] != h.numel():
        raise ValueError(
            "detector_flux penultimate dimension must match h_grid_km."
        )
    return torch.trapezoid(detector_flux, x=h, dim=-2)


def sum_detected_flavours(
    integrated_flux_by_production_flavour: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Sum incoherently the detector fluxes from all produced flavours."""
    values = list(integrated_flux_by_production_flavour.values())
    if not values:
        raise ValueError("At least one produced-flavour detector flux is required.")
    reference_shape = values[0].shape
    if any(value.shape != reference_shape for value in values[1:]):
        raise ValueError("All produced-flavour detector fluxes must share a shape.")
    return torch.stack(values, dim=0).sum(dim=0)
