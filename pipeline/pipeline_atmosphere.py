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
Workflow helpers for atmosphere-neutrino production and detector fluxes.

Physical scenario: unlike the solar pipelines (``pipeline_coherent``,
``pipeline_incoherent``, ``pipeline_legacypeanuts``), which propagate
neutrinos produced deep inside the Sun across vacuum to Earth, this module
handles neutrinos produced by cosmic-ray air showers in the Earth's
atmosphere at a height h above the surface, arriving at a detector at a
given zenith angle theta after crossing only atmosphere and (optionally)
Earth matter — there is no vacuum/solar stage. Each produced flavour's
height-and-energy-differential flux Phi_beta(E,h,theta) (typically built by
an external generator such as MCEq/Honda) is propagated coherently from its
production altitude, through the atmosphere, to the Earth's surface, then
optionally further through Earth matter to a detector at
``detector_depth_m``, and finally integrated over production height to give
the total detector-level flux summed over all produced flavours.

The intended workflow is:

1. Select Phi_beta(E,h,theta) for one produced particle and one detector angle.
2. Keep the selected mceq grids and metadata together.
3. Propagate flavour states from production height h to the earth surface.
4. Keep atmosphere propagation coherent for each produced neutrino.
5. Propagate the surface states coherently through earth to the detector.
6. Build beta -> i probabilities at the detector.
7. Integrate over production height.
8. Sum contributions from all produced flavours.

Flavour convention follows the rest of tpeanuts:

    [nue, numu, nutau] -> [0, 1, 2]

Module functions:
    select_particle_angle_flux(...)
        Select one particle and angle slice from a loaded height-flux dataset.
    build_atmosphere_trajectories(...)
        Build atmosphere path lengths and altitude grids for the selected
        production-height grid.
    propagate_atmosphere_coherent(...)
        Propagate one coherent flavour state from production altitude to the
        Earth surface through the atmosphere.
    propagate_earth_coherent(...)
        Continue coherent propagation from the Earth surface to the detector.
    integrate_initial_and_surface_fluxes(...)
        Integrate initial and atmosphere-surface fluxes over production height.
    integrate_height_and_sum_flavours(...)
        Integrate detector fluxes over height and sum produced-flavour
        contributions.
"""



from __future__ import annotations

from typing import Dict, Optional, Union

import torch

from tpeanuts.medium.atmosphere.geometry import atmosphere_path_length, atmosphere_path_grid
from tpeanuts.core.common.probability import probability_coherent_state
from tpeanuts.core.common.neutrino import flavour_index
from tpeanuts.medium.earth.profile import EarthProfile
from tpeanuts.medium.earth.evolutor import earth_evolutor_from_zenith
from tpeanuts.pipeline.config import PropagationConfig
from tpeanuts.pipeline.pipeline_common import prepare_earth_profile
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import as_tensor, cdtype_from_real




def _canonical_flavour_name(name: Union[str, int]) -> Union[str, int]:
    """Normalize particle labels to the flavour keys understood by core."""
    if isinstance(name, int):
        return name

    key = str(name).lower()
    key = key.replace("total_", "")
    key = key.replace("anti", "")
    key = key.replace("bar", "")
    key = key.strip("_")

    return key


def flavour_state(
    flavour: Union[str, int],
    *,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> torch.Tensor:
    """
    Build a coherent flavour-basis unit state.

    Args:
        flavour: Flavour label or index accepted by ``flavour_index``.
        context: Runtime device/dtype for the returned state.

    Returns:
        Complex tensor with shape ``(3,)`` and unit amplitude in the selected
        flavour component.
    """
    cdtype = cdtype_from_real(context.dtype)
    idx = flavour_index(_canonical_flavour_name(flavour))

    state = torch.zeros(3, device=context.device, dtype=cdtype)
    state[idx] = 1.0 + 0.0j

    return state


def _angle_distance(
    angle_grid: torch.Tensor,
    *,
    alpha_deg: Optional[float],
    theta_deg: Optional[float],
    angle_mode: str,
) -> torch.Tensor:
    """Return absolute angular distance on the selected angle convention."""
    if angle_mode == "alpha":
        if alpha_deg is None:
            raise ValueError("alpha_deg is required when angle_mode='alpha'.")
        return torch.abs(angle_grid - float(alpha_deg))

    if angle_mode == "theta":
        if theta_deg is None:
            raise ValueError("theta_deg is required when angle_mode='theta'.")
        return torch.abs(angle_grid - float(theta_deg))

    raise ValueError("angle_mode must be 'alpha' or 'theta'.")


@torch.no_grad()
def select_particle_angle_flux(
    flux_data: Dict[str, dict],
    particle: str,
    *,
    alpha_deg: Optional[float] = None,
    theta_deg: Optional[float] = None,
    angle_index: Optional[int] = None,
    angle_mode: str = "theta",
    angle_tolerance_deg: Optional[float] = None,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> Dict[str, object]:
    """
    Select one particle and angle slice from loaded atmosphere flux data.

    Args:
        flux_data: Dictionary produced by the atmosphere I/O loaders, keyed by
            particle/flavour name.
        particle: Particle key to select.
        alpha_deg: Surface/source angle used when angle_mode is ``"alpha"``.
        theta_deg: Detector zenith angle used when angle_mode is ``"theta"``.
        angle_index: Optional explicit angle-bin index. If provided, angular
            matching is skipped.
        angle_mode: Angle convention used for nearest-bin selection:
            ``"theta"`` for detector angles or ``"alpha"`` for
            surface/source angles.
        angle_tolerance_deg: Optional maximum accepted angular mismatch.
        context: Runtime device/dtype for returned tensors.

    Returns:
        Dictionary with selected grids, flux slice, metadata, particle label,
        angle values, and source path.
    """
    dev = context.device
    dtype = context.dtype

    if particle not in flux_data:
        raise KeyError(
            f"Particle '{particle}' not found. Available: {sorted(flux_data.keys())}"
        )

    group = flux_data[particle]

    if angle_mode == "alpha" and "alpha_grid_deg" not in group:
        angle_mode = "theta"

    angle_grid = (
        group["alpha_grid_deg"]
        if angle_mode == "alpha"
        else group["theta_grid_deg"]
    )
    angle_grid = as_tensor(angle_grid, device=dev, dtype=dtype)

    if angle_index is None:
        distance = _angle_distance(
            angle_grid,
            alpha_deg=alpha_deg,
            theta_deg=theta_deg,
            angle_mode=angle_mode,
        )
        angle_index = int(torch.argmin(distance).item())

        if angle_tolerance_deg is not None:
            best = float(distance[angle_index].item())
            if best > angle_tolerance_deg:
                raise ValueError(
                    f"Closest {angle_mode} angle is {best:.6g} deg away, "
                    f"larger than tolerance {angle_tolerance_deg}."
                )
    else:
        angle_index = int(angle_index)

    if angle_index < 0 or angle_index >= angle_grid.numel():
        raise IndexError("angle_index is outside the available angle grid.")

    E = as_tensor(group["E_grid_GeV"], device=dev, dtype=dtype)
    h = as_tensor(group["h_grid_km"], device=dev, dtype=dtype)
    theta_grid = as_tensor(group["theta_grid_deg"], device=dev, dtype=dtype)

    phi_Eh = as_tensor(
        group["phi_E_theta_h"][angle_index],
        device=dev,
        dtype=dtype,
    )

    entry = group["entries"][angle_index]
    metadata = group["metadata"][angle_index]

    alpha_value = None
    if "alpha_grid_deg" in group:
        alpha_value = float(
            as_tensor(group["alpha_grid_deg"][angle_index], device="cpu", dtype=dtype).item()
        )

    theta_value = float(theta_grid[angle_index].detach().cpu().item())

    return {
        "particle": particle,
        "flavour_index": flavour_index(_canonical_flavour_name(particle)),
        "angle_index": angle_index,
        "angle_mode": angle_mode,
        "alpha_deg": alpha_value,
        "theta_deg": theta_value,
        "E_grid_GeV": E,
        "h_grid_km": h,
        "phi_Eh": phi_Eh,
        "phi_E_theta": as_tensor(
            group["phi_E_theta"][angle_index],
            device=dev,
            dtype=dtype,
        ),
        "f_Eh": as_tensor(
            group["f_theta_E_h"][angle_index],
            device=dev,
            dtype=dtype,
        ),
        "metadata": metadata,
        "entry": entry,
        "path": group["paths"][angle_index],
    }


@torch.no_grad()
def build_atmosphere_trajectories(
    selected: Dict[str, object],
    *,
    detector_depth_m: float = 0.0,
    trajectory_steps: int = 200,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> Dict[str, torch.Tensor]:
    """
    Build atmosphere trajectory diagnostics for a selected flux slice.

    Args:
        selected: Selection dictionary returned by
            ``select_particle_angle_flux``.
        detector_depth_m: Detector depth below the Earth surface in metres.
        trajectory_steps: Number of points in the altitude/path diagnostic
            grid.
        context: Runtime device/dtype for geometry tensors.

    Returns:
        Dictionary containing atmosphere path length, atmosphere path grid,
        and altitude grid along the detector-to-production ray.
    """
    dev, dtype = context.device, context.dtype
    h = as_tensor(selected["h_grid_km"], device=dev, dtype=dtype)
    theta = as_tensor(selected["theta_deg"], device=dev, dtype=dtype)
    depth_km = torch.as_tensor(detector_depth_m / 1.0e3, device=dev, dtype=dtype)

    L_atm = atmosphere_path_length(
        h_km=h,
        theta_deg=theta,
        depth_km=depth_km,
        device=dev,
        dtype=dtype,
    )

    s_grid, h_path_grid = atmosphere_path_grid(
        h_km=h,
        theta_deg=theta,
        depth_km=depth_km,
        n_steps=trajectory_steps,
        device=dev,
        dtype=dtype,
    )

    return {
        "L_atm_km": L_atm,
        "s_atm_grid_km": s_grid,
        "h_path_grid_km": h_path_grid,
    }


@torch.no_grad()
def propagate_atmosphere_coherent(
    selected: Dict[str, object],
    config: PropagationConfig,
    *,
    initial_flavour: Optional[Union[str, int]] = None,
    trajectory_steps: int = 200,
    debug: bool = False,
) -> Dict[str, object]:
    """
    Propagate a coherent flavour state through the atmosphere.

    Args:
        selected: Flux slice returned by ``select_particle_angle_flux``.
        config: Runtime, oscillation, Earth, and atmosphere settings shared
            by every tpeanuts pipeline.
        initial_flavour: Optional initial flavour. Defaults to the selected
            particle key.
        trajectory_steps: Number of diagnostic trajectory grid points.
        debug: If True, print grid-size diagnostics.

    Returns:
        Dictionary with the initial state, surface coherent states,
        atmosphere evolutor, surface probabilities, and trajectory metadata.
    """
    context = config.runtime
    dev, dtype = context.device, context.dtype

    from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
    E = as_tensor(selected["E_grid_GeV"], device=dev, dtype=dtype)
    h = as_tensor(selected["h_grid_km"], device=dev, dtype=dtype)
    theta = as_tensor(selected["theta_deg"], device=dev, dtype=dtype)
    depth_km = torch.as_tensor(config.detector_depth_m / 1.0e3, device=dev, dtype=dtype)

    flavour = initial_flavour if initial_flavour is not None else selected["particle"]
    psi0 = flavour_state(flavour, context=context)

    E_grid, h_grid = torch.meshgrid(E, h, indexing="ij")

    if debug:
        print(
            "atmosphere coherent batched grid: "
            f"n_E={E.numel()}, n_h={h.numel()}, n_steps={config.atmosphere.nsteps}"
        )

    S_atm, _ = atmosphere_evolutor(
        config.oscillation,
        1.0e3 * E_grid,
        h_grid,
        theta,
        depth_km,
        atmosphere=config.atmosphere,
        context=context,
    )

    surface_states = torch.einsum("...ab,b->...a", S_atm, psi0)

    trajectories = build_atmosphere_trajectories(
        selected,
        detector_depth_m=config.detector_depth_m,
        trajectory_steps=trajectory_steps,
        context=context,
    )

    return {
        "mode": "coherent",
        "selected": selected,
        "initial_state": psi0,
        "surface_states": surface_states,
        "S_atm": S_atm,
        "surface_probabilities": probability_coherent_state(
            surface_states,
            real_dtype=dtype,
        ),
        "trajectories": trajectories,
    }


@torch.no_grad()
def propagate_earth_coherent(
    atmosphere_result: Dict[str, object],
    config: PropagationConfig,
    *,
    profile_earth: Optional[EarthProfile] = None,
    debug: bool = False,
) -> Dict[str, object]:
    """
    Continue coherent propagation from Earth surface to detector.

    Args:
        atmosphere_result: Result returned by
            ``propagate_atmosphere_coherent``.
        config: Runtime, oscillation, and Earth settings shared by every
            tpeanuts pipeline.
        profile_earth: Optional already-built EarthProfile instance. When
            omitted, one is built from ``config.earth``.
        debug: If True, print grid-size diagnostics.

    Returns:
        Dictionary with Earth evolutors, detector coherent states, and
        beta-to-flavour detector probabilities.
    """
    context = config.runtime
    dev, dtype = context.device, context.dtype
    cdtype = cdtype_from_real(dtype)

    selected = atmosphere_result["selected"]
    E = as_tensor(selected["E_grid_GeV"], device=dev, dtype=dtype)
    h = as_tensor(selected["h_grid_km"], device=dev, dtype=dtype)
    theta = as_tensor(selected["theta_deg"], device=dev, dtype=dtype)

    surface_states = atmosphere_result.get("surface_states", None)

    if surface_states is None:
        raise ValueError(
            "propagate_earth_coherent requires coherent atmosphere states. "
            "Use propagate_atmosphere_coherent before this function."
        )

    surface_states = as_tensor(surface_states, device=dev, dtype=cdtype)

    profile_earth, _ = prepare_earth_profile(
        profile_earth,
        earth=config.earth,
        context=context,
    )

    if debug:
        print(f"earth coherent batched grid: n_E={E.numel()}, n_h={h.numel()}")

    earth_operators = earth_evolutor_from_zenith(
        profile_earth=profile_earth,
        oscillation=config.oscillation,
        E_MeV=1.0e3 * E,
        theta_deg=theta,
        depth_m=float(config.detector_depth_m),
        reunitarize=config.reunitarize_earth,
    )

    detector_states = torch.matmul(
        earth_operators.unsqueeze(-3),
        surface_states.unsqueeze(-1),
    ).squeeze(-1)

    probabilities_beta_to_i = probability_coherent_state(
        detector_states,
        real_dtype=dtype,
    )

    return {
        "mode": "coherent",
        "selected": selected,
        "atmosphere": atmosphere_result,
        "S_earth": earth_operators,
        "detector_states": detector_states,
        "probabilities_beta_to_i": probabilities_beta_to_i,
    }


@torch.no_grad()
def integrate_initial_and_surface_fluxes(
    selected_by_flavour: Dict[str, Dict[str, object]],
    atmosphere_by_flavour: Optional[Dict[str, Dict[str, object]]] = None,
    *,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> Dict[str, object]:
    """
    Integrate initial and atmosphere-surface fluxes over production height.

    Args:
        selected_by_flavour: Mapping from produced flavour to selected flux
            slices.
        atmosphere_by_flavour: Optional mapping from produced flavour to
            atmosphere propagation results.
        context: Runtime device/dtype for flux tensors.

    Returns:
        Dictionary with flavour-resolved and total initial/surface fluxes and
        corresponding height-integrated probabilities.
    """
    dev, dtype = context.device, context.dtype

    initial_flux_by_beta = {}
    surface_flux_by_beta = {}
    initial_probability_by_beta = {}
    surface_probability_by_beta = {}

    initial_flux_total = None
    surface_flux_total = None

    for beta, selected in selected_by_flavour.items():
        h = as_tensor(selected["h_grid_km"], device=dev, dtype=dtype)
        phi_Eh = as_tensor(selected["phi_Eh"], device=dev, dtype=dtype)
        beta_index = int(selected.get("flavour_index", flavour_index(_canonical_flavour_name(beta))))

        source_flux_E = torch.trapezoid(phi_Eh, x=h, dim=-1)

        initial_flux_Ei = torch.zeros(
            (*source_flux_E.shape, 3),
            device=dev,
            dtype=dtype,
        )
        initial_flux_Ei[..., beta_index] = source_flux_E

        initial_prob_Ei = torch.zeros_like(initial_flux_Ei)
        initial_prob_Ei[..., beta_index] = torch.where(
            source_flux_E > 0.0,
            torch.ones_like(source_flux_E),
            torch.zeros_like(source_flux_E),
        )

        initial_flux_by_beta[beta] = initial_flux_Ei
        initial_probability_by_beta[beta] = initial_prob_Ei

        if initial_flux_total is None:
            initial_flux_total = initial_flux_Ei.clone()
        else:
            initial_flux_total = initial_flux_total + initial_flux_Ei

        if atmosphere_by_flavour is None or beta not in atmosphere_by_flavour:
            continue

        atmosphere_result = atmosphere_by_flavour[beta]
        probs = as_tensor(
            atmosphere_result["surface_probabilities"],
            device=dev,
            dtype=dtype,
        )

        surface_flux_Ehi = phi_Eh[..., None] * probs
        surface_flux_Ei = torch.trapezoid(surface_flux_Ehi, x=h, dim=-2)

        denom = source_flux_E[..., None].clamp_min(torch.finfo(dtype).tiny)
        surface_prob_Ei = surface_flux_Ei / denom

        surface_flux_by_beta[beta] = surface_flux_Ei
        surface_probability_by_beta[beta] = surface_prob_Ei

        if surface_flux_total is None:
            surface_flux_total = surface_flux_Ei.clone()
        else:
            surface_flux_total = surface_flux_total + surface_flux_Ei

    return {
        "initial_flux_by_beta": initial_flux_by_beta,
        "surface_flux_by_beta": surface_flux_by_beta,
        "initial_probability_by_beta": initial_probability_by_beta,
        "surface_probability_by_beta": surface_probability_by_beta,
        "initial_flux_total_Ei": initial_flux_total,
        "surface_flux_total_Ei": surface_flux_total,
    }


@torch.no_grad()
def integrate_height_and_sum_flavours(
    propagated_by_flavour: Dict[str, Dict[str, object]],
    *,
    context: RuntimeContext = RuntimeContext.resolve(None, torch.float64),
) -> Dict[str, object]:
    """
    Integrate detector fluxes over height and sum produced flavours.

    Args:
        propagated_by_flavour: Mapping from produced flavour to coherent
            Earth-propagation results.
        context: Runtime device/dtype for flux tensors.

    Returns:
        Dictionary with height-resolved fluxes, height-integrated fluxes,
        optional weighted coherent states, and total detector flux.
    """
    dev, dtype = context.device, context.dtype
    cdtype = cdtype_from_real(dtype)

    height_resolved_flux_by_beta = {}
    integrated_flux_by_beta = {}
    weighted_detector_states_by_beta = {}
    integrated_weighted_states_by_beta = {}

    detector_flux_total = None

    for beta, result in propagated_by_flavour.items():
        selected = result["selected"]
        h = as_tensor(selected["h_grid_km"], device=dev, dtype=dtype)
        phi_Eh = as_tensor(selected["phi_Eh"], device=dev, dtype=dtype)
        probs = as_tensor(result["probabilities_beta_to_i"], device=dev, dtype=dtype)

        flux_Ehi = phi_Eh[..., None] * probs
        flux_Ei = torch.trapezoid(flux_Ehi, x=h, dim=-2)

        height_resolved_flux_by_beta[beta] = flux_Ehi
        integrated_flux_by_beta[beta] = flux_Ei

        states = result.get("detector_states", None)
        if states is not None:
            states = as_tensor(states, device=dev, dtype=cdtype)
            weighted_states = phi_Eh[..., None].to(cdtype) * states
            integrated_states = torch.trapezoid(weighted_states, x=h, dim=-2)
            weighted_detector_states_by_beta[beta] = weighted_states
            integrated_weighted_states_by_beta[beta] = integrated_states

        if detector_flux_total is None:
            detector_flux_total = flux_Ei.clone()
        else:
            detector_flux_total = detector_flux_total + flux_Ei

    return {
        "height_resolved_flux_by_beta": height_resolved_flux_by_beta,
        "integrated_flux_by_beta": integrated_flux_by_beta,
        "weighted_detector_states_by_beta": weighted_detector_states_by_beta,
        "integrated_weighted_states_by_beta": integrated_weighted_states_by_beta,
        "detector_flux_total_Ei": detector_flux_total,
    }
