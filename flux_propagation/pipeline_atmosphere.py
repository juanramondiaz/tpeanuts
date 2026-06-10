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
Workflow helpers for atmospheric-neutrino production and detector fluxes.

The intended workflow is:

1. Select Phi_beta(E,h,theta) for one produced particle and one detector angle.
2. Keep the selected mceq grids and metadata together.
3. Propagate flavour states from production height h to the earth surface.
4. Keep atmospheric propagation coherent for each produced neutrino.
5. Propagate the surface states coherently through earth to the detector.
6. Build beta -> i probabilities at the detector.
7. Integrate over production height.
8. Sum contributions from all produced flavours.

Flavour convention follows the rest of tpeanuts:

    [nue, numu, nutau] -> [0, 1, 2]
"""



from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import torch

from tpeanuts.atmosphere.earth import earth_evolution_operator
from tpeanuts.atmosphere.geometry import atmospheric_path_length, atmospheric_path_grid
from tpeanuts.core.probabilities import flavour_index
from tpeanuts.earth.density import EarthDensity
from tpeanuts.util.torch_util import _default_device
from tpeanuts.util.type import _as_tensor, _cdtype_from_real


TensorLike = Union[float, int, torch.Tensor]


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if callable(device):
        device = device()
    return _default_device(device)


def _canonical_flavour_name(name: Union[str, int]) -> Union[str, int]:
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
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    dev = _resolve_device(device)
    cdtype = _cdtype_from_real(dtype)
    idx = flavour_index(_canonical_flavour_name(flavour))

    state = torch.zeros(3, device=dev, dtype=cdtype)
    state[idx] = 1.0 + 0.0j

    return state


def _angle_distance(
    angle_grid: torch.Tensor,
    *,
    alpha_deg: Optional[float],
    theta_deg: Optional[float],
    angle_mode: str,
) -> torch.Tensor:
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
    angle_mode: str = "alpha",
    angle_tolerance_deg: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, object]:
    dev = _resolve_device(device)

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
    angle_grid = _as_tensor(angle_grid, device=dev, dtype=dtype)

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

    E = _as_tensor(group["E_grid_GeV"], device=dev, dtype=dtype)
    h = _as_tensor(group["h_grid_km"], device=dev, dtype=dtype)
    theta_grid = _as_tensor(group["theta_grid_deg"], device=dev, dtype=dtype)

    phi_Eh = _as_tensor(
        group["phi_E_theta_h"][angle_index],
        device=dev,
        dtype=dtype,
    )

    entry = group["entries"][angle_index]
    metadata = group["metadata"][angle_index]

    alpha_value = None
    if "alpha_grid_deg" in group:
        alpha_value = float(
            _as_tensor(group["alpha_grid_deg"][angle_index], device="cpu", dtype=dtype).item()
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
        "phi_E_theta": _as_tensor(
            group["phi_E_theta"][angle_index],
            device=dev,
            dtype=dtype,
        ),
        "f_Eh": _as_tensor(
            group["f_theta_E_h"][angle_index],
            device=dev,
            dtype=dtype,
        ),
        "metadata": metadata,
        "entry": entry,
        "path": group["paths"][angle_index],
    }


@torch.no_grad()
def build_atmospheric_trajectories(
    selected: Dict[str, object],
    *,
    detector_depth_m: float = 0.0,
    trajectory_steps: int = 200,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    dev = _resolve_device(device)
    h = _as_tensor(selected["h_grid_km"], device=dev, dtype=dtype)
    theta = _as_tensor(selected["theta_deg"], device=dev, dtype=dtype)
    depth_km = torch.as_tensor(detector_depth_m / 1.0e3, device=dev, dtype=dtype)

    L_atm = atmospheric_path_length(
        h_km=h,
        theta_deg=theta,
        depth_km=depth_km,
        device=dev,
        dtype=dtype,
    )

    s_grid, h_path_grid = atmospheric_path_grid(
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
    pmns,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    *,
    initial_flavour: Optional[Union[str, int]] = None,
    detector_depth_m: float = 0.0,
    antinu: bool = False,
    matter: bool = True,
    ne_profile: Optional[Callable] = None,
    atmosphere_n_steps: int = 600,
    trajectory_steps: int = 200,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    debug: bool = False,
) -> Dict[str, object]:
    dev = _resolve_device(device)
    cdtype = _cdtype_from_real(dtype)

    from tpeanuts.atmosphere.propagation import atmospheric_evolution_operator
    if ne_profile is None:
        from tpeanuts.atmosphere.density import atmospheric_electron_density_profile
        ne_profile = atmospheric_electron_density_profile

    E = _as_tensor(selected["E_grid_GeV"], device=dev, dtype=dtype)
    h = _as_tensor(selected["h_grid_km"], device=dev, dtype=dtype)
    theta = _as_tensor(selected["theta_deg"], device=dev, dtype=dtype)
    depth_km = torch.as_tensor(detector_depth_m / 1.0e3, device=dev, dtype=dtype)

    flavour = initial_flavour if initial_flavour is not None else selected["particle"]
    psi0 = flavour_state(flavour, device=dev, dtype=dtype)

    E_grid, h_grid = torch.meshgrid(E, h, indexing="ij")

    if debug:
        print(
            "atmosphere coherent batched grid: "
            f"n_E={E.numel()}, n_h={h.numel()}, n_steps={atmosphere_n_steps}"
        )

    S_atm, _ = atmospheric_evolution_operator(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=1.0e3 * E_grid,
        h_km=h_grid,
        theta_deg=theta,
        depth_km=depth_km,
        antinu=antinu,
        ne_profile=ne_profile,
        n_steps=atmosphere_n_steps,
        matter=matter,
        device=dev,
        dtype=dtype,
    )

    surface_states = torch.einsum("...ab,b->...a", S_atm, psi0)

    trajectories = build_atmospheric_trajectories(
        selected,
        detector_depth_m=detector_depth_m,
        trajectory_steps=trajectory_steps,
        device=dev,
        dtype=dtype,
    )

    return {
        "mode": "coherent",
        "selected": selected,
        "initial_state": psi0,
        "surface_states": surface_states,
        "S_atm": S_atm,
        "surface_probabilities": torch.abs(surface_states) ** 2,
        "trajectories": trajectories,
    }


@torch.no_grad()
def propagate_earth_coherent(
    atmosphere_result: Dict[str, object],
    pmns,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    *,
    detector_depth_m: float = 0.0,
    density: Optional[EarthDensity] = None,
    density_file: Optional[str] = None,
    antinu: bool = False,
    reunitarize_earth: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    debug: bool = False,
) -> Dict[str, object]:
    dev = _resolve_device(device)
    cdtype = _cdtype_from_real(dtype)

    selected = atmosphere_result["selected"]
    E = _as_tensor(selected["E_grid_GeV"], device=dev, dtype=dtype)
    h = _as_tensor(selected["h_grid_km"], device=dev, dtype=dtype)
    theta = _as_tensor(selected["theta_deg"], device=dev, dtype=dtype)

    surface_states = atmosphere_result.get("surface_states", None)

    if surface_states is None:
        raise ValueError(
            "propagate_earth_coherent requires coherent atmospheric states. "
            "Use propagate_atmosphere_coherent before this function."
        )

    surface_states = _as_tensor(surface_states, device=dev, dtype=cdtype)

    if debug:
        print(f"earth coherent batched grid: n_E={E.numel()}, n_h={h.numel()}")

    earth_operators = earth_evolution_operator(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=1.0e3 * E,
        theta_deg=theta,
        detector_depth_m=detector_depth_m,
        density=density,
        density_file=density_file,
        antinu=antinu,
        reunitarize=reunitarize_earth,
        device=dev,
        dtype=dtype,
    )

    detector_states = torch.matmul(
        earth_operators.unsqueeze(-3),
        surface_states.unsqueeze(-1),
    ).squeeze(-1)

    probabilities_beta_to_i = torch.abs(detector_states) ** 2

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
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, object]:
    dev = _resolve_device(device)

    initial_flux_by_beta = {}
    surface_flux_by_beta = {}
    initial_probability_by_beta = {}
    surface_probability_by_beta = {}

    initial_flux_total = None
    surface_flux_total = None

    for beta, selected in selected_by_flavour.items():
        h = _as_tensor(selected["h_grid_km"], device=dev, dtype=dtype)
        phi_Eh = _as_tensor(selected["phi_Eh"], device=dev, dtype=dtype)
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
        probs = _as_tensor(
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
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, object]:
    dev = _resolve_device(device)
    cdtype = _cdtype_from_real(dtype)

    height_resolved_flux_by_beta = {}
    integrated_flux_by_beta = {}
    weighted_detector_states_by_beta = {}
    integrated_weighted_states_by_beta = {}

    detector_flux_total = None

    for beta, result in propagated_by_flavour.items():
        selected = result["selected"]
        h = _as_tensor(selected["h_grid_km"], device=dev, dtype=dtype)
        phi_Eh = _as_tensor(selected["phi_Eh"], device=dev, dtype=dtype)
        probs = _as_tensor(result["probabilities_beta_to_i"], device=dev, dtype=dtype)

        flux_Ehi = phi_Eh[..., None] * probs
        flux_Ei = torch.trapezoid(flux_Ehi, x=h, dim=-2)

        height_resolved_flux_by_beta[beta] = flux_Ehi
        integrated_flux_by_beta[beta] = flux_Ei

        states = result.get("detector_states", None)
        if states is not None:
            states = _as_tensor(states, device=dev, dtype=cdtype)
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
