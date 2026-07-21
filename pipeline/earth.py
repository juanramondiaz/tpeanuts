"""Pure Earth workflow from the Earth surface to a detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.medium.earth.exposure_integration import earth_probability_exposure
from tpeanuts.medium.earth.exposure_table import prepare_nadir_exposure
from tpeanuts.medium.earth.probability import earth_probability_state
from tpeanuts.medium.earth.profile import EarthProfile, build_earth_profile
from tpeanuts.util.torch_util import as_1d_tensor
from tpeanuts.util.type import TensorLike, cdtype_from_real


@dataclass(frozen=True)
class EarthDetectorResult:
    """Earth propagation result for an incident flavour or mass state."""

    incident_state: torch.Tensor
    incident_basis: Literal["flavour", "mass"]
    E_MeV: torch.Tensor
    profile: EarthProfile
    eta: Optional[torch.Tensor]
    exposure: Optional[torch.Tensor]
    probabilities_eta: Optional[torch.Tensor]
    probabilities_exposure: Optional[torch.Tensor]


def _earth_inputs(
    incident_state: torch.Tensor,
    E_MeV: TensorLike,
    config: PropagationConfig,
    earth_profile: Optional[EarthProfile],
) -> tuple[torch.Tensor, torch.Tensor, EarthProfile]:
    context = config.runtime
    raw_state = torch.as_tensor(incident_state)
    state = raw_state.to(
        device=context.device,
        dtype=(
            cdtype_from_real(context.dtype)
            if raw_state.is_complex()
            else context.dtype
        ),
    )
    energy = as_1d_tensor(
        E_MeV,
        name="E_MeV",
        device=context.device,
        dtype=context.dtype,
    )
    profile = build_earth_profile(
        earth_profile,
        params=config.earth,
        context=context,
    )
    return state, energy, profile


@torch.no_grad()
def propagate_earth_to_detector(
    incident_state: torch.Tensor,
    *,
    E_MeV: TensorLike,
    config: PropagationConfig,
    incident_basis: Literal["flavour", "mass"] = "flavour",
    earth_profile: Optional[EarthProfile] = None,
    eta: Optional[TensorLike] = None,
) -> EarthDetectorResult:
    """Propagate an incident state over an explicit or configured eta grid."""
    state, energy, profile = _earth_inputs(
        incident_state, E_MeV, config, earth_profile
    )
    eta_grid, exposure, _ = prepare_nadir_exposure(
        eta,
        exposure=config.exposure,
        context=config.runtime,
    )
    # The low-level evolutor also accepts paired one-dimensional E/eta
    # samples. A pipeline grid is instead always the Cartesian product, so
    # make both axes explicit even when N_E == N_eta.
    energy_grid = energy[:, None]
    eta_evaluation_grid = eta_grid[None, :]
    probabilities = earth_probability_state(
        nustate=state,
        profile_earth=profile,
        oscillation=config.oscillation,
        E_MeV=energy_grid,
        eta=eta_evaluation_grid,
        depth_m=config.detector_depth_m,
        method=config.earth.method,
        massbasis=incident_basis == "mass",
        nsteps=config.earth.nsteps,
        ode_method=config.earth.ode_method,
        context=config.runtime,
        reunitarize=config.reunitarize_earth,
    )
    return EarthDetectorResult(
        incident_state=state,
        incident_basis=incident_basis,
        E_MeV=energy,
        profile=profile,
        eta=eta_grid,
        exposure=exposure,
        probabilities_eta=probabilities,
        probabilities_exposure=None,
    )


@torch.no_grad()
def propagate_earth_to_detector_exposure(
    incident_state: torch.Tensor,
    *,
    E_MeV: TensorLike,
    config: PropagationConfig,
    incident_basis: Literal["flavour", "mass"] = "flavour",
    earth_profile: Optional[EarthProfile] = None,
) -> EarthDetectorResult:
    """Propagate an incident state and average over the configured exposure."""
    state, energy, profile = _earth_inputs(
        incident_state, E_MeV, config, earth_profile
    )
    probabilities = earth_probability_exposure(
        nustate=state,
        profile_earth=profile,
        oscillation=config.oscillation,
        E_MeV=energy,
        depth_m=config.detector_depth_m,
        method=config.earth.method,
        massbasis=incident_basis == "mass",
        exposure=config.exposure,
        context=config.runtime,
        chunk_eta=config.earth.chunk_eta,
        reunitarize=config.reunitarize_earth,
        nsteps=config.earth.nsteps,
        ode_method=config.earth.ode_method,
    )
    return EarthDetectorResult(
        incident_state=state,
        incident_basis=incident_basis,
        E_MeV=energy,
        profile=profile,
        eta=None,
        exposure=None,
        probabilities_eta=None,
        probabilities_exposure=probabilities,
    )
