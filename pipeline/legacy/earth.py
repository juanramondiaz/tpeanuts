"""Legacy Earth workflow from the surface to a detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.medium.earth.validation import (
    default_legacy_earth_density_path,
    legacy_earth_density,
    legacy_earth_probabilities,
    legacy_earth_probabilities_integrated,
    legacy_integrate_probabilities,
    legacy_nadir_exposure,
)
from tpeanuts.pipeline.legacy._common import validate_legacy_configuration
from tpeanuts.util.torch_util import as_1d_tensor
from tpeanuts.util.type import TensorLike, cdtype_from_real


@dataclass(frozen=True)
class LegacyEarthDetectorResult:
    """Legacy Earth probabilities for an incident state."""

    incident_state: torch.Tensor
    incident_basis: Literal["flavour", "mass"]
    E_MeV: torch.Tensor
    legacy_density: object
    eta: Optional[torch.Tensor]
    exposure: Optional[torch.Tensor]
    probabilities_eta: Optional[torch.Tensor]
    probabilities_integrated: Optional[torch.Tensor]
    integration_method: Optional[str]


def _legacy_earth_inputs(
    incident_state: TensorLike,
    E_MeV: TensorLike,
    config: PropagationConfig,
    earth_density: Optional[object],
    incident_basis: Literal["flavour", "mass"],
) -> tuple[torch.Tensor, torch.Tensor, object]:
    validate_legacy_configuration(config)
    context = config.runtime
    raw_state = torch.as_tensor(incident_state)
    state = raw_state.to(
        device=context.device,
        dtype=(
            context.dtype
            if incident_basis == "mass"
            else cdtype_from_real(context.dtype)
        ),
    )
    energy = as_1d_tensor(
        E_MeV, name="E_MeV", device=context.device, dtype=context.dtype
    )
    if earth_density is None:
        kwargs = config.earth.profile_perturbative_kwargs or {}
        earth_density = legacy_earth_density(
            kwargs.get("density_file") or default_legacy_earth_density_path(),
            tabulated_density=kwargs.get("tabulated_density", False),
        )
    return state, energy, earth_density


@torch.no_grad()
def propagate_legacy_earth_to_detector(
    incident_state: TensorLike,
    *,
    E_MeV: TensorLike,
    config: PropagationConfig,
    incident_basis: Literal["flavour", "mass"] = "flavour",
    earth_density: Optional[object] = None,
    eta: Optional[TensorLike] = None,
    exposure_normalized: bool = True,
) -> LegacyEarthDetectorResult:
    """Evaluate eta-resolved legacy Earth probabilities."""
    state, energy, density = _legacy_earth_inputs(
        incident_state, E_MeV, config, earth_density, incident_basis
    )
    eta_np, exposure_np, _, _ = legacy_nadir_exposure(
        eta,
        exposure=config.exposure,
        normalized=exposure_normalized,
    )
    probabilities = legacy_earth_probabilities(
        state,
        config.oscillation,
        energy,
        eta_np,
        config.detector_depth_m,
        density=density,
        massbasis=incident_basis == "mass",
        method=config.earth.method,
    )
    context = config.runtime
    return LegacyEarthDetectorResult(
        incident_state=state,
        incident_basis=incident_basis,
        E_MeV=energy,
        legacy_density=density,
        eta=torch.as_tensor(eta_np, device=context.device, dtype=context.dtype),
        exposure=torch.as_tensor(
            exposure_np, device=context.device, dtype=context.dtype
        ),
        probabilities_eta=torch.as_tensor(
            probabilities, device=context.device, dtype=context.dtype
        ),
        probabilities_integrated=None,
        integration_method=None,
    )


@torch.no_grad()
def propagate_legacy_earth_to_detector_integrated(
    incident_state: TensorLike,
    *,
    E_MeV: TensorLike,
    config: PropagationConfig,
    incident_basis: Literal["flavour", "mass"] = "flavour",
    earth_density: Optional[object] = None,
    eta: Optional[TensorLike] = None,
    exposure_normalized: bool = True,
    integration_method: Literal[
        "peanuts_integrated", "legacy_rectangle", "trapezoid"
    ] = "peanuts_integrated",
) -> LegacyEarthDetectorResult:
    """Evaluate and integrate legacy Earth probabilities with an explicit rule."""
    if integration_method == "peanuts_integrated":
        if eta is not None:
            raise ValueError(
                "eta cannot be supplied with integration_method='peanuts_integrated'; "
                "use 'legacy_rectangle' or 'trapezoid' for an explicit grid."
            )
        state, energy, density = _legacy_earth_inputs(
            incident_state, E_MeV, config, earth_density, incident_basis
        )
        integrated = legacy_earth_probabilities_integrated(
            state,
            config.oscillation,
            energy,
            config.detector_depth_m,
            density=density,
            exposure=config.exposure,
            normalized=exposure_normalized,
            method=config.earth.method,
        )
        return LegacyEarthDetectorResult(
            incident_state=state,
            incident_basis=incident_basis,
            E_MeV=energy,
            legacy_density=density,
            eta=None,
            exposure=None,
            probabilities_eta=None,
            probabilities_integrated=torch.as_tensor(
                integrated,
                device=config.runtime.device,
                dtype=config.runtime.dtype,
            ),
            integration_method=integration_method,
        )

    angular = propagate_legacy_earth_to_detector(
        incident_state,
        E_MeV=E_MeV,
        config=config,
        incident_basis=incident_basis,
        earth_density=earth_density,
        eta=eta,
        exposure_normalized=exposure_normalized,
    )
    eta_np = angular.eta.detach().cpu().numpy()
    exposure_np = angular.exposure.detach().cpu().numpy()
    _, _, _, deta = legacy_nadir_exposure(
        eta,
        exposure=config.exposure,
        normalized=exposure_normalized,
    )
    integrated = legacy_integrate_probabilities(
        angular.probabilities_eta.detach().cpu().numpy(),
        eta_np,
        exposure_np,
        method=integration_method,
        deta=deta,
    )
    return LegacyEarthDetectorResult(
        incident_state=angular.incident_state,
        incident_basis=angular.incident_basis,
        E_MeV=angular.E_MeV,
        legacy_density=angular.legacy_density,
        eta=angular.eta,
        exposure=angular.exposure,
        probabilities_eta=angular.probabilities_eta,
        probabilities_integrated=torch.as_tensor(
            integrated,
            device=config.runtime.device,
            dtype=config.runtime.dtype,
        ),
        integration_method=integration_method,
    )
