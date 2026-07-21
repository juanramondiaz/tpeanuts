"""Legacy solar workflow from production to the solar surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.medium.solar.validation import (
    legacy_solar_flavour_probabilities,
    legacy_solar_mass_weights,
    legacy_solar_model,
)
from tpeanuts.pipeline.legacy._common import validate_legacy_configuration
from tpeanuts.util.torch_util import as_1d_tensor
from tpeanuts.util.type import TensorLike


@dataclass(frozen=True)
class LegacySolarSurfaceResult:
    """Legacy incoherent mass mixture at the solar surface."""

    source: str
    E_MeV: torch.Tensor
    radius: torch.Tensor
    production_fraction: torch.Tensor
    mass_weights: torch.Tensor
    flavour_probabilities: Optional[torch.Tensor]
    legacy_model: object


@torch.no_grad()
def propagate_legacy_solar_to_surface(
    *,
    E_MeV: TensorLike,
    config: PropagationConfig,
    source: str,
    solar_model: Optional[object] = None,
    solar_model_file: Optional[str] = None,
    solar_flux_file: Optional[str] = None,
    validate_psolar: bool = False,
) -> LegacySolarSurfaceResult:
    """Evaluate legacy solar mass weights without any Earth propagation."""
    validate_legacy_configuration(config)
    context = config.runtime
    energy = as_1d_tensor(
        E_MeV, name="E_MeV", device=context.device, dtype=context.dtype
    )
    model = legacy_solar_model(
        solar_model,
        solar_model_file=solar_model_file,
        solar_flux_file=solar_flux_file,
    )
    energy_numpy = energy.detach().cpu().numpy()
    mass_weights = legacy_solar_mass_weights(
        config.oscillation,
        energy_numpy,
        source,
        solar_model=model,
    )
    flavour = (
        legacy_solar_flavour_probabilities(
            config.oscillation,
            energy_numpy,
            source,
            solar_model=model,
        )
        if validate_psolar
        else None
    )
    fraction = np.asarray(model.fraction(source), dtype=np.float64)
    radius = np.asarray(model.radius(), dtype=np.float64)
    norm = np.trapz(fraction, x=radius)
    fraction = fraction / max(float(norm), np.finfo(np.float64).tiny)
    return LegacySolarSurfaceResult(
        source=source,
        E_MeV=energy,
        radius=torch.as_tensor(radius, device=context.device, dtype=context.dtype),
        production_fraction=torch.as_tensor(
            fraction, device=context.device, dtype=context.dtype
        ),
        mass_weights=torch.as_tensor(
            mass_weights, device=context.device, dtype=context.dtype
        ),
        flavour_probabilities=(
            None
            if flavour is None
            else torch.as_tensor(flavour, device=context.device, dtype=context.dtype)
        ),
        legacy_model=model,
    )
