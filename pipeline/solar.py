"""Solar-neutrino workflow from production to the solar surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.medium.solar.probability import solar_probability_mass, solar_probability_state
from tpeanuts.medium.solar.profile import SolarProfile, build_solar_profile
from tpeanuts.util.torch_util import as_1d_tensor


@dataclass(frozen=True)
class SolarSurfaceResult:
    """Incoherent mass mixture produced inside the Sun at its surface."""

    source: str
    E_MeV: torch.Tensor
    profile: SolarProfile
    production_fraction: torch.Tensor
    mass_weights: torch.Tensor
    flavour_probabilities: torch.Tensor


@torch.no_grad()
def propagate_solar_to_surface(
    *,
    E_MeV,
    config: PropagationConfig,
    source: str,
    solar_profile: Optional[SolarProfile] = None,
    legacy_precision: bool = False,
) -> SolarSurfaceResult:
    """Build the incoherent solar-surface mass mixture for one source."""
    context = config.runtime
    energy = as_1d_tensor(
        E_MeV,
        name="E_MeV",
        device=context.device,
        dtype=context.dtype,
    )
    profile = build_solar_profile(
        solar_profile,
        params=config.solar,
        context=context,
    )
    fraction = profile.normalized_fraction(source)
    mass_weights = solar_probability_mass(
        config.oscillation,
        energy,
        profile,
        source,
        legacy_precision=legacy_precision,
    )
    flavour_probabilities = solar_probability_state(
        config.oscillation,
        energy,
        profile,
        source,
        legacy_precision=legacy_precision,
    )
    return SolarSurfaceResult(
        source=source,
        E_MeV=energy,
        profile=profile,
        production_fraction=fraction,
        mass_weights=mass_weights,
        flavour_probabilities=flavour_probabilities,
    )
