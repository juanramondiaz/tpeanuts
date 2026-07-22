"""Solar-neutrino workflow from production to the solar surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.probability import probability_incoherent
from tpeanuts.medium.solar.probability import solar_probability_mass
from tpeanuts.medium.solar.profile import SolarProfile, build_solar_profile
from tpeanuts.util.torch_util import as_1d_tensor
from tpeanuts.util.type import cdtype_from_real


@dataclass(frozen=True)
class SolarSurfaceResult:
    """Incoherent mass mixture produced inside the Sun at its surface."""

    source: str
    E_MeV: torch.Tensor
    profile: SolarProfile
    production_distribution: torch.Tensor
    mass_weights: torch.Tensor
    flavour_probabilities: torch.Tensor


@torch.no_grad()
def propagate_solar_to_surface(
    *,
    E_MeV,
    config: PropagationConfig,
    source: str,
    solar_profile: Optional[SolarProfile] = None,
    method: str = "adiabatic",
    legacy_precision: bool = False,
    include_matter_nc: Optional[bool] = None,
) -> SolarSurfaceResult:
    """Build the incoherent solar-surface mass mixture for one source.

    Args:
        E_MeV: Neutrino energy in MeV.
        config: Propagation configuration bundling ``oscillation``, the
            solar profile construction settings (``config.solar``), and the
            runtime device/dtype.
        source: Solar source key.
        solar_profile: Optional pre-built ``SolarProfile``; None loads the
            default/configured one.
        method: ``"adiabatic"`` (default) or ``"numerical"`` (see
            ``medium.solar.probability.solar_probability_mass``).
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation.
        include_matter_nc: If True/False, applied/not applied. If ``None``
            (the default), auto-resolved per-call: the 3+1 sterile
            extension's neutral-current matter term is included whenever
            ``config.oscillation`` is sterile and the profile has
            neutron-density data available, and omitted otherwise (with a
            ``RuntimeWarning`` if sterile was requested but the data is
            missing) -- see ``core.common.oscillation.
            resolve_include_matter_nc``. Always omitted for the plain
            3-flavour case.
    """
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
    fraction = profile.production_distribution(source)
    mass_weights = solar_probability_mass(
        config.oscillation,
        energy,
        profile,
        source,
        method=method,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
    )
    identity = torch.eye(
        int(config.oscillation.pmns.n_flavours),
        device=mass_weights.device,
        dtype=cdtype_from_real(mass_weights.dtype),
    )
    flavour_probabilities = probability_incoherent(
        identity,
        mass_weights,
        pmns=config.oscillation.pmns,
        antinu=config.oscillation.antinu,
        real_dtype=mass_weights.dtype,
    )
    return SolarSurfaceResult(
        source=source,
        E_MeV=energy,
        profile=profile,
        production_distribution=fraction,
        mass_weights=mass_weights,
        flavour_probabilities=flavour_probabilities,
    )
