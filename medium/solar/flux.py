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
Solar source flux helpers.

This module sits above the adiabatic solar-probability functions. It selects
one or more production sources from a SolarProfile, computes their final
flavour probabilities with ``solar_probability_state``, and delegates the
generic probability-to-flux multiplication and energy integration to
``core.common.flux``.

Module functions:
    solar_flux_state(...)
        Weight source probabilities by total fluxes and profile spectra.
    solar_flux_integrated(...)
        Integrate the energy-resolved solar flux over energy to obtain a
        neutrino rate. Uses the profile spectrum unless explicitly overridden.
"""



from __future__ import annotations

import torch

from tpeanuts.core.common.flux import flux_integrated, flux_state
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.solar.geometry import sun_earth_distance_factor
from tpeanuts.medium.solar.probability import solar_probability_state


def solar_flux_state(
    sources: str | list[str] | tuple[str, ...],
    profile,
    oscillation: OscillationParameters,
    E_MeV,
    source_spectrum: torch.Tensor | None = None,
    *,
    method: str = "adiabatic",
    legacy_precision: bool = False,
    include_matter_nc: bool | None = None,
    date: str | None = None,
) -> torch.Tensor:
    """Compute flavour-resolved solar flux for one or more sources.

    Args:
        sources: Solar source key or ordered source keys available in
            ``profile``.
        profile: SolarProfile-like object exposing source fluxes and
            production fractions.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        source_spectrum: Optional spectral override. None interpolates the
            spectrum stored in ``profile``. For several sources, an override
            must include the leading source dimension when needed.
        method: ``"adiabatic"`` (default) or ``"numerical"`` (see
            ``medium.solar.probability.solar_probability_mass``).
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation (see ``medium.solar.matter_mixing``).
        include_matter_nc: If True/False, applied/not applied. If ``None``
            (the default), auto-resolved per-call (see
            ``core.common.oscillation.resolve_include_matter_nc``): the 3+1
            sterile extension's neutral-current matter term is included
            whenever ``oscillation`` is sterile and ``profile`` has
            neutron-density data available, and omitted otherwise (with a
            ``RuntimeWarning`` if sterile was requested but the data is
            missing). Always omitted for the plain 3-flavour case.
        date: Optional ISO ``"YYYY-MM-DD"`` calendar date. None (the
            default) leaves the flux at its 1 AU reference normalization,
            matching every solar-model flux table. When given, the result is
            scaled by the instantaneous Sun-Earth distance modulation
            ``(1 AU / R(date))^2`` (see
            ``medium.solar.geometry.sun_earth_distance_factor``); Earth's
            orbit is elliptical, so this varies by about +-3.4% over the
            year. For a period average instead of one date, see
            ``pipeline.solar_earth.propagate_solar_to_earth_detector``'s
            ``average_sun_earth_distance`` option.

    Returns:
        Flavour-resolved flux with optional leading source dimensions and final
        flavour dimension 3.
    """
    probabilities = solar_probability_state(
        oscillation,
        E_MeV,
        profile,
        sources,
        method=method,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
    )

    if source_spectrum is None:
        source_spectrum = profile.spectrum(sources, E_MeV)

    if isinstance(sources, str):
        fluxes = profile.flux(sources)
    else:
        fluxes = torch.stack(
            [
                profile.flux(source)
                for source in sources
            ],
            dim=0,
        )

    flux = flux_state(probabilities, fluxes, source_spectrum)

    if date is not None:
        flux = flux * sun_earth_distance_factor(date, device=flux.device, dtype=flux.dtype)

    return flux


def solar_flux_integrated(
    sources: str | list[str] | tuple[str, ...],
    profile,
    oscillation: OscillationParameters,
    E_MeV,
    source_spectrum: torch.Tensor | None = None,
    *,
    method: str = "adiabatic",
    legacy_precision: bool = False,
    energy_dim: int = -2,
    include_matter_nc: bool | None = None,
    date: str | None = None,
) -> torch.Tensor:
    """Integrate the energy-resolved solar flux over energy.

    Builds the flavour-resolved solar flux with ``solar_flux_state`` and
    integrates it over energy with ``core.common.flux.flux_integrated``,
    obtaining a physical rate (unnormalized, unlike
    ``solar_probability_integrated``).

    ``profile.flux(source)`` is the source's *total* (already energy
    -integrated) flux, not a spectral density: multiplying it by
    ``P(E)`` gives ``Phi_total * P(E)``, which does not have units of
    ``dPhi/dE`` on its own. A normalized spectrum is therefore obtained from
    ``profile`` by default, or accepted as an explicit override, so that the
    quantity handed to ``flux_integrated`` is a genuine differential flux;
    without it, integrating over energy would silently pick up a spurious
    factor of the energy grid's units and depend on the grid's spacing/range,
    matching the convention enforced by ``core.common.probability``.

    Args:
        sources: Solar source key or ordered source keys available in
            ``profile``.
        profile: SolarProfile-like object exposing source fluxes and
            production fractions.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy grid in MeV, one-dimensional.
        source_spectrum: Optional normalized production spectral-density
            override. None uses ``profile.spectrum(sources, E_MeV)``.
        method: ``"adiabatic"`` (default) or ``"numerical"`` (see
            ``medium.solar.probability.solar_probability_mass``).
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation (see ``medium.solar.matter_mixing``).
        energy_dim: Axis of the resulting flux tensor holding the energy
            grid. Must not be the final (flavour) axis.
        include_matter_nc: If True/False, applied/not applied. If ``None``
            (the default), auto-resolved per-call (see
            ``solar_flux_state``/``core.common.oscillation.
            resolve_include_matter_nc``).
        date: Optional ISO ``"YYYY-MM-DD"`` calendar date, forwarded to
            ``solar_flux_state`` (see there). None (the default) leaves the
            rate at its 1 AU reference normalization.

    Returns:
        Flux integrated over energy (a rate), with the energy axis removed.

    """
    flux_grid = solar_flux_state(
        sources,
        profile,
        oscillation,
        E_MeV,
        source_spectrum,
        method=method,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
        date=date,
    )

    return flux_integrated(flux_grid, E_MeV, energy_dim=energy_dim)
