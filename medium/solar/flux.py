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

This module sits above the adiabatic solar-probability functions. It
computes final flavour probabilities for one or more production sources
with ``solar_probability_state`` (taking a ``medium`` and a ``source``
separately, see ``medium.solar.probability``), and delegates the generic
probability-to-flux multiplication and energy integration to
``core.common.flux``. The solar flux normalization and spectrum come from
``source`` (``source.solar.SolarNeutrinoSource``) rather than the medium,
since the solar flux is a property of what the Sun emits, not of how it
propagates.

Module functions:
    solar_flux_state(...)
        Weight source probabilities by total fluxes and source spectra.
    solar_flux_integrated(...)
        Integrate the energy-resolved solar flux over energy to obtain a
        neutrino rate. Uses the source spectrum unless explicitly overridden.
"""



from __future__ import annotations

from typing import Optional, Sequence

import torch

from tpeanuts.core.common.flux import flux_integrated, flux_state
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.numerical.geometry import OdeMethod
from tpeanuts.medium.vacuum.solar_geometry import sun_earth_distance_factor
from tpeanuts.medium.solar.probability import solar_probability_state
from tpeanuts.source.solar import ContinuousSolarSpectrum, SolarLineSpectrum
from tpeanuts.util.type import TensorLike


def solar_flux_state(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    medium: object,
    source: object,
    sources: str | Sequence[str],
    source_spectrum: torch.Tensor | None = None,
    *,
    method: str = "adiabatic_approximated",
    use_LZ: bool = False,
    legacy_precision: bool = False,
    include_matter_nc: Optional[bool] = None,
    numerical_sampling: Optional[OdeMethod] = "midpoint",
    date: str | None = None,
) -> torch.Tensor:
    """Compute flavour-resolved solar flux for one or more sources.

    Args:
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        medium: ``medium.solar.profile.SolarMediumProfile``-like object
            (see ``medium.solar.probability.solar_probability_mass``).
        source: ``source.solar.SolarNeutrinoSource``-like object exposing
            source fluxes and production fractions.
        sources: Solar source key or ordered source keys available in
            ``source``.
        source_spectrum: Optional spectral override. None interpolates the
            spectrum stored in ``source``. For several sources, an override
            must include the leading source dimension when needed. If
            ``sources`` is a single line-spectrum source (e.g. "7Be",
            "pep") and this is None, ``E_MeV`` must equal
            ``source.spectrum_table(sources).energy_MeV`` exactly -- the
            source's tabulated line weights are used directly instead of
            ``source.spectrum()`` (which raises ``TypeError`` for a line
            source, since a discrete spectrum has no dPhi/dE to
            interpolate).
        method: ``"numerical"``, ``"adiabatic_approximated"`` (default), or
            ``"adiabatic_exact"`` (see
            ``medium.solar.probability.solar_probability_mass``).
        use_LZ: If True, apply the local two-level Landau-Zener correction
            (see ``solar_probability_mass``).
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation (see ``medium.solar.matter_mixing``).
        include_matter_nc: If True/False, applied/not applied. If ``None``
            (the default), auto-resolved per-call (see
            ``core.common.oscillation.resolve_include_matter_nc``): the 3+1
            sterile extension's neutral-current matter term is included
            whenever ``oscillation`` is sterile and ``medium`` has
            neutron-density data available, and omitted otherwise (with a
            ``RuntimeWarning`` if sterile was requested but the data is
            missing). Always omitted for the plain 3-flavour case.
        numerical_sampling: Segment density-sampling rule, only used when
            ``method="numerical"``.
        date: Optional ISO ``"YYYY-MM-DD"`` calendar date. None (the
            default) leaves the flux at its 1 AU reference normalization,
            matching every solar-model flux table. When given, the result is
            scaled by the instantaneous Sun-Earth distance modulation
            ``(1 AU / R(date))^2`` (see
            ``medium.vacuum.solar_geometry.sun_earth_distance_factor``); Earth's
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
        medium,
        source,
        sources,
        method=method,
        use_LZ=use_LZ,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
        numerical_sampling=numerical_sampling,
    )

    if source_spectrum is None:
        spectrum_model = source.spectrum_table(sources) if isinstance(sources, str) else None
        if isinstance(spectrum_model, SolarLineSpectrum):
            E_MeV_t = torch.as_tensor(
                E_MeV, device=spectrum_model.energy_MeV.device, dtype=spectrum_model.energy_MeV.dtype,
            )
            if E_MeV_t.shape != spectrum_model.energy_MeV.shape or not torch.allclose(
                E_MeV_t, spectrum_model.energy_MeV,
            ):
                raise ValueError(
                    "A solar line source must be evaluated at its exact line "
                    "energies; pass E_MeV=source.spectrum_table(sources)."
                    "energy_MeV, or an explicit source_spectrum override."
                )
            source_spectrum = spectrum_model.weights
        else:
            source_spectrum = source.spectrum(sources, E_MeV)

    if isinstance(sources, str):
        fluxes = source.total_flux(sources)
    else:
        fluxes = torch.stack(
            [
                source.total_flux(one_source)
                for one_source in sources
            ],
            dim=0,
        )

    flux = flux_state(probabilities, fluxes, source_spectrum)

    if date is not None:
        flux = flux * source.flux_reference_distance_au ** 2 * sun_earth_distance_factor(
            date, device=flux.device, dtype=flux.dtype,
        )

    return flux


def solar_flux_integrated(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    medium: object,
    source: object,
    sources: str | Sequence[str],
    source_spectrum: torch.Tensor | ContinuousSolarSpectrum | SolarLineSpectrum | None = None,
    *,
    method: str = "adiabatic_approximated",
    use_LZ: bool = False,
    legacy_precision: bool = False,
    energy_dim: int = -2,
    include_matter_nc: Optional[bool] = None,
    numerical_sampling: Optional[OdeMethod] = "midpoint",
    date: str | None = None,
) -> torch.Tensor:
    """Integrate the energy-resolved solar flux over energy.

    Builds the flavour-resolved solar flux with ``solar_flux_state`` and
    integrates it over energy with ``core.common.flux.flux_integrated``,
    obtaining a physical rate (unnormalized, unlike
    ``solar_probability_integrated``).

    ``source.total_flux(sources)`` is the source's *total* (already energy
    -integrated) flux, not a spectral density: multiplying it by
    ``P(E)`` gives ``Phi_total * P(E)``, which does not have units of
    ``dPhi/dE`` on its own. A normalized spectrum is therefore obtained from
    ``source`` by default, or accepted as an explicit override, so that the
    quantity handed to ``flux_integrated`` is a genuine differential flux;
    without it, integrating over energy would silently pick up a spurious
    factor of the energy grid's units and depend on the grid's spacing/range,
    matching the convention enforced by ``core.common.probability``.

    Args:
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy grid in MeV, one-dimensional.
        medium: ``SolarMediumProfile``-like object (see
            ``medium.solar.probability.solar_probability_mass``).
        source: ``SolarNeutrinoSource``-like object exposing source fluxes
            and production fractions.
        sources: Solar source key or ordered source keys available in
            ``source``.
        source_spectrum: Optional normalized production spectral-density
            override. None uses ``source.spectrum(sources, E_MeV)``.
        method: ``"numerical"``, ``"adiabatic_approximated"`` (default), or
            ``"adiabatic_exact"`` (see
            ``medium.solar.probability.solar_probability_mass``).
        use_LZ: If True, apply the local two-level Landau-Zener correction
            (see ``solar_probability_mass``).
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation (see ``medium.solar.matter_mixing``).
        energy_dim: Axis of the resulting flux tensor holding the energy
            grid. Must not be the final (flavour) axis.
        include_matter_nc: If True/False, applied/not applied. If ``None``
            (the default), auto-resolved per-call (see
            ``solar_flux_state``/``core.common.oscillation.
            resolve_include_matter_nc``).
        numerical_sampling: Segment density-sampling rule, only used when
            ``method="numerical"``.
        date: Optional ISO ``"YYYY-MM-DD"`` calendar date, forwarded to
            ``solar_flux_state`` (see there). None (the default) leaves the
            rate at its 1 AU reference normalization.

    Returns:
        Flux integrated over energy (a rate), with the energy axis removed.

    """
    if source_spectrum is None:
        if not isinstance(sources, str):
            raise ValueError("Automatic spectrum selection requires exactly one solar source.")
        source_spectrum = source.spectrum_table(sources)

    if isinstance(source_spectrum, SolarLineSpectrum):
        if not isinstance(sources, str):
            raise ValueError("A SolarLineSpectrum can only be integrated for one source at a time.")
        probabilities = solar_probability_state(
            oscillation, source_spectrum.energy_MeV, medium, source, sources,
            method=method, use_LZ=use_LZ, legacy_precision=legacy_precision,
            include_matter_nc=include_matter_nc, numerical_sampling=numerical_sampling,
        )
        result = source.total_flux(sources) * (
            probabilities * source_spectrum.weights[..., None]
        ).sum(dim=-2)
        if date is not None:
            result = result * source.flux_reference_distance_au ** 2 * sun_earth_distance_factor(
                date, device=result.device, dtype=result.dtype,
            )
        return result

    if isinstance(source_spectrum, ContinuousSolarSpectrum):
        E_MeV = source_spectrum.energy_MeV
        source_spectrum = source_spectrum.density_MeV_inverse

    flux_grid = solar_flux_state(
        oscillation,
        E_MeV,
        medium,
        source,
        sources,
        source_spectrum,
        method=method,
        use_LZ=use_LZ,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
        numerical_sampling=numerical_sampling,
        date=date,
    )

    return flux_integrated(flux_grid, E_MeV, energy_dim=energy_dim)
