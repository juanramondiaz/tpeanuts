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
Solar-neutrino source: production, total flux, and production spectra.

``SolarNeutrinoSource`` describes what the Sun produces, where, and with
what energy -- independent of how those neutrinos then propagate through
the solar medium (``medium.solar.profile.SolarMediumProfile``). This split
mirrors ``source.atmosphere``/``medium.atmosphere`` and
``source.reactor``/``detector.dayabay``: production physics lives here,
propagation physics lives in ``medium.*``.

``medium.solar.probability``/``medium.solar.flux`` depend on this module
(passing a built ``SolarNeutrinoSource`` alongside a
``SolarMediumProfile``); this module never depends back on ``medium.solar``.

Module contents:
    SolarSpectrum
        Tabulated normalized production spectrum for one solar source.
    SolarSourceParameters
        Construction settings for the tabulated solar source (production,
        flux, and spectrum tables).
    SolarNeutrinoSource
        Torch container for production distributions, total fluxes, and
        production spectra, plus the radial reduction used to average
        medium-computed mass weights over the production volume.
    build_solar_source(...)
        Return an existing SolarNeutrinoSource on the requested
        device/dtype, or load the default one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, TypeAlias

import torch

import tpeanuts.config.default as default
from tpeanuts.source.solar.io import (
    available_solar_spectrum_sources,
    load_solar_fluxes,
    load_solar_production,
    load_solar_spectrum,
)
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import interp1d_linear


Tensor = torch.Tensor


@dataclass(frozen=True)
class ContinuousSolarSpectrum:
    """Normalized continuous solar spectrum ``g(E)`` in MeV^-1."""

    energy_MeV: Tensor
    density_MeV_inverse: Tensor

    def __post_init__(self) -> None:
        energy = self.energy_MeV
        density = self.density_MeV_inverse
        if energy.ndim != 1 or energy.numel() < 2 or density.shape != energy.shape:
            raise ValueError("A continuous solar spectrum requires matching 1-D arrays with at least two points.")
        if energy.device != density.device or energy.dtype != density.dtype:
            raise ValueError("Continuous solar-spectrum tensors must share device and dtype.")
        if not torch.isfinite(energy).all() or torch.any(energy < 0) or torch.any(torch.diff(energy) <= 0):
            raise ValueError("Continuous solar-spectrum energies must be finite, non-negative, and strictly increasing.")
        if not torch.isfinite(density).all() or torch.any(density < 0):
            raise ValueError("Continuous solar-spectrum density must be finite and non-negative.")
        normalization = torch.trapezoid(density, x=energy)
        if not torch.isfinite(normalization) or normalization <= 0:
            raise ValueError("Continuous solar spectrum must have a positive finite integral.")
        object.__setattr__(self, "density_MeV_inverse", density / normalization)

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "ContinuousSolarSpectrum":
        return ContinuousSolarSpectrum(
            self.energy_MeV.to(device=device, dtype=dtype),
            self.density_MeV_inverse.to(device=device, dtype=dtype),
        )

    def evaluate(self, E_MeV: Tensor) -> Tensor:
        energy = torch.as_tensor(E_MeV, device=self.energy_MeV.device, dtype=self.energy_MeV.dtype)
        zero = torch.zeros((), device=energy.device, dtype=energy.dtype)
        return interp1d_linear(
            x=energy, xp=self.energy_MeV, fp=self.density_MeV_inverse,
            left=zero, right=zero, device=energy.device, dtype=energy.dtype,
        )


@dataclass(frozen=True)
class SolarLineSpectrum:
    """Discrete solar-neutrino lines with normalized dimensionless weights."""

    energy_MeV: Tensor
    weights: Tensor

    def __post_init__(self) -> None:
        energy = self.energy_MeV
        weights = self.weights
        if energy.ndim != 1 or energy.numel() < 1 or weights.shape != energy.shape:
            raise ValueError("A solar line spectrum requires matching non-empty 1-D arrays.")
        if energy.device != weights.device or energy.dtype != weights.dtype:
            raise ValueError("Solar line-spectrum tensors must share device and dtype.")
        if not torch.isfinite(energy).all() or torch.any(energy <= 0) or (energy.numel() > 1 and torch.any(torch.diff(energy) <= 0)):
            raise ValueError("Solar line energies must be finite, positive, unique, and strictly increasing.")
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("Solar line weights must be finite and non-negative.")
        normalization = weights.sum()
        if not torch.isfinite(normalization) or normalization <= 0:
            raise ValueError("Solar line weights must have a positive finite sum.")
        object.__setattr__(self, "weights", weights / normalization)

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "SolarLineSpectrum":
        return SolarLineSpectrum(
            self.energy_MeV.to(device=device, dtype=dtype),
            self.weights.to(device=device, dtype=dtype),
        )


SolarSpectrum: TypeAlias = ContinuousSolarSpectrum | SolarLineSpectrum


@dataclass(frozen=True)
class SolarSourceParameters:
    """
    Construction settings for the tabulated solar-neutrino source.

    Parameters
    ----------
    production_provider, flux_provider:
        Independent canonical providers for the radial-production and total-
        flux tables. Explicit paths take precedence for *loading*, and also
        for the *recorded* provenance: giving ``production_path``/
        ``fluxes_path`` always records ``"custom"`` as the built source's
        ``production_provider``/``flux_provider``, even if a provider name
        is also given, so the metadata never claims a canonical table that
        was not actually the one loaded (mirroring
        ``medium.solar.profile.SolarMediumParameters``'s ``provider``/
        ``density_path`` resolution). None (with no path override) resolves
        to the configured default provider.

    spectrum_provider:
        Provider for production energy spectra. Defaults to ``"legacy"``
        independently of the production/flux provider.

    spectrum_variants:
        Optional source-to-variant mapping, for example
        ``{"8B": "ortiz", "7Be": "ground"}``.

    production_path:
        Optional radial-production-table override. None loads the configured
        default provider table.

    fluxes_path:
        Optional override path to the per-source total-flux CSV. None loads
        the configured default provider table.
    """

    production_provider: Optional[str] = None
    flux_provider: Optional[str] = None
    spectrum_provider: Optional[str] = default.solar_spectrum_provider
    spectrum_variants: Optional[Mapping[str, str]] = None
    production_path: Optional[str] = None
    fluxes_path: Optional[str] = None
    flux_reference_distance_au: float = 1.0


def build_solar_source(
    solar_source: "SolarNeutrinoSource | None",
    *,
    params: Optional[SolarSourceParameters] = None,
    context: RuntimeContext,
) -> "SolarNeutrinoSource":
    """Return a SolarNeutrinoSource on the requested device and dtype.

    Args:
        solar_source: Existing source or None to load the default source.
        params: Solar source construction settings used when
            ``solar_source`` is None.
        context: Target device/dtype for source tensors.

    Returns:
        SolarNeutrinoSource with production distributions, fluxes, and
        spectra on the target device and dtype.
    """
    if solar_source is None:
        return SolarNeutrinoSource.default(params=params, context=context)

    device, dtype = context.device, context.dtype
    if solar_source.production_radius.device == device and solar_source.production_radius.dtype == dtype:
        return solar_source

    return SolarNeutrinoSource(
        production_radius=solar_source.production_radius.to(device=device, dtype=dtype),
        fractions={
            key: value.to(device=device, dtype=dtype)
            for key, value in solar_source.fractions.items()
        },
        fluxes={
            key: value.to(device=device, dtype=dtype)
            for key, value in solar_source.fluxes.items()
        },
        spectra={
            key: value.to(device=device, dtype=dtype)
            for key, value in solar_source.spectra.items()
        },
        production_provider=solar_source.production_provider,
        flux_provider=solar_source.flux_provider,
        spectrum_provider=solar_source.spectrum_provider,
        spectrum_variants=solar_source.spectrum_variants,
        flux_reference_distance_au=solar_source.flux_reference_distance_au,
        flux_unit=solar_source.flux_unit,
        production_measure=solar_source.production_measure,
    )


@dataclass
class SolarNeutrinoSource:
    """Torch representation of the solar-neutrino production source.

    ``production_radius`` is the independent grid on which ``fractions`` is
    defined; ``medium.solar`` interpolates its own (independently
    validated) density profile at these points, or over the full structural
    grid for the numerical propagation path.

    ``production_measure`` records how ``fractions`` reduce over radius:
    ``"shell_fraction"`` (discrete per-shell weights that already sum to ~1,
    e.g. Bahcall's bp2004_production table) or ``"radial_pdf"`` (a
    continuous density dN/dr requiring trapezoidal integration to reach 1,
    e.g. zenodo SF3 or legacy B16). ``SolarNeutrinoSource.__post_init__``
    normalizes with that measure once; ``medium.solar.probability`` then
    uses the stored distribution without applying a second normalization.
    """

    production_radius: Tensor
    fractions: dict[str, Tensor]
    fluxes: dict[str, Tensor]
    spectra: dict[str, SolarSpectrum] = field(default_factory=dict)
    production_provider: Optional[str] = None
    flux_provider: Optional[str] = None
    spectrum_provider: Optional[str] = None
    spectrum_variants: Mapping[str, str] = field(default_factory=dict)
    flux_reference_distance_au: float = 1.0
    flux_unit: str = "cm^-2 s^-1"
    production_measure: str = "radial_pdf"

    def __post_init__(self) -> None:
        """Validate, sanitize, and normalize every production distribution."""
        if self.production_measure not in {"radial_pdf", "shell_fraction"}:
            raise ValueError(
                "production_measure must be 'radial_pdf' or 'shell_fraction', "
                f"got {self.production_measure!r}."
            )
        if self.production_radius.ndim != 1 or self.production_radius.numel() < 2:
            raise ValueError("SolarNeutrinoSource.production_radius must be one-dimensional with at least two points.")
        if not torch.isfinite(self.production_radius).all() or torch.any(torch.diff(self.production_radius) <= 0):
            raise ValueError("SolarNeutrinoSource.production_radius must be finite and strictly increasing.")
        if torch.any(self.production_radius < 0) or torch.any(self.production_radius > 1):
            raise ValueError("SolarNeutrinoSource.production_radius must lie in [0, 1] R_sun.")
        if not torch.isfinite(torch.as_tensor(self.flux_reference_distance_au)) or self.flux_reference_distance_au <= 0:
            raise ValueError("flux_reference_distance_au must be positive and finite.")

        sanitized: dict[str, Tensor] = {}
        for source, fraction in self.fractions.items():
            fraction = torch.as_tensor(
                fraction, device=self.production_radius.device, dtype=self.production_radius.dtype
            )
            if fraction.shape != self.production_radius.shape:
                raise ValueError(
                    f"Production distribution {source!r} must have shape "
                    f"{tuple(self.production_radius.shape)}, got {tuple(fraction.shape)}."
                )
            if not torch.isfinite(fraction).all():
                raise ValueError(
                    f"Production distribution {source!r} contains non-finite values."
                )
            scale = torch.clamp(fraction.abs().max(), min=1.0)
            tolerance = max(1.0e-10, 32 * torch.finfo(self.dtype).eps) * scale
            if torch.any(fraction < -tolerance):
                raise ValueError(
                    f"Production distribution {source!r} contains a significant "
                    f"negative value: {float(fraction.min()):.6e}."
                )
            fraction = fraction.clamp_min(0.0)
            normalization = (
                fraction.sum()
                if self.production_measure == "shell_fraction"
                else torch.trapezoid(fraction, x=self.production_radius)
            )
            if not torch.isfinite(normalization) or normalization <= 0:
                raise ValueError(
                    f"Production distribution {source!r} has non-positive normalization."
                )
            sanitized[source] = fraction / normalization
        self.fractions = sanitized

        if set(self.fractions) != set(self.fluxes):
            raise ValueError(
                "Solar production/flux source mismatch: "
                f"production={sorted(self.fractions)}; flux={sorted(self.fluxes)}."
            )
        validated_fluxes: dict[str, Tensor] = {}
        for source, flux in self.fluxes.items():
            flux = torch.as_tensor(flux, device=self.device, dtype=self.dtype)
            if flux.ndim != 0 or not torch.isfinite(flux) or flux < 0:
                raise ValueError(f"Solar total flux {source!r} must be a finite non-negative scalar.")
            validated_fluxes[source] = flux
        self.fluxes = validated_fluxes
        if self.flux_unit != "cm^-2 s^-1":
            raise ValueError("Solar total fluxes must use the canonical unit 'cm^-2 s^-1'.")

        unknown_spectra = set(self.spectra) - set(self.fractions)
        if unknown_spectra:
            raise ValueError(f"Solar spectra have no matching production source: {sorted(unknown_spectra)}.")
        self.spectra = {
            key: spectrum.to(device=self.device, dtype=self.dtype)
            for key, spectrum in self.spectra.items()
        }

    @classmethod
    def default(
        cls,
        *,
        params: Optional[SolarSourceParameters] = None,
        context: RuntimeContext,
    ) -> "SolarNeutrinoSource":
        """Load the default solar production, flux, and spectrum tables.

        The default data files are controlled by ``tpeanuts.config.default``:
        currently the Zenodo SF-III AGSS09 radial production distributions
        and total fluxes, together with the independently configured legacy
        production-spectrum provider. Pass an explicit
        ``SolarSourceParameters`` to override them.

        Args:
            params: Solar source construction settings (CSV path overrides).
                None loads the package defaults (see ``config.default``).
            context: Runtime device/dtype used for the loaded tensors.

        Returns:
            SolarNeutrinoSource built from the configured production, flux,
            and production-spectrum files.
        """
        params = params or SolarSourceParameters()
        device, dtype = context.device, context.dtype
        production_provider = "custom" if params.production_path else (params.production_provider or default.solar_provider)
        flux_provider = "custom" if params.fluxes_path else (params.flux_provider or default.solar_provider)
        fluxes = load_solar_fluxes(
            params.fluxes_path, provider=None if params.fluxes_path else flux_provider,
            device=device, dtype=dtype,
        )

        production = load_solar_production(
            params.production_path,
            provider=None if params.production_path else production_provider,
            device=device, dtype=dtype,
        )
        production_radius = production["radius"]
        fractions = production["fractions"]
        production_measure = production.get("production_measure", "radial_pdf")
        spectra: dict[str, SolarSpectrum] = {}
        if params.spectrum_provider is not None:
            variants = dict(params.spectrum_variants or {})
            resolved_variants: dict[str, str] = {}
            for source in available_solar_spectrum_sources(params.spectrum_provider):
                if source not in fractions:
                    continue
                variant = variants.get(source, "default")
                table = load_solar_spectrum(
                    source,
                    provider=params.spectrum_provider,
                    variant=variant,
                    device=device,
                    dtype=dtype,
                )
                if table["kind"] == "line":
                    spectra[source] = SolarLineSpectrum(table["energy"], table["spectrum"])
                else:
                    spectra[source] = ContinuousSolarSpectrum(table["energy"], table["spectrum"])
                resolved_variants[source] = variant
        production_sources = set(fractions)
        flux_sources = set(fluxes)
        if production_sources != flux_sources:
            missing_flux = sorted(production_sources - flux_sources)
            missing_production = sorted(flux_sources - production_sources)
            raise ValueError(
                "Solar production/flux source mismatch: "
                f"production without flux={missing_flux}; "
                f"flux without production={missing_production}."
            )

        return cls(
            production_radius=production_radius,
            fractions=fractions,
            fluxes=fluxes,
            spectra=spectra,
            production_provider=production_provider,
            flux_provider=flux_provider,
            spectrum_provider=params.spectrum_provider,
            spectrum_variants=resolved_variants if params.spectrum_provider is not None else {},
            flux_reference_distance_au=params.flux_reference_distance_au,
            production_measure=production_measure,
        )

    @property
    def device(self) -> torch.device:
        """Return the device used by the source's production-radius tensor."""
        return self.production_radius.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the real dtype used by the source's production-radius tensor."""
        return self.production_radius.dtype

    def production_distribution(
        self,
        sources: str | Sequence[str],
        radius: Tensor | None = None,
    ) -> Tensor:
        """Return normalized production distributions for one or more sources.

        Args:
            sources: Source key or ordered source keys stored in ``fractions``.
            radius: Optional query radii. None returns the native source grid.

        Returns:
            One source has shape ``radius.shape``; multiple sources are stacked
            on a leading source axis in the requested order.

        Raises:
            KeyError: If ``source`` is not present in the source.
        """
        if not isinstance(sources, str):
            return torch.stack(
                [self.production_distribution(source, radius) for source in sources],
                dim=0,
            )
        if sources not in self.fractions:
            raise KeyError(f"Unknown solar source: {sources}")
        distribution = self.fractions[sources]
        if radius is None:
            return distribution
        return interp1d_linear(
            x=radius,
            xp=self.production_radius,
            fp=distribution,
            left=torch.zeros((), device=self.device, dtype=self.dtype),
            right=torch.zeros((), device=self.device, dtype=self.dtype),
            device=self.device,
            dtype=self.dtype,
        )

    def mass_weights_integrate(
        self,
        weights_r: Tensor,
        fractions: Tensor,
        energy_ndim: int,
    ) -> Tensor:
        """Reduce per-radius mass-basis weights against a production distribution.

        Used identically by the adiabatic (``medium.solar.adiabatic``) and
        numerical (``medium.solar.evolutor.mass_weights_numerical``)
        mass-weight computations dispatched from
        ``medium.solar.probability.solar_probability_mass``, which all
        produce ``weights_r`` on this source's ``production_radius`` grid
        in the same ``(..., n_r, N)`` shape convention.

        ``self.production_measure`` selects how ``fractions`` reduce over
        radius (see the class docstring):

            "shell_fraction" -- fractions are already discrete per-shell
                weights that sum (plain sum, not integrated) to ~1 over the
                table's own -- possibly non-uniform -- shells (e.g.
                Bahcall's bp2004_production table). Reduced with a plain
                weighted sum: each tabulated fraction already carries its
                own shell's share of production, so re-weighting by local
                radius spacing (as trapezoidal integration would)
                double-counts the (non-uniform) grid spacing and biases the
                result toward whichever shells happen to be more finely
                sampled.
            "radial_pdf" -- fractions are a continuous production density
                dN/dr (e.g. zenodo SF3 or legacy B16), reduced with
                trapezoidal integration over ``self.production_radius``.

        Args:
            weights_r: Per-radius mass-basis weights, shape ``(..., n_r, N)``.
            fractions: Production distribution(s), shape ``(..., n_r)`` --
                as returned by ``production_distribution``.
            energy_ndim: Number of leading energy dimensions folded into
                ``weights_r`` between ``fractions``'s source dimensions and
                its radius axis.

        Returns:
            Weights reduced over radius, with the radius axis removed.
        """
        source_shape = fractions.shape[:-1]

        fractions_lifted = fractions.reshape(
            *source_shape,
            *((1,) * energy_ndim),
            fractions.shape[-1],
        )

        weights_lifted = weights_r.reshape(
            *((1,) * len(source_shape)),
            *weights_r.shape,
        )

        weighted = weights_lifted * fractions_lifted[..., None]

        if self.production_measure == "shell_fraction":
            return weighted.sum(dim=-2)
        return torch.trapz(weighted, x=self.production_radius, dim=-2)

    def total_flux(self, source: str) -> Tensor:
        """Return the total flux for one solar source.

        Args:
            source: Solar source key stored in ``self.fluxes``.

        Returns:
            Total source flux tensor.

        Raises:
            KeyError: If ``source`` is not present in the flux table.
        """
        if source not in self.fluxes:
            raise KeyError(f"Unknown solar flux source: {source}")

        return self.fluxes[source]

    def has_spectrum(self, source: str) -> bool:
        """Return whether an energy spectrum is available for ``source``."""
        return source in self.spectra

    def spectrum_table(self, source: str) -> SolarSpectrum:
        """Return the native tabulated spectrum for one source."""
        if source not in self.spectra:
            raise KeyError(
                f"No production spectrum for solar source {source!r} in "
                f"spectrum provider {self.spectrum_provider!r}."
            )
        return self.spectra[source]

    def spectrum(
        self,
        sources: str | Sequence[str],
        E_MeV: Tensor,
    ) -> Tensor:
        """Interpolate normalized source spectra onto an energy grid.

        Values outside each tabulated support are zero. Multiple sources are
        stacked on a leading source axis in the requested order.
        """
        if not isinstance(sources, str):
            return torch.stack([self.spectrum(source, E_MeV) for source in sources])
        table = self.spectrum_table(sources)
        if isinstance(table, SolarLineSpectrum):
            raise TypeError(
                f"Solar source {sources!r} is a discrete line spectrum; it cannot be interpolated as dPhi/dE. "
                "Evaluate probabilities/rates at spectrum_table(source).energy_MeV and sum with its weights."
            )
        return table.evaluate(E_MeV)

    def __str__(self) -> str:
        """Return a compact summary of the solar source configuration."""
        n_r = self.production_radius.numel()
        r0 = float(self.production_radius[0])
        r1 = float(self.production_radius[-1])
        sources = ", ".join(sorted(self.fractions.keys()))
        spectrum_sources = ", ".join(sorted(self.spectra))
        return (
            f"SolarNeutrinoSource | "
            f"n_r={n_r} | "
            f"r=[{r0:.3f}, {r1:.3f}] R_sun | "
            f"sources=[{sources}] | "
            f"providers=production:{self.production_provider},flux:{self.flux_provider},spectrum:{self.spectrum_provider} | "
            f"spectra=[{spectrum_sources}] | ref={self.flux_reference_distance_au:g} AU | "
            f"{self.device} / {self.dtype}"
        )

    __repr__ = __str__
