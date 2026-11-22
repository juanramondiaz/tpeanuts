#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Solar-neutrino source: production, total flux, and production spectra.

``SolarSource`` describes what the Sun produces, where, and with
what energy -- independent of how those neutrinos then propagate through
the solar medium (a future ``tpeanuts.medium.solar``, not ported yet).
This split mirrors ``tpeanuts.source.atmosphere``/
``tpeanuts.medium.atmosphere`` (and ``tpeanuts.source.reactor.reactor``/
``tpeanuts.medium.vacuum.vacuum``): production physics lives here,
propagation physics will live in ``medium.solar``, which will depend on
this module (passing a built ``SolarSource`` alongside its own
medium profile) -- this module never depends back on ``medium.solar``.

Ported from v1_0's ``source/solar/model.py``, unchanged in behaviour;
only the data-loading calls into ``tpeanuts.source.solar.io`` follow this
rewrite's own conventions (``RuntimeContext``, no ``pandas``), and the
linear interpolation this module needs uses
``tpeanuts.util.math.interp1d_linear`` (promoted from a private copy
here once ``tpeanuts.source.reactor.reactor``/``tpeanuts.experiments.dayabay``
needed the same "zero/constant outside the tabulated support"
interpolation independently).

Module contents:
    SolarContinuousSpectrum
        Normalized continuous production spectrum ``g(E)`` in MeV^-1.
    SolarLineSpectrum
        Normalized discrete production-line spectrum.
    SolarSpectrum
        Type alias: ``SolarContinuousSpectrum | SolarLineSpectrum``.
    SolarSourceParameters
        Construction settings for the tabulated solar source (production,
        flux, and spectrum tables).
    SolarSource
        Torch container for production distributions, total fluxes, and
        production spectra, plus the radial reduction used to average
        medium-computed mass weights over the production volume, and
        ``flux()``, which presents the chosen sources as a ``SourceFlux``.
    build_solar_source(...)
        Return an existing SolarSource on the requested
        device/dtype, or load the default one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union, cast

import torch

import tpeanuts.config.default as default
from tpeanuts.interfaces.flux import Flux, ProductionDistribution, SourceFlux, integrate_radial
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.source.solar.io import (
    available_solar_spectrum_sources,
    load_solar_flux,
    load_solar_production,
    load_solar_spectrum,
)
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import interp1d_linear
from tpeanuts.util.tensor import as_tensor

Tensor = torch.Tensor


@dataclass(frozen=True)
class SolarContinuousSpectrum:
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
            raise ValueError(
                "Continuous solar-spectrum energies must be finite, non-negative, and strictly increasing."
            )
        if not torch.isfinite(density).all() or torch.any(density < 0):
            raise ValueError("Continuous solar-spectrum density must be finite and non-negative.")
        normalization = torch.trapezoid(density, x=energy)
        if not torch.isfinite(normalization) or normalization <= 0:
            raise ValueError("Continuous solar spectrum must have a positive finite integral.")
        object.__setattr__(self, "density_MeV_inverse", density / normalization)

    def to(self, *, context: RuntimeContext) -> "SolarContinuousSpectrum":
        """Return a copy with both tensors moved to the device and dtype of ``context``."""
        return SolarContinuousSpectrum(
            self.energy_MeV.to(device=context.device, dtype=context.dtype),
            self.density_MeV_inverse.to(device=context.device, dtype=context.dtype),
        )

    def evaluate(self, E_MeV: Tensor) -> Tensor:
        """Evaluate ``g(E)`` in MeV^-1 at ``E_MeV`` by linear interpolation (zero outside the tabulated range)."""
        energy = as_tensor(E_MeV, device=self.energy_MeV.device, dtype=self.energy_MeV.dtype)
        zero = torch.zeros((), device=energy.device, dtype=energy.dtype)
        return interp1d_linear(energy, self.energy_MeV, self.density_MeV_inverse, left=zero, right=zero)


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
        if (
            not torch.isfinite(energy).all()
            or torch.any(energy <= 0)
            or (energy.numel() > 1 and torch.any(torch.diff(energy) <= 0))
        ):
            raise ValueError("Solar line energies must be finite, positive, unique, and strictly increasing.")
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("Solar line weights must be finite and non-negative.")
        normalization = weights.sum()
        if not torch.isfinite(normalization) or normalization <= 0:
            raise ValueError("Solar line weights must have a positive finite sum.")
        object.__setattr__(self, "weights", weights / normalization)

    def to(self, *, context: RuntimeContext) -> "SolarLineSpectrum":
        """Return a copy with both tensors moved to the device and dtype of ``context``."""
        return SolarLineSpectrum(
            self.energy_MeV.to(device=context.device, dtype=context.dtype),
            self.weights.to(device=context.device, dtype=context.dtype),
        )


SolarSpectrum = Union[SolarContinuousSpectrum, SolarLineSpectrum]


@dataclass(frozen=True)
class SolarSourceParameters:
    """Construction settings for the tabulated solar-neutrino source.

    Attributes:
        production_provider, flux_provider: Independent canonical
            providers for the radial-production and total-flux tables.
            Explicit paths take precedence for *loading*, and also for
            the *recorded* provenance: giving ``production_path``/
            ``fluxes_path`` always records ``"custom"`` as the built
            source's ``production_provider``/``flux_provider``, even if a
            provider name is also given, so the metadata never claims a
            canonical table that was not actually the one loaded. None
            (with no path override) resolves to the configured default
            provider (``tpeanuts.config.default.SOLAR_PROVIDER``).
        spectrum_provider: Provider for production energy spectra.
            Defaults to ``tpeanuts.config.default.SOLAR_SPECTRUM_PROVIDER``
            independently of the production/flux provider.
        spectrum_variants: Optional source-to-variant mapping, e.g.
            ``{"8B": "ortiz", "7Be": "ground"}``.
        production_path: Optional radial-production-table override. None
            loads the configured default provider table.
        fluxes_path: Optional override path to the per-source total-flux
            CSV. None loads the configured default provider table.
        flux_reference_distance_au: Distance, in AU, the tabulated total
            fluxes are normalized at.
    """

    production_provider: Optional[str] = None
    flux_provider: Optional[str] = None
    spectrum_provider: Optional[str] = default.SOLAR_SPECTRUM_PROVIDER
    spectrum_variants: Optional[Mapping[str, str]] = None
    production_path: Optional[Union[str, Path]] = None
    fluxes_path: Optional[Union[str, Path]] = None
    flux_reference_distance_au: float = 1.0


def build_solar_source(
    solar_source: Optional["SolarSource"],
    *,
    params: Optional[SolarSourceParameters] = None,
    context: RuntimeContext,
) -> "SolarSource":
    """Return a SolarSource on the requested device and dtype.

    Args:
        solar_source: Existing source or None to load the default source.
        params: Solar source construction settings used when
            ``solar_source`` is None.
        context: Target device/dtype for source tensors.

    Returns:
        SolarSource with production distributions, fluxes, and
        spectra on the target device and dtype.
    """
    if solar_source is None:
        return SolarSource.default(params=params, context=context)

    if solar_source.device == context.device and solar_source.dtype == context.dtype:
        return solar_source

    return SolarSource(
        production_radius=solar_source.production_radius.to(device=context.device, dtype=context.dtype),
        fractions={
            key: value.to(device=context.device, dtype=context.dtype)
            for key, value in solar_source.fractions.items()
        },
        fluxes={
            key: value.to(device=context.device, dtype=context.dtype)
            for key, value in solar_source.fluxes.items()
        },
        context=context,
        spectra={
            key: spectrum.to(context=context)
            for key, spectrum in solar_source.spectra.items()
        },
        production_provider=solar_source.production_provider,
        flux_provider=solar_source.flux_provider,
        spectrum_provider=solar_source.spectrum_provider,
        spectrum_variants=solar_source.spectrum_variants,
        flux_reference_distance_au=solar_source.flux_reference_distance_au,
        flux_unit=solar_source.flux_unit,
        production_measure=solar_source.production_measure,
    )


@dataclass(frozen=True)
class SolarSource:
    """Torch representation of the solar-neutrino production source.

    ``production_radius`` is the independent grid on which ``fractions``
    is defined; a future ``medium.solar`` will interpolate its own
    (independently validated) density profile at these points, or over
    the full structural grid for a numerical propagation path.

    ``production_measure`` records how ``fractions`` reduce over radius:
    ``"shell_fraction"`` (discrete per-shell weights that already sum to
    ~1, e.g. Bahcall's bp2004_production table) or ``"radial_pdf"`` (a
    continuous density dN/dr requiring trapezoidal integration to reach
    1, e.g. zenodo SF3 or legacy B16). ``__post_init__`` normalizes with
    that measure once; downstream consumers use the stored distribution
    without applying a second normalization.

    Attributes:
        production_radius: Radial grid on which the production distributions are defined, in units of the solar
            radius, shape ``(n_r,)``, finite, strictly increasing within ``[0, 1]``, with at least two points.
        fractions: ``{source: distribution}``, the production distribution of each channel on ``production_radius``
            (shape ``(n_r,)``). Normalised on construction according to ``production_measure``.
        fluxes: ``{source: total flux}``, scalar total flux of each channel in ``flux_unit`` at
            ``flux_reference_distance_au``. Must name exactly the channels of ``fractions``.
        context: RuntimeContext every tensor here is built on (the source is converted to it on construction).
        spectra: ``{source: spectrum}`` production-energy spectra, ``SolarContinuousSpectrum`` or ``SolarLineSpectrum``;
            may be a subset of the channels, but none can be unknown to ``fractions``.
        production_provider: Provenance of the production distributions (``"zenodo"``, ``"bahcall"``, ``"legacy"``
            or ``"custom"``); no physics.
        flux_provider: Provenance of the total fluxes.
        spectrum_provider: Provenance of the spectra.
        spectrum_variants: ``{source: variant}`` of the spectrum table each channel was loaded from.
        flux_reference_distance_au: Distance from the Sun, in AU, at which ``fluxes`` are given (positive, finite).
        flux_unit: Unit of ``fluxes``; only ``"cm^-2 s^-1"`` is accepted.
        production_measure: ``"radial_pdf"`` or ``"shell_fraction"``, see above.

    Raises:
        ValueError: If ``production_measure`` is unknown; ``production_radius`` is not a finite, strictly
            increasing grid of at least two points within ``[0, 1]``; ``flux_reference_distance_au`` is not positive;
            a distribution has the wrong shape, a non-finite or significantly negative value or no positive
            normalisation; ``fractions`` and ``fluxes`` name different channels; a total flux is not a finite
            non-negative scalar; ``flux_unit`` is not canonical; or a spectrum has no production channel.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.source import SolarSource
        >>> source = SolarSource.default(context=context)   # the packaged production profiles and fluxes
        >>> source.has_spectrum("8B"), f"{float(source.total_flux('8B')):.3e}"   # flux in the units of the flux table
        (True, '4.135e+06')
    """

    production_radius: Tensor
    fractions: dict[str, Tensor]
    fluxes: dict[str, Tensor]
    context: RuntimeContext
    spectra: dict[str, SolarSpectrum] = field(default_factory=dict)
    production_provider: Optional[str] = None
    flux_provider: Optional[str] = None
    spectrum_provider: Optional[str] = None
    spectrum_variants: Mapping[str, str] = field(default_factory=dict)
    flux_reference_distance_au: float = 1.0
    flux_unit: str = "cm^-2 s^-1"
    production_measure: str = "radial_pdf"

    def __post_init__(self) -> None:
        """Validate, sanitize, and normalize every production distribution, on ``context``'s device and dtype."""
        object.__setattr__(
            self, "production_radius",
            as_tensor(self.production_radius, device=self.context.device, dtype=self.context.dtype),
        )
        if self.production_measure not in {"radial_pdf", "shell_fraction"}:
            raise ValueError(
                "production_measure must be 'radial_pdf' or 'shell_fraction', "
                f"got {self.production_measure!r}."
            )
        if self.production_radius.ndim != 1 or self.production_radius.numel() < 2:
            raise ValueError("SolarSource.production_radius must be one-dimensional with at least two points.")
        if not torch.isfinite(self.production_radius).all() or torch.any(torch.diff(self.production_radius) <= 0):
            raise ValueError("SolarSource.production_radius must be finite and strictly increasing.")
        if torch.any(self.production_radius < 0) or torch.any(self.production_radius > 1):
            raise ValueError("SolarSource.production_radius must lie in [0, 1] R_sun.")
        if (
            not torch.isfinite(torch.as_tensor(float(self.flux_reference_distance_au)))
            or self.flux_reference_distance_au <= 0
        ):
            raise ValueError("flux_reference_distance_au must be positive and finite.")

        sanitized: dict[str, Tensor] = {}
        for source, fraction in self.fractions.items():
            fraction = as_tensor(fraction, device=self.production_radius.device, dtype=self.production_radius.dtype)
            if fraction.shape != self.production_radius.shape:
                raise ValueError(
                    f"Production distribution {source!r} must have shape "
                    f"{tuple(self.production_radius.shape)}, got {tuple(fraction.shape)}."
                )
            if not torch.isfinite(fraction).all():
                raise ValueError(f"Production distribution {source!r} contains non-finite values.")
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
                raise ValueError(f"Production distribution {source!r} has non-positive normalization.")
            sanitized[source] = fraction / normalization
        object.__setattr__(self, "fractions", sanitized)

        if set(self.fractions) != set(self.fluxes):
            raise ValueError(
                "Solar production/flux source mismatch: "
                f"production={sorted(self.fractions)}; flux={sorted(self.fluxes)}."
            )
        validated_fluxes: dict[str, Tensor] = {}
        for source, flux in self.fluxes.items():
            flux = as_tensor(flux, device=self.device, dtype=self.dtype)
            if flux.ndim != 0 or not torch.isfinite(flux) or flux < 0:
                raise ValueError(f"Solar total flux {source!r} must be a finite non-negative scalar.")
            validated_fluxes[source] = flux
        object.__setattr__(self, "fluxes", validated_fluxes)
        if self.flux_unit != "cm^-2 s^-1":
            raise ValueError("Solar total fluxes must use the canonical unit 'cm^-2 s^-1'.")

        unknown_spectra = set(self.spectra) - set(self.fractions)
        if unknown_spectra:
            raise ValueError(f"Solar spectra have no matching production source: {sorted(unknown_spectra)}.")
        spectra = {key: spectrum.to(context=self.context) for key, spectrum in self.spectra.items()}
        object.__setattr__(self, "spectra", spectra)

    @classmethod
    def default(
        cls,
        *,
        params: Optional[SolarSourceParameters] = None,
        context: RuntimeContext,
    ) -> "SolarSource":
        """Load the default solar production, flux, and spectrum tables.

        The default data files are controlled by
        ``tpeanuts.config.default`` -- currently the Zenodo SF-III AGSS09
        radial production distributions and total fluxes, together with
        the independently configured legacy production-spectrum
        provider. Pass an explicit ``SolarSourceParameters`` to override
        them.

        Args:
            params: Solar source construction settings (CSV path
                overrides). None loads the package defaults (see
                ``config.default``).
            context: Runtime device/dtype used for the loaded tensors.

        Returns:
            SolarSource built from the configured production,
            flux, and production-spectrum files.

        Raises:
            ValueError: If the production and flux tables disagree on
                which sources are present.
        """
        params = params or SolarSourceParameters()
        production_provider = (
            "custom" if params.production_path else (params.production_provider or default.SOLAR_PROVIDER)
        )
        flux_provider = "custom" if params.fluxes_path else (params.flux_provider or default.SOLAR_PROVIDER)
        fluxes = load_solar_flux(
            params.fluxes_path, provider=None if params.fluxes_path else flux_provider, context=context,
        )

        production = load_solar_production(
            params.production_path,
            provider=None if params.production_path else production_provider,
            context=context,
        )
        production_radius = cast(Tensor, production["radius"])
        fractions = cast(dict[str, Tensor], production["fractions"])
        production_measure = cast(str, production.get("production_measure", "radial_pdf"))
        spectra: dict[str, SolarSpectrum] = {}
        resolved_variants: dict[str, str] = {}
        if params.spectrum_provider is not None:
            variants = dict(params.spectrum_variants or {})
            for source in available_solar_spectrum_sources(params.spectrum_provider):
                if source not in fractions:
                    continue
                variant = variants.get(source, "default")
                table = load_solar_spectrum(
                    source, provider=params.spectrum_provider, variant=variant, context=context,
                )
                energy, spectrum = cast(Tensor, table["energy"]), cast(Tensor, table["spectrum"])
                if table["kind"] == "line":
                    spectra[source] = SolarLineSpectrum(energy, spectrum)
                else:
                    spectra[source] = SolarContinuousSpectrum(energy, spectrum)
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
            context=context,
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
        """Return the device of this source's tensors (``context.device``)."""
        return self.context.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the real dtype of this source's tensors (``context.dtype``)."""
        return self.context.dtype

    def production_distribution(
        self,
        sources: Union[str, Sequence[str]],
        radius: Optional[Tensor] = None,
    ) -> Tensor:
        """Return normalized production distributions for one or more sources.

        Args:
            sources: Source key or ordered source keys stored in ``fractions``.
            radius: Optional query radii. None returns the native source grid.

        Returns:
            One source has shape ``radius.shape``; multiple sources are
            stacked on a leading source axis in the requested order.

        Raises:
            KeyError: If ``sources`` names a source not present in the source.
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
        radius = as_tensor(radius, device=self.device, dtype=self.dtype)
        zero = torch.zeros((), device=self.device, dtype=self.dtype)
        return interp1d_linear(radius, self.production_radius, distribution, left=zero, right=zero)

    def mass_weights_integrate(
        self,
        weights_r: Tensor,
        fractions: Tensor,
        energy_ndim: int,
    ) -> Tensor:
        """Reduce per-radius mass-basis weights against a production distribution.

        Intended to be used identically by a future ``medium.solar``'s
        adiabatic and numerical mass-weight computations, which will
        produce ``weights_r`` on this source's ``production_radius`` grid
        in the same ``(..., n_r, N)`` shape convention -- ported now,
        unused for now, since it is a pure reduction method of this class
        with no propagation dependency of its own (see this module's own
        docstring).

        ``self.production_measure`` selects how ``fractions`` reduce over
        radius (see the class docstring):

            "shell_fraction" -- fractions are already discrete per-shell
                weights that sum (plain sum, not integrated) to ~1 over
                the table's own -- possibly non-uniform -- shells (e.g.
                Bahcall's bp2004_production table). Reduced with a plain
                weighted sum: each tabulated fraction already carries its
                own shell's share of production, so re-weighting by local
                radius spacing (as trapezoidal integration would)
                double-counts the (non-uniform) grid spacing and biases
                the result toward whichever shells happen to be more
                finely sampled.
            "radial_pdf" -- fractions are a continuous production density
                dN/dr (e.g. zenodo SF3 or legacy B16), reduced with
                trapezoidal integration over ``self.production_radius``.

        Args:
            weights_r: Per-radius mass-basis weights, shape ``(..., n_r, N)``.
            fractions: Production distribution(s), shape ``(..., n_r)`` --
                as returned by ``production_distribution``.
            energy_ndim: Number of leading energy dimensions folded into
                ``weights_r`` between ``fractions``'s source dimensions
                and its radius axis.

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
        return integrate_radial(weighted, self.production_radius, self.production_measure)

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
        sources: Union[str, Sequence[str]],
        E_MeV: Tensor,
    ) -> Tensor:
        """Interpolate normalized source spectra onto an energy grid.

        Values outside each tabulated support are zero. Multiple sources
        are stacked on a leading source axis in the requested order.

        Raises:
            TypeError: If ``sources`` (or any source in it) is a discrete
                line spectrum, which cannot be interpolated as dPhi/dE.
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

    def flux(
        self,
        sources: Union[str, Sequence[str]],
        energy_MeV: Optional[Tensor] = None,
        *,
        intensity: bool = True,
    ) -> SourceFlux:
        """Present the chosen production channels as a ``SourceFlux``: one ``Flux`` per channel.

        Every neutrino is born as an electron neutrino, so each component is the pure electron-flavour state; what
        distinguishes the components is their intensity and where in the Sun they are produced.

        * A **continuous** channel is sampled at ``energy_MeV``: ``differential_flux`` is the total flux times the
          normalised spectrum ``g(E)``, per MeV. If ``energy_MeV`` is omitted the tabulated grid is used, without its
          zero-energy point if it has one.
        * A **line** channel (``7Be``, ``pep``) lives at its own energies, which replace ``energy_MeV``: its
          ``differential_flux`` is the flux carried by each line (no "per MeV").

        Each component also carries its ``production`` distribution in the Sun, as the source stores it.

        With ``intensity=False`` the components carry no intensity at all: only the energy grid, exactly as given
        (for a line channel too), and the production distribution. That is what the survival probability at chosen
        energies needs, and it asks nothing of the flux and spectrum tables.

        Args:
            sources: Source key or ordered source keys, each with a production distribution (and, unless
                ``intensity=False``, a total flux and a spectrum).
            energy_MeV: Energy grid in MeV: for the continuous channels, and for every channel if
                ``intensity=False`` (then required).
            intensity: Fill ``differential_flux`` (default), or leave it out.

        Returns:
            SourceFlux with one component per key, in the requested order, all at ``location="source"``.

        Raises:
            KeyError: If a key is missing from the production distributions, or (with intensity) the flux table or
                the spectra.
            ValueError: If ``energy_MeV`` is omitted and a channel needs it but has no positive tabulated energy, or
                is omitted with ``intensity=False``.
        """
        keys = [sources] if isinstance(sources, str) else list(sources)
        context = self.context
        state = DensityMatrix.basis_state(0, FlavourSpace(), context)
        if not intensity and energy_MeV is None:
            raise ValueError("intensity=False needs energy_MeV: the flux has no spectrum to take a grid from.")
        components = {}
        for key in keys:
            if key not in self.fractions:
                raise KeyError(f"Unknown solar source: {key}")
            production = ProductionDistribution(
                radius=self.production_radius, weight=self.fractions[key], context=context,
                measure=self.production_measure,
            )
            if not intensity:
                energy = as_tensor(energy_MeV, device=self.device, dtype=self.dtype)
                components[key] = Flux(energy_MeV=energy, state=state, antinu=False, production=production)
                continue
            table = self.spectrum_table(key)
            total = self.total_flux(key)
            if isinstance(table, SolarLineSpectrum):
                energy, intensity_value, units = table.energy_MeV, total * table.weights, self.flux_unit
            else:
                if energy_MeV is None:
                    energy = table.energy_MeV[table.energy_MeV > 0]
                else:
                    energy = as_tensor(energy_MeV, device=self.device, dtype=self.dtype)
                intensity_value, units = total * table.evaluate(energy), f"{self.flux_unit} MeV^-1"
            components[key] = Flux(
                energy_MeV=energy, state=state, antinu=False, differential_flux=intensity_value, flux_units=units,
                production=production,
            )
        return SourceFlux(components)

    def __str__(self) -> str:
        """Return a compact summary of the solar source configuration."""
        n_r = self.production_radius.numel()
        r0 = float(self.production_radius[0])
        r1 = float(self.production_radius[-1])
        sources = ", ".join(sorted(self.fractions.keys()))
        spectrum_sources = ", ".join(sorted(self.spectra))
        return (
            f"SolarSource | "
            f"n_r={n_r} | "
            f"r=[{r0:.3f}, {r1:.3f}] R_sun | "
            f"sources=[{sources}] | "
            f"providers=production:{self.production_provider},flux:{self.flux_provider},"
            f"spectrum:{self.spectrum_provider} | "
            f"spectra=[{spectrum_sources}] | ref={self.flux_reference_distance_au:g} AU | "
            f"{self.device} / {self.dtype}"
        )

    __repr__ = __str__
