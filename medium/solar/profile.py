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
Solar profile containers and interpolation utilities.

This module defines the torch-native SolarProfile container used by the solar
probability and integration layers. It stores tabulated radius, electron
density, production fractions, total source fluxes, and production spectra,
and provides small
helpers for interpolation and normalization on the profile grid.

Module functions:
    build_solar_profile(...)
        Return an existing SolarProfile on the requested device/dtype or load
        the default profile.
    SolarProfile.default(...)
        Load the configured default solar model, flux, and composition
        tables.
    SolarProfile.device
        Return the torch device used by the profile tensors.
    SolarProfile.dtype
        Return the real dtype used by the profile tensors.
    SolarProfile.electron_density(...)
        Interpolate electron density at requested solar radii.
    SolarProfile.neutron_density(...)
        Interpolate neutron density (3+1 sterile neutral-current term) at
        requested solar radii.
    SolarProfile.production_distribution(...)
        Return normalized production distributions for one or more sources,
        optionally interpolated to another radial grid.
    SolarProfile.mass_weights_integrate(...)
        Reduce per-radius mass-basis weights against a production
        distribution, over the production radius grid.
    SolarProfile.flux(...)
        Return the total flux for one solar source.
"""



from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import torch

from tpeanuts.medium.solar.io import (
    load_solar_composition,
    load_solar_density,
    load_solar_fluxes,
    load_solar_production,
    available_solar_spectrum_sources,
    load_solar_spectrum,
)
import tpeanuts.config.default as default
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import interp1d_linear


Tensor = torch.Tensor


@dataclass(frozen=True)
class SolarSpectrum:
    """Tabulated normalized production spectrum for one solar source."""

    energy_MeV: Tensor
    density_MeV_inverse: Tensor

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "SolarSpectrum":
        """Return the spectrum on a requested torch runtime."""
        return SolarSpectrum(
            energy_MeV=self.energy_MeV.to(device=device, dtype=dtype),
            density_MeV_inverse=self.density_MeV_inverse.to(device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class SolarParameters:
    """
    Construction settings for the tabulated solar model.

    Parameters
    ----------
    provider:
        Optional canonical provider name (``"zenodo"``, ``"bahcall"`` or
        ``"legacy"``). Explicit table paths take precedence product by
        product. None uses the configured default filenames.

    density_path:
        Optional density-table override. None loads the selected provider's
        electron/neutron-density table.

    spectrum_provider:
        Provider for production energy spectra. Defaults to ``"legacy"``
        independently of the structural/flux provider.

    spectrum_variants:
        Optional source-to-variant mapping, for example
        ``{"8B": "ortiz", "7Be": "ground"}``.

    production_path:
        Optional radial-production-table override. None loads the configured
        Bahcall production table.

    fluxes_path:
        Optional override path to the per-source total-flux CSV
        None loads the configured Bahcall BP2004 flux table.

    composition_path:
        Optional override path to the solar structure+composition table
        (``struct+nu_SF3_*.dat`` format) used to derive the neutron-density
        profile for the 3+1 sterile neutral-current term
        (``medium.solar.io.load_solar_composition``). None loads the default
        composition table. It is only needed as a fallback when the selected
        density table has no neutron-density column.

    """

    provider: Optional[str] = None
    spectrum_provider: Optional[str] = default.solar_spectrum_provider
    spectrum_variants: Optional[Mapping[str, str]] = None
    density_path: Optional[str] = None
    production_path: Optional[str] = None
    fluxes_path: Optional[str] = None
    composition_path: Optional[str] = None


def build_solar_profile(
    solar_profile: "SolarProfile | None",
    *,
    params: Optional[SolarParameters] = None,
    context: RuntimeContext,
) -> "SolarProfile":
    """Return a SolarProfile on the requested device and dtype.

    Args:
        solar_profile: Existing profile or None to load the default profile.
        params: Solar model construction settings used when ``solar_profile``
            is None.
        context: Target device/dtype for profile tensors.

    Returns:
        SolarProfile with radius, density, fractions, fluxes, and spectra on the
        target device and dtype.
    """
    if solar_profile is None:
        return SolarProfile.default(params=params, context=context)

    device, dtype = context.device, context.dtype
    if solar_profile.radius.device == device and solar_profile.radius.dtype == dtype:
        return solar_profile

    return SolarProfile(
        radius=solar_profile.radius.to(device=device, dtype=dtype),
        production_radius=solar_profile.production_radius.to(device=device, dtype=dtype),
        density=solar_profile.density.to(device=device, dtype=dtype),
        fractions={
            key: value.to(device=device, dtype=dtype)
            for key, value in solar_profile.fractions.items()
        },
        fluxes={
            key: value.to(device=device, dtype=dtype)
            for key, value in solar_profile.fluxes.items()
        },
        spectra={
            key: value.to(device=device, dtype=dtype)
            for key, value in solar_profile.spectra.items()
        },
        spectrum_provider=solar_profile.spectrum_provider,
        use_LZ=solar_profile.use_LZ,
        production_measure=solar_profile.production_measure,
        density_n=(
            None if solar_profile.density_n is None
            else solar_profile.density_n.to(device=device, dtype=dtype)
        ),
    )


@dataclass
class SolarProfile:
    """Torch representation of a solar model.

    ``radius``/``density``/``density_n`` store the full structural density
    grid, normally extending to r = 1 R_sun. ``production_radius`` is the
    independent grid on which ``fractions`` is defined. The adiabatic path
    interpolates the full densities at those production points; the numerical
    path propagates over the full structural grid.

    ``production_measure`` records how ``fractions`` reduce over radius:
    ``"shell_fraction"`` (discrete per-shell weights that already sum to ~1,
    e.g. Bahcall's bp2004_production table) or ``"radial_pdf"`` (a
    continuous density dN/dr requiring trapezoidal integration to reach 1,
    e.g. zenodo SF3 or legacy B16). ``SolarProfile.__post_init__`` normalizes
    with that measure once; ``medium.solar.probability`` then uses the stored
    distribution without applying a second normalization.
    """

    radius: Tensor
    density: Tensor
    production_radius: Tensor
    fractions: dict[str, Tensor]
    fluxes: dict[str, Tensor]
    spectra: dict[str, SolarSpectrum] = field(default_factory=dict)
    spectrum_provider: Optional[str] = None
    use_LZ: bool = False
    production_measure: str = "radial_pdf"
    density_n: Optional[Tensor] = None

    def __post_init__(self) -> None:
        """Validate, sanitize, and normalize every production distribution."""
        if self.production_measure not in {"radial_pdf", "shell_fraction"}:
            raise ValueError(
                "production_measure must be 'radial_pdf' or 'shell_fraction', "
                f"got {self.production_measure!r}."
            )
        if self.radius.ndim != 1 or self.radius.numel() < 2:
            raise ValueError("SolarProfile.radius must be one-dimensional with at least two points.")
        if not torch.isfinite(self.radius).all() or torch.any(torch.diff(self.radius) <= 0):
            raise ValueError("SolarProfile.radius must be finite and strictly increasing.")
        if self.density.shape != self.radius.shape:
            raise ValueError("SolarProfile.density must have the same shape as radius.")
        if not torch.isfinite(self.density).all() or torch.any(self.density < 0):
            raise ValueError("SolarProfile.density must be finite and non-negative.")
        if self.density_n is not None:
            if self.density_n.shape != self.radius.shape:
                raise ValueError("SolarProfile.density_n must have the same shape as radius.")
            if not torch.isfinite(self.density_n).all() or torch.any(self.density_n < 0):
                raise ValueError("SolarProfile.density_n must be finite and non-negative.")

        if self.production_radius.ndim != 1 or self.production_radius.numel() < 2:
            raise ValueError("SolarProfile.production_radius must be one-dimensional with at least two points.")
        if not torch.isfinite(self.production_radius).all() or torch.any(torch.diff(self.production_radius) <= 0):
            raise ValueError("SolarProfile.production_radius must be finite and strictly increasing.")
        if self.production_radius[0] < self.radius[0] or self.production_radius[-1] > self.radius[-1]:
            raise ValueError("SolarProfile.production_radius must lie within the structural radius grid.")

        sanitized: dict[str, Tensor] = {}
        for source, fraction in self.fractions.items():
            fraction = torch.as_tensor(
                fraction, device=self.radius.device, dtype=self.radius.dtype
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
            tolerance = 1.0e-10 * scale
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

    @classmethod
    def default(
        cls,
        *,
        params: Optional[SolarParameters] = None,
        context: RuntimeContext,
    ) -> "SolarProfile":
        """Load the default solar model, flux, and composition tables.

        The default data files are controlled by ``tpeanuts.config.default``:
        currently the Zenodo SF-III AGSS09 density profile, radial production
        distributions, and total fluxes, together with the independently
        configured legacy production-spectrum provider. Pass an
        explicit ``SolarParameters`` to override them.

        ``density_n`` (needed for the 3+1 sterile neutral-current term) is
        loaded directly when the density table contains it. For legacy or
        electron-only tables, an explicitly selected structure/composition
        table can provide the interpolated ratio ``n_n/n_e`` as a fallback.

        Args:
            params: Solar model construction settings (CSV path overrides).
                None loads the package defaults (see ``config.default``).
            context: Runtime device/dtype used for the loaded tensors.

        Returns:
            SolarProfile built from the configured structure, flux, and
            production-spectrum files.
        """
        params = params or SolarParameters()
        device, dtype = context.device, context.dtype
        fluxes = load_solar_fluxes(
            params.fluxes_path, provider=params.provider, device=device, dtype=dtype
        )

        production = load_solar_production(
            params.production_path, provider=params.provider, device=device, dtype=dtype,
        )
        density_table = load_solar_density(
            params.density_path, provider=params.provider, device=device, dtype=dtype,
        )
        spectra: dict[str, SolarSpectrum] = {}
        if params.spectrum_provider is not None:
            variants = dict(params.spectrum_variants or {})
            for source in available_solar_spectrum_sources(params.spectrum_provider):
                table = load_solar_spectrum(
                    source,
                    provider=params.spectrum_provider,
                    variant=variants.get(source, "default"),
                    device=device,
                    dtype=dtype,
                )
                spectra[source] = SolarSpectrum(
                    energy_MeV=table["energy"],
                    density_MeV_inverse=table["spectrum"],
                )
        production_radius = production["radius"]
        fractions = production["fractions"]
        production_measure = production.get("production_measure", "radial_pdf")
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
        radius = density_table["radius"]
        density = density_table["electron_density"]
        density_n = None
        if "neutron_density" in density_table:
            density_n = density_table["neutron_density"]
        elif params.composition_path is not None:
            composition = load_solar_composition(
                    params.composition_path, device=device, dtype=dtype,
                )
            ratio_on_grid = interp1d_linear(
                    x=radius, xp=composition["radius"], fp=composition["neutron_to_electron_ratio"],
                    left=composition["neutron_to_electron_ratio"][0],
                    right=composition["neutron_to_electron_ratio"][-1],
                    device=device, dtype=dtype,
                )
            density_n = density * ratio_on_grid

        return cls(
            radius=radius,
            density=density,
            production_radius=production_radius,
            fractions=fractions,
            fluxes=fluxes,
            spectra=spectra,
            spectrum_provider=params.spectrum_provider,
            production_measure=production_measure,
            density_n=density_n,
        )

    @property
    def device(self) -> torch.device:
        """Return the device used by the profile radius tensor.

        Returns:
            Torch device shared by the profile tensors.
        """
        return self.radius.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the real dtype used by the profile radius tensor.

        Returns:
            Torch dtype shared by the profile tensors.
        """
        return self.radius.dtype

    def electron_density(self, r_query: Tensor) -> Tensor:
        """Interpolate electron density at requested solar radii.

        Args:
            r_query: Query radii in the same units as ``self.radius``.

        Returns:
            Electron-density tensor interpolated on ``r_query``.
        """
        return interp1d_linear(
            x=r_query,
            xp=self.radius,
            fp=self.density,
            left=self.density[0],
            right=self.density[-1],
            device=self.device,
            dtype=self.dtype,
        )

    def neutron_density(self, r_query: Tensor) -> Tensor:
        """Interpolate neutron density at requested solar radii.

        Used for the 3+1 sterile neutral-current term
        (``core.common.hamiltonian``); only meaningful when this profile was
        built with composition data (``SolarProfile.default()`` always
        populates ``density_n``; a manually constructed ``SolarProfile`` may
        not).

        Args:
            r_query: Query radii in the same units as ``self.radius``.

        Returns:
            Neutron-density tensor interpolated on ``r_query``.

        Raises:
            ValueError: If ``self.density_n`` is None.
        """
        if self.density_n is None:
            raise ValueError(
                "SolarProfile.density_n is not set: this profile was not "
                "built with composition data (see "
                "medium.solar.io.load_solar_composition), so the sterile "
                "neutral-current term cannot be evaluated on it."
            )

        return interp1d_linear(
            x=r_query,
            xp=self.radius,
            fp=self.density_n,
            left=self.density_n[0],
            right=self.density_n[-1],
            device=self.device,
            dtype=self.dtype,
        )

    def production_distribution(
        self,
        sources: str | Sequence[str],
        radius: Tensor | None = None,
    ) -> Tensor:
        """Return normalized production distributions for one or more sources.

        Args:
            sources: Source key or ordered source keys stored in ``fractions``.
            radius: Optional query radii. None returns the native profile grid.

        Returns:
            One source has shape ``radius.shape``; multiple sources are stacked
            on a leading source axis in the requested order.

        Raises:
            KeyError: If ``source`` is not present in the profile.
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

        Used identically by the adiabatic (``Tei``) and numerical mass-weight
        computations in ``medium.solar.probability``, which both produce
        ``weights_r`` on this profile's ``production_radius`` grid in the
        same ``(..., n_r, N)`` shape convention.

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

    def flux(self, source: str) -> Tensor:
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
        energy = torch.as_tensor(E_MeV, device=self.device, dtype=self.dtype)
        zero = torch.zeros((), device=self.device, dtype=self.dtype)
        return interp1d_linear(
            x=energy,
            xp=table.energy_MeV,
            fp=table.density_MeV_inverse,
            left=zero,
            right=zero,
            device=self.device,
            dtype=self.dtype,
        )

    def __str__(self) -> str:
        """Return a compact summary of the solar profile configuration."""
        n_r = self.radius.numel()
        r0  = float(self.radius[0])
        r1  = float(self.radius[-1])
        ne_min = float(self.density.min())
        ne_max = float(self.density.max())
        sources = ", ".join(sorted(self.fractions.keys()))
        spectrum_sources = ", ".join(sorted(self.spectra))
        if self.density_n is None:
            nn_summary = "unavailable"
        else:
            nn_summary = f"[{float(self.density_n.min()):.2e}, {float(self.density_n.max()):.2e}] mol/cm³"
        return (
            f"SolarProfile | "
            f"n_r={n_r} | "
            f"r=[{r0:.3f}, {r1:.3f}] R☉ | "
            f"n_e=[{ne_min:.2e}, {ne_max:.2e}] mol/cm³ | "
            f"n_n={nn_summary} | "
            f"sources=[{sources}] | "
            f"spectra={self.spectrum_provider}[{spectrum_sources}] | "
            f"use_LZ={self.use_LZ} | "
            f"{self.device} / {self.dtype}"
        )

    __repr__ = __str__
