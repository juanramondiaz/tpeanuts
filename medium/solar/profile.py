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
density, production fractions, and total source fluxes, and provides small
helpers for interpolation and normalization on the profile grid.

Module functions:
    build_solar_profile(...)
        Return an existing SolarProfile on the requested device/dtype or load
        the default profile.
    SolarProfile.default(...)
        Load the default B16 solar model and flux tables.
    SolarProfile.device
        Return the torch device used by the profile tensors.
    SolarProfile.dtype
        Return the real dtype used by the profile tensors.
    SolarProfile.electron_density(...)
        Interpolate electron density at requested solar radii.
    SolarProfile.production_fraction(...)
        Return or interpolate a source production fraction.
    SolarProfile.source_fractions(...)
        Return one or several source production fractions on the profile grid.
    SolarProfile.normalized_fraction(...)
        Normalize a source production fraction over the radius grid.
    SolarProfile.flux(...)
        Return the total flux for one solar source.
"""



from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from tpeanuts.medium.solar.io import load_b16_fluxes, load_b16_solar_model
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import interp1d_linear, normalize_trapz


Tensor = torch.Tensor


@dataclass(frozen=True)
class SolarParameters:
    """
    Construction settings for the tabulated solar model.

    Parameters
    ----------
    model_path:
        Optional override path to the solar density/composition CSV
        (``nudistr_b16_agss09.csv`` format). None loads the default model
        (currently zenodo SF3-AGSS09, r ∈ [0, 1.0] R_sun).

    fluxes_path:
        Optional override path to the per-source total-flux CSV
        (``fluxes_b16.csv`` format). None loads the default flux table
        (currently zenodo SF3-AGSS09).
    """

    model_path: Optional[str] = None
    fluxes_path: Optional[str] = None


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
        SolarProfile with radius, density, fractions, and fluxes on the
        target device and dtype.
    """
    if solar_profile is None:
        return SolarProfile.default(params=params, context=context)

    device, dtype = context.device, context.dtype
    if solar_profile.radius.device == device and solar_profile.radius.dtype == dtype:
        return solar_profile

    return SolarProfile(
        radius=solar_profile.radius.to(device=device, dtype=dtype),
        density=solar_profile.density.to(device=device, dtype=dtype),
        fractions={
            key: value.to(device=device, dtype=dtype)
            for key, value in solar_profile.fractions.items()
        },
        fluxes={
            key: value.to(device=device, dtype=dtype)
            for key, value in solar_profile.fluxes.items()
        },
        use_LZ=solar_profile.use_LZ,
    )


@dataclass
class SolarProfile:
    """Torch representation of a solar model."""

    radius: Tensor
    density: Tensor
    fractions: dict[str, Tensor]
    fluxes: dict[str, Tensor]
    use_LZ: bool = False

    @classmethod
    def default(
        cls,
        *,
        params: Optional[SolarParameters] = None,
        context: RuntimeContext,
    ) -> "SolarProfile":
        """Load the default solar model and flux tables.

        The default data files are controlled by ``tpeanuts.util.default``:
        currently the zenodo SF3-AGSS09 extended profile covering
        r ∈ [0, 1.0] R_sun. Pass an explicit ``SolarParameters`` to override.

        Args:
            params: Solar model construction settings (CSV path overrides).
                None loads the package default data (see ``util.default``).
            context: Runtime device/dtype used for the loaded tensors.

        Returns:
            SolarProfile built from the configured solar data files.
        """
        params = params or SolarParameters()
        device, dtype = context.device, context.dtype
        model = load_b16_solar_model(params.model_path, device=device, dtype=dtype)
        fluxes = load_b16_fluxes(params.fluxes_path, device=device, dtype=dtype)

        return cls(
            radius=model["radius"],
            density=model["density"],
            fractions=model["fractions"],
            fluxes=fluxes,
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

    def production_fraction(self, source: str, r_query: Tensor | None = None) -> Tensor:
        """Return or interpolate one source production fraction.

        Args:
            source: Solar source key stored in ``self.fractions``.
            r_query: Optional query radii. When omitted, returns the tabulated
                fraction on ``self.radius``.

        Returns:
            Source production fraction on the tabulated or queried grid.

        Raises:
            KeyError: If ``source`` is not present in the profile.
        """
        if source not in self.fractions:
            raise KeyError(f"Unknown solar source: {source}")

        fraction = self.fractions[source]

        if r_query is None:
            return fraction

        return interp1d_linear(
            x=r_query,
            xp=self.radius,
            fp=fraction,
            left=fraction[0],
            right=fraction[-1],
            device=self.device,
            dtype=self.dtype,
        )

    def source_fractions(self, sources: str | Sequence[str]) -> Tensor:
        """Return one or several source production fractions on the grid.

        Args:
            sources: Source key or ordered source keys stored in
                ``self.fractions``.

        Returns:
            Source production fraction tensor. A single source returns shape
            ``(n_radius,)``; multiple sources return shape
            ``(n_sources, n_radius)``.

        Raises:
            KeyError: If any requested source is not present in the profile.
        """
        if isinstance(sources, str):
            return self.production_fraction(sources)

        return torch.stack(
            [
                self.production_fraction(source)
                for source in sources
            ],
            dim=0,
        )

    def normalized_fraction(self, source: str) -> Tensor:
        """Normalize one source production fraction over the radius grid.

        Args:
            source: Solar source key stored in ``self.fractions``.

        Returns:
            Production fraction normalized by trapezoidal integration over
            ``self.radius``.
        """
        return normalize_trapz(
            self.production_fraction(source),
            x=self.radius,
            clamp_min=0.0,
        )

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

    def __str__(self) -> str:
        """Return a compact summary of the solar profile configuration."""
        n_r = self.radius.numel()
        r0  = float(self.radius[0])
        r1  = float(self.radius[-1])
        ne_min = float(self.density.min())
        ne_max = float(self.density.max())
        sources = ", ".join(sorted(self.fractions.keys()))
        return (
            f"SolarProfile | "
            f"n_r={n_r} | "
            f"r=[{r0:.3f}, {r1:.3f}] R☉ | "
            f"n_e=[{ne_min:.2e}, {ne_max:.2e}] mol/cm³ | "
            f"sources=[{sources}] | "
            f"use_LZ={self.use_LZ} | "
            f"{self.device} / {self.dtype}"
        )

    __repr__ = __str__
