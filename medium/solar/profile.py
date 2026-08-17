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
Solar medium (structural) profile container and interpolation utilities.

This module defines the torch-native ``SolarMediumProfile`` container: the
electron/neutron density structure of the Sun, independent of *which*
neutrinos are produced where (``source.solar.model.SolarNeutrinoSource``).
``medium.solar.probability``/``medium.solar.flux`` take both a
``SolarMediumProfile`` and a ``SolarNeutrinoSource``; this module never
depends on ``source.solar``.

``config.solar.SolarParameters`` composes this module's medium settings with
``source.solar.SolarSourceParameters`` without introducing a reverse source
dependency here.

Module functions:
    build_solar_medium(...)
        Return an existing SolarMediumProfile on the requested device/dtype
        or load the default profile.
    SolarMediumProfile.default(...)
        Load the configured default solar density/composition tables.
    SolarMediumProfile.device
        Return the torch device used by the profile tensors.
    SolarMediumProfile.dtype
        Return the real dtype used by the profile tensors.
    SolarMediumProfile.electron_density(...)
        Interpolate electron density at requested solar radii.
    SolarMediumProfile.neutron_density(...)
        Interpolate neutron density (3+1 sterile neutral-current term) at
        requested solar radii.
"""



from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.medium.solar.io import (
    load_solar_composition,
    load_solar_density,
)
import tpeanuts.config.default as default
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import interp1d_linear


Tensor = torch.Tensor


@dataclass(frozen=True)
class SolarMediumParameters:
    """
    Construction settings for the tabulated solar medium (density structure).

    Parameters
    ----------
    provider:
        Optional canonical provider name (``"zenodo"``, ``"bahcall"`` or
        ``"legacy"``). Explicit table paths take precedence product by
        product. None uses the configured default filenames.

    density_path:
        Optional density-table override. None loads the selected provider's
        electron/neutron-density table.

    composition_path:
        Optional override path to the solar structure+composition table
        (``struct+nu_SF3_*.dat`` format) used to derive the neutron-density
        profile for the 3+1 sterile neutral-current term
        (``medium.solar.io.load_solar_composition``). None loads the default
        composition table. It is only needed as a fallback when the selected
        density table has no neutron-density column.
    """

    provider: Optional[str] = None
    density_path: Optional[str] = None
    composition_path: Optional[str] = None


def build_solar_medium(
    solar_medium: "SolarMediumProfile | None",
    *,
    params: Optional[SolarMediumParameters] = None,
    context: RuntimeContext,
) -> "SolarMediumProfile":
    """Return a SolarMediumProfile on the requested device and dtype.

    Args:
        solar_medium: Existing profile or None to load the default profile.
        params: Solar medium construction settings used when
            ``solar_medium`` is None.
        context: Target device/dtype for profile tensors.

    Returns:
        SolarMediumProfile with radius/density on the target device and
        dtype.
    """
    if solar_medium is None:
        return SolarMediumProfile.default(params=params, context=context)

    device, dtype = context.device, context.dtype
    if solar_medium.radius.device == device and solar_medium.radius.dtype == dtype:
        return solar_medium

    return SolarMediumProfile(
        radius=solar_medium.radius.to(device=device, dtype=dtype),
        density=solar_medium.density.to(device=device, dtype=dtype),
        density_n=(
            None if solar_medium.density_n is None
            else solar_medium.density_n.to(device=device, dtype=dtype)
        ),
        provider=solar_medium.provider,
    )


@dataclass
class SolarMediumProfile:
    """Torch representation of the solar density structure.

    ``radius``/``density``/``density_n`` store the full structural density
    grid, normally extending to r = 1 R_sun. The adiabatic propagation path
    interpolates these densities at the production points supplied by a
    ``source.solar.SolarNeutrinoSource``; the numerical path propagates over
    this full structural grid, merged with the source's production grid
    (see ``medium.solar.evolutor.build_solar_trajectory``).
    """

    radius: Tensor
    density: Tensor
    density_n: Optional[Tensor] = None
    provider: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the structural density grid."""
        if self.radius.ndim != 1 or self.radius.numel() < 2:
            raise ValueError("SolarMediumProfile.radius must be one-dimensional with at least two points.")
        if not torch.isfinite(self.radius).all() or torch.any(torch.diff(self.radius) <= 0):
            raise ValueError("SolarMediumProfile.radius must be finite and strictly increasing.")
        if torch.any(self.radius < 0) or torch.any(self.radius > 1):
            raise ValueError("SolarMediumProfile.radius must lie in [0, 1] R_sun.")
        if self.radius[-1] < 1.0 - 32 * torch.finfo(self.radius.dtype).eps:
            raise ValueError("SolarMediumProfile must extend to the solar surface r/R_sun = 1.")
        if self.density.shape != self.radius.shape:
            raise ValueError("SolarMediumProfile.density must have the same shape as radius.")
        if self.density.device != self.radius.device or self.density.dtype != self.radius.dtype:
            raise ValueError("SolarMediumProfile radius and density must share device and dtype.")
        if not torch.isfinite(self.density).all() or torch.any(self.density < 0):
            raise ValueError("SolarMediumProfile.density must be finite and non-negative.")
        if self.density_n is not None:
            if self.density_n.shape != self.radius.shape:
                raise ValueError("SolarMediumProfile.density_n must have the same shape as radius.")
            if not torch.isfinite(self.density_n).all() or torch.any(self.density_n < 0):
                raise ValueError("SolarMediumProfile.density_n must be finite and non-negative.")
            if self.density_n.device != self.radius.device or self.density_n.dtype != self.radius.dtype:
                raise ValueError("SolarMediumProfile density_n must share radius device and dtype.")

    @classmethod
    def default(
        cls,
        *,
        params: Optional[SolarMediumParameters] = None,
        context: RuntimeContext,
    ) -> "SolarMediumProfile":
        """Load the default solar density/composition tables.

        The default data files are controlled by ``tpeanuts.config.default``:
        currently the Zenodo SF-III AGSS09 density profile. Pass an explicit
        ``SolarMediumParameters`` to override them.

        ``density_n`` (needed for the 3+1 sterile neutral-current term) is
        loaded directly when the density table contains it. For legacy or
        electron-only tables, an explicitly selected structure/composition
        table can provide the interpolated ratio ``n_n/n_e`` as a fallback.

        Args:
            params: Solar medium construction settings (CSV path overrides).
                None loads the package defaults (see ``config.default``).
            context: Runtime device/dtype used for the loaded tensors.

        Returns:
            SolarMediumProfile built from the configured structure files.
        """
        params = params or SolarMediumParameters()
        device, dtype = context.device, context.dtype
        effective_provider = params.provider or default.solar_provider
        density_table = load_solar_density(
            params.density_path,
            provider=None if params.density_path else effective_provider,
            device=device,
            dtype=dtype,
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
            density_n=density_n,
            provider="custom" if params.density_path else effective_provider,
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
        built with composition data (``SolarMediumProfile.default()`` always
        populates ``density_n``; a manually constructed ``SolarMediumProfile``
        may not).

        Args:
            r_query: Query radii in the same units as ``self.radius``.

        Returns:
            Neutron-density tensor interpolated on ``r_query``.

        Raises:
            ValueError: If ``self.density_n`` is None.
        """
        if self.density_n is None:
            raise ValueError(
                "SolarMediumProfile.density_n is not set: this profile was not "
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

    def __str__(self) -> str:
        """Return a compact summary of the solar medium configuration."""
        n_r = self.radius.numel()
        r0  = float(self.radius[0])
        r1  = float(self.radius[-1])
        ne_min = float(self.density.min())
        ne_max = float(self.density.max())
        if self.density_n is None:
            nn_summary = "unavailable"
        else:
            nn_summary = f"[{float(self.density_n.min()):.2e}, {float(self.density_n.max()):.2e}] mol/cm³"
        return (
            f"SolarMediumProfile | "
            f"n_r={n_r} | "
            f"r=[{r0:.3f}, {r1:.3f}] R☉ | "
            f"n_e=[{ne_min:.2e}, {ne_max:.2e}] mol/cm³ | "
            f"n_n={nn_summary} | "
            f"{self.device} / {self.dtype}"
        )

    __repr__ = __str__
