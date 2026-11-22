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

"""The Sun as a propagation medium: its density profile, the mass-basis weights at the surface and ``propagate``.

``SolarMedium`` is the solar counterpart of ``tpeanuts.medium.earth.earth.EarthMedium``,
``tpeanuts.medium.atmosphere.atmosphere.AtmosphereMedium`` and ``tpeanuts.medium.vacuum.vacuum.VacuumMedium``: it holds
the validated, interpolatable radial density table (the analogue of
``tpeanuts.medium.earth.density.PremTable`` + ``interpolate_prem_density``) and propagates a ``Flux`` through it
(``mass_weights``, ``propagate``). What it does not do is return a ``tpeanuts.interfaces.medium.MediumProfile`` from a
``profile()`` method like the other media: solar propagation is not one path per call (Earth's single nadir angle) but
many production points sharing one destination (the solar surface), which needs
``tpeanuts.physics.evolution.numerical.shared_history.merge_origin_grid``'s own grid-merging geometry (via
``numerical_segments``) rather than a single ``length_m``/``n_e_mol_cm3`` pair. That contract has no concept of multiple
shared-endpoint origins (an ``origin_index``), and generalising it before a second medium needs the shared-origin
technique would be guessing at a shared contract from a single consumer.

Ported from v1_0's ``medium/solar/profile.py`` (its own ``io.py::
load_solar_density`` is this rewrite's ``density.py``), translated to
this project's established conventions
(``tpeanuts.medium.solar.density``, ``RuntimeContext``). v1_0's
``composition_path``/``load_solar_composition`` fallback (deriving
neutron density from a raw stellar-structure file's mass fractions when
a provider's density table lacks a neutron column) is not ported: none
of the three providers actually ported here needs it (zenodo and
bahcall's tables already carry neutron density directly; legacy's does
not, and has no default composition-path resolution in v1_0 either --
see ``tpeanuts.data.solar``'s own ``README.md``). ``density_n`` is
simply ``None`` for legacy, matching v1_0's own unconfigured behaviour
exactly.

Module contents:
    SolarMediumParameters
        Construction settings for the tabulated solar density profile.
    SolarMedium
        The Sun as a medium: torch container for the density table, with radius-interpolation
        methods, ``numerical_segments`` (path segmentation for the
        shared-origin numerical propagation method), ``mass_weights`` (the
        mass-basis weights at the surface of a neutrino born at each radius)
        and ``propagate`` (a Flux, or a SourceFlux, from its production
        inside the Sun to the solar surface).
    build_solar_medium(...)
        Return an existing SolarMedium on the requested
        device/dtype, or load the default one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

import tpeanuts.config.default as default
from tpeanuts.interfaces.flux import Flux, SourceFlux
from tpeanuts.medium.common.propagation import check_compatible
from tpeanuts.medium.solar.density import (
    SolarDensityTable,
    interpolate_solar_density,
    load_solar_density_table,
)
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution.adiabatic import adiabatic
from tpeanuts.physics.evolution.numerical import shared_history
from tpeanuts.physics.evolution.numerical.shared_history import merge_origin_grid
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import as_tensor

Tensor = torch.Tensor

_SOLAR_METHODS = ("adiabatic_approximated", "adiabatic_exact", "numerical")


@dataclass(frozen=True)
class SolarMediumParameters:
    """Construction settings for the tabulated solar density profile.

    Attributes:
        provider: Canonical provider (``"bahcall"``, ``"zenodo"`` or
            ``"legacy"``). None resolves to the configured default
            (``tpeanuts.config.default.SOLAR_PROVIDER``).
        density_path: Optional density-table override. None loads the
            configured provider's table; giving an explicit path always
            records ``"custom"`` as the built profile's own ``provider``
            (see ``SolarMedium.default``), even if ``provider`` is
            also given, so the metadata never claims a canonical table
            that was not actually the one loaded.
    """

    provider: Optional[str] = None
    density_path: Optional[Union[str, Path]] = None


def build_solar_medium(
    solar_medium: Optional["SolarMedium"],
    *,
    params: Optional[SolarMediumParameters] = None,
    context: RuntimeContext,
) -> "SolarMedium":
    """Return a SolarMedium on the requested device and dtype.

    Args:
        solar_medium: Existing profile or None to load the default
            profile.
        params: Solar medium construction settings used when
            ``solar_medium`` is None.
        context: Target device/dtype for the profile's tensors.

    Returns:
        SolarMedium on the target device and dtype.
    """
    if solar_medium is None:
        return SolarMedium.default(params=params, context=context)

    if solar_medium.device == context.device and solar_medium.dtype == context.dtype:
        return solar_medium

    return SolarMedium(
        radius=solar_medium.radius.to(device=context.device, dtype=context.dtype),
        density=solar_medium.density.to(device=context.device, dtype=context.dtype),
        density_n=(
            solar_medium.density_n.to(device=context.device, dtype=context.dtype)
            if solar_medium.density_n is not None
            else None
        ),
        provider=solar_medium.provider,
    )


@dataclass(frozen=True)
class SolarMedium:
    """The Sun as a propagation medium: the solar electron/neutron density profile and its propagation.

    ``radius`` is the independent grid on which ``density``/``density_n``
    are defined, in units of ``r / R_sun`` -- ``electron_density``/
    ``neutron_density`` interpolate at arbitrary query radii within
    ``[0, 1]``.

    Attributes:
        radius: ``r / R_sun``, shape ``(n_r,)``, strictly increasing,
            reaching the solar surface (``radius[-1] ~= 1.0``).
        density: Electron density at each radius, mol/cm^3, shape
            ``(n_r,)``.
        density_n: Neutron density at each radius, mol/cm^3, shape
            ``(n_r,)``, or None if the underlying table has no neutron
            column (true of the ``"legacy"`` provider -- see this
            module's own docstring).
        provider: Canonical provider name this profile was built from
            (``"bahcall"``, ``"zenodo"``, ``"legacy"`` or ``"custom"``),
            or None for a hand-built profile.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.medium import SolarMedium
        >>> sun = SolarMedium.default(context=context)   # the packaged standard solar model
        >>> float(sun.radius[0]), float(sun.radius[-1])   # radii in units of the solar radius
        (0.0, 1.0)
        >>> centre = torch.tensor([0.0], dtype=torch.float64)
        >>> round(float(sun.electron_density(centre)[0]), 1)   # mol/cm^3 at the centre
        100.7
    """

    radius: Tensor
    density: Tensor
    density_n: Optional[Tensor] = None
    provider: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the profile's radius and density tensors."""
        if self.radius.ndim != 1 or self.radius.numel() < 2:
            raise ValueError("SolarMedium.radius must be one-dimensional with at least two points.")
        if not torch.isfinite(self.radius).all() or torch.any(torch.diff(self.radius) <= 0):
            raise ValueError("SolarMedium.radius must be finite and strictly increasing.")
        if torch.any(self.radius < 0) or torch.any(self.radius > 1):
            raise ValueError("SolarMedium.radius must lie in [0, 1] R_sun.")
        surface_tolerance = 32 * torch.finfo(self.dtype).eps
        if self.radius[-1] < 1.0 - surface_tolerance:
            raise ValueError(
                f"SolarMedium.radius must reach the solar surface (radius[-1] ~= 1.0), "
                f"got {float(self.radius[-1]):.6f}."
            )

        density = as_tensor(self.density, device=self.device, dtype=self.dtype)
        if density.shape != self.radius.shape:
            raise ValueError(
                f"SolarMedium.density must have shape {tuple(self.radius.shape)}, got {tuple(density.shape)}."
            )
        if not torch.isfinite(density).all() or torch.any(density < 0):
            raise ValueError("SolarMedium.density must be finite and non-negative.")
        object.__setattr__(self, "density", density)

        if self.density_n is not None:
            density_n = as_tensor(self.density_n, device=self.device, dtype=self.dtype)
            if density_n.shape != self.radius.shape:
                raise ValueError(
                    f"SolarMedium.density_n must have shape {tuple(self.radius.shape)}, "
                    f"got {tuple(density_n.shape)}."
                )
            if not torch.isfinite(density_n).all() or torch.any(density_n < 0):
                raise ValueError("SolarMedium.density_n must be finite and non-negative.")
            object.__setattr__(self, "density_n", density_n)

    @classmethod
    def default(
        cls,
        *,
        params: Optional[SolarMediumParameters] = None,
        context: RuntimeContext,
    ) -> "SolarMedium":
        """Load the default solar density profile.

        The default data file is controlled by
        ``tpeanuts.config.default`` -- currently the Zenodo SF-III
        AGSS09 electron/neutron density table. Pass an explicit
        ``SolarMediumParameters`` to override it.

        Args:
            params: Solar medium construction settings (CSV path
                override). None loads the package default (see
                ``config.default``).
            context: Runtime device/dtype used for the loaded tensors.

        Returns:
            SolarMedium built from the configured density table.
        """
        params = params or SolarMediumParameters()
        provider = "custom" if params.density_path else (params.provider or default.SOLAR_PROVIDER)
        table: SolarDensityTable = load_solar_density_table(
            params.density_path,
            provider=None if params.density_path else provider,
            context=context,
        )
        return cls(radius=table.radius, density=table.n_e_mol_cm3, density_n=table.n_n_mol_cm3, provider=provider)

    @property
    def device(self) -> torch.device:
        """Return the device used by the profile's radius tensor."""
        return self.radius.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the real dtype used by the profile's radius tensor."""
        return self.radius.dtype

    def electron_density(self, r_query: Tensor) -> Tensor:
        """Interpolate electron density at arbitrary query radii within [0, 1]."""
        n_e, _ = interpolate_solar_density(r_query, self._as_table())
        return n_e

    def neutron_density(self, r_query: Tensor) -> Tensor:
        """Interpolate neutron density at arbitrary query radii within [0, 1].

        Raises:
            ValueError: If this profile has no neutron-density column
                (``self.density_n is None``).
        """
        if self.density_n is None:
            raise ValueError(
                f"SolarMedium (provider={self.provider!r}) has no neutron-density column."
            )
        _, n_n = interpolate_solar_density(r_query, self._as_table())
        assert n_n is not None  # density_n is set, so the table has the column
        return n_n

    def _as_table(self) -> SolarDensityTable:
        """Wrap this profile's own tensors (no copy) as a SolarDensityTable for interpolation."""
        return SolarDensityTable(radius=self.radius, n_e_mol_cm3=self.density, n_n_mol_cm3=self.density_n)

    def numerical_segments(
        self,
        origin_radius: Tensor,
        *,
        sampling: str = "midpoint",
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
        """Build the merged-grid segmentation the shared-origin numerical method needs.

        Merges ``self.radius`` and ``origin_radius`` into one trajectory
        (``tpeanuts.physics.evolution.numerical.shared_history.
        merge_origin_grid``) and interpolates this profile's own
        density at each merged segment's sample point -- the two steps
        every consumer of the shared-origin technique needs done before
        it can build a per-segment generator, done here once rather than
        by each caller.

        Args:
            origin_radius: Production-radius grid to merge as exact
                trajectory boundaries, shape ``(n_r,)`` -- typically a
                ``tpeanuts.source.solar.SolarSource``'s own
                ``production_radius``.
            sampling: Segment density-sampling rule, forwarded to
                ``merge_origin_grid``.

        Returns:
            ``(dx_evolution, n_e_mol_cm3, n_n_mol_cm3, origin_index)``:
            ``dx_evolution`` and the density samples are shaped
            ``(n,)`` (the merged grid's own segment count), ``n_n_mol_cm3``
            is None whenever ``self.density_n`` is (see
            ``SolarMedium``'s own docstring), and ``origin_index``
            locates each ``origin_radius`` entry within the merged grid,
            shape ``(n_r,)`` -- see ``merge_origin_grid``'s own docstring
            for exactly what each of the four returned tensors means.
        """
        _, dx_evolution, sample_x, origin_index = merge_origin_grid(self.radius, origin_radius, sampling=sampling)
        n_e_mol_cm3, n_n_mol_cm3 = interpolate_solar_density(sample_x, self._as_table())
        return dx_evolution, n_e_mol_cm3, n_n_mol_cm3, origin_index

    def mass_weights(
        self,
        model: PropagationModel,
        E_MeV: Tensor,
        radius_samples: Tensor,
        *,
        method: str = "adiabatic_approximated",
        use_LZ: bool = False,
        numerical_sampling: str = "midpoint",
    ) -> Tensor:
        """Mass-basis weights of an electron neutrino produced at each of ``radius_samples``, at the solar surface.

        The one place that validates every method/``use_LZ``/domain combination and dispatches to exactly one
        computation: the coherent evolution through the Sun (``"numerical"``) or the adiabatic mass weights
        (``"adiabatic_exact"``, ``"adiabatic_approximated"``, the latter optionally with the Landau-Zener correction).
        It does not know where the neutrinos are produced: the average over the production radius is the caller's
        (``ProductionDistribution.integrate``).

        Args:
            model: Built PropagationModel.
            E_MeV: Neutrino energy in MeV, scalar or any shape (``use_LZ`` needs at most 1-D).
            radius_samples: Radii of production, as a fraction of the solar radius, shaped ``(n_r,)``.
            method: ``"adiabatic_approximated"`` (default; the model may carry no extensions), ``"adiabatic_exact"`` or
                ``"numerical"``.
            use_LZ: Landau-Zener correction, only with ``method="adiabatic_approximated"``.
            numerical_sampling: Segment density-sampling rule, only used by ``method="numerical"`` (see
                ``numerical_segments``).

        Returns:
            Tensor shaped ``(*E_MeV.shape, n_r, n_flavour)``.

        Raises:
            ValueError: If ``method`` is unknown; if ``use_LZ`` is combined with another method or with a
                multi-dimensional energy; if ``radius_samples`` lies outside this profile's radial domain or on
                another device/dtype; or (``adiabatic_approximated``) if the model carries active extensions or an
                extended flavour space.
        """
        if method not in _SOLAR_METHODS:
            raise ValueError(f"method must be one of {_SOLAR_METHODS!r}, got {method!r}.")
        if self.device != radius_samples.device or self.dtype != radius_samples.dtype:
            raise ValueError(
                "Solar medium and source must share device and dtype; build both with the same RuntimeContext."
            )
        tolerance = 32 * torch.finfo(self.dtype).eps
        if radius_samples[0] < self.radius[0] - tolerance or radius_samples[-1] > self.radius[-1] + tolerance:
            raise ValueError(
                "Solar production radii must lie inside the solar-medium radial domain; "
                f"source=[{float(radius_samples[0])}, {float(radius_samples[-1])}], "
                f"medium=[{float(self.radius[0])}, {float(self.radius[-1])}]."
            )
        E_t = as_tensor(E_MeV, device=radius_samples.device, dtype=radius_samples.dtype)
        if use_LZ and method != "adiabatic_approximated":
            raise ValueError(
                f"use_LZ=True is only supported with method='adiabatic_approximated' "
                f"(got method={method!r}): "
                + (
                    "the coherent evolutor already captures every non-adiabatic transition "
                    "directly, so there is nothing for Landau-Zener to correct."
                    if method == "numerical"
                    else "the two-level Landau-Zener crossing formula has no closed-form "
                    "generalisation to off-diagonal NSI couplings or a fourth level."
                )
            )
        if use_LZ and E_t.ndim > 1:
            raise ValueError(
                "use_LZ=True requires a scalar or 1-D energy grid: the Landau-Zener correction "
                f"is not implemented for E_MeV.ndim={E_t.ndim}. Reshape E_MeV to at most 1-D, or "
                "set use_LZ=False."
            )
        if method == "numerical":
            dx_evolution, n_e_samples, n_n_samples, origin_index = self.numerical_segments(
                radius_samples, sampling=numerical_sampling,
            )
            return shared_history.mass_weights_numerical(
                model, E_t, dx_evolution, n_e_samples, n_n_samples, origin_index,
            )
        density = self.electron_density(radius_samples)
        if method == "adiabatic_exact":
            n_n = self.neutron_density(radius_samples) if self.density_n is not None else None
            environment = PropagationEnvironment(E_MeV=E_t[..., None], n_e_mol_cm3=density, n_n_mol_cm3=n_n)
            return adiabatic.mass_weights_adiabatic(model, environment, variant="exact")
        environment = PropagationEnvironment(E_MeV=E_t[..., None], n_e_mol_cm3=density)
        return adiabatic.mass_weights_adiabatic(
            model, environment, variant="approximated",
            medium_radius=self.radius if use_LZ else None,
            medium_density=self.density if use_LZ else None,
            radius_samples=radius_samples if use_LZ else None,
        )

    def propagate(
        self,
        flux: Union[Flux, SourceFlux],
        model: PropagationModel,
        *,
        method: str = "adiabatic_approximated",
        use_LZ: bool = False,
        numerical_sampling: str = "midpoint",
        location: Optional[str] = None,
    ) -> Union[Flux, SourceFlux]:
        """Carry a flux produced inside the Sun to its surface.

        Every solar neutrino is born an electron neutrino at some radius; what leaves the Sun is an incoherent mixture
        of mass eigenstates, the weights of ``mass_weights`` averaged over the flux's ``production`` distribution and
        rotated into the flavour basis. The flux's own state is therefore not evolved but required to be the pure
        electron-flavour state a solar source produces.

        A ``SourceFlux`` is carried component by component, computing the radial weights once for all the components
        that share an energy grid and production radius (the eight solar channels do, mostly).

        Args:
            flux: A Flux (or a SourceFlux of them) with a ``production`` distribution, a pure electron-flavour state,
                and the neutrino convention of ``model``.
            model: Built PropagationModel.
            method, use_LZ, numerical_sampling: See ``mass_weights``.
            location: ``location`` of the result; ``"surface"`` if omitted.

        Returns:
            The same kind of object, in the flavour state at the solar surface, with ``"sun"`` appended to ``history``.

        Raises:
            ValueError: If a flux has no ``production`` distribution, is not a pure electron-flavour state, disagrees
                with ``model`` or this profile (device, dtype, antinu), or as ``mass_weights``.
        """
        if isinstance(flux, SourceFlux):
            cache: list = []
            return flux.map(
                lambda component: self._propagate_one(
                    component, model, method, use_LZ, numerical_sampling, location, cache
                )
            )
        return self._propagate_one(flux, model, method, use_LZ, numerical_sampling, location, [])

    def _propagate_one(self, flux, model, method, use_LZ, numerical_sampling, location, cache) -> Flux:
        """``propagate`` for one Flux; ``cache`` holds ``(energy, radius, weights_r)`` of those already done."""
        context = RuntimeContext(device=self.device, dtype=self.dtype)
        check_compatible(flux, model, context)
        if flux.production is None:
            raise ValueError("a flux propagated through the Sun needs a production distribution (flux.production).")
        electron = flux.state.probabilities()[..., 0]
        if not bool(torch.all(torch.abs(electron - 1.0) < 1e-6)):
            raise ValueError("solar propagation assumes the neutrinos are produced as electron neutrinos.")

        production = flux.production
        weights_r = next(
            (w for e, r, w in cache if torch.equal(e, flux.energy_MeV) and torch.equal(r, production.radius)), None
        )
        if weights_r is None:
            weights_r = self.mass_weights(
                model, flux.energy_MeV, production.radius,
                method=method, use_LZ=use_LZ, numerical_sampling=numerical_sampling,
            )
            cache.append((flux.energy_MeV, production.radius, weights_r))

        weights = production.integrate(weights_r)
        rho_mass = DensityMatrix.from_weights(weights, model.space, context)
        state = rho_mass.conjugate_by(model.mixing.pmns_matrix(antinu=model.antinu))
        return flux.with_state(state, location=location or "surface", step="sun")

    def __str__(self) -> str:
        """Return a compact summary of the solar medium configuration."""
        n_r = self.radius.numel()
        has_n = self.density_n is not None
        return (
            f"SolarMedium | "
            f"n_r={n_r} | "
            f"r=[{float(self.radius[0]):.3f}, {float(self.radius[-1]):.3f}] R_sun | "
            f"neutron_density={'yes' if has_n else 'no'} | "
            f"provider={self.provider} | "
            f"{self.device} / {self.dtype}"
        )

    __repr__ = __str__
