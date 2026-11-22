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

"""A production-to-detector atmosphere-only path, sampled for the numerical or analytic evolver.

``AtmosphereMedium.profile()`` mirrors
``tpeanuts.medium.earth.earth.EarthMedium``'s own ``NUMERICAL``/``ANALYTIC``
case-A sampling exactly: ``nsteps`` equal-length segments tiling the
chord from the detector's own radius fraction (``s=0``, the surface when
``depth_m=0``) to the production altitude (``s=`` the chord's own
length), each sampled at its own midpoint -- only here there is no
``x1`` offset (the detector is one genuine endpoint of this chord, not a
point of closest approach in its middle), and density comes from
``tpeanuts.medium.atmosphere.density.atmosphere_density`` rather than a
tabulated PREM table. ``ANALYTIC`` needs no different segment data at all
-- it is an exact drop-in replacement for ``NUMERICAL``'s ``matrix_exp``
on a constant-density segment (see ``tpeanuts.physics.evolution``'s own
module docstring), so the same discretization already serves both; only
``MediumProfile.method`` itself differs.

``PropagationMethod.ANALYTIC_PERTURBATIVE`` instead splits the chord into
``perturbative_segments`` equal pieces (not ``nsteps`` -- ``nsteps`` is
ignored for this method, mirroring ``EarthMedium``'s own convention),
each carrying a genuine within-segment density *variation* -- a local
polynomial fitted from sampled density via
``tpeanuts.medium.atmosphere.polynomial_profile.
atmosphere_perturbative_profile`` -- rather than a single per-segment
point sample. Unlike Earth's own even-power fit (globally tabulated
ahead of time, shell-aligned), this one is fitted fresh from
``self.density_source`` for every call: the atmosphere-only chord has no
single tabulated density source to pre-fit coefficients for, and, unlike
Earth's shell-crossing segment count, ``perturbative_segments`` is a
fixed user choice independent of ``eta_rad``/``depth_m`` -- so, unlike
``EarthMedium``'s own restriction (scalar ``eta_rad``/``depth_m`` as soon as
``depth_m > 0``, for all three methods), no scalar requirement is needed
here even at ``depth_m > 0`` (confirmed by this module's own tests, not
just expected).

**This medium never builds an underground segment, regardless of
``depth_m``**: ``depth_m`` only relocates where the atmosphere-only
chord starts (see ``tpeanuts.medium.atmosphere.geometry``'s own module
docstring for why, confirmed against v1_0's own
``medium/atmosphere/evolutor.py``). A buried detector's own underground
leg is a separate ``tpeanuts.medium.earth.earth.EarthMedium`` evolutor,
composed afterwards -- see ``tpeanuts.pipeline.atmospheric_neutrino``.

Module contents:
    AtmosphereMedium
        Frozen dataclass: a nadir angle, a production altitude, a density
        backend choice, a propagation method, a segment/fit configuration,
        and a RuntimeContext, with a ``profile()`` method building the
        corresponding ``MediumProfile`` and an ``evolutor()`` method building
        the evolution operator of the atmosphere-only leg, and a ``propagate()``
        method applying it to a ``Flux``; ``from_flux`` builds the medium for a flux with a direction axis and a
        production altitude (the atmospheric source), whose ``propagate`` also averages over the production altitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from tpeanuts.config import default
from tpeanuts.interfaces.flux import Flux
from tpeanuts.interfaces.medium import MediumProfile
from tpeanuts.medium.atmosphere.density import atmosphere_density
from tpeanuts.medium.atmosphere.geometry import altitude_along_path_m, atmosphere_chord_length_m
from tpeanuts.medium.atmosphere.polynomial_profile import atmosphere_perturbative_profile
from tpeanuts.medium.common.propagation import check_compatible, evolve_flux
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.evolution.dispatch import evolve_path
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor

_ZERO: Any = 0.0  # default depth; __post_init__ turns it into a tensor on the context


@dataclass(frozen=True)
class AtmosphereMedium:
    """A source-to-surface atmosphere path, sampled from a chosen density backend.

    Attributes:
        eta_rad: Nadir angle, in radians, at the detector. Scalar or any
            batch shape -- see
            ``tpeanuts.medium.atmosphere.geometry.atmosphere_chord_length_m``
            for its full ``[0, pi]`` domain (no case split needed here).
        h_m: Production altitude above the surface, in metres; must be
            strictly positive.
        context: RuntimeContext the returned profile's tensors are built
            on.
        depth_m: Detector depth below the surface, in metres. Defaults
            to ``0.0`` (a surface detector, this package's original
            scope, fully backward compatible). Scalar or any batch
            shape, broadcastable against ``eta_rad`` -- see this module's
            own docstring for why no scalar restriction is needed here,
            unlike ``EarthMedium``'s own.
        nsteps: Number of equal-length path segments to discretize into.
            Defaults to ``tpeanuts.config.default.ATMOSPHERE_NSTEPS``.
        density_source: Which ``tpeanuts.medium.atmosphere.density.
            atmosphere_density`` backend to sample -- ``"exponential"``
            (default) or ``"file"`` this phase.
        density_kwargs: Extra keyword arguments forwarded to the selected
            density backend (e.g. ``density_file`` for ``"file"``,
            ``rho0_gcm3``/``scale_height_m`` overrides for
            ``"exponential"``).
        method: Which ``PropagationMethod`` this profile is built for --
            see this module's own docstring for what each one produces.
            Defaults to ``PropagationMethod.NUMERICAL``.
        perturbative_segments: Number of equal-length chord segments for
            ``ANALYTIC_PERTURBATIVE`` only (ignored otherwise -- ``nsteps``
            is ignored for this method instead). Defaults to
            ``tpeanuts.config.default.ATMOSPHERE_PERTURBATIVE_SEGMENTS``.
        perturbative_degree: Local polynomial degree fitted within each
            ``ANALYTIC_PERTURBATIVE`` segment (ignored otherwise).
            Defaults to
            ``tpeanuts.config.default.ATMOSPHERE_PERTURBATIVE_DEGREE``.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.medium import AtmosphereMedium
        >>> medium = AtmosphereMedium(   # straight up from the detector to a production height of 15 km
        ...     eta_rad=torch.tensor(math.pi, dtype=torch.float64), h_m=torch.tensor(15.0e3, dtype=torch.float64),
        ...     context=context, nsteps=10)
        >>> profile = medium.profile()
        >>> profile.n_segments, round(float(profile.length_m.sum()))
        (10, 15000)
    """

    eta_rad: torch.Tensor
    h_m: torch.Tensor
    context: RuntimeContext
    depth_m: torch.Tensor = _ZERO
    nsteps: int = default.ATMOSPHERE_NSTEPS
    density_source: str = default.ATMOSPHERE_SOURCE_DENSITY
    density_kwargs: Dict[str, Any] = field(default_factory=dict)
    method: PropagationMethod = PropagationMethod.NUMERICAL
    perturbative_segments: int = default.ATMOSPHERE_PERTURBATIVE_SEGMENTS
    perturbative_degree: int = default.ATMOSPHERE_PERTURBATIVE_DEGREE

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            eta_rad: TensorLike,
            h_m: TensorLike,
            context: RuntimeContext,
            depth_m: TensorLike = ...,
            nsteps: int = ...,
            density_source: str = ...,
            density_kwargs: Dict[str, Any] = ...,
            method: PropagationMethod = ...,
            perturbative_segments: int = ...,
            perturbative_degree: int = ...,
        ) -> None: ...

    @classmethod
    def from_flux(cls, flux: Flux, *, depth_m: TensorLike = _ZERO, **options: Any) -> "AtmosphereMedium":
        """The medium that carries ``flux`` (a direction axis and a production altitude) to the surface.

        ``eta_rad`` becomes the column of the flux's nadir angles and ``h_m`` the row of its strictly positive
        production altitudes, so that angle, height and energy are three independent batch axes (see ``evolutor``). A
        production point at the surface (``h = 0``, the first point of every generated grid) has no atmosphere to cross
        and is left out of the medium; ``propagate`` treats it as the identity.

        Args:
            flux: A Flux with ``direction`` and ``production`` (e.g. a component of
                ``AtmosphericSource.flux()``).
            depth_m: Detector depth below the surface, in metres.
            **options: Any other field of this class (``nsteps``, ``density_source``, ``method``, ...).

        Returns:
            An AtmosphereMedium on the flux's context.

        Raises:
            ValueError: If the flux has no direction or no production altitude, or no production point above the
                surface.
        """
        if flux.direction is None or flux.production is None:
            raise ValueError("an atmospheric flux needs a direction axis and a production altitude (flux.direction, "
                             "flux.production).")
        heights = flux.production.radius
        heights = heights[heights > 0.0]
        if heights.numel() == 0:
            raise ValueError("the production altitudes of the flux are all at or below the surface.")
        return cls(
            eta_rad=flux.direction.reshape(-1, 1), h_m=heights.reshape(1, -1), context=flux.context, depth_m=depth_m,
            **options,
        )

    def profile(self) -> MediumProfile:
        """Build this trajectory's MediumProfile, shaped for ``self.method``.

        Returns:
            See this module's own docstring for what each ``method``
            produces.

        Raises:
            ValueError: If ``eta_rad``/``h_m`` is outside its valid domain
                (raised by ``atmosphere_chord_length_m``), or if
                ``density_source``/its ``density_kwargs`` are invalid
                (raised by ``atmosphere_density``), or (``ANALYTIC_PERTURBATIVE``
                only) if ``perturbative_segments``/``perturbative_degree``
                is invalid.
        """
        if self.method is PropagationMethod.ANALYTIC_PERTURBATIVE:
            return self._profile_analytic_perturbative()
        return self._profile_sampled()

    def evolutor(self, model: PropagationModel, energy_MeV: TensorLike) -> torch.Tensor:
        """Build the evolution operator of the atmosphere-only leg for ``model`` at ``energy_MeV``.

        ``eta_rad``, ``h_m`` and ``energy_MeV`` are all independently batchable, each on its own orthogonal axis,
        whether or not the other two are scalar or batched: ``eta_rad`` shaped ``(n_angle, 1)`` and ``h_m`` shaped
        ``(1, n_h)`` broadcast through this medium into one ``(n_angle, n_h, nsteps)`` profile, and a non-scalar
        energy then opens its *own* new axis -- inserted by unsqueezing the profile's tensors at ``-2``, just before
        the trailing segment axis, not by unsqueezing the energy alone, which would collide with whichever axis
        happens to be trailing in the combined ``eta_rad``/``h_m`` batch shape (checked against a manual per-element
        loop, not just by shape). This differs from ``EarthMedium.evolutor``, where energy and ``eta_rad`` broadcast
        like ordinary tensors.

        Args:
            model: Built PropagationModel (mixing, mass spectrum, and any active extensions/dissipators).
            energy_MeV: Neutrino energy in MeV, any shape (a scalar included).

        Returns:
            The evolution operator, shaped ``(*eta_and_h_m_batch_shape, n_E, n_flavour, n_flavour)`` when the energy
            is non-scalar (``(*eta_and_h_m_batch_shape, n_flavour, n_flavour)`` when it is a scalar), or the
            Liouville-space shape if ``model`` has active dissipators.

        Raises:
            ValueError: Via ``profile()`` (an invalid ``eta_rad``/``h_m``/``depth_m``, or invalid
                ``density_source``/``density_kwargs``/segment configuration), or via ``evolve_path`` (an
                incompatible ``method`` against active dissipators).
        """
        E_MeV = as_tensor(energy_MeV, device=self.context.device, dtype=self.context.dtype)
        # ANALYTIC_PERTURBATIVE carries its density in the profile object, whose tensors cannot be unsqueezed after
        # the fact: it is built with the energy slot already open (see ``_profile_analytic_perturbative``).
        energy_slot_in_profile = self.method is PropagationMethod.ANALYTIC_PERTURBATIVE and E_MeV.ndim > 0
        medium_profile = (
            self._profile_analytic_perturbative(energy_axis=True) if energy_slot_in_profile else self.profile()
        )
        n_e_mol_cm3 = medium_profile.n_e_mol_cm3
        n_n_mol_cm3 = medium_profile.n_n_mol_cm3
        length_m = medium_profile.length_m
        if E_MeV.ndim > 0:
            if not energy_slot_in_profile:
                n_e_mol_cm3 = n_e_mol_cm3.unsqueeze(-2)
                n_n_mol_cm3 = n_n_mol_cm3.unsqueeze(-2) if n_n_mol_cm3 is not None else None
                length_m = length_m.unsqueeze(-2)
            E_MeV = E_MeV.reshape(-1, 1)

        environment = PropagationEnvironment(E_MeV=E_MeV, n_e_mol_cm3=n_e_mol_cm3, n_n_mol_cm3=n_n_mol_cm3)
        # The profile is tiled from the detector (segment 0) up to the production point, while a state passes the
        # production end first: the segments are composed in reverse (checked against nuSQuIDS and an independent
        # NumPy propagation, notebooks/3_validation/2_external/1_nusquids/03_atmosphere).
        return evolve_path(
            model,
            environment,
            length_m / R_E,
            method=medium_profile.method,
            perturbative_profile=medium_profile.perturbative_profile,
            evolution_scale_m=R_E,
            reverse=True,
        )

    def propagate(self, flux: Flux, model: PropagationModel, *, location: Optional[str] = None) -> Flux:
        """Propagate ``flux`` from its production altitude down to the Earth's surface and return it evolved.

        The operator is ``evolutor(model, flux.energy_MeV)``, applied to ``flux.state`` (a Liouville evolution if
        ``model`` has dissipators). Angle, height and energy are three independent batch axes, see ``evolutor``.

        Args:
            flux: The flux to propagate, on this medium's device/dtype and with the same neutrino/antineutrino
                convention as ``model``.
            model: Built PropagationModel.
            location: ``location`` of the returned flux; ``"surface"`` if omitted (the atmosphere ends where the Earth
                begins -- continue with ``EarthMedium.propagate``).

        Returns:
            A new Flux, with ``"atmosphere"`` appended to its ``history``.

        Raises:
            ValueError: If ``flux`` and ``model`` or this medium disagree (device, dtype, antinu), or as
                ``evolutor``.
        """
        check_compatible(flux, model, self.context)
        if flux.production is not None:
            return self._propagate_from_production(flux, model, location or "surface")
        operator = self.evolutor(model, flux.energy_MeV)
        return evolve_flux(flux, model, operator, location=location or "surface", step="atmosphere")

    def _propagate_from_production(self, flux: Flux, model: PropagationModel, location: str) -> Flux:
        """Propagate a flux with a direction axis and a production altitude, averaging over the altitude.

        Neutrinos born at different altitudes do not interfere, so the state at the surface is the average of the
        evolved states weighted by the production distribution ``f(h | direction, E)``, a density in ``h``: the
        average of the density matrices, not of the probabilities, since the Earth leg that follows needs the
        coherences. A production point at the surface contributes the unevolved state.
        """
        production, direction = flux.production, flux.direction
        if direction is None or flux.energy_MeV.ndim != 1:
            raise ValueError("a flux with a production altitude needs a direction axis and a one-dimensional energy.")
        positive = production.radius > 0.0
        if (
            self.eta_rad.shape != (direction.numel(), 1)
            or self.h_m.shape != (1, int(positive.sum()))
            or not torch.allclose(self.eta_rad.reshape(-1), direction)
            or not torch.allclose(self.h_m.reshape(-1), production.radius[positive])
        ):
            raise ValueError("this medium does not match the flux's directions and production altitudes: build it "
                             "with AtmosphereMedium.from_flux(flux).")

        evolved = evolve_flux(flux, model, self.evolutor(model, flux.energy_MeV), location=location, step="atmosphere")
        states = evolved.state.tensor  # (n_direction, n_height, n_energy, dim, dim)
        if not bool(positive[0]):
            surface = flux.state.tensor.expand(direction.numel(), 1, flux.energy_MeV.numel(), *states.shape[-2:])
            states = torch.cat([surface, states], dim=1)
        dim = states.shape[-1]
        # (n_direction, n_energy, n_height, dim*dim): the height axis next to the matrix, as `integrate` expects
        by_height = states.reshape(*states.shape[:3], dim * dim).transpose(1, 2)
        average = production.integrate(by_height).reshape(*by_height.shape[:2], dim, dim)
        return flux.with_state(DensityMatrix(average, flux.space, flux.context), location=location, step="atmosphere")

    def _profile_sampled(self) -> MediumProfile:
        """``NUMERICAL``/``ANALYTIC``: ``nsteps`` equal segments, each sampled at its own midpoint."""
        eta_t = as_tensor(self.eta_rad, device=self.context.device, dtype=self.context.dtype)
        depth_t = as_tensor(self.depth_m, device=self.context.device, dtype=self.context.dtype)
        chord_length_m = atmosphere_chord_length_m(eta_t, self.h_m, depth_m=depth_t, context=self.context)

        step_length_m = chord_length_m / self.nsteps
        step_index = torch.arange(self.nsteps, device=self.context.device, dtype=self.context.dtype)
        # Midpoint of each of nsteps equal segments tiling [0, chord_length_m].
        midpoints_s_m = (step_index + 0.5) * step_length_m.unsqueeze(-1)

        altitude_m = altitude_along_path_m(
            midpoints_s_m,
            eta_t.unsqueeze(-1),
            depth_m=depth_t.unsqueeze(-1) if depth_t.ndim > 0 else depth_t,
            context=self.context,
        )

        n_e_mol_cm3 = atmosphere_density(
            altitude_m, source=self.density_source, density_type="electron_density",
            context=self.context, **self.density_kwargs,
        )
        n_n_mol_cm3 = atmosphere_density(
            altitude_m, source=self.density_source, density_type="neutron_density",
            context=self.context, **self.density_kwargs,
        )

        length_m = step_length_m.unsqueeze(-1).expand_as(n_e_mol_cm3).clone()
        return MediumProfile(
            length_m=length_m,
            n_e_mol_cm3=n_e_mol_cm3,
            n_n_mol_cm3=n_n_mol_cm3,
            context=self.context,
            method=self.method,
        )

    def _profile_analytic_perturbative(self, energy_axis: bool = False) -> MediumProfile:
        """``ANALYTIC_PERTURBATIVE``: ``perturbative_segments`` segments, each with its own density variation.

        With ``energy_axis`` the profile is built with a size-1 batch slot just before the segment axis, where a
        non-scalar energy goes (``evolutor``): ``eta_rad``, ``h_m`` and ``depth_m`` each get a trailing axis before the
        density is fitted, so every tensor of the profile object carries the slot, as the sampled profiles do after
        ``unsqueeze(-2)``.
        """
        eta_rad, h_m, depth_m = self.eta_rad, self.h_m, self.depth_m
        if energy_axis:
            eta_rad, h_m = (as_tensor(item, device=self.context.device, dtype=self.context.dtype).unsqueeze(-1)
                        for item in (eta_rad, h_m))
            depth = as_tensor(depth_m, device=self.context.device, dtype=self.context.dtype)
            depth_m = depth.unsqueeze(-1) if depth.ndim > 0 else depth
        profile = atmosphere_perturbative_profile(
            eta_rad, h_m,
            n_segments=self.perturbative_segments, degree=self.perturbative_degree,
            density_source=self.density_source, density_kwargs=self.density_kwargs,
            depth_m=depth_m, evolution_scale_m=R_E, context=self.context,
        )
        # profile.x1/x2 are dimensionless (evolution_scale_m units) --
        # MediumProfile.length_m needs metres, so scale back up.
        return MediumProfile(
            length_m=(profile.x2 - profile.x1) * R_E,
            n_e_mol_cm3=profile.average,
            n_n_mol_cm3=profile.average_n,
            context=self.context,
            method=PropagationMethod.ANALYTIC_PERTURBATIVE,
            perturbative_profile=profile,
        )
