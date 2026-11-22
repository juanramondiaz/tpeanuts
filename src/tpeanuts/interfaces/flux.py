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

"""What flows through a chain of media: an energy spectrum, a propagating state and an optional absolute intensity.

A ``Flux`` is the one object that travels from the source to the detector. It is created by a source, handed to a
medium, and comes back from it with the state evolved (``with_state``); it never changes type along the way.
Three things are kept apart on purpose:

* the **energy grid** (``energy_MeV``): the true neutrino energy, which propagation conserves, so it is the same
  quantity in the source, in every medium and in the detector (the energy a detector *measures* is a different variable,
  named differently where it appears);
* the **flavour content** (``state``): a ``DensityMatrix`` normalised to trace one, holding the populations *and* the
  coherences between flavours -- which is what propagation needs to act on;
* the **intensity** (``differential_flux``): the absolute number of neutrinos per unit energy, a plain scale factor.
  The flux of one flavour is the product ``differential_flux * state.probabilities()[..., alpha]`` (``per_flavour``),
  never stored separately, so there is a single place where the flavour lives.

``production`` describes where inside an extended source the neutrinos are born (the radial profile of the Sun, the
production altitude of an atmospheric neutrino), which a medium that starts inside the source needs. ``direction`` is
the optional axis of arrival directions of a flux that depends on them (the atmospheric flux): the nadir angles at
which ``differential_flux`` is given, matched by the batch axis of the state just before the energy. A source made of
several components with their own spectra (the eight solar channels, the four fissile isotopes of a reactor) is a
``SourceFlux``: one ``Flux`` per component.

Module contents:
    integrate_radial(...)
        Reduce a production-weighted quantity over the radius axis, by the production's measure.
    ProductionDistribution
        Frozen dataclass: a radial grid and the normalised weight of the production at each radius, with an
        ``integrate`` method averaging per-radius values over it.
    Flux
        Frozen dataclass: ``energy_MeV``, ``state`` (a ``DensityMatrix``), ``antinu``, and the optional
        ``differential_flux``, ``flux_units``, ``production``, ``location``, ``history`` and ``direction``.
    SourceFlux
        Frozen dataclass: named ``Flux`` components of one source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Dict, Iterator, Optional, Tuple, Union

import torch

from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import TensorLike, as_tensor

LOCATIONS = ("source", "surface", "detector")


def integrate_radial(weighted: torch.Tensor, radius: torch.Tensor, measure: str) -> torch.Tensor:
    """Reduce a quantity already multiplied by the production weight over the radius axis (``-2``).

    ``"shell_fraction"`` weights are discrete per-shell fractions that already add up to one, so they reduce with a
    plain sum (re-weighting by the local radius spacing, as an integral would, double-counts the grid spacing and biases
    the result towards the more finely sampled shells); ``"radial_pdf"`` weights are a density in radius, reduced with
    the trapezoidal rule over ``radius``.

    Args:
        weighted: Tensor shaped ``(..., n_radius, n_last)``, already multiplied by the production weight.
        radius: Radial grid, shaped ``(n_radius,)``.
        measure: ``"radial_pdf"`` or ``"shell_fraction"``.

    Returns:
        The reduction over the radius axis, shaped ``(..., n_last)``.
    """
    if measure == "shell_fraction":
        return weighted.sum(dim=-2)
    return torch.trapz(weighted, x=radius, dim=-2)


@dataclass(frozen=True)
class ProductionDistribution:
    """Where inside an extended source the neutrinos are produced.

    Attributes:
        radius: Grid of the production coordinate, shaped ``(n_radius,)`` and increasing: the radius as a fraction of
            the source radius for the Sun, the altitude above the surface in metres for the atmosphere.
        weight: Normalised weight of the production at each point of the grid, shaped ``(..., n_radius)``,
            non-negative; leading axes are those of a distribution that depends on direction and energy (the
            atmospheric production altitude does). It is the distribution as the source stores it (a discrete
            per-shell fraction or a density to be integrated, see ``measure``); it is used as given and not
            normalised a second time.
        measure: ``"radial_pdf"`` (``weight`` is a density in radius, integrated with the trapezoidal rule) or
            ``"shell_fraction"`` (``weight`` are fractions of shells that already add up to one).
        context: RuntimeContext the tensors are stored on.

    Raises:
        ValueError: If ``radius`` is not one-dimensional or the last axis of ``weight`` does not match it, ``weight``
            has a negative entry, or ``measure`` is not one of the two above.

    Examples:
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.interfaces import ProductionDistribution
        >>> production = ProductionDistribution(radius=[0.0, 0.1, 0.2], weight=[0.2, 0.5, 0.3], context=context,
        ...                                     measure="shell_fraction")
        >>> production.radius.shape
        torch.Size([3])
    """

    radius: torch.Tensor
    weight: torch.Tensor
    context: RuntimeContext
    measure: str = "radial_pdf"

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            radius: TensorLike,
            weight: TensorLike,
            context: RuntimeContext,
            measure: str = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        device, dtype = self.context.device, self.context.dtype
        radius = as_tensor(self.radius, device=device, dtype=dtype)
        weight = as_tensor(self.weight, device=device, dtype=dtype)
        if radius.ndim != 1 or weight.ndim < 1 or weight.shape[-1] != radius.shape[0]:
            raise ValueError(
                f"radius must be one-dimensional and the last axis of weight must match it, got radius "
                f"{tuple(radius.shape)} and weight {tuple(weight.shape)}."
            )
        if torch.any(weight < 0):
            raise ValueError("weight must be non-negative everywhere.")
        if self.measure not in ("radial_pdf", "shell_fraction"):
            raise ValueError(f"measure must be 'radial_pdf' or 'shell_fraction', got {self.measure!r}.")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "weight", weight)

    def integrate(self, values_r: torch.Tensor) -> torch.Tensor:
        """Average per-radius values over this production distribution.

        Args:
            values_r: Values on the radial grid, shaped ``(..., n_radius, n_last)`` (e.g. the mass-basis weights of a
                neutrino produced at each radius, with the mass index last); the leading axes broadcast against those
                of ``weight``.

        Returns:
            Tensor shaped ``(..., n_last)``: the values weighted by the production and reduced over the radius, as the
            ``measure`` prescribes.
        """
        return integrate_radial(values_r * self.weight.unsqueeze(-1), self.radius, self.measure)


@dataclass(frozen=True)
class Flux:
    """An energy spectrum, the state it is in, and optionally its absolute intensity.

    Attributes:
        energy_MeV: True neutrino energy, in MeV, shaped ``(..., n_energy)`` -- the grid this flux is sampled at.
            Cast to ``state``'s own device/dtype.
        state: DensityMatrix of the flux, e.g. ``DensityMatrix.basis_state(0, space, context)`` for a source producing
            a pure electron-flavour state, or the evolved state after a medium. Its ``batch_shape`` must be
            broadcastable against ``energy_MeV``'s -- an energy-independent state (``batch_shape == ()``, the common
            case for a source with fixed flavour composition) always is.
        antinu: Bool or boolean tensor selecting the antineutrino convention this flux is made of (e.g. ``True`` for a
            reactor's electron antineutrinos). Kept separate from ``state``: matching every other antinu-aware object
            in ``tpeanuts.physics``, a ``DensityMatrix``'s own basis indices never encode this.
        differential_flux: Optional absolute intensity at each energy, non-negative, broadcastable against
            ``energy_MeV`` (shape ``(..., n_energy)``). A pure scale factor: the flavour content is ``state``'s. For a
            continuous spectrum it is per unit energy; for a line spectrum it is the integrated flux of each line --
            ``flux_units`` says which.
        flux_units: Free-text unit of ``differential_flux`` (e.g. ``"cm^-2 s^-1 MeV^-1"``). Empty if there is no
            ``differential_flux``.
        production: Optional ProductionDistribution of an extended source, needed by a medium that starts inside it.
        location: Where along the chain the flux is: ``"source"``, ``"surface"`` (leaving the source's own medium) or
            ``"detector"`` (after the last medium).
        history: The steps applied so far, as short labels, in order; a record of the provenance, not used for physics.
        direction: Optional nadir angles in radians (0 = straight down, as in ``medium.earth``), shaped
            ``(n_direction,)``, for a flux that depends on the arrival direction. ``differential_flux`` is then shaped
            ``(n_direction, n_energy)``, and the batch of the state ends in ``(n_direction, n_energy)`` once a medium
            has acted on it. None for a flux without a direction axis.

    Raises:
        ValueError: If ``energy_MeV`` has any non-positive entry, ``differential_flux`` has a negative entry or does not
            broadcast against ``energy_MeV``, ``flux_units`` is set without a ``differential_flux``, ``location`` is
            not one of the three above, or ``direction`` is not a one-dimensional set of angles in ``[0, pi]`` that
            ``differential_flux`` broadcasts against.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.interfaces import Flux
        >>> from tpeanuts.physics import DensityMatrix
        >>> electron = DensityMatrix.basis_state(0, tp.FlavourSpace(), context)
        >>> flux = Flux(energy_MeV=[1.0, 2.0, 5.0], state=electron)
        >>> flux.energy_MeV.shape, flux.antinu
        (torch.Size([3]), False)
        >>> weighted = Flux(energy_MeV=[1.0, 2.0, 5.0], state=electron, differential_flux=[3.0, 2.0, 1.0],
        ...                 flux_units="cm^-2 s^-1 MeV^-1")
        >>> weighted.per_flavour().shape   # the flux of each flavour at each energy
        torch.Size([3, 3])
    """

    energy_MeV: torch.Tensor
    state: DensityMatrix
    antinu: Union[bool, torch.Tensor] = False
    differential_flux: Optional[torch.Tensor] = None
    flux_units: str = ""
    production: Optional[ProductionDistribution] = None
    location: str = "source"
    history: Tuple[str, ...] = field(default_factory=tuple)
    direction: Optional[torch.Tensor] = None

    if TYPE_CHECKING:  # the runtime __init__ is the dataclass one; tensor fields accept any TensorLike

        def __init__(
            self,
            energy_MeV: TensorLike,
            state: DensityMatrix,
            antinu: Union[bool, torch.Tensor] = ...,
            differential_flux: Optional[TensorLike] = ...,
            flux_units: str = ...,
            production: Optional[ProductionDistribution] = ...,
            location: str = ...,
            history: Tuple[str, ...] = ...,
            direction: Optional[TensorLike] = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        device, dtype = self.state.context.device, self.state.context.dtype
        energy = as_tensor(self.energy_MeV, device=device, dtype=dtype)
        if torch.any(energy <= 0):
            raise ValueError("energy_MeV must be strictly positive everywhere.")
        object.__setattr__(self, "energy_MeV", energy)

        grid = energy
        if self.direction is not None:
            direction = as_tensor(self.direction, device=device, dtype=dtype)
            if direction.ndim != 1 or direction.numel() == 0:
                raise ValueError(f"direction must be one-dimensional and not empty, got {tuple(direction.shape)}.")
            if torch.any(direction < 0) or torch.any(direction > math.pi):
                raise ValueError("direction (a nadir angle) must lie in [0, pi].")
            object.__setattr__(self, "direction", direction)
            grid = torch.broadcast_to(energy, (direction.numel(), *energy.shape[-1:]))

        if self.differential_flux is None:
            if self.flux_units:
                raise ValueError("flux_units is set but there is no differential_flux.")
        else:
            intensity = as_tensor(self.differential_flux, device=device, dtype=dtype)
            if torch.any(intensity < 0):
                raise ValueError("differential_flux must be non-negative everywhere.")
            try:
                torch.broadcast_shapes(intensity.shape, grid.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"differential_flux (shape {tuple(intensity.shape)}) cannot be broadcast against energy_MeV "
                    f"(shape {tuple(grid.shape)}, with the direction axis if there is one)."
                ) from exc
            object.__setattr__(self, "differential_flux", intensity)

        if self.location not in LOCATIONS:
            raise ValueError(f"location must be one of {LOCATIONS}, got {self.location!r}.")
        object.__setattr__(self, "history", tuple(self.history))

    @property
    def space(self) -> FlavourSpace:
        """FlavourSpace this flux's state is expressed in (``state.space``)."""
        return self.state.space

    @property
    def context(self) -> RuntimeContext:
        """RuntimeContext this flux's tensors are stored on (``state.context``)."""
        return self.state.context

    def with_state(
        self,
        state: DensityMatrix,
        *,
        location: Optional[str] = None,
        step: Optional[str] = None,
    ) -> Flux:
        """The same flux in a new state -- the result of propagating it through a medium.

        The energy grid, intensity and production distribution are shared, not copied; the original is untouched.

        Args:
            state: The evolved DensityMatrix.
            location: New ``location``; unchanged if omitted.
            step: Label appended to ``history`` (e.g. ``"earth"``); nothing is appended if omitted.

        Returns:
            A new Flux.

        Raises:
            ValueError: If ``location`` is not a valid location, or the new state is on another device/dtype.
        """
        if state.context.device != self.context.device or state.context.dtype != self.context.dtype:
            raise ValueError("the new state must be on the same device and dtype as the flux.")
        history = self.history + ((step,) if step else ())
        return replace(self, state=state, location=self.location if location is None else location, history=history)

    def per_flavour(self) -> torch.Tensor:
        """The flux of each flavour: ``differential_flux`` times the flavour probabilities of ``state``.

        Returns:
            Real tensor shaped ``(..., n_energy, n_flavour)``.

        Raises:
            ValueError: If the flux has no ``differential_flux``.
        """
        if self.differential_flux is None:
            raise ValueError("this flux has no differential_flux, so it has no absolute flux per flavour.")
        return self.differential_flux.unsqueeze(-1) * self.state.probabilities()

    def total(self) -> torch.Tensor:
        """The flux integrated over energy (trapezoidal rule), for a continuous spectrum.

        Returns:
            Real tensor, the integral over the last (energy) axis of ``differential_flux``.

        Raises:
            ValueError: If the flux has no ``differential_flux`` or fewer than two energies.
        """
        if self.differential_flux is None:
            raise ValueError("this flux has no differential_flux, so it has no total flux.")
        if self.energy_MeV.ndim == 0 or self.energy_MeV.shape[-1] < 2:
            raise ValueError("integrating over energy needs at least two energies.")
        shape = torch.broadcast_shapes(self.energy_MeV.shape, self.differential_flux.shape)
        energy = torch.broadcast_to(self.energy_MeV, shape)
        intensity = torch.broadcast_to(self.differential_flux, energy.shape)
        return torch.trapezoid(intensity, energy, dim=-1)


@dataclass(frozen=True)
class SourceFlux:
    """The components of a source that has several, each with its own spectrum: one ``Flux`` per name.

    Examples are the eight solar production channels and the four fissile isotopes of a reactor. The components may be
    sampled on different energy grids, so they are kept apart rather than added together.

    Attributes:
        components: Mapping from a component name (``"8B"``, ``"U235"``, ...) to its Flux. Not empty.

    Raises:
        ValueError: If ``components`` is empty, or the components are not all on the same device and dtype.

    Examples:
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> from tpeanuts.interfaces import Flux, SourceFlux
        >>> from tpeanuts.physics import DensityMatrix
        >>> electron = DensityMatrix.basis_state(0, tp.FlavourSpace(), context)
        >>> source = SourceFlux({"a": Flux(energy_MeV=[1.0, 2.0], state=electron),
        ...                      "b": Flux(energy_MeV=[3.0], state=electron)})
        >>> list(source), len(source)
        (['a', 'b'], 2)
    """

    components: Dict[str, Flux]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("a SourceFlux needs at least one component.")
        contexts = {(f.context.device, f.context.dtype) for f in self.components.values()}
        if len(contexts) > 1:
            raise ValueError("all components must be on the same device and dtype.")
        object.__setattr__(self, "components", dict(self.components))

    def __iter__(self) -> Iterator[str]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, name: str) -> Flux:
        return self.components[name]

    def items(self):
        """The ``(name, Flux)`` pairs, in order."""
        return self.components.items()

    @property
    def context(self) -> RuntimeContext:
        """RuntimeContext shared by every component."""
        return next(iter(self.components.values())).context

    def map(self, function: Callable[[Flux], Flux]) -> SourceFlux:
        """Apply ``function`` (e.g. propagating through a medium) to every component, keeping the names."""
        return SourceFlux({name: function(flux) for name, flux in self.components.items()})

    def totals(self) -> Dict[str, torch.Tensor]:
        """The energy-integrated flux of every component (see ``Flux.total``)."""
        return {name: flux.total() for name, flux in self.components.items()}
