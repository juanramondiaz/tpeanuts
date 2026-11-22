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

"""The options of each leg of a scenario, and the exposure average over the Earth leg.

A scenario pipeline (``solar_flux``, ``atmospheric_flux``) crosses up to three legs -- the Sun, the atmosphere, the
Earth -- and each has a handful of choices (the propagation method, the chord discretisation, the detector depth, which
nadir angle or exposure table). They are carried here as small immutable dataclasses, so that a call reads
``solar_flux(model, source, sun, ["8B"], energy, earth=EarthLeg(depth_m=2070.0, exposure=table))`` and the options of
one leg never leak into another's keyword arguments.

A leg only *selects* the medium and its options; every piece of physics stays in ``tpeanuts.medium`` and
``tpeanuts.physics.evolution``. The one thing a leg does that no medium can is the **exposure average**: a detector
that sees the Sun at every nadir angle during the day or the night averages the Earth's effect over a
``NadirExposureTable``, and the average needs the model (``PropagationModel``), which ``medium`` may not import. It is
written once here (``exposure_average``) and used by ``EarthLeg``.

Module contents:
    EarthLeg
        Detector depth, method, chord steps, and either one nadir angle (``eta_rad``), an exposure table (``exposure``)
        or neither (the angles are then the flux's own directions); ``propagate`` crosses the Earth with them.
    SolarLeg
        Method, Landau-Zener flag and sampling of the solar leg; ``propagate`` carries a flux to the solar surface.
    AtmosphereLeg
        Method and discretisation of the atmosphere leg; ``medium`` builds the ``AtmosphereMedium`` of a flux.
    exposure_average(...)
        Average the flavour probabilities that a medium gives at each angle of an exposure table, weighted by the
        exposure, and return the flux in the incoherent state those probabilities define.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import torch

from tpeanuts.config import default
from tpeanuts.interfaces.flux import Flux, SourceFlux
from tpeanuts.medium.atmosphere.atmosphere import AtmosphereMedium
from tpeanuts.medium.earth.earth import EarthMedium
from tpeanuts.medium.earth.exposure.table import NadirExposureTable, integrate_exposure
from tpeanuts.medium.solar.solar import SolarMedium
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.tensor import TensorLike

_F = TypeVar("_F", Flux, SourceFlux)
_SOLAR_METHODS = ("adiabatic_approximated", "adiabatic_exact", "numerical")
_SAMPLINGS = ("midpoint", "endpoint")


def exposure_average(
    flux: Flux,
    model: PropagationModel,
    table: NadirExposureTable,
    medium_at: Callable[[torch.Tensor], Any],
    **propagate_options: Any,
) -> Flux:
    """Average the probabilities of a medium over the nadir-angle exposure of a detector.

    For each nadir angle of ``table`` the flux is propagated through ``medium_at(eta_rad)``; the flavour probabilities
    of the results are averaged with the exposure weights (``integrate_exposure``) and the flux is returned in the
    incoherent state those probabilities define (``DensityMatrix.from_weights``). Anything that is linear in the
    probabilities -- an event rate, a count -- can be computed from the result as if it were averaged over the angles.

    Args:
        flux: The flux entering the medium (a ``Flux``; see ``EarthLeg.propagate`` for a ``SourceFlux``).
        model: Built PropagationModel.
        table: Already-built exposure table; it is expensive and independent of the scenario, so the caller builds it
            once.
        medium_at: ``eta_rad -> medium``: the medium for one scalar nadir angle (anything with
            ``propagate(flux, model, **options)``).
        **propagate_options: Forwarded to every ``propagate`` call (e.g. ``P``, ``P_nc``).

    Returns:
        A Flux like the propagated ones (same ``location`` and ``history``), whose state is diagonal and whose
        probabilities are the exposure average, shaped like one angle's.
    """
    propagated = [medium_at(eta_i).propagate(flux, model, **propagate_options) for eta_i in table.eta_rad]
    probabilities = torch.stack([item.state.probabilities() for item in propagated])
    average = integrate_exposure(probabilities, table.eta_rad, table.exposure)
    state = DensityMatrix.from_weights(average, model.space, flux.context)
    return propagated[0].with_state(state)


@dataclass(frozen=True)
class EarthLeg:
    """How the neutrinos cross the Earth to the detector.

    Attributes:
        depth_m: Detector depth below the surface, in metres.
        method: ``PropagationMethod`` that builds the evolution operator (see ``EarthMedium``).
        nsteps: Chord discretisation for ``NUMERICAL``/``ANALYTIC`` (ignored otherwise).
        eta_rad: One nadir angle (or a batch), in radians. Exclusive with ``exposure``.
        exposure: A ``NadirExposureTable``: the result is averaged over its angles. Exclusive with ``eta_rad``.
        P: Optional matter-potential direction matrix for ``ANALYTIC_PERTURBATIVE`` with extensions, forwarded to
            the medium unchanged.
        P_nc: Optional neutral-current direction matrix, forwarded unchanged.

    With neither ``eta_rad`` nor ``exposure`` the angles are the flux's own directions (an atmospheric flux).

    Raises:
        ValueError: If ``nsteps`` is not positive, or both ``eta_rad`` and ``exposure`` are given.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> space = tp.FlavourSpace()
        >>> params = tp.StandardMixingParameters(
        ...     theta12=math.radians(33.4), theta13=math.radians(8.6), theta23=math.radians(42.0),
        ...     delta13=math.radians(230.0), context=context)
        >>> mixing = tp.MixingMatrix(space=space, standard=params, nature=tp.Dirac(), context=context)
        >>> masses = tp.MassSpectrum(space=space, DeltamSq21=7.4e-5, DeltamSq3l=2.5e-3, context=context)
        >>> model = tp.PropagationModel(mixing=mixing, mass_spectrum=masses)
        >>> from tpeanuts.interfaces import Flux
        >>> from tpeanuts.physics import DensityMatrix
        >>> from tpeanuts.pipeline import EarthLeg
        >>> muon = DensityMatrix.basis_state(1, space, context)
        >>> flux = Flux(energy_MeV=[1000.0, 5000.0], state=muon)   # a muon-neutrino beam
        >>> leg = EarthLeg(depth_m=2070.0, eta_rad=0.5, nsteps=20)   # one nadir angle, a detector 2070 m down
        >>> leg.propagate(flux, model).state.probabilities().shape
        torch.Size([2, 3])
    """

    depth_m: TensorLike = 0.0
    method: PropagationMethod = PropagationMethod.NUMERICAL
    nsteps: int = default.EARTH_NSTEPS
    eta_rad: Optional[TensorLike] = None
    exposure: Optional[NadirExposureTable] = None
    P: Optional[torch.Tensor] = None
    P_nc: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.nsteps <= 0:
            raise ValueError(f"nsteps must be strictly positive, got {self.nsteps!r}.")
        if self.eta_rad is not None and self.exposure is not None:
            raise ValueError("give either eta_rad (one nadir angle) or exposure (an average over angles), not both.")

    def medium(self, flux: Flux, eta_rad: Optional[TensorLike] = None) -> EarthMedium:
        """The ``EarthMedium`` of this leg for ``flux``: at ``eta_rad`` if given, else this leg's ``eta_rad``, else the
        flux's.

        Args:
            flux: The flux about to cross the Earth (its context, and its directions if no angle is given).
            eta_rad: A nadir angle overriding this leg's (the exposure average passes each table angle).

        Returns:
            The medium, on the flux's context.

        Raises:
            ValueError: If there is no angle at all and the flux has no direction axis.
        """
        angle = self.eta_rad if eta_rad is None else eta_rad
        options = dict(depth_m=self.depth_m, method=self.method, nsteps=self.nsteps)
        if angle is None:
            return EarthMedium.from_flux(flux, **options)
        return EarthMedium(eta_rad=angle, context=flux.context, **options)

    def propagate(self, flux: _F, model: PropagationModel) -> _F:
        """Cross the Earth: at this leg's angle, along the flux's directions, or averaged over its exposure table.

        A ``SourceFlux`` is carried component by component.

        Args:
            flux: The flux at the Earth's surface (or at the source, for an atmospheric one past its own leg).
            model: Built PropagationModel.

        Returns:
            The flux at the detector; with an exposure table, in the incoherent state of the averaged probabilities.

        Raises:
            ValueError: As ``EarthMedium`` (angle outside its domain at this depth, model and flux disagreeing).
        """
        if isinstance(flux, SourceFlux):
            return flux.map(lambda component: self.propagate(component, model))
        options: Dict[str, Any] = dict(P=self.P, P_nc=self.P_nc)
        if self.exposure is not None:
            return exposure_average(
                flux, model, self.exposure, lambda eta_i: self.medium(flux, eta_i), **options,
            )
        return self.medium(flux).propagate(flux, model, **options)


@dataclass(frozen=True)
class SolarLeg:
    """How the neutrinos cross the Sun (see ``SolarMedium.propagate``).

    Attributes:
        method: ``"adiabatic_approximated"``, ``"adiabatic_exact"`` or ``"numerical"``.
        use_LZ: Add the Landau-Zener non-adiabatic correction (adiabatic methods only).
        numerical_sampling: ``"midpoint"`` or ``"endpoint"`` radial sampling of the numerical solver.

    Raises:
        ValueError: If ``method`` or ``numerical_sampling`` is not one of the above.

    Examples:
        >>> import math
        >>> import torch
        >>> import tpeanuts as tp
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> space = tp.FlavourSpace()
        >>> params = tp.StandardMixingParameters(
        ...     theta12=math.radians(33.4), theta13=math.radians(8.6), theta23=math.radians(42.0),
        ...     delta13=math.radians(230.0), context=context)
        >>> mixing = tp.MixingMatrix(space=space, standard=params, nature=tp.Dirac(), context=context)
        >>> masses = tp.MassSpectrum(space=space, DeltamSq21=7.4e-5, DeltamSq3l=2.5e-3, context=context)
        >>> model = tp.PropagationModel(mixing=mixing, mass_spectrum=masses)
        >>> sun = tp.SolarMedium.default(context=context)
        >>> source = tp.SolarSource.default(context=context)
        >>> from tpeanuts.pipeline import SolarLeg
        >>> emitted = source.flux(["8B"], torch.tensor([5.0], dtype=torch.float64))
        >>> at_surface = SolarLeg(method="adiabatic_exact").propagate(emitted, sun, model)
        >>> at_surface["8B"].history, at_surface["8B"].location
        (('sun',), 'surface')
    """

    method: str = "adiabatic_approximated"
    use_LZ: bool = False
    numerical_sampling: str = "midpoint"

    def __post_init__(self) -> None:
        if self.method not in _SOLAR_METHODS:
            raise ValueError(f"method must be one of {_SOLAR_METHODS}, got {self.method!r}.")
        if self.numerical_sampling not in _SAMPLINGS:
            raise ValueError(f"numerical_sampling must be one of {_SAMPLINGS}, got {self.numerical_sampling!r}.")

    def propagate(self, flux: _F, sun: SolarMedium, model: PropagationModel) -> _F:
        """Carry a flux produced inside the Sun to its surface.

        Args:
            flux: A Flux (or SourceFlux) with a ``production`` distribution, as the solar source emits.
            sun: The solar medium.
            model: Built PropagationModel.

        Returns:
            The flux at the solar surface.
        """
        return sun.propagate(
            flux, model, method=self.method, use_LZ=self.use_LZ, numerical_sampling=self.numerical_sampling,
        )


@dataclass(frozen=True)
class AtmosphereLeg:
    """How the neutrinos cross the atmosphere from their production height to the surface.

    Attributes:
        method: ``PropagationMethod`` that builds the evolution operator (see ``AtmosphereMedium``).
        nsteps: Chord discretisation for ``NUMERICAL``/``ANALYTIC``.
        perturbative_segments: Segment count of ``ANALYTIC_PERTURBATIVE`` only.
        perturbative_degree: Local polynomial degree of ``ANALYTIC_PERTURBATIVE`` only.
        density_source: Density model of the atmosphere, forwarded to ``AtmosphereMedium``.
        density_kwargs: Extra keywords of that density model.

    Raises:
        ValueError: If ``nsteps``, ``perturbative_segments`` or ``perturbative_degree`` is not positive.

    Examples:
        >>> from tpeanuts.pipeline import AtmosphereLeg
        >>> leg = AtmosphereLeg(nsteps=40, density_kwargs={"rho0_gcm3": 0.0})   # the same path without air
        >>> leg.nsteps, leg.method.name
        (40, 'NUMERICAL')
    """

    method: PropagationMethod = PropagationMethod.NUMERICAL
    nsteps: int = default.ATMOSPHERE_NSTEPS
    perturbative_segments: int = default.ATMOSPHERE_PERTURBATIVE_SEGMENTS
    perturbative_degree: int = default.ATMOSPHERE_PERTURBATIVE_DEGREE
    density_source: str = default.ATMOSPHERE_SOURCE_DENSITY
    density_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        for name in ("nsteps", "perturbative_segments", "perturbative_degree"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be strictly positive, got {getattr(self, name)!r}.")

    def medium(self, flux: Flux, depth_m: Union[float, torch.Tensor] = 0.0) -> AtmosphereMedium:
        """The ``AtmosphereMedium`` that carries ``flux`` (a direction and a production height) to the surface.

        Args:
            flux: A Flux with ``direction`` and ``production``, e.g. a component of the atmospheric source's flux.
            depth_m: Detector depth below the surface, in metres.

        Returns:
            The medium, on the flux's context.

        Raises:
            ValueError: As ``AtmosphereMedium.from_flux``.
        """
        return AtmosphereMedium.from_flux(
            flux, depth_m=depth_m, method=self.method, nsteps=self.nsteps,
            perturbative_segments=self.perturbative_segments, perturbative_degree=self.perturbative_degree,
            density_source=self.density_source, density_kwargs=self.density_kwargs or {},
        )
