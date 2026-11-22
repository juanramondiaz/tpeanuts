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

"""``solar_flux``: the solar scenario, from production in the Sun to the surface of the Sun or the detector.

A solar neutrino is born as an electron neutrino at some radius of the Sun, leaves it as an incoherent mixture of mass
eigenstates, and, for a detector underground, crosses the Earth. The scenario is ``source.flux()`` followed by the Sun
(``SolarLeg``) and, optionally, the Earth (``EarthLeg``), channel by channel::

    arrived = solar_flux(model, source, sun, ["pp", "8B"], energies, earth=EarthLeg(depth_m=2070.0, exposure=night))
    arrived["8B"].state.probabilities()        # (n_E, n_flavour) at the detector, averaged over the exposure
    detector.observe(arrived["8B"])            # the event rate

The result is a ``SourceFlux`` with one ``Flux`` per channel (the channels keep their own energy grids: a line channel
has its own energies). Every algorithm lives below this layer: the adiabatic weights, the Landau-Zener correction and
the shared-history numerical solver in ``tpeanuts.physics.evolution``, the radial average in
``SolarMedium.propagate`` (which computes the radial weights once for all the channels that share an energy grid
and production radius), the Earth chord in ``EarthMedium``. The exposure average is the one piece of this layer that
needs the model, and lives in ``tpeanuts.pipeline._legs``.

Module contents:
    solar_flux(...)
        The solar scenario: ``source.flux(...)`` through the Sun and, optionally, the Earth: a ``SourceFlux`` with one
        ``Flux`` per channel at the surface of the Sun or at the detector.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

from tpeanuts.interfaces.flux import SourceFlux
from tpeanuts.medium.solar.solar import SolarMedium
from tpeanuts.physics.model import PropagationModel
from tpeanuts.pipeline._legs import EarthLeg, SolarLeg
from tpeanuts.source.solar.solar import SolarSource
from tpeanuts.util.tensor import TensorLike


def solar_flux(
    model: PropagationModel,
    source: SolarSource,
    sun: SolarMedium,
    sources: Union[str, Sequence[str]],
    energy_MeV: Optional[TensorLike] = None,
    *,
    solar_leg: Optional[SolarLeg] = None,
    earth: Optional[EarthLeg] = None,
    intensity: bool = True,
) -> SourceFlux:
    """The solar scenario: the chosen channels, from their production in the Sun to the surface or the detector.

    ``source.flux(sources, energy_MeV)`` followed by the Sun (``solar_leg.propagate``) and, if ``earth`` is given, the
    Earth (``earth.propagate``); nothing else. The probabilities are ``flux.state.probabilities()`` of each
    component, and the rate is ``detector.observe(flux)``.

    Args:
        model: Built PropagationModel (a neutrino one: the Sun emits neutrinos).
        source: The solar source (production distributions, fluxes and spectra).
        sun: The solar medium (density profile).
        sources: Channel key or ordered keys (``"pp"``, ``"8B"``, ``"7Be"``, ...).
        energy_MeV: Energy grid in MeV for the continuous channels (a line channel keeps its own energies, unless
            ``intensity=False``); None uses the source's tabulated grid.
        solar_leg: Method, Landau-Zener flag and sampling of the solar leg; defaults to ``SolarLeg()``.
        earth: The Earth leg, with one nadir angle (``eta_rad``), an exposure table (``exposure``, which averages the
            Earth's effect over the detector's day or night) and the detector depth. None stops at the solar surface.
        intensity: Fill each flux's ``differential_flux`` (default). With False the components carry only the energy
            grid, as given, and their production: the probabilities at chosen energies, with no flux or spectrum
            table needed.

    Returns:
        SourceFlux with one component per channel: ``location="surface"`` and ``history=("sun",)``, or
        ``location="detector"`` and ``history=("sun", "earth")`` with ``earth``. With an exposure table the state is
        the incoherent one of the exposure-averaged probabilities.

    Raises:
        ValueError: From ``SolarSource.flux`` (``energy_MeV`` missing with ``intensity=False``),
            ``SolarMedium.propagate`` (method against model, e.g. NSI with an approximated adiabatic method)
            or ``EarthLeg.propagate`` (angle outside its domain at this depth).
        KeyError: For a channel the source does not know.

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
        >>> energies = torch.tensor([1.0, 10.0], dtype=torch.float64)   # MeV
        >>> from tpeanuts.pipeline.solar import solar_flux
        >>> arrived = solar_flux(model, source, sun, ["8B"], energies)
        >>> arrived["8B"].state.probabilities().shape   # (energy, flavour) at the solar surface
        torch.Size([2, 3])
    """
    emitted = source.flux(sources, energy_MeV, intensity=intensity)
    at_surface = (solar_leg or SolarLeg()).propagate(emitted, sun, model)
    return at_surface if earth is None else earth.propagate(at_surface, model)
