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

"""``propagate_chain``: apply several media to one flux, in order.

Sugar over ``medium.propagate``: ``propagate_chain(flux, [sun, vacuum, earth], model)`` is exactly
``earth.propagate(vacuum.propagate(sun.propagate(flux, model), model), model)``, and nothing more. It implements no
physics (that is each medium's and ``tpeanuts.physics.evolution``'s job); it only fixes the order and lets each leg
carry its own options.

A step is one of:
    * a medium (anything with a ``propagate(flux, model, **options)`` method);
    * a ``(medium, options)`` pair, ``options`` being the keyword arguments forwarded to that medium's ``propagate``;
    * a callable ``flux -> medium`` (or ``flux -> (medium, options)``), for a leg that can only be built from the flux
      it receives, such as ``AtmosphereMedium.from_flux``/``EarthMedium.from_flux`` (which read the direction axis
      and the production altitudes off the flux).

Module contents:
    propagate_chain(...)
        Apply the steps to the flux in order and return the final ``Flux`` (or ``SourceFlux``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple, TypeVar

from tpeanuts.physics.model import PropagationModel

_F = TypeVar("_F")


def _resolve(step: Any, flux: Any) -> Tuple[Any, Mapping[str, Any]]:
    """Return ``(medium, options)`` for one step, building the medium from ``flux`` if the step is a factory."""
    if not hasattr(step, "propagate") and callable(step):
        step = step(flux)
    if isinstance(step, tuple):
        if len(step) != 2 or not isinstance(step[1], Mapping):
            raise ValueError("a (medium, options) step must be a pair whose second item is a mapping of keywords")
        medium, options = step
    else:
        medium, options = step, {}
    if not hasattr(medium, "propagate"):
        raise ValueError(f"a chain step must provide propagate(flux, model); got {type(medium).__name__}")
    return medium, options


def propagate_chain(flux: _F, steps: Sequence[Any], model: PropagationModel) -> _F:
    """Propagate ``flux`` through ``steps`` in order.

    Args:
        flux: A Flux (or a SourceFlux) in the state the first step expects.
        steps: The media to cross, nearest to the source first; see the module docstring for the accepted forms.
        model: Built PropagationModel, shared by every step.

    Returns:
        The flux after the last step; its ``history`` lists every step's own entry in order.

    Raises:
        ValueError: If ``steps`` is empty or a step is not a medium, a ``(medium, options)`` pair or a factory of
            one; or as the media's own ``propagate``.
    """
    if len(steps) == 0:
        raise ValueError("propagate_chain needs at least one step")
    for step in steps:
        medium, options = _resolve(step, flux)
        flux = medium.propagate(flux, model, **options)
    return flux
