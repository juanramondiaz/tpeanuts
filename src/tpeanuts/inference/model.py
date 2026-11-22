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

"""Structural contract shared by every fittable model.

``DifferentiableModel`` is a ``typing.Protocol``: any object exposing a
``free`` parameter-name tuple and a single-argument ``predict(theta)``
satisfies it, with no need to inherit from it.
``tpeanuts.inference.fit.fit`` requires nothing else from a model.

Module contents:
    DifferentiableModel
        The structural type ``fit`` requires.
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class DifferentiableModel(Protocol):
    """Structural type every model fit by ``tpeanuts.inference`` must satisfy.

    No inheritance required -- any object exposing these two members
    satisfies it (``isinstance(obj, DifferentiableModel)`` works, via
    ``runtime_checkable``, though it only checks that the members exist,
    not their exact signatures). ``fit`` only ever calls ``predict``
    and reads ``free``; this Protocol documents that minimal surface
    instead of hardcoding one concrete model class into the fitting code.

    Attributes:
        free: Names of this model's free parameters, in the same order
            ``predict``'s ``theta`` argument expects them.
    """

    @property
    def free(self) -> Tuple[str, ...]:
        """Names of this model's free parameters (read-only: a stored field or a computed property both satisfy this).
        """
        ...

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """Return this model's prediction at the given free-parameter vector.

        Args:
            theta: Free-parameter vector, shaped ``(len(self.free),)``,
                same order as ``self.free``.

        Returns:
            Prediction tensor, differentiable w.r.t. ``theta``.
        """
        ...
