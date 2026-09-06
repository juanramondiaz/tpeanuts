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
Structural contract shared by every fittable model.

``DifferentiableModel`` is a ``typing.Protocol``: any object exposing a
``free`` parameter-name tuple and a single-argument ``predict(theta)``
satisfies it, with no need to inherit from it. ``fit.fit_lbfgs`` and
``scan.loglik_grid`` require nothing else from a model.

Module contents:
    DifferentiableModel
        The structural type ``fit_lbfgs``/``loglik_grid`` require.
"""

from __future__ import annotations

from typing import Protocol

import torch


class DifferentiableModel(Protocol):
    """Structural type every model fit by ``tpeanuts.inference`` must satisfy.

    ``fit_lbfgs``/``loglik_grid``/``calibrate_delta_threshold`` only ever
    call ``predict`` and read ``free``; this Protocol documents that
    minimal surface instead of hardcoding one concrete model class into the
    fitting code.
    """

    free: tuple[str, ...]

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        ...
