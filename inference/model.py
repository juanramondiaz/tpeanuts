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

``DifferentiableModel`` is the entire interface ``tpeanuts.inference.fit
.fit_lbfgs``/``tpeanuts.inference.scan.loglik_grid`` require: a free-
parameter name tuple and a single-argument ``predict(theta)``. It carries
no medium or detector concept -- every concrete model lives with the
physics it wraps instead:

    inference.solar_model
        SolarSMOscillationModel, SolarNSIOscillationModel, SolarPointModel
        -- solar P_ee(E), shared by any solar-fed detector.
    medium.vacuum.oscillation_model
        VacuumOscillationModel -- reactor P_ee(E, L) in vacuum, shared by
        any reactor experiment.
    detector.borexino.inference_model, detector.sno.inference_model,
    detector.dayabay.inference_model
        Detector-composition layers wrapping the medium models above with
        each detector's own target/response/binning, already implementing
        ``predict(theta)`` directly.

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
