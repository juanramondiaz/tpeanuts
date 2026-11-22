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

"""Fitting parameters against observed data via autograd.

``tpeanuts.inference.model.DifferentiableModel`` is the only structural
contract this package requires from a model: a ``free`` parameter-name
tuple and a single-argument ``predict(theta) -> Tensor``. Every
model-specific input (source, medium, detector, a fixed energy grid, ...)
is bound as a field on the concrete model itself (e.g.
``tpeanuts.inference.model_reactor.ReactorEventRateModel``); this package's
own code -- ``tpeanuts.inference.likelihood``,
``tpeanuts.inference.fit`` -- never imports ``tpeanuts.source``/
``tpeanuts.medium``/``tpeanuts.detector``/``tpeanuts.physics`` directly,
only ever calling ``model.predict(theta)`` and reading ``model.free``.

``tpeanuts.inference.fit.fit`` is the single-point fitting mechanism
(gradient-based point estimate via a caller-selected, typed
``tpeanuts.inference.fit.OptimizerConfig`` -- ``"lbfgs"`` by default,
``"adam"`` also validated, or a custom dataclass for any other
``torch.optim.Optimizer`` -- plus a Laplace/Fisher covariance from the
autograd Hessian at the optimum). ``tpeanuts.inference.scan`` builds a
confidence *region* on top of it instead of a single point: a
profile-likelihood ``(param_x, param_y)`` grid
(``scan.loglik_grid``), plus a parametric-bootstrap calibration of its
own confidence-level threshold when Wilks' theorem's asymptotic
normality is not trusted at a given exposure/statistics level
(``scan.calibrate_delta_threshold``) -- both model-agnostic, applying
unchanged to any already-built ``DifferentiableModel``.
"""

from __future__ import annotations

__all__: list = []  # experimental in 1.0: no stability guarantee, import from the submodules
