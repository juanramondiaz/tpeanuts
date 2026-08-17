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
Gradient-based inference of oscillation parameters from observatory data.

This package sits above ``tpeanuts.medium``/``tpeanuts.core`` the same way
``tpeanuts.pipeline`` does, but for the inverse problem: given observed
probabilities/rates, recover the oscillation parameters that best reproduce
them, using PyTorch autograd rather than a derivative-free optimizer.

Medium/detector-agnostic by design: this package carries no medium or
detector concept of its own. Every concrete differentiable model lives with
the physics it wraps instead --
``tpeanuts.inference.solar_model``/``tpeanuts.medium.vacuum
.oscillation_model`` for medium-level P_ee models, and each
``tpeanuts.detector.<name>.inference_model`` for the detector-composition
layer built on top -- and only needs to satisfy ``model.DifferentiableModel``
(a ``free`` tuple and a single-argument ``predict(theta)``) to be usable by
every function here.

Package modules:
    model
        DifferentiableModel: the structural contract every fittable model
        satisfies.
    likelihood
        Asymmetric-Gaussian chi-square and Poisson likelihood-ratio
        statistics used to compare predictions against observed data.
    fit
        Gradient-based point estimate (torch.optim.LBFGS) plus a
        Laplace/Fisher uncertainty estimate from the autograd Hessian at
        the optimum.
    scan
        2-D -2 ln L grid scans and Monte-Carlo threshold calibration.
"""
