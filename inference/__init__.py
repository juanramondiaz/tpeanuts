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

Medium/detector-agnostic by design: the medium-level P_ee models --
``tpeanuts.inference.model_solar``, ``tpeanuts.inference.model_vacuum``,
``tpeanuts.inference.model_atmosphere``, one per propagation medium -- live
here directly, while each ``tpeanuts.detector.<name>.inference_model``
composes the relevant one for the detector-composition layer built on top.
Every such model, medium-level or detector-composed, only needs to satisfy
``model.DifferentiableModel`` (a ``free`` tuple and a single-argument
``predict(theta)``) to be usable by every function here.

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
