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
MCEq-based Atmosphere flux generation utilities.

This package is the tpeanuts wrapper layer around the external MCEq
(Matrix Cascade Equation) package and the crflux primary cosmic-ray flux
models. MCEq numerically solves the coupled cascade equations that
describe how a primary cosmic-ray spectrum, injected at the top of the
atmosphere, propagates and produces secondary hadrons, muons and
neutrinos as a function of energy E (GeV) and atmospheric slant depth X
(g/cm^2, the column density of air integrated along the particle's
path). tpeanuts itself does not reimplement this transport physics; it
only configures MCEq runs (interaction model, primary flux model,
density/atmosphere model, energy and depth grids), drives MCEq's solver,
and converts its outputs into torch tensors suitable for the rest of the
pipeline (height-differential production profiles f(h | E, alpha) and
fluxes Phi(E, h, alpha) as used downstream in oscillation-in-matter
calculations).

Submodules:
    config: Dataclasses and constants enumerating the available MCEq
        interaction models, crflux primary cosmic-ray models, and
        density/atmosphere models, plus grid/smoothing/output
        configuration objects. Pure tpeanuts-native configuration; no
        MCEq call is made here.
    core: Thin wrapper that constructs and configures an MCEqRun
        instance (the external MCEq solver object) for a given zenith
        angle and physics-model selection.
    solver: Direct calls into an MCEqRun instance to solve the cascade
        equations and extract the differential flux Phi(E, X, alpha) on
        a depth grid or at a single observation depth.
    density: Wraps the MCEq atmosphere density-model object to evaluate
        mass density rho(h) and mass overburden X(h) at given altitudes.
    depth: Conversions between altitude h (km) and atmospheric slant
        depth X (g/cm^2) along a line of sight at zenith angle theta,
        built on top of the MCEq density model.
    smoothing: Tensor-only post-processing (smoothing and depth
        derivatives dPhi/dX) applied to MCEq flux output; does not call
        MCEq itself.
    profiles: Combines the solver, depth and smoothing utilities to
        reconstruct height-dependent production profiles f(h | E, alpha)
        from the depth-differential flux gradient.
    generator: High-level, per-particle/per-angle orchestration that
        produces and saves height-differential flux files from the MCEq
        backend.
    diagnostics: Tensor-only consistency checks (normalization, NaN/Inf,
        monotonicity, flux-reconstruction accuracy) for the datasets
        produced by this package; does not call MCEq.
"""



