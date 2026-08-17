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
Daya Bay-specific detector wiring: real 6-reactor / 8-detector IBD geometry.

This package models the real Daya Bay Collaboration experiment: 6 reactor
cores (R1-R6) and 8 antineutrino detectors (AD11/12, AD21/22,
AD31/32/33/34) at their real published baselines, a real per-isotope
Huber-Mueller antineutrino flux, and a fit to Daya Bay's own real published
IBD prompt-energy spectra and backgrounds.

Data source: the official Daya Bay Collaboration "Full Data Release"
(Zenodo DOI 10.5281/zenodo.17587229, supplementary to F. P. An et al.,
Phys. Rev. Lett. 130, 161802 (2023)), analysis-dataset (TSV) tier, fetched
by ``notebooks/external/dayabay/DayaBay1_generator.ipynb`` from the
official GitHub mirror (``zenodo.org`` itself is Cloudflare-blocked from
this project's development sandbox) into ``data/detector/dayabay/``.

**Scope -- real vs. explicitly deferred.** This package uses the data
release's real baselines, real per-isotope Huber-Mueller flux, real
time-averaged fission fractions/thermal power/energy-per-fission, real
observed IBD spectra and backgrounds, real target proton counts, real
exposure, the real 3-term energy-resolution formula, the real IAV
energy-redistribution matrix and LSNL nonlinearity curve, the real
non-equilibrium and spent-nuclear-fuel (SNF) flux corrections, a real
per-reactor relative power weight derived from the real weekly antineutrino-
rate history, Daya Bay's own real order-1/M multi-parameter IBD cross
section (``ibd_constants.yaml``'s f/g/f2, via
``tpeanuts.detector.interaction.inverse_beta_decay
.ibd_cross_section_grid_precise``), and Daya Bay's own real free
``global_normalization`` nuisance parameter. It does **not** reproduce: the
LSNL/IAV/background pull-curve systematic uncertainties or the full
correlated systematic/covariance treatment (a plain Poisson point-estimate
fit is used instead, via ``tpeanuts.inference.likelihood``/``fit``,
unchanged, with ``global_normalization`` as the one explicit normalization
nuisance); or the 6AD/7AD data-taking periods (only the 8AD period -- all 8
detectors online -- is modeled).

Package modules:
    parameters
        Constants loaded/derived at import time: detector/reactor names,
        real fission fractions, thermal power, energy-per-fission, real
        energy-resolution parameters, real analysis bin edges, the true/
        reconstructed energy grids, real IBD cross-section constants, the
        real IAV matrix and LSNL curve, real non-equilibrium/SNF
        corrections, the real per-reactor relative weight, and the real
        nominal global normalization.
    io
        Loaders for the per-detector real data (IBD spectra, backgrounds,
        exposure, baselines, target protons, Huber-Mueller spectra, IBD
        constants, IAV matrix, LSNL curve, non-equilibrium/SNF corrections,
        weekly reactor rate, global normalization).
    flux
        Real per-isotope Huber-Mueller spectrum (with the real non-
        equilibrium correction folded in) and the real per-reactor face
        flux (real SNF correction and relative power weight applied).
    response
        The real IAV -> LSNL -> Gaussian-resolution response chain.
    event_rate
        The multi-reactor IBD fold: sums each detector's 6 real reactor
        baselines (oscillated individually) before folding with the real
        order-1/M IBD cross section, real response, real target protons,
        and real background, with an optional real signal-normalization
        scale.
    inference_model
        The detector-composition layer: ``DayaBayDetectorModel`` and
        ``JointDayaBayModel``, wrapping
        ``tpeanuts.medium.vacuum.oscillation_model.VacuumOscillationModel``
        (vacuum, antinu=True) and Daya Bay's own real
        ``global_normalization`` nuisance.
"""
