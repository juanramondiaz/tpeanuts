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

Models the real Daya Bay Collaboration experiment -- 6 reactor cores and 8
antineutrino detectors at their published baselines -- and fits it to Daya
Bay's own published IBD prompt-energy spectra and backgrounds.

Notes:
    - Data comes from the official Daya Bay "Full Data Release" (Zenodo DOI
      10.5281/zenodo.17587229; An et al., PRL 130, 161802 (2023)), cached
      under ``data/detector/dayabay/``.
    - Modeled at full realism: baselines, per-isotope Huber-Mueller flux,
      fission fractions/thermal power, observed spectra and backgrounds,
      target protons, exposure, the energy-resolution/IAV/LSNL response
      chain, non-equilibrium/SNF flux corrections, the per-reactor relative
      power weight, Daya Bay's own order-1/M IBD cross section, and its own
      free ``global_normalization`` nuisance.
    - Not modeled: the LSNL/IAV/background pull-curve systematics or a full
      correlated covariance (a plain Poisson fit is used instead, with
      ``global_normalization`` as the only normalization nuisance); and the
      6AD/7AD periods (only the 8AD period is modeled).

Package modules:
    parameters
        Constants loaded/derived at import time: names, fission fractions,
        thermal power, resolution/binning, cross-section and response
        constants.
    io
        Loaders for the per-detector data release files.
    flux
        Per-isotope Huber-Mueller spectrum and per-reactor face flux.
    response
        The IAV -> LSNL -> Gaussian-resolution detector-response chain.
    event_rate
        The multi-reactor IBD fold into a predicted event-rate spectrum.
    inference_model
        The detector-composition layer: ``DayaBayDetectorModel`` and
        ``JointDayaBayModel``, wrapping ``VacuumOscillationModel``.
"""
