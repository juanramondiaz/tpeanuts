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

"""A real Earth medium: a PREM density chord, for a detector at any depth.

Supports all three ``tpeanuts.physics.evolution.PropagationMethod``
values (``NUMERICAL``/``ANALYTIC``/``ANALYTIC_PERTURBATIVE``), a detector
at any depth (not just the surface), and (``exposure`` subpackage)
nadir-angle exposure-weighted time-averaging for a detector at fixed
latitude -- see ``tpeanuts.medium.earth.earth``'s and
``tpeanuts.medium.earth.exposure``'s own module docstrings.

Module contents:
    ``tpeanuts.medium.earth.density``
        PREM electron/neutron density table and linear-in-radius
        interpolation.
    ``tpeanuts.medium.earth.even_power``
        The coarser, shell-aligned even-power density fit
        ``ANALYTIC_PERTURBATIVE`` uses in place of ``density``'s own
        direct interpolation.
    ``tpeanuts.medium.earth.geometry``
        Chord geometry through a spherical Earth, for a detector at any
        depth.
    ``tpeanuts.medium.earth.earth``
        ``EarthMedium``, the ``MediumProfile`` producer.
    ``tpeanuts.medium.earth.exposure``
        Nadir-angle exposure (time-averaging) for a detector at fixed
        latitude.
"""

from __future__ import annotations
