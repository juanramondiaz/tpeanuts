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

"""Geometry shared by more than one medium (today: ``medium.earth`` and ``medium.atmosphere``).

Populated incrementally, matching ``tpeanuts.util.constant``'s own stated
policy -- today that is exactly the spherical-shell chord-length formula
both media need (a straight line through two concentric spheres, seen
from a point on or inside the inner one), factored out once a second
consumer (``medium.atmosphere``) needed the exact same algebra
``medium.earth.geometry.earth_chord_length_m`` already had, rather than
reimplementing it a second time (v1_0 itself has two independent copies
of this formula, one per medium -- confirmed by direct comparison, not
assumed).

Module contents:
    chord_length_to_radius_fraction(...)
        Length of the straight chord from a point at radius fraction
        ``r_start_fraction``, in the direction ``eta_rad`` (nadir angle) from
        that point's own local vertical, to where it crosses radius
        fraction ``r_target_fraction``.
"""

from __future__ import annotations
