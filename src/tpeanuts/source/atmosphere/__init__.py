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

"""Atmospheric-neutrino production data: tabulated flux, generators, and the source-side domain model.

The flat (no production-height axis) Honda/HKKM surface-flux table
(``io.py``), two independent generators that each *build* a
height-differential ``Phi(E, h, angle)`` table from scratch and save it
in the same per-angle CSV schema ``io.py`` already reads (via
``io.write_angle_flux_csv``) -- no separate loader needed for either
generator's output, and no format distinction between them once
written to disk: ``mceq.py`` solves the atmospheric cascade equations
live with the external MCEq package, ``honda.py`` reads and interpolates
the external Honda/HKKM ``.d.gz`` tables instead -- and ``model.py``,
which turns a generated file into ``AtmosphericSource``, a
validated in-memory domain object with real behaviour
(``production_distribution``/``total_flux``), and bridges it into
``tpeanuts.interfaces.Flux`` via ``particle_flux``/``flux``, mirroring
``tpeanuts.source.solar.solar.SolarSource``'s own role for the
solar source (see ``model.py``'s own module docstring for exactly how
the two differ physically). Nothing here depends on
``tpeanuts.medium.atmosphere`` other than pure zenith-angle geometry
conversions (``tpeanuts.util.geometry``'s
``alpha_surface_to_theta_detector``/``theta_detector_to_alpha_surface``,
used by both generators and by ``model.py``), and nothing in
``tpeanuts.medium.atmosphere``'s live propagation path depends on this
package -- ``particle_flux``'s own output already works
directly with the existing, unmodified ``tpeanuts.pipeline.atmosphere``
(see ``model.py``'s own docstring for a worked example).

Module contents:
    ``tpeanuts.source.atmosphere.io``
        ``AtmosphericFluxTable`` and its loader,
        ``load_atmospheric_flux``; plus the shared per-angle CSV writer,
        ``write_angle_flux_csv``/``angle_output_path``, both generators
        below use.
    ``tpeanuts.source.atmosphere.mceq``
        ``MCEqConfig`` and ``generate_mceq_flux``: solve the
        atmospheric cascade equations with MCEq and save one
        height-differential flux CSV per zenith angle.
    ``tpeanuts.source.atmosphere.honda``
        ``HondaConfig`` and ``generate_honda_flux``: read the external
        Honda/HKKM flux and production-height tables and save one
        height-differential flux CSV per zenith angle.
    ``tpeanuts.source.atmosphere.atmosphere``
        ``AtmosphericSource`` (the torch-native, validated
        source domain object, with a ``.default()`` loader and ``.particle_flux()``), ``AtmosphericSourceParameters``
        and ``build_atmospheric_source``.
"""

from __future__ import annotations
