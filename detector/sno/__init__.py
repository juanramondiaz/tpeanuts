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
SNO-specific detector wiring: CC, ES, and NC event rates.

Built from ``detector.common``/``detector.interaction.deuteron``/
``detector.interaction.neutrino_electron`` the same way
``detector.borexino`` is built from ``detector.common``/
``detector.interaction.neutrino_electron``. All three of SNO's reactions
(CC, ES, NC) are modeled as separate signal channels
(``detector.sno.event_rate``'s module docstring); the real published SNO
day/night background counts (``data/detector/sno/observation/backgrounds.csv``,
via ``detector.sno.backgrounds``) remain a background correction to the
CC/ES analysis window, not a stand-in for the modeled NC channel.

Package modules:
    parameters
        Heavy-water target composition (deuteron and electron counts) and
        energy grids.
    response
        SNO's real (Phase-I parametrization) energy-response matrix,
        shared by the CC/ES continuum fold and the NC capture-gamma
        response.
    backgrounds
        Loader for the real published day/night background counts.
    event_rate
        Composed CC/ES/NC event-rate functions, using ``tpeanuts.medium.earth``
        for Earth matter regeneration (day/night, via
        ``tpeanuts.detector.sno.inference_model``).
    io
        Loaders for the real day/night spectrum and cos-zenith exposure.
    inference_model
        The oscillation-parameter-fit composition layer, ``SNODayNightModel``.
"""
