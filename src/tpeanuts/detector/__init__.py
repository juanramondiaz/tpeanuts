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

"""Generic detector physics: what turns a propagated probability into an observable rate.

A detector here consumes the ``Flux`` that reached it (or bare ``(energy_MeV, probabilities)`` tensors) and produces an
``ObservableSpectrum`` or a ``BinnedSpectrum`` of ``tpeanuts.interfaces.observable`` -- the only things that cross this
package's boundary.
``tpeanuts.detector`` never imports ``tpeanuts.source``, ``tpeanuts.medium`` or ``tpeanuts.pipeline``, and knows nothing
about propagation itself (that is ``tpeanuts.physics``'s job) or about oscillation at all -- it only knows how a given
interaction reacts to whatever flavour survival probability it is handed.

``tpeanuts.detector.ibd.IBDDetector`` (inverse beta decay) returns a rate over the true energy;
``tpeanuts.detector.folded`` (``FoldedDetector``, ``FixedEnergyDetector``) fold a flux through a cross section and a
response into counts per bin (a ``BinnedSpectrum``), and are what the Borexino and SNO channels are built from.
``tpeanuts.detector.ibd_cross_section`` (inverse beta decay), ``tpeanuts.detector.neutrino_electron`` (elastic nu-e
scattering) and ``tpeanuts.detector.deuteron`` (CC/NC deuteron breakup) hold the reusable interaction *cross sections*,
physics independent of any one experiment.
``tpeanuts.detector.common`` holds the detector-agnostic building blocks (energy response, efficiency, the
flux -> counts folding chain, published observations, target particle counts) any experiment composes from.

The experiments themselves (Daya Bay, Borexino, SNO, SNO-II, IceCube) -- a source, the media in between, a detector
and the data wired together -- live in ``tpeanuts.experiments``.
"""

from __future__ import annotations

__all__: list = []  # experimental in 1.0: no stability guarantee, import from the submodules
