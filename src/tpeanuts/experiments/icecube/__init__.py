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

"""IceCube DeepCore-specific detector wiring: real atmospheric-neutrino oscillation, Earth matter.

Ported from v1_0's ``detector/icecube/`` package (see
[[project_tpeanuts_v2_rewrite]] for the round-by-round porting log).
This package models the real IceCube DeepCore "9-year golden event"
atmospheric-neutrino oscillation analysis (Phys. Rev. D 108, 012014
(2023), arXiv:2304.12236), from the collaboration's own official
public data release (Harvard Dataverse, DOI 10.7910/DVN/B4RITM): real
observed counts, real event-by-event Monte Carlo (true + reconstructed
kinematics, real physical weights), a real pre-binned atmospheric-muon
background, and real per-bin detector-systematics hypersurfaces.

Unlike Daya Bay/Borexino/SNO (a continuous differential cross section
folded through a Gaussian energy response,
``tpeanuts.detector.common.event_rate``), IceCube's real data release
is event-by-event weighted Monte Carlo, so this package reweights real
simulated events by a real 3-flavour Earth-matter transition
probability (``tpeanuts.medium.earth.earth.EarthMedium.transition_matrix``,
the squared modulus of the medium's evolution operator, which exists for
every ``PropagationMethod`` -- a reusable method, not detector-specific) rather than folding a
cross-section grid.

Real atmospheric flux (electron/muon neutrino and antineutrino, needed
to convert the release's per-event weight into a physical rate) comes
from this project's own already-cached Honda flux table
(``tpeanuts.source.atmosphere.io.load_atmospheric_flux``), not a new
external dependency -- see ``tpeanuts.experiments.icecube.flux``.

Module contents:
    ``tpeanuts.experiments.icecube.io``
        Loaders for the cached real data release.
    ``tpeanuts.experiments.icecube.parameters``
        Real analysis binning, detector depth, and fixed real-data
        reference points (systematics, published best fit).
    ``tpeanuts.experiments.icecube.flux``
        Real Honda atmospheric flux, interpolated at each MC event's
        true (energy, coszen).
    ``tpeanuts.experiments.icecube.event_rate``
        The real event-by-event MC reweighting forward model.
    ``tpeanuts.experiments.icecube.inference_model``
        ``IceCubeModel``, the differentiable oscillation x
        Earth x detector composition.
"""

from __future__ import annotations
