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
IceCube DeepCore-specific detector wiring: real atmospheric-neutrino oscillation, Earth matter.

This package models the real IceCube DeepCore "9-year golden event"
atmospheric-neutrino oscillation analysis (Phys. Rev. D 108, 012014
(2023), arXiv:2304.12236), from the collaboration's own official public
data release (Harvard Dataverse, DOI 10.7910/DVN/B4RITM): real observed
counts, real event-by-event Monte Carlo (true + reconstructed kinematics,
real physical weights), a real pre-binned atmospheric-muon background, and
real per-bin detector-systematics hypersurfaces.

Unlike Borexino/SNO/Daya Bay (a continuous differential cross section
folded through a Gaussian energy response, ``detector.common.event_rate``),
IceCube's real data release is event-by-event weighted Monte Carlo, so this
package reweights real simulated events by a real 3-flavour Earth-matter
oscillation probability (``tpeanuts.medium.earth.probability
.earth_probability_transition``) rather than folding a cross-section grid
-- see ``detector.icecube.event_rate``'s module docstring for the full
reweighting formula and the scope decisions made (fixed detector
systematics at their real published best-fit point; free overall
normalization scales, since this release states no independently usable
absolute livetime).

Real atmospheric flux (electron/muon neutrino and antineutrino, needed to
convert the release's per-event weight into a physical rate) comes from
this project's own already-cached Honda flux table
(``tpeanuts.source.atmosphere.io.load_atmospheric_flux``), not a new
external dependency -- see ``detector.icecube.flux``.

Data provenance: ``notebooks/external/icecube/IceCube1_generator.ipynb``
fetches and validates the release verbatim into ``data/detector/icecube/raw/``.

Package modules:
    parameters
        Real analysis binning, detector depth, and fixed real-data
        reference points (systematics, published best fit).
    io
        Loaders for the cached real data release.
    flux
        Real Honda atmospheric flux, interpolated at each MC event's true
        (energy, coszen).
    event_rate
        The real event-by-event MC reweighting forward model.
    inference_model
        The oscillation-parameter-fit composition layer,
        ``IceCubeDetectorModel``.
"""
