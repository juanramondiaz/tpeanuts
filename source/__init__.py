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
Neutrino source data and production physics, independent of propagation medium and detector.

Holds the pieces of "where do the neutrinos come from" that are genuinely
reusable across more than one detector/experiment -- if a table or formula
is universal physics (a per-isotope reactor antineutrino spectrum, a
provider-neutral atmospheric flux table format), it belongs here rather
than duplicated inside each detector package that happens to consume it.
Experiment-specific source configuration (which reactor cores, which
fission fractions, which baselines) stays in that experiment's own
``detector.<name>`` package instead, since it is not shared physics.

Package contents:
    reactor
        Generic (isotope-level) Huber-Mueller antineutrino spectrum
        physics, reusable by any reactor experiment.
    atmosphere
        Provider-neutral I/O for tabulated atmospheric-neutrino flux
        datasets (Honda/MCEq/Bartol), independent of atmosphere
        propagation physics (``medium.atmosphere``).
    solar
        Radial production distributions, total fluxes, and production
        spectra per solar-neutrino source, independent of solar
        propagation physics (``medium.solar``).
"""
