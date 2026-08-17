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
Detector-level forward folding: oscillated flux -> predicted event counts.

This package sits downstream of ``tpeanuts.pipeline``/``tpeanuts.medium`` the
same way those sit downstream of ``tpeanuts.core``: it consumes an already
flavour-resolved, already-oscillated differential flux (e.g.
``tpeanuts.core.common.flux.flux_state``'s output) and turns it into
predicted per-bin event counts, comparable to a real detector's measured
spectrum via ``tpeanuts.inference.likelihood.poisson_nll``. It knows nothing
about oscillation parameters or propagation media -- composing "oscillated
flux" with "detector response" is a job for ``tpeanuts.inference.model``.

Package layout:
    common/
        Detector-agnostic building blocks: observation containers, target
        stoichiometry, Gaussian energy-response construction, efficiency
        application, and the single event-rate folding assembly (mirroring
        ``core.common.hamiltonian``'s "only Hamiltonian assembly code in the
        project" principle -- this is the only event-rate folding code).
    interaction/
        Neutrino interaction cross sections, organized by physical process
        (not by detector) so a channel shared by two experiments -- e.g.
        neutrino-electron elastic scattering, used by both Borexino and
        SNO's ES channel -- is implemented once.
    borexino/
        Borexino-specific wiring: target composition, energy response,
        background placeholder, and the composed event-rate function, built
        entirely from ``common``/``interaction`` pieces.

Every function in this package that takes only energy/observable grids
(cross sections, response matrices, efficiency curves) is safe to leave
``@torch.no_grad()``-decorated where relevant, exactly like
``tpeanuts.util.math.interp1d_linear``: none of them take a fit parameter as
input, only the oscillated flux does, and multiplying a differentiable flux
by a detached (grid-only) constant preserves the correct gradient by the
product rule. No ``core.common.hamiltonian``-style audit was needed to build
this package; one would only become necessary if a future systematic
(energy scale, efficiency normalization, ...) were promoted to a fit
parameter itself.
"""
