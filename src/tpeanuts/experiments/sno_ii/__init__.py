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

"""SNO salt-phase (Phase II) published-observable analysis.

Ported from v1_0's ``experiments/sno_ii/`` package (see
[[project_tpeanuts_v2_rewrite]] for the round-by-round porting log). A
sibling package to ``tpeanuts.experiments.sno`` (SNO Phase I, pure D2O),
not a subpackage of it: the two phases share the same physical
detector (D2O target with NaCl added for the salt phase) but
essentially nothing else -- different statistical separation of
CC/ES/NC (event-by-event isotropy fit vs. spectral-shape fit),
different response-function coefficients, different published
observable set, and a different likelihood (correlated Gaussian over
38 observables vs. this project's existing diagonal/Poisson
binned-count statistics, see ``tpeanuts.inference.likelihood
.correlated_gaussian_nll``, added this round). Following this
project's one-experiment-per-top-level-package convention
(``experiments.borexino``, ``experiments.dayabay``, ``experiments.icecube``,
``experiments.sno``), Phase II gets its own top-level package rather than
nesting inside ``experiments.sno``.

**Reproduces SNO's own officially-extracted observables, not raw MC
events.** Unlike Phase I's
``experiments.sno.inference_model.SNOModel`` (which sums
forward-modeled CC+ES+NC event rates because SNO Phase I did not
statistically separate the three reactions), SNO's salt-phase analysis
*did* separate CC, ES, and NC event-by-event (via the isotropy
parameter beta_14, radius, and angle to the Sun -- see the primary
source below) and published the result as 38 correlated
pseudo-observables: a 17-bin CC recoil-electron spectrum plus
integrated NC and ES fluxes, for day and night separately.
Reconstructing SNO's own multi-dimensional PDF fit from this project's
own simulated MC events is not attempted (the PDFs themselves were
never published); instead this package predicts the same 38-observable
vector SNO itself published, for direct comparison via a
correlated-Gaussian chi-square -- the same "use the officially
extracted observable, not the raw data" pattern already used for
IceCube's published detector-systematics hypersurfaces and Daya Bay's
published IBD spectra.

Primary source
    B. Aharmim et al. (SNO Collaboration), "Electron Energy Spectra,
    Fluxes, and Day-Night Asymmetries of 8B Solar Neutrinos from the
    391-Day Salt Phase SNO Data Set", Phys. Rev. C 72, 055502 (2005),
    arXiv:nucl-ex/0502021. Appendix A (Tables XXX-XXXIV) is this
    package's data source, transcribed by hand into
    ``data/experiments/sno_ii/`` -- see
    ``data/experiments/sno_ii/metadata/source.json`` for the exact
    cross-checks performed and the (small) subset of transcribed
    values still flagged for manual verification (carried forward
    verbatim, not silently resolved -- see
    ``tpeanuts.experiments.sno_ii.io``'s own module docstring).

Module contents:
    ``tpeanuts.experiments.sno_ii.parameters``
        Real salt-phase constants: the 19-channel observable ordering,
        the 17 CC spectral bin edges, the salt-phase energy-resolution
        coefficients (Eq. A3), and the fixed hep flux.
    ``tpeanuts.experiments.sno_ii.response``
        Salt-phase Gaussian response matrix.
    ``tpeanuts.experiments.sno_ii.io``
        Loaders for ``data/experiments/sno_ii/``.
    ``tpeanuts.experiments.sno_ii.observable``
        The CC/NC/ES "equivalent flux" observable functions, Eq. A1 of
        the primary source.
    ``tpeanuts.experiments.sno_ii.inference_model``
        ``SNOPhaseIIModel``, predicting the 38-observable
        vector from oscillation parameters.
"""

from __future__ import annotations
