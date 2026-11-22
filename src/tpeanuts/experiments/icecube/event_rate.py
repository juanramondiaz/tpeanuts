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

"""IceCube DeepCore event-by-event MC reweighting: real weighted-Monte-Carlo forward model.

Ported from v1_0's ``detector/icecube/event_rate.py``. Unlike every
other detector in this project (Daya Bay/Borexino/SNO, which fold a
continuous differential cross section through a Gaussian energy
response, ``tpeanuts.detector.common.event_rate``), IceCube's public
release supplies **event-by-event simulated Monte Carlo** with a real
physical weight (GeV cm^2 sr, see
``tpeanuts.experiments.icecube.io.load_mc_events``) rather than a
cross-section table -- the standard "weighted MC" technique every
current atmospheric-neutrino oscillation analysis uses, and the
technique the release's own ``example.ipynb`` demonstrates. This
module reweights those real events by a *real* 3-flavour Earth-matter
transition probability
(``tpeanuts.medium.earth.earth.EarthMedium.transition_matrix``
-- differentiable end to end, no ``@torch.no_grad()`` anywhere
in its own call chain, confirmed directly) instead of the official
example's simplified 2-flavour vacuum toy formula, then histograms
them into the release's own real analysis bins.

**Reweighting formula.** For each simulated event (true energy/coszen,
PDG code, weight), the parent flavour(s) it could have started as
before oscillating into the flavour its own PDG code/reaction file
already tags are ``nu_e``/``nu_mu`` (the real Honda atmospheric flux
table, ``tpeanuts.experiments.icecube.flux``, has no primary nu_tau
entry -- physically correct, since prompt/conventional atmospheric
tau-neutrino production is negligible at these energies):

    predicted_weight = weight * [Phi_e(E,coszen) * P(e->beta) + Phi_mu(E,coszen) * P(mu->beta)]

with ``beta`` the event's own tagged flavour (from its PDG code) and
``Phi``/``P`` both evaluated at the event's *true* (not reconstructed)
energy and cos(zenith). This applies uniformly to NC and CC events (an
NC interaction requires the neutrino to still be flavour ``beta`` at
the interaction point, exactly like a CC interaction of that flavour,
up to the small cross-section differences already baked into
``weight`` -- the official example notebook's own reweighting logic
makes the same choice).

**Splitting events by antineutrino sign, not a single batched
``antinu`` mask.** v1_0's own ``OscillationParameters`` accepts
``antinu`` as a per-event boolean tensor directly; confirmed directly
that v2's ``PropagationModel``/``MixingMatrix`` does not broadcast a
batched ``antinu`` mask against a batched energy/eta_rad axis at the same
time (``RuntimeError`` inside ``MixingMatrix._select_antinu`` -- same
class of batching-axis collision already documented twice,
[[reference_batching_axis_collision]],
[[reference_earth_analytic_perturbative_batching]]). So each channel's
events are split into two homogeneous subsets (neutrino/antineutrino)
here, each folded through its own ``PropagationModel(..., antinu=...)``,
and the two subsets' results are merged back into the original
per-event order before histogramming -- safe, and does not touch
shared, already-tested physics code.

**Real detector-systematics hypersurfaces are applied per bin**
(``tpeanuts.experiments.icecube.io.load_hypersurfaces``), interpolated
over the release's own tabulated DeltamSq31 grid (piecewise-linear,
differentiable w.r.t. the fitted DeltamSq3l) at the *fixed* real
best-fit systematics point
(``tpeanuts.experiments.icecube.parameters.BESTFIT_SYSTEMATICS``, Table
IV of the publication) -- the 5 detector-calibration parameters
themselves are not re-fit here (a natural, clearly scoped future
extension, matching this project's established "rate-only first"
precedent for Daya Bay/SNO).

**Real atmospheric-muon background**
(``tpeanuts.experiments.icecube.io.load_muon_background``, already
pre-binned) is added with its own free normalization scale, alongside
a free overall neutrino normalization scale -- this release does not
state a usable absolute livetime for converting ``weight`` into a
physical rate on its own (see
``tpeanuts.experiments.icecube.parameters``'s own module docstring), so
both scales are genuine fit parameters rather than externally
pre-computed constants, unlike the official example notebook's
closed-form ratio.

Module contents:
    CHANNELS
        The four real reaction channels this release ships.
    ChannelEvents
        Precomputed, theta-independent per-channel event tensors (true
        kinematics, real Honda flux, real bin index) -- built once per
        model, reused across every ``predict(theta)`` call.
    build_channel_events(...)
        Build one channel's ``ChannelEvents`` from the real cached MC.
    HypersurfaceTable
        One channel's real per-bin correction factor at each real
        tabulated DeltamSq31 slice.
    build_hypersurface_table(...)
        Precompute one channel's per-bin correction factor at each real
        tabulated DeltamSq31 slice, at the fixed real best-fit
        systematics.
    interpolate_hypersurface(...)
        Differentiable piecewise-linear interpolation of a
        hypersurface table over its DeltamSq31 axis.
    real_observed_counts(...)
        Real observed counts, ordered to match every other function
        here's bin index.
    muon_background_histogram(...)
        Real pre-binned atmospheric-muon background as a bin-ordered
        tensor.
    predicted_neutrino_counts(...)
        The differentiable, per-``theta`` forward model: reweight
        every channel's real MC and sum into the real analysis bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

from tpeanuts.config import default
from tpeanuts.experiments.icecube import io as icecube_io
from tpeanuts.experiments.icecube.flux import flux_at_events
from tpeanuts.experiments.icecube.parameters import (
    BESTFIT_SYSTEMATICS,
    N_BINS,
    N_COSZEN_BINS,
    N_ENERGY_BINS,
    NOMINAL_SYSTEMATICS,
    PID_BIN_EDGES,
    RECO_COSZEN_BIN_EDGES,
    RECO_ENERGY_BIN_EDGES_GEV,
)
from tpeanuts.medium.earth.earth import EarthMedium
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix, StandardMixingParameters
from tpeanuts.physics.common.nature import Dirac
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.evolution import PropagationMethod
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import searchsorted_contiguous
from tpeanuts.util.tensor import TensorLike

Tensor = torch.Tensor

CHANNELS: Tuple[str, ...] = ("nc", "nue_cc", "numu_cc", "nutau_cc")

# Downsampling factor applied to the real event-by-event MC for
# computational tractability during a gradient-based fit (weights are
# rescaled to compensate, so the *expectation* is unbiased) -- the same
# scope choice the official example.ipynb itself makes ("Downsize input
# MC data for expediency", SCALING_FACTOR = 10) before any fit is
# attempted.
DEFAULT_DOWNSAMPLE: int = 10

_RECO_ENERGY_EDGES_NP = RECO_ENERGY_BIN_EDGES_GEV.numpy()
_RECO_COSZEN_EDGES_NP = RECO_COSZEN_BIN_EDGES.numpy()
_PID_EDGES_NP = PID_BIN_EDGES.numpy()

# PDG code -> flavour index (0=e, 1=mu, 2=tau).
_PDG_FLAVOUR = {12: 0, 14: 1, 16: 2}


def _bin_index(reco_energy_GeV: np.ndarray, reco_coszen: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """Map real reconstructed (energy, coszen, pid) triples to a flat 0..N_BINS-1 bin index."""
    ie = np.clip(np.searchsorted(_RECO_ENERGY_EDGES_NP, reco_energy_GeV, side="right") - 1, 0, N_ENERGY_BINS - 1)
    ic = np.clip(np.searchsorted(_RECO_COSZEN_EDGES_NP, reco_coszen, side="right") - 1, 0, N_COSZEN_BINS - 1)
    ip = np.clip(np.searchsorted(_PID_EDGES_NP, pid, side="right") - 1, 0, _PID_EDGES_NP.size - 2)
    return (ie * N_COSZEN_BINS + ic) * (_PID_EDGES_NP.size - 1) + ip


@dataclass(frozen=True)
class ChannelEvents:
    """Precomputed, theta-independent per-event tensors for one reaction channel.

    Everything here is fixed real data (true kinematics, real Honda
    flux, real bin assignment) -- none of it depends on the
    oscillation parameters being fit, so it is built once
    (``build_channel_events``) and reused across every
    ``predict(theta)`` call.

    Attributes:
        true_energy_MeV, eta_rad: True neutrino energy and Earth nadir
            angle (``eta_rad = arccos(-true_coszen)``, see module
            docstring), the two arguments
            ``EarthMedium.transition_matrix`` needs.
        weight: Real per-event weight (GeV cm^2 sr), already
            downsampling-corrected.
        flux_e, flux_mu: Real Honda flux (m^-2 s^-1 sr^-1 GeV^-1, see
            ``tpeanuts.experiments.icecube.flux``'s own module docstring)
            of the electron/muon parent species (neutrino or
            antineutrino, matching each event's own PDG sign) at the
            event's true (energy, coszen).
        antinu: Boolean tensor, True where the event's PDG code is
            negative.
        beta_index: Long tensor, the flavour index (0=e, 1=mu, 2=tau)
            this event's own PDG code/reaction file tags it as.
        bin_index: Long tensor, this event's flat analysis-bin index
            (0..N_BINS-1).
    """

    true_energy_MeV: Tensor
    eta_rad: Tensor
    weight: Tensor
    flux_e: Tensor
    flux_mu: Tensor
    antinu: Tensor
    beta_index: Tensor
    bin_index: Tensor


def build_channel_events(
    channel: str, *, downsample: int = DEFAULT_DOWNSAMPLE, context: RuntimeContext,
) -> ChannelEvents:
    """Build one channel's real, theta-independent event tensors from the cached MC.

    Args:
        channel: One of ``CHANNELS``.
        downsample: Keep every ``downsample``-th event (weights
            rescaled to compensate); 1 uses every real simulated
            event.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        A ``ChannelEvents`` instance.
    """
    mc = icecube_io.load_mc_events(channel)
    if downsample > 1:
        mc = {name: col[::downsample] for name, col in mc.items()}
        weight = mc["weight"] * downsample
    else:
        weight = mc["weight"]

    pdg = mc["pdg"]
    abs_pdg = np.abs(pdg)
    antinu_np = pdg < 0
    beta_index_np = np.array([_PDG_FLAVOUR[p] for p in abs_pdg], dtype=np.int64)

    true_energy_GeV = mc["true_energy"]
    true_coszen = mc["true_coszen"]

    flux_e = np.where(
        antinu_np, flux_at_events(true_energy_GeV, true_coszen, "nuebar"),
        flux_at_events(true_energy_GeV, true_coszen, "nue"),
    )
    flux_mu = np.where(
        antinu_np, flux_at_events(true_energy_GeV, true_coszen, "numubar"),
        flux_at_events(true_energy_GeV, true_coszen, "numu"),
    )

    bin_index_np = _bin_index(mc["reco_energy"], mc["reco_coszen"], mc["pid"])
    eta_np = np.arccos(np.clip(-true_coszen, -1.0, 1.0))

    return ChannelEvents(
        true_energy_MeV=torch.as_tensor(true_energy_GeV * 1.0e3, dtype=context.dtype, device=context.device),
        eta_rad=torch.as_tensor(eta_np, dtype=context.dtype, device=context.device),
        weight=torch.as_tensor(weight, dtype=context.dtype, device=context.device),
        flux_e=torch.as_tensor(flux_e, dtype=context.dtype, device=context.device),
        flux_mu=torch.as_tensor(flux_mu, dtype=context.dtype, device=context.device),
        antinu=torch.as_tensor(antinu_np, dtype=torch.bool, device=context.device),
        beta_index=torch.as_tensor(beta_index_np, dtype=torch.long, device=context.device),
        bin_index=torch.as_tensor(bin_index_np, dtype=torch.long, device=context.device),
    )


@dataclass(frozen=True)
class HypersurfaceTable:
    """One channel's real per-bin correction factor at each tabulated DeltamSq31 slice.

    Attributes:
        deltam31_grid: Real tabulated DeltamSq31 slice values (eV^2),
            ascending, shape ``(n_slices,)``.
        correction: Real per-bin, per-slice multiplicative correction
            factor at the fixed systematics point used to build this
            table, shape ``(n_slices, N_BINS)``.
    """

    deltam31_grid: Tensor
    correction: Tensor


def build_hypersurface_table(
    channel: str, *, systematics: Dict[str, float] = BESTFIT_SYSTEMATICS, context: RuntimeContext,
) -> HypersurfaceTable:
    """Precompute one channel's real hypersurface correction factor at a fixed systematics point.

    ``factor(bin, slice) = intercept + sum_n slope_n * (systematics[n] - nominal[n])``
    (the release's own formula, see
    ``tpeanuts.experiments.icecube.parameters``'s own module docstring),
    evaluated once per real tabulated DeltamSq31 slice.

    Args:
        channel: One of ``CHANNELS``.
        systematics: Fixed detector-systematics point (see module
            docstring for why these are not free fit parameters here).
        context: RuntimeContext the returned tensors are built on.

    Returns:
        A ``HypersurfaceTable``.
    """
    hs = icecube_io.load_hypersurfaces(channel)
    slices = np.sort(np.unique(hs["deltam31"]))

    correction = np.empty((slices.size, N_BINS), dtype=np.float64)
    for k, dm31 in enumerate(slices):
        rows = hs["deltam31"] == dm31
        bins = _bin_index(hs["reco_energy"][rows], hs["reco_coszen"][rows], hs["pid"][rows])
        factor = hs["intercept"][rows].copy()
        for name, value in systematics.items():
            factor = factor + hs[name][rows] * (value - NOMINAL_SYSTEMATICS[name])
        correction[k, bins] = factor

    return HypersurfaceTable(
        deltam31_grid=torch.as_tensor(slices, dtype=context.dtype, device=context.device),
        correction=torch.as_tensor(correction, dtype=context.dtype, device=context.device),
    )


def interpolate_hypersurface(table: HypersurfaceTable, deltam3l: Tensor) -> Tensor:
    """Piecewise-linear interpolation of a hypersurface table over its real DeltamSq31 axis.

    Differentiable with respect to ``deltam3l`` (the query point, e.g.
    a fitted ``DeltamSq3l``); the tabulated grid/correction values
    themselves are fixed real data.

    Args:
        table: A ``HypersurfaceTable``.
        deltam3l: Scalar query point (eV^2), clamped to the table's own
            tabulated range.

    Returns:
        Real tensor shaped ``(N_BINS,)``.
    """
    grid = table.deltam31_grid
    query = deltam3l.clamp(grid[0], grid[-1])
    idx_hi = searchsorted_contiguous(grid, query.detach()).clamp(1, grid.numel() - 1)
    idx_lo = idx_hi - 1
    x_lo, x_hi = grid[idx_lo], grid[idx_hi]
    w = (query - x_lo) / (x_hi - x_lo)
    return table.correction[idx_lo] + w * (table.correction[idx_hi] - table.correction[idx_lo])


def real_observed_counts(*, context: RuntimeContext) -> Tensor:
    """Real observed IceCube DeepCore counts, ordered to match ``predicted_neutrino_counts``'s bin index.

    Built via the same ``_bin_index`` function ``ChannelEvents``/
    ``HypersurfaceTable``/``muon_background_histogram`` use, rather
    than trusting ``data.tab``'s own row order to already match it --
    a ``model.predict(theta)`` output is only comparable bin-for-bin
    against this function's output, never against a raw read of
    ``data.tab``.

    Args:
        context: RuntimeContext the returned tensor is built on.

    Returns:
        Real tensor shaped ``(N_BINS,)``.
    """
    obs = icecube_io.load_observed_counts()
    bins = _bin_index(obs["reco_energy"], obs["reco_coszen"], obs["pid"])
    hist = np.zeros(N_BINS, dtype=np.float64)
    hist[bins] = obs["count"]
    return torch.as_tensor(hist, dtype=context.dtype, device=context.device)


def muon_background_histogram(*, context: RuntimeContext) -> Tensor:
    """Real pre-binned atmospheric-muon background, ordered to match the analysis bin index.

    Args:
        context: RuntimeContext the returned tensor is built on.

    Returns:
        Real tensor shaped ``(N_BINS,)``.
    """
    mu = icecube_io.load_muon_background()
    bins = _bin_index(mu["reco_energy"], mu["reco_coszen"], mu["pid"])
    hist = np.zeros(N_BINS, dtype=np.float64)
    hist[bins] = mu["count"]
    return torch.as_tensor(hist, dtype=context.dtype, device=context.device)


def _transition_probabilities_from_e_and_mu(
    events: ChannelEvents,
    mixing: MixingMatrix,
    mass_spectrum: MassSpectrum,
    *,
    detector_depth_m: float,
    earth_method: PropagationMethod,
    earth_nsteps: int,
    extensions: Tuple[Any, ...] = (),
) -> Tuple[Tensor, Tensor]:
    """P(e->beta)/P(mu->beta) for every event, splitting by antineutrino sign (see module docstring).

    Args:
        events: One channel's ``ChannelEvents``.
        mixing, mass_spectrum: Shared oscillation-parameter pieces
            (built once by the caller); only ``antinu`` differs
            between the two ``PropagationModel``s built here.
        detector_depth_m, earth_method, earth_nsteps: Forwarded to
            ``tpeanuts.medium.earth.earth.EarthMedium``.
        extensions: Propagation extensions (for example NSI), given to both ``PropagationModel``s.

    Returns:
        ``(p_from_e, p_from_mu)``, each shape ``(n_events,)``,
        differentiable w.r.t. ``mixing``/``mass_spectrum``'s own
        parameters.
    """
    context = mixing.context
    n = events.true_energy_MeV.shape[0]
    p_from_e = torch.zeros(n, device=context.device, dtype=context.dtype)
    p_from_mu = torch.zeros(n, device=context.device, dtype=context.dtype)

    for antinu, mask in ((False, ~events.antinu), (True, events.antinu)):
        if not bool(mask.any()):
            continue
        model = PropagationModel(mixing=mixing, mass_spectrum=mass_spectrum, antinu=antinu, extensions=extensions)
        earth = EarthMedium(
            eta_rad=events.eta_rad[mask], context=context, depth_m=detector_depth_m, method=earth_method,
            nsteps=earth_nsteps,
        )
        # (n_masked, 3, 3): [final beta, initial alpha]
        P = earth.transition_matrix(model, events.true_energy_MeV[mask])
        idx = torch.arange(P.shape[0], device=context.device)
        beta = events.beta_index[mask]
        p_from_e = p_from_e.masked_scatter(mask, P[idx, beta, 0])
        p_from_mu = p_from_mu.masked_scatter(mask, P[idx, beta, 1])

    return p_from_e, p_from_mu


def predicted_neutrino_counts(
    theta23: Tensor,
    deltam_sq_3l: Tensor,
    *,
    theta12: TensorLike,
    theta13: TensorLike,
    delta13: TensorLike,
    deltam_sq_21: TensorLike,
    context: RuntimeContext,
    detector_depth_m: float,
    channels: Dict[str, ChannelEvents],
    hypersurfaces: Dict[str, HypersurfaceTable],
    earth_method: PropagationMethod = PropagationMethod.ANALYTIC,
    earth_nsteps: int = default.EARTH_NSTEPS,
    extensions: Tuple[Any, ...] = (),
) -> Tensor:
    """Differentiable real-MC-reweighted neutrino prediction, summed into the real analysis bins.

    Args:
        theta23, deltam_sq_3l: The two free oscillation parameters.
        theta12, theta13, delta13, deltam_sq_21: Fixed solar/reactor
            oscillation inputs (see
            ``tpeanuts.experiments.icecube.parameters``).
        context: RuntimeContext every tensor this function builds is
            on.
        detector_depth_m: Detector depth below the Earth's surface,
            metres.
        channels: ``{channel: ChannelEvents}`` for every entry of
            ``CHANNELS``.
        hypersurfaces: ``{channel: HypersurfaceTable}`` for every entry
            of ``CHANNELS``.
        earth_method, earth_nsteps, extensions: Forwarded to
            ``tpeanuts.medium.earth.earth.EarthMedium``.

    Returns:
        Real tensor shaped ``(N_BINS,)``, differentiable w.r.t.
        ``theta23``/``deltam_sq_3l``.
    """
    space = FlavourSpace()
    mixing = MixingMatrix(
        space=space,
        standard=StandardMixingParameters(
            theta12=theta12, theta13=theta13, theta23=theta23, delta13=delta13, context=context
        ),
        nature=Dirac(),
        context=context,
    )
    mass_spectrum = MassSpectrum(space=space, DeltamSq21=deltam_sq_21, DeltamSq3l=deltam_sq_3l, context=context)

    total = None
    for channel in CHANNELS:
        events = channels[channel]
        p_from_e, p_from_mu = _transition_probabilities_from_e_and_mu(
            events, mixing, mass_spectrum,
            detector_depth_m=detector_depth_m, earth_method=earth_method, earth_nsteps=earth_nsteps,
            extensions=extensions,
        )
        predicted_weight = events.weight * (events.flux_e * p_from_e + events.flux_mu * p_from_mu)

        hist = torch.zeros(N_BINS, dtype=predicted_weight.dtype, device=predicted_weight.device)
        hist = hist.index_add(0, events.bin_index, predicted_weight)

        correction = interpolate_hypersurface(hypersurfaces[channel], deltam_sq_3l)
        contribution = hist * correction
        total = contribution if total is None else total + contribution

    assert total is not None  # CHANNELS is a non-empty constant tuple, so the loop always runs at least once
    return total
