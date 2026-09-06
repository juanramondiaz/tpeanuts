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
IceCube DeepCore event-by-event MC reweighting: real weighted-Monte-Carlo forward model.

Unlike every other detector in this project (Borexino/SNO/Daya Bay, which
fold a continuous differential cross section through a Gaussian energy
response, ``detector.common.event_rate``), IceCube's public release supplies
**event-by-event simulated Monte Carlo** with a real physical weight (GeV
cm^2 sr, see ``detector.icecube.io.load_mc_events``) rather than a
cross-section table -- the standard "weighted MC" technique every current
atmospheric-neutrino oscillation analysis uses, and the technique the
release's own ``example.ipynb`` demonstrates. This module reweights those
real events by a *real* 3-flavour Earth-matter oscillation probability
(``tpeanuts.medium.earth.probability.earth_probability_transition``,
already differentiable end-to-end -- verified this session) instead of the
official example's simplified 2-flavour vacuum toy formula, then
histograms them into the release's own real analysis bins.

**Reweighting formula.** For each simulated event (true energy/coszen, PDG
code, weight), the parent flavour(s) it could have started as before
oscillating into the flavour its own PDG code/reaction-file already tags
are ``nu_e``/``nu_mu`` (the real Honda atmospheric flux table,
``detector.icecube.flux``, has no primary nu_tau entry -- physically
correct, since prompt/conventional atmospheric tau-neutrino production is
negligible at these energies):

    predicted_weight = weight * [Phi_e(E,coszen) * P(e->beta) + Phi_mu(E,coszen) * P(mu->beta)]

with ``beta`` the event's own tagged flavour (from its PDG code) and
``Phi``/``P`` both evaluated at the event's *true* (not reconstructed)
energy and cos(zenith), matching the same convention this project's other
weighted-MC-style folds use. This applies uniformly to NC and CC events
(an NC interaction requires the neutrino to still be flavour ``beta`` at
the interaction point, exactly like a CC interaction of that flavour, up
to the small cross-section differences already baked into ``weight`` --
the official example notebook's own reweighting logic makes the same
choice).

**Real detector-systematics hypersurfaces are applied per bin**
(``detector.icecube.io.load_hypersurfaces``), interpolated over the
release's own tabulated DeltamSq31 grid (piecewise-linear, differentiable
w.r.t. the fitted DeltamSq3l) at the *fixed* real best-fit systematics
point (``detector.icecube.parameters.BESTFIT_SYSTEMATICS``, Table IV of
the publication) -- the 5 detector-calibration parameters themselves are
not re-fit here (a natural, clearly scoped future extension, matching this
project's established "rate-only first" precedent for Daya Bay/SNO).

**Real atmospheric-muon background** (``detector.icecube.io
.load_muon_background``, already pre-binned) is added with its own free
normalization scale, alongside a free overall neutrino normalization scale
-- this release does not state a usable absolute livetime for converting
``weight`` into a physical rate on its own (see ``detector.icecube
.parameters``' module docstring), so both scales are genuine fit
parameters rather than externally pre-computed constants, unlike the
official example notebook's closed-form ratio.

Module contents:
    ChannelEvents
        Precomputed, theta-independent per-channel event tensors (true
        kinematics, real Honda flux, real bin index) -- built once per
        model, reused across every ``predict(theta)`` call.
    prepare_channel_events(...)
        Build one channel's ``ChannelEvents`` from the real cached MC.
    prepare_hypersurface_table(...)
        Precompute one channel's per-bin correction factor at each real
        tabulated DeltamSq31 slice, at the fixed real best-fit systematics.
    interpolate_hypersurface(...)
        Differentiable piecewise-linear interpolation of a hypersurface
        table over its DeltamSq31 axis.
    real_observed_counts(...)
        Real observed counts, ordered to match every other function here's
        bin index.
    muon_background_histogram(...)
        Real pre-binned atmospheric-muon background as a bin-ordered tensor.
    predicted_neutrino_counts(...)
        The differentiable, per-``theta`` forward model: reweight every
        channel's real MC and sum into the real analysis bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.detector.icecube import io as icecube_io
from tpeanuts.detector.icecube.flux import flux_at_events
from tpeanuts.detector.icecube.parameters import (
    BESTFIT_SYSTEMATICS,
    N_BINS,
    N_COSZEN_BINS,
    N_ENERGY_BINS,
    NOMINAL_SYSTEMATICS,
    PID_BIN_EDGES,
    RECO_COSZEN_BIN_EDGES,
    RECO_ENERGY_BIN_EDGES_GEV,
)
from tpeanuts.medium.earth.probability import earth_probability_transition

CHANNELS: tuple[str, ...] = ("nc", "nue_cc", "numu_cc", "nutau_cc")

# Downsampling factor applied to the real event-by-event MC for
# computational tractability during a gradient-based fit (weights are
# rescaled to compensate, so the *expectation* is unbiased) -- the same
# scope choice the official example.ipynb itself makes ("Downsize input MC
# data for expediency", SCALING_FACTOR = 10) before any fit is attempted.
DEFAULT_DOWNSAMPLE: int = 10


def _bin_index(reco_energy_GeV: np.ndarray, reco_coszen: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """Map real reconstructed (energy, coszen, pid) triples to a flat 0..N_BINS-1 bin index."""
    ie = np.clip(np.searchsorted(RECO_ENERGY_BIN_EDGES_GEV.numpy(), reco_energy_GeV, side="right") - 1, 0, N_ENERGY_BINS - 1)
    ic = np.clip(np.searchsorted(RECO_COSZEN_BIN_EDGES.numpy(), reco_coszen, side="right") - 1, 0, N_COSZEN_BINS - 1)
    ip = np.clip(np.searchsorted(PID_BIN_EDGES.numpy(), pid, side="right") - 1, 0, PID_BIN_EDGES.numel() - 2)
    return (ie * N_COSZEN_BINS + ic) * (PID_BIN_EDGES.numel() - 1) + ip


# PDG code -> (flavour index 0=e/1=mu/2=tau, is_antineutrino).
_PDG_FLAVOUR = {12: 0, 14: 1, 16: 2}


@dataclass(frozen=True)
class ChannelEvents:
    """Precomputed, theta-independent per-event tensors for one reaction channel.

    Everything here is fixed real data (true kinematics, real Honda flux,
    real bin assignment) -- none of it depends on the oscillation
    parameters being fit, so it is built once (``prepare_channel_events``)
    and reused across every ``predict(theta)`` call.

    Parameters
    ----------
    true_energy_MeV, eta:
        True neutrino energy and Earth nadir angle (``eta = arccos(-true_
        coszen)``, see module docstring), the two arguments
        ``earth_probability_transition`` needs.
    weight:
        Real per-event weight (GeV cm^2 sr), already downsampling-corrected.
    flux_e, flux_mu:
        Real Honda flux (cm^-2 s^-1 sr^-1 GeV^-1) of the electron/muon
        parent species (neutrino or antineutrino, matching each event's own
        PDG sign) at the event's true (energy, coszen).
    antinu:
        Boolean tensor, True where the event's PDG code is negative.
    beta_index:
        Long tensor, the flavour index (0=e, 1=mu, 2=tau) this event's own
        PDG code/reaction file tags it as.
    bin_index:
        Long tensor, this event's flat analysis-bin index (0..N_BINS-1).
    """

    true_energy_MeV: torch.Tensor
    eta: torch.Tensor
    weight: torch.Tensor
    flux_e: torch.Tensor
    flux_mu: torch.Tensor
    antinu: torch.Tensor
    beta_index: torch.Tensor
    bin_index: torch.Tensor

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "ChannelEvents":
        return ChannelEvents(
            true_energy_MeV=self.true_energy_MeV.to(device=device, dtype=dtype),
            eta=self.eta.to(device=device, dtype=dtype),
            weight=self.weight.to(device=device, dtype=dtype),
            flux_e=self.flux_e.to(device=device, dtype=dtype),
            flux_mu=self.flux_mu.to(device=device, dtype=dtype),
            antinu=self.antinu.to(device=device),
            beta_index=self.beta_index.to(device=device),
            bin_index=self.bin_index.to(device=device),
        )


def prepare_channel_events(
    channel: str,
    *,
    downsample: int = DEFAULT_DOWNSAMPLE,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> ChannelEvents:
    """Build one channel's real, theta-independent event tensors from the cached MC.

    Args:
        channel: One of ``detector.icecube.event_rate.CHANNELS``.
        downsample: Keep every ``downsample``-th event (weights rescaled to
            compensate); 1 uses every real simulated event.
        device, dtype: Target tensor device/dtype.

    Returns:
        A ``ChannelEvents`` instance.
    """
    df = icecube_io.load_mc_events(channel)
    if downsample > 1:
        df = df.iloc[::downsample].reset_index(drop=True)
        weight = df["weight"].to_numpy() * downsample
    else:
        weight = df["weight"].to_numpy()

    pdg = df["pdg"].to_numpy()
    abs_pdg = np.abs(pdg)
    antinu_np = pdg < 0
    beta_index_np = np.array([_PDG_FLAVOUR[p] for p in abs_pdg], dtype=np.int64)

    true_energy_GeV = df["true_energy"].to_numpy()
    true_coszen = df["true_coszen"].to_numpy()

    flux_e = np.where(
        antinu_np,
        flux_at_events(true_energy_GeV, true_coszen, "nuebar"),
        flux_at_events(true_energy_GeV, true_coszen, "nue"),
    )
    flux_mu = np.where(
        antinu_np,
        flux_at_events(true_energy_GeV, true_coszen, "numubar"),
        flux_at_events(true_energy_GeV, true_coszen, "numu"),
    )

    bin_index_np = _bin_index(
        df["reco_energy"].to_numpy(), df["reco_coszen"].to_numpy(), df["pid"].to_numpy(),
    )

    eta_np = np.arccos(np.clip(-true_coszen, -1.0, 1.0))

    return ChannelEvents(
        true_energy_MeV=torch.as_tensor(true_energy_GeV * 1.0e3, dtype=dtype, device=device),
        eta=torch.as_tensor(eta_np, dtype=dtype, device=device),
        weight=torch.as_tensor(weight, dtype=dtype, device=device),
        flux_e=torch.as_tensor(flux_e, dtype=dtype, device=device),
        flux_mu=torch.as_tensor(flux_mu, dtype=dtype, device=device),
        antinu=torch.as_tensor(antinu_np, dtype=torch.bool, device=device),
        beta_index=torch.as_tensor(beta_index_np, dtype=torch.long, device=device),
        bin_index=torch.as_tensor(bin_index_np, dtype=torch.long, device=device),
    )


@dataclass(frozen=True)
class HypersurfaceTable:
    """One channel's real per-bin correction factor at each tabulated DeltamSq31 slice.

    Parameters
    ----------
    deltam31_grid:
        Real tabulated DeltamSq31 slice values (eV^2), ascending, shape
        ``(n_slices,)``.
    correction:
        Real per-bin, per-slice multiplicative correction factor at the
        fixed systematics point used to build this table, shape
        ``(n_slices, N_BINS)``.
    """

    deltam31_grid: torch.Tensor
    correction: torch.Tensor


def prepare_hypersurface_table(
    channel: str,
    *,
    systematics: dict[str, float] = BESTFIT_SYSTEMATICS,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> HypersurfaceTable:
    """Precompute one channel's real hypersurface correction factor at a fixed systematics point.

    ``factor(bin, slice) = intercept + sum_n slope_n * (systematics[n] - nominal[n])``
    (the release's own formula, see ``detector.icecube.parameters``' module
    docstring), evaluated once per real tabulated DeltamSq31 slice.

    Args:
        channel: One of ``detector.icecube.event_rate.CHANNELS``.
        systematics: Fixed detector-systematics point (see module
            docstring for why these are not free fit parameters here).
        device, dtype: Target tensor device/dtype.

    Returns:
        A ``HypersurfaceTable``.
    """
    df = icecube_io.load_hypersurfaces(channel)
    slices = np.sort(df["deltam31"].unique())

    correction = np.empty((slices.size, N_BINS), dtype=np.float64)
    for k, dm31 in enumerate(slices):
        rows = df[df["deltam31"] == dm31].reset_index(drop=True)
        bins = _bin_index(rows["reco_energy"].to_numpy(), rows["reco_coszen"].to_numpy(), rows["pid"].to_numpy())
        factor = rows["intercept"].to_numpy().copy()
        for name, value in systematics.items():
            factor = factor + rows[name].to_numpy() * (value - NOMINAL_SYSTEMATICS[name])
        correction[k, bins] = factor

    return HypersurfaceTable(
        deltam31_grid=torch.as_tensor(slices, dtype=dtype, device=device),
        correction=torch.as_tensor(correction, dtype=dtype, device=device),
    )


def interpolate_hypersurface(table: HypersurfaceTable, deltam3l: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of a hypersurface table over its real DeltamSq31 axis.

    Differentiable with respect to ``deltam3l`` (the query point, e.g. a
    fitted ``DeltamSq3l``); the tabulated grid/correction values themselves
    are fixed real data.

    Args:
        table: A ``HypersurfaceTable``.
        deltam3l: Scalar query point (eV^2), clamped to the table's own
            tabulated range.

    Returns:
        Real tensor shaped ``(N_BINS,)``.
    """
    grid = table.deltam31_grid
    query = deltam3l.clamp(grid[0], grid[-1])
    idx_hi = torch.searchsorted(grid, query.detach()).clamp(1, grid.numel() - 1)
    idx_lo = idx_hi - 1
    x_lo, x_hi = grid[idx_lo], grid[idx_hi]
    w = (query - x_lo) / (x_hi - x_lo)
    return table.correction[idx_lo] + w * (table.correction[idx_hi] - table.correction[idx_lo])


def real_observed_counts(*, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Real observed IceCube DeepCore counts, ordered to match ``predicted_neutrino_counts``'s bin index.

    Built via the same ``_bin_index`` function ``ChannelEvents``/
    ``HypersurfaceTable``/``muon_background_histogram`` use, rather than
    trusting ``data.tab``'s own row order to already match it -- a
    ``model.predict(theta)`` output is only comparable bin-for-bin against
    this function's output, never against a raw ``pandas`` read of
    ``data.tab``.

    Returns:
        Real tensor shaped ``(N_BINS,)``.
    """
    df = icecube_io.load_observed_counts()
    bins = _bin_index(df["reco_energy"].to_numpy(), df["reco_coszen"].to_numpy(), df["pid"].to_numpy())
    hist = np.zeros(N_BINS, dtype=np.float64)
    hist[bins] = df["count"].to_numpy()
    return torch.as_tensor(hist, dtype=dtype, device=device)


def muon_background_histogram(*, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Real pre-binned atmospheric-muon background, ordered to match the analysis bin index.

    Returns:
        Real tensor shaped ``(N_BINS,)``.
    """
    df = icecube_io.load_muon_background()
    bins = _bin_index(df["reco_energy"].to_numpy(), df["reco_coszen"].to_numpy(), df["pid"].to_numpy())
    hist = np.zeros(N_BINS, dtype=np.float64)
    hist[bins] = df["count"].to_numpy()
    return torch.as_tensor(hist, dtype=dtype, device=device)


def predicted_neutrino_counts(
    theta23: torch.Tensor,
    deltam_sq_3l: torch.Tensor,
    *,
    oscillation_model,
    earth_profile: object,
    detector_depth_m: float,
    channels: dict[str, ChannelEvents],
    hypersurfaces: dict[str, HypersurfaceTable],
) -> torch.Tensor:
    """Differentiable real-MC-reweighted neutrino prediction, summed into the real analysis bins.

    Args:
        theta23, deltam_sq_3l: The two free oscillation parameters (see
            ``tpeanuts.inference.model_atmosphere.AtmosphericOscillationModel``).
        oscillation_model: An ``AtmosphericOscillationModel``.
        earth_profile: Earth density profile
            (``tpeanuts.medium.earth.profile.build_earth_profile``'s return
            value).
        detector_depth_m: Detector depth below the Earth's surface, metres.
        channels: ``{channel: ChannelEvents}`` for every entry of
            ``CHANNELS``.
        hypersurfaces: ``{channel: HypersurfaceTable}`` for every entry of
            ``CHANNELS``.

    Returns:
        Real tensor shaped ``(N_BINS,)``, differentiable w.r.t.
        ``theta23``/``deltam_sq_3l``.
    """
    theta = torch.stack([theta23, deltam_sq_3l])
    total = None
    for channel in CHANNELS:
        events = channels[channel]
        oscillation: OscillationParameters = oscillation_model.oscillation(theta, antinu=events.antinu)
        P = earth_probability_transition(
            earth_profile, oscillation, events.true_energy_MeV, events.eta, detector_depth_m,
        )  # (n_events, 3, 3): [final beta, initial alpha]

        n = events.true_energy_MeV.shape[0]
        idx = torch.arange(n, device=events.true_energy_MeV.device)
        p_from_e = P[idx, events.beta_index, 0]
        p_from_mu = P[idx, events.beta_index, 1]

        predicted_weight = events.weight * (events.flux_e * p_from_e + events.flux_mu * p_from_mu)

        hist = torch.zeros(N_BINS, dtype=predicted_weight.dtype, device=predicted_weight.device)
        hist = hist.index_add(0, events.bin_index, predicted_weight)

        correction = interpolate_hypersurface(hypersurfaces[channel], deltam_sq_3l)
        contribution = hist * correction
        total = contribution if total is None else total + contribution

    return total
