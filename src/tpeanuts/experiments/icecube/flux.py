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

"""Real Honda atmospheric flux, interpolated at IceCube MC events' true (energy, coszen).

Ported from v1_0's ``detector/icecube/flux.py``. Each IceCube Monte
Carlo event's ``weight`` column (GeV cm^2 sr, see
``tpeanuts.experiments.icecube.io.load_mc_events``) must be multiplied by
the *unoscillated* atmospheric flux of the event's own true parent
flavour at its own true (energy, coszen) to become a physical rate --
the official data release's own ``example.ipynb`` does this with
``daemonflux``; this project instead reuses its own already-fetched,
already-tested provider-neutral Honda flux table
(``tpeanuts.source.atmosphere.io.load_atmospheric_flux``), avoiding a
new external dependency.

**Unit note.** v1_0's own version of this module documented its
returned flux as cm^-2 s^-1 sr^-1 GeV^-1; confirmed directly (identical
``flux_at_events`` output, bit-for-bit, on the same query points) that
v1_0's own underlying Honda table is numerically the very same data
this project's own ported table is (``data/atmosphere/honda/
README.md`` documents it as m^-2 s^-1 sr^-1 GeV^-1) -- v1_0's own
docstring comment was simply stale/incorrect, not a real data or
physics difference. Harmless either way:
``tpeanuts.experiments.icecube.inference_model.IceCubeModel``'s
own ``nu_norm`` is a free fit parameter precisely because this data
release states no independently usable absolute livetime either (see
that module's own docstring), so any constant unit factor is absorbed
into ``nu_norm`` regardless.

This module is explicitly *not* part of the differentiable forward
model: the atmospheric flux does not depend on the oscillation
parameters being fit, so ``flux_at_events`` is called once at
model-construction time
(``tpeanuts.experiments.icecube.event_rate``) with plain floats/NumPy,
not inside the per-``theta`` ``predict`` call. Consistently, it never
takes a caller-supplied ``RuntimeContext``: the Honda table is always
loaded on a fixed CPU/float64 context (matching v1_0's own hardcoded
``device="cpu", dtype=torch.float64``) and immediately converted to
plain numpy, since a GPU device would only add a pointless
host-device round trip here (confirmed directly: passing a CUDA
context through to ``tpeanuts.source.atmosphere.io
.load_atmospheric_flux`` and then straight to ``.numpy()`` raises,
since a CUDA tensor cannot convert to numpy without an explicit
``.cpu()`` first).

Module contents:
    flux_at_events(...)
        Interpolate the real Honda nu_e/nu_mu (and antineutrino) flux
        at a batch of (true_energy_GeV, true_coszen) points.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from tpeanuts.source.atmosphere.io import load_atmospheric_flux
from tpeanuts.util.context import RuntimeContext

_LOAD_CONTEXT = RuntimeContext(device=torch.device("cpu"), dtype=torch.float64)


def _particle_grid(particle: str) -> RegularGridInterpolator:
    """Build a log(E)-coszen bilinear interpolator for one Honda flux particle species.

    Azimuth-averages the real long-form Honda table (12 azimuth bins,
    IceCube's DeepCore oscillation binning has no azimuth axis) onto a
    regular (log10 true_energy_GeV, true_coszen) grid, then wraps it in
    a ``scipy`` bilinear interpolator (bounds-clamped: the table's own
    tabulated cos(zenith) range, [-0.95, 0.95] bin centers, does not
    quite reach the physical edges [-1, 1], so out-of-range queries --
    e.g. IceCube's most upgoing bin -- are clamped to the nearest
    tabulated value rather than extrapolated).

    Args:
        particle: One of ``"nue"``, ``"nuebar"``, ``"numu"``, ``"numubar"``.

    Returns:
        A ``RegularGridInterpolator`` mapping
        ``(log10(true_energy_GeV), true_coszen)`` to flux in
        m^-2 s^-1 sr^-1 GeV^-1 (see module docstring's unit note).
    """
    table = load_atmospheric_flux(context=_LOAD_CONTEXT)
    particle_arr = np.asarray(table.particle)
    mask = particle_arr == particle
    energy_GeV = (table.energy_MeV[mask] / 1.0e3).numpy()
    coszen = table.cos_zenith[mask].numpy()
    flux = table.flux[mask].numpy()

    energy_grid = np.sort(np.unique(energy_GeV))
    coszen_grid = np.sort(np.unique(coszen))

    grid = np.empty((energy_grid.size, coszen_grid.size), dtype=np.float64)
    for i, e in enumerate(energy_grid):
        for j, c in enumerate(coszen_grid):
            grid[i, j] = flux[(energy_GeV == e) & (coszen == c)].mean()

    return RegularGridInterpolator(
        (np.log10(energy_grid), coszen_grid), grid,
        method="linear", bounds_error=False, fill_value=None,
    )


_INTERPOLATORS: Dict[str, RegularGridInterpolator] = {}


def _get_interpolator(particle: str) -> RegularGridInterpolator:
    if particle not in _INTERPOLATORS:
        _INTERPOLATORS[particle] = _particle_grid(particle)
    return _INTERPOLATORS[particle]


def flux_at_events(true_energy_GeV: np.ndarray, true_coszen: np.ndarray, particle: str) -> np.ndarray:
    """Interpolate the real Honda flux of one particle species at a batch of events.

    Args:
        true_energy_GeV: True neutrino energy, shape ``(n,)``.
        true_coszen: True cosine(zenith), shape ``(n,)``.
        particle: One of ``"nue"``, ``"nuebar"``, ``"numu"``, ``"numubar"``.

    Returns:
        Real array shaped ``(n,)``, flux in m^-2 s^-1 sr^-1 GeV^-1 (see
        module docstring's unit note).
    """
    interpolator = _get_interpolator(particle)
    coszen_clamped = np.clip(true_coszen, -0.95, 0.95)
    log_energy = np.log10(np.clip(true_energy_GeV, 1.0e-3, None))
    return interpolator(np.stack([log_energy, coszen_clamped], axis=-1))
