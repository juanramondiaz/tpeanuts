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

"""Numerical (non-adiabatic) solar-interior propagation.

``medium.solar.probability``'s ``method="adiabatic"`` path never builds a
coherent evolutor: it invokes the adiabatic theorem directly at the local
production density (``Tei``), with a closed-form Landau-Zener correction for
the plain 3-flavour case only (``landau_zener.py``) -- that correction has no
simple generalisation once NSI and/or the 3+1 sterile extension distort the
resonance structure.

This module instead builds a genuine coherent flavour-basis evolutor
``S(r_end, r)`` with the same ``core.numerical`` machinery used for
Earth-crossing propagation (NSI/sterile/NC generic, no Landau-Zener
approximation needed since every non-adiabatic transition is captured by
construction). Because it integrates the Schroedinger equation outward
instead of invoking the adiabatic theorem, it needs the full structural
density grid (``profile.radius``/``density``, extending to near-vacuum), not
the production-restricted grid ``Tei`` alone needs.

Every production point shares the same trajectory endpoint, so
``solar_evolutor_numerical`` builds the segment history S(r_j, 0) once over
the merged (production points + full density grid) trajectory and recovers
every point's evolutor to the endpoint via S(r_end, r_k) = S(r_end, 0) @
S(r_k, 0)^dagger (valid since each S(r_j, 0) is unitary), instead of
integrating a separate trajectory per production point.

Accuracy is tied to the density table's own tabulated grid (no separate
refinement), so cross-validate against ``method="adiabatic"`` in the plain SM
limit (where both must agree) before trusting non-standard configurations.

Module functions:
    build_solar_trajectory(...)
        Merge the full density grid and production-point grid into one
        trajectory, so every production point is an exact segment boundary.
    solar_evolutor_numerical_history(...)
        Compute S(r_j, 0) for every trajectory boundary point in one batch.
    solar_evolutor_numerical(...)
        Production-point-to-endpoint evolutor for every production radius.
    mass_weights_numerical(...)
        Exact (non-adiabatic) per-production-point mass-basis weights --
        the numerical counterpart of ``Tei``.
"""

from __future__ import annotations

from typing import Optional

import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.evolutor import apply_evolutor_to_state
from tpeanuts.core.common.neutrino import flavour_state
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.numerical.evolutor import evolutor_numerical
from tpeanuts.core.numerical.geometry import OdeMethod, Trajectory, segment_sample_points
from tpeanuts.util.math import interp1d_linear
from tpeanuts.util.type import TensorLike


def build_solar_trajectory(
    profile,
    *,
    method: Optional[OdeMethod] = "midpoint",
) -> Trajectory:
    """Merge the full density grid and production grid into one trajectory.

    Segment boundaries are the sorted union of ``profile.radius`` (full
    density table) and ``profile.production_radius``, so every production
    point is an exact boundary -- required for the shared-history endpoint
    trick in ``solar_evolutor_numerical`` to be exact rather than
    interpolated.

    Returns:
        ``Trajectory`` with the merged grid ``x``, dimensionless
        ``dx_evolution``, per-segment ``sample_x``, and
        ``meta["production_index"]`` locating ``profile.production_radius``
        within ``x``.
    """
    x = torch.unique(torch.cat([profile.radius, profile.production_radius]), sorted=True)
    dx_evolution = x[1:] - x[:-1]
    sample_x = segment_sample_points(x, method)
    production_index = torch.searchsorted(x, profile.production_radius)

    return Trajectory(
        x=x,
        dx_evolution=dx_evolution,
        sample_x=sample_x,
        meta={"kind": "solar", "production_index": production_index},
    )


@torch.no_grad()
def solar_evolutor_numerical_history(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    profile,
    *,
    method: Optional[OdeMethod] = "midpoint",
    include_matter_nc: bool = False,
    legacy_precision: bool = False,
) -> tuple[torch.Tensor, Trajectory]:
    """Compute S(r_j, 0) for every point on the merged numerical trajectory.

    Args:
        oscillation: Built pmns object (3-flavour or 3+1 sterile) plus mass
            splittings, antinu selection, and the optional ``nsi`` attribute
            -- read generically by ``core.numerical.evolutor_numerical``.
        E_MeV: Neutrino energy in MeV, scalar or batched.
        profile: SolarProfile-like object exposing the full ``radius``/
            ``density`` grid and, when ``include_matter_nc=True``,
            ``density_n``.
        include_matter_nc: If True, also sample and apply the 3+1 sterile
            neutral-current term via ``profile.density_n``. Only meaningful
            for a 4-flavour ``oscillation.pmns``; silently ignored otherwise.

    Returns:
        ``(S_history, trajectory)``: complex tensor shaped ``(..., n, N, N)``
        (N = 3 or 4), the accumulated evolutor from the trajectory start to
        each merged-grid point (identity at index 0), plus the ``Trajectory``
        used to build it.

    Raises:
        ValueError: If ``include_matter_nc=True`` and ``profile.density_n``
            is not set.
    """
    trajectory = build_solar_trajectory(profile, method=method)

    n_e_samples = interp1d_linear(
        x=trajectory.sample_x,
        xp=profile.radius,
        fp=profile.density,
        left=profile.density[0],
        right=profile.density[-1],
        device=profile.device,
        dtype=profile.dtype,
    )

    n_n_samples = None
    if include_matter_nc:
        if profile.density_n is None:
            raise ValueError(
                "include_matter_nc=True requires profile.density_n to be set "
                "(the full-range neutron-density table); this profile does "
                "not expose one."
            )
        n_n_samples = interp1d_linear(
            x=trajectory.sample_x,
            xp=profile.radius,
            fp=profile.density_n,
            left=profile.density_n[0],
            right=profile.density_n[-1],
            device=profile.device,
            dtype=profile.dtype,
        )

    # Add the trailing segment-broadcast dimension explicitly (mirrors
    # Tei's E_t[..., None] convention) so a batched E broadcasts against the
    # 1-D segment array instead of clashing with it.
    E_t = torch.as_tensor(E_MeV, device=profile.device, dtype=profile.dtype)[..., None]

    S_history = evolutor_numerical(
        oscillation,
        E_t,
        n_e_samples,
        trajectory.dx_evolution,
        n_n_mol_cm3=n_n_samples,
        return_history=True,
        device=profile.device,
        dtype=profile.dtype,
        evolution_scale_m=constant.R_SUN,
        legacy_precision=legacy_precision,
    )

    return S_history, trajectory


@torch.no_grad()
def solar_evolutor_numerical(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    profile,
    *,
    method: Optional[OdeMethod] = "midpoint",
    include_matter_nc: bool = False,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Production-point-to-endpoint evolutor for every production radius.

    Recovers S(r_end, r_k) from the shared history via
    S(r_end, r_k) = S(r_end, 0) @ S(r_k, 0)^dagger (exact since every
    S(r_j, 0) is unitary), then reads off the production-radius rows.

    Returns:
        Complex tensor shaped ``(..., n_r, N, N)``
        (``n_r = profile.production_radius.numel()``).
    """
    S_history, trajectory = solar_evolutor_numerical_history(
        oscillation,
        E_MeV,
        profile,
        method=method,
        include_matter_nc=include_matter_nc,
        legacy_precision=legacy_precision,
    )

    S_to_endpoint = S_history[..., -1:, :, :] @ S_history.conj().transpose(-1, -2)
    return S_to_endpoint[..., trajectory.meta["production_index"], :, :]


def mass_weights_numerical(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    profile,
    *,
    method: Optional[OdeMethod] = "midpoint",
    include_matter_nc: bool = False,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Exact per-production-point mass-basis weights (numerical counterpart of Tei).

    Propagates a pure electron-flavour state from every production radius to
    the trajectory endpoint with ``solar_evolutor_numerical``, then projects
    onto the vacuum mass basis. Shape ``(..., n_r, N)``, matching ``Tei``'s
    convention so both feed ``SolarProfile.mass_weights_integrate`` the same
    way.
    """
    n_flavours = int(oscillation.pmns.n_flavours)

    S = solar_evolutor_numerical(
        oscillation,
        E_MeV,
        profile,
        method=method,
        include_matter_nc=include_matter_nc,
        legacy_precision=legacy_precision,
    )  # (..., n_r, N, N)

    psi_e = flavour_state("e", device=profile.device, dtype=profile.dtype, n_flavours=n_flavours)
    amplitude_flavour = apply_evolutor_to_state(S, psi_e)  # (..., n_r, N)

    U = oscillation.pmns.pmns_matrix(antinu=oscillation.antinu)
    amplitude_mass = torch.einsum(
        "...ij,...j->...i", U.conj().transpose(-2, -1), amplitude_flavour,
    )
    return amplitude_mass.abs() ** 2  # (..., n_r, N)
