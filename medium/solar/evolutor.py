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

"""Coherent numerical propagation through the solar density profile.

The profile is represented by constant-density segments and evolved with
``core.numerical``. No adiabatic or separate Landau--Zener approximation is
imposed, but accuracy depends on the radial grid and segment sampling rule.

One history ``S(r_j, 0)`` is reused for every production point through
``S(r_end, r_k) = S(r_end, 0) @ S(r_k, 0)^dagger``. The Hamiltonian supports
the same SM, NSI, sterile and neutral-current terms as the common evolutor.
"""

from __future__ import annotations

from typing import Optional

import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.evolutor import apply_evolutor_to_state
from tpeanuts.core.common.neutrino import flavour_state
from tpeanuts.core.common.oscillation import OscillationParameters, oscillation_needs_neutron_composition
from tpeanuts.core.numerical.evolutor import evolutor_numerical
from tpeanuts.core.numerical.geometry import OdeMethod, Trajectory, segment_sample_points
from tpeanuts.util.math import interp1d_linear
from tpeanuts.util.type import TensorLike


def build_solar_trajectory(
    medium,
    initial_radius: torch.Tensor,
    *,
    method: Optional[OdeMethod] = "midpoint",
) -> Trajectory:
    """Merge the full density grid and production grid into one trajectory.

    Segment boundaries are the sorted union of ``medium.radius`` (full
    density table) and ``initial_radius`` (the source's production-radius
    grid), so every production point is an exact boundary -- required for
    the shared-history endpoint trick in ``solar_evolutor_numerical`` to be
    exact rather than interpolated.

    Returns:
        ``Trajectory`` with the merged grid ``x``, dimensionless
        ``dx_evolution``, per-segment ``sample_x``, and
        ``meta["production_index"]`` locating ``initial_radius`` within
        ``x``.
    """
    x = torch.unique(torch.cat([medium.radius, initial_radius]), sorted=True)
    dx_evolution = x[1:] - x[:-1]
    sample_x = segment_sample_points(x, method)
    production_index = torch.searchsorted(x, initial_radius)

    return Trajectory(
        x=x,
        dx_evolution=dx_evolution,
        sample_x=sample_x,
        meta={"kind": "solar", "production_index": production_index},
    )


def solar_evolutor_numerical_history(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    medium,
    initial_radius: torch.Tensor,
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
        medium: SolarMediumProfile-like object exposing the full ``radius``/
            ``density`` grid and, when ``include_matter_nc=True``,
            ``density_n``.
        initial_radius: Production-radius grid to merge as exact trajectory
            boundaries (typically a ``source.solar.SolarNeutrinoSource``'s
            ``production_radius``).
        include_matter_nc: If True, also sample and apply the 3+1 sterile
            neutral-current term via ``medium.density_n``. Only meaningful
            for a 4-flavour ``oscillation.pmns``; silently ignored otherwise.
            Independent of and orthogonal to the NSI composition term:
            ``medium.density_n`` is sampled whenever this is True *or*
            ``oscillation.nsi.has_neutron_coupling`` is True (auto-detected,
            no separate flag needed, any flavour count -- see
            ``core.BSM.bsm_nsi``'s "Composition dependence" section);
            ``core.numerical.evolutor_numerical``/``hamiltonian_reduced``
            then apply each term on its own once the sample is supplied.

    Returns:
        ``(S_history, trajectory)``: complex tensor shaped ``(..., n, N, N)``
        (N = 3 or 4), the accumulated evolutor from the trajectory start to
        each merged-grid point (identity at index 0), plus the ``Trajectory``
        used to build it.

    Raises:
        ValueError: If ``include_matter_nc=True`` and/or
            ``oscillation.nsi.has_neutron_coupling`` is True and
            ``medium.density_n`` is not set.
    """
    trajectory = build_solar_trajectory(medium, initial_radius, method=method)

    n_e_samples = interp1d_linear(
        x=trajectory.sample_x,
        xp=medium.radius,
        fp=medium.density,
        left=medium.density[0],
        right=medium.density[-1],
        device=medium.device,
        dtype=medium.dtype,
    )

    needs_eps_n = oscillation_needs_neutron_composition(oscillation)
    n_n_samples = None
    if include_matter_nc or needs_eps_n:
        if medium.density_n is None:
            reason = (
                "oscillation.nsi has non-zero eps_*_n (composition-dependent NSI)"
                if needs_eps_n and not include_matter_nc
                else "include_matter_nc=True"
            )
            raise ValueError(
                f"{reason} requires medium.density_n to be set "
                "(the full-range neutron-density table); this medium does "
                "not expose one."
            )
        n_n_samples = interp1d_linear(
            x=trajectory.sample_x,
            xp=medium.radius,
            fp=medium.density_n,
            left=medium.density_n[0],
            right=medium.density_n[-1],
            device=medium.device,
            dtype=medium.dtype,
        )

    # Add the trailing segment-broadcast dimension explicitly (mirrors
    # Tei's E_t[..., None] convention) so a batched E broadcasts against the
    # 1-D segment array instead of clashing with it.
    E_t = torch.as_tensor(E_MeV, device=medium.device, dtype=medium.dtype)[..., None]

    S_history = evolutor_numerical(
        oscillation,
        E_t,
        n_e_samples,
        trajectory.dx_evolution,
        n_n_mol_cm3=n_n_samples,
        return_history=True,
        device=medium.device,
        dtype=medium.dtype,
        evolution_scale_m=constant.R_SUN,
        legacy_precision=legacy_precision,
    )

    return S_history, trajectory


def solar_evolutor_numerical(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    medium,
    initial_radius: torch.Tensor,
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
        Complex tensor shaped ``(..., n_r, N, N)`` (``n_r =
        initial_radius.numel()``).
    """
    S_history, trajectory = solar_evolutor_numerical_history(
        oscillation,
        E_MeV,
        medium,
        initial_radius,
        method=method,
        include_matter_nc=include_matter_nc,
        legacy_precision=legacy_precision,
    )

    S_to_endpoint = S_history[..., -1:, :, :] @ S_history.conj().transpose(-1, -2)
    return S_to_endpoint[..., trajectory.meta["production_index"], :, :]


def mass_weights_numerical(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    medium,
    initial_radius: torch.Tensor,
    *,
    method: Optional[OdeMethod] = "midpoint",
    include_matter_nc: bool = False,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Compute numerical mass weights at every production point.

    Propagates a pure electron-flavour state from every production radius to
    the trajectory endpoint with ``solar_evolutor_numerical``, then projects
    onto the vacuum mass basis. Shape ``(..., n_r, N)``, matching ``Tei``'s
    convention so both feed
    ``source.solar.SolarNeutrinoSource.mass_weights_integrate`` the same
    way.
    """
    n_flavours = int(oscillation.pmns.n_flavours)

    S = solar_evolutor_numerical(
        oscillation,
        E_MeV,
        medium,
        initial_radius,
        method=method,
        include_matter_nc=include_matter_nc,
        legacy_precision=legacy_precision,
    )  # (..., n_r, N, N)

    psi_e = flavour_state("e", device=medium.device, dtype=medium.dtype, n_flavours=n_flavours)
    amplitude_flavour = apply_evolutor_to_state(S, psi_e)  # (..., n_r, N)

    U = oscillation.pmns.pmns_matrix(antinu=oscillation.antinu)
    amplitude_mass = torch.einsum(
        "...ij,...j->...i", U.conj().transpose(-2, -1), amplitude_flavour,
    )
    return amplitude_mass.abs() ** 2  # (..., n_r, N)
