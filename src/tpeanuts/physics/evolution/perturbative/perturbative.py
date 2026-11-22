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

"""First-order perturbative correction for within-segment density variation.

``tpeanuts.physics.evolution.perturbative.analytic`` is exact only while
the Hamiltonian is genuinely constant across a segment. This module adds
the one thing that isn't true for -- density that varies continuously
within the segment, described by a
``tpeanuts.physics.evolution.perturbative.profile.EvenPowerProfile``-like
object -- on top of ``analytic``'s own exact zero-order result, reusing
its ``_segment_evolutor_from_spectral``/``hamiltonian_spectral_data``
machinery rather than a second, redundant spectral decomposition. Lives
alongside ``analytic.py`` in the same ``perturbative/`` folder for exactly
this reason: the two are variants of one closed-form technique, not
independent methods (see this package's own ``__init__.py`` docstring).

This is the one place in ``tpeanuts.physics.evolution`` where the result is
a genuine *approximation*, not exact: the correction is first order in the
density's own deviation from its segment average. It is strictly additive
and optional, though -- a profile with no variation to correct for
(``has_perturbation()`` False everywhere) reproduces
``tpeanuts.physics.evolution.perturbative.analytic``'s exact zero-order
result bit for bit (see this module's own tests), never a forced
approximation.

Module contents:
    PerturbativeProfile
        Protocol the density-profile argument must satisfy (``EvenPowerProfile`` and ``PolynomialProfile`` do).
    segment_evolutor_analytic_perturbative(...)
        ``tpeanuts.physics.evolution.perturbative.analytic.segment_evolutor_analytic``'s
        zero-order result, plus the first-order correction, for one
        segment (or an entire single-segment path).
    evolve_path_analytic_perturbative(...)
        Build and compose the per-segment perturbative evolutors for a
        whole path into one total evolution operator, via
        ``tpeanuts.physics.common.composition.compose_segment_evolutors``,
        unchanged and shared with every other evolver in this package.
"""

from __future__ import annotations

from typing import Optional, Protocol, Union

import torch

from tpeanuts.physics.common.composition import compose_segment_evolutors
from tpeanuts.physics.common.potential import matter_potential_cc, matter_potential_nc
from tpeanuts.physics.evolution.perturbative.analytic import _segment_evolutor_from_spectral
from tpeanuts.physics.evolution.perturbative.spectral import hamiltonian_spectral_data
from tpeanuts.util.constant import R_E
from tpeanuts.util.tensor import TensorLike


class PerturbativeProfile(Protocol):
    """The density-profile interface the first-order correction consumes (see ``profile.py``'s two implementations)."""

    def residual_integral(self, la: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        """Electron-density residual integral between eigenvalue pairs ``la``, ``lb``."""
        ...

    def residual_integral_neutron(self, la: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        """Neutron-density counterpart of ``residual_integral``."""
        ...

    def has_perturbation(self) -> torch.Tensor:
        """Boolean mask, True where the electron density deviates from its segment average."""
        ...

    def has_perturbation_neutron(self) -> torch.Tensor:
        """Neutron-density counterpart of ``has_perturbation``."""
        ...


def _evolutor_first_order(
    M: torch.Tensor,
    lam: torch.Tensor,
    trace_H: torch.Tensor,
    profile: PerturbativeProfile,
    *,
    antinu: Union[bool, torch.Tensor] = False,
    evolution_scale_m: TensorLike = R_E,
    P: Optional[torch.Tensor] = None,
    P_nc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the first-order correction for a within-segment density-variation profile.

    ``U1 = -i sum_{a,b} I_ab (M_a @ P @ M_b)``, where ``I_ab`` is the
    profile's own spectral oscillatory residual integral and ``P`` is the
    (position-independent) matter-potential direction matrix -- see
    ``tpeanuts.physics.model.PropagationModel``'s/
    ``tpeanuts.physics.extensions.nsi.NSITerm``'s/``tpeanuts.physics.
    extensions.sterile.SterileNCTerm``'s own optional ``direction_reduced``.
    When ``P_nc`` is also supplied, a second, independent sandwich term is
    added using the profile's own neutron-density residual integral
    (``profile.residual_integral_neutron``) and
    ``tpeanuts.physics.common.potential.matter_potential_nc``, mirroring
    the sterile neutral-current matter term's own zero-order contribution.

    Args:
        M: Spectral projectors shaped ``(..., N, N, N)``, N in ``{3, 4}``.
        lam: Traceless-Hamiltonian eigenvalues shaped ``(..., N)``.
        trace_H: Trace of the full dimensionless Hamiltonian.
        profile: Density-profile object exposing ``residual_integral(la,
            lb)`` (and, when ``P_nc`` is given, ``residual_integral_neutron``)
            -- see ``tpeanuts.physics.evolution.perturbative.profile.EvenPowerProfile``.
        antinu: Select the antineutrino matter-potential sign.
        evolution_scale_m: Length scale used to normalize the potential --
            must match whatever ``M``/``lam``/``trace_H``'s own generator
            was built with.
        P: Optional matter-potential direction matrix shaped ``(..., N,
            N)``, broadcastable against ``M``'s batch shape but
            independent of it (evaluated once "at unit potential" by the
            caller -- see ``direction_reduced``'s own docstrings). When
            None (the common plain-Standard-Model case), uses the
            closed-form rank-1 shortcut equivalent to ``P =
            diag(1,0,...,0)`` -- bit-identical to, and cheaper than, the
            general sandwich formula with that same ``P``.
        P_nc: Optional neutral-current matter-potential direction matrix,
            same shape/batch constraints as ``P``. None (the default)
            omits the neutral-current correction entirely.

    Returns:
        First-order correction U1 shaped ``(..., N, N)``.
    """
    N = M.shape[-1]
    eigenvalues_H = lam + trace_H[..., None] / N
    la = eigenvalues_H[..., :, None]
    lb = eigenvalues_H[..., None, :]

    integral = profile.residual_integral(la=la, lb=lb)
    potential_correction = matter_potential_cc(
        integral, antinu=antinu, evolution_scale_m=evolution_scale_m
    ).to(dtype=M.dtype)

    if P is None:
        U1 = (-1j) * torch.einsum(
            "...ab,...ai,...bj->...ij", potential_correction, M[..., :, :, 0], M[..., :, 0, :]
        )
    else:
        P = P.to(device=M.device, dtype=M.dtype)
        U1 = (-1j) * torch.einsum("...ab,...aik,...kl,...blj->...ij", potential_correction, M, P, M)

    if P_nc is not None:
        integral_n = profile.residual_integral_neutron(la=la, lb=lb)
        potential_correction_nc = matter_potential_nc(
            integral_n, antinu=antinu, evolution_scale_m=evolution_scale_m
        ).to(dtype=M.dtype)
        P_nc = P_nc.to(device=M.device, dtype=M.dtype)
        U1 = U1 + (-1j) * torch.einsum(
            "...ab,...aik,...kl,...blj->...ij", potential_correction_nc, M, P_nc, M
        )

    return U1


def segment_evolutor_analytic_perturbative(
    H: torch.Tensor,
    dx: TensorLike,
    perturbative_profile: PerturbativeProfile,
    *,
    antinu: Union[bool, torch.Tensor] = False,
    evolution_scale_m: TensorLike = R_E,
    P: Optional[torch.Tensor] = None,
    P_nc: Optional[torch.Tensor] = None,
    analytic_eigenvalues: bool = True,
) -> torch.Tensor:
    """Build the exact zero-order evolutor, plus the first-order perturbative correction.

    Args:
        H: Hermitian generator, shaped ``(..., N, N)``, ``N in {3, 4}`` --
            same contract as
            ``tpeanuts.physics.evolution.perturbative.analytic.segment_evolutor_analytic``'s
            own ``H``. Its own average matter term must correspond to the
            *same* density ``perturbative_profile.average``/``average_n``
            represents -- see
            ``tpeanuts.interfaces.medium.MediumProfile.
            perturbative_profile``'s own docstring for this consistency
            requirement.
        dx: Dimensionless path length(s) -- same contract as
            ``segment_evolutor_analytic``'s own ``dx``.
        perturbative_profile: Density-profile object exposing
            ``has_perturbation()``/``residual_integral`` (and, when ``P_nc``
            is given, ``has_perturbation_neutron()``/
            ``residual_integral_neutron``) -- see
            ``tpeanuts.physics.evolution.perturbative.profile.EvenPowerProfile``.
        antinu: Forwarded to ``_evolutor_first_order``.
        evolution_scale_m: Forwarded to ``_evolutor_first_order`` -- must
            match whatever ``H`` was itself built with.
        P: Optional matter-potential direction matrix forwarded to
            ``_evolutor_first_order``. None reproduces the plain Standard
            Model/sterile ``diag(1,0,...,0)`` direction.
        P_nc: Optional neutral-current direction matrix forwarded to
            ``_evolutor_first_order``. None (the default) omits the
            neutral-current correction entirely.
        analytic_eigenvalues: Forwarded to ``hamiltonian_spectral_data``.

    Returns:
        Complex tensor shaped ``(..., N, N)``: ``U0 + U1``, or exactly
        ``U0`` (bit for bit) wherever ``perturbative_profile.
        has_perturbation()`` (combined with ``has_perturbation_neutron()``
        when ``P_nc`` is given) is False.

    Raises:
        ValueError: If ``H.shape[-1]`` is not 3 or 4.
    """
    spectral = hamiltonian_spectral_data(H, analytic_eigenvalues=analytic_eigenvalues)
    U0 = _segment_evolutor_from_spectral(spectral, dx, device=H.device, real_dtype=H.real.dtype)

    has_perturbation = perturbative_profile.has_perturbation()
    if P_nc is not None:
        has_perturbation = has_perturbation | perturbative_profile.has_perturbation_neutron()

    if not bool(torch.any(has_perturbation)):
        # Every segment is constant -- skip the residual-integral computation
        # entirely when its result is guaranteed to be discarded.
        return U0

    U1 = _evolutor_first_order(
        spectral["M"], spectral["lam"], spectral["trace"], perturbative_profile,
        antinu=antinu, evolution_scale_m=evolution_scale_m, P=P, P_nc=P_nc,
    )
    mask = has_perturbation
    while mask.ndim < U1.ndim - 2:
        mask = mask.unsqueeze(-1)
    U1 = torch.where(mask[..., None, None], U1, torch.zeros_like(U1))
    return U0 + U1


def evolve_path_analytic_perturbative(
    H_segments: torch.Tensor,
    dx_segments: TensorLike,
    perturbative_profile: PerturbativeProfile,
    *,
    reverse: bool = False,
    antinu: Union[bool, torch.Tensor] = False,
    evolution_scale_m: TensorLike = R_E,
    P: Optional[torch.Tensor] = None,
    P_nc: Optional[torch.Tensor] = None,
    analytic_eigenvalues: bool = True,
) -> torch.Tensor:
    """Build and compose the per-segment perturbative evolutors for a whole path.

    Args:
        H_segments: Hermitian generators, one per segment, shaped
            ``(..., S, N, N)`` (``S`` segments).
        dx_segments: Dimensionless segment lengths, shaped ``(..., S)`` (or
            broadcastable against it).
        perturbative_profile: Forwarded to
            ``segment_evolutor_analytic_perturbative`` for every segment --
            a single profile shared across the whole path (build one per
            segment and stack it along the segment axis for a genuinely
            varying-by-segment medium).
        reverse: Forwarded to
            ``tpeanuts.physics.common.composition.compose_segment_evolutors``:
            if False (default), segment 0 is the first one a state passes
            through -- identical convention to every other evolver in this
            package.
        antinu: Forwarded to ``segment_evolutor_analytic_perturbative``.
        evolution_scale_m: Forwarded to
            ``segment_evolutor_analytic_perturbative`` -- must match
            whatever ``H_segments`` was itself built with.
        P: Forwarded to ``segment_evolutor_analytic_perturbative``.
        P_nc: Forwarded to ``segment_evolutor_analytic_perturbative``.
        analytic_eigenvalues: Forwarded to
            ``segment_evolutor_analytic_perturbative``.

    Returns:
        Complex tensor shaped ``(..., N, N)``, the total evolution operator
        for the whole path.

    Raises:
        ValueError: If ``H_segments.shape[-1]`` is not 3 or 4.
    """
    S_segments = segment_evolutor_analytic_perturbative(
        H_segments, dx_segments, perturbative_profile,
        antinu=antinu, evolution_scale_m=evolution_scale_m, P=P, P_nc=P_nc,
        analytic_eigenvalues=analytic_eigenvalues,
    )
    return compose_segment_evolutors(S_segments, reverse=reverse)
