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

"""Numerical propagation from many origin points to one shared endpoint, without re-propagating per origin.

Generalises v1_0's own solar-specific "shared-history" trick (originally
``medium/solar/evolutor.py``) into a medium-agnostic technique: given a
structural grid and a set of origin points merged into it as exact
boundaries, propagate the numerical evolutor *once* along the whole
grid, then recover the evolutor from any origin to the shared endpoint
via unitarity --

    S(r_end, r_k) = S(r_end, 0) @ S(r_k, 0)^dagger

-- without a second matrix-exponential integration per origin. Nothing
here mentions the Sun: the same technique applies to any coherent
numerical propagation with many origin points sharing one destination
(today, only ``tpeanuts.pipeline.solar``'s own ``"numerical"`` method
uses it).

v2 has no equivalent of v1_0's ``evolutor_numerical(...,
return_history=True)`` (confirmed by direct search:
``tpeanuts.physics.evolution.numerical.numerical.evolve_path_numerical`` only ever
returns the fully-composed endpoint operator, via
``tpeanuts.physics.common.composition.compose_segment_evolutors``'s
O(log N)-depth tree reduction, never the intermediate partial products)
-- ``accumulate_history`` below is new code, a plain sequential
accumulation (``S_j = U_j @ S_{j-1}``), matching v1_0's own
implementation rather than a tree reduction.

``merge_origin_grid``/``accumulate_history``/``endpoint_evolutors_from_history``/
``mass_weights_from_evolutor`` are pure functions: building the per
-segment generator ``H_segments`` itself (which needs
``PropagationModel.generator_flavour``) is not their job -- same
"generator is a plain tensor argument" contract as
``tpeanuts.physics.evolution.numerical.numerical.segment_evolutor_numerical``.
``mass_weights_numerical`` (below) is the model-aware entry point built on top of
them: it takes a ``PropagationModel`` directly, builds ``H_segments`` via
``model.generator_flavour``, and sequences the four pure functions --
callers outside ``tpeanuts.physics.evolution`` itself (e.g.
``tpeanuts.pipeline.solar``) should use it rather than the pure functions
directly, exactly as
``tpeanuts.physics.evolution.dispatch.evolve_path``/
``tpeanuts.physics.evolution.adiabatic.adiabatic.mass_weights_adiabatic`` are the
model-aware entry points for their own families.

Module contents:
    merge_origin_grid(...)
        Merge a structural grid and a set of origin points into one
        sorted grid, with per-segment sample points and each origin's
        own index -- pure tensor geometry, no medium dependency.
    accumulate_history(...)
        Sequentially compose per-segment evolutors into the accumulated
        history from the grid's start to every boundary point.
    endpoint_evolutors_from_history(...)
        Recover the evolutor from every origin point to the shared
        endpoint, via unitarity.
    mass_weights_from_evolutor(...)
        Propagate a pure electron-flavour state through an evolutor and
        project the result onto the mass basis.
    mass_weights_numerical(...)
        Model-aware entry point: takes a ``PropagationModel`` and an
        already-merged grid (typically
        ``tpeanuts.medium.solar.solar.SolarMedium.
        numerical_segments``'s own output), and sequences the four pure
        functions above into the final mass-basis weights.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.common.space import FlavourSpace
from tpeanuts.physics.common.state import DensityMatrix
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution.numerical.numerical import segment_evolutor_numerical
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.constant import R_SUN
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import searchsorted_contiguous
from tpeanuts.util.tensor import TensorLike, as_tensor


def _segment_sample_points(x: torch.Tensor, method: str = "midpoint") -> torch.Tensor:
    """Per-segment sample points along ``x``'s boundaries -- see ``merge_origin_grid``.

    Args:
        x: Segment boundary coordinates, shape ``(N+1,)``.
        method: ``"midpoint"`` (default) returns ``0.5*(x[i]+x[i+1])``;
            ``"left"`` returns ``x[i]``; ``"right"`` returns ``x[i+1]``.

    Returns:
        Per-segment sample coordinates, shape ``(N,)``.

    Raises:
        ValueError: If ``method`` is not one of the three above.
    """
    if method == "midpoint":
        return 0.5 * (x[:-1] + x[1:])
    if method == "left":
        return x[:-1]
    if method == "right":
        return x[1:]
    raise ValueError(f"method must be 'midpoint', 'left', or 'right', got {method!r}.")


def merge_origin_grid(
    boundary_points: torch.Tensor,
    origin_points: torch.Tensor,
    *,
    sampling: str = "midpoint",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge a structural grid and a set of origin points into one trajectory.

    Every entry of ``origin_points`` is merged as an *exact* boundary of
    the returned grid ``x`` (not interpolated), which is what makes
    ``endpoint_evolutors_from_history``'s shared-history trick exact
    rather than approximate.

    Args:
        boundary_points: The medium's own structural grid, shape
            ``(n_b,)``.
        origin_points: Origin points to merge as exact trajectory
            boundaries, shape ``(n_r,)``.
        sampling: Segment density-sampling rule, forwarded to
            ``_segment_sample_points``.

    Returns:
        ``(x, dx_evolution, sample_x, origin_index)``: ``x`` is the
        merged, sorted, deduplicated grid, shape ``(n+1,)``;
        ``dx_evolution`` is its own segment lengths, shape ``(n,)``;
        ``sample_x`` is the per-segment density-sampling points, shape
        ``(n,)``; ``origin_index`` locates each ``origin_points`` entry
        within ``x``, shape ``(n_r,)``.
    """
    x = torch.unique(torch.cat([boundary_points, origin_points]), sorted=True)
    dx_evolution = x[1:] - x[:-1]
    sample_x = _segment_sample_points(x, sampling)
    origin_index = searchsorted_contiguous(x, origin_points)
    return x, dx_evolution, sample_x, origin_index


def accumulate_history(U_steps: torch.Tensor) -> torch.Tensor:
    """Sequentially compose per-segment evolutors into the accumulated history.

    Args:
        U_steps: Per-segment evolutors, shaped ``(..., n, N, N)`` -- e.g.
            ``tpeanuts.physics.evolution.numerical.numerical.segment_evolutor_numerical(H, dx)``.

    Returns:
        ``S_history``, shaped ``(..., n+1, N, N)``: the accumulated
        evolutor from the trajectory start (identity at index 0) to
        each boundary point, ``S_j = U_j @ S_{j-1}``.
    """
    batch_shape = U_steps.shape[:-3]
    n_flavour = U_steps.shape[-1]
    identity = torch.eye(n_flavour, device=U_steps.device, dtype=U_steps.dtype)
    S = identity.expand(*batch_shape, n_flavour, n_flavour).clone()

    S_list = [S.clone()]
    for j in range(U_steps.shape[-3]):
        S = U_steps[..., j, :, :] @ S
        S_list.append(S.clone())
    return torch.stack(S_list, dim=-3)


def endpoint_evolutors_from_history(S_history: torch.Tensor, origin_index: torch.Tensor) -> torch.Tensor:
    """Recover the evolutor from every origin point to the shared endpoint, via unitarity.

    ``S(r_end, r_k) = S(r_end, 0) @ S(r_k, 0)^dagger`` -- exact since
    every ``S(r_j, 0)`` is unitary, and cheaper than a second
    propagation per origin.

    Args:
        S_history: Accumulated evolutor history, shaped ``(..., n+1, N, N)``
            -- see ``accumulate_history``.
        origin_index: Index of each origin point within the grid
            ``S_history`` was built from, shape ``(n_r,)`` -- see
            ``merge_origin_grid``.

    Returns:
        Complex tensor shaped ``(..., n_r, N, N)``.
    """
    S_to_endpoint = S_history[..., -1:, :, :] @ S_history.conj().transpose(-1, -2)
    return S_to_endpoint[..., origin_index, :, :]


def mass_weights_from_evolutor(
    S: torch.Tensor,
    mixing: MixingMatrix,
    antinu: Union[bool, torch.Tensor],
    space: FlavourSpace,
    context: RuntimeContext,
) -> torch.Tensor:
    """Propagate a pure electron-flavour state through ``S`` and project onto the mass basis.

    ``DensityMatrix.basis_state(0, ...).conjugate_by(S)`` gives the
    evolved flavour-basis density matrix; conjugating that by
    ``mixing.pmns_matrix(antinu).conj().T`` rotates it into the mass
    basis, whose diagonal is exactly ``|amplitude_mass_i|^2``.

    Args:
        S: Evolutor(s) from a production point to the endpoint, shaped
            ``(..., N, N)``.
        mixing: Built MixingMatrix supplying the PMNS matrix.
        antinu: Selects the antineutrino PMNS convention -- must match
            whatever convention ``S`` was itself built with.
        space: FlavourSpace ``S`` acts on.
        context: RuntimeContext for the intermediate DensityMatrix
            construction.

    Returns:
        Real tensor shaped ``(..., N)``.
    """
    rho_e = DensityMatrix.basis_state(0, space, context)
    rho_flavour = rho_e.conjugate_by(S)

    U_dagger = mixing.pmns_matrix(antinu=antinu).conj().transpose(-2, -1)
    rho_mass = rho_flavour.conjugate_by(U_dagger)

    return rho_mass.probabilities()


def mass_weights_numerical(
    model: PropagationModel,
    E_MeV: TensorLike,
    dx_evolution: torch.Tensor,
    n_e_mol_cm3: torch.Tensor,
    n_n_mol_cm3: Optional[torch.Tensor],
    origin_index: torch.Tensor,
    *,
    evolution_scale_m: TensorLike = R_SUN,
) -> torch.Tensor:
    """Numerical mass-eigenstate weights at every production point, no adiabatic approximation.

    The model-aware entry point for the shared-origin technique: builds
    the per-segment generator via ``model.generator_flavour``, then
    sequences ``segment_evolutor_numerical`` -> ``accumulate_history`` ->
    ``endpoint_evolutors_from_history`` -> ``mass_weights_from_evolutor``
    -- propagating the numerical evolutor once along the whole merged
    grid rather than once per production point.

    Args:
        model: Built PropagationModel (mixing, mass spectrum, antinu, and
            any active extensions). Must have no active dissipators --
            this is a purely unitary path.
        E_MeV: Neutrino energy in MeV, scalar or batched.
        dx_evolution: Merged-grid segment lengths, shape ``(n,)`` --
            typically ``tpeanuts.medium.solar.solar.SolarMedium.
            numerical_segments``'s own first return value.
        n_e_mol_cm3: Electron density at each merged segment's sample
            point, shape ``(n,)`` -- paired with ``dx_evolution``.
        n_n_mol_cm3: Neutron density at each merged segment's sample
            point, shape ``(n,)``, or None if the medium has no neutron
            column.
        origin_index: Index of each production radius within the merged
            grid, shape ``(n_r,)`` -- see ``merge_origin_grid``.
        evolution_scale_m: Positive length scale in metres used to
            non-dimensionalize ``generator_flavour``. Defaults to the
            solar radius, since the shared-origin technique's only
            current use is solar-neutrino propagation.

    Returns:
        Real tensor shaped ``(..., n_r, N)``, matching
        ``tpeanuts.physics.evolution.adiabatic.adiabatic.mass_weights_adiabatic``'s
        own shape convention so both feed
        ``tpeanuts.source.solar.SolarSource.mass_weights_integrate``
        the same way.

    Raises:
        ValueError: If ``model.dissipators`` is non-empty.
    """
    if model.dissipators:
        raise ValueError("mass_weights_numerical is a purely unitary path -- model must have no active dissipators.")

    E_t = as_tensor(E_MeV, device=n_e_mol_cm3.device, dtype=n_e_mol_cm3.dtype)
    if E_t.ndim > 0:
        E_t = E_t.unsqueeze(-1)

    environment = PropagationEnvironment(E_MeV=E_t, n_e_mol_cm3=n_e_mol_cm3, n_n_mol_cm3=n_n_mol_cm3)
    H = model.generator_flavour(environment, evolution_scale_m=evolution_scale_m)  # (..., n_segments, N, N)
    U_steps = segment_evolutor_numerical(H, dx_evolution)  # (..., n_segments, N, N)
    S_history = accumulate_history(U_steps)
    S = endpoint_evolutors_from_history(S_history, origin_index)  # (..., n_r, N, N)

    return mass_weights_from_evolutor(S, model.mixing, model.antinu, model.space, model.context)  # (..., n_r, N)
