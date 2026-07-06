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
earth evolution operators for peanuts-torch.

This module computes the full flavour-basis evolution operator for neutrinos
crossing the earth.

It sits above the core peanuts modules and delegates generic tensor
broadcasting and flattening to tpeanuts.util.torch_util. The core module
computes one local matter-segment evolutor. This earth module:

    1. Classifies the trajectory angle eta.
    2. Computes earth geometry.
    3. Queries the EarthProfile object for crossed shells.
    4. Builds and multiplies segment evolutors.
    5. Dresses the reduced evolution operator into the full flavour basis.
    6. Optionally projects the result onto the nearest unitary matrix.

Module functions:
    earth_evolutor(...)
        Build Earth-crossing operators while allowing independent polynomial
        profile and Hamiltonian evolution scales.
    earth_evolutor_from_zenith(...)
        Convert detector zenith angles and call ``earth_evolutor`` with the
        same scale configuration.
"""



from __future__ import annotations

import dataclasses
from typing import Union

import torch

import tpeanuts.util.default as default
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.util.type import TensorLike, as_tensor, cdtype_from_real
from tpeanuts.util.math import project_to_unitary
from tpeanuts.util.torch_util import (
    infer_device_dtype,
    broadcast_tensor,
    flat_tensor,
)
from tpeanuts.util.constant import R_E

from tpeanuts.core.common.evolutor import compose_segment_evolutors
from tpeanuts.core.perturbative.evolutor import evolutor_perturbative_segment
from tpeanuts.core.perturbative.models import PerturbativeSegmentBatch

from tpeanuts.medium.earth.geometry import (
    detector_radius_fraction,
    eta_prime_from_eta,
    detector_x_coordinate,
    chord_length_case_b,
    classify_eta_regions,
    validate_eta_range,
)

@torch.no_grad()
def _earth_evolutor_case_a_batched(
    profile_earth: object,
    oscillation: OscillationParameters,
    E_b: torch.Tensor,
    eta_b: torch.Tensor,
    depth_m: float,
    antinu: Union[bool, torch.Tensor],
    device: torch.device,
    rdtype: torch.dtype,
    cdtype: torch.dtype,
    profile_scale_m: TensorLike,
    evolution_scale_m: TensorLike,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Compute Earth evolution operators for trajectories crossing Earth shells.

    Args:
        profile_earth: EarthProfile object exposing ``trajectory_profile``.
        oscillation: Built pmns object plus mass splittings. Its own
            ``antinu`` is ignored here -- the per-region-masked ``antinu``
            argument below is the one actually used (and is folded into a
            fresh oscillation via ``dataclasses.replace`` before being
            passed to the segment evolutor).
        E_b: Flattened batch of neutrino energies in MeV for case A entries.
        eta_b: Flattened batch of nadir angles in radians for case A entries.
        depth_m: Detector depth in meters.
        antinu: Scalar or batched antineutrino selector for these entries.
        device: Torch device used for intermediate tensors.
        rdtype: Real floating dtype.
        cdtype: Complex dtype matching rdtype.
        profile_scale_m: Positive scale defining the profile coordinate passed
            to the core evolutor.
        evolution_scale_m: Positive scale defining H and evolution lengths.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor inside perturbative segment models and corrections.

    Returns:
        Tensor with shape `(batch_size, 3, 3)` containing full flavour-basis
        evolution operators.

    Notes:
        Case A covers paths that enter Earth matter deeply enough to cross one
        or more tabulated Earth shells. The implementation builds the shell
        segments for half of the chord, composes them, adds the detector-side
        segment, and dresses the reduced operator into the full flavour basis.
    """
    pmns = oscillation.pmns
    batch_size = eta_b.numel()

    if batch_size == 0:
        return torch.empty((0, 3, 3), device=device, dtype=cdtype)

    # Move the detector from the surface radius to its actual underground
    # radius. All subsequent chord coordinates are expressed in Earth-radius
    # units.
    r_d = detector_radius_fraction(
        depth_m,
        device=device,
        dtype=rdtype,
    )

    # Convert the detector nadir angle to the equivalent surface-crossing angle
    # used by the Earth-layer geometry.
    eta_prime = eta_prime_from_eta(
        eta_b,
        r_d,
    )

    # Coordinate of the detector point along the trajectory chord.
    x_d = detector_x_coordinate(
        eta_b,
        r_d,
    )

    # Build the model-layer view in trajectory coordinates. From this point on
    # Earth handles geometry and ordering, while the core model owns all
    # coefficient operations.
    trajectory_profile, xj_all, crossed = profile_earth.trajectory_profile(eta_prime)

    batch_size, ns = xj_all.shape

    # Find the outermost crossed shell. This gives the first material segment
    # between the Earth surface and the detector-side half path.
    outer = trajectory_profile.outermost_segment(
        xj_all=xj_all,
        crossed=crossed,
    )

    # Build ordered shell segments for the mirrored half-chord. The segment
    # fields include endpoints, opaque model data, and a boolean mask
    # indicating whether the segment is physically crossed for each trajectory.
    segments = trajectory_profile.ordered_segments(
        xj_all=xj_all,
        crossed=crossed,
    )

    profile_scale = as_tensor(profile_scale_m, device=device, dtype=rdtype)
    if torch.any(profile_scale <= 0):
        raise ValueError("profile_scale_m must be positive.")
    earth_to_profile = R_E / profile_scale
    coefficient_ratio = profile_scale / R_E

    # Add a segment dimension to energy so the perturbative segment evolutor
    # broadcasts over both batch and crossed-shell segments.
    E_seg = E_b.unsqueeze(-1)

    antinu_segments = antinu.unsqueeze(-1) if torch.is_tensor(antinu) else antinu

    # Compute one reduced matter evolutor per segment. Non-crossed segments are
    # replaced by identities below so they do not affect the composition.
    segment_model = trajectory_profile.segment_model(
        PerturbativeSegmentBatch(
            x1=segments.x1 * earth_to_profile,
            x2=segments.x2 * earth_to_profile,
            crossed=segments.crossed,
            model_data=segments.model_data,
        ),
        coordinate_ratio=coefficient_ratio,
        antinu=antinu_segments,
        profile_scale_m=profile_scale_m,
        evolution_scale_m=evolution_scale_m,
        device=device,
        dtype=rdtype,
        legacy_precision=legacy_precision,
    )
    U_segments = evolutor_perturbative_segment(
        dataclasses.replace(oscillation, antinu=antinu_segments),
        E_MeV=E_seg,
        profile_model=segment_model,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    I = torch.eye(3, device=device, dtype=cdtype).view(1, 1, 3, 3)

    # Mask out geometrically absent shell segments. This keeps a rectangular
    # tensor shape while preserving the correct product for each trajectory.
    U_segments = torch.where(
        segments.crossed.view(batch_size, ns, 1, 1),
        U_segments,
        I,
    )

    # Compose all crossed segments along the half chord. The multiplication
    # order follows the path ordering convention implemented in the core helper.
    U_half_full = compose_segment_evolutors(
        U_segments,
        segment_dim=1,
        multiply="right",
    )

    # Compute the final detector-side segment from the outer shell start to the
    # actual detector coordinate.
    detector_model = trajectory_profile.segment_model(
        PerturbativeSegmentBatch(
            x1=outer.x_start * earth_to_profile,
            x2=x_d * earth_to_profile,
            crossed=outer.has_any,
            model_data=outer.model_data,
        ),
        coordinate_ratio=coefficient_ratio,
        antinu=antinu,
        profile_scale_m=profile_scale_m,
        evolution_scale_m=evolution_scale_m,
        device=device,
        dtype=rdtype,
        legacy_precision=legacy_precision,
    )
    U0_det = evolutor_perturbative_segment(
        dataclasses.replace(oscillation, antinu=antinu),
        E_MeV=E_b,
        profile_model=detector_model,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    # Compose the detector-side half path excluding the duplicated first segment
    # and prepend the detector segment.
    U_half_det = U0_det @ compose_segment_evolutors(
        U_segments[:, 1:],
        segment_dim=1,
        multiply="right",
    )

    # Combine detector-side and far-side half paths into the reduced evolution
    # operator. The transpose implements the mirrored traversal of the first
    # half under the real symmetric reduced Hamiltonian convention.
    U_red = U_half_det @ U_half_full.transpose(-1, -2)

    # Transform the reduced evolution operator to the full flavour basis.
    U = pmns.operator_flavour_basis(
        U_red,
        antinu=antinu,
        device=device,
        dtype=cdtype,
    )

    I_full = torch.eye(3, device=device, dtype=cdtype).view(1, 3, 3)

    # Some case-A entries can still have no material shell after geometric
    # filtering. Keep those as identity operators.
    U = torch.where(
        outer.has_any.view(batch_size, 1, 1),
        U,
        I_full,
    )

    return U


@torch.no_grad()
def _earth_evolutor_case_b_batched(
    profile_earth: object,
    oscillation: OscillationParameters,
    E_b: torch.Tensor,
    eta_b: torch.Tensor,
    depth_m: float,
    antinu: Union[bool, torch.Tensor],
    device: torch.device,
    rdtype: torch.dtype,
    cdtype: torch.dtype,
    profile_scale_m: TensorLike,
    evolution_scale_m: TensorLike,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Compute Earth evolution operators for shallow detector-near trajectories.

    Args:
        profile_earth: EarthProfile object exposing ``call(x, eta)``.
        oscillation: Built pmns object plus mass splittings. Its own
            ``antinu`` is ignored here -- the per-region-masked ``antinu``
            argument below is the one actually used (and is folded into a
            fresh oscillation via ``dataclasses.replace`` before being
            passed to the segment evolutor).
        E_b: Flattened batch of neutrino energies in MeV for case B entries.
        eta_b: Flattened batch of nadir angles in radians for case B entries.
        depth_m: Detector depth in meters.
        antinu: Scalar or batched antineutrino selector for these entries.
        device: Torch device used for intermediate tensors.
        rdtype: Real floating dtype.
        cdtype: Complex dtype matching rdtype.
        profile_scale_m: Positive scale defining the segment coordinate.
        evolution_scale_m: Positive scale defining H and evolution lengths.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor inside the constant segment model and correction.

    Returns:
        Tensor with shape `(batch_size, 3, 3)` containing full flavour-basis
        evolution operators.

    Notes:
        Case B represents short paths between the Earth surface and the
        detector. The matter profile is approximated at the mid-depth radius
        and applied as a single constant-profile segment.
    """
    pmns = oscillation.pmns
    batch_size = eta_b.numel()

    if batch_size == 0:
        return torch.empty((0, 3, 3), device=device, dtype=cdtype)

    # Case B covers short trajectories from the surface to a shallow detector.
    # The detector radius determines the chord length inside matter.
    r_d = detector_radius_fraction(
        depth_m,
        device=device,
        dtype=rdtype,
    )

    # Approximate the profile along the short path by evaluating it at the
    # midpoint radius between the surface and the detector.
    h = float(depth_m) / float(R_E)
    r_mid = 1.0 - h / 2.0

    n1 = profile_earth.call(
        torch.tensor(r_mid, device=device, dtype=rdtype),
        torch.tensor(0.0, device=device, dtype=rdtype),
    )

    # Physical path length inside Earth matter for each shallow trajectory.
    deltax = chord_length_case_b(
        eta_b,
        r_d,
    )

    profile_scale = as_tensor(profile_scale_m, device=device, dtype=rdtype)
    if torch.any(profile_scale <= 0):
        raise ValueError("profile_scale_m must be positive.")
    earth_to_profile = R_E / profile_scale

    # Treat the shallow path as a single constant-profile matter segment in the
    # reduced basis.
    segment_model = profile_earth.constant_segment_model(
        x2=deltax * earth_to_profile,
        x1=torch.zeros_like(deltax),
        density=n1,
        antinu=antinu,
        profile_scale_m=profile_scale_m,
        evolution_scale_m=evolution_scale_m,
        device=device,
        dtype=rdtype,
        legacy_precision=legacy_precision,
    )
    U_red = evolutor_perturbative_segment(
        dataclasses.replace(oscillation, antinu=antinu),
        E_MeV=E_b,
        profile_model=segment_model,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    # Transform the reduced evolution operator to the full flavour basis.
    U = pmns.operator_flavour_basis(
        U_red,
        antinu=antinu,
        device=device,
        dtype=cdtype,
    )

    return U


@torch.no_grad()
def earth_evolutor(
    profile_earth: object,
    oscillation: OscillationParameters,
    E: TensorLike,
    eta: TensorLike,
    depth_m: float,
    *,
    reunitarize: bool = default.earth_reunitarize,
    profile_scale_m: TensorLike = R_E,
    evolution_scale_m: TensorLike = R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Compute the full Earth matter evolution operator in flavour basis.

    Args:
        profile_earth: EarthProfile object. It must provide the trajectory-profile
            interface used by case A and the direct profile call used by case B.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E: Neutrino energy in MeV. Scalars, tensors, and broadcastable grids are
            accepted.
        eta: Detector nadir angle in radians. Scalars, tensors, and
            broadcastable grids are accepted.
        depth_m: Detector depth below the Earth surface in meters.
        reunitarize: If True, project computed evolution matrices onto the
            nearest unitary matrix.
        profile_scale_m: Positive scale in metres used to reexpress the Earth
            perturbative profile before calling the core evolutor.
        evolution_scale_m: Positive scale in metres used to normalize the
            Hamiltonian and propagation coordinate.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in Earth matter propagation.
    Returns:
        Complex tensor with shape `(*broadcast_shape(E, eta), 3, 3)`. Entries
        above the Earth horizon remain the identity operator; case A and case B
        entries are filled with their corresponding matter evolution operators.

    Notes:
        - `U_a` and `U_b` only contain the subset of operators selected by their
        masks. 
        - `flat_out` is a flattened view used for indexed assignment.
        - The returned `out` preserves the original broadcast grid shape, 
        which is why returning `out` is the correct final result.
    """
    antinu = oscillation.antinu

    # Build a common E/eta grid. For independent 1D grids this creates the
    # outer-product shape, so the final result can preserve `(n_E, n_eta, 3, 3)`.
    device, rdtype = infer_device_dtype(E, eta)
    cdtype = cdtype_from_real(rdtype)
    E_b, eta_b = broadcast_tensor(
        E,
        eta,
        device=device,
        dtype=rdtype,
        independent_1d=True,
    )
    # Broadcast the boolean selector over the same generic grid as energy and
    # angle while retaining its logical dtype.
    antinu_b, _ = broadcast_tensor(
        antinu,
        E_b,
        device=device,
        dtype=(torch.bool, rdtype),
    )

    validate_eta_range(eta_b)

    # The masks split the broadcast grid into trajectories handled by different
    # approximations. Entries outside case A/B stay as identity operators.
    above, mask_a, mask_b = classify_eta_regions(
        eta_b,
        depth_m,
    )
    _ = above  # Above-horizon entries are intentionally left as identities.

    identity = torch.eye(
        3,
        device=device,
        dtype=cdtype,
    )

    # `out` has the final broadcast shape. It starts as identity everywhere so
    # untouched entries already represent no Earth matter propagation.
    out = identity.expand(*eta_b.shape, 3, 3).clone()

    # `flat_out` is a view into `out`; assigning to selected flat indices updates
    # the final-shaped output without losing the original broadcast dimensions.
    flat_out = out.reshape(-1, 3, 3)
    E_flat = E_b.reshape(-1)
    eta_flat = eta_b.reshape(-1)

    # Case A: shell-crossing Earth trajectories.
    flat_idx_a = torch.nonzero(
        mask_a.reshape(-1),
        as_tuple=False,
    ).squeeze(-1)

    U_a = _earth_evolutor_case_a_batched(
        profile_earth=profile_earth,
        oscillation=oscillation,
        E_b=E_flat[flat_idx_a],
        eta_b=eta_flat[flat_idx_a],
        depth_m=depth_m,
        antinu=flat_tensor(antinu_b, flat_idx_a),
        device=device,
        rdtype=rdtype,
        cdtype=cdtype,
        profile_scale_m=profile_scale_m,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    if reunitarize:
        U_a = project_to_unitary(U_a)

    # Insert only the case-A subset into the flattened view of the full output.
    flat_out[flat_idx_a] = U_a

    # Case B: shallow detector-near trajectories.
    flat_idx_b = torch.nonzero(
        mask_b.reshape(-1),
        as_tuple=False,
    ).squeeze(-1)

    U_b = _earth_evolutor_case_b_batched(
        profile_earth=profile_earth,
        oscillation=oscillation,
        E_b=E_flat[flat_idx_b],
        eta_b=eta_flat[flat_idx_b],
        depth_m=depth_m,
        antinu=flat_tensor(antinu_b, flat_idx_b),
        device=device,
        rdtype=rdtype,
        cdtype=cdtype,
        profile_scale_m=profile_scale_m,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    if reunitarize:
        U_b = project_to_unitary(U_b)

    # Insert the case-B subset. Entries in neither mask are still identities.
    flat_out[flat_idx_b] = U_b

    if reunitarize:
        # Reproject the final shaped tensor as a last numerical cleanup after
        # indexed assembly. This preserves the broadcast dimensions.
        out = project_to_unitary(out)

    return out


@torch.no_grad()
def earth_evolutor_from_zenith(
    profile_earth: object,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    theta_deg: TensorLike,
    depth_m: float,
    *,
    reunitarize: bool = default.earth_reunitarize,
    profile_scale_m: TensorLike = R_E,
    evolution_scale_m: TensorLike = R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Build the Earth evolutor from zenith angles with configurable scales.

    Convenience wrapper around ``earth_evolutor`` for callers that work with
    the detector zenith angle ``theta`` (angle from the local vertical,
    measured in degrees) instead of the nadir angle ``eta`` used internally.
    The two are related by ``eta = pi - theta_rad``, so a neutrino arriving
    from directly overhead (``theta=0``) maps to ``eta=pi`` (no Earth
    crossing), while one arriving from directly below (``theta=180``) maps
    to ``eta=0`` (straight through the Earth's centre).

    Args:
        profile_earth: EarthProfile object. It must provide the
            trajectory-profile interface used by case A and the direct
            profile call used by case B.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV. Scalars, tensors, and broadcastable
            grids are accepted.
        theta_deg: Detector zenith angle in degrees, measured from the local
            vertical (0 = overhead, 180 = straight up through the Earth).
            Scalars, tensors, and broadcastable grids are accepted.
        depth_m: Detector depth below the Earth surface in metres.
        reunitarize: If True, project computed evolution matrices onto the
            nearest unitary matrix.
        profile_scale_m: Positive scale in metres used to reexpress the Earth
            perturbative profile before calling the core evolutor.
        evolution_scale_m: Positive scale in metres used to normalize the
            Hamiltonian and propagation coordinate.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in Earth matter propagation.

    Returns:
        Complex tensor with shape `(*broadcast_shape(E_MeV, theta_deg), 3, 3)`
        containing full flavour-basis Earth evolution operators, identical in
        meaning to the output of ``earth_evolutor``.
    """
    device, dtype = infer_device_dtype(E_MeV, theta_deg)
    theta = as_tensor(theta_deg, device=device, dtype=dtype)
    eta = torch.pi - torch.deg2rad(theta)

    return earth_evolutor(
        profile_earth=profile_earth,
        oscillation=oscillation,
        E=E_MeV,
        eta=eta,
        depth_m=depth_m,
        reunitarize=reunitarize,
        profile_scale_m=profile_scale_m,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )
