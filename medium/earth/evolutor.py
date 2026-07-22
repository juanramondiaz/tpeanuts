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

from typing import Optional, Union

import torch

import tpeanuts.config.default as default
from tpeanuts.core.common.oscillation import OscillationParameters, resolve_include_matter_nc
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
    identity3: torch.Tensor,
    profile_scale_m: TensorLike,
    evolution_scale_m: TensorLike,
    n_flavours: int = 3,
    legacy_precision: bool = False,
    include_matter_nc: bool = False,
) -> torch.Tensor:
    """
    Compute Earth evolution operators for trajectories crossing Earth shells.

    Args:
        profile_earth: EarthProfile object exposing ``trajectory_profile``.
        oscillation: Built PMNS object plus mass splittings and the optional
            ``nsi`` (NSIConfig) attribute. The per-region ``antinu`` argument
            below is passed separately to the segment evolutor.
        E_b: Flattened batch of neutrino energies in MeV for case A entries.
        eta_b: Flattened batch of nadir angles in radians for case A entries.
        depth_m: Detector depth in meters.
        antinu: Scalar or batched antineutrino selector for these entries.
        device: Torch device used for intermediate tensors.
        rdtype: Real floating dtype.
        cdtype: Complex dtype matching rdtype.
        identity3: NxN identity matrix (name kept for historical reasons; N is
            ``n_flavours``, not always 3).
        profile_scale_m: Positive scale defining the profile coordinate passed
            to the core evolutor.
        evolution_scale_m: Positive scale defining H and evolution lengths.
        n_flavours: Flavour count of ``oscillation.pmns`` (3 for the SM, 4 for
            the 3+1 sterile extension).
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor inside perturbative segment models and corrections.
        include_matter_nc: If True, also apply the 3+1 sterile extension's
            neutral-current matter term using ``profile_earth``'s
            neutron-density data (only meaningful for a 4-flavour
            ``oscillation.pmns``; requires the profile to have been built
            with neutron-density coefficients, see
            ``core.perturbative.evolutor.evolutor_perturbative_segment``).

    Returns:
        Tensor with shape `(batch_size, n_flavours, n_flavours)` containing
        full flavour-basis evolution operators.

    Notes:
        Case A covers paths that enter Earth matter deeply enough to cross one
        or more tabulated Earth shells. The implementation builds the shell
        segments for half of the chord, composes them, adds the detector-side
        segment, and dresses the reduced operator into the full flavour basis.
    """
    pmns = oscillation.pmns
    batch_size = eta_b.numel()

    if batch_size == 0:
        return torch.empty((0, n_flavours, n_flavours), device=device, dtype=cdtype)

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
        oscillation,
        E_MeV=E_seg,
        profile_model=segment_model,
        antinu=antinu_segments,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
    )

    I = identity3.view(1, 1, n_flavours, n_flavours)

    # Mask out geometrically absent shell segments. This keeps a rectangular
    # tensor shape while preserving the correct product for each trajectory.
    U_segments = torch.where(
        segments.crossed.view(batch_size, ns, 1, 1),
        U_segments,
        I,
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
        oscillation,
        E_MeV=E_b,
        profile_model=detector_model,
        antinu=antinu,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
    )

    # Compose the detector-side half path excluding the duplicated first segment
    # and prepend the detector segment.
    U_half_det = U0_det @ compose_segment_evolutors(
        U_segments[:, 1:],
        segment_dim=1,
        multiply="right",
    )

    # Build the far half directly from its own mirrored segments, rather than
    # reusing U_segments via a transpose shortcut. The chord coordinate x
    # satisfies r**2 = x**2 + sin(eta)**2, so the physical density profile is
    # exactly even in x (n(x) == n(-x)) and every crossed shell on the near
    # side [x1, x2] (0 <= x1 <= x2) has a mirror-image counterpart on the far
    # side spanning [-x2, -x1] with identical even-power coefficients -- the
    # "even_power" model's polynomial n(x) = a + b*x**2 + c*x**4 + ... is
    # invariant under this substitution by construction.
    #
    # This mirrored range is NOT merely a relabelling: the first-order
    # correction's oscillatory residual integral (see
    # ``EvenPowerProfileSegment.residual_integral``) depends on x1/x2 through
    # a complex phase that is not itself even in x, so the mirrored segment's
    # evolutor genuinely differs from the near-side one and must be computed
    # directly through the same (already-validated)
    # ``evolutor_perturbative_segment`` pipeline -- not derived from it via
    # transpose or reordering. A transpose-based shortcut (or a reordering of
    # the same near-side operators) is only proven correct when the reduced
    # Hamiltonian is real (delta14 == 0 for the 3+1 sterile extension); this
    # direct recomputation is correct unconditionally, independent of that
    # assumption, at the cost of one extra segment-evolutor pass.
    mirrored_segment_model = trajectory_profile.segment_model(
        PerturbativeSegmentBatch(
            x1=-segments.x2 * earth_to_profile,
            x2=-segments.x1 * earth_to_profile,
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
    U_segments_mirrored = evolutor_perturbative_segment(
        oscillation,
        E_MeV=E_seg,
        profile_model=mirrored_segment_model,
        antinu=antinu_segments,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
    )
    U_segments_mirrored = torch.where(
        segments.crossed.view(batch_size, ns, 1, 1),
        U_segments_mirrored,
        I,
    )
    # segments is ordered outermost-first (index 0) to innermost-last (index
    # ns-1) for the near-side "exit" traversal (multiply="right" composes it
    # innermost-applied-first, outermost-applied-last, i.e. propagating
    # outward from the point of closest approach). The mirrored far-side
    # traversal enters from the far surface inward, so it encounters the
    # mirrored outermost shell first and the mirrored innermost shell last --
    # the reverse composition order (multiply="left") of the very same
    # (freshly computed, correctly mirrored) per-segment operators.
    U_half_full_mirrored = compose_segment_evolutors(
        U_segments_mirrored,
        segment_dim=1,
        multiply="left",
    )

    # Combine detector-side and far-side half paths into the reduced
    # evolution operator.
    U_red = U_half_det @ U_half_full_mirrored

    # Transform the reduced evolution operator to the full flavour basis.
    U = pmns.flavour_basis(
        U_red,
        antinu=antinu,
        device=device,
        dtype=cdtype,
    )

    I_full = identity3.view(1, n_flavours, n_flavours)

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
    n_flavours: int = 3,
    legacy_precision: bool = False,
    include_matter_nc: bool = False,
) -> torch.Tensor:
    """
    Compute Earth evolution operators for shallow detector-near trajectories.

    Args:
        profile_earth: EarthProfile object exposing ``call(x, eta)``.
        oscillation: Built PMNS object plus mass splittings and the optional
            ``nsi`` (NSIConfig) attribute. The per-region ``antinu`` argument
            below is passed separately to the segment evolutor.
        E_b: Flattened batch of neutrino energies in MeV for case B entries.
        eta_b: Flattened batch of nadir angles in radians for case B entries.
        depth_m: Detector depth in meters.
        antinu: Scalar or batched antineutrino selector for these entries.
        device: Torch device used for intermediate tensors.
        rdtype: Real floating dtype.
        cdtype: Complex dtype matching rdtype.
        profile_scale_m: Positive scale defining the segment coordinate.
        evolution_scale_m: Positive scale defining H and evolution lengths.
        n_flavours: Flavour count of ``oscillation.pmns`` (3 for the SM, 4 for
            the 3+1 sterile extension).
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor inside the constant segment model and correction.
        include_matter_nc: If True, also apply the 3+1 sterile extension's
            neutral-current matter term using ``profile_earth``'s
            neutron-density data (only meaningful for a 4-flavour
            ``oscillation.pmns``; requires ``profile_earth`` to expose
            ``call_neutron`` -- i.e. the profile was built with
            neutron-density coefficients).

    Returns:
        Tensor with shape `(batch_size, n_flavours, n_flavours)` containing
        full flavour-basis evolution operators.

    Notes:
        Case B represents short paths between the Earth surface and the
        detector. The matter profile is approximated at the mid-depth radius
        and applied as a single constant-profile segment.
    """
    pmns = oscillation.pmns
    batch_size = eta_b.numel()

    if batch_size == 0:
        return torch.empty((0, n_flavours, n_flavours), device=device, dtype=cdtype)

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

    n1 = profile_earth.density_x_eta(
        torch.tensor(r_mid, device=device, dtype=rdtype),
        torch.tensor(0.0, device=device, dtype=rdtype),
    )
    n1_n = (
        profile_earth.density_n_x_eta(
            torch.tensor(r_mid, device=device, dtype=rdtype),
            torch.tensor(0.0, device=device, dtype=rdtype),
        )
        if include_matter_nc
        else None
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
        density_n=n1_n,
        antinu=antinu,
        profile_scale_m=profile_scale_m,
        evolution_scale_m=evolution_scale_m,
        device=device,
        dtype=rdtype,
        legacy_precision=legacy_precision,
    )
    U_red = evolutor_perturbative_segment(
        oscillation,
        E_MeV=E_b,
        profile_model=segment_model,
        antinu=antinu,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
    )

    # Transform the reduced evolution operator to the full flavour basis.
    U = pmns.flavour_basis(
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
    include_matter_nc: Optional[bool] = None,
) -> torch.Tensor:
    """
    Compute the full Earth matter evolution operator in flavour basis.

    Supports both the 3-flavour Standard Model and the 3+1 sterile extension
    (``oscillation.pmns.n_flavours == 4``, consumed by
    ``evolutor_perturbative_segment`` via ``oscillation.mass_spectrum.DeltamSq41``), and
    optional NSI via ``oscillation.nsi``, active for either flavour count. The
    identity operator and empty-batch tensors used internally are sized by
    ``oscillation.pmns.n_flavours``, derived once here and threaded through
    both trajectory cases.

    Args:
        profile_earth: EarthProfile object. It must provide the trajectory-profile
            interface used by case A and the direct profile call used by case B.
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
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
        include_matter_nc: If True/False, applied/not applied using
            ``profile_earth``'s neutron-density data (only meaningful when
            ``oscillation.pmns`` is 4-flavour); an explicit ``True`` still
            raises if ``profile_earth`` lacks neutron-density coefficients
            (e.g. ``EvenPowerProfileLayered``/``PremTabulatedProfile`` built
            without ``include_neutron=True``). If ``None`` (the default),
            auto-resolved per-call by ``core.common.oscillation.
            resolve_include_matter_nc``: ``True`` when ``oscillation`` is
            the 3+1 sterile extension and ``profile_earth.
            has_neutron_density`` is True, ``False`` otherwise (with a
            ``RuntimeWarning`` if sterile was requested but the profile
            lacks neutron-density data). Always ``False`` for the plain
            3-flavour case.
    Returns:
        Complex tensor with shape `(*broadcast_shape(E, eta), N, N)`, N in
        {3, 4}. Entries above the Earth horizon remain the identity operator;
        case A and case B entries are filled with their corresponding matter
        evolution operators.

    Notes:
        - `U_a` and `U_b` only contain the subset of operators selected by their
        masks.
        - `flat_out` is a flattened view used for indexed assignment.
        - The returned `out` preserves the original broadcast grid shape,
        which is why returning `out` is the correct final result.
    """
    antinu = oscillation.antinu
    n_flavours = int(oscillation.pmns.n_flavours)
    include_matter_nc = resolve_include_matter_nc(
        include_matter_nc,
        oscillation,
        has_neutron_data=getattr(profile_earth, "has_neutron_density", False),
        context_name="earth_evolutor",
    )

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
        n_flavours,
        device=device,
        dtype=cdtype,
    )

    # `out` has the final broadcast shape. It starts as identity everywhere so
    # untouched entries already represent no Earth matter propagation.
    out = identity.expand(*eta_b.shape, n_flavours, n_flavours).clone()

    # `flat_out` is a view into `out`; assigning to selected flat indices updates
    # the final-shaped output without losing the original broadcast dimensions.
    flat_out = out.reshape(-1, n_flavours, n_flavours)
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
        identity3=identity,
        profile_scale_m=profile_scale_m,
        evolution_scale_m=evolution_scale_m,
        n_flavours=n_flavours,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
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
        n_flavours=n_flavours,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
    )

    if reunitarize:
        U_b = project_to_unitary(U_b)

    # Insert the case-B subset. Entries in neither mask are still identities.
    flat_out[flat_idx_b] = U_b

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
    include_matter_nc: Optional[bool] = None,
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
        include_matter_nc: If True, also apply the 3+1 sterile extension's
            neutral-current matter term (see ``earth_evolutor``).

    Returns:
        Complex tensor with shape `(*broadcast_shape(E_MeV, theta_deg), N, N)`,
        N in {3, 4}, containing full flavour-basis Earth evolution operators,
        identical in meaning to the output of ``earth_evolutor``.
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
        include_matter_nc=include_matter_nc,
    )
