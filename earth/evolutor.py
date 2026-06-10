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

It sits above the core peanuts modules. The core module computes one local
matter-segment evolutor. This earth module:

    1. Classifies the trajectory angle eta.
    2. Computes earth geometry.
    3. Queries the earth density model for crossed shells.
    4. Builds and multiplies segment evolutors.
    5. Dresses the reduced evolution operator into the full flavour basis.
    6. Optionally projects the result onto the nearest unitary matrix.

The main public function is earth_evolutor(...).
"""



from __future__ import annotations

from typing import Union

import torch

import tpeanuts.util.default as default
from tpeanuts.util.type import _cdtype_from_real
from tpeanuts.util.math import project_to_unitary

from tpeanuts.core.segment_evolution import (
    compose_segment_evolutors,
    perturbative_segment_evolutor,
)
from tpeanuts.core.dressing import (
    earth_dressing_matrices,
    dress_reduced_evolutor,
)
from tpeanuts.core.hamiltonian import _infer_device_dtype

from tpeanuts.earth.geometry import (
    detector_radius_fraction,
    eta_prime_from_eta,
    detector_x_coordinate,
    chord_length_case_b,
    classify_eta_regions,
    validate_eta_range,
)

from tpeanuts.earth.layers import (
    normalize_density_layer_output,
    extract_quartic_coefficients,
    outermost_crossed_shell,
    build_flipped_shell_segments,
)


TensorLike = Union[float, int, torch.Tensor]


def _broadcast_energy_and_eta(
    E: TensorLike,
    eta: TensorLike,
) -> tuple[torch.Tensor, torch.Tensor, torch.device, torch.dtype, torch.dtype]:
    """
    Convert energy and nadir angle inputs to broadcast-compatible tensors.

    Args:
        E: Neutrino energy in MeV. Can be scalar-like or a torch tensor.
        eta: Detector nadir angle in radians. Can be scalar-like or a torch
            tensor.

    Returns:
        Tuple containing broadcast energy tensor, broadcast eta tensor, common
        device, real dtype, and matching complex dtype.

    Notes:
        If both E and eta are one-dimensional and have different lengths, they
        are interpreted as independent grids and expanded to an outer product
        shape `(n_energy, n_eta)`.
    """
    device, rdtype = _infer_device_dtype(E, eta)
    E_t = torch.as_tensor(E, device=device, dtype=rdtype)

    cdtype = _cdtype_from_real(rdtype)

    if torch.is_tensor(eta):
        eta_t = eta.to(device=device, dtype=rdtype)
    else:
        eta_t = torch.tensor(eta, device=device, dtype=rdtype)

    if E_t.ndim == 1 and eta_t.ndim == 1 and E_t.numel() != eta_t.numel():
        E_t = E_t[:, None]
        eta_t = eta_t[None, :]

    E_b, eta_b = torch.broadcast_tensors(E_t, eta_t)

    return E_b, eta_b, device, rdtype, cdtype


def _broadcast_antinu(
    antinu: Union[bool, torch.Tensor],
    target: torch.Tensor,
) -> Union[bool, torch.Tensor]:
    """
    Broadcast the neutrino/antineutrino selector to a target grid.

    Args:
        antinu: Boolean scalar or tensor. True selects antineutrino matter
            effects.
        target: Tensor whose broadcast shape is used when antinu is a tensor.

    Returns:
        A boolean scalar when antinu is scalar, otherwise a boolean tensor with
        the same broadcast shape as target.
    """
    if isinstance(antinu, bool):
        return antinu

    antinu_t = antinu.to(device=target.device, dtype=torch.bool)
    antinu_b, _ = torch.broadcast_tensors(antinu_t, target)

    return antinu_b


def _flat_antinu(
    antinu: Union[bool, torch.Tensor],
    flat_idx: torch.Tensor,
) -> Union[bool, torch.Tensor]:
    """
    Select antinu flags for a flattened subset of the broadcast grid.

    Args:
        antinu: Scalar boolean or broadcast antinu tensor.
        flat_idx: Flat indices of the entries being evaluated.

    Returns:
        Scalar antinu unchanged, or a one-dimensional tensor matching flat_idx.
    """
    if isinstance(antinu, bool):
        return antinu

    return antinu.reshape(-1)[flat_idx]


def _masked_identity_output(
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Allocate an uninitialized output tensor with trailing evolution dimensions.

    Args:
        shape: Broadcast batch shape.
        device: Target torch device.
        dtype: Complex dtype for the evolution operators.

    Returns:
        Tensor with shape `(*shape, 3, 3)`.

    Notes:
        This helper currently only centralizes the target shape convention. The
        public evolutor initializes its output explicitly with identity
        matrices because above-horizon trajectories should remain identities.
    """
    return torch.empty((*shape, 3, 3), device=device, dtype=dtype)


@torch.no_grad()
def _earth_evolutor_case_a_batched(
    density: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    pmns: object,
    E_b: torch.Tensor,
    eta_b: torch.Tensor,
    depth_m: float,
    antinu: Union[bool, torch.Tensor],
    device: torch.device,
    rdtype: torch.dtype,
    cdtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute Earth evolution operators for trajectories crossing density shells.

    Args:
        density: Earth density model exposing `parameters_abc(eta_prime)`.
        DeltamSq21: Solar mass-squared splitting in eV^2.
        DeltamSq3l: Atmospheric mass-squared splitting in eV^2.
        pmns: PMNS mixing object used by the segment evolutor.
        E_b: Flattened batch of neutrino energies in MeV for case A entries.
        eta_b: Flattened batch of nadir angles in radians for case A entries.
        depth_m: Detector depth in meters.
        antinu: Scalar or batched antineutrino selector for these entries.
        device: Torch device used for intermediate tensors.
        rdtype: Real floating dtype.
        cdtype: Complex dtype matching rdtype.

    Returns:
        Tensor with shape `(batch_size, 3, 3)` containing full flavour-basis
        evolution operators.

    Notes:
        Case A covers paths that enter Earth matter deeply enough to cross one
        or more tabulated density shells. The implementation builds the shell
        segments for half of the chord, composes them, adds the detector-side
        segment, and dresses the reduced operator into the full flavour basis.
    """
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
    # used by the density-layer geometry.
    eta_prime = eta_prime_from_eta(
        eta_b,
        r_d,
    )

    # Coordinate of the detector point along the trajectory chord.
    x_d = detector_x_coordinate(
        eta_b,
        r_d,
    )

    # Query the density model for the polynomial coefficients and shell
    # intersections crossed by each trajectory.
    coeffs_all, xj_all, crossed = density.parameters_abc(eta_prime)

    # Normalize density output shapes so downstream code can treat single and
    # batched trajectories uniformly.
    coeffs_all, xj_all, crossed = normalize_density_layer_output(
        coeffs_all,
        xj_all,
        crossed,
    )

    # Split the quadratic-density coefficients n_e(x) = a*x^2 + b*x + c.
    a, b, c = extract_quartic_coefficients(coeffs_all)

    batch_size, ns = xj_all.shape

    # Find the outermost crossed shell. This gives the first material segment
    # between the Earth surface and the detector-side half path.
    outer = outermost_crossed_shell(
        a=a,
        b=b,
        c=c,
        xj_all=xj_all,
        crossed=crossed,
        dtype=rdtype,
        device=device,
    )

    # Build ordered shell segments for the mirrored half-chord. The segment
    # fields include endpoints, coefficients, and a boolean mask indicating
    # whether the segment is physically crossed for each trajectory.
    segments = build_flipped_shell_segments(
        xj_all=xj_all,
        a=a,
        b=b,
        c=c,
        crossed=crossed,
    )

    # Add a segment dimension to energy so perturbative_segment_evolutor
    # broadcasts over both batch and crossed-shell segments.
    E_seg = E_b.unsqueeze(-1)

    antinu_segments = antinu.unsqueeze(-1) if torch.is_tensor(antinu) else antinu

    # Compute one reduced matter evolutor per segment. Non-crossed segments are
    # replaced by identities below so they do not affect the composition.
    U_segments = perturbative_segment_evolutor(
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        pmns=pmns,
        E_MeV=E_seg,
        x2=segments["x_hi"],
        x1=segments["x_lo"],
        a=segments["a2"],
        b=segments["b2"],
        c=segments["c2"],
        antinu=antinu_segments,
    )

    I = torch.eye(3, device=device, dtype=cdtype).view(1, 1, 3, 3)

    # Mask out geometrically absent shell segments. This keeps a rectangular
    # tensor shape while preserving the correct product for each trajectory.
    U_segments = torch.where(
        segments["crossed2"].view(batch_size, ns, 1, 1),
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
    U0_det = perturbative_segment_evolutor(
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        pmns=pmns,
        E_MeV=E_b,
        x2=x_d,
        x1=outer["x_start"],
        a=outer["a_o"],
        b=outer["b_o"],
        c=outer["c_o"],
        antinu=antinu,
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

    # The segment evolutor works in the reduced basis. Build the external
    # rotation/phase matrices needed to recover the full flavour-basis operator.
    r23, delta = earth_dressing_matrices(
        pmns,
        antinu=antinu,
        device=device,
        dtype=cdtype,
    )

    # Dress the reduced operator into full flavour basis.
    U = dress_reduced_evolutor(
        U_red,
        r23,
        delta,
    )

    I_full = torch.eye(3, device=device, dtype=cdtype).view(1, 3, 3)

    # Some case-A entries can still have no material shell after geometric
    # filtering. Keep those as identity operators.
    U = torch.where(
        outer["has_any"].view(batch_size, 1, 1),
        U,
        I_full,
    )

    return U


@torch.no_grad()
def _earth_evolutor_case_b_batched(
    density: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    pmns: object,
    E_b: torch.Tensor,
    eta_b: torch.Tensor,
    depth_m: float,
    antinu: Union[bool, torch.Tensor],
    device: torch.device,
    rdtype: torch.dtype,
    cdtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute Earth evolution operators for shallow detector-near trajectories.

    Args:
        density: Earth density model exposing `call(x, eta)`.
        DeltamSq21: Solar mass-squared splitting in eV^2.
        DeltamSq3l: Atmospheric mass-squared splitting in eV^2.
        pmns: PMNS mixing object used by the segment evolutor.
        E_b: Flattened batch of neutrino energies in MeV for case B entries.
        eta_b: Flattened batch of nadir angles in radians for case B entries.
        depth_m: Detector depth in meters.
        antinu: Scalar or batched antineutrino selector for these entries.
        device: Torch device used for intermediate tensors.
        rdtype: Real floating dtype.
        cdtype: Complex dtype matching rdtype.

    Returns:
        Tensor with shape `(batch_size, 3, 3)` containing full flavour-basis
        evolution operators.

    Notes:
        Case B represents short paths between the Earth surface and the
        detector. The matter density is approximated at the mid-depth radius
        and applied as a single constant-density segment.
    """
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

    # Approximate the density along the short path by evaluating it at the
    # midpoint radius between the surface and the detector.
    h = float(depth_m) / float(__import__("tpeanuts.util.constant").util.constant.R_E)
    r_mid = 1.0 - h / 2.0

    n1 = density.call(
        torch.tensor(r_mid, device=device, dtype=rdtype),
        torch.tensor(0.0, device=device, dtype=rdtype),
    )

    # Physical path length inside Earth matter for each shallow trajectory.
    deltax = chord_length_case_b(
        eta_b,
        r_d,
    )

    zeros = torch.zeros_like(deltax)

    # Treat the shallow path as a single constant-density matter segment in the
    # reduced basis.
    U_red = perturbative_segment_evolutor(
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        pmns=pmns,
        E_MeV=E_b,
        x2=deltax,
        x1=zeros,
        a=n1,
        b=zeros,
        c=zeros,
        antinu=antinu,
    )

    # Build the matrices that map the reduced operator back to full flavour
    # basis, including the antineutrino convention when requested.
    r23, delta = earth_dressing_matrices(
        pmns,
        antinu=antinu,
        device=device,
        dtype=cdtype,
    )

    # Return the full flavour-basis operator for the shallow segment.
    U = dress_reduced_evolutor(
        U_red,
        r23,
        delta,
    )

    return U


@torch.no_grad()
def earth_evolutor(
    density: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    pmns: object,
    E: TensorLike,
    eta: TensorLike,
    depth_m: float,
    antinu: Union[bool, torch.Tensor] = default.earth_antinu,
    *,
    reunitarize: bool = default.earth_reunitarize,
) -> torch.Tensor:
    """
    Compute the full Earth matter evolution operator in flavour basis.

    Args:
        density: Earth density model. It must provide the shell polynomial
            interface used by case A and the direct density call used by case B.
        DeltamSq21: Solar mass-squared splitting in eV^2.
        DeltamSq3l: Atmospheric mass-squared splitting in eV^2.
        pmns: PMNS mixing object.
        E: Neutrino energy in MeV. Scalars, tensors, and broadcastable grids are
            accepted.
        eta: Detector nadir angle in radians. Scalars, tensors, and
            broadcastable grids are accepted.
        depth_m: Detector depth below the Earth surface in meters.
        antinu: Boolean scalar or tensor selecting antineutrino propagation.
        reunitarize: If True, project computed evolution matrices onto the
            nearest unitary matrix.

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
    # Build a common E/eta grid. For independent 1D grids this creates the
    # outer-product shape, so the final result can preserve `(n_E, n_eta, 3, 3)`.
    E_b, eta_b, device, rdtype, cdtype = _broadcast_energy_and_eta(
        E,
        eta,
    )
    # The antineutrino flag may be a scalar or a tensor defined on the same
    # broadcast grid as E and eta.
    antinu_b = _broadcast_antinu(
        antinu,
        E_b,
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
        density=density,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        pmns=pmns,
        E_b=E_flat[flat_idx_a],
        eta_b=eta_flat[flat_idx_a],
        depth_m=depth_m,
        antinu=_flat_antinu(antinu_b, flat_idx_a),
        device=device,
        rdtype=rdtype,
        cdtype=cdtype,
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
        density=density,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        pmns=pmns,
        E_b=E_flat[flat_idx_b],
        eta_b=eta_flat[flat_idx_b],
        depth_m=depth_m,
        antinu=_flat_antinu(antinu_b, flat_idx_b),
        device=device,
        rdtype=rdtype,
        cdtype=cdtype,
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
