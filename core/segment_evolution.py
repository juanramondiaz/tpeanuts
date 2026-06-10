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
Segment evolution utilities for peanuts-torch.

This module provides the high-level interface that builds the evolution
operator for a single matter segment.

A peanuts segment is defined by:

    - Initial coordinate x1.
    - Final coordinate x2.
    - Polynomial density coefficients a, b, c.
    - Neutrino energy E.
    - Oscillation parameters.
    - Reduced PMNS mixing matrix.

The density profile inside the segment is assumed to be

    n_e(x) = a + b x^2 + c x^4.

The segment evolution operator is computed as

    U_segment = u0 + u1,

where:

    u0
        Constant-density evolution operator computed using the average
        density over the segment.

    u1
        First-order perturbative correction due to the density variation
        inside the segment.

This module connects the lower-level core components:

    hamiltonian.py
        Builds the reduced Hamiltonian H.

    spectral.py
        Decomposes H into eigenvalues and projectors.

    evolution.py
        Computes the constant-density evolutor u0.

    perturbation.py
        Computes the first-order correction u1.

The main public function is:

    perturbative_segment_evolutor(...)

It replaces the old Upert_torch(...) routine.

Module functions:
    perturbative_segment_evolutor(...), constant_density_segment_evolutor(...)
        Build segment-level evolution operators with perturbative or
        constant-density treatment.
    compose_segment_evolutors(...)
        Multiply ordered segment evolution matrices along a chosen segment
        dimension.
    apply_segment_sequence_to_state(...)
        Apply a composed segment sequence to an input flavour or
        reduced-basis state.
"""



from __future__ import annotations

from typing import Union
import torch

from tpeanuts.util.type import _as_tensor, _cdtype_from_real

from tpeanuts.core.hamiltonian import (
    _infer_device_dtype,
    kinetic_mass_vector,
    matter_potential_from_polynomial_average,
    reduced_mixing_matrix,
    kinetic_hamiltonian_reduced,
    matter_hamiltonian_reduced,
)

from tpeanuts.core.perturbation import (
    perturbative_segment_evolutor as _perturbative_segment_evolutor_core,
)

TensorLike = Union[float, int, torch.Tensor]


@torch.no_grad()
def perturbative_segment_evolutor(
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    pmns: object,
    E_MeV: TensorLike,
    x2: TensorLike,
    x1: TensorLike,
    a: TensorLike,
    b: TensorLike,
    c: TensorLike,
    antinu: Union[bool, torch.Tensor] = False,
    debug: bool = False,
) -> torch.Tensor:
    """
    Build a reduced Hamiltonian for one polynomial-density segment and evolve it with first-order perturbation theory.
    
    Args:
        DeltamSq21: Solar mass splitting Delta m^2_21 in eV^2.
        DeltamSq3l: Atmospheric mass splitting Delta m^2_3l in eV^2; sign selects normal or inverted ordering.
        pmns: PMNS object exposing full and reduced mixing matrices plus R23 and Delta builders.
        E_MeV: Neutrino energy in MeV; scalar or tensor broadcastable with other inputs.
        x2: Final segment coordinate in Earth-radius units or polynomial coordinate.
        x1: Initial segment coordinate in Earth-radius units or polynomial coordinate.
        a: Constant coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        b: Quadratic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        c: Quartic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
        debug: If True, return diagnostic intermediate tensors together with the evolutor.
    
    Returns:
        Complex segment evolutor, or (evolutor, diagnostics) when debug=True.
    """
    device, rdtype = _infer_device_dtype(E_MeV, x1, x2, a, b, c)

    cdtype = _cdtype_from_real(rdtype)

    E_MeV = _as_tensor(E_MeV, device=device, dtype=rdtype)
    x1 = _as_tensor(x1, device=device, dtype=rdtype)
    x2 = _as_tensor(x2, device=device, dtype=rdtype)
    a = _as_tensor(a, device=device, dtype=rdtype)
    b = _as_tensor(b, device=device, dtype=rdtype)
    c = _as_tensor(c, device=device, dtype=rdtype)

    ki = kinetic_mass_vector(
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        device=device,
        dtype=rdtype,
    )

    V, naverage, L, zero_mask = matter_potential_from_polynomial_average(
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
        antinu=antinu,
        device=device,
        dtype=rdtype,
    )

    Ured = reduced_mixing_matrix(
        pmns,
        antinu=antinu,
        device=device,
        dtype=cdtype,
    )

    Hkin = kinetic_hamiltonian_reduced(
        ki=ki,
        Ured=Ured,
    )

    Hmat = matter_hamiltonian_reduced(
        V=V,
    )

    H = Hkin + Hmat

    trace_H = (
        ki[..., 0]
        + ki[..., 1]
        + ki[..., 2]
        + V
    ).to(dtype=cdtype)

    if debug:
        herm_err = (
            H
            - H.conj().transpose(-1, -2)
        ).abs().max()

        print(
            f"[DEBUG] H Hermitian error: max|H-H†| = {herm_err.item():.4e}"
        )

        if not torch.isfinite(naverage).all():
            print("[DEBUG] naverage contains NaN or Inf.")

        if not torch.isfinite(V).all():
            print("[DEBUG] V contains NaN or Inf.")

        if not torch.isfinite(H).all():
            print("[DEBUG] H contains NaN or Inf.")

        if not torch.isfinite(L).all():
            print("[DEBUG] L contains NaN or Inf.")

    U_segment = _perturbative_segment_evolutor_core(
        H=H,
        L=L,
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
        naverage=naverage,
        trace_H=trace_H,
        zero_mask=zero_mask,
        antinu=antinu,
    )

    return U_segment


@torch.no_grad()
def constant_density_segment_evolutor(
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    pmns: object,
    E_MeV: TensorLike,
    x2: TensorLike,
    x1: TensorLike,
    a: TensorLike,
    b: TensorLike,
    c: TensorLike,
    antinu: Union[bool, torch.Tensor] = False,
    debug: bool = False,
) -> torch.Tensor:
    """
    Compute the exact evolutor for a segment approximated by its average constant density.
    
    Args:
        DeltamSq21: Solar mass splitting Delta m^2_21 in eV^2.
        DeltamSq3l: Atmospheric mass splitting Delta m^2_3l in eV^2; sign selects normal or inverted ordering.
        pmns: PMNS object exposing full and reduced mixing matrices plus R23 and Delta builders.
        E_MeV: Neutrino energy in MeV; scalar or tensor broadcastable with other inputs.
        x2: Final segment coordinate in Earth-radius units or polynomial coordinate.
        x1: Initial segment coordinate in Earth-radius units or polynomial coordinate.
        a: Constant coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        b: Quadratic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        c: Quartic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
        debug: If True, return diagnostic intermediate tensors together with the evolutor.
    
    Returns:
        Complex segment evolutor, or (evolutor, diagnostics) when debug=True.
    """
    from tpeanuts.core.evolution import constant_density_evolutor

    device, rdtype = _infer_device_dtype(E_MeV, x1, x2, a, b, c)

    cdtype = _cdtype_from_real(rdtype)

    E_MeV = _as_tensor(E_MeV, device=device, dtype=rdtype)

    ki = kinetic_mass_vector(
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        device=device,
        dtype=rdtype,
    )

    V, naverage, L, zero_mask = matter_potential_from_polynomial_average(
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
        antinu=antinu,
        device=device,
        dtype=rdtype,
    )

    Ured = reduced_mixing_matrix(
        pmns,
        antinu=antinu,
        device=device,
        dtype=cdtype,
    )

    H = (
        kinetic_hamiltonian_reduced(ki, Ured)
        + matter_hamiltonian_reduced(V)
    )

    trace_H = (
        ki[..., 0]
        + ki[..., 1]
        + ki[..., 2]
        + V
    ).to(dtype=cdtype)

    if debug:
        herm_err = (
            H
            - H.conj().transpose(-1, -2)
        ).abs().max()

        print(
            f"[DEBUG] H Hermitian error: max|H-H†| = {herm_err.item():.4e}"
        )

    U0 = constant_density_evolutor(
        H=H,
        L=L,
        trace_H=trace_H,
        zero_mask=zero_mask,
    )

    return U0


@torch.no_grad()
def compose_segment_evolutors(
    U_segments: torch.Tensor,
    *,
    segment_dim: int = -3,
    multiply: str = "left",
) -> torch.Tensor:
    """
    Compose ordered segment evolutors into a single total evolution operator.
    
    Args:
        U_segments: Ordered segment-evolutor tensor shaped (..., N, 3, 3) or iterable with segment matrices in propagation order.
        segment_dim: Axis of U_segments that enumerates propagation segments.
        multiply: Composition convention; "left" applies each new segment as U_seg @ U_total, "right" as U_total @ U_seg.
    
    Returns:
        Total evolutor tensor shaped (..., 3, 3).
    """
    if U_segments.shape[-2:] != (3, 3):
        raise ValueError("U_segments must have final shape (..., 3, 3).")

    if multiply not in ("left", "right"):
        raise ValueError("multiply must be either 'left' or 'right'.")

    U_segments = torch.movedim(U_segments, segment_dim, -3)
    batch_shape = U_segments.shape[:-3]

    U_total = torch.eye(
        3,
        device=U_segments.device,
        dtype=U_segments.dtype,
    ).expand(*batch_shape, 3, 3)

    for U_seg in U_segments.unbind(dim=-3):
        if multiply == "left":
            U_total = U_seg @ U_total
        else:
            U_total = U_total @ U_seg

    return U_total


@torch.no_grad()
def apply_segment_sequence_to_state(
    U_segments: torch.Tensor,
    state_initial: torch.Tensor,
    *,
    segment_dim: int = -3,
) -> torch.Tensor:
    """
    Apply an ordered sequence of segment evolutors to an initial flavour state vector.
    
    Args:
        U_segments: Ordered segment-evolutor tensor shaped (..., N, 3, 3) or iterable with segment matrices in propagation order.
        state_initial: Initial flavour state vector shaped (..., 3).
        segment_dim: Axis of U_segments that enumerates propagation segments.
    
    Returns:
        Final state vector shaped (..., 3).
    """
    U_total = compose_segment_evolutors(
        U_segments,
        segment_dim=segment_dim,
    )

    if state_initial.shape[-1] != 3:
        raise ValueError("state_initial must have last dimension equal to 3.")

    state_initial = state_initial.to(
        device=U_total.device,
        dtype=U_total.dtype,
    )

    return torch.einsum("...ij,...j->...i", U_total, state_initial)
