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
Atmospheric propagation utilities for atmospheric neutrinos.

This module implements the atmospheric part of the propagation:

    production height h  ->  earth surface

The propagation can be computed either as vacuum propagation or including
a atmospheric matter density profile.

Module functions:
    
    kinetic_terms(...)
        Builds the mass-basis kinetic eigenvalues used in the atmospheric
        Hamiltonian from mass splittings and neutrino energy.
    
    atmospheric_hamiltonian(...)
        Constructs the full flavour-basis Hamiltonian H = H_kin + H_mat for
        atmospheric matter density and optional antineutrino propagation.
    
    atmospheric_evolution_operator(...)
        Segments the atmospheric trajectory, evaluates the density profile,
        exponentiates local Hamiltonians, and composes the evolution operator.
    
    propagate_atmosphere(...)
        Applies the atmospheric evolution operator to a coherent flavour
        state and returns the surface state together with the operator.
"""




from __future__ import annotations

from typing import  Union, Callable
import torch

from tpeanuts.util.type import _as_tensor, _cdtype_from_real
from tpeanuts.util.torch_util import _default_device
from tpeanuts.util.constant import R_E_KM
from tpeanuts.core.potential import matter_potential
from tpeanuts.core.hamiltonian import (
    _infer_device_dtype,
    kinetic_mass_vector,
    reduced_mixing_matrix,
    kinetic_hamiltonian_reduced,
    matter_hamiltonian_reduced,
)
from tpeanuts.core.dressing import earth_dressing_matrices, dress_reduced_evolutor
from tpeanuts.core.segment_evolution import compose_segment_evolutors

from tpeanuts.atmosphere.geometry import (
    atmospheric_path_length,
    altitude_along_detector_path,
    underground_path_length,
)

from tpeanuts.atmosphere.density import atmospheric_electron_density_profile

TensorLike = Union[float, int, torch.Tensor]


# ============================================================
# Atmospheric Hamiltonian
# ============================================================

@torch.no_grad()
def kinetic_terms(
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    *,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """
    Build mass-basis kinetic terms for atmospheric propagation.

    Args:
        DeltamSq21: Solar mass splitting Delta m^2_21 in eV^2. Scalar or tensor.
        DeltamSq3l: Atmospheric mass splitting Delta m^2_3l in eV^2.
        E_MeV: Neutrino energy in MeV. Scalar or tensor broadcastable with the
            mass splittings.
        device: Optional torch device.
        dtype: Real dtype for the kinetic terms.

    Returns:
        Tensor of kinetic eigenvalues k_i with final dimension 3, in the
        conventions used by tpeanuts.core.hamiltonian.
    """
    dev, dtype = _infer_device_dtype(
        E_MeV,
        DeltamSq21,
        DeltamSq3l,
        device=device,
        dtype=dtype,
    )

    return kinetic_mass_vector(
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        device=dev,
        dtype=dtype,
    )


@torch.no_grad()
def atmospheric_hamiltonian(
    pmns,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    ne_molcm3: TensorLike,
    antinu: Union[bool, torch.Tensor] = False,
    *,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """
    Construct the full flavour-basis atmospheric Hamiltonian.

    Args:
        pmns: PMNS object or compatible mixing container accepted by core
            mixing helpers.
        DeltamSq21: Delta m^2_21 in eV^2. Scalar or tensor.
        DeltamSq3l: Delta m^2_3l in eV^2. Scalar or tensor.
        E_MeV: Neutrino energy in MeV, broadcastable to density inputs.
        ne_molcm3: Electron density in mol/cm^3. Scalar or tensor, commonly
            shaped (..., n_steps).
        antinu: Bool or tensor mask selecting antineutrino matter sign and
            conjugated mixing matrix.
        device: Optional torch device.
        dtype: Real dtype for inputs; complex dtype is inferred internally.

    Returns:
        Complex tensor with shape (..., 3, 3) containing H = H_kin + H_mat in
        the dressed flavour basis.
    """
    dev, dtype = _infer_device_dtype(
        E_MeV,
        ne_molcm3,
        DeltamSq21,
        DeltamSq3l,
        device=device,
        dtype=dtype,
    )
    cdtype = _cdtype_from_real(dtype)

    ki = kinetic_terms(
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        device=dev,
        dtype=dtype,
    )

    U = reduced_mixing_matrix(
        pmns,
        antinu=antinu,
        device=dev,
        dtype=cdtype,
    )

    Hk = kinetic_hamiltonian_reduced(
        ki=ki,
        Ured=U,
    )
    
    V = matter_potential(
        _as_tensor(ne_molcm3, device=dev, dtype=dtype),
        antinu=antinu,
    )

    Hmatter = matter_hamiltonian_reduced(V)

    H_reduced = Hk + Hmatter

    r23, delta = earth_dressing_matrices(
        pmns,
        antinu=antinu,
        device=dev,
        dtype=cdtype,
    )

    return dress_reduced_evolutor(H_reduced, r23, delta)


# ============================================================
# Atmospheric evolution
# ============================================================

@torch.no_grad()
def atmospheric_evolution_operator(
    pmns,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    antinu: Union[bool, torch.Tensor] = False,
    ne_profile: Callable = atmospheric_electron_density_profile,
    n_steps: int = 600,
    matter: bool = True,
    *,
    device=None,
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the atmospheric evolution operator over a segmented trajectory.

    Args:
        pmns: PMNS object or compatible mixing container.
        DeltamSq21: Delta m^2_21 in eV^2.
        DeltamSq3l: Delta m^2_3l in eV^2.
        E_MeV: Neutrino energy in MeV. Scalar or tensor.
        h_km: Production altitude in km. Scalar or tensor broadcastable with
            E_MeV and theta_deg.
        theta_deg: Atmospheric zenith angle in degrees.
        depth_km: Detector depth below surface in km.
        antinu: Bool or tensor mask for antineutrino propagation.
        ne_profile: Callable returning electron density in mol/cm^3 for an
            altitude tensor; defaults to atmospheric_electron_density_profile.
        n_steps: Number of path grid points used for segmented propagation.
        matter: If False, replaces the matter profile by zero density.
        device: Optional torch device.
        dtype: Real dtype for trajectory and Hamiltonian inputs.

    Returns:
        Pair (S, x_grid), where S has shape (..., 3, 3) and is the complex
        atmospheric evolution operator, and x_grid is the dimensionless path
        grid L/R_E with final dimension n_steps.
    """
    dev, dtype = _infer_device_dtype(
        E_MeV,
        h_km,
        theta_deg,
        depth_km,
        device=device,
        dtype=dtype,
    )
    cdtype = _cdtype_from_real(dtype)

    h_km = _as_tensor(h_km, device=dev, dtype=dtype)
    theta_deg = _as_tensor(theta_deg, device=dev, dtype=dtype)
    depth_km = _as_tensor(depth_km, device=dev, dtype=dtype)

    L_atm_km = atmospheric_path_length(
        h_km=h_km,
        theta_deg=theta_deg,
        depth_km=depth_km,
        device=dev,
        dtype=dtype,
        check_geometry=False,
    )

    L_und_km = underground_path_length(
        theta_deg=theta_deg,
        depth_km=depth_km,
        device=dev,
        dtype=dtype,
        check_geometry=False,
    )

    u_grid = torch.linspace(
        0.0,
        1.0,
        int(n_steps),
        device=dev,
        dtype=dtype,
    )

    x_grid = (L_atm_km / R_E_KM)[..., None] * u_grid
    x_mid = 0.5 * (x_grid[..., :-1] + x_grid[..., 1:])
    dx = x_grid[..., 1:] - x_grid[..., :-1]

    s_atm_km = x_mid * R_E_KM
    s_detector_km = L_und_km[..., None] + s_atm_km

    altitude_km = altitude_along_detector_path(
        s_km=s_detector_km,
        theta_deg=theta_deg[..., None],
        depth_km=depth_km[..., None],
        device=dev,
        dtype=dtype,
    )

    if matter:
        ne_molcm3 = ne_profile(
            altitude_km,
            device=dev,
            dtype=dtype,
        )
    else:
        ne_molcm3 = torch.zeros_like(altitude_km)

    E_steps = _as_tensor(E_MeV, device=dev, dtype=dtype)[..., None]
    antinu_steps = antinu.unsqueeze(-1) if torch.is_tensor(antinu) else antinu

    H = atmospheric_hamiltonian(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_steps,
        ne_molcm3=ne_molcm3,
        antinu=antinu_steps,
        device=dev,
        dtype=dtype,
    )

    S_steps = torch.linalg.matrix_exp(
        -1j * H * dx[..., None, None].to(dtype=cdtype)
    )

    S = compose_segment_evolutors(
        S_steps,
        segment_dim=-3,
        multiply="left",
    )

    identity = torch.eye(3, device=dev, dtype=cdtype)
    S = torch.where(
        (L_atm_km <= 0.0)[..., None, None],
        identity.expand(*S.shape[:-2], 3, 3),
        S,
    )

    return S, x_grid


@torch.no_grad()
def propagate_atmosphere(
    nustate: TensorLike,
    pmns,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    antinu: Union[bool, torch.Tensor] = False,
    ne_profile: Callable = atmospheric_electron_density_profile,
    n_steps: int = 600,
    matter: bool = True,
    *,
    device=None,
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate a coherent flavour state through the atmospheric segment.

    Args:
        nustate: Initial flavour-amplitude tensor with last dimension 3.
        pmns: PMNS object or compatible mixing container.
        DeltamSq21: Delta m^2_21 in eV^2.
        DeltamSq3l: Delta m^2_3l in eV^2.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude in km.
        theta_deg: Atmospheric zenith angle in degrees.
        depth_km: Detector depth in km.
        antinu: Bool or tensor mask for antineutrino propagation.
        ne_profile: Callable mapping altitude km to electron density mol/cm^3.
        n_steps: Number of trajectory grid points.
        matter: If False, perform vacuum atmospheric propagation.
        device: Optional torch device.
        dtype: Real dtype for geometry and Hamiltonian quantities.

    Returns:
        Pair (nu_surface, S_atm). nu_surface has last dimension 3 and contains
        propagated amplitudes at the Earth surface; S_atm is the complex
        atmospheric evolution operator with shape (..., 3, 3).
    """
    dev = _default_device(device)
    cdtype = _cdtype_from_real(dtype)

    nustate = _as_tensor(
        nustate,
        device=dev,
        dtype=cdtype,
    )

    if nustate.shape[-1] != 3:
        raise ValueError("nustate must have last dimension 3.")

    S_atm, _ = atmospheric_evolution_operator(
        pmns=pmns,
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        depth_km=depth_km,
        antinu=antinu,
        ne_profile=ne_profile,
        n_steps=n_steps,
        matter=matter,
        device=dev,
        dtype=dtype,
    )

    while nustate.ndim < S_atm.ndim - 1:
        nustate = nustate.unsqueeze(-2)

    nu_surface = torch.matmul(
        S_atm,
        nustate[..., None],
    ).squeeze(-1)

    return nu_surface, S_atm
