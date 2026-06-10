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
earth probability utilities for peanuts-torch.

This module converts the full earth evolution operator into physical flavour
probabilities.

It sits above:

    earth.evolutor
        Computes the full earth evolution operator U_earth(E, eta).

    core.probabilities
        Provides generic probability and flux utilities.

The main public function is:

    pearth(..., method="analytical" | "numerical")

which dispatches earth matter-regeneration probabilities either from:

    1. Incoherent mass-basis weights.
    2. A coherent flavour-basis input state.

Conventions
-----------
For mass-basis input, the initial state is represented by weights

    w_i

and the final flavour probability is

    P_alpha =
        sum_i |(U_earth U_PMNS)_{alpha i}|^2 w_i.

For antineutrinos, the PMNS matrix is complex-conjugated.

For flavour-basis input, the initial state is treated as a coherent flavour
amplitude vector,

    psi_final = U_earth psi_initial,

and

    P_alpha = |psi_final_alpha|^2.

This module does not build density profiles, shell crossings, Hamiltonians,
or segment evolutors. It only computes probabilities from the earth evolutor.
"""



from __future__ import annotations

from typing import Literal, Optional, Union
import torch
from torch import Tensor

import tpeanuts.util.default as default

TensorLike = Union[float, int, Tensor]
PearthMethod = Literal["analytical", "numerical"]

from tpeanuts.earth.evolutor import earth_evolutor

from tpeanuts.util.type import _state_tensor, _cdtype_from_real, _as_complex_tensor
from tpeanuts.core.hamiltonian import _select_antinu_matrix
from tpeanuts.earth.numerical import numerical_solution
Tensor = torch.Tensor
TensorLike = Union[float, int, torch.Tensor]


def _broadcast_mass_weights(
    weights: Tensor,
    probs_i_to_alpha: Tensor,
) -> Tensor:
    target_ndim = probs_i_to_alpha.ndim - 1

    while weights.ndim < target_ndim:
        weights = weights.unsqueeze(-2)

    return weights


@torch.no_grad()
def pearth_analytical(
    nustate: Tensor,
    density: object,
    pmns: object,
    dm21_eV2: TensorLike,
    dm3l_eV2: TensorLike,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    *,
    antinu: Union[bool, Tensor] = default.earth_antinu,
    massbasis: bool = default.earth_massbasis,
    reunitarize: bool = default.earth_reunitarize,
) -> Tensor:
    U_earth = earth_evolutor(
        density=density,
        DeltamSq21=dm21_eV2,
        DeltamSq3l=dm3l_eV2,
        pmns=pmns,
        E=E_MeV,
        eta=eta,
        depth_m=depth_m,
        antinu=antinu,
        reunitarize=reunitarize,
    )

    if massbasis:

        U_pmns = pmns.pmns_matrix().to(
            device=U_earth.device,
            dtype=U_earth.dtype,
        )

        U_pmns = _select_antinu_matrix(
            U_pmns,
            antinu,
        )

        U_total = U_earth @ U_pmns

        probs_i_to_alpha = torch.abs(U_total) ** 2

        weights = _state_tensor(
            nustate,
            device=U_earth.device,
            dtype=probs_i_to_alpha.real.dtype,
        )

        weights = _broadcast_mass_weights(
            weights,
            probs_i_to_alpha,
        )

        probabilities = torch.sum(
            probs_i_to_alpha * weights[..., None, :],
            dim=-1,
        )

        return probabilities.real

    psi0 = _state_tensor(
        nustate,
        device=U_earth.device,
        dtype=U_earth.dtype,
    )

    while psi0.ndim < U_earth.ndim - 1:
        psi0 = psi0.unsqueeze(-2)

    psi_final = torch.einsum(
        "...ab,...b->...a",
        U_earth,
        psi0,
    )

    return (torch.abs(psi_final) ** 2).real


@torch.no_grad()
def pearth_numerical(
    nustate: Tensor,
    density: object,
    pmns: object,
    dm21_eV2: TensorLike,
    dm3l_eV2: TensorLike,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    *,
    antinu: Union[bool, Tensor] = default.earth_antinu,
    massbasis: bool = default.earth_massbasis,
    full_oscillation: bool = default.earth_full_oscillation,
    nsteps: int = default.earth_probability_nsteps,
    rtol: float = default.earth_rtol,
    atol: float = default.earth_atol,
    ode_method: str | None = None,
    device: Union[str, torch.device] = default.earth_device,
    dtype: torch.dtype = default.dtype,
) -> Tensor | None:
    if torch.is_tensor(antinu):
        if antinu.numel() != 1:
            raise ValueError("pearth_numerical only supports scalar antinu.")
        antinu = bool(antinu.item())

    cdtype = _cdtype_from_real(dtype)
    
    state = nustate.to(device=device)

    Sx, x = numerical_solution(
        density=density,
        pmns=pmns,
        dm21_eV2=dm21_eV2,
        dm3l_eV2=dm3l_eV2,
        E_MeV=E_MeV,
        eta=eta,
        depth_m=depth_m,
        antinu=antinu,
        nsteps=nsteps,
        method=ode_method,
        device=device,
        dtype=dtype,
    )

    if not massbasis:

        amp0 = state.to(dtype=cdtype)

        amp_x = torch.einsum(
            "xij,j->xi",
            Sx,
            amp0,
        )

        evolution = torch.abs(amp_x) ** 2

    else:

        weights = state.to(dtype=dtype)

        U = pmns.U.to(device=device, dtype=cdtype)

        if antinu:
            U = torch.conj(U)

        A = torch.einsum(
            "xij,jk->xik",
            Sx,
            U,
        )

        P_mass_to_flavour = torch.abs(A) ** 2

        evolution = torch.einsum(
            "xij,j->xi",
            P_mass_to_flavour.real,
            weights,
        )

    if full_oscillation:
        return evolution, x

    return evolution[-1]


@torch.no_grad()
def pearth(
    nustate: Tensor,
    density: object,
    pmns: object,
    dm21_eV2: TensorLike,
    dm3l_eV2: TensorLike,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    *,
    method: PearthMethod = default.earth_method,
    antinu: Union[bool, Tensor] = default.earth_antinu,
    massbasis: bool = default.earth_massbasis,
    full_oscillation: bool = default.earth_full_oscillation,
    nsteps: int = default.earth_probability_nsteps,
    rtol: float = default.earth_rtol,
    atol: float = default.earth_atol,
    ode_method: str | None = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
    reunitarize: bool = default.earth_reunitarize,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Dispatch Earth matter-regeneration probabilities by calculation method.

    Args:
        nustate: Initial state with last dimension 3. Interpreted as mass
            weights when massbasis=True, otherwise as flavour amplitudes.
        density: EarthDensity-compatible density model.
        pmns: PMNS object or compatible mixing container.
        dm21_eV2: Solar mass splitting in eV^2.
        dm3l_eV2: Atmospheric mass splitting in eV^2.
        E_MeV: Neutrino energy in MeV.
        eta: Peanuts nadir angle in radians.
        depth_m: Detector depth in meters.
        method: "analytical" or "numerical".
        antinu: Bool or tensor selecting antineutrino propagation.
        massbasis: Select incoherent mass-basis weights or coherent flavour
            amplitudes.
        full_oscillation: For method="numerical", return the full path
            evolution and x grid instead of only the final probability.
        nsteps: Numerical integration steps for method="numerical".
        rtol: Numerical tolerance placeholder for ODE-style methods.
        atol: Numerical tolerance placeholder for ODE-style methods.
        ode_method: Numerical stepping method passed to numerical_solution.
        device: Device for method="numerical"; analytical infers from inputs.
        dtype: Real dtype for numerical calculations.
        reunitarize: For method="analytical", project evolution operators to
            the nearest unitary matrix.

    Returns:
        Probability tensor with final dimension 3. If method="numerical" and
        full_oscillation=True, returns (probabilities_along_path, x_grid).
    """
    method = str(method).lower().strip()

    if method == "analytical":
        return pearth_analytical(
            nustate=nustate,
            density=density,
            pmns=pmns,
            dm21_eV2=dm21_eV2,
            dm3l_eV2=dm3l_eV2,
            E_MeV=E_MeV,
            eta=eta,
            depth_m=depth_m,
            antinu=antinu,
            massbasis=massbasis,
            reunitarize=reunitarize,
        )

    if method == "numerical":
        return pearth_numerical(
            nustate=nustate,
            density=density,
            pmns=pmns,
            dm21_eV2=dm21_eV2,
            dm3l_eV2=dm3l_eV2,
            E_MeV=E_MeV,
            eta=eta,
            depth_m=depth_m,
            antinu=antinu,
            massbasis=massbasis,
            full_oscillation=full_oscillation,
            nsteps=nsteps,
            rtol=rtol,
            atol=atol,
            ode_method=ode_method,
            device=device if device is not None else default.earth_device,
            dtype=dtype,
        )

    raise ValueError("method must be either 'analytical' or 'numerical'.")
