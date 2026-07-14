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
Earth matter-regeneration probabilities.

This module converts Earth evolution operators into final flavour
probabilities. It sits above ``medium.earth.evolutor`` and
``core.numerical.evolutor``: those modules build evolution operators, while
this module only interprets an initial state and projects the result to
probabilities.

Two input conventions are supported:

    massbasis=True
        ``nustate`` is an incoherent mass-basis weight vector ``w_i``. The
        final flavour probability is
        ``P_alpha = sum_i |(U_earth U_PMNS)_{alpha i}|^2 w_i``.

    massbasis=False
        ``nustate`` is a coherent flavour-basis amplitude vector. The final
        state is ``psi_final = U_earth psi_initial`` and
        ``P_alpha = |psi_final_alpha|^2``.

Module functions:
    pearth_analytical(...)
        Compute final Earth probabilities using the perturbative analytical
        Earth evolutor.
    pearth_numerical(...)
        Compute final Earth probabilities using the medium-independent
        numerical evolutor sampled along an Earth trajectory.
    pearth(...)
        Dispatch to the analytical or numerical Earth probability pipeline.
"""



from __future__ import annotations

import dataclasses
from typing import Literal, Optional, Union
import torch
from torch import Tensor

import tpeanuts.util.default as default

PearthMethod = Literal["analytical", "numerical"]

from tpeanuts.medium.earth.evolutor import earth_evolutor

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor, cdtype_from_real, state_tensor
from tpeanuts.core.common.probability import (
    probability_from_evolutor,
)
from tpeanuts.core.numerical.evolutor import evolutor_numerical
from tpeanuts.core.numerical.geometry import OdeMethod
from tpeanuts.medium.earth.geometry import build_earth_trajectory
from tpeanuts.util.constant import R_E


@torch.no_grad()
def pearth_analytical(
    nustate: Tensor,
    profile_earth: object,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    *,
    massbasis: bool = default.earth_massbasis,
    reunitarize: bool = default.earth_reunitarize,
    legacy_precision: bool = False,
) -> Tensor:
    """Compute Earth probabilities with the analytical perturbative evolutor.

    Args:
        nustate: Initial state with final dimension 3. Interpreted as
            incoherent mass weights when ``massbasis=True`` and coherent
            flavour amplitudes otherwise.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        eta: Detector nadir angle in radians.
        depth_m: Detector depth in metres.
        massbasis: Selects the interpretation of ``nustate``.
        reunitarize: Project the Earth evolutor to the nearest unitary matrix.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in the Earth evolutor.

    Returns:
        Final flavour probabilities with final dimension 3.
    """
    U_earth = earth_evolutor(
        profile_earth=profile_earth,
        oscillation=oscillation,
        E=E_MeV,
        eta=eta,
        depth_m=depth_m,
        reunitarize=reunitarize,
        legacy_precision=legacy_precision,
    )

    state = state_tensor(
        nustate,
        device=U_earth.device,
        dtype=U_earth.real.dtype if massbasis else U_earth.dtype,
    )
    probabilities = probability_from_evolutor(
        U_earth,
        state,
        pmns=oscillation.pmns,
        massbasis=massbasis,
        antinu=oscillation.antinu,
    )

    return probabilities.real


@torch.no_grad()
def pearth_numerical(
    nustate: Tensor,
    profile_earth: object,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    *,
    massbasis: bool = default.earth_massbasis,
    full_oscillation: bool = default.earth_full_oscillation,
    nsteps: int = default.earth_probability_nsteps,
    ode_method: OdeMethod | None = None,
    context: RuntimeContext = RuntimeContext.resolve(default.earth_device, default.dtype),
    epsilon: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> Tensor | None:
    """Compute Earth probabilities with the numerical segment evolutor.

    Args:
        nustate: Initial state with final dimension 3. Interpreted as
            incoherent mass weights when ``massbasis=True`` and coherent
            flavour amplitudes otherwise.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings and a scalar
            antinu selection.
        E_MeV: Neutrino energy in MeV.
        eta: Detector nadir angle in radians. Numerical mode currently
            supports scalar trajectories.
        depth_m: Detector depth in metres.
        massbasis: Selects the interpretation of ``nustate``.
        full_oscillation: Return probabilities along the full trajectory plus
            the sampled x grid instead of only the final point.
        nsteps: Number of numerical trajectory segments.
        ode_method: Sampling rule passed to the numerical Earth profile.
        context: Runtime device/dtype used by the numerical calculation.
        epsilon: Optional NSI matrix for the numerical Hamiltonian. The
            analytical perturbative Earth path does not use this argument.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in numerical segment Hamiltonians.

    Returns:
        Final flavour probabilities. If ``full_oscillation=True``, returns
        ``(probabilities_along_path, x_grid)``.
    """
    antinu = oscillation.antinu
    if torch.is_tensor(antinu):
        if antinu.numel() != 1:
            raise ValueError("pearth_numerical only supports scalar antinu.")
        antinu = bool(antinu.item())
        oscillation = dataclasses.replace(oscillation, antinu=antinu)

    dev, dtype = context.device, context.dtype
    trajectory = build_earth_trajectory(
        profile_earth=profile_earth,
        eta=eta,
        depth_m=depth_m,
        nsteps=nsteps,
        method=ode_method,
        device=dev,
        dtype=dtype,
        evolution_scale_m=R_E,
    )

    if trajectory.meta["mode"] == "earth_crossing":
        n_e = profile_earth.call(
            trajectory.sample_x,
            trajectory.meta["eta_prime"],
        )
    else:
        r_mid = 0.5 * (1.0 + trajectory.meta["r_d"])
        n_1 = profile_earth.call(
            r_mid,
            torch.tensor(0.0, device=dev, dtype=dtype),
        )
        n_e = torch.ones_like(trajectory.sample_x) * as_tensor(
            n_1,
            device=dev,
            dtype=dtype,
        )

    n_e = as_tensor(n_e, device=dev, dtype=dtype)
    Sx = evolutor_numerical(
        oscillation,
        E_MeV=E_MeV,
        n_e_mol_cm3=n_e,
        dx_evolution=trajectory.dx_evolution,
        return_history=full_oscillation,
        device=dev,
        dtype=dtype,
        epsilon=epsilon,
        legacy_precision=legacy_precision,
    )
    x = trajectory.x

    state = state_tensor(
        nustate,
        device=Sx.device,
        dtype=dtype if massbasis else cdtype_from_real(dtype),
    )
    evolution = probability_from_evolutor(
        Sx,
        state,
        pmns=oscillation.pmns,
        massbasis=massbasis,
        antinu=antinu,
        real_dtype=dtype,
    )

    if full_oscillation:
        return evolution, x

    return evolution


@torch.no_grad()
def pearth(
    nustate: Tensor,
    profile_earth: object,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    *,
    method: PearthMethod = default.earth_method,
    massbasis: bool = default.earth_massbasis,
    full_oscillation: bool = default.earth_full_oscillation,
    nsteps: int = default.earth_probability_nsteps,
    ode_method: OdeMethod | None = None,
    context: Optional[RuntimeContext] = None,
    reunitarize: bool = default.earth_reunitarize,
    epsilon: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Dispatch Earth matter-regeneration probabilities by method.

    This is the main public probability entry point for Earth propagation. It
    selects either the analytical perturbative Earth pipeline or the numerical
    segment pipeline, then returns final flavour probabilities.

    Args:
        nustate: Initial state with last dimension 3. Interpreted as mass
            weights when massbasis=True, otherwise as flavour amplitudes.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        eta: Peanuts nadir angle in radians.
        depth_m: Detector depth in meters.
        method: "analytical" or "numerical".
        massbasis: Select incoherent mass-basis weights or coherent flavour
            amplitudes.
        full_oscillation: For method="numerical", return the full path
            evolution and x grid instead of only the final probability.
        nsteps: Numerical integration steps for method="numerical".
        ode_method: Numerical profile sampling rule passed to the Earth
            numerical profile builder.
        context: Runtime device/dtype for method="numerical"; analytical
            infers from inputs.
        reunitarize: For method="analytical", project evolution operators to
            the nearest unitary matrix.
        epsilon: Optional NSI matrix used only with method="numerical".
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor throughout Earth propagation.

    Returns:
        Probability tensor with final dimension 3. If method="numerical" and
        full_oscillation=True, returns (probabilities_along_path, x_grid).
    """
    method = str(method).lower().strip()

    if method == "analytical":
        return pearth_analytical(
            nustate=nustate,
            profile_earth=profile_earth,
            oscillation=oscillation,
            E_MeV=E_MeV,
            eta=eta,
            depth_m=depth_m,
            massbasis=massbasis,
            reunitarize=reunitarize,
            legacy_precision=legacy_precision,
        )

    if method == "numerical":
        return pearth_numerical(
            nustate=nustate,
            profile_earth=profile_earth,
            oscillation=oscillation,
            E_MeV=E_MeV,
            eta=eta,
            depth_m=depth_m,
            massbasis=massbasis,
            full_oscillation=full_oscillation,
            nsteps=nsteps,
            ode_method=ode_method,
            context=context if context is not None else RuntimeContext.resolve(default.earth_device, default.dtype),
            epsilon=epsilon,
            legacy_precision=legacy_precision,
        )

    raise ValueError("method must be either 'analytical' or 'numerical'.")
