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
    earth_probability_transition(...)
        Build the full Earth flavour-transition probability matrix
        |S_earth[beta, alpha]|^2 from the analytical Earth evolutor.
    earth_probability_state_analytical(...)
        Compute final Earth probabilities using the perturbative analytical
        Earth evolutor.
    earth_probability_state_numerical(...)
        Compute final Earth probabilities using the medium-independent
        numerical evolutor sampled along an Earth trajectory.
    earth_probability_state(...)
        Dispatch to the analytical or numerical Earth probability pipeline.
    earth_probability_integrated(...)
        Average final Earth flavour probabilities over energy, weighted by an
        explicit production spectrum.
"""



from __future__ import annotations

import dataclasses
from typing import Literal, Optional, Union
import torch
from torch import Tensor

import tpeanuts.config.default as default

PearthMethod = Literal["analytical", "numerical"]

from tpeanuts.medium.earth.evolutor import earth_evolutor

from tpeanuts.core.common.oscillation import OscillationParameters, resolve_include_matter_nc
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor, cdtype_from_real, state_tensor
from tpeanuts.core.common.probability import (
    probability_coherent,
    probability_integrated,
    probability_state,
    probability_incoherent,
    probability_transition,
)
from tpeanuts.core.numerical.evolutor import evolutor_numerical
from tpeanuts.core.numerical.geometry import OdeMethod
from tpeanuts.medium.earth.geometry import build_earth_trajectory
from tpeanuts.util.constant import R_E


@torch.no_grad()
def earth_probability_transition(
    profile_earth: object,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    *,
    reunitarize: bool = default.earth_reunitarize,
    legacy_precision: bool = False,
    include_matter_nc: Optional[bool] = None,
) -> Tensor:
    """Build the full Earth flavour-transition probability matrix.

    Uses the analytical perturbative Earth evolutor (the only Earth evolutor
    that produces a full flavour-basis matrix independent of an initial
    state); the numerical segment pipeline (``earth_probability_state_numerical``)
    only evolves an explicit initial state and has no matrix-level
    counterpart.

    Args:
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV.
        eta: Detector nadir angle in radians.
        depth_m: Detector depth in metres.
        reunitarize: Project the Earth evolutor to the nearest unitary matrix.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in the Earth evolutor.
        include_matter_nc: If True/False, applied/not applied (see
            ``medium.earth.evolutor.earth_evolutor``; an explicit ``True``
            still raises if ``profile_earth`` lacks neutron-density
            coefficients). If ``None`` (the default), auto-resolved
            per-call (see ``core.common.oscillation.
            resolve_include_matter_nc``).

    Returns:
        Real tensor |S_earth[beta, alpha]|^2 with final two dimensions final
        flavour and initial flavour.
    """
    U_earth = earth_evolutor(
        profile_earth=profile_earth,
        oscillation=oscillation,
        E=E_MeV,
        eta=eta,
        depth_m=depth_m,
        reunitarize=reunitarize,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
    )

    return probability_transition(U_earth)


@torch.no_grad()
def earth_probability_state_analytical(
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
    include_matter_nc: Optional[bool] = None,
) -> Tensor:
    """Compute Earth probabilities with the analytical perturbative evolutor.

    Args:
        nustate: Initial state with final dimension 3. Interpreted as
            incoherent mass weights when ``massbasis=True`` and coherent
            flavour amplitudes otherwise.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV.
        eta: Detector nadir angle in radians.
        depth_m: Detector depth in metres.
        massbasis: Selects the interpretation of ``nustate``.
        reunitarize: Project the Earth evolutor to the nearest unitary matrix.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in the Earth evolutor.
        include_matter_nc: If True/False, applied/not applied (see
            ``medium.earth.evolutor.earth_evolutor``; an explicit ``True``
            still raises if ``profile_earth`` lacks neutron-density
            coefficients). If ``None`` (the default), auto-resolved
            per-call.

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
        include_matter_nc=include_matter_nc,
    )

    state = state_tensor(
        nustate,
        device=U_earth.device,
        dtype=U_earth.real.dtype if massbasis else U_earth.dtype,
    )

    if massbasis:
        return probability_incoherent(
            U_earth,
            state,
            pmns=oscillation.pmns,
            antinu=oscillation.antinu,
        ).real

    return probability_coherent(
        U_earth,
        state,
    ).real


@torch.no_grad()
def earth_probability_state_numerical(
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
    legacy_precision: bool = False,
    include_matter_nc: Optional[bool] = None,
) -> Tensor | None:
    """Compute Earth probabilities with the numerical segment evolutor.

    Args:
        nustate: Initial state with final dimension 3. Interpreted as
            incoherent mass weights when ``massbasis=True`` and coherent
            flavour amplitudes otherwise.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings, a scalar antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
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
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in numerical segment Hamiltonians.
        include_matter_nc: If True, also sample neutron density along the
            trajectory and forward it as ``n_n_mol_cm3``, enabling the 3+1
            sterile extension's neutral-current matter term (only meaningful
            when ``oscillation.pmns`` is 4-flavour); an explicit ``True``
            still raises ValueError if ``profile_earth`` lacks
            neutron-density coefficients (see
            ``EarthProfile.density_n_x_eta``,
            ``EvenPowerProfileLayered``/``PremTabulatedProfile``
            ``include_neutron=True``). If ``None`` (the default),
            auto-resolved per-call by ``core.common.oscillation.
            resolve_include_matter_nc``: ``True`` when ``oscillation`` is
            the 3+1 sterile extension and
            ``profile_earth.has_neutron_density`` is True, ``False``
            otherwise (with a ``RuntimeWarning`` if sterile was requested
            but the profile lacks neutron-density data). Always ``False``
            for the plain 3-flavour case.

    Returns:
        Final flavour probabilities. If ``full_oscillation=True``, returns
        ``(probabilities_along_path, x_grid)``.
    """
    antinu = oscillation.antinu
    if torch.is_tensor(antinu):
        if antinu.numel() != 1:
            raise ValueError("earth_probability_state_numerical only supports scalar antinu.")
        antinu = bool(antinu.item())
        oscillation = dataclasses.replace(oscillation, antinu=antinu)

    include_matter_nc = resolve_include_matter_nc(
        include_matter_nc,
        oscillation,
        has_neutron_data=getattr(profile_earth, "has_neutron_density", False),
        context_name="earth_probability_state_numerical",
    )

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
        n_e = profile_earth.density_x_eta(
            trajectory.sample_x,
            trajectory.meta["eta_prime"],
        )
        n_n = (
            profile_earth.density_n_x_eta(
                trajectory.sample_x,
                trajectory.meta["eta_prime"],
            )
            if include_matter_nc
            else None
        )
    else:
        r_mid = 0.5 * (1.0 + trajectory.meta["r_d"])
        n_1 = profile_earth.density_x_eta(
            r_mid,
            torch.tensor(0.0, device=dev, dtype=dtype),
        )
        n_e = torch.ones_like(trajectory.sample_x) * as_tensor(
            n_1,
            device=dev,
            dtype=dtype,
        )
        if include_matter_nc:
            n_1n = profile_earth.density_n_x_eta(
                r_mid,
                torch.tensor(0.0, device=dev, dtype=dtype),
            )
            n_n = torch.ones_like(trajectory.sample_x) * as_tensor(
                n_1n,
                device=dev,
                dtype=dtype,
            )
        else:
            n_n = None

    n_e = as_tensor(n_e, device=dev, dtype=dtype)
    n_n = None if n_n is None else as_tensor(n_n, device=dev, dtype=dtype)
    Sx = evolutor_numerical(
        oscillation,
        E_MeV=E_MeV,
        n_e_mol_cm3=n_e,
        n_n_mol_cm3=n_n,
        dx_evolution=trajectory.dx_evolution,
        return_history=full_oscillation,
        device=dev,
        dtype=dtype,
        legacy_precision=legacy_precision,
    )
    x = trajectory.x

    state = state_tensor(
        nustate,
        device=Sx.device,
        dtype=dtype if massbasis else cdtype_from_real(dtype),
    )
    evolution = probability_state(
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
def earth_probability_state(
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
    legacy_precision: bool = False,
    include_matter_nc: Optional[bool] = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Dispatch Earth matter-regeneration probabilities by method.

    This is the main public probability entry point for Earth propagation. It
    selects either the analytical perturbative Earth pipeline or the numerical
    segment pipeline, then returns final flavour probabilities.

    Args:
        nustate: Initial state with last dimension 3. Interpreted as mass
            weights when massbasis=True, otherwise as flavour amplitudes.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute (used
            by both methods).
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
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor throughout Earth propagation.
        include_matter_nc: If True/False, applied/not applied for either
            method (see ``earth_probability_state_analytical``/
            ``earth_probability_state_numerical``). If ``None`` (the
            default), auto-resolved per-call.

    Returns:
        Probability tensor with final dimension 3. If method="numerical" and
        full_oscillation=True, returns (probabilities_along_path, x_grid).
    """
    method = str(method).lower().strip()

    if method == "analytical":
        return earth_probability_state_analytical(
            nustate=nustate,
            profile_earth=profile_earth,
            oscillation=oscillation,
            E_MeV=E_MeV,
            eta=eta,
            depth_m=depth_m,
            massbasis=massbasis,
            reunitarize=reunitarize,
            legacy_precision=legacy_precision,
            include_matter_nc=include_matter_nc,
        )

    if method == "numerical":
        return earth_probability_state_numerical(
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
            legacy_precision=legacy_precision,
            include_matter_nc=include_matter_nc,
        )

    raise ValueError("method must be either 'analytical' or 'numerical'.")


@torch.no_grad()
def earth_probability_integrated(
    nustate: Tensor,
    profile_earth: object,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    spectrum: Tensor,
    *,
    method: PearthMethod = default.earth_method,
    massbasis: bool = default.earth_massbasis,
    nsteps: int = default.earth_probability_nsteps,
    ode_method: OdeMethod | None = None,
    context: Optional[RuntimeContext] = None,
    reunitarize: bool = default.earth_reunitarize,
    legacy_precision: bool = False,
    energy_dim: int = -2,
    include_matter_nc: Optional[bool] = None,
) -> Tensor:
    """Average final Earth flavour probabilities over energy.

    Builds the energy-resolved probabilities with ``earth_probability_state``
    at a fixed nadir angle ``eta`` and averages them with
    ``core.common.probability.probability_integrated``, weighted by an
    explicit production ``spectrum``. This is distinct from
    ``earth_probability_exposure`` (``medium.earth.exposure_integration``),
    which time-averages over the nadir angle instead of integrating over
    energy.

    Args:
        nustate: Initial state passed to ``earth_probability_state``.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy grid in MeV, one-dimensional, matching
            ``E_grid_MeV`` of ``probability_integrated``.
        eta: Detector nadir angle in radians.
        depth_m: Detector depth in metres.
        spectrum: Spectral weight w(E), required (no default).
        method: "analytical" or "numerical" Earth probability pipeline.
        massbasis: Selects the interpretation of ``nustate``.
        nsteps: Numerical integration steps for method="numerical".
        ode_method: Numerical profile sampling rule.
        context: Runtime device/dtype for method="numerical"; analytical
            infers from inputs.
        reunitarize: For method="analytical", project evolution operators to
            the nearest unitary matrix.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor throughout Earth propagation.
        energy_dim: Axis of the resulting probability tensor holding the
            energy grid. Must not be the final (flavour) axis.
        include_matter_nc: If True/False, applied/not applied for either
            method (see ``earth_probability_state``). If ``None`` (the
            default), auto-resolved per-call.

    Returns:
        Spectrum-weighted average probability, with the energy axis removed.
    """
    probabilities = earth_probability_state(
        nustate=nustate,
        profile_earth=profile_earth,
        oscillation=oscillation,
        E_MeV=E_MeV,
        eta=eta,
        depth_m=depth_m,
        method=method,
        massbasis=massbasis,
        full_oscillation=False,
        nsteps=nsteps,
        ode_method=ode_method,
        context=context,
        reunitarize=reunitarize,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
    )

    return probability_integrated(
        probabilities,
        E_MeV,
        spectrum,
        energy_dim=energy_dim,
    )
