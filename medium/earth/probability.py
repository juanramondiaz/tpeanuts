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

from tpeanuts.medium.earth.evolutor import EarthPerturbativeDiagnostics, earth_evolutor

from tpeanuts.core.common.oscillation import (
    OscillationParameters,
    oscillation_needs_neutron_composition,
    resolve_include_matter_nc,
)
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor, cdtype_from_real, state_tensor
from tpeanuts.util.torch_util import broadcast_tensor
from tpeanuts.core.common.probability import (
    probability_coherent,
    probability_integrated,
    probability_state,
    probability_incoherent,
    probability_transition,
)
from tpeanuts.core.numerical.evolutor import evolutor_numerical
from tpeanuts.core.numerical.geometry import OdeMethod
from tpeanuts.medium.earth.geometry import build_earth_trajectory, validate_eta_range
from tpeanuts.util.constant import R_E


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
    analytic_eigenvalues: bool = False,
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
        analytic_eigenvalues: If True, compute the Earth evolutor's
            eigenvalues with the closed-form Cardano/Ferrari solution
            instead of ``torch.linalg.eigvalsh`` (see
            ``medium.earth.evolutor.earth_evolutor``).

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
        analytic_eigenvalues=analytic_eigenvalues,
    )

    return probability_transition(U_earth)


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
    analytic_eigenvalues: bool = False,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, EarthPerturbativeDiagnostics]:
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
        analytic_eigenvalues: If True, compute the Earth evolutor's
            eigenvalues with the closed-form Cardano/Ferrari solution
            instead of ``torch.linalg.eigvalsh`` (see
            ``medium.earth.evolutor.earth_evolutor``).

    Returns:
        Final flavour probabilities with final dimension 3.
    """
    evolutor_result = earth_evolutor(
        profile_earth=profile_earth,
        oscillation=oscillation,
        E=E_MeV,
        eta=eta,
        depth_m=depth_m,
        reunitarize=reunitarize,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
        analytic_eigenvalues=analytic_eigenvalues,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        U_earth, diagnostics = evolutor_result
    else:
        U_earth = evolutor_result

    state = state_tensor(
        nustate,
        device=U_earth.device,
        dtype=U_earth.real.dtype if massbasis else U_earth.dtype,
    )

    if massbasis:
        probabilities = probability_incoherent(
            U_earth,
            state,
            pmns=oscillation.pmns,
            antinu=oscillation.antinu,
        ).real
    else:
        probabilities = probability_coherent(
            U_earth,
            state,
        ).real

    if return_diagnostics:
        return probabilities, diagnostics
    return probabilities


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
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute Earth probabilities with the numerical segment evolutor.

    Args:
        nustate: Initial state with final dimension 3. Interpreted as
            incoherent mass weights when ``massbasis=True`` and coherent
            flavour amplitudes otherwise.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings, an antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
            ``antinu`` may be a bool or a tensor broadcastable to the
            (energy, eta) grid shape formed from ``E_MeV``/``eta`` (see
            ``util.torch_util.broadcast_tensor``); a tensor that is not
            broadcastable to that shape raises ``ValueError``.
        E_MeV: Neutrino energy in MeV. Scalar or tensor.
        eta: Detector nadir angle(s) in radians, each required to lie in
            [0, pi] (see ``medium.earth.geometry.validate_eta_range``, the
            same check ``method="analytical"`` applies via
            ``earth_evolutor``). Scalar or any batch shape; broadcast
            against ``E_MeV`` the same way ``method="analytical"`` does (two
            differently-sized 1D grids form an independent outer product,
            see ``util.torch_util.broadcast_tensor``), so a full (energy,
            eta) grid is built and propagated in one batched call rather
            than looping in Python.
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
            for the plain 3-flavour case. Independent of and orthogonal to
            the NSI composition term: neutron density is sampled whenever
            this resolves True *or* ``oscillation.nsi.has_neutron_coupling``
            is True (auto-detected, no separate flag needed, any flavour
            count -- see ``core.BSM.bsm_nsi``'s "Composition dependence"
            section).

    Returns:
        Final flavour probabilities. If ``full_oscillation=True``, returns
        ``(probabilities_along_path, x_grid)``.

    Raises:
        ValueError: If any broadcast ``eta`` value lies outside [0, pi], if
            a tensor ``oscillation.antinu`` is not broadcastable to the
            (energy, eta) grid shape, or if
            ``oscillation.nsi.has_neutron_coupling`` is True and
            ``profile_earth`` lacks neutron-density data.
    """
    dev, dtype = context.device, context.dtype
    E_b, eta_b = broadcast_tensor(
        E_MeV, eta, device=dev, dtype=dtype, independent_1d=True,
    )
    validate_eta_range(eta_b)

    # A scalar/bool antinu broadcasts trivially; a tensor antinu must itself
    # be broadcastable *to* the (energy, eta) grid shape of E_b/eta_b (not
    # symmetrically broadcast against it, which could silently pull eta_b
    # into an unrelated extra dimension antinu introduced on its own, e.g. a
    # stray antinu shape with nothing to do with the actual eta grid).
    # Matching eta_b's shape means antinu also lines up with dx_evolution/
    # n_e's leading batch dims below, and evolutor_numerical_segment pads it
    # with one more trailing dim to reach the segment axis (same mechanism
    # the analytical path already relies on via PMNS.select_antinu).
    antinu = oscillation.antinu
    if torch.is_tensor(antinu):
        original_antinu_shape = tuple(antinu.shape)
        antinu = antinu.to(device=dev, dtype=torch.bool)
        try:
            antinu = torch.broadcast_to(antinu, eta_b.shape)
        except RuntimeError as exc:
            raise ValueError(
                "oscillation.antinu must be broadcastable to the (energy, "
                f"eta) grid shape {tuple(eta_b.shape)}; got antinu shape "
                f"{original_antinu_shape}."
            ) from exc
        oscillation = dataclasses.replace(oscillation, antinu=antinu)

    include_matter_nc = resolve_include_matter_nc(
        include_matter_nc,
        oscillation,
        has_neutron_data=getattr(profile_earth, "has_neutron_density", False),
        context_name="earth_probability_state_numerical",
    )

    trajectory = build_earth_trajectory(
        profile_earth=profile_earth,
        eta=eta_b,
        depth_m=depth_m,
        nsteps=nsteps,
        method=ode_method,
        device=dev,
        dtype=dtype,
        evolution_scale_m=R_E,
    )

    # Case A (earth-crossing) and case B (local-constant) entries are both
    # present in a mixed eta batch; compute both branches with batched
    # tensor ops and select per entry -- no Python loop over eta.
    is_earth_crossing = trajectory.meta["is_earth_crossing"]
    eta_prime_seg = trajectory.meta["eta_prime"][..., None]
    r_mid = 0.5 * (1.0 + trajectory.meta["r_d"])
    zero = torch.tensor(0.0, device=dev, dtype=dtype)

    n_e_crossing = profile_earth.density_x_eta(trajectory.sample_x, eta_prime_seg)
    n_e_constant = profile_earth.density_x_eta(r_mid, zero)
    n_e = torch.where(is_earth_crossing[..., None], n_e_crossing, n_e_constant)

    # Sampling neutron density is needed for the sterile NC term
    # (include_matter_nc) and/or the NSI composition term
    # (oscillation.nsi.epsilon_n, any flavour count -- see
    # core.BSM.bsm_nsi's "Composition dependence" section); the two are
    # independent, so this ORs them rather than gating solely on the
    # sterile-specific flag.
    needs_eps_n = oscillation_needs_neutron_composition(oscillation)
    if needs_eps_n and not getattr(profile_earth, "has_neutron_density", False):
        raise ValueError(
            "oscillation.nsi has non-zero eps_*_n (composition-dependent "
            "NSI) but profile_earth was not built with neutron-density "
            "coefficients (see EarthProfile.has_neutron_density, "
            "EvenPowerProfileLayered/PremTabulatedProfile "
            "include_neutron=True)."
        )
    if include_matter_nc or needs_eps_n:
        n_n_crossing = profile_earth.density_n_x_eta(trajectory.sample_x, eta_prime_seg)
        n_n_constant = profile_earth.density_n_x_eta(r_mid, zero)
        n_n = torch.where(is_earth_crossing[..., None], n_n_crossing, n_n_constant)
    else:
        n_n = None

    n_e = as_tensor(n_e, device=dev, dtype=dtype)
    n_n = None if n_n is None else as_tensor(n_n, device=dev, dtype=dtype)
    Sx = evolutor_numerical(
        oscillation,
        E_MeV=E_b,
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
    analytic_eigenvalues: bool = False,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, EarthPerturbativeDiagnostics]:
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
            the nearest unitary matrix. Must be False (the default) for
            method="numerical", which has no unitary-projection step.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor throughout Earth propagation.
        include_matter_nc: If True/False, applied/not applied for either
            method (see ``earth_probability_state_analytical``/
            ``earth_probability_state_numerical``). If ``None`` (the
            default), auto-resolved per-call.
        analytic_eigenvalues: If True, compute the analytical evolutor's
            eigenvalues with the closed-form Cardano/Ferrari solution
            instead of ``torch.linalg.eigvalsh`` (see
            ``earth_probability_state_analytical``). Only meaningful with
            ``method="analytical"``.
        return_diagnostics: Return analytical first-order validity
            diagnostics together with the probabilities. Rejected for the
            numerical method.

    Returns:
        Probability tensor with final dimension 3. With analytical
        ``return_diagnostics=True``, returns ``(probabilities, diagnostics)``.
        If method="numerical" and
        full_oscillation=True, returns (probabilities_along_path, x_grid).

    Raises:
        ValueError: If ``method`` is not "analytical"/"numerical", if
            ``reunitarize=True`` is requested with ``method="numerical"``, if
            ``full_oscillation=True`` is requested with
            ``method="analytical"`` -- both combinations would otherwise
            silently drop the flag instead of applying it, since
            ``earth_probability_state_analytical`` has no ``full_oscillation``
            parameter and ``earth_probability_state_numerical`` never
            re-unitarizes -- or if ``analytic_eigenvalues=True`` is requested
            with ``method="numerical"``, which propagates via
            ``torch.linalg.matrix_exp`` and never diagonalizes.
    """
    method = str(method).lower().strip()

    if method not in ("analytical", "numerical"):
        raise ValueError("method must be either 'analytical' or 'numerical'.")

    if method == "numerical" and reunitarize:
        raise ValueError(
            "reunitarize=True has no effect for method='numerical': only "
            "the analytical perturbative evolutor is re-unitarized. Pass "
            "reunitarize=False (the default), or use method='analytical'."
        )
    if method == "analytical" and full_oscillation:
        raise ValueError(
            "full_oscillation=True has no effect for method='analytical': "
            "only method='numerical' can return the full trajectory. Pass "
            "full_oscillation=False (the default), or use method='numerical'."
        )
    if method == "numerical" and analytic_eigenvalues:
        raise ValueError(
            "analytic_eigenvalues=True has no effect for method='numerical': "
            "only the analytical perturbative evolutor supports closed-form "
            "eigenvalues (method='numerical' propagates via "
            "torch.linalg.matrix_exp and never diagonalizes). Pass "
            "analytic_eigenvalues=False (the default), or use "
            "method='analytical'."
        )
    if method == "numerical" and return_diagnostics:
        raise ValueError(
            "return_diagnostics=True is only available for "
            "method='analytical', which has an explicit first-order term."
        )

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
            analytic_eigenvalues=analytic_eigenvalues,
            return_diagnostics=return_diagnostics,
        )

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
    analytic_eigenvalues: bool = False,
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
        analytic_eigenvalues: If True, use the closed-form Cardano/Ferrari
            eigenvalues instead of ``torch.linalg.eigvalsh`` (see
            ``earth_probability_state``). Only meaningful with
            ``method="analytical"``.

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
        analytic_eigenvalues=analytic_eigenvalues,
    )

    return probability_integrated(
        probabilities,
        E_MeV,
        spectrum,
        energy_dim=energy_dim,
    )
