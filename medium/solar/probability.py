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

"""Solar-neutrino mass and flavour probabilities.

Three propagation methods
--------------------------
``method="numerical"`` performs coherent, segment-discretised propagation
through the solar profile (``medium.solar.evolutor.mass_weights_numerical``)
without imposing adiabaticity or a separate Landau--Zener correction. Its
accuracy depends on the radial discretisation. Supports the plain SM, NSI,
the 3+1 sterile extension, or both.

``method="adiabatic_approximated"`` (the default) evaluates the closed-form
matter-mixing angles theta12^M/theta13^M (``medium.solar.matter_mixing``,
via ``medium.solar.adiabatic.mass_weights_adiabatic_approximated``) at each
production point, under the adiabatic approximation. Restricted to the
plain 3-flavour Standard Model (no NSI, no 3+1 sterile extension): the
formulas are a 2-level reduction with no NSI/sterile generalisation. May
apply the local two-level Landau-Zener correction via ``profile.use_LZ``.

``method="adiabatic_exact"`` diagonalises the full flavour-basis Hamiltonian
pointwise (``medium.solar.adiabatic.mass_weights_adiabatic_exact``), still
under the adiabatic approximation but without the 2-level reduction
``"adiabatic_approximated"`` relies on -- so it is strictly more general
(plain SM, NSI, the 3+1 sterile extension, or both) and, for the plain SM
case, more exact than ``"adiabatic_approximated"``. Has no Landau-Zener
counterpart: the two-level crossing formula has no closed-form
generalisation to off-diagonal NSI couplings or a fourth level, so
``profile.use_LZ=True`` raises with this method.

Notes
------
``solar_probability_mass`` is the single entry point that validates every
method / ``use_LZ`` / BSM-extension combination and raises a specific error
for each unsupported one (see ``Raises`` on its docstring);
``medium.solar.adiabatic``'s two functions and
``medium.solar.evolutor.mass_weights_numerical`` assume their caller has
already validated the combination and perform no such checks themselves.

``profile.use_LZ`` is valid only for ``method="adiabatic_approximated"`` and
scalar or 1-D energy grids. It applies the crossing probability only to
neutrinos produced above the 1--2 resonance density (see
``medium.solar.landau_zener.landau_zener_spatial_correction``).

The adiabatic paths produce mass weights rather than a coherent transition
operator, so this module exposes state probabilities but no adiabatic
``solar_probability_transition``.
"""


from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from tpeanuts.core.common.oscillation import OscillationParameters, resolve_include_matter_nc
from tpeanuts.core.common.probability import (
    probability_incoherent,
    probability_integrated,
)
from tpeanuts.core.numerical.geometry import OdeMethod
from tpeanuts.medium.solar.adiabatic import (
    mass_weights_adiabatic_approximated,
    mass_weights_adiabatic_exact,
)
from tpeanuts.medium.solar.evolutor import mass_weights_numerical
from tpeanuts.medium.solar.landau_zener import landau_zener_spatial_correction
from tpeanuts.util.type import cdtype_from_real


TensorLike = Union[float, int, torch.Tensor]

_SOLAR_METHODS = ("numerical", "adiabatic_approximated", "adiabatic_exact")


def solar_probability_mass(
    oscillation: OscillationParameters,
    E: TensorLike,
    profile: object,
    sources: str | Sequence[str],
    *,
    method: str = "adiabatic_approximated",
    legacy_precision: bool = False,
    include_matter_nc: Optional[bool] = None,
    numerical_sampling: Optional[OdeMethod] = "midpoint",
) -> torch.Tensor:
    """Integrate solar production profiles into mass-basis probabilities.

    This is the single entry point that validates every method / ``use_LZ``
    / BSM-extension combination (see ``Raises``) and then dispatches to
    exactly one weight-computation path, none of which repeat these checks:

        ``method="numerical"``
            ``medium.solar.evolutor.mass_weights_numerical``.
        ``method="adiabatic_approximated"``
            ``medium.solar.adiabatic.mass_weights_adiabatic_approximated``,
            with an optional Landau-Zener correction built here via
            ``medium.solar.landau_zener.landau_zener_spatial_correction``
            when ``profile.use_LZ`` is True.
        ``method="adiabatic_exact"``
            ``medium.solar.adiabatic.mass_weights_adiabatic_exact``.

    Args:
        oscillation: Oscillation parameters supplying theta12, theta13,
            DeltamSq21, DeltamSq3l, and the optional ``nsi`` (NSIConfig)
            attribute.
        E: Neutrino energy in MeV. Scalar or 1-D tensor in the standard
            pipeline; multi-dimensional E is supported for
            ``method="adiabatic_approximated"``/``"adiabatic_exact"`` as long
            as ``profile.use_LZ`` is False.
        profile: SolarProfile-like object exposing radius, density,
            production_distribution(), mass_weights_integrate(), the
            optional ``use_LZ`` boolean flag, (when ``include_matter_nc=True``)
            ``density_n``, and (when ``method="numerical"``) the full
            ``radius``/``density`` grid.
        sources: Source key or ordered source keys available in ``profile``.
        method: ``"numerical"``, ``"adiabatic_approximated"`` (default), or
            ``"adiabatic_exact"`` (see module docstring).
        legacy_precision: If True, evaluate the matter-mixing angles/potential
            with the legacy peanuts ``Vk``/prefactor for bit-comparable
            validation (see ``medium.solar.matter_mixing``). Ignored on
            ``method="adiabatic_exact"``, which has no legacy peanuts
            counterpart to compare against (legacy peanuts is 3-flavour SM
            only).
        include_matter_nc: If True, also apply the 3+1 sterile extension's
            neutral-current matter term. If False, never apply it. If
            ``None`` (the default), resolved automatically by
            ``core.common.oscillation.resolve_include_matter_nc``: ``True``
            when ``oscillation`` is the 3+1 sterile extension and the
            profile has neutron-density data available, ``False`` otherwise
            (with a ``RuntimeWarning`` if sterile was requested but the data
            is not available -- see that function's docstring). Always
            ``False`` for the plain 3-flavour case regardless (V_NC is an
            unobservable common phase there, mirroring
            ``hamiltonian_matter_reduced``'s own convention).
        numerical_sampling: Segment density-sampling rule passed to
            ``medium.solar.evolutor.build_solar_trajectory``. Only used when
            ``method="numerical"``.

    Returns:
        Normalized incoherent mass-basis probabilities with leading source
        dimensions, optional energy dimensions, and final mass-index
        dimension N (3 or 4, matching ``oscillation.pmns.n_flavours``).

    Raises:
        ValueError: If ``method`` is not one of ``"numerical"``,
            ``"adiabatic_approximated"``, ``"adiabatic_exact"``; if
            ``method="adiabatic_approximated"`` and ``oscillation`` carries
            NSI and/or the 3+1 sterile extension; if ``profile.use_LZ`` is
            True together with a ``method`` other than
            ``"adiabatic_approximated"``; if ``profile.use_LZ`` is True with
            a multi-dimensional energy grid; or if ``include_matter_nc``
            resolves to True (explicitly or via auto-resolution) and the
            required neutron-density field is not set on ``profile``.
    """
    if method not in _SOLAR_METHODS:
        raise ValueError(
            f"method must be one of {_SOLAR_METHODS!r}, got {method!r}."
        )

    fractions = profile.production_distribution(sources)
    radius_samples = profile.production_radius
    E_t = torch.as_tensor(E, device=radius_samples.device, dtype=radius_samples.dtype)

    n_flavours = int(oscillation.pmns.n_flavours)
    use_lz = getattr(profile, "use_LZ", False)

    if method == "adiabatic_approximated" and oscillation.BSM_extension:
        raise ValueError(
            "method='adiabatic_approximated' only supports the plain "
            "3-flavour Standard Model: its closed-form matter-mixing-angle "
            "formulas (theta12_M/theta13_M) have no generalisation to "
            "off-diagonal NSI couplings or the 3+1 sterile extension. Use "
            "method='adiabatic_exact' (pointwise Hamiltonian "
            "diagonalisation, still adiabatic) or method='numerical' "
            "(full coherent propagation) instead."
        )

    if use_lz and method == "numerical":
        raise ValueError(
            "profile.use_LZ=True has no effect with method='numerical': the "
            "coherent evolutor already captures every non-adiabatic "
            "transition directly, so there is nothing for Landau-Zener to "
            "correct. Set profile.use_LZ=False before calling with "
            "method='numerical'."
        )
    if use_lz and method == "adiabatic_exact":
        raise ValueError(
            "profile.use_LZ=True is not supported with "
            "method='adiabatic_exact': the two-level Landau-Zener crossing "
            "formula has no closed-form generalisation to off-diagonal NSI "
            "couplings or a fourth level. Set profile.use_LZ=False, or use "
            "method='adiabatic_approximated' for the plain 3-flavour "
            "analytic path with Landau-Zener support."
        )
    if use_lz and E_t.ndim > 1:
        raise ValueError(
            "profile.use_LZ=True requires a scalar or 1-D energy grid: the "
            f"Landau-Zener correction is not implemented for E.ndim={E_t.ndim}. "
            "Reshape E to at most 1-D, or set profile.use_LZ=False."
        )

    has_neutron_data = getattr(profile, "density_n", None) is not None
    include_matter_nc = resolve_include_matter_nc(
        include_matter_nc,
        oscillation,
        has_neutron_data=has_neutron_data,
        context_name="solar_probability_mass",
    )

    ########################################
    # method == "numerical"
    ########################################
    if method == "numerical":
        weights_r = mass_weights_numerical(
            oscillation,
            E_t,
            profile,
            method=numerical_sampling,
            include_matter_nc=include_matter_nc,
            legacy_precision=legacy_precision,
        )
        return profile.mass_weights_integrate(weights_r, fractions, E_t.ndim)

    density = profile.electron_density(radius_samples)

    ########################################
    # method == "adiabatic_exact"
    ########################################
    if method == "adiabatic_exact":
        n_n_for_exact: Optional[torch.Tensor] = None
        if include_matter_nc and n_flavours == 4:
            density_n = getattr(profile, "density_n", None)
            if density_n is None:
                raise ValueError(
                    "include_matter_nc=True requires profile.density_n to be "
                    "set (e.g. the default SolarProfile.default() "
                    "construction, which derives it from the struct+nu "
                    "composition table via medium.solar.io.load_solar_"
                    "composition); this profile does not expose a "
                    "neutron-density companion."
                )
            n_n_for_exact = profile.neutron_density(radius_samples)

        weights_r = mass_weights_adiabatic_exact(
            oscillation,
            E_t[..., None],
            density,
            n_n_mol_cm3=n_n_for_exact,
        )
        return profile.mass_weights_integrate(weights_r, fractions, E_t.ndim)

    ########################################
    # method == "adiabatic_approximated"
    ########################################
    p_lz_spatial: Optional[torch.Tensor] = None
    if use_lz:
        p_lz_spatial = landau_zener_spatial_correction(
            oscillation, E_t, profile, radius_samples, legacy_precision=legacy_precision,
        )

    weights_r = mass_weights_adiabatic_approximated(
        oscillation,
        E_t[..., None],
        density,
        p_lz=p_lz_spatial,
        legacy_precision=legacy_precision,
    )
    return profile.mass_weights_integrate(weights_r, fractions, E_t.ndim)


def solar_probability_state(
    oscillation: OscillationParameters,
    E: TensorLike,
    profile: object,
    sources: str | Sequence[str],
    *,
    method: str = "adiabatic_approximated",
    legacy_precision: bool = False,
    include_matter_nc: Optional[bool] = None,
    numerical_sampling: Optional[OdeMethod] = "midpoint",
) -> torch.Tensor:
    """Compute solar flavour probabilities for one or more sources.

    Integrates the source production profile in the mass basis and projects
    the resulting incoherent weights through
    ``P_alpha = sum_i |U_alpha_i|^2 P_i``. The result is a probability, not a
    flux; multiple sources retain a leading source dimension.

    Args:
        oscillation: Built pmns object (3-flavour or 3+1 sterile) plus mass
            splittings, antinu selection, and the optional ``nsi``
            (NSIConfig) attribute.
        E: Neutrino energy in MeV.
        profile: SolarProfile-like object exposing radius, density,
            production_distribution(), the optional ``use_LZ`` boolean flag, and
            (when ``include_matter_nc=True``) ``density_n``.
        sources: Source key or ordered source keys available in ``profile``.
        method: ``"numerical"``, ``"adiabatic_approximated"`` (default), or
            ``"adiabatic_exact"`` (see ``solar_probability_mass``).
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation (see ``medium.solar.matter_mixing``).
            Ignored on ``method="adiabatic_exact"`` and
            ``method="numerical"``.
        include_matter_nc: If True/False, applied/not applied. If ``None``
            (the default), auto-resolved per-call (see
            ``solar_probability_mass``/``core.common.oscillation.
            resolve_include_matter_nc``).
        numerical_sampling: Segment density-sampling rule, only used when
            ``method="numerical"`` (see ``solar_probability_mass``).

    Returns:
        Final flavour probabilities with leading source dimensions, optional
        energy dimensions, and final flavour dimension N (3 or 4, matching
        ``oscillation.pmns.n_flavours``).
    """
    weights = solar_probability_mass(
        oscillation,
        E,
        profile,
        sources,
        method=method,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
        numerical_sampling=numerical_sampling,
    )

    n_flavours = int(oscillation.pmns.n_flavours)
    identity = torch.eye(
        n_flavours,
        device=profile.production_radius.device,
        dtype=cdtype_from_real(weights.dtype),
    )

    return probability_incoherent(
        identity,
        weights,
        pmns=oscillation.pmns,
        antinu=oscillation.antinu,
        real_dtype=weights.dtype,
    )


def solar_probability_integrated(
    oscillation: OscillationParameters,
    E: TensorLike,
    profile: object,
    sources: str | Sequence[str],
    spectrum: torch.Tensor,
    *,
    method: str = "adiabatic_approximated",
    legacy_precision: bool = False,
    energy_dim: int = -2,
    include_matter_nc: Optional[bool] = None,
    numerical_sampling: Optional[OdeMethod] = "midpoint",
) -> torch.Tensor:
    """Average final solar flavour probabilities over energy.

    Uses ``solar_probability_state`` followed by the normalized spectral
    average from ``core.common.probability.probability_integrated``.

    Args:
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
        E: Neutrino energy grid in MeV, one-dimensional, matching
            ``E_grid_MeV`` of ``probability_integrated``.
        profile: SolarProfile-like object exposing radius, density,
            production_distribution(), and the optional ``use_LZ`` boolean flag.
        sources: Source key or ordered source keys available in ``profile``.
        spectrum: Spectral weight w(E), required (no default).
        method: ``"numerical"``, ``"adiabatic_approximated"`` (default), or
            ``"adiabatic_exact"`` (see ``solar_probability_mass``).
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation (see ``medium.solar.matter_mixing``).
        energy_dim: Axis of the resulting probability tensor holding the
            energy grid. Must not be the final (flavour) axis.
        include_matter_nc: If True, also apply the 3+1 sterile extension's
            neutral-current matter term (see ``solar_probability_mass``).
        numerical_sampling: Segment density-sampling rule, only used when
            ``method="numerical"``.

    Returns:
        Spectrum-weighted average probability, with the energy axis removed.
    """
    probabilities = solar_probability_state(
        oscillation,
        E,
        profile,
        sources,
        method=method,
        legacy_precision=legacy_precision,
        include_matter_nc=include_matter_nc,
        numerical_sampling=numerical_sampling,
    )

    return probability_integrated(
        probabilities,
        E,
        spectrum,
        energy_dim=energy_dim,
    )
