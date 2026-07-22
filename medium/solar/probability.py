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
Solar neutrino probabilities.

This module computes the incoherent mass-eigenstate weights produced in the
solar interior and converts them into final flavour probabilities using the
vacuum PMNS projection. It works directly with tabulated solar radius,
electron-density, and production-fraction samples.

Two propagation methods, selected by the ``method`` argument of
``solar_probability_mass``/``solar_probability_state``/
``solar_probability_integrated``:

    method="adiabatic" (the default)
        Reads off local matter-eigenstate weights at each production point
        via the adiabatic theorem (``Tei``). For the plain 3-flavour case
        this uses closed-form mixing-angle formulas, optionally corrected
        for non-adiabatic level jumps at the MSW resonance with the
        closed-form two-level Landau-Zener formula (``landau_zener.py``);
        for NSI and/or the 3+1 sterile extension, where no closed form
        exists, ``Tei`` instead diagonalises the full Hamiltonian at each
        grid point (still the same adiabatic-theorem approximation, just
        without a formula to fall back on -- see "NSI and 3+1 sterile
        extension" below). Landau-Zener has no generalisation to that
        diagonalization sub-path, so combining ``profile.use_LZ=True`` with
        NSI/sterile raises (see "Landau-Zener (LZ) corrections" below).
    method="numerical"
        Builds a genuine coherent evolutor across the tabulated density
        profile (``medium.solar.evolutor``, reusing ``core.numerical``) and
        projects onto the vacuum mass basis -- no adiabatic assumption, no
        Landau-Zener approximation, exact for any NSI/sterile/NC
        combination, at the cost of the same accuracy-vs-resolution
        trade-off as ``core.numerical`` elsewhere in the project (see
        ``evolutor.py``'s module docstring).

Landau-Zener (LZ) corrections
------------------------------
By default the calculation is purely adiabatic (P_LZ = 0 everywhere). When
``solar_profile.use_LZ`` is True, ``solar_probability_mass`` queries
``landau_zener.plz`` and ``landau_zener.resonance_radius`` to compute a
spatially resolved LZ transition probability that is passed to ``Tei``,
applied only to production points above the resonance density (radii
smaller than r_res(E)).

``profile.use_LZ=True`` raises ``ValueError`` (rather than silently ignoring
the flag) when combined with: ``method="numerical"`` (the coherent evolutor
already captures every non-adiabatic transition, so there is nothing for LZ
to correct); NSI and/or the 3+1 sterile extension (the two-level LZ formula
has no closed-form generalisation to off-diagonal couplings or a fourth
level); or a multi-dimensional energy grid (not implemented). See
``solar_probability_mass``'s ``Raises`` section.

NSI and 3+1 sterile extension
------------------------------
When ``oscillation.nsi`` is set, or ``oscillation.pmns.n_flavours == 4`` (the
3+1 sterile extension), ``Tei``, ``solar_probability_mass``, and
``solar_probability_state`` replace the analytic 3-flavour mixing-angle path
by diagonalising the total Hamiltonian
``H = H_kin + H_mat`` (including ``H_mat^NSI`` when NSI is set, and the
sterile neutral-current term when ``include_matter_nc=True`` and neutron
density is supplied), assembled via
``tpeanuts.core.common.hamiltonian.hamiltonian_flavour`` and diagonalised
with ``torch.linalg.eigh`` at each (energy, density) grid point. The squared
modulus of the electron-flavour component of each matter eigenstate gives
the production weights directly, generalising to any flavour count N:

    w_i(E, r) = |<ν_e | ν_i^M(E, r)>|²  =  |eigvec[..., 0, i]|²,  i = 1..N

``eigh`` sorts columns by ascending eigenvalue, not by the physical mass
index -- the two agree for normal ordering but not for inverted ordering
(Δm²₃₁ < 0), so the raw columns are reindexed with a fixed permutation
computed once per ``oscillation`` from the vacuum limit; see ``Tei``'s
docstring and implementation.

This diagonalization path supports arbitrary off-diagonal NSI, the 3+1
sterile extension (θ14, θ24, θ34, Δm²41, all carried by
``oscillation.pmns``/``oscillation.mass_spectrum`` and read generically by
``hamiltonian_flavour`` -- nothing here is 3-flavour-specific once this path
is taken), and both simultaneously, and enables the study of LMA-Dark and
other NSI-degenerate scenarios. See ``include_matter_nc`` on
``solar_probability_mass`` for the sterile neutral-current term, which
requires ``profile.density_n`` (derived from the solar composition table,
see ``medium.solar.io``); omitting it (the default) reproduces the CC-only
sterile matter term used throughout the rest of this project.

No transition function
-----------------------
Unlike the other media, this module does not expose a
``solar_probability_transition``. The adiabatic solar model has no coherent
evolution operator ``S`` to build a transition matrix from: production
weights are computed directly in the mass basis via MSW level crossing
(``Tei``/``solar_probability_mass``), not by evolving a flavour-basis state
through an evolutor. ``solar_probability_state`` is therefore the lowest-tier
public probability function for this medium.

Module functions:
    Tei(...)
        Compute matter-basis production weights for one radius/density grid.
    solar_probability_mass(...)
        Integrate source production profiles into mass-basis probabilities.
    solar_probability_state(...)
        Convert source mass-basis probabilities into flavour probabilities.
    solar_probability_integrated(...)
        Average final flavour probabilities over energy, weighted by an
        explicit production spectrum.
"""



from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.hamiltonian import hamiltonian_flavour
from tpeanuts.core.common.oscillation import OscillationParameters, resolve_include_matter_nc
from tpeanuts.core.common.probability import (
    probability_incoherent,
    probability_integrated,
)
from tpeanuts.core.numerical.geometry import OdeMethod
from tpeanuts.medium.solar.evolutor import mass_weights_numerical
from tpeanuts.medium.solar.landau_zener import plz, resonance_radius
from tpeanuts.medium.solar.matter_mixing import th12_M, th13_M
from tpeanuts.util.type import cdtype_from_real


TensorLike = Union[float, int, torch.Tensor]


def Tei(
    oscillation: OscillationParameters,
    E: TensorLike,
    ne: TensorLike,
    *,
    p_lz: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
    n_n_mol_cm3: Optional[TensorLike] = None,
) -> torch.Tensor:
    """Compute adiabatic mass-eigenstate production weights.

    Returns the probability that an electron neutrino produced in matter
    is projected onto each of the N effective mass eigenstates (N = 3 for
    the plain Standard Model, N = 4 for the 3+1 sterile extension).

    Diagonalization path  — ``oscillation.nsi`` is set, or
    ``oscillation.pmns.n_flavours == 4`` (3+1 sterile)
    ------------------------------------------------------------------------
    The total Hamiltonian ``H = H_kin + H_mat`` is assembled via
    ``tpeanuts.core.common.hamiltonian.hamiltonian_flavour`` -- which reads
    ``oscillation.nsi.epsilon`` when NSI is set, and generalises to N = 4
    automatically from ``oscillation.pmns``/``oscillation.mass_spectrum``
    when the 3+1 sterile extension is active (θ14, θ24, θ34, Δm²41), with
    the optional sterile neutral-current term when ``n_n_mol_cm3`` is
    supplied -- and diagonalised with ``torch.linalg.eigh``. Production
    weights are the squared electron-flavour components of each matter
    eigenstate:

        w_i(E, r) = |<nu_e | nu_i^M(E, r)>|^2 = |eigvec[..., 0, i]|^2,  i = 1..N

    ``eigh`` returns columns sorted by ascending eigenvalue, which is *not*
    the same as the physical mass index ``i`` whenever the vacuum
    eigenvalue order and the ascending order disagree -- true for normal
    ordering, false for inverted ordering. The raw ``eigh`` output is
    therefore reindexed with a fixed (E/ne-independent) permutation, derived
    by diagonalising the vacuum-limit (n_e = n_n = 0) Hamiltonian and
    matching its eigenvectors against the actual PMNS matrix columns, before
    being returned -- so index ``i`` always means the physical vacuum mass
    eigenstate ``nu_{i+1}`` regardless of ordering.

    This diagonalization path supports arbitrary off-diagonal NSI, the 3+1
    sterile extension, or both simultaneously -- nothing here branches
    explicitly on which combination is active, since ``hamiltonian_flavour``
    already does. The ``p_lz`` argument is ignored on this path (with a
    ``RuntimeWarning`` if supplied) because the two-level Landau-Zener
    crossing formula has no closed-form generalisation to off-diagonal NSI
    or a fourth level.

    Analytic (SM) path  — ``oscillation.nsi`` is None and
    ``oscillation.pmns.n_flavours == 3``
    ------------------------------------------------------------------------
    The matter mixing angles theta_12^M and theta_13^M are evaluated
    analytically via ``matter_mixing.th12_M`` and ``matter_mixing.th13_M``.
    When ``p_lz`` is also provided, the nu_1^M / nu_2^M weights are mixed to
    account for the Landau-Zener transition at the MSW resonance:

        w_1 = (1 - P_LZ) cos^2(theta_12^M) + P_LZ sin^2(theta_12^M)
        w_2 = (1 - P_LZ) sin^2(theta_12^M) + P_LZ cos^2(theta_12^M)

    The nu_3^M weight (sin^2 theta_13^M) is unaffected. At standard solar
    neutrino energies, the density required for the theta_13 resonance is
    higher than the maximum density reached inside the Sun, so standard solar
    neutrinos do not cross a physical 1--3 resonance.

    Args:
        oscillation: Oscillation parameters supplying theta12, theta13,
            DeltamSq21, DeltamSq3l, and the optional ``nsi`` (NSIConfig)
            attribute. A 4-flavour ``oscillation.pmns`` (3+1 sterile)
            selects the diagonalization path regardless of ``nsi``.
        E: Neutrino energy in MeV.
        ne: Electron density samples in mol/cm^3.
        p_lz: Optional Landau-Zener transition probability tensor,
            broadcastable with the internal cos^2(theta_12^M) /
            sin^2(theta_12^M) outputs. Ignored (with a warning) on the
            diagonalization path.
        legacy_precision: If True, evaluate the analytic matter-mixing angles
            with the legacy peanuts ``Vk`` prefactor for bit-comparable
            validation (see ``medium.solar.matter_mixing``). Ignored on the
            diagonalization path.
        n_n_mol_cm3: Optional neutron density samples in mol/cm^3, enabling
            the 3+1 sterile neutral-current term (see
            ``core.common.hamiltonian.hamiltonian_matter_reduced``). Only
            meaningful together with a 4-flavour ``oscillation.pmns``;
            silently ignored otherwise (mirroring
            ``hamiltonian_matter_reduced``'s own convention, since V_NC is an
            unobservable common phase in the plain 3-flavour case).

    Returns:
        Real tensor of matter-production weights with final mass-index
        dimension N (3 or 4), shape broadcast-compatible with ``(E, ne)``.
    """
    n_flavours = int(oscillation.pmns.n_flavours)
    diagonalization_path = oscillation.nsi is not None or n_flavours == 4

    if diagonalization_path:
        if p_lz is not None:
            warnings.warn(
                "Tei: p_lz is ignored on the diagonalization path (NSI "
                "and/or the 3+1 sterile extension are active): the "
                "two-level Landau-Zener crossing formula has no closed-form "
                "generalisation to off-diagonal NSI or a fourth level.",
                RuntimeWarning,
                stacklevel=2,
            )

        H = hamiltonian_flavour(
            oscillation,
            E,
            ne,
            n_n_mol_cm3=n_n_mol_cm3,
            evolution_scale_m=constant.R_SUN,
        )
        # eigh returns columns as eigenvectors; row 0 = electron-flavour
        _, eigvec = torch.linalg.eigh(H)          # (..., N, N)
        weights_eigh_order = eigvec[..., 0, :].abs() ** 2  # (..., N)

        # eigh sorts columns by ascending eigenvalue, not by physical mass
        # index -- the two agree for normal ordering (0 < Delta_m^2_21 <
        # Delta_m^2_31) but not for inverted ordering (Delta_m^2_31 < 0), so
        # eigh's raw column order silently swaps which weight belongs to
        # which vacuum mass eigenstate under IO. At n_e = n_n = 0 the
        # NSI/NC matter terms vanish exactly (both proportional to
        # density), leaving the pure vacuum kinetic Hamiltonian
        # H_kin = U diag(k_1,...,k_N) U^dagger, whose eigenvectors are
        # therefore exactly U's columns (up to phase); neither this
        # correspondence nor eigh's ordering depends on E (each k_i is a
        # positive multiple of Delta_m^2_i1) or antinu (the kinetic term
        # does not depend on it), so comparing eigh's vacuum-limit
        # eigenvectors against U once gives a fixed permutation to relabel
        # eigh's raw column order by physical index -- a no-op for normal
        # ordering, required for inverted ordering.
        device = oscillation.mass_spectrum.DeltamSq21.device
        dtype = oscillation.mass_spectrum.DeltamSq21.dtype
        H_vacuum = hamiltonian_flavour(
            oscillation,
            torch.ones((), device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype),
            n_n_mol_cm3=None,
            evolution_scale_m=constant.R_SUN,
        )
        _, eigvec_vacuum = torch.linalg.eigh(H_vacuum)  # (N, N), ascending
        U = oscillation.pmns.pmns_matrix(antinu=oscillation.antinu)  # (N, N)
        # overlap[j, i] = |<eigh column j | vacuum mass state i>|^2
        overlap = (eigvec_vacuum.conj().transpose(-2, -1) @ U).abs() ** 2  # (N, N)
        inv_perm = torch.argsort(overlap.argmax(dim=-1))

        return weights_eigh_order[..., inv_perm]   # (..., N)

    # --- analytic SM path --------------------------------------------------
    th13m = th13_M(oscillation, E, ne, legacy_precision=legacy_precision)
    th12m = th12_M(oscillation, E, ne, legacy_precision=legacy_precision, th13m=th13m)

    c13m = torch.cos(th13m)
    s13m = torch.sin(th13m)
    c12m = torch.cos(th12m)
    s12m = torch.sin(th12m)

    c12m_sq = c12m ** 2
    s12m_sq = s12m ** 2

    if p_lz is not None:
        w1 = (1.0 - p_lz) * c12m_sq + p_lz * s12m_sq
        w2 = (1.0 - p_lz) * s12m_sq + p_lz * c12m_sq
    else:
        w1 = c12m_sq
        w2 = s12m_sq

    weights = torch.stack(
        [
            (c13m ** 2) * w1,
            (c13m ** 2) * w2,
            s13m ** 2,
        ],
        dim=-1,
    )
    return weights


def solar_probability_mass(
    oscillation: OscillationParameters,
    E: TensorLike,
    profile: object,
    sources: str | Sequence[str],
    *,
    method: str = "adiabatic",
    legacy_precision: bool = False,
    include_matter_nc: Optional[bool] = None,
    numerical_sampling: Optional[OdeMethod] = "midpoint",
) -> torch.Tensor:
    """Integrate solar production profiles into mass-basis probabilities.

    ``method="adiabatic"`` (the default)
    ------------------------------------------------------------------------
    Standard (SM) path -- ``oscillation.nsi`` is None and
    ``oscillation.pmns.n_flavours == 3``: if ``profile.use_LZ`` is True, a
    spatially resolved Landau-Zener correction is applied per (energy,
    production radius) pair, only to production radii above the resonance
    density (r_prod < r_res(E)). Diagonalization sub-path -- ``oscillation.nsi``
    is set, or ``oscillation.pmns.n_flavours == 4`` (3+1 sterile): the full
    Hamiltonian ``H = H_kin + H_mat`` is diagonalised at each (E, r) grid
    point (see ``Tei``); ``profile.use_LZ=True`` raises here (see ``Raises``)
    since the two-level Landau-Zener formula has no closed-form
    generalisation to off-diagonal NSI or a fourth level.

    ``method="numerical"``
    ------------------------------------------------------------------------
    Exact (no adiabatic assumption, no Landau-Zener approximation)
    alternative: propagates a coherent flavour state from each production
    radius out through the tabulated density profile with
    ``medium.solar.evolutor.solar_evolutor_numerical`` (reusing
    ``core.numerical``, which already supports NSI/sterile/NC generically),
    then projects onto the vacuum mass basis. See ``evolutor.py``'s module
    docstring for the efficiency and accuracy trade-offs relative to
    ``"adiabatic"``. ``profile.use_LZ=True`` raises here too: the coherent
    evolutor already captures every non-adiabatic transition, so there is
    nothing for Landau-Zener to correct.

    Args:
        oscillation: Oscillation parameters supplying theta12, theta13,
            DeltamSq21, DeltamSq3l, and the optional ``nsi`` (NSIConfig)
            attribute. A 4-flavour ``oscillation.pmns`` (3+1 sterile)
            selects the diagonalization sub-path of ``"adiabatic"``
            regardless of ``nsi``.
        E: Neutrino energy in MeV. Scalar or 1-D tensor in the standard
            pipeline; multi-dimensional E is supported for
            ``method="adiabatic"`` as long as ``profile.use_LZ`` is False.
        profile: SolarProfile-like object exposing radius, density,
            production_distribution(), mass_weights_integrate(), the
            optional ``use_LZ`` boolean flag, (when ``include_matter_nc=True``)
            ``density_n``, and (when ``method="numerical"``) the full
            ``radius``/``density`` grid.
        sources: Source key or ordered source keys available in ``profile``.
        method: ``"adiabatic"`` (default) or ``"numerical"`` (see above).
        legacy_precision: If True, evaluate the matter-mixing angles/potential
            with the legacy peanuts ``Vk``/prefactor for bit-comparable
            validation (see ``medium.solar.matter_mixing``). Ignored on the
            diagonalization sub-path of ``"adiabatic"``.
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
        ValueError: If ``method`` is not ``"adiabatic"`` or ``"numerical"``;
            if ``profile.use_LZ`` is True together with ``method="numerical"``;
            if ``profile.use_LZ`` is True together with NSI and/or the 3+1
            sterile extension; if ``profile.use_LZ`` is True with a
            multi-dimensional energy grid; or if ``include_matter_nc``
            resolves to True (explicitly or via auto-resolution) and the
            required neutron-density field is not set on ``profile``.
    """
    if method not in ("adiabatic", "numerical"):
        raise ValueError(
            f"method must be 'adiabatic' or 'numerical', got {method!r}."
        )

    fractions = profile.production_distribution(sources)
    radius_samples = profile.production_radius
    E_t = torch.as_tensor(E, device=radius_samples.device, dtype=radius_samples.dtype)

    n_flavours = int(oscillation.pmns.n_flavours)
    diagonalization_path = oscillation.nsi is not None or n_flavours == 4
    use_lz = getattr(profile, "use_LZ", False)

    if use_lz and method == "numerical":
        raise ValueError(
            "profile.use_LZ=True has no effect with method='numerical': the "
            "coherent evolutor already captures every non-adiabatic "
            "transition directly, so there is nothing for Landau-Zener to "
            "correct. Set profile.use_LZ=False before calling with "
            "method='numerical'."
        )
    if use_lz and diagonalization_path:
        raise ValueError(
            "profile.use_LZ=True is not supported together with NSI and/or "
            "the 3+1 sterile extension: the two-level Landau-Zener crossing "
            "formula has no closed-form generalisation to off-diagonal NSI "
            "couplings or a fourth level. Set profile.use_LZ=False, or use "
            "method='numerical' for an exact (non-adiabatic) treatment."
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

    n_n_for_tei: Optional[torch.Tensor] = None
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
        n_n_for_tei = profile.neutron_density(radius_samples)

    # Landau-Zener correction (analytic SM path only -- validated above to
    # exclude NSI/sterile and multi-dimensional E).
    p_lz_spatial: Optional[torch.Tensor] = None
    if use_lz:
        E_1d = E_t.reshape(-1) if E_t.ndim == 1 else E_t.reshape(1)  # (n_E,)
        r_res = resonance_radius(
            oscillation, E_1d, profile, legacy_precision=legacy_precision,
        )  # (n_E,) NaN if absent
        p_lz_e = plz(
            oscillation, E_1d, profile, legacy_precision=legacy_precision,
        )  # (n_E,)

        # above_res[e, r]: True where r_prod < r_res(E) (above resonance
        # density). NaN comparisons evaluate to False, so energies without a
        # resonance contribute a zero mask automatically.
        above_res = radius_samples[None, :] < r_res[:, None]          # (n_E, n_r)
        p_lz_2d = p_lz_e[:, None] * above_res.to(dtype=p_lz_e.dtype)  # (n_E, n_r)
        p_lz_spatial = p_lz_2d.squeeze(0) if E_t.ndim == 0 else p_lz_2d

    weights_r = Tei(
        oscillation,
        E_t[..., None],
        density,
        p_lz=p_lz_spatial,
        legacy_precision=legacy_precision,
        n_n_mol_cm3=n_n_for_tei,
    )

    return profile.mass_weights_integrate(weights_r, fractions, E_t.ndim)


def solar_probability_state(
    oscillation: OscillationParameters,
    E: TensorLike,
    profile: object,
    sources: str | Sequence[str],
    *,
    method: str = "adiabatic",
    legacy_precision: bool = False,
    include_matter_nc: Optional[bool] = None,
    numerical_sampling: Optional[OdeMethod] = "midpoint",
) -> torch.Tensor:
    """Compute solar flavour probabilities for one or more sources.

    This is the torch-native analogue of the legacy ``Psolar`` convention. It
    assumes electron neutrinos are produced in the solar medium and propagate
    adiabatically to vacuum mass eigenstates. The calculation has two steps:

        1. ``solar_probability_mass`` integrates each source production profile
           over the solar radius and builds incoherent mass-basis weights
           ``P_i(E)``, optionally including Landau-Zener corrections when
           ``profile.use_LZ`` is True (analytic SM path), or using
           diagonalisation when ``oscillation.nsi`` is set and/or
           ``oscillation.pmns`` is 4-flavour (3+1 sterile) (diagonalization
           sub-path).
        2. The mass weights are projected to final flavour probabilities with
           the vacuum PMNS probabilities,
           ``P_alpha(E) = sum_i |U_{alpha i}|^2 P_i(E)``, sized generically by
           ``oscillation.pmns.n_flavours`` (3 or 4) rather than assuming 3.

    The returned quantity is therefore a flavour probability vector, not a
    flux. For several sources, a leading source dimension is preserved.

    Args:
        oscillation: Built pmns object (3-flavour or 3+1 sterile) plus mass
            splittings, antinu selection, and the optional ``nsi``
            (NSIConfig) attribute.
        E: Neutrino energy in MeV.
        profile: SolarProfile-like object exposing radius, density,
            production_distribution(), the optional ``use_LZ`` boolean flag, and
            (when ``include_matter_nc=True``) ``density_n``.
        sources: Source key or ordered source keys available in ``profile``.
        method: ``"adiabatic"`` (default) or ``"numerical"`` (see
            ``solar_probability_mass``).
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation (see ``medium.solar.matter_mixing``).
            Ignored on the diagonalization sub-path and on
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
    method: str = "adiabatic",
    legacy_precision: bool = False,
    energy_dim: int = -2,
    include_matter_nc: Optional[bool] = None,
    numerical_sampling: Optional[OdeMethod] = "midpoint",
) -> torch.Tensor:
    """Average final solar flavour probabilities over energy.

    Builds the energy-resolved probabilities with ``solar_probability_state``
    and averages them with ``core.common.probability.probability_integrated``,
    weighted by an explicit production ``spectrum``.

    Args:
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
        E: Neutrino energy grid in MeV, one-dimensional, matching
            ``E_grid_MeV`` of ``probability_integrated``.
        profile: SolarProfile-like object exposing radius, density,
            production_distribution(), and the optional ``use_LZ`` boolean flag.
        sources: Source key or ordered source keys available in ``profile``.
        spectrum: Spectral weight w(E), required (no default).
        method: ``"adiabatic"`` (default) or ``"numerical"`` (see
            ``solar_probability_mass``).
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
