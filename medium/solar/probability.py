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
Solar neutrino probabilities in the adiabatic approximation.

This module computes the incoherent mass-eigenstate weights produced in the
solar interior and converts them into final flavour probabilities using the
vacuum PMNS projection. It works directly with tabulated solar radius,
electron-density, and production-fraction samples.

Landau-Zener (LZ) corrections
------------------------------
By default the calculation is purely adiabatic (P_LZ = 0 everywhere). When
``solar_profile.use_LZ`` is True, ``solar_probability_mass`` queries
``landau_zener.plz`` and ``landau_zener.resonance_radius`` to compute a
spatially resolved LZ transition probability that is passed to ``Tei``.

The LZ correction is applied only to production points located *above* the
resonance density (i.e. at solar radii smaller than r_res(E)), since only
those neutrinos cross the MSW resonance while propagating outward. Production
below the resonance is purely adiabatic and its weights are left unchanged.

NSI extension
-------------
When ``epsilon`` (a Hermitian 3×3 NSI coupling matrix) is supplied to
``Tei``, ``solar_probability_mass``, or ``psolar``, the analytic mixing-angle
path is replaced by a fully numerical one: the total Hamiltonian
``H = H_kin + H_mat^NSI`` is assembled via
``tpeanuts.core.BSM.hamiltonian.hamiltonian_flavour_bsm`` at each
(energy, density) grid point and diagonalised with ``torch.linalg.eigh``.
The squared modulus of the electron-flavour component of each matter
eigenstate gives the production weights directly:

    w_i(E, r) = |<ν_e | ν_i^M(E, r)>|²  =  |eigvec[..., 0, i]|²

This path supports arbitrary off-diagonal NSI and enables the study of
LMA-Dark and other NSI-degenerate scenarios. The LZ correction from
``landau_zener`` is skipped when NSI is active, since the LZ formula and
resonance condition both change in the presence of non-diagonal ε.

Module functions:
    Tei(...)
        Compute matter-basis production weights for one radius/density grid.
    solar_probability_mass(...)
        Integrate source production profiles into mass-basis probabilities.
    psolar(...)
        Convert source mass-basis probabilities into flavour probabilities.
"""



from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.probability import (
    probability_incoherent,
)
from tpeanuts.medium.solar.matter_mixing import th12_M, th13_M
from tpeanuts.util.type import cdtype_from_real


TensorLike = Union[float, int, torch.Tensor]


def Tei(
    oscillation: OscillationParameters,
    E: TensorLike,
    ne: TensorLike,
    *,
    p_lz: Optional[torch.Tensor] = None,
    epsilon: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Compute adiabatic mass-eigenstate production weights.

    Returns the probability that an electron neutrino produced in matter
    is projected onto each of the three effective mass eigenstates.

    Standard (SM) path  — ``epsilon`` is None
    ------------------------------------------
    The matter mixing angles theta_12^M and theta_13^M are evaluated
    analytically via ``matter_mixing.th12_M`` and ``matter_mixing.th13_M``.
    When ``p_lz`` is also provided, the nu_1^M / nu_2^M weights are mixed to
    account for the Landau-Zener transition at the MSW resonance:

        w_1 = (1 - P_LZ) cos^2(theta_12^M) + P_LZ sin^2(theta_12^M)
        w_2 = (1 - P_LZ) sin^2(theta_12^M) + P_LZ cos^2(theta_12^M)

    The nu_3^M weight (sin^2 theta_13^M) is unaffected because the theta_13
    resonance lies in the deep solar core and is negligible for standard
    solar sources.

    NSI path  — ``epsilon`` is not None
    -------------------------------------
    The total Hamiltonian ``H = H_kin + H_mat^NSI`` is assembled via
    ``tpeanuts.core.BSM.hamiltonian.hamiltonian_flavour_bsm`` and
    diagonalised with ``torch.linalg.eigh``. Production weights are the
    squared electron-flavour components of each matter eigenstate:

        w_i(E, r) = |<nu_e | nu_i^M(E, r)>|^2 = |eigvec[..., 0, i]|^2

    The ``p_lz`` argument is silently ignored in the NSI path because the
    Landau-Zener resonance condition and adiabaticity parameter change in
    the presence of off-diagonal epsilon.

    Args:
        oscillation: Oscillation parameters supplying theta12, theta13,
            DeltamSq21, and DeltamSq3l.
        E: Neutrino energy in MeV.
        ne: Electron density samples in mol/cm^3.
        p_lz: Optional Landau-Zener transition probability tensor,
            broadcastable with the internal cos^2(theta_12^M) /
            sin^2(theta_12^M) outputs. Ignored when ``epsilon`` is not None.
        epsilon: Optional 3×3 complex NSI coupling matrix (Hermitian). When
            supplied, activates the numerical diagonalisation path. Accepts
            any shape broadcastable over the ``(E, ne)`` batch dimensions as
            long as the final two dimensions are ``(3, 3)``.
        legacy_precision: If True, evaluate the analytic matter-mixing angles
            with the legacy peanuts ``Vk`` prefactor for bit-comparable
            validation (see ``medium.solar.matter_mixing``). Ignored in the
            NSI path.

    Returns:
        Real tensor of matter-production weights with final mass-index
        dimension 3, shape broadcast-compatible with ``(E, ne)``.
    """
    if epsilon is not None:
        from tpeanuts.core.BSM.hamiltonian import hamiltonian_flavour_bsm

        H = hamiltonian_flavour_bsm(
            oscillation,
            E,
            ne,
            epsilon=epsilon,
            evolution_scale_m=constant.R_SUN,
        )
        # eigh returns columns as eigenvectors; row 0 = electron-flavour
        _, eigvec = torch.linalg.eigh(H)          # (..., 3, 3)
        return eigvec[..., 0, :].abs() ** 2        # (..., 3)

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
    epsilon: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Integrate solar production profiles into mass-basis probabilities.

    Standard (SM) path  — ``epsilon`` is None
    ------------------------------------------
    If ``profile.use_LZ`` is True, a spatially resolved Landau-Zener
    correction is applied per (energy, production radius) pair: the LZ
    transition probability P_LZ(E) is multiplied by a boolean mask that is
    True only for production radii located above the resonance density (i.e.
    r_prod < r_res(E)), so that only those neutrinos effectively cross the
    MSW resonance.

    NSI path  — ``epsilon`` is not None
    -------------------------------------
    The full Hamiltonian ``H = H_kin + H_mat^NSI`` is diagonalised at each
    (E, r) grid point (see ``Tei``). The ``profile.use_LZ`` flag is ignored
    in this path.

    Args:
        oscillation: Oscillation parameters supplying theta12, theta13,
            DeltamSq21, and DeltamSq3l.
        E: Neutrino energy in MeV. Scalar or 1-D tensor in the standard
            pipeline; multi-dimensional E is supported for the adiabatic path
            but LZ corrections are applied only when E is 0-D or 1-D.
        profile: SolarProfile-like object exposing radius, density,
            source_fractions(), and the optional ``use_LZ`` boolean flag.
        sources: Source key or ordered source keys available in ``profile``.
        epsilon: Optional 3×3 complex NSI coupling matrix (Hermitian).
            When supplied, activates the numerical diagonalisation path in
            ``Tei`` and disables Landau-Zener corrections.
        legacy_precision: If True, evaluate ``Tei`` with the legacy peanuts
            ``Vk`` prefactor for bit-comparable validation (see
            ``medium.solar.matter_mixing``). Ignored in the NSI path.

    Returns:
        Normalized incoherent mass-basis probabilities with leading source
        dimensions, optional energy dimensions, and final mass-index dimension
        3.
    """
    fractions = profile.source_fractions(sources)
    radius_samples = profile.radius
    density = profile.density

    E_t = torch.as_tensor(E, device=radius_samples.device, dtype=radius_samples.dtype)

    # --- Landau-Zener correction (SM path only) ---------------------------
    # Skipped when NSI epsilon is active: the SM resonance condition and
    # adiabaticity parameter no longer apply with off-diagonal epsilon.
    use_lz = getattr(profile, "use_LZ", False)
    p_lz_spatial: Optional[torch.Tensor] = None

    if use_lz and epsilon is None and E_t.ndim <= 1:
        from tpeanuts.medium.solar.landau_zener import (
            plz as _plz,
            resonance_radius as _resonance_radius,
        )
        # Work with a guaranteed 1-D energy array
        E_1d = E_t.reshape(-1) if E_t.ndim == 1 else E_t.reshape(1)  # (n_E,)
        r_res = _resonance_radius(oscillation, E_1d, profile)          # (n_E,) NaN if absent
        p_lz_e = _plz(oscillation, E_1d, profile)                     # (n_E,)

        # above_res[e, r] = True if r_prod < r_res(E) (above resonance density)
        # NaN comparisons evaluate to False, so energies without a resonance
        # contribute a zero mask automatically.
        above_res = radius_samples[None, :] < r_res[:, None]          # (n_E, n_r)

        # Effective LZ probability per (energy, production radius).
        # For scalar E_t: shape (n_r,); for 1-D E_t: shape (n_E, n_r).
        # Either shape broadcasts correctly with Tei's c12m_sq output.
        p_lz_2d = p_lz_e[:, None] * above_res.to(dtype=p_lz_e.dtype)  # (n_E, n_r)
        p_lz_spatial = p_lz_2d.squeeze(0) if E_t.ndim == 0 else p_lz_2d

    # --- Adiabatic (+ optional LZ / NSI) weights --------------------------
    weights_r = Tei(
        oscillation,
        E_t[..., None],
        density,
        p_lz=p_lz_spatial,
        epsilon=epsilon,
        legacy_precision=legacy_precision,
    )

    source_shape = fractions.shape[:-1]
    energy_ndim = E_t.ndim

    fractions_lifted = fractions.reshape(
        *source_shape,
        *((1,) * energy_ndim),
        fractions.shape[-1],
    )

    weights_lifted = weights_r.reshape(
        *((1,) * len(source_shape)),
        *weights_r.shape,
    )

    weighted = weights_lifted * fractions_lifted[..., None]
    norm = torch.trapz(fractions, x=radius_samples, dim=-1)
    integral = torch.trapz(weighted, x=radius_samples, dim=-2)

    norm_lifted = norm.reshape(
        source_shape + (1,) * energy_ndim
    )

    return integral / torch.clamp(norm_lifted, min=torch.finfo(radius_samples.dtype).tiny)[..., None]


def psolar(
    oscillation: OscillationParameters,
    E: TensorLike,
    profile: object,
    sources: str | Sequence[str],
    *,
    epsilon: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Compute adiabatic solar flavour probabilities for one or more sources.

    This is the torch-native analogue of the legacy ``Psolar`` convention. It
    assumes electron neutrinos are produced in the solar medium and propagate
    adiabatically to vacuum mass eigenstates. The calculation has two steps:

        1. ``solar_probability_mass`` integrates each source production profile
           over the solar radius and builds incoherent mass-basis weights
           ``P_i(E)``, optionally including Landau-Zener corrections when
           ``profile.use_LZ`` is True (SM path), or using numerical
           diagonalisation when ``epsilon`` is supplied (NSI path).
        2. The mass weights are projected to final flavour probabilities with
           the vacuum PMNS probabilities,
           ``P_alpha(E) = sum_i |U_{alpha i}|^2 P_i(E)``.

    The returned quantity is therefore a flavour probability vector, not a
    flux. For several sources, a leading source dimension is preserved.

    Args:
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E: Neutrino energy in MeV.
        profile: SolarProfile-like object exposing radius, density,
            source_fractions(), and the optional ``use_LZ`` boolean flag.
        sources: Source key or ordered source keys available in ``profile``.
        epsilon: Optional 3×3 complex NSI coupling matrix (Hermitian).
            When supplied, activates the numerical diagonalisation path (see
            ``solar_probability_mass`` and ``Tei``).
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation (see ``medium.solar.matter_mixing``).
            Ignored in the NSI path.

    Returns:
        Final flavour probabilities with leading source dimensions, optional
        energy dimensions, and final flavour dimension 3.
    """
    weights = solar_probability_mass(
        oscillation,
        E,
        profile,
        sources,
        epsilon=epsilon,
        legacy_precision=legacy_precision,
    )

    identity = torch.eye(
        3,
        device=profile.radius.device,
        dtype=cdtype_from_real(weights.dtype),
    )

    return probability_incoherent(
        identity,
        weights,
        pmns=oscillation.pmns,
        antinu=oscillation.antinu,
        real_dtype=weights.dtype,
    )
