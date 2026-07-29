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

"""Pointwise adiabatic mass-eigenstate production weights.

Both functions return the probability that an electron neutrino produced at
a given (E, n_e) point projects onto each vacuum-mass-adjacent matter
eigenstate, under the solar adiabatic approximation: the neutrino is assumed
to remain in the local instantaneous matter eigenstate as it propagates
outward, so the production-point weight already equals the eventual vacuum
mass-eigenstate weight. Neither function propagates anything or builds an
evolution operator -- that is ``medium.solar.evolutor``'s job for the
non-adiabatic ``method="numerical"`` path.

``mass_weights_adiabatic_approximated`` and ``mass_weights_adiabatic_exact``
are two independent implementations of the same physical quantity, differing
in generality and in the approximation made:

    mass_weights_adiabatic_approximated
        Plain 3-flavour Standard Model only (no NSI, no 3+1 sterile
        extension). Evaluates the closed-form matter-mixing angles
        theta12^M/theta13^M (``medium.solar.matter_mixing``), a well-known
        but not exact reduction to two sequential 2-level problems (valid
        because DeltamSq21 << |DeltamSq3l|). Supports an optional local
        Landau-Zener correction at the 1-2 resonance (see
        ``medium.solar.landau_zener.landau_zener_spatial_correction``).

    mass_weights_adiabatic_exact
        General N-flavour path (plain SM, NSI, the 3+1 sterile extension, or
        NSI and sterile combined). Diagonalises the full flavour-basis
        Hamiltonian pointwise with ``torch.linalg.eigh`` -- exact given the
        adiabatic assumption, i.e. it makes no further approximation beyond
        adiabaticity itself. Has no Landau-Zener counterpart: the two-level
        crossing formula has no closed-form generalisation to off-diagonal
        NSI couplings or a fourth level.

Callers should not call these functions directly with mismatched physics
(NSI/sterile into the approximated path, or ``p_lz`` into the exact path --
the latter is not even accepted as an argument). All method/use_LZ/BSM
compatibility checks live in ``medium.solar.probability.
solar_probability_mass``, which is the intended entry point; these two
functions assume their caller has already validated the combination.
"""



from __future__ import annotations

from typing import Optional, Union

import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.hamiltonian import hamiltonian_flavour
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.solar.matter_mixing import th12_M, th13_M

TensorLike = Union[float, int, torch.Tensor]


def mass_weights_adiabatic_approximated(
    oscillation: OscillationParameters,
    E: TensorLike,
    ne: TensorLike,
    *,
    p_lz: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Analytic (closed-form) adiabatic mass-eigenstate weights, plain SM only.

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

    This function does not check whether ``oscillation`` carries NSI or a
    4-flavour ``pmns`` -- the caller (``solar_probability_mass``) is
    responsible for only calling it in the plain 3-flavour SM case, since
    ``matter_mixing.th12_M/th13_M`` are 3-flavour-only formulas.

    Args:
        oscillation: Oscillation parameters supplying theta12, theta13, and
            mass_spectrum.DeltamSq21/DeltamSq3l.
        E: Neutrino energy in MeV.
        ne: Electron density samples in mol/cm^3.
        p_lz: Optional Landau-Zener transition probability tensor,
            broadcastable with the internal cos^2(theta_12^M) /
            sin^2(theta_12^M) outputs (see ``medium.solar.landau_zener.
            landau_zener_spatial_correction``).
        legacy_precision: If True, evaluate the analytic matter-mixing angles
            with the legacy peanuts ``Vk`` prefactor for bit-comparable
            validation (see ``medium.solar.matter_mixing``).

    Returns:
        Real tensor of matter-production weights with final mass-index
        dimension 3, shape broadcast-compatible with ``(E, ne)``.
    """
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


def mass_weights_adiabatic_exact(
    oscillation: OscillationParameters,
    E: TensorLike,
    ne: TensorLike,
    *,
    n_n_mol_cm3: Optional[TensorLike] = None,
) -> torch.Tensor:
    """Numerical (pointwise-diagonalised) adiabatic mass-eigenstate weights.

    The total Hamiltonian ``H = H_kin + H_mat`` is assembled via
    ``tpeanuts.core.common.hamiltonian.hamiltonian_flavour`` -- which reads
    ``oscillation.nsi.epsilon`` when NSI is set, and generalises to N = 4
    automatically from ``oscillation.pmns``/``oscillation.mass_spectrum``
    when the 3+1 sterile extension is active (theta14, theta24, theta34,
    DeltamSq41), with the optional sterile neutral-current term when
    ``n_n_mol_cm3`` is supplied -- and diagonalised with
    ``torch.linalg.eigh``. Production weights are the squared
    electron-flavour components of each matter eigenstate:

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

    This path supports arbitrary off-diagonal NSI, the 3+1 sterile
    extension, or both simultaneously -- nothing here branches explicitly on
    which combination is active, since ``hamiltonian_flavour`` already does.
    It also works for the plain 3-flavour SM case (no NSI, N=3), giving
    numerically exact adiabatic weights rather than
    ``mass_weights_adiabatic_approximated``'s closed-form 2-level reduction.

    There is no ``p_lz``/Landau-Zener parameter: the two-level crossing
    formula has no closed-form generalisation to off-diagonal NSI couplings
    or a fourth level, and unlike ``mass_weights_adiabatic_approximated``
    this function is not restricted to a 2-level reducible problem in the
    first place.

    Args:
        oscillation: Oscillation parameters supplying theta12, theta13,
            DeltamSq21, DeltamSq3l, and the optional ``nsi`` (NSIConfig)
            attribute.
        E: Neutrino energy in MeV.
        ne: Electron density samples in mol/cm^3.
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
