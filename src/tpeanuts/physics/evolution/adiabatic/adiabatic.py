#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Pointwise adiabatic mass-eigenstate production weights: closed-form and diagonalised.

Returns the probability that an electron neutrino produced at a given
``(E, n_e)`` point projects onto each vacuum-mass-adjacent matter
eigenstate, under the adiabatic approximation: the neutrino is assumed
to remain in the local instantaneous matter eigenstate as it propagates
outward, so the production-point weight already equals the eventual
vacuum mass-eigenstate weight. Neither function propagates anything or
builds an evolution operator over a path -- see
``tpeanuts.physics.evolution.adiabatic``'s own package docstring for why
that makes this family structurally different from
``numerical``/``analytic``/``perturbative``.

``mass_weights_adiabatic_approximated`` and ``mass_weights_adiabatic_exact``
are two independent implementations of the same physical quantity,
differing in generality and in the approximation made:

    mass_weights_adiabatic_approximated
        Plain 3-flavour Standard Model only (no NSI, no sterile
        extension). Evaluates the closed-form matter-mixing angles
        theta12^M/theta13^M (``tpeanuts.physics.evolution.adiabatic.
        matter_mixing``), a well-known but not exact reduction to two
        sequential 2-level problems (valid because DeltamSq21 <<
        |DeltamSq3l|). Supports an optional local Landau-Zener
        correction at the 1-2 resonance (see
        ``tpeanuts.physics.evolution.adiabatic.landau_zener``).
    mass_weights_adiabatic_exact
        General N-flavour path (plain SM, NSI, a sterile extension, or
        both). Diagonalises the full flavour-basis Hamiltonian pointwise
        with ``torch.linalg.eigh`` -- exact given the adiabatic
        assumption. Has no Landau-Zener counterpart: the two-level
        crossing formula has no closed-form generalisation to
        off-diagonal NSI couplings or a fourth level.

``mass_weights_adiabatic_exact`` takes the flavour-basis generator ``H``
(and a vacuum-limit ``H_vacuum``) as caller-supplied arguments, rather
than a whole ``tpeanuts.physics.model.PropagationModel`` -- the same
"generator is a plain tensor argument" contract
``tpeanuts.physics.evolution.numerical.numerical.segment_evolutor_numerical``
already uses. Building ``H`` (which needs
``PropagationModel.generator_flavour`` -- private assembly logic that
already handles NSI/sterile/helicity-doubling correctly, and whose own
docstring documents a real double-counted-term bug an earlier
hand-written version had) is therefore ``mass_weights_adiabatic``'s job (below,
the model-aware entry point), not this pure function's.

``mass_weights_adiabatic_approximated`` still validates what is
knowable from a ``MixingMatrix`` alone (rejecting a sterile-/helicity
-extended space) so that calling it standalone -- bypassing
``mass_weights_adiabatic`` -- cannot silently return physically wrong weights;
``mass_weights_adiabatic`` performs the fuller check (also rejecting active
``extensions``, invisible at the ``MixingMatrix`` level) before
delegating here.

Module contents:
    mass_weights_adiabatic_approximated(...)
        Analytic (closed-form) adiabatic mass-eigenstate weights, plain
        SM only. Pure function, no ``PropagationModel``.
    mass_weights_adiabatic_exact(...)
        Numerical (pointwise-diagonalised) adiabatic mass-eigenstate
        weights, supporting NSI/sterile extensions. Pure function, no
        ``PropagationModel``.
    mass_weights_adiabatic(...)
        Model-aware entry point: takes a ``PropagationModel`` directly,
        builds whichever tensor(s) it needs (including the Landau-Zener
        correction, when requested), and delegates to whichever of the
        two functions above ``variant`` selects.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import torch

from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.common.terms import PropagationEnvironment
from tpeanuts.physics.evolution.adiabatic.landau_zener import landau_zener_spatial_correction
from tpeanuts.physics.evolution.adiabatic.matter_mixing import theta12_in_matter, theta13_in_matter
from tpeanuts.physics.model import PropagationModel
from tpeanuts.util.constant import R_SUN
from tpeanuts.util.math import eigh_chunked
from tpeanuts.util.tensor import TensorLike


def _reject_extended_space(mixing: MixingMatrix) -> None:
    """Raise if ``mixing``'s space is sterile- or helicity-extended.

    Raises:
        ValueError: If ``mixing.space.is_sterile_extended`` or
            ``mixing.space.is_helicity_extended``.
    """
    if mixing.space.is_sterile_extended:
        raise ValueError(
            "mass_weights_adiabatic_approximated is a plain 3-flavour "
            "Standard Model closed-form reduction -- it cannot represent "
            "a sterile-extended FlavourSpace. Use mass_weights_adiabatic_exact "
            "for a sterile-extended space."
        )
    if mixing.space.is_helicity_extended:
        raise ValueError(
            "mass_weights_adiabatic_approximated is a plain 3-flavour "
            "Standard Model closed-form reduction -- it cannot represent "
            "a helicity-extended FlavourSpace."
        )


def mass_weights_adiabatic_approximated(
    mixing: MixingMatrix,
    mass_spectrum: MassSpectrum,
    E_MeV: TensorLike,
    ne_mol_cm3: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
    p_lz: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Analytic (closed-form) adiabatic mass-eigenstate weights, plain SM only.

    The matter mixing angles theta_12^M and theta_13^M are evaluated
    analytically via ``matter_mixing.theta12_in_matter`` and ``matter_mixing.theta13_in_matter``.
    When ``p_lz`` is also provided, the nu_1^M / nu_2^M weights are mixed to
    account for the Landau-Zener transition at the MSW resonance:

        w_1 = (1 - P_LZ) cos^2(theta_12^M) + P_LZ sin^2(theta_12^M)
        w_2 = (1 - P_LZ) sin^2(theta_12^M) + P_LZ cos^2(theta_12^M)

    The nu_3^M weight (sin^2 theta_13^M) is unaffected.

    Args:
        mixing: Built MixingMatrix supplying theta12/theta13. Rejected if
            its space is sterile- or helicity-extended (see this
            module's own docstring for why NSI cannot be rejected here
            too).
        mass_spectrum: Built MassSpectrum supplying DeltamSq21/DeltamSq3l.
        E_MeV: Neutrino energy in MeV.
        ne_mol_cm3: Electron density samples in mol/cm^3.
        antinu: If True, flip the matter-potential sign for antineutrinos.
        p_lz: Optional Landau-Zener transition probability tensor,
            broadcastable with the internal cos^2(theta_12^M) /
            sin^2(theta_12^M) outputs -- see
            ``tpeanuts.physics.evolution.adiabatic.landau_zener``.

    Returns:
        Real tensor of matter-production weights with final mass-index
        dimension 3, shape broadcast-compatible with ``(E, ne)``.

    Raises:
        ValueError: If ``mixing.space`` is sterile- or helicity-extended.
    """
    _reject_extended_space(mixing)

    th13m = theta13_in_matter(mixing, mass_spectrum, E_MeV, ne_mol_cm3, antinu=antinu)
    th12m = theta12_in_matter(mixing, mass_spectrum, E_MeV, ne_mol_cm3, antinu=antinu, th13m=th13m)

    c13m = torch.cos(th13m)
    s13m = torch.sin(th13m)
    c12m = torch.cos(th12m)
    s12m = torch.sin(th12m)

    c12m_sq = c12m**2
    s12m_sq = s12m**2

    if p_lz is not None:
        w1 = (1.0 - p_lz) * c12m_sq + p_lz * s12m_sq
        w2 = (1.0 - p_lz) * s12m_sq + p_lz * c12m_sq
    else:
        w1 = c12m_sq
        w2 = s12m_sq

    return torch.stack(
        [
            (c13m**2) * w1,
            (c13m**2) * w2,
            s13m**2,
        ],
        dim=-1,
    )


def mass_weights_adiabatic_exact(
    H: torch.Tensor,
    H_vacuum: torch.Tensor,
    mixing: MixingMatrix,
    *,
    antinu: Union[bool, torch.Tensor] = False,
) -> torch.Tensor:
    """Numerical (pointwise-diagonalised) adiabatic mass-eigenstate weights.

    Diagonalises the caller-supplied flavour-basis generator ``H`` with
    ``torch.linalg.eigh`` -- ``H`` already includes whichever extensions
    were active when it was built (NSI, a sterile neutral-current term,
    ...) and generalises to any ``H.shape[-1]`` automatically. Production
    weights are the squared electron-flavour components of each matter
    eigenstate:

        w_i(E, r) = |<nu_e | nu_i^M(E, r)>|^2 = |eigvec[..., 0, i]|^2,  i = 1..N

    ``eigh`` returns columns sorted by ascending eigenvalue, which is
    *not* the same as the physical mass index ``i`` whenever the vacuum
    eigenvalue order and the ascending order disagree -- true for normal
    ordering, false for inverted ordering. The raw ``eigh`` output is
    therefore reindexed with a fixed (E/ne-independent) permutation,
    derived by diagonalising ``H_vacuum`` and matching its eigenvectors
    against ``mixing``'s own PMNS matrix columns, before being returned
    -- so index ``i`` always means the physical vacuum mass eigenstate
    ``nu_{i+1}`` regardless of ordering.

    This path supports arbitrary off-diagonal NSI, a sterile extension,
    or both simultaneously -- nothing here branches explicitly on which
    combination is active, since that is already baked into ``H`` by
    whoever built it. It also works for the plain 3-flavour SM case (``H``
    with no extensions), giving numerically exact adiabatic weights
    rather than ``mass_weights_adiabatic_approximated``'s closed-form
    2-level reduction.

    There is no ``p_lz``/Landau-Zener parameter: the two-level crossing
    formula has no closed-form generalisation to off-diagonal NSI
    couplings or a fourth level, and unlike
    ``mass_weights_adiabatic_approximated`` this function is not
    restricted to a 2-level reducible problem in the first place.

    Args:
        H: Flavour-basis generator at the query point, shaped
            ``(..., N, N)`` -- typically
            ``PropagationModel.generator_flavour(environment, ...)``.
        H_vacuum: The same generator at ``n_e = n_n = 0`` (any positive
            energy -- the vacuum kinetic term's eigenvectors do not
            depend on it), shaped ``(N, N)``, used only for the
            eigenvector-permutation-fixing trick described above.
        mixing: Built MixingMatrix supplying the PMNS matrix
            (``mixing.pmns_matrix(antinu=antinu)``) the permutation is
            matched against.
        antinu: Selects the antineutrino PMNS convention for the
            permutation-fixing step -- must match whatever convention
            ``H``/``H_vacuum`` were themselves built with.

    Returns:
        Real tensor of matter-production weights with final mass-index
        dimension ``H.shape[-1]``, shape broadcast-compatible with
        ``H``'s own batch shape.
    """
    # eigh returns columns as eigenvectors; row 0 = electron-flavour
    _, eigvec = eigh_chunked(H)  # (..., N, N)
    weights_eigh_order = eigvec[..., 0, :].abs() ** 2  # (..., N)

    # eigh sorts columns by ascending eigenvalue, not by physical mass
    # index -- the two agree for normal ordering but not for inverted
    # ordering. At n_e = n_n = 0 every matter/NSI/NC term vanishes
    # exactly (all proportional to density), leaving the pure vacuum
    # kinetic generator H_kin = U diag(k_1,...,k_N) U^dagger, whose
    # eigenvectors are therefore exactly U's columns (up to phase);
    # neither this correspondence nor eigh's ordering depends on E
    # (each k_i is a positive multiple of a mass-squared splitting) or
    # antinu (the kinetic term does not depend on it), so comparing
    # eigh's vacuum-limit eigenvectors against U once gives a fixed
    # permutation to relabel eigh's raw column order by physical index
    # -- a no-op for normal ordering, required for inverted ordering.
    _, eigvec_vacuum = eigh_chunked(H_vacuum)  # (N, N), ascending
    U = mixing.pmns_matrix(antinu=antinu)  # (N, N)
    # overlap[j, i] = |<eigh column j | vacuum mass state i>|^2
    overlap = (eigvec_vacuum.conj().transpose(-2, -1) @ U).abs() ** 2  # (N, N)
    inv_perm = torch.argsort(overlap.argmax(dim=-1))

    return weights_eigh_order[..., inv_perm]  # (..., N)


def mass_weights_adiabatic(
    model: PropagationModel,
    environment: PropagationEnvironment,
    *,
    variant: Literal["approximated", "exact"] = "approximated",
    p_lz: Optional[torch.Tensor] = None,
    medium_radius: Optional[torch.Tensor] = None,
    medium_density: Optional[torch.Tensor] = None,
    radius_samples: Optional[torch.Tensor] = None,
    evolution_scale_m: TensorLike = R_SUN,
) -> torch.Tensor:
    """Pointwise adiabatic mass-eigenstate production weights, model-aware entry point.

    Mirrors ``tpeanuts.physics.evolution.dispatch.evolve_path``'s role for
    the adiabatic family: unlike ``mass_weights_adiabatic_approximated``/
    ``_exact`` above (which only ever see bare tensors and know nothing
    about a model's own extensions or dissipators), this function performs
    the full model-level validation and, for ``variant="exact"``, builds
    the flavour-basis generator(s) via ``model.generator_flavour`` before
    delegating.

    ``variant="approximated"`` additionally rejects any active
    ``model.extensions`` -- invisible to ``mass_weights_adiabatic_approximated``
    itself, which can only see (and already rejects) a sterile-/
    helicity-extended ``mixing.space``, not an NSI/sterile-NC term active
    on the model.

    When ``variant="approximated"``, ``p_lz`` is None, and
    ``medium_radius``/``medium_density``/``radius_samples`` are all given,
    the Landau-Zener correction is computed internally via
    ``tpeanuts.physics.evolution.adiabatic.landau_zener.
    landau_zener_spatial_correction`` (using ``model.mixing``/
    ``model.mass_spectrum``/``model.antinu``) before being forwarded to
    ``mass_weights_adiabatic_approximated`` -- the caller (typically
    ``tpeanuts.pipeline.solar``) only ever needs to pass the medium's own
    ``radius``/``density`` arrays and the production radii being
    evaluated, never call ``landau_zener_spatial_correction`` itself.

    Args:
        model: Built PropagationModel (mixing, mass spectrum, antinu, and
            any active extensions/dissipators) -- the only source this
            function reads its generator assembly and validation from.
        environment: Neutrino/medium data at the query point(s) (``E_MeV``,
            ``n_e_mol_cm3``, optionally ``n_n_mol_cm3``).
        variant: ``"approximated"`` (default) uses the closed-form
            2-level reduction, plain SM only; ``"exact"`` diagonalises
            the full flavour-basis generator pointwise, supporting any
            active extension.
        p_lz: Optional Landau-Zener transition probability, forwarded to
            ``mass_weights_adiabatic_approximated`` directly. Takes
            precedence over ``medium_radius``/``medium_density``/
            ``radius_samples`` when given. Ignored when
            ``variant="exact"`` (see that function's own docstring for
            why there is no exact-path counterpart).
        medium_radius: Whole-medium radius grid, for locating the 1-2
            resonance -- forwarded to ``landau_zener_spatial_correction``
            when ``p_lz`` is not given directly. Only used together with
            ``medium_density``/``radius_samples``.
        medium_density: Whole-medium density grid, paired with
            ``medium_radius``.
        radius_samples: The specific production radii ``environment``'s
            own ``n_e_mol_cm3`` was evaluated at -- distinct from
            ``medium_radius`` (the whole star), needed so the Landau
            -Zener correction knows which points to evaluate the
            resonance-crossing formula at.
        evolution_scale_m: Positive length scale in metres used to
            non-dimensionalize ``generator_flavour`` when
            ``variant="exact"`` -- unused for ``variant="approximated"``
            (which needs no non-dimensionalization at all, and is also
            unused by the Landau-Zener correction, itself scale-free in
            units of ``R_SUN``). Defaults to the solar radius, since the
            adiabatic family's only current use is solar-neutrino
            propagation.

    Returns:
        Real tensor of matter-production weights with final mass-index
        dimension 3 (``variant="approximated"``) or ``model.space.dim``
        (``variant="exact"``).

    Raises:
        ValueError: If ``model.dissipators`` is non-empty (the adiabatic
            approximation has no open-system counterpart here), if
            ``variant="approximated"`` and ``model.extensions`` is
            non-empty or ``model.space`` is sterile-/helicity-extended,
            or if ``variant`` is neither ``"approximated"`` nor
            ``"exact"``.
    """
    if model.dissipators:
        raise ValueError(
            f"mass_weights_adiabatic has no open-system counterpart, but this "
            f"model has active dissipators ({model.dissipators!r}). Construct "
            "this model with dissipators=() to use mass_weights_adiabatic."
        )

    if variant == "approximated":
        if model.extensions:
            raise ValueError(
                "mass_weights_adiabatic(variant='approximated') is a plain "
                "3-flavour Standard Model closed-form reduction, but this "
                f"model has active extensions ({model.extensions!r}). Use "
                "variant='exact' instead."
            )
        if p_lz is None and medium_radius is not None and medium_density is not None and radius_samples is not None:
            # the pipeline gives the energy a trailing singleton axis that broadcasts against the production radii; the
            # correction takes the bare scalar or 1-D grid
            energy = environment.E_MeV
            E_grid = energy[..., 0] if energy.ndim >= 1 and energy.shape[-1] == 1 else energy
            p_lz = landau_zener_spatial_correction(
                model.mixing, model.mass_spectrum, E_grid,
                medium_radius, medium_density, radius_samples,
                antinu=model.antinu,
            )
        return mass_weights_adiabatic_approximated(
            model.mixing,
            model.mass_spectrum,
            environment.E_MeV,
            environment.n_e_mol_cm3,
            antinu=model.antinu,
            p_lz=p_lz,
        )

    if variant == "exact":
        H = model.generator_flavour(environment, evolution_scale_m=evolution_scale_m)
        context = model.context
        one = torch.ones((), device=context.device, dtype=context.dtype)
        zero = torch.zeros((), device=context.device, dtype=context.dtype)
        environment_vacuum = PropagationEnvironment(
            E_MeV=one,
            n_e_mol_cm3=zero,
            n_n_mol_cm3=zero if environment.n_n_mol_cm3 is not None else None,
        )
        H_vacuum = model.generator_flavour(environment_vacuum, evolution_scale_m=evolution_scale_m)
        return mass_weights_adiabatic_exact(H, H_vacuum, model.mixing, antinu=model.antinu)

    raise ValueError(f"variant must be 'approximated' or 'exact', got {variant!r}.")
