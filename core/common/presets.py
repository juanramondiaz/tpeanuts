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
Single source of truth for every named-preset registry in the project:
registry mechanics (register/get/list) AND the preset data itself
(3-flavour SM and 3+1 sterile oscillation parameters, NSI couplings).


CP phase counting for 3+1 Dirac neutrinos
------------------------------------------
An N-generation Dirac mixing matrix contains (N-1)(N-2)/2 independent CP
phases.  For N = 4 this gives exactly **three** Dirac phases:

    delta_13  (standard SM phase, carried by R_13 via Delta)
    delta_14  (e-s sector, carried by R_14)
    delta_24  (mu-s sector, carried by R_24)

R_34 is always a **real** rotation: the fourth phase can be absorbed into the
charged-lepton or neutrino fields.  Sterile presets therefore carry no
``delta34_deg`` entry; ``OscillationParameters.from_preset`` always builds
``PMNS_sterile`` with delta34 = 0.

Module contents
---------------
register_preset(registry, name, **kwargs)
    Add a new preset to ``registry``, raising if ``name`` is already used.
get_preset(registry, name, *, kind="preset")
    Look up a preset by name, raising ``ValueError`` with the sorted
    available names if unknown.
list_presets(registry)
    Sorted list of all names currently in ``registry``.

OSCILLATION_PRESETS
    Registry of 3-flavour SM and 3+1 sterile oscillation parameter sets
    (degrees/eV^2). A preset is a sterile preset iff it carries a
    ``theta14_deg`` key; ``OscillationParameters.from_preset`` dispatches on
    that to build either a plain ``PMNS_SM`` or a ``PMNS_sterile``.

NSI_PRESETS
    Registry of Non-Standard Interaction coupling sets, consumed by
    ``tpeanuts.core.BSM.NSIConfig.NSIConfig.from_preset``.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Generic registry mechanics
# ---------------------------------------------------------------------------

def register_preset(
    registry: dict[str, dict],
    name: str,
    **kwargs: float | str,
) -> None:
    """Register a new preset in ``registry``.

    Args:
        registry: The ``dict[str, dict]`` registry to mutate.
        name: Unique string identifier for the preset.
        **kwargs: Preset data, stored verbatim as a dict under ``name``.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    if name in registry:
        raise ValueError(
            f'Preset "{name}" is already registered. '
            "Use a different name or call registry.pop(name) first."
        )
    registry[name] = dict(kwargs)


def get_preset(
    registry: dict[str, dict],
    name: str,
    *,
    kind: str = "preset",
) -> dict:
    """Return the preset data dict for ``name``.

    Args:
        registry: The ``dict[str, dict]`` registry to read.
        name: Preset identifier. Call ``list_presets(registry)`` for all
            available names.
        kind: Human-readable noun used in the error message (e.g. "sterile
            preset", "NSI preset").

    Returns:
        The preset data dict stored under ``name``.

    Raises:
        ValueError: If ``name`` is not in ``registry``.
    """
    if name not in registry:
        available = ", ".join(f'"{k}"' for k in sorted(registry))
        raise ValueError(
            f'Unknown {kind} "{name}". Available {kind}s: {available}'
        )
    return registry[name]


def list_presets(registry: dict[str, dict]) -> list[str]:
    """Return the names of all presets in ``registry`` (sorted).

    Args:
        registry: The ``dict[str, dict]`` registry to read.

    Returns:
        Sorted list of preset name strings.
    """
    return sorted(registry.keys())


# ---------------------------------------------------------------------------
# Oscillation presets (3-flavour SM and 3+1 sterile)
# ---------------------------------------------------------------------------

# SM baseline shared by every oscillation preset below.
# Reference: http://www.nu-fit.org, v5.2 (2022), Table 1, with SK data.
_SM_NUFIT52_NO: dict = dict(
    theta12_deg=33.41,
    theta13_deg=8.58,
    theta23_deg=49.0,
    delta13_deg=197.0,
    DeltamSq21=7.41e-5,
    DeltamSq3l=+2.511e-3,   # positive -> Delta m^2_31, normal ordering
    ordering="NO",
)

OSCILLATION_PRESETS: dict[str, dict] = {}

# 1. Plain 3-flavour SM best fit (no sterile fields -> builds PMNS_SM).
register_preset(
    OSCILLATION_PRESETS,
    "_SM_NUFIT52_NO",
    **_SM_NUFIT52_NO,
    label="_SM_NUFIT52_NO",
    description=(
        "Standard 3-flavor SM best fit. NuFIT 5.2 (2022), normal ordering, "
        "with SK atmospheric data. http://www.nu-fit.org."
    ),
)

# 1b. LMA-Dark angular parameterization.
# Reference: Esteban, Gonzalez-Garcia, Maltoni, Martinez-Soler, Schwetz
#   (2018), JHEP 08:180, arXiv:1805.04530, Sections 4-5.
#
# The LMA solar solution admits a discrete degeneracy: replacing
#   theta12 -> pi/2 - theta12  (56.59 deg in the second octant)
# combined with
#   eps_ee -> -2.0  (NSI that flips the sign of the MSW potential)
# leaves the solar neutrino survival probability invariant at all energies.
#
# This preset carries only the angular rotation; the NSI counterpart is
# registered separately as "nsi_lma_dark_esteban2018" in NSI_PRESETS.
#
# Key degeneracy relation:
#   P_ee^adiabatic(LMA-Dark) = cos^2(theta12_dark) = sin^2(theta12_LMA) ~ 0.303
#   P_ee^vacuum (LMA-Dark)   = 1 - sin^2(2*theta12)/2  (same as LMA, sin^2 is symmetric)
_theta12_dark_deg: float = 90.0 - _SM_NUFIT52_NO["theta12_deg"]   # = 56.59 deg
register_preset(
    OSCILLATION_PRESETS,
    "_LMA_DARK_NUFIT52_NO",
    **{**_SM_NUFIT52_NO, "theta12_deg": _theta12_dark_deg},
    label="_LMA_DARK_NUFIT52_NO",
    description=(
        "LMA-Dark angular parameterization. NuFIT 5.2 (2022) normal ordering "
        "with theta12 reflected into the second octant: "
        "theta12_dark = 90 - 33.41 = 56.59 deg (sin^2 theta12 = 0.697). "
        "All other parameters identical to _SM_NUFIT52_NO. "
        "Use together with nsi_lma_dark_esteban2018 (eps_ee=-2.0) for the full "
        "degenerate LMA-Dark solution (Esteban et al. 2018, arXiv:1805.04530). "
        "Degeneracy: cos^2(theta12_dark) = sin^2(theta12_LMA) = 0.303, "
        "i.e. the adiabatic solar P_ee is identical for both solutions."
    ),
)

# 1c. Updated SM baseline: NuFIT 6.0 (2024).
# Reference: Esteban, Gonzalez-Garcia, Maltoni, Martinez-Soler, Pinheiro,
#   Schwetz, "NuFIT-6.0: Updated global analysis of three-flavor neutrino
#   oscillations", JHEP 12 (2024) 216, arXiv:2410.05380.
# Table "IC24 with SK atmospheric data", Normal Ordering (best fit).
# Values read directly from the official parameter-range table,
# https://www.nu-fit.org/?q=node/294 (v6.0.tbl-parameters.pdf).
_SM_NUFIT60_NO: dict = dict(
    theta12_deg=33.68,
    theta13_deg=8.56,
    theta23_deg=43.3,
    delta13_deg=212.0,
    DeltamSq21=7.49e-5,
    DeltamSq3l=+2.513e-3,   # positive -> Delta m^2_31, normal ordering
    ordering="NO",
)
register_preset(
    OSCILLATION_PRESETS,
    "_SM_NUFIT60_NO",
    **_SM_NUFIT60_NO,
    label="_SM_NUFIT60_NO",
    description=(
        "Standard 3-flavor SM best fit. NuFIT 6.0 (2024), normal ordering, "
        "with SK atmospheric data. Esteban et al., JHEP 12 (2024) 216, "
        "arXiv:2410.05380. http://www.nu-fit.org."
    ),
)

# 1d. Updated SM baseline: NuFIT 6.1 (2025), the most recent release at the
#     time of writing -- first global fit to include JUNO's initial
#     theta12/Delta m^2_21 measurement.
# Reference: same base analysis as NuFIT 6.0 (Esteban et al., JHEP 12 (2024)
#   216, arXiv:2410.05380), updated per "NuFIT 6.1 (2025), www.nu-fit.org"
#   (the citation instruction given on the NuFIT results page itself; no
#   separate arXiv preprint is published for the v6.1 website update).
# Table "IC24 with SK atmospheric data", Normal Ordering (best fit).
# Values read directly from the official parameter-range table,
# https://www.nu-fit.org/?q=node/309 (v6.1.tbl-parameters.pdf).
_SM_NUFIT61_NO: dict = dict(
    theta12_deg=33.76,
    theta13_deg=8.62,
    theta23_deg=43.29,
    delta13_deg=212.0,
    DeltamSq21=7.537e-5,
    DeltamSq3l=+2.511e-3,   # positive -> Delta m^2_31, normal ordering
    ordering="NO",
)
register_preset(
    OSCILLATION_PRESETS,
    "_SM_NUFIT61_NO",
    **_SM_NUFIT61_NO,
    label="_SM_NUFIT61_NO",
    description=(
        "Standard 3-flavor SM best fit. NuFIT 6.1 (2025), normal ordering, "
        "with SK atmospheric data -- first NuFIT release including JUNO's "
        "initial theta12/Delta m^2_21 measurement. Base analysis: Esteban "
        "et al., JHEP 12 (2024) 216, arXiv:2410.05380; see NuFIT 6.1 (2025), "
        "http://www.nu-fit.org for the updated tables."
    ),
)

# 2. Standard 3-flavor SM, run through the 4-flavour sterile machinery with
#    all sterile mixing angles zero (has theta14_deg -> builds PMNS_sterile).
register_preset(
    OSCILLATION_PRESETS,
    "standard_3nu",
    **_SM_NUFIT52_NO,
    theta14_deg=0.0,
    theta24_deg=0.0,
    theta34_deg=0.0,
    delta14_deg=0.0,
    delta24_deg=0.0,
    DeltamSq41=1.0,   # arbitrary; no physical effect when all sterile angles = 0
    label="standard_3nu",
    description=(
        "Standard 3-flavor SM best fit. NuFIT 5.2 (2022), normal ordering, "
        "with SK atmospheric data. No sterile mixing."
    ),
)

# 3. 3+1 framework with null sterile mixing (SM limit inside 4-flavor code).
register_preset(
    OSCILLATION_PRESETS,
    "sterile_3p1_null_mixing",
    **_SM_NUFIT52_NO,
    theta14_deg=0.0,
    theta24_deg=0.0,
    theta34_deg=0.0,
    delta14_deg=0.0,
    delta24_deg=0.0,
    DeltamSq41=1.0,
    label="sterile_3p1_null_mixing",
    description=(
        "3+1 scenario with all active-sterile mixing angles set to zero. "
        "Physically equivalent to standard_3nu but run through PMNS_sterile. "
        "Use as null-hypothesis reference in sterile analyses."
    ),
)

# 4. Global 3+1 best fit from Giunti, Marrone & Palazzo (2017).
# Reference: arXiv:1612.01087, Table 3 (combined global fit).
# Sterile CP phases marginalized over (effectively 0 in the published fit).
# sin^2(2 theta14) ~ 0.085 -> theta14 ~ 8.5 deg
# sin^2(2 theta24) ~ 0.068 -> theta24 ~ 7.5 deg
# Delta m^2_41 ~ 1.7 eV^2 (reactor + gallium + solar best fit)
# SM phase delta_13 = 197 deg from NuFIT 5.2 (marginalized in the original analysis).
register_preset(
    OSCILLATION_PRESETS,
    "sterile_3p1_bestfit_giunti2017",
    **_SM_NUFIT52_NO,
    theta14_deg=8.5,
    theta24_deg=7.5,
    theta34_deg=0.0,
    delta14_deg=0.0,
    delta24_deg=0.0,
    DeltamSq41=1.7,
    label="sterile_3p1_bestfit_giunti2017",
    description=(
        "Global 3+1 best fit (Giunti, Marrone, Palazzo 2017, arXiv:1612.01087). "
        "Combines reactor anomaly, gallium anomaly, and solar data. "
        "sin^2(2 theta14)=0.085, sin^2(2 theta24)=0.068, Delta m^2_41=1.7 eV^2. "
        "Sterile CP phases assumed zero (no sensitivity in reactor/gallium data). "
        "SM delta_13 from NuFIT 5.2."
    ),
)

# 5. IceCube 2020 atmospheric nu_mu -> nu_s disappearance benchmark.
# Reference: IceCube Collaboration, arXiv:2005.12943 (PRL 125, 141801).
# Benchmark point: sin^2(2 theta_24) = 0.10, Delta m^2_41 = 0.3 eV^2.
# Conversion: sin^2(2 theta_24) = 0.10  ->  theta_24 = 1/2 arcsin(sqrt(0.10)) ~ 9.22 deg
_theta24_ic: float = 0.5 * math.degrees(math.asin(math.sqrt(0.10)))   # ~ 9.217 deg
register_preset(
    OSCILLATION_PRESETS,
    "sterile_3p1_benchmark_icecube",
    **_SM_NUFIT52_NO,
    theta14_deg=0.0,
    theta24_deg=round(_theta24_ic, 4),
    theta34_deg=0.0,
    delta14_deg=0.0,
    delta24_deg=0.0,
    DeltamSq41=0.3,
    label="sterile_3p1_benchmark_icecube",
    description=(
        "Benchmark inspired by IceCube 2020 nu_mu disappearance search "
        "(arXiv:2005.12943). sin^2(2 theta24)=0.10 -> theta24~9.22 deg, "
        "Delta m^2_41=0.3 eV^2. theta14 and theta34 set to zero (IceCube is "
        "primarily sensitive to theta24)."
    ),
)


# ---------------------------------------------------------------------------
# NSI presets
# ---------------------------------------------------------------------------
#
# epsilon matrix structure and bounds (Biggio, Blennow, Fernandez-Martinez
# 2009, arXiv:0907.0097):
#   |eps_ee|  < 0.50    |eps_emu|  < 0.033
#   |eps_mumu| < 0.078   |eps_etau|  < 0.27
#   |eps_tautau| < 0.33    |eps_mutau|  < 0.040
# See ``tpeanuts.core.BSM.NSIConfig`` for the full physics background and the
# ``NSIConfig.epsilon_tensor()`` builder consuming these presets.

NSI_PRESETS: dict[str, dict] = {}

# 1. SM limit - all eps = 0.
register_preset(
    NSI_PRESETS,
    "sm_no_nsi",
    label="sm_no_nsi",
    description=(
        "Standard Model: all NSI parameters set to zero. "
        "Passing this epsilon to hamiltonian_reduced_bsm is equivalent to "
        "the default epsilon=None (SM MSW matter potential)."
    ),
)

# 2. Diagonal-only NSI - representative allowed values.
# Reference: Biggio, Blennow, Fernandez-Martinez (2009), JHEP 08:090,
#   arXiv:0907.0097, Table 1.
# Model-independent 90% CL upper bounds from CC universality and NC data:
#   |eps_ee| < 0.50,  |eps_mumu| < 0.078,  |eps_tautau| < 0.33.
# Chosen values are within bounds but sizable; eps_mumu = 0 (WLOG convention).
register_preset(
    NSI_PRESETS,
    "nsi_diagonal_biggio2009",
    eps_ee=0.30,
    eps_mumu=0.0,    # set to 0 without loss of generality
    eps_tautau=0.15,
    label="nsi_diagonal_biggio2009",
    description=(
        "Diagonal-only NSI at representative allowed values. "
        "Within the model-independent 90% CL bounds of "
        "Biggio, Blennow, Fernandez-Martinez (2009), arXiv:0907.0097. "
        "eps_ee=0.30, eps_tautau=0.15, eps_mumu=0 (WLOG convention). "
        "All off-diagonal entries zero."
    ),
)

# 3. LMA-Dark degenerate solution.
# Reference: Esteban, Gonzalez-Garcia, Maltoni, Martinez-Soler, Schwetz
#   (2018), JHEP 08:180, arXiv:1805.04530, Sections 4-5.
#
# The standard LMA solar solution admits a discrete degeneracy: replacing
#   theta12 -> pi/2 - theta12  (~ 56.6 deg)
# combined with
#   eps_ee -> -(2 + eps_ee)  ~ -2.0
# produces an equivalent fit to all solar and reactor data. This is the
# LMA-Dark solution. The large negative eps_ee effectively flips the sign
# of the MSW potential seen by nu_e, compensating the change in theta12.
# Value eps_ee = -2.0 corresponds to the simplest LMA-D benchmark with all
# other eps = 0; the global fit allows additional small eps_etau that help
# fit atmospheric data (not included here for clarity).
register_preset(
    NSI_PRESETS,
    "nsi_lma_dark_esteban2018",
    eps_ee=-2.0,
    label="nsi_lma_dark_esteban2018",
    description=(
        "LMA-Dark degenerate solution. eps_ee=-2.0 effectively flips the "
        "sign of the MSW potential, mimicking theta12 -> pi/2 - theta12 ~ "
        "56.6 deg. Esteban, Gonzalez-Garcia, Maltoni, Martinez-Soler, "
        "Schwetz (2018), arXiv:1805.04530. All other eps = 0. Combine with "
        "theta12 -> 56.6 deg for the full degenerate solution."
    ),
)

# 4. Off-diagonal NSI in the mu-tau sector - IceCube DeepCore benchmark.
# Reference: IceCube Collaboration (2022), Phys. Rev. D 106, 032009,
#   arXiv:2112.09122.
#
# IceCube DeepCore is primarily sensitive to eps_mutau and (eps_tautau -
# eps_mumu) through their effect on atmospheric nu_mu survival. The 90% CL
# bounds from the DeepCore analysis are approximately:
#   |eps_mutau|        < 0.0060
#   |eps_tautau - eps_mumu| < 0.018
# This preset sits near the edge of the allowed region:
#   eps_mutau  = 0.005 (real, Im = 0)
#   eps_tautau = 0.015 (with eps_mumu = 0)
register_preset(
    NSI_PRESETS,
    "nsi_offdiag_icecube2022",
    eps_tautau=0.015,
    eps_mutau_re=0.005,
    label="nsi_offdiag_icecube2022",
    description=(
        "Off-diagonal NSI in the mu-tau sector near the IceCube DeepCore "
        "90% CL boundary. eps_mutau=0.005 (real), eps_tautau=0.015 "
        "(with eps_mumu=0). Dominant NSI for atmospheric neutrinos. "
        "IceCube Collaboration (2022), arXiv:2112.09122: "
        "|eps_mutau| < 0.0060, |eps_tautau - eps_mumu| < 0.018."
    ),
)

# 5. DUNE flavour-changing propagation benchmark.
# Reference:
#   DUNE Collaboration, Abi et al. (2020),
#   "Prospects for beyond the Standard Model physics searches at the DUNE
#   Near Detector", Eur. Phys. J. C 81, 322 (2021),
#   arXiv:2008.12769.
#
# A single flavour-changing NSI parameter eps_e_tau is introduced while
# all remaining NSI couplings are set to zero. Values around |eps_e_tau|
# = 0.1-0.2 are commonly adopted in DUNE sensitivity studies because this
# parameter generates the strongest degeneracies with delta_CP, the mass
# ordering and the determination of theta23. The value eps_e_tau = 0.15
# represents a realistic benchmark inside the region typically explored
# by DUNE simulations, although it is not a global best-fit.
register_preset(
    NSI_PRESETS,
    "nsi_dune_etau",
    eps_etau_re=0.15,
    eps_etau_im=0.0,
    label="nsi_dune_etau",
    description=(
        "Representative DUNE propagation NSI benchmark with "
        "eps_e_tau=0.15. Widely used to study CP-violation and mass-"
        "ordering degeneracies. Inspired by DUNE Collaboration "
        "sensitivity analyses (arXiv:2008.12769)."
    ),
)

# 6. Hyper-Kamiokande propagation NSI benchmark.
# Reference:
#   Hyper-Kamiokande Proto-Collaboration (Abe et al.),
#   Prog. Theor. Exp. Phys. 2015, 053C02,
#   arXiv:1502.05199.
#
# This benchmark combines a moderate diagonal matter correction with a
# flavour-changing interaction. Hyper-K sensitivity studies frequently
# investigate this combination because eps_ee modifies the MSW potential
# while eps_e_tau introduces additional interference with the standard
# oscillation parameters. The values below are representative benchmark
# values rather than a preferred fit.
register_preset(
    NSI_PRESETS,
    "nsi_hyperk_etau",
    eps_ee=0.20,
    eps_etau_re=0.10,
    eps_etau_im=0.0,
    label="nsi_hyperk_etau",
    description=(
        "Representative Hyper-K propagation NSI benchmark with "
        "eps_ee=0.20 and eps_e_tau=0.10. Illustrates the combined "
        "effect of diagonal and flavour-changing matter interactions. "
        "Inspired by Hyper-K NSI sensitivity studies."
    ),
)

# 7. Global-fit inspired propagation NSI benchmark.
# Reference:
#   Esteban, Gonzalez-Garcia, Maltoni, Martinez-Soler, Schwetz
#   (2018), JHEP 08:180,
#   arXiv:1805.04530.
#
# Recent global oscillation analyses including propagation NSI constrain
# the allowed multidimensional epsilon parameter space. The values below
# are representative of the currently allowed region and provide a realistic
# benchmark for numerical simulations. They should not be interpreted as a
# unique global best-fit, since significant parameter correlations and
# degeneracies remain.
register_preset(
    NSI_PRESETS,
    "nsi_globalfit_esteban2018",
    eps_ee=0.25,
    eps_etau_re=0.12,
    eps_etau_im=0.0,
    eps_mutau_re=0.01,
    eps_tautau=0.08,
    label="nsi_globalfit_esteban2018",
    description=(
        "Propagation NSI benchmark inspired by the global analysis of "
        "Esteban et al. (2018), arXiv:1805.04530. Representative values "
        "within the allowed parameter space for realistic oscillation "
        "simulations."
    ),
)

# 8. Diagonal matter-potential benchmark.
# Reference:
#   Biggio, Blennow, Fernandez-Martinez (2009),
#   JHEP 08:090,
#   arXiv:0907.0097.
#
# Only the diagonal coupling eps_ee is allowed to differ from zero.
# This benchmark isolates the modification of the effective MSW matter
# potential without introducing flavour-changing interactions. It is
# frequently used as the simplest NSI scenario for comparison with the
# Standard Model.
register_preset(
    NSI_PRESETS,
    "nsi_ee_only",
    eps_ee=0.30,
    label="nsi_ee_only",
    description=(
        "Single-parameter propagation NSI benchmark with "
        "eps_ee=0.30. Modifies only the effective MSW matter potential "
        "while all flavour-changing NSI couplings remain zero. "
        "Representative value within the Biggio et al. bounds."
    ),
)

# 9. Atmospheric mu-tau propagation benchmark.
# Reference:
#   IceCube Collaboration,
#   Coloma et al.,
#   Phys. Rev. D 96, 115007 (2017),
#   arXiv:1708.02899.
#
# The flavour-changing parameter eps_mu_tau primarily affects atmospheric
# neutrinos traversing the Earth. High-energy atmospheric experiments such
# as IceCube and KM3NeT are particularly sensitive to this coupling because
# it modifies nu_mu <-> nu_tau propagation through the Earth's matter.
# The value eps_mu_tau = 0.01 represents a typical benchmark used in
# sensitivity studies.
register_preset(
    NSI_PRESETS,
    "nsi_mutau_only",
    eps_mutau_re=0.01,
    label="nsi_mutau_only",
    description=(
        "Atmospheric propagation NSI benchmark with eps_mu_tau=0.01. "
        "Representative of IceCube and KM3NeT sensitivity studies of "
        "flavour-changing mu-tau matter interactions."
    ),
)