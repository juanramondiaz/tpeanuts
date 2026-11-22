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

"""Independent NumPy reference physics shared by every external validation: mixing, vacuum, segments, adiabatic weights.

Everything an external comparison needs as its *independent* reference, written in plain NumPy and importing neither
``tpeanuts`` nor any external solver, so that a defect of tpeanuts cannot reach both sides of a comparison and
the module can be imported from the environment of any package that is validated (the original PEANUTS runs without
torch). The notebooks of ``notebooks/3_validation/2_external`` used to carry their own copy of each function;
there is one copy here.

Conventions (the same in every function):

- Energies are in GeV unless the name says MeV, baselines and segment lengths in km or m as the argument says,
  mass-squared differences in eV^2, densities in mol/cm^3 (the electron density ``N_e = rho Y_e`` and the neutron
  density ``N_n = rho (1 - Y_e)`` with ``rho`` in g/cm^3 and one gram per mole, as nuSQuIDS does).
- ``mixing`` is the matrix ``U`` of the neutrino; for the antineutrino it is conjugated here and the potentials change
  sign, so callers always pass the neutrino matrix and ``antinu``.
- Probabilities are indexed ``[initial, final]``; ``mass_to_flavour`` is indexed ``[final, mass]``.
- With a sterile neutrino the neutral-current potential of the sterile state relative to the active ones is
  ``+V_CC N_n / 2`` (verified against nuSQuIDS in ``06_sterile_3p1``).

The constants are written out, not imported from tpeanuts, to keep the independence; ``tests/validation/test_common.py``
asserts that they equal those of ``tpeanuts.util.constant``.

Module contents:
    HBARC_EV_M, V_CC_EV, R_EARTH_M
        Constants of the reference: hbar c in eV m, the charged-current potential per mol/cm^3 in eV, and the Earth
        radius in m.
    rotation(...)
        Complex rotation in the (i, j) plane with the phase on the upper off-diagonal element.
    mixing_matrix(...)
        The 3x3 PMNS matrix or the 4x4 matrix ``R34 R24 R14 R23 R13 R12`` of nuSQuIDS, from a configuration object.
    mass_squared(...)
        The vector ``(0, dm21, dm3l[, dm41])`` of a configuration object.
    vacuum_probability(...)
        Probabilities after a vacuum baseline, indexed ``[initial, final]``.
    evolution_operator(...)
        Operator of a list of segments of constant density, by diagonalising the Hamiltonian of each.
    probabilities(...), mass_to_flavour(...)
        Probabilities of an operator, and probabilities from each vacuum mass eigenstate.
    adiabatic_weights(...)
        Weights of the vacuum mass eigenstates of a neutrino produced as ``nu_e`` in matter and leaving adiabatically.
    chord_midpoints(...)
        Lengths and radial midpoints of the segments of a chord of the Earth, cut at the nodes of a radial table.
    atmosphere_path(...)
        Lengths and mass density of the segments of the atmosphere crossed from the production point to the surface.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

# hbar c in eV m: hbar = 6.582119569509067e-22 MeV s (exact SI) times c = 299 792 458 m/s.
HBARC_EV_M: float = 6.582119569509067e-22 * 299_792_458.0 * 1.0e6
_G_F_MEV_M2 = 1.1663787e-11
_N_A = 6.02214076e23
# sqrt(2) G_F N_A in eV for one mol/cm^3 of electrons: G_F in eV^-2 and (hbar c)^3 to turn cm^-3 into eV^3.
V_CC_EV: float = math.sqrt(2.0) * _G_F_MEV_M2 * _N_A * (6.582119569509067e-22 * 299_792_458.0 * 1.0e2) ** 3 * 1.0e6
R_EARTH_M: float = 6.371e6


def rotation(i: int, j: int, theta: float, phase: float = 0.0, n: int = 4) -> np.ndarray:
    """Complex ``n x n`` rotation in the ``(i, j)`` plane with the phase on the upper off-diagonal element.

    Args:
        i: First index of the plane.
        j: Second index of the plane.
        theta: Rotation angle in radians.
        phase: CP phase in radians, ``exp(-i phase)`` on element ``(i, j)``.
        n: Dimension of the matrix.

    Returns:
        Complex array shaped ``(n, n)``.
    """
    matrix = np.eye(n, dtype=complex)
    matrix[i, i] = matrix[j, j] = math.cos(theta)
    matrix[i, j] = math.sin(theta) * np.exp(-1j * phase)
    matrix[j, i] = -math.sin(theta) * np.exp(1j * phase)
    return matrix


def mixing_matrix(
    config: object,
    *,
    order: Sequence[str] = ("34", "24", "14", "23", "13", "12"),
    sign: int = 1,
    antinu: bool = False,
) -> np.ndarray:
    """The mixing matrix of a configuration: ``R23 R13 R12`` for three flavours, ``R34 R24 R14 R23 R13 R12`` for four.

    Args:
        config: Object with ``theta12``, ``theta13``, ``theta23``, ``delta_cp`` and, for ``numneu == 4``, ``theta14``,
            ``theta24``, ``theta34``, ``delta14`` and ``delta24`` (``NuSQuIDSConfig`` has them all). A missing
            ``numneu`` means three flavours.
        order: Order of the rotations of the four-flavour product, for the search of a convention; the default is
            that of nuSQuIDS. Ignored for three flavours.
        sign: Sign of the phases, ``-1`` for the conjugate convention.
        antinu: If True, return the conjugate matrix.

    Returns:
        Complex array shaped ``(3, 3)`` or ``(4, 4)``.
    """
    n = int(getattr(config, "numneu", 3))
    if n == 3:
        U = (
            rotation(1, 2, config.theta23, n=3)
            @ rotation(0, 2, config.theta13, sign * config.delta_cp, n=3)
            @ rotation(0, 1, config.theta12, n=3)
        )
    else:
        rotations = {
            "12": rotation(0, 1, config.theta12),
            "13": rotation(0, 2, config.theta13, sign * config.delta_cp),
            "23": rotation(1, 2, config.theta23),
            "14": rotation(0, 3, config.theta14, sign * config.delta14),
            "24": rotation(1, 3, config.theta24, sign * config.delta24),
            "34": rotation(2, 3, config.theta34),
        }
        U = np.eye(4, dtype=complex)
        for name in order:
            U = U @ rotations[name]
    return U.conj() if antinu else U


def mass_squared(config: object) -> np.ndarray:
    """The mass-squared offsets ``(0, dm21, dm3l)`` or ``(0, dm21, dm3l, dm41)`` of a configuration, in eV^2.

    Args:
        config: Object with ``DeltamSq21``, ``DeltamSq3l`` and, for ``numneu == 4``, ``DeltamSq41``.

    Returns:
        Array of length 3 or 4.
    """
    values = [0.0, config.DeltamSq21, config.DeltamSq3l]
    if int(getattr(config, "numneu", 3)) == 4:
        values.append(config.DeltamSq41)
    return np.array(values, dtype=float)


def vacuum_probability(
    mixing: np.ndarray,
    masses_eV2: np.ndarray,
    energy_gev: float,
    baseline_km: float,
    *,
    antinu: bool = False,
    hbarc_ev_m: float = HBARC_EV_M,
) -> np.ndarray:
    """Flavour probabilities after a baseline in vacuum, ``P[initial, final]``.

    Args:
        mixing: Neutrino mixing matrix, from ``mixing_matrix``.
        masses_eV2: Mass-squared offsets in eV^2.
        energy_gev: Neutrino energy in GeV.
        baseline_km: Baseline in km.
        antinu: If True, conjugate the mixing.
        hbarc_ev_m: Value of hbar c in eV m, to test the sensitivity of a comparison to the constant.

    Returns:
        Real array shaped ``(n, n)``.
    """
    U = mixing.conj() if antinu else mixing
    phase = masses_eV2 * baseline_km * 1.0e3 / (2.0 * energy_gev * 1.0e9) / hbarc_ev_m
    amplitude = (U * np.exp(-1j * phase)[None, :]) @ U.conj().T
    return (np.abs(amplitude) ** 2).T


def _hamiltonian(
    mixing: np.ndarray,
    masses_eV2: np.ndarray,
    energy_ev: float,
    electron_density: np.ndarray,
    neutron_density: Optional[np.ndarray],
    antinu: bool,
    neutral_current: bool,
) -> np.ndarray:
    """Hamiltonians in eV of every density: vacuum term plus the charged-current and sterile neutral-current ones."""
    U = mixing.conj() if antinu else mixing
    n = len(masses_eV2)
    kinetic = U @ np.diag(masses_eV2) @ U.conj().T / (2.0 * energy_ev)
    sign = -1.0 if antinu else 1.0
    electron = np.zeros((n, n))
    electron[0, 0] = 1.0
    H = kinetic[None] + sign * V_CC_EV * np.asarray(electron_density, dtype=float)[:, None, None] * electron[None]
    if neutral_current and n > 3 and neutron_density is not None:
        sterile = np.zeros((n, n))
        sterile[n - 1, n - 1] = 1.0
        H = H + sign * 0.5 * V_CC_EV * np.asarray(neutron_density, dtype=float)[:, None, None] * sterile[None]
    return H


def evolution_operator(
    energy_gev: float,
    lengths_m: np.ndarray,
    electron_density: np.ndarray,
    neutron_density: Optional[np.ndarray] = None,
    *,
    mixing: np.ndarray,
    masses_eV2: np.ndarray,
    antinu: bool = False,
    neutral_current: bool = True,
) -> np.ndarray:
    """Evolution operator of consecutive segments of constant density, the first segment crossed first.

    The Hamiltonian of each segment is diagonalised and its exponential applied; there is no free numerical parameter
    once the segments are fine enough, which is what makes it a reference.

    Args:
        energy_gev: Neutrino energy in GeV.
        lengths_m: Segment lengths in m, shaped ``(n_segments,)``.
        electron_density: Electron density of each segment in mol/cm^3.
        neutron_density: Neutron density of each segment in mol/cm^3, used for the sterile neutral-current potential;
            ignored for three flavours.
        mixing: Neutrino mixing matrix (conjugated here for the antineutrino).
        masses_eV2: Mass-squared offsets in eV^2, length 3 or 4.
        antinu: If True, the antineutrino: conjugate mixing and opposite potentials.
        neutral_current: If False, leave out the sterile neutral-current potential.

    Returns:
        Complex array shaped ``(n, n)``, ``S[final, initial]`` in the flavour basis.
    """
    H = _hamiltonian(mixing, masses_eV2, energy_gev * 1.0e9, electron_density, neutron_density, antinu, neutral_current)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    lengths = np.asarray(lengths_m, dtype=float)
    segments = (
        eigenvectors * np.exp(-1j * eigenvalues * lengths[:, None] / HBARC_EV_M)[:, None, :]
    ) @ eigenvectors.conj().transpose(0, 2, 1)
    S = np.eye(len(masses_eV2), dtype=complex)
    for segment in segments:
        S = segment @ S
    return S


def probabilities(operator: np.ndarray) -> np.ndarray:
    """Probabilities of an evolution operator, ``P[initial, final]``."""
    return (np.abs(operator) ** 2).T


def mass_to_flavour(operator: np.ndarray, mixing: np.ndarray, *, antinu: bool = False) -> np.ndarray:
    """Probability that each vacuum mass eigenstate ends in each flavour, ``M[final, mass]`` ``= |S U|^2``."""
    U = mixing.conj() if antinu else mixing
    return np.abs(operator @ U) ** 2


def adiabatic_weights(
    electron_density: float,
    energy_mev: float,
    *,
    mixing: np.ndarray,
    masses_eV2: np.ndarray,
    neutron_density: float = 0.0,
    antinu: bool = False,
    neutral_current: bool = True,
) -> np.ndarray:
    """Weights of the vacuum mass eigenstates for a ``nu_e`` produced in matter that leaves adiabatically.

    The weight of each eigenstate is the electron content of the eigenstate of the Hamiltonian in matter that connects
    with it, ``|<nu_e | nu_i^m>|^2``; the eigenstates of the Hamiltonian, sorted by energy, are put back in the order
    of the vacuum masses, which matters when ``dm41 < dm31``.

    Args:
        electron_density: Electron density at the production point in mol/cm^3.
        energy_mev: Neutrino energy in MeV.
        mixing: Neutrino mixing matrix.
        masses_eV2: Mass-squared offsets in eV^2, length 3 or 4.
        neutron_density: Neutron density at the production point in mol/cm^3 (sterile neutral-current potential).
        antinu: If True, the antineutrino.
        neutral_current: If False, leave out the sterile neutral-current potential.

    Returns:
        Weights shaped ``(n,)``, in the order of ``masses_eV2``, adding to one.
    """
    H = _hamiltonian(
        mixing,
        masses_eV2,
        energy_mev * 1.0e6,
        np.array([electron_density]),
        np.array([neutron_density]),
        antinu,
        neutral_current,
    )[0]
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    by_energy = np.abs(eigenvectors[0]) ** 2
    return by_energy[np.argsort(np.argsort(masses_eV2))]


def chord_midpoints(
    cos_zenith: float,
    node_fractions: np.ndarray,
    subdivisions: int = 16,
    earth_radius_m: float = R_EARTH_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segments of a chord of the Earth, cut at every node of a radial table and subdivided.

    The chord of a neutrino that arrives with ``cos_zenith < 0`` has length ``2 R cos(eta)`` with ``eta = arccos(-cos)``
    and a closest approach ``R sin(eta)`` to the centre. Its boundaries are the intersections with every radius of the
    table, so that no segment straddles a discontinuity, and each interval is divided in ``subdivisions`` equal parts.

    Args:
        cos_zenith: cosine of the zenith angle, strictly negative.
        node_fractions: Radii of the table in units of the Earth radius.
        subdivisions: Number of equal parts of each interval.
        earth_radius_m: Earth radius in m.

    Returns:
        ``(lengths_m, midpoint_fractions)``: the length of each segment and the radius, in units of the Earth radius,
        of its middle, where the caller evaluates the density.
    """
    eta = math.acos(-cos_zenith)
    radius, rmin, length = earth_radius_m, earth_radius_m * math.sin(eta), 2.0 * earth_radius_m * math.cos(eta)
    radii = np.unique(node_fractions[node_fractions * radius > rmin]) * radius
    root = np.sqrt(np.maximum(radii**2 - rmin**2, 0.0))
    nodes = np.unique(
        np.clip(
            np.concatenate([[0.0, length], radius * math.cos(eta) - root, radius * math.cos(eta) + root]), 0.0, length
        )
    )
    parts = [np.linspace(a, b, subdivisions + 1) for a, b in zip(nodes[:-1], nodes[1:], strict=True)]
    nodes = np.unique(np.concatenate(parts))
    middle = 0.5 * (nodes[1:] + nodes[:-1])
    fraction = np.sqrt(np.maximum(radius**2 - 2.0 * radius * middle * math.cos(eta) + middle**2, 0.0)) / radius
    return np.diff(nodes), fraction


def atmosphere_path(
    cos_zenith: float,
    height_m: float,
    segments: int,
    rho0_gcm3: float,
    scale_height_km: float,
    earth_radius_m: float = R_EARTH_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segments of the atmosphere crossed by a neutrino produced at ``height_m``, production end first, and density.

    The path has length ``-R cos + sqrt(R^2 cos^2 + 2 R h + h^2)``; for an upgoing neutrino (``cos < 0``) the part that
    lies below the horizon is the Earth, so the atmosphere leg starts at ``-2 R cos``. The density is exponential,
    ``rho0 exp(-altitude / scale_height)``.

    Args:
        cos_zenith: cosine of the zenith angle at the detector.
        height_m: Production altitude above the surface in m.
        segments: Number of equal segments.
        rho0_gcm3: Mass density at the surface in g/cm^3.
        scale_height_km: Scale height in km.
        earth_radius_m: Earth radius in m.

    Returns:
        ``(lengths_m, rho_gcm3)`` in the order in which the neutrino crosses them; multiply the density by the electron
        fraction, or by one minus it, for the electron or the neutron density.
    """
    radius = earth_radius_m
    length = -radius * cos_zenith + math.sqrt((radius * cos_zenith) ** 2 + 2.0 * radius * height_m + height_m**2)
    start = max(0.0, -2.0 * radius * cos_zenith)
    x = np.linspace(start, length, segments + 1)
    middle = 0.5 * (x[1:] + x[:-1])
    altitude = np.sqrt(radius**2 + middle**2 + 2.0 * radius * middle * cos_zenith) - radius
    density = rho0_gcm3 * np.exp(-altitude / 1.0e3 / scale_height_km)
    return np.diff(x)[::-1].copy(), density[::-1].copy()
