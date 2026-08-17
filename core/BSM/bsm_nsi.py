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
Non-Standard Interaction (NSI) parameter configuration, for neutrino
propagation.

Physics background
------------------
NSI are effective four-fermion operators that modify the coherent forward
scattering of neutrinos in matter beyond the MSW potential [1].  The matter
Hamiltonian in the reduced flavour basis becomes

    H_mat^NSI = V_CC · (δ_{αe} δ_{βe} + ε_{αβ})

where V_CC = ±√2 G_F n_e L_scale is the standard CC potential and ε is a
3×3 Hermitian matrix of dimensionless couplings:

    ε = | ε_ee    ε_eμ    ε_eτ  |
        | ε_eμ*   ε_μμ    ε_μτ  |
        | ε_eτ*   ε_μτ*   ε_ττ  |

Diagonal entries are real; off-diagonal entries are generally complex.  The
SM limit is ε = 0.  For oscillations only the traceless part of ε matters;
setting ε_μμ = 0 is a valid convention for the diagonal sector.

Composition dependence (electron/proton vs neutron)
-----------------------------------------------------
NSI couple to individual matter fermions (e, u, d), not to "matter" as a
single substance.  The most general propagation NSI Lagrangian carries three
independent per-flavour-pair couplings ε^e, ε^u, ε^d (one per fermion
species); the effective matrix seen by a Hamiltonian built from number
densities n_e, n_p, n_n is

    ε_αβ(x) = ε_αβ^e + (n_p(x)/n_e(x))·ε_αβ^p + (n_n(x)/n_e(x))·ε_αβ^n,

with the proton coupling itself a fixed combination of the quark ones,
ε^p = 2ε^u + ε^d, and the neutron coupling the complementary one,
ε^n = ε^u + 2ε^d.  Since matter is electrically neutral (n_p = n_e
everywhere), n_p(x)/n_e(x) ≡ 1 identically, so the three-fermion sum
collapses to exactly two *effective* pieces, not three independent physical
ones:

    ε_αβ(x) = ε_αβ^(e+p)  +  (n_n(x) / n_e(x)) · ε_αβ^n,     ε^(e+p) ≡ ε^e + 2ε^u + ε^d

``eps_*`` (equivalently ``epsilon``) is the first, electron/proton-
normalized piece ε^(e+p); ``eps_*_n`` (equivalently ``epsilon_n``) is the
second, neutron-normalized piece ε^n = ε^u + 2ε^d. Calling the first block
"electron/proton" is a normalization label, not a claim that it is a
single fundamental coupling: it is the fixed combination ε^e + 2ε^u + ε^d
that happens to multiply n_e(x) (via n_p(x) ≡ n_e(x)) in any electrically
neutral medium. ``epsilon_n`` defaults to the zero matrix, so every preset
and every existing caller reproduces the pre-existing single-matrix ε
exactly.

Published NSI bounds and best-fit values (including every preset in
``NSI_PRESETS``, all of which leave ``eps_*_n`` at zero) are fits of the
*effective*, single-medium-composition ε -- typically Earth-like matter,
n_n(x)/n_e(x) ≈ 1 -- not of ε^e/ε^u/ε^d individually. ``epsilon_n = 0`` is
therefore a modelling hypothesis carried by the preset (or by the caller
who built a config with only ``eps_*`` set), not a physical fact derived
from the quoted bound: two different experiments with different target
composition can each report an "ε_ee" consistent with a common ε^e, ε^u,
ε^d but numerically different ε^(e+p), and neither alone fixes ε^n.

``n_n(x)/n_e(x)`` is not a universal number: it is ~0 in the Sun's
hydrogen-rich envelope, ~1 in its helium/metal core, and ~1.0-1.15 across
the Earth's mantle/core boundary.  A single constant ε (``epsilon_n = 0``)
implicitly assumes the medium composition used wherever that ε was fit or
quoted; this is a good approximation for propagation confined to one
roughly isoscalar medium, but not across media of very different
composition.  See ``core.common.hamiltonian.hamiltonian_matter_reduced``
for how ``epsilon_n`` is combined with local ``n_n_mol_cm3`` data (reusing
``core.common.potential.matter_potential_cc`` on the neutron density,
which is algebraically equivalent to the ratio above without ever dividing
by n_e). A non-zero ``eps_*_n`` is treated as a firm request, not an
optional refinement: every consumer (``hamiltonian_matter_reduced``,
``core.perturbative.evolutor.evolutor_perturbative_segment``, and every
``medium.*`` pipeline) raises ``ValueError`` rather than silently
reproducing the ``epsilon_n = 0`` result when the local ``n_n`` data it
needs is unavailable.

References
----------
[1] Grossman (1995), Phys. Lett. B 359, 141.
    arXiv:hep-ph/9507344.  Original NSI proposal for propagation.

[2] Biggio, Blennow, Fernandez-Martinez (2009), JHEP 08:090.
    arXiv:0907.0097.  Model-independent NSI bounds.

[3] Esteban, Gonzalez-Garcia, Maltoni, Martinez-Soler, Schwetz (2018),
    JHEP 08:180.  arXiv:1805.04530.
    Global oscillation fit including NSI; establishes LMA-Dark degeneracy.

[4] IceCube Collaboration (2022), Phys. Rev. D 106, 032009.
    arXiv:2106.07755.
    All-flavour NSI constraints from IceCube DeepCore atmospheric neutrinos.

Usage
-----
::

    from tpeanuts.core.BSM.bsm_nsi import NSIConfig
    from tpeanuts.config.propagation import PropagationConfig

    # Usual path: attach the NSI preset directly to OscillationParameters
    # (this builds the NSIConfig internally; epsilon is always recomputed
    # automatically from the eps_* fields in __post_init__) so every
    # downstream builder --
    # core.common.hamiltonian.hamiltonian_reduced/hamiltonian_flavour,
    # core.numerical.evolutor.evolutor_numerical,
    # core.perturbative.evolutor.evolutor_perturbative_segment -- reads
    # oscillation.nsi.epsilon automatically; none of them take a separate
    # epsilon argument.
    oscillation = PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO", NSI_extension="nsi_lma_dark_esteban2018",
    )
    print(oscillation.nsi)
    eps = oscillation.nsi.epsilon          # torch.Tensor, shape (3, 3), complex

    # SM limit — equivalent to oscillation.nsi = None
    cfg0 = NSIConfig.from_preset("sm_no_nsi")
    assert cfg0.is_sm_limit

The preset registry itself (data and bounds/citations per preset) lives in
``tpeanuts.config.presets.NSI_PRESETS`` — see that module for the
available preset names and their physics justification.

Module contents
---------------
NSIConfig
    Frozen dataclass storing all NSI parameters.  ``__post_init__`` always
    recomputes the complex 3×3 ``epsilon`` field from the ``eps_*`` fields
    (via ``epsilon_tensor_base``), on every construction and every
    ``dataclasses.replace(...)`` call, so ``eps_*`` is always the single
    source of truth and ``epsilon`` can never silently go stale.
    ``from_preset`` builds a named parameter set the same way;
    ``from_raw_epsilon`` is the one escape hatch for an arbitrary epsilon
    tensor that does not decompose into the scalar parametrization.
    ``epsilon_tensor`` embeds ``self.epsilon`` for a Hamiltonian of arbitrary
    flavour count (used by
    ``tpeanuts.core.common.hamiltonian.hamiltonian_matter_reduced`` for the
    3+1 sterile extension). To list available preset names, call
    ``tpeanuts.config.presets.list_presets(NSI_PRESETS)`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from tpeanuts.config.presets import NSI_PRESETS, get_preset
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import as_tensor, cdtype_from_real

FloatOrTensor = Union[float, torch.Tensor]

# Hermiticity-check tolerance for from_raw_epsilon, expressed as a multiplier
# of torch.finfo(dtype).eps so it scales automatically with precision
# (float32/complex64 vs float64/complex128), mirroring the convention already
# used for the degeneracy thresholds in core.perturbative.spectral.
_HERMITICITY_RELATIVE_EPS: float = 1.0e2


def _py_float(x: FloatOrTensor) -> float:
    """Convert a float-or-tensor scalar field to a plain Python float.

    Used only by non-differentiable convenience/display accessors
    (``__str__``, ``eps_emu``/``eps_etau``/``eps_mutau``/their ``_n``
    counterparts): detaching a tensor there is correct, not a gradient
    leak, since the differentiable path is ``epsilon``/``epsilon_n``
    themselves (built by ``_hermitian_3x3``), never these accessors.

    Args:
        x: Python float, or a real-valued torch.Tensor (any shape that
            ``float()`` accepts, i.e. exactly one element).

    Returns:
        Plain Python float.
    """
    if torch.is_tensor(x):
        return float(x.detach())
    return float(x)


def _hermitian_3x3(
    ee: FloatOrTensor, mumu: FloatOrTensor, tautau: FloatOrTensor,
    emu_re: FloatOrTensor, emu_im: FloatOrTensor,
    etau_re: FloatOrTensor, etau_im: FloatOrTensor,
    mutau_re: FloatOrTensor, mutau_im: FloatOrTensor,
    *,
    device: Optional[torch.device],
    real_dtype: torch.dtype,
) -> torch.Tensor:
    """Assemble a Hermitian 3x3 complex tensor from 9 real scalar couplings.

    Shared by ``NSIConfig.epsilon_tensor_base`` (electron/proton-normalized
    ε) and ``NSIConfig.epsilon_n_tensor_base`` (neutron-normalized ε^n),
    which differ only in which of the ``eps_*``/``eps_*_n`` field groups
    they pass in.

    Built entirely with ``torch`` ops (``as_tensor``/``.to()``/
    ``torch.complex``/``torch.stack``), never Python's ``complex()`` builtin
    or ``torch.tensor(nested_python_list)``: unlike those, every step here
    is differentiable, so a ``torch.Tensor`` argument with
    ``requires_grad=True`` stays connected to the returned matrix (and hence
    to ``NSIConfig.epsilon``/``epsilon_n``) without needing the
    ``from_raw_epsilon`` escape hatch. A plain ``float`` argument still
    produces a plain (non-grad-tracking) leaf tensor, exactly as before.

    Args:
        ee, mumu, tautau: Real diagonal entries, each a Python float or a
            real-valued 0-d/broadcastable ``torch.Tensor``.
        emu_re, emu_im, etau_re, etau_im, mutau_re, mutau_im: Real/imaginary
            parts of the three off-diagonal entries, same float-or-tensor
            convention.
        device: Target torch device. ``None`` resolves to CPU -- every
            field is moved (differentiably) onto this single resolved
            device, so mixing a GPU-resident tensor field with plain-float
            fields never produces a mixed-device result.
        real_dtype: Real base dtype; the complex dtype is inferred
            (float32 → complex64, float64 → complex128). Every field is
            differentiably cast to this dtype via ``.to()``.

    Returns:
        Complex tensor shaped (3, 3).
    """
    cdtype = cdtype_from_real(real_dtype)
    resolved_device = device if device is not None else torch.device("cpu")

    def _rt(x: FloatOrTensor) -> torch.Tensor:
        return as_tensor(x, dtype=real_dtype, device=resolved_device)

    zero = torch.zeros((), dtype=real_dtype, device=resolved_device)
    ee_c = torch.complex(_rt(ee), zero)
    mumu_c = torch.complex(_rt(mumu), zero)
    tautau_c = torch.complex(_rt(tautau), zero)
    emu_c = torch.complex(_rt(emu_re), _rt(emu_im))
    etau_c = torch.complex(_rt(etau_re), _rt(etau_im))
    mutau_c = torch.complex(_rt(mutau_re), _rt(mutau_im))

    row0 = torch.stack((ee_c, emu_c, etau_c))
    row1 = torch.stack((emu_c.conj(), mumu_c, mutau_c))
    row2 = torch.stack((etau_c.conj(), mutau_c.conj(), tautau_c))
    return torch.stack((row0, row1, row2)).to(dtype=cdtype)


# ---------------------------------------------------------------------------
# NSIConfig dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class NSIConfig:
    """Complete NSI parameter set for neutrino propagation in matter.

    The Hermitian 3×3 matrix ε is stored as nine real numbers:

    * Three real **diagonal** entries: ``eps_ee``, ``eps_mumu``, ``eps_tautau``.
    * Three **complex off-diagonal** entries decomposed into real and imaginary
      parts: ``eps_emu_re / _im``, ``eps_etau_re / _im``, ``eps_mutau_re / _im``.

    The SM limit is all fields at 0.  Use ``epsilon_tensor_base()`` to obtain
    the corresponding complex 3×3 torch tensor.

    Only the traceless part of ε is observable; ``eps_mumu = 0`` can be set
    without loss of generality for the diagonal sector.

    Equality and hashing are deliberately **identity-based**
    (``eq=False``, so ``__eq__``/``__hash__`` are inherited from ``object``
    rather than auto-generated from the fields): every ``eps_*``/``eps_*_n``
    field now accepts a differentiable ``torch.Tensor`` (see
    ``epsilon_tensor_base``), and ``from_raw_epsilon`` stores a caller
    tensor directly while leaving the scalar fields at their 0.0 defaults --
    both make a field-tuple-based structural ``__eq__`` unsound (comparing
    tensors with ``==``/hashing them can raise, and two configs with very
    different physics but the same 0.0 scalar defaults would otherwise
    compare equal). Two ``NSIConfig`` instances built with identical
    arguments therefore compare unequal (``is`` identity only); use
    ``torch.equal(a.epsilon, b.epsilon)`` (and ``a.epsilon_n``/``b.epsilon_n``)
    directly for value comparison.

    Parameters
    ----------
    eps_ee : float
        ε_ee — real, diagonal e-neutrino entry.  Defaults to 0.
    eps_mumu : float
        ε_μμ — real, diagonal μ-neutrino entry.  Defaults to 0.
    eps_tautau : float
        ε_ττ — real, diagonal τ-neutrino entry.  Defaults to 0.
    eps_emu_re, eps_emu_im : float
        Real and imaginary parts of ε_eμ.  Defaults to 0.
    eps_etau_re, eps_etau_im : float
        Real and imaginary parts of ε_eτ.  Defaults to 0.
    eps_mutau_re, eps_mutau_im : float
        Real and imaginary parts of ε_μτ.  Defaults to 0.
    eps_ee_n, eps_mumu_n, eps_tautau_n, eps_emu_n_re/_im, eps_etau_n_re/_im,
    eps_mutau_n_re/_im : float
        Neutron-normalized counterparts of the nine fields above (see the
        module docstring's "Composition dependence" section): the effective
        matter epsilon is ``epsilon + (n_n/n_e) * epsilon_n``.  All default
        to 0, reproducing the electron/proton-only epsilon above exactly.
    label : str
        Short identifier string (e.g. the preset name).
    description : str
        Human-readable description and literature reference.
    device : Optional[torch.device]
        Target torch device for the auto-computed ``epsilon`` tensor.
        Defaults to CPU.
    real_dtype : torch.dtype
        Real base dtype for the auto-computed ``epsilon`` tensor; the
        complex dtype is inferred (float32 -> complex64, float64 ->
        complex128).
    epsilon : Optional[torch.Tensor]
        Complex 3x3 epsilon tensor. Recomputed automatically in
        ``__post_init__`` from the ``eps_*`` fields (via
        ``epsilon_tensor_base()``) on *every* construction, including every
        ``dataclasses.replace(...)`` call -- so ``eps_*`` is always the
        single source of truth and this field can never silently diverge
        from it. Any value passed explicitly to the constructor is
        discarded and overwritten; use ``NSIConfig.from_raw_epsilon`` for
        the rare case of an arbitrary epsilon tensor that does not
        decompose into the standard scalar parametrization. This is the
        field ``OscillationParameters.nsi.epsilon`` exposes.
    epsilon_n : Optional[torch.Tensor]
        Complex 3x3 neutron-normalized epsilon tensor, mirroring
        ``epsilon`` exactly but recomputed from the ``eps_*_n`` fields (via
        ``epsilon_n_tensor_base()``). Defaults to the zero matrix.
    """

    # Diagonal entries (real). Each accepts a plain float or a differentiable
    # real-valued torch.Tensor (0-d, or broadcastable) -- see
    # epsilon_tensor_base()/_hermitian_3x3.
    eps_ee:      FloatOrTensor = 0.0
    eps_mumu:    FloatOrTensor = 0.0
    eps_tautau:  FloatOrTensor = 0.0

    # Off-diagonal ε_eμ (complex)
    eps_emu_re:  FloatOrTensor = 0.0
    eps_emu_im:  FloatOrTensor = 0.0

    # Off-diagonal ε_eτ (complex)
    eps_etau_re: FloatOrTensor = 0.0
    eps_etau_im: FloatOrTensor = 0.0

    # Off-diagonal ε_μτ (complex)
    eps_mutau_re: FloatOrTensor = 0.0
    eps_mutau_im: FloatOrTensor = 0.0

    # Neutron-normalized couplings ε^n (composition dependence -- see the
    # module docstring's "Composition dependence" section). All default to
    # 0.0, so the effective epsilon = epsilon + (n_n/n_e)*epsilon_n reduces
    # exactly to the electron/proton-only epsilon above for every existing
    # preset and caller.
    eps_ee_n:      FloatOrTensor = 0.0
    eps_mumu_n:    FloatOrTensor = 0.0
    eps_tautau_n:  FloatOrTensor = 0.0

    eps_emu_n_re:  FloatOrTensor = 0.0
    eps_emu_n_im:  FloatOrTensor = 0.0

    eps_etau_n_re: FloatOrTensor = 0.0
    eps_etau_n_im: FloatOrTensor = 0.0

    eps_mutau_n_re: FloatOrTensor = 0.0
    eps_mutau_n_im: FloatOrTensor = 0.0

    # Metadata
    label:       str = ""
    description: str = ""

    # Device/dtype used to build epsilon in __post_init__.
    device: Optional[torch.device] = None
    real_dtype: torch.dtype = torch.float64

    # Derived tensors: always recomputed from eps_*/eps_*_n in __post_init__
    # (see from_raw_epsilon for the one legitimate way to override them).
    epsilon: Optional[torch.Tensor] = field(default=None, compare=False)
    epsilon_n: Optional[torch.Tensor] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Recompute ``epsilon``/``epsilon_n`` from the ``eps_*``/``eps_*_n`` fields.

        Runs on every construction, including every ``dataclasses.replace``
        call (which re-invokes ``__init__``/``__post_init__``). This makes
        ``eps_*``/``eps_*_n`` the single source of truth for
        ``epsilon``/``epsilon_n``: tweaking a single field via ``replace``
        can never leave a stale tensor behind, closing the divergence risk
        that merely-cached fields would otherwise allow.
        """
        object.__setattr__(
            self,
            "epsilon",
            self.epsilon_tensor_base(device=self.device, real_dtype=self.real_dtype),
        )
        object.__setattr__(
            self,
            "epsilon_n",
            self.epsilon_n_tensor_base(device=self.device, real_dtype=self.real_dtype),
        )

    # ------------------------------------------------------------------
    # Preset interface
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        device: Optional[torch.device] = None,
        real_dtype: torch.dtype = torch.float64,
    ) -> "NSIConfig":
        """Build an ``NSIConfig`` from a named preset.

        Args:
            name: Preset identifier.  Call
                ``tpeanuts.config.presets.list_presets(NSI_PRESETS)``
                for all names.
            device: Target torch device for the auto-computed ``epsilon``
                tensor.  Defaults to CPU (see ``epsilon_tensor``).
            real_dtype: Real base dtype for ``epsilon``; the complex dtype is
                inferred (float32 -> complex64, float64 -> complex128).

        Returns:
            Fully initialized ``NSIConfig`` instance; ``epsilon`` is built
            automatically by ``__post_init__``.

        Raises:
            ValueError: If ``name`` is not in ``tpeanuts.config.presets.NSI_PRESETS``.
        """
        return cls(
            **get_preset(NSI_PRESETS, name, kind="NSI preset"),
            device=device,
            real_dtype=real_dtype,
        )

    @classmethod
    def from_raw_epsilon(
        cls,
        epsilon: torch.Tensor,
        *,
        epsilon_n: Optional[torch.Tensor] = None,
        label: str = "",
        description: str = "",
    ) -> "NSIConfig":
        """Build an ``NSIConfig`` from an arbitrary epsilon tensor.

        Bypasses the ``eps_*``/``eps_*_n`` scalar parametrization entirely:
        those fields are left at their defaults (0.0) and do NOT reflect
        ``epsilon``/``epsilon_n``'s actual entries. Use this only when
        ``epsilon`` does not decompose into the standard 3x3 Hermitian
        parametrization (e.g. a precomputed matrix, or a non-3x3 shape for
        ``epsilon_tensor`` embedding tests) -- for every other case, prefer
        the ``eps_*``/``eps_*_n`` fields (directly or via ``from_preset``),
        which keep ``epsilon``/``epsilon_n`` self-consistent automatically
        *and* stay differentiable (``epsilon_tensor_base`` is now built
        entirely with ``torch`` ops -- ``torch.complex``/``torch.stack`` --
        so a ``torch.Tensor`` with ``requires_grad=True`` passed as e.g.
        ``eps_ee`` reaches ``epsilon`` intact; this used to be the main
        reason to reach for ``from_raw_epsilon``, see ``_hermitian_3x3``).

        **Do not call ``dataclasses.replace()`` on a config built this
        way.** ``__post_init__`` always recomputes ``epsilon``/``epsilon_n``
        from the ``eps_*``/``eps_*_n`` fields on *every* construction,
        including every ``replace()`` call (this is what keeps ``eps_*`` a
        reliable single source of truth for the normal scalar-field path,
        see ``__post_init__``'s docstring) -- for a config built via
        ``from_raw_epsilon``, those fields are still at their 0.0 defaults,
        so ``replace()`` silently resets ``epsilon``/``epsilon_n`` back to
        (at best) zero, discarding the raw tensor (and any gradient
        connected to it) with no warning:
        ``dataclasses.replace(NSIConfig.from_raw_epsilon(eps), label="x")``
        does **not** carry ``eps`` over. Build a fresh config with
        ``from_raw_epsilon`` again instead of replacing an existing one.

        The only validation performed here is that ``epsilon`` (and
        ``epsilon_n``, if given) is square, finite, and Hermitian (within a
        dtype-scaled tolerance) -- the minimum any coherent-forward-
        scattering matter potential must satisfy for the resulting
        Hamiltonian to be Hermitian, and hence the propagation unitary. It
        deliberately does *not* enforce a 3x3/4x4 shape: that is
        ``epsilon_tensor``'s responsibility (see its own ``ValueError`` for
        an incompatible flavour count), so a caller can still build a
        deliberately mismatched-shape config here to exercise that check.

        Args:
            epsilon: Tensor stored as-is on the returned config. Must be
                square, finite, and Hermitian.
            epsilon_n: Optional neutron-normalized epsilon tensor (see the
                module docstring's "Composition dependence" section),
                stored as-is when given. Must satisfy the same validation as
                ``epsilon``. Left at the ``__post_init__``-built zero matrix
                (i.e. no composition dependence) when omitted.
            label: Optional short identifier string.
            description: Optional human-readable description.

        Returns:
            ``NSIConfig`` with ``epsilon`` (and, if given, ``epsilon_n``) set
            to exactly the given tensor(s).

        Raises:
            ValueError: If ``epsilon`` or ``epsilon_n`` is not square,
                contains non-finite (NaN/Inf) entries, or is not Hermitian
                within tolerance; or if both are given with different
                active-flavour block sizes (final two dimensions).
        """
        cls._validate_raw_epsilon(epsilon, name="epsilon")
        if epsilon_n is not None:
            cls._validate_raw_epsilon(epsilon_n, name="epsilon_n")
            if tuple(epsilon.shape[-2:]) != tuple(epsilon_n.shape[-2:]):
                raise ValueError(
                    "epsilon and epsilon_n must have the same active-flavour "
                    f"block size (final two dimensions): got {tuple(epsilon.shape[-2:])} "
                    f"and {tuple(epsilon_n.shape[-2:])}. Both are embedded into the "
                    "same n_flavours Hamiltonian block by "
                    "epsilon_tensor/epsilon_n_tensor, so a mismatch here would "
                    "only surface later as a confusing shape error there."
                )

        cfg = cls(label=label, description=description)
        object.__setattr__(cfg, "epsilon", epsilon)
        if epsilon_n is not None:
            object.__setattr__(cfg, "epsilon_n", epsilon_n)
        return cfg

    @staticmethod
    def _validate_raw_epsilon(epsilon: torch.Tensor, *, name: str) -> None:
        """Shared square/finite/Hermitian validation for ``from_raw_epsilon``.

        Args:
            epsilon: Tensor to validate.
            name: Field name, used only to make the error messages point at
                ``epsilon`` or ``epsilon_n`` correctly.

        Raises:
            ValueError: If ``epsilon`` has fewer than 2 dimensions, is not
                square, contains non-finite (NaN/Inf) entries, or is not
                Hermitian within tolerance.
        """
        if epsilon.ndim < 2:
            raise ValueError(
                f"{name} must have at least 2 dimensions (a square matrix "
                f"in the final two), got shape {tuple(epsilon.shape)}."
            )
        if epsilon.shape[-1] != epsilon.shape[-2]:
            raise ValueError(
                f"{name} must be square, got final dimensions "
                f"{tuple(epsilon.shape[-2:])}."
            )
        if not torch.isfinite(epsilon).all():
            raise ValueError(f"{name} contains non-finite (NaN/Inf) entries.")

        real_dtype = epsilon.real.dtype if epsilon.is_complex() else epsilon.dtype
        eps_mach = torch.finfo(real_dtype).eps
        scale = epsilon.abs().max().clamp_min(1.0)
        residual = (epsilon - epsilon.conj().transpose(-2, -1)).abs().max()
        tolerance = _HERMITICITY_RELATIVE_EPS * eps_mach * scale
        if residual > tolerance:
            raise ValueError(
                f"{name} must be Hermitian ({name} == {name}.conj().transpose(-2, -1)): "
                f"max|{name} - {name}^dagger| = {float(residual):.3e} exceeds "
                f"tolerance {float(tolerance):.3e}. The propagation matter "
                "potential must be Hermitian for the resulting Hamiltonian "
                "-- and hence the evolution -- to be unitary."
            )

    # ------------------------------------------------------------------
    # Tensor builder
    # ------------------------------------------------------------------

    def epsilon_tensor_base(
        self,
        device: Optional[torch.device] = None,
        real_dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Build the base 3x3 Hermitian ε matrix as a complex torch tensor.

        The tensor is constructed once per call (no caching). This is the
        "base" (3x3, not antineutrino-selected) matrix; ``epsilon_tensor``
        embeds ``self.epsilon`` for a Hamiltonian of arbitrary flavour count.

        Args:
            device: Target torch device.  Defaults to CPU.
            real_dtype: Real base dtype; the complex dtype is inferred
                (float32 → complex64, float64 → complex128).

        Returns:
            Complex tensor shaped (3, 3) representing ε.
        """
        return _hermitian_3x3(
            self.eps_ee, self.eps_mumu, self.eps_tautau,
            self.eps_emu_re, self.eps_emu_im,
            self.eps_etau_re, self.eps_etau_im,
            self.eps_mutau_re, self.eps_mutau_im,
            device=device, real_dtype=real_dtype,
        )

    def epsilon_n_tensor_base(
        self,
        device: Optional[torch.device] = None,
        real_dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Build the base 3x3 Hermitian ε^n (neutron-normalized) matrix.

        Mirrors ``epsilon_tensor_base`` exactly, reading the ``eps_*_n``
        fields instead of ``eps_*``. See the module docstring's "Composition
        dependence" section: the effective epsilon combining both blocks is
        ``epsilon + (n_n/n_e) * epsilon_n``, built by
        ``core.common.hamiltonian.hamiltonian_matter_reduced``.

        Args:
            device: Target torch device. Defaults to CPU.
            real_dtype: Real base dtype; the complex dtype is inferred
                (float32 → complex64, float64 → complex128).

        Returns:
            Complex tensor shaped (3, 3) representing ε^n.
        """
        return _hermitian_3x3(
            self.eps_ee_n, self.eps_mumu_n, self.eps_tautau_n,
            self.eps_emu_n_re, self.eps_emu_n_im,
            self.eps_etau_n_re, self.eps_etau_n_im,
            self.eps_mutau_n_re, self.eps_mutau_n_im,
            device=device, real_dtype=real_dtype,
        )

    def epsilon_tensor(
        self,
        *,
        n_flavours: int,
        context: Optional[RuntimeContext] = None,
    ) -> torch.Tensor:
        """Embed ``self.epsilon`` for a Hamiltonian with *n_flavours* flavours.

        If ``self.epsilon`` is already shaped ``(..., n_flavours, n_flavours)``
        it is returned as-is (after a device/dtype cast). If *n_flavours* > 3
        and ``self.epsilon`` is shaped ``(..., 3, 3)``, it is embedded in the
        top-left corner of an ``(n_flavours, n_flavours)`` zero matrix -- the
        sterile-sector rows and columns carry no NSI coupling. This is the
        base (not antineutrino-selected) matrix; callers needing the
        antineutrino convention must select it themselves on the result
        (e.g. via ``PMNS.select_antinu``).

        Args:
            n_flavours: Total number of flavours of the target Hamiltonian.
            context: Optional runtime device/dtype. When omitted, both are
                inferred from ``self.epsilon``.

        Returns:
            Complex tensor shaped ``(..., n_flavours, n_flavours)``.

        Raises:
            ValueError: If ``self.epsilon`` is ``None``, or if it has an
                incompatible shape.
        """
        return self._embed_active(self.epsilon, n_flavours=n_flavours, context=context, name="epsilon")

    def epsilon_n_tensor(
        self,
        *,
        n_flavours: int,
        context: Optional[RuntimeContext] = None,
    ) -> torch.Tensor:
        """Embed ``self.epsilon_n`` for a Hamiltonian with *n_flavours* flavours.

        Mirrors ``epsilon_tensor`` exactly, embedding ``self.epsilon_n``
        instead of ``self.epsilon``.

        Args:
            n_flavours: Total number of flavours of the target Hamiltonian.
            context: Optional runtime device/dtype. When omitted, both are
                inferred from ``self.epsilon_n``.

        Returns:
            Complex tensor shaped ``(..., n_flavours, n_flavours)``.

        Raises:
            ValueError: If ``self.epsilon_n`` is ``None``, or if it has an
                incompatible shape.
        """
        return self._embed_active(self.epsilon_n, n_flavours=n_flavours, context=context, name="epsilon_n")

    def _embed_active(
        self,
        eps: Optional[torch.Tensor],
        *,
        n_flavours: int,
        context: Optional[RuntimeContext],
        name: str,
    ) -> torch.Tensor:
        """Shared embedding logic for ``epsilon_tensor``/``epsilon_n_tensor``.

        Args:
            eps: The tensor to embed (``self.epsilon`` or ``self.epsilon_n``).
            n_flavours: Total number of flavours of the target Hamiltonian.
            context: Optional runtime device/dtype. When omitted, both are
                inferred from ``eps``.
            name: Field name, used only to point the error messages at
                ``epsilon`` or ``epsilon_n`` correctly.

        Returns:
            Complex tensor shaped ``(..., n_flavours, n_flavours)``.

        Raises:
            ValueError: If ``eps`` is ``None``, or if it has an incompatible
                shape.
        """
        if eps is None:
            raise ValueError(
                f"No {name} matrix available: use an NSIConfig with {name} "
                "already populated (e.g. from_preset)."
            )
        if context is not None:
            device, dtype = context.device, context.dtype
        else:
            device = eps.device
            dtype = eps.real.dtype if eps.is_complex() else eps.dtype
        cdtype = cdtype_from_real(dtype)
        eps = eps.to(device=device, dtype=cdtype)

        if eps.shape[-2:] == (n_flavours, n_flavours):
            return eps

        if n_flavours > 3 and eps.shape[-2:] == (3, 3):
            out = torch.zeros(
                (*eps.shape[:-2], n_flavours, n_flavours),
                device=device,
                dtype=cdtype,
            )
            out[..., :3, :3] = eps
            return out

        raise ValueError(
            f"{name} must have final dimensions "
            f"({n_flavours}, {n_flavours}) or active block (3, 3); "
            f"got {tuple(eps.shape[-2:])}."
        )

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    @property
    def is_sm_limit(self) -> bool:
        """True when both epsilon and epsilon_n are exactly zero (Standard Model limit).

        Inspects ``self.epsilon``/``self.epsilon_n`` directly, not the
        ``eps_*``/``eps_*_n`` scalar fields: those are the same information
        for a config built from them (directly or via ``from_preset``),
        since ``__post_init__`` always recomputes the tensors from the
        scalars, but ``from_raw_epsilon`` leaves the scalars at their
        defaults while setting the tensor(s) to the caller's actual
        matrices. Reading the tensors themselves keeps this property correct
        in both cases, rather than silently reporting the SM limit for a
        non-zero raw epsilon/epsilon_n.
        """
        if self.epsilon is not None and bool(torch.any(self.epsilon != 0)):
            return False
        if self.epsilon_n is not None and bool(torch.any(self.epsilon_n != 0)):
            return False
        return True

    @property
    def has_neutron_coupling(self) -> bool:
        """True when epsilon_n is not exactly the zero matrix.

        Gates the composition-dependent contribution
        ``V_CC(n_n) * epsilon_n`` in ``core.common.hamiltonian.
        hamiltonian_matter_reduced``: the default (all-zero) ``epsilon_n``
        costs nothing extra there, keeping the pre-existing electron/proton-
        only NSI Hamiltonian bit-identical when this is False. Also used by
        ``core.perturbative.evolutor.evolutor_perturbative_segment`` to
        raise rather than silently ignore a composition-dependent NSI
        coupling it does not yet support.
        """
        if self.epsilon_n is None:
            return False
        return bool(torch.any(self.epsilon_n != 0))

    @property
    def has_cp_violation(self) -> bool:
        """True when any off-diagonal entry of epsilon or epsilon_n has a non-zero imaginary part.

        Inspects ``self.epsilon``/``self.epsilon_n`` directly, excluding the
        diagonal (always real for a Hermitian matrix) -- see ``is_sm_limit``
        above for why this reads the tensors rather than the
        ``eps_*``/``eps_*_n`` scalar fields.
        """
        return self._has_offdiag_imag(self.epsilon) or self._has_offdiag_imag(self.epsilon_n)

    @staticmethod
    def _has_offdiag_imag(eps: Optional[torch.Tensor]) -> bool:
        """True when ``eps`` has any non-zero off-diagonal imaginary part."""
        if eps is None:
            return False
        n = eps.shape[-1]
        eye = torch.eye(n, dtype=torch.bool, device=eps.device)
        off_diagonal_imag = eps.imag.masked_fill(eye, 0.0)
        return bool(torch.any(off_diagonal_imag != 0))

    @property
    def eps_emu(self) -> complex:
        """ε_eμ as a Python complex number.

        Convenience/display accessor only, not part of the differentiable
        path: detaches a tensor field via ``_py_float`` before building the
        Python ``complex``. Use ``self.epsilon`` directly for a
        gradient-connected value.
        """
        return complex(_py_float(self.eps_emu_re), _py_float(self.eps_emu_im))

    @property
    def eps_etau(self) -> complex:
        """ε_eτ as a Python complex number. See ``eps_emu`` for the caveat."""
        return complex(_py_float(self.eps_etau_re), _py_float(self.eps_etau_im))

    @property
    def eps_mutau(self) -> complex:
        """ε_μτ as a Python complex number. See ``eps_emu`` for the caveat."""
        return complex(_py_float(self.eps_mutau_re), _py_float(self.eps_mutau_im))

    @property
    def eps_emu_n(self) -> complex:
        """ε^n_eμ as a Python complex number. See ``eps_emu`` for the caveat."""
        return complex(_py_float(self.eps_emu_n_re), _py_float(self.eps_emu_n_im))

    @property
    def eps_etau_n(self) -> complex:
        """ε^n_eτ as a Python complex number. See ``eps_emu`` for the caveat."""
        return complex(_py_float(self.eps_etau_n_re), _py_float(self.eps_etau_n_im))

    @property
    def eps_mutau_n(self) -> complex:
        """ε^n_μτ as a Python complex number. See ``eps_emu`` for the caveat."""
        return complex(_py_float(self.eps_mutau_n_re), _py_float(self.eps_mutau_n_im))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Return a multi-line human-readable summary of all NSI parameters."""
        label_str = f" [{self.label}]" if self.label else ""

        def _fmt_complex(re: FloatOrTensor, im: FloatOrTensor) -> str:
            re, im = _py_float(re), _py_float(im)
            if im == 0.0:
                return f"{re:+.4f}"
            return f"{re:+.4f} {'+' if im >= 0 else ''}{im:.4f}i"

        lines = [
            f"NSIConfig{label_str}",
            f"  Diagonal (real):              Off-diagonal (complex):",
            f"    ε_ee   = {_py_float(self.eps_ee):+.4f}          ε_eμ  = {_fmt_complex(self.eps_emu_re,   self.eps_emu_im)}",
            f"    ε_μμ   = {_py_float(self.eps_mumu):+.4f}          ε_eτ  = {_fmt_complex(self.eps_etau_re,  self.eps_etau_im)}",
            f"    ε_ττ   = {_py_float(self.eps_tautau):+.4f}          ε_μτ  = {_fmt_complex(self.eps_mutau_re, self.eps_mutau_im)}",
            f"  SM limit: {self.is_sm_limit}   CP violation: {self.has_cp_violation}",
        ]
        if self.has_neutron_coupling:
            lines.extend([
                f"  Neutron-normalized (ε^n = composition-dependent term, ε_eff = ε + (n_n/n_e)·ε^n):",
                f"    ε^n_ee = {_py_float(self.eps_ee_n):+.4f}          ε^n_eμ = {_fmt_complex(self.eps_emu_n_re,   self.eps_emu_n_im)}",
                f"    ε^n_μμ = {_py_float(self.eps_mumu_n):+.4f}          ε^n_eτ = {_fmt_complex(self.eps_etau_n_re,  self.eps_etau_n_im)}",
                f"    ε^n_ττ = {_py_float(self.eps_tautau_n):+.4f}          ε^n_μτ = {_fmt_complex(self.eps_mutau_n_re, self.eps_mutau_n_im)}",
            ])
        if self.description:
            words = self.description.split()
            line, wrapped = "  Note: ", []
            for word in words:
                if len(line) + len(word) + 1 > 72:
                    wrapped.append(line)
                    line = "        " + word
                else:
                    line += (" " if line.strip() else "") + word
            wrapped.append(line)
            lines.extend(wrapped)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a compact one-line repr with label and SM-limit flag."""
        label = self.label or "<unlabeled>"
        return f"NSIConfig(label={label!r}, is_sm_limit={self.is_sm_limit})"
