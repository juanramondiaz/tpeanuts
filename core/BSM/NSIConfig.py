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
Non-Standard Interaction (NSI) parameter configuration for neutrino propagation.

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
    arXiv:2112.09122.
    NSI constraints from IceCube DeepCore atmospheric neutrinos.

Usage
-----
::

    from tpeanuts.core.BSM.NSIConfig import NSIConfig

    cfg  = NSIConfig.from_preset("nsi_lma_dark_esteban2018")
    print(cfg)

    eps  = cfg.epsilon_tensor()            # torch.Tensor, shape (3, 3), complex
    H    = hamiltonian_reduced_bsm(..., epsilon=eps)

    # SM limit — equivalent to passing epsilon=None
    cfg0 = NSIConfig.from_preset("sm_no_nsi")
    assert cfg0.is_sm_limit

The preset registry itself (data and bounds/citations per preset) lives in
``tpeanuts.core.common.presets.NSI_PRESETS`` — see that module for the
available preset names and their physics justification.

Module contents
---------------
NSIConfig
    Frozen dataclass storing all NSI parameters.  Provides
    ``epsilon_tensor`` to build the complex 3×3 ε matrix, and the
    ``from_preset`` classmethod for named parameter sets. To list available
    preset names, call
    ``tpeanuts.core.common.presets.list_presets(NSI_PRESETS)`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tpeanuts.core.common.presets import NSI_PRESETS, get_preset


# ---------------------------------------------------------------------------
# NSIConfig dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NSIConfig:
    """Complete NSI parameter set for neutrino propagation in matter.

    The Hermitian 3×3 matrix ε is stored as nine real numbers:

    * Three real **diagonal** entries: ``eps_ee``, ``eps_mumu``, ``eps_tautau``.
    * Three **complex off-diagonal** entries decomposed into real and imaginary
      parts: ``eps_emu_re / _im``, ``eps_etau_re / _im``, ``eps_mutau_re / _im``.

    The SM limit is all fields at 0.  Use ``epsilon_tensor()`` to obtain the
    corresponding complex 3×3 torch tensor.

    Only the traceless part of ε is observable; ``eps_mumu = 0`` can be set
    without loss of generality for the diagonal sector.

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
    label : str
        Short identifier string (e.g. the preset name).
    description : str
        Human-readable description and literature reference.
    """

    # Diagonal entries (real)
    eps_ee:      float = 0.0
    eps_mumu:    float = 0.0
    eps_tautau:  float = 0.0

    # Off-diagonal ε_eμ (complex)
    eps_emu_re:  float = 0.0
    eps_emu_im:  float = 0.0

    # Off-diagonal ε_eτ (complex)
    eps_etau_re: float = 0.0
    eps_etau_im: float = 0.0

    # Off-diagonal ε_μτ (complex)
    eps_mutau_re: float = 0.0
    eps_mutau_im: float = 0.0

    # Metadata
    label:       str = ""
    description: str = ""

    # ------------------------------------------------------------------
    # Preset interface
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, name: str) -> "NSIConfig":
        """Build an ``NSIConfig`` from a named preset.

        Args:
            name: Preset identifier.  Call
                ``tpeanuts.core.common.presets.list_presets(NSI_PRESETS)``
                for all names.

        Returns:
            Fully initialized ``NSIConfig`` instance.

        Raises:
            ValueError: If ``name`` is not in ``tpeanuts.core.common.presets.NSI_PRESETS``.
        """
        return cls(**get_preset(NSI_PRESETS, name, kind="NSI preset"))

    # ------------------------------------------------------------------
    # Tensor builder
    # ------------------------------------------------------------------

    def epsilon_tensor(
        self,
        device: Optional[torch.device] = None,
        real_dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Build the 3×3 Hermitian ε matrix as a complex torch tensor.

        The tensor is constructed once per call (no caching).  Pass the
        result as the ``epsilon`` argument of ``hamiltonian_reduced_bsm`` or
        ``hamiltonian_flavour_bsm``.

        Args:
            device: Target torch device.  Defaults to CPU.
            real_dtype: Real base dtype; the complex dtype is inferred
                (float32 → complex64, float64 → complex128).

        Returns:
            Complex tensor shaped (3, 3) representing ε.
        """
        cdtype = (
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        )

        eps_emu   = complex(self.eps_emu_re,   self.eps_emu_im)
        eps_etau  = complex(self.eps_etau_re,  self.eps_etau_im)
        eps_mutau = complex(self.eps_mutau_re, self.eps_mutau_im)

        data = [
            [complex(self.eps_ee,     0.0), eps_emu,              eps_etau           ],
            [eps_emu.conjugate(),           complex(self.eps_mumu, 0.0), eps_mutau   ],
            [eps_etau.conjugate(),          eps_mutau.conjugate(), complex(self.eps_tautau, 0.0)],
        ]

        return torch.tensor(data, dtype=cdtype, device=device)

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    @property
    def is_sm_limit(self) -> bool:
        """True when all ε parameters are zero (Standard Model limit)."""
        return all(
            v == 0.0
            for v in (
                self.eps_ee, self.eps_mumu, self.eps_tautau,
                self.eps_emu_re, self.eps_emu_im,
                self.eps_etau_re, self.eps_etau_im,
                self.eps_mutau_re, self.eps_mutau_im,
            )
        )

    @property
    def has_cp_violation(self) -> bool:
        """True when any off-diagonal imaginary part is non-zero."""
        return any(
            v != 0.0
            for v in (self.eps_emu_im, self.eps_etau_im, self.eps_mutau_im)
        )

    @property
    def eps_emu(self) -> complex:
        """ε_eμ as a Python complex number."""
        return complex(self.eps_emu_re, self.eps_emu_im)

    @property
    def eps_etau(self) -> complex:
        """ε_eτ as a Python complex number."""
        return complex(self.eps_etau_re, self.eps_etau_im)

    @property
    def eps_mutau(self) -> complex:
        """ε_μτ as a Python complex number."""
        return complex(self.eps_mutau_re, self.eps_mutau_im)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Return a multi-line human-readable summary of all NSI parameters."""
        label_str = f" [{self.label}]" if self.label else ""

        def _fmt_complex(re: float, im: float) -> str:
            if im == 0.0:
                return f"{re:+.4f}"
            return f"{re:+.4f} {'+' if im >= 0 else ''}{im:.4f}i"

        lines = [
            f"NSIConfig{label_str}",
            f"  Diagonal (real):              Off-diagonal (complex):",
            f"    ε_ee   = {self.eps_ee:+.4f}          ε_eμ  = {_fmt_complex(self.eps_emu_re,   self.eps_emu_im)}",
            f"    ε_μμ   = {self.eps_mumu:+.4f}          ε_eτ  = {_fmt_complex(self.eps_etau_re,  self.eps_etau_im)}",
            f"    ε_ττ   = {self.eps_tautau:+.4f}          ε_μτ  = {_fmt_complex(self.eps_mutau_re, self.eps_mutau_im)}",
            f"  SM limit: {self.is_sm_limit}   CP violation: {self.has_cp_violation}",
        ]
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
