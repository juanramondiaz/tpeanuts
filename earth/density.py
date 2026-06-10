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
earth density model utilities for peanuts-torch.

This module defines the torch-native earth electron-density model used by the
earth matter-regeneration block.

The density profile is represented shell by shell as an even polynomial in the
dimensionless earth radius r:

    n_e(r) =
        alpha
        + beta r^2
        + gamma r^4
        + sum_n delta_n r^{2(n+3)}.

For a neutrino trajectory with nadir angle eta, the radial coordinate is written
as

    r^2 = x^2 + sin^2(eta),

where x is the dimensionless coordinate along the trajectory. Therefore, inside
each shell, the same density can be rewritten as an even polynomial in x:

    n_e(x, eta) =
        alpha'(eta)
        + beta'(eta) x^2
        + gamma'(eta) x^4
        + sum_n delta'_n(eta) x^{2(n+3)}.

The main class is:

    earthdensity
        Torch-based earth electron-density container.

The main methods are:

    shells_x(...)
        Computes shell-crossing coordinates x_j for a given eta.

    parameters(...)
        Computes all transformed polynomial coefficients along the trajectory.

    parameters_abc(...)
        Computes only the alpha', beta', gamma' coefficients used by the
        perturbative core segment evolutor.

    density_x_eta(...)
        Evaluates n_e(x, eta) along the trajectory.

    call(...)
        peanuts-compatible density evaluation wrapper.

    __call__(...)
        Allows direct usage as density(r, eta).

This module does not compute Hamiltonians, evolution operators, oscillation
probabilities, or exposure averages. It only defines the earth matter-density
model and the trajectory-dependent polynomial coefficients.
"""



from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor

from tpeanuts.util.math import _binom

@dataclass
class EarthDensity:
    """
    Torch representation of the peanuts earth electron-density profile.

    Parameters
    ----------
    rj:
        Shell radii in dimensionless earth-radius units. Shape: (Nshell,).

    alpha:
        Constant density coefficients per shell. Shape: (Nshell,).

    beta:
        Quadratic radial density coefficients per shell. Shape: (Nshell,).

    gamma:
        Quartic radial density coefficients per shell. Shape: (Nshell,).

    deltas:
        Higher-order density coefficients. Shape: (Nd, Nshell), or
        (0, Nshell) if no higher-order terms are present.

    tabulated:
        If True, the profile was loaded as a tabulated constant-density model.
    """

    rj: Tensor
    alpha: Tensor
    beta: Tensor
    gamma: Tensor
    deltas: Tensor
    tabulated: bool = False

    @property
    def device(self) -> torch.device:
        return self.rj.device

    @property
    def dtype(self) -> torch.dtype:
        return self.rj.dtype

    @torch.no_grad()
    def shells_x(
        self,
        eta: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dev = self.device
        dt = self.dtype

        if not torch.is_tensor(eta):
            eta = torch.tensor(eta, device=dev, dtype=dt)
        else:
            eta = eta.to(device=dev, dtype=dt)

        s = torch.sin(eta)
        s_col = s[..., None]

        r = self.rj.reshape(*((1,) * eta.ndim), -1)

        crossed = r > s_col

        idx0 = torch.searchsorted(
            self.rj,
            s,
        )

        x2 = torch.clamp(
            r * r - s_col * s_col,
            min=0.0,
        )

        xj_all = torch.sqrt(x2)

        return xj_all, crossed, idx0

    @torch.no_grad()
    def parameters(
        self,
        eta: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dev = self.device
        dt = self.dtype

        if not torch.is_tensor(eta):
            eta = torch.tensor(eta, device=dev, dtype=dt)
        else:
            eta = eta.to(device=dev, dtype=dt)

        s = torch.sin(eta)[..., None]
        s2 = s * s
        s4 = s2 * s2

        xj_all, crossed, _ = self.shells_x(eta)

        ns = self.rj.numel()
        lead_shape = eta.shape
        base_view = (1,) * eta.ndim + (ns,)

        alpha0 = self.alpha.view(base_view).expand(*lead_shape, ns)
        beta0 = self.beta.view(base_view).expand(*lead_shape, ns)
        gamma0 = self.gamma.view(base_view).expand(*lead_shape, ns)

        alpha_p = alpha0 + beta0 * s2 + gamma0 * s4
        beta_p = beta0 + 2.0 * gamma0 * s2
        gamma_p = gamma0

        nd = int(self.deltas.shape[0])

        if nd > 0:
            deltas_view = (1,) * eta.ndim + (nd, ns)

            deltas = self.deltas.view(deltas_view).expand(*lead_shape,nd,ns)

            for n in range(nd):
                alpha_p = alpha_p + deltas[..., n, :] * (s ** (2 * (n + 3)))
                beta_p = beta_p + (n + 3) * deltas[..., n, :] * (s ** (2 * (n + 2)))

                coefficient = _binom(n + 3, 2, dev, dt)

                gamma_p = gamma_p + coefficient * deltas[..., n, :] * (s ** (2 * (n + 1)))

            deltas_p = deltas.clone()

            for n in range(nd):
                for k in range(n + 1, nd):
                    coefficient = _binom(k + 3, n + 3, dev, dt)

                    deltas_p[..., n, :] = (deltas_p[..., n, :] + coefficient
                        * deltas[..., k, :] * (s ** (2 * (k - n)))
                    )

            deltas_p = deltas_p.transpose(-2, -1)

        else:
            deltas_p = torch.zeros(
                (*alpha_p.shape, 0),
                device=dev,
                dtype=dt,
            )

        coeffs = torch.cat(
            [
                alpha_p[..., None],
                beta_p[..., None],
                gamma_p[..., None],
                deltas_p,
            ],
            dim=-1,
        )

        return coeffs, xj_all, crossed

    @torch.no_grad()
    def parameters_abc(
        self,
        eta: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dev = self.device
        dt = self.dtype

        if not torch.is_tensor(eta):
            eta = torch.tensor(eta, device=dev, dtype=dt)
        else:
            eta = eta.to(device=dev, dtype=dt)

        s = torch.sin(eta)[..., None]
        s2 = s * s
        s4 = s2 * s2

        xj_all, crossed, _ = self.shells_x(eta)

        ns = self.rj.numel()
        lead_shape = eta.shape
        base_view = (1,) * eta.ndim + (ns,)

        alpha0 = self.alpha.view(base_view).expand(*lead_shape, ns)
        beta0 = self.beta.view(base_view).expand(*lead_shape, ns)
        gamma0 = self.gamma.view(base_view).expand(*lead_shape, ns)

        alpha_p = alpha0 + beta0 * s2 + gamma0 * s4
        beta_p = beta0 + 2.0 * gamma0 * s2
        gamma_p = gamma0

        coeffs_abc = torch.stack(
            [alpha_p, beta_p,gamma_p],
            dim=-1
        )

        return coeffs_abc, xj_all, crossed

    @torch.no_grad()
    def density_x_eta(
        self,
        x: Tensor,
        eta: Tensor,
    ) -> Tensor:
        dev = self.device
        dt = self.dtype

        if not torch.is_tensor(x):
            x = torch.tensor(x, device=dev, dtype=dt)
        else:
            x = x.to(device=dev, dtype=dt)

        if not torch.is_tensor(eta):
            eta = torch.tensor(eta, device=dev, dtype=dt)
        else:
            eta = eta.to(device=dev, dtype=dt)

        xabs = torch.abs(x)

        eta_b = eta + torch.zeros_like(xabs)

        cos_eta = torch.cos(eta_b)

        outside = xabs > cos_eta

        coeffs, xj, crossed = self.parameters(eta_b)

        neg_inf = torch.tensor(
            float("-inf"),
            device=dev,
            dtype=dt,
        )

        xj_eff = torch.where(
            crossed,
            xj,
            neg_inf,
        )

        idx = torch.searchsorted(
            xj_eff,
            xabs[..., None],
            right=False,
        ).squeeze(-1)

        idx = torch.clamp(
            idx,
            0,
            xj_eff.shape[-1] - 1,
        )

        n_coeffs = coeffs.shape[-1]

        gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(
            *idx.shape,
            1,
            n_coeffs,
        )

        selected = torch.gather(
            coeffs,
            dim=-2,
            index=gather_idx,
        ).squeeze(-2)

        a = selected[..., 0]
        b = selected[..., 1]
        c = selected[..., 2]
        deltas = selected[..., 3:]

        x2 = xabs * xabs

        n_e = a + b * x2 + c * (x2 * x2)

        if deltas.numel() > 0:
            powers = torch.stack(
                [
                    x2 ** (3 + i)
                    for i in range(deltas.shape[-1])
                ],
                dim=-1,
            )

            n_e = n_e + torch.sum(
                deltas * powers,
                dim=-1,
            )

        n_e = torch.where(
            outside,
            torch.zeros_like(n_e),
            n_e,
        )

        return n_e

    def call(
        self,
        r: Tensor,
        eta: Tensor,
    ) -> Tensor:
        return self.density_x_eta(
            r,
            eta,
        )

    def __call__(
        self,
        r: Tensor,
        eta: Tensor,
    ) -> Tensor:
        return self.call(
            r,
            eta,
        )