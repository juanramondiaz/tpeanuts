
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

from __future__ import annotations

from typing import Optional, Union
import math
import torch
from torch import Tensor

import tpeanuts.util.default as default

TensorLike = Union[float, int, Tensor]



from tpeanuts.core.potential import kinetic_potential, matter_potential
from tpeanuts.util.type import _as_tensor, _as_complex_tensor
from tpeanuts.util.constant import R_E

@torch.no_grad()
def numerical_solution(
    density: object,
    pmns: object,
    dm21_eV2: TensorLike,
    dm3l_eV2: TensorLike,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    *,
    antinu: bool = default.earth_antinu,
    nsteps: int = default.earth_nsteps,
    method: str | None = default.earth_numerical_method,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = default.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:

    if torch.is_tensor(antinu):
        if antinu.numel() != 1:
            raise ValueError("numerical_solution only supports scalar antinu.")
        antinu = bool(antinu.item())

    if device is None:
        device = (
            E_MeV.device
            if isinstance(E_MeV, torch.Tensor)
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    eta = _as_tensor(eta, device=device, dtype=dtype)
    E = _as_tensor(E_MeV, device=device, dtype=dtype)
    dm21 = _as_tensor(dm21_eV2, device=device, dtype=dtype)
    dm3l = _as_tensor(dm3l_eV2, device=device, dtype=dtype)

    
    # --------------------------------------------------------
    # PMNS objects
    # --------------------------------------------------------

    U = pmns.U.to(device=device, dtype=cdtype)

    r23 = pmns.R23().to(device=device, dtype=cdtype)

    delta = pmns.Delta().to(device=device, dtype=cdtype)

    if antinu:
        U = torch.conj(U)
        r23 = torch.conj(r23)
        delta = torch.conj(delta)

    if dm3l > 0:
        mSq = torch.stack(
            [
                torch.zeros_like(dm21),
                dm21,
                dm3l,
            ]
        )
    else:
        mSq = torch.stack(
            [
                -dm21,
                torch.zeros_like(dm21),
                dm3l,
            ]
        )

    # --------------------------------------------------------
    # vacuum Hamiltonian
    # --------------------------------------------------------

    k = kinetic_potential(
        mSq_eV2=mSq,
        E_MeV=E,
    )

    Hk = U @ torch.diag(k.to(cdtype)) @ U.T

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    h = depth_m / R_E
    r_d = 1.0 - h

    x_d = r_d * torch.cos(eta)

    delta_x = (
        r_d * torch.cos(eta)
        + torch.sqrt(
            torch.clamp(
                1.0 - r_d**2 * torch.sin(eta) ** 2,
                min=0.0,
            )
        )
    )

    eta_prime = torch.asin(r_d * torch.sin(eta))

    eta_float = float(eta.detach().cpu())
    eta_prime_float = float(eta_prime.detach().cpu())

    if 0.0 <= eta_float < math.pi / 2.0:
        xj_all, crossed, _ = density.shells_x(eta_prime)
        xj_crossed = torch.where(crossed, xj_all, torch.zeros_like(xj_all))
        x1 = -float(torch.max(xj_crossed).detach().cpu())
        x2 = float(x_d.detach().cpu())
    else:
        x1 = 0.0
        x2 = float(delta_x.detach().cpu())

    x = torch.linspace(
        x1,
        x2,
        nsteps + 1,
        device=device,
        dtype=dtype,
    )

    dx = x[1:] - x[:-1]

    n_1 = density.call(
        1.0 - h / 2.0,
        0.0,
    )
    # --------------------------------------------------------
    # Evolution
    # --------------------------------------------------------

    S = torch.eye(
        3,
        device=device,
        dtype=cdtype,
    )

    S_list = [S.clone()]

    for j in range(nsteps):

        if method in (None, "midpoint"):
            t = 0.5 * (x[j] + x[j + 1])
        elif method == "left":
            t = x[j]
        elif method == "right":
            t = x[j + 1]
        else:
            raise ValueError("method must be None, 'midpoint', 'left' or 'right'.")

        if 0.0 <= eta_float < math.pi / 2.0:
            n_e = density.call(
                float(t.detach().cpu()),
                eta_prime_float,
            )
        else:
            n_e = n_1

        n_e = _as_tensor(
            n_e,
            device=device,
            dtype=dtype,
        )

        V = matter_potential(
            n_mol_cm3=n_e,
            antinu=antinu,
        ).to(cdtype)

        Hm = torch.diag(
            torch.stack(
                [
                    V,
                    torch.zeros_like(V),
                    torch.zeros_like(V),
                ]
            )
        )

        H = r23 @ delta @ (Hk + Hm) @ torch.conj(delta).T @ r23.T

        U_step = torch.linalg.matrix_exp(
            -1j * H * dx[j].to(cdtype)
        )

        S = U_step @ S
        S_list.append(S.clone())

    Sx = torch.stack(S_list, dim=0)

    return Sx, x
