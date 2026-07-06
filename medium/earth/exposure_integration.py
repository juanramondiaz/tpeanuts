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
Earth exposure integration for matter-regeneration probabilities.

This module evaluates Earth probabilities over a nadir-angle exposure table
and integrates the result into an exposure-averaged probability vector. It is
part of the observable layer of the Earth pipeline: it does not build
Hamiltonians or evolution operators directly, but delegates those operations
to ``medium.earth.probability.pearth``.

The calculation performed here is

    P_int(E) = integral d eta W(eta) P_earth(E, eta),

where ``W(eta)`` is obtained from ``medium.earth.exposure_table`` and
``P_earth(E, eta)`` is computed either with the perturbative analytical
pipeline or with the numerical pipeline. The implementation supports chunking
over the eta grid to control memory usage when large energy-angle batches are
evaluated.

Module functions:
    _prepare_energy_grid(...)
        Convert scalar or vector energies into a one-dimensional tensor and
        record whether the scalar dimension must be removed at the end.
    pearth_integrated(...)
        Compute exposure-integrated Earth flavour probabilities for one or
        more neutrino energies.
"""



from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from tpeanuts.util.type import TensorLike

import tpeanuts.util.default as default
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.numerical.geometry import OdeMethod
from tpeanuts.medium.earth.probability import PearthMethod, pearth
from tpeanuts.medium.earth.exposure_table import ExposureParameters, build_nadir_exposure
from tpeanuts.util.context import RuntimeContext

Tensor = torch.Tensor


def _prepare_energy_grid(
    E_MeV: TensorLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, bool]:
    """Prepare the energy axis used by the exposure integration.

    Args:
        E_MeV: Scalar or tensor-like neutrino energy in MeV.
        device: Device where the energy tensor is allocated.
        dtype: Real dtype used for the energy tensor.

    Returns:
        Tuple ``(E, squeeze_E)`` where ``E`` is one-dimensional and
        ``squeeze_E`` indicates whether the input was scalar.
    """
    if torch.is_tensor(E_MeV):
        E = E_MeV.to(device=device, dtype=dtype)
    else:
        E = torch.tensor(E_MeV, device=device, dtype=dtype)

    if E.ndim == 0:
        E = E[None]
        squeeze_E = True
    else:
        squeeze_E = False

    return E, squeeze_E

  
@torch.no_grad()
def pearth_integrated(
    nustate: Tensor,
    profile_earth: object,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    depth_m: float,
    *,
    method: PearthMethod = default.earth_method,
    massbasis: bool = default.earth_massbasis,
    exposure: ExposureParameters = ExposureParameters(),
    normalized_exposure: bool = default.earth_normalized_exposure,
    context: RuntimeContext = RuntimeContext.resolve(default.earth_device, default.dtype),
    chunk_eta: Optional[int] = default.earth_chunk_eta,
    reunitarize: bool = default.earth_reunitarize,
    nsteps: int = default.earth_probability_nsteps,
    ode_method: OdeMethod | None = default.earth_numerical_method,
) -> Tensor:
    """Compute Earth probabilities averaged over a nadir exposure table.

    The function builds or loads an exposure table ``W(eta)``, evaluates
    ``pearth`` on the corresponding energy-angle grid, and accumulates
    ``P_int(E) = integral d eta W(eta) P_earth(E, eta)``. Analytical mode is
    evaluated as a batched tensor operation, while numerical mode currently
    loops over scalar trajectories.

    Args:
        nustate: Initial state with final dimension 3. Interpreted as
            incoherent mass weights when ``massbasis=True`` and coherent
            flavour amplitudes otherwise.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Scalar or vector of neutrino energies in MeV.
        depth_m: Detector depth in metres.
        method: Earth probability method, either ``"analytical"`` or
            ``"numerical"``.
        massbasis: Selects the interpretation of ``nustate``.
        exposure: Exposure-table construction settings. The
            ``exposure_source`` selector is passed through unchanged to
            ``build_nadir_exposure`` and can be "math", "cache", "csv", or
            "legacy". The default ``ExposureParameters()`` selects "math".
        normalized_exposure: Normalize the exposure weights before
            integration.
        context: Runtime device/dtype used by the integration.
        chunk_eta: Number of eta samples evaluated per batch. ``None`` or a
            non-positive value evaluates the full eta grid at once.
        reunitarize: For analytical propagation, project evolution operators
            to the nearest unitary matrix.
        nsteps: Number of numerical trajectory samples for numerical mode.
        ode_method: Numerical profile sampling rule for numerical mode.

    Returns:
        Exposure-integrated final flavour probabilities with final dimension
        3. A scalar energy input returns a single probability vector; vector
        energy input preserves the leading energy dimension.
    """
    dev, dtype = context.device, context.dtype
    antinu = oscillation.antinu

    if method not in ("analytical", "numerical"):
        raise ValueError("method must be either 'analytical' or 'numerical'.")

    source = exposure.exposure_source
    if source not in ("math", "cache", "csv", "legacy"):
        raise ValueError("exposure_source must be 'math', 'cache', 'csv' or 'legacy'.")

    exposure_table = build_nadir_exposure(
        exposure=exposure,
        context=context,
        normalized=normalized_exposure,
    )

    eta_grid = exposure_table.eta
    w_eta = exposure_table.exposure

    deta = torch.pi / exposure.exposure_ns

    E, squeeze_E = _prepare_energy_grid(
        E_MeV,
        device=dev,
        dtype=dtype,
    )

    n_energy = E.shape[0]

    antinu_t = None
    if torch.is_tensor(antinu):
        antinu_t = antinu.to(device=dev, dtype=torch.bool)

    out = torch.zeros(
        (n_energy, 3),
        device=dev,
        dtype=dtype,
    )

    if chunk_eta is None or chunk_eta <= 0:
        chunk_eta = eta_grid.numel()

    for start in range(0, eta_grid.numel(), chunk_eta):

        eta_chunk = eta_grid[start:start + chunk_eta]
        w_chunk = w_eta[start:start + chunk_eta]

        E_grid = E[:, None].expand(
            n_energy,
            eta_chunk.numel(),
        )

        eta_grid_chunk = eta_chunk[None, :].expand(
            n_energy,
            eta_chunk.numel(),
        )

        antinu_chunk = antinu
        if antinu_t is not None:
            if antinu_t.shape == eta_grid.shape:
                antinu_chunk = antinu_t[start:start + chunk_eta]
            elif antinu_t.ndim >= 1 and antinu_t.shape[-1] == eta_grid.numel():
                antinu_chunk = antinu_t[..., start:start + chunk_eta]
            else:
                antinu_chunk = antinu_t

        if method == "analytical":
            P_chunk = pearth(
                nustate=nustate,
                profile_earth=profile_earth,
                oscillation=dataclasses.replace(oscillation, antinu=antinu_chunk),
                E_MeV=E_grid,
                eta=eta_grid_chunk,
                depth_m=depth_m,
                method="analytical",
                massbasis=massbasis,
                reunitarize=reunitarize,
            )
        else:
            if torch.is_tensor(antinu_chunk) and antinu_chunk.numel() != 1:
                raise ValueError("Numerical earth integration only supports scalar antinu.")

            antinu_scalar = bool(antinu_chunk.item()) if torch.is_tensor(antinu_chunk) else antinu_chunk
            oscillation_scalar = dataclasses.replace(oscillation, antinu=antinu_scalar)

            P_rows = []

            for i_energy in range(n_energy):
                P_eta = []

                for eta_value in eta_chunk:
                    P_eta.append(
                        pearth(
                            nustate=nustate,
                            profile_earth=profile_earth,
                            oscillation=oscillation_scalar,
                            E_MeV=E[i_energy],
                            eta=eta_value,
                            depth_m=depth_m,
                            method="numerical",
                            massbasis=massbasis,
                            full_oscillation=False,
                            nsteps=nsteps,
                            ode_method=ode_method,
                            context=context,
                        )
                    )

                P_rows.append(torch.stack(P_eta, dim=0))

            P_chunk = torch.stack(P_rows, dim=0)

        out = out + torch.sum(
            P_chunk * w_chunk[None, :, None],
            dim=1,
        ) * deta

    if squeeze_E:
        return out[0]

    return out
