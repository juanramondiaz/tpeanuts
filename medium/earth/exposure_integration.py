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
to ``medium.earth.probability.earth_probability_state``.

The calculation performed here is

    P_exp(E) = integral d eta W(eta) P_earth(E, eta)
             ~ sum_i W(eta_i) P_earth(E, eta_i) * deta,

where ``W(eta)`` is obtained from ``medium.earth.exposure_table`` and
``P_earth(E, eta)`` is computed either with the perturbative analytical
pipeline or with the numerical pipeline. The implementation supports chunking
over the eta grid to control memory usage when large energy-angle batches are
evaluated. This time-averages over the nadir angle and is distinct from
``core.common.probability.probability_integrated_angular``, which performs a
geometric solid-angle average rather than an exposure-weighted one.

Quadrature rule (deliberately a rectangle sum, not a trapezoid): the eta
integral above is evaluated as a plain Riemann sum with a single grid spacing
``deta = eta[1] - eta[0]``, not ``torch.trapezoid``. This reproduces the
legacy peanuts ``Pearth_integrated`` convention (once its ``deta = pi / ns``
grid-spacing bug -- inconsistent with its own ``ns``-point grid, which is
actually spaced ``pi / (ns - 1)`` -- is corrected to the true spacing used
here), and is cross-validated against it in
``medium.earth.test.test7_legacy_validation``. ``medium.earth.exposure_table.
integrate_exposure`` is a separate, genuinely trapezoidal reduction of the
same physical quantity; it is not used by this function and will not
reproduce these numbers bit-for-bit (the two differ by the usual
O(1/n_eta) trapezoid-vs-rectangle boundary term). Prefer this module's
``earth_probability_exposure`` when matching the legacy reference matters;
prefer ``integrate_exposure`` when you already have an arbitrary
``(eta, exposure)`` pair and want the more accurate rule with no legacy
constraint. Because the rectangle sum uses one shared ``deta``, this function
requires ``exposure_table.eta`` to be uniformly spaced (true for every
built-in exposure source: "math"/"legacy" via ``make_eta_grid``, "csv" via
its uniform re-gridding, "cache" via round-tripping one of the former) and
raises ``ValueError`` if it is not.

Module functions:
    _prepare_energy_grid(...)
        Convert scalar or vector energies into a one-dimensional tensor and
        record whether the scalar dimension must be removed at the end.
    earth_probability_exposure(...)
        Compute exposure-integrated Earth flavour probabilities for one or
        more neutrino energies.
"""



from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from tpeanuts.util.type import TensorLike

import tpeanuts.config.default as default
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.numerical.geometry import OdeMethod
from tpeanuts.medium.earth.probability import PearthMethod, earth_probability_state
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
def earth_probability_exposure(
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
    include_matter_nc: Optional[bool] = None,
    analytic_eigenvalues: bool = False,
) -> Tensor:
    """Compute Earth probabilities averaged over a nadir exposure table.

    The function builds or loads an exposure table ``W(eta)``, evaluates
    ``earth_probability_state`` on the corresponding energy-angle grid, and
    accumulates ``P_exp(E) = integral d eta W(eta) P_earth(E, eta)``.
    Both analytical and numerical mode are evaluated as a single batched
    tensor operation over the full (energy, eta) grid -- numerical mode no
    longer loops over scalar trajectories now that
    ``medium.earth.geometry.build_earth_trajectory`` accepts a batched eta.

    Args:
        nustate: Initial state with final dimension matching
            ``oscillation.pmns.n_flavours`` (3, or 4 for the 3+1 sterile
            extension). Interpreted as incoherent mass weights when
            ``massbasis=True`` and coherent flavour amplitudes otherwise.
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
        include_matter_nc: If True/False, applied/not applied (see
            ``medium.earth.probability.earth_probability_state``). If
            ``None`` (the default), auto-resolved per-call.
        analytic_eigenvalues: If True, use the closed-form Cardano/Ferrari
            eigenvalues instead of ``torch.linalg.eigvalsh`` for analytical
            propagation (see ``medium.earth.probability.
            earth_probability_state``). Only meaningful with
            ``method="analytical"``; forwarded to every
            ``earth_probability_state`` call below, whose own validation
            raises if combined with ``method="numerical"``.

    Returns:
        Exposure-integrated final flavour probabilities with final dimension
        matching ``oscillation.pmns.n_flavours`` (3, or 4 for the 3+1
        sterile extension). A scalar energy input returns a single
        probability vector; vector energy input preserves the leading
        energy dimension.
    """
    dev, dtype = context.device, context.dtype
    antinu = oscillation.antinu
    n_flavours = int(oscillation.pmns.n_flavours)

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

    deta = eta_grid[1] - eta_grid[0]
    if eta_grid.numel() > 2:
        spacing = eta_grid[1:] - eta_grid[:-1]
        if not torch.allclose(spacing, deta.expand_as(spacing), rtol=1.0e-6, atol=1.0e-10):
            raise ValueError(
                "earth_probability_exposure integrates with a single shared "
                "'deta' (a rectangle-sum rule, see the module docstring), "
                "which requires a uniformly spaced eta grid; the exposure "
                f"table built from exposure_source={source!r} is not "
                "uniformly spaced."
            )

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
        (n_energy, n_flavours),
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

        # earth_probability_state_numerical now batches over the full
        # (energy, eta) grid in one call and accepts a tensor antinu
        # broadcast against that grid -- same as method="analytical" -- so
        # both methods share this one call. nsteps/ode_method/context are
        # accepted (and ignored) for method="analytical"; reunitarize is
        # forwarded for method="numerical" too (even though the numerical
        # evolutor never re-unitarizes) so that reunitarize=True raises the
        # same "has no effect" ValueError here as it does for a direct
        # earth_probability_state call, instead of being silently dropped.
        P_chunk = earth_probability_state(
            nustate=nustate,
            profile_earth=profile_earth,
            oscillation=dataclasses.replace(oscillation, antinu=antinu_chunk),
            E_MeV=E_grid,
            eta=eta_grid_chunk,
            depth_m=depth_m,
            method=method,
            massbasis=massbasis,
            full_oscillation=False,
            nsteps=nsteps,
            ode_method=ode_method,
            context=context,
            reunitarize=reunitarize,
            include_matter_nc=include_matter_nc,
            analytic_eigenvalues=analytic_eigenvalues,
        )

        out = out + torch.sum(
            P_chunk * w_chunk[None, :, None],
            dim=1,
        ) * deta

    if squeeze_E:
        return out[0]

    return out
