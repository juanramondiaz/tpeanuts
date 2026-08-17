"""Pure Earth workflow from the Earth surface to a detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, cast

import torch

from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.medium.earth.exposure_integration import earth_probability_exposure
from tpeanuts.medium.earth.exposure_table import prepare_nadir_exposure
from tpeanuts.medium.earth.probability import earth_probability_state
from tpeanuts.medium.earth.profile import EarthProfile, build_earth_profile
from tpeanuts.medium.earth.evolutor import EarthPerturbativeDiagnostics
from tpeanuts.util.torch_util import as_1d_tensor
from tpeanuts.util.type import TensorLike, cdtype_from_real


@dataclass(frozen=True)
class EarthDetectorResult:
    """Earth propagation result for an incident flavour or mass state."""

    incident_state: torch.Tensor
    incident_basis: Literal["flavour", "mass"]
    E_MeV: torch.Tensor
    profile: EarthProfile
    eta: Optional[torch.Tensor]
    exposure: Optional[torch.Tensor]
    probabilities_eta: Optional[torch.Tensor]
    probabilities_exposure: Optional[torch.Tensor]
    perturbative_diagnostics: Optional[EarthPerturbativeDiagnostics] = None


def _earth_inputs(
    incident_state: torch.Tensor,
    E_MeV: TensorLike,
    config: PropagationConfig,
    earth_profile: Optional[EarthProfile],
) -> tuple[torch.Tensor, torch.Tensor, EarthProfile]:
    context = config.runtime
    raw_state = torch.as_tensor(incident_state)
    state = raw_state.to(
        device=context.device,
        dtype=(
            cdtype_from_real(context.dtype)
            if raw_state.is_complex()
            else context.dtype
        ),
    )
    energy = as_1d_tensor(
        E_MeV,
        name="E_MeV",
        device=context.device,
        dtype=context.dtype,
    )
    profile = build_earth_profile(
        earth_profile,
        params=config.earth,
        context=context,
    )
    return state, energy, profile


@torch.no_grad()
def propagate_earth_to_detector(
    incident_state: torch.Tensor,
    *,
    E_MeV: TensorLike,
    config: PropagationConfig,
    incident_basis: Literal["flavour", "mass"] = "flavour",
    earth_profile: Optional[EarthProfile] = None,
    eta: Optional[TensorLike] = None,
    return_diagnostics: bool = False,
) -> EarthDetectorResult:
    """Propagate an incident state over an explicit or configured eta grid.

    With ``return_diagnostics=True`` the analytical method also stores the
    first-order validity diagnostics in ``result.perturbative_diagnostics``.
    Numerical propagation rejects that option because it has no explicit
    perturbative correction.

    ``config.earth.chunk_eta`` is honoured here the same way
    ``propagate_earth_to_detector_exposure``/``earth_probability_exposure``
    already do: it splits the eta axis into sub-batches evaluated one at a
    time (``None`` or a non-positive value evaluates the full grid at once),
    to control memory usage for large energy-angle grids. Previously this
    field had no effect at all through this entry point -- only through the
    ``_exposure`` variant.
    """
    state, energy, profile = _earth_inputs(
        incident_state, E_MeV, config, earth_profile
    )
    eta_grid, exposure, _ = prepare_nadir_exposure(
        eta,
        exposure=config.exposure,
        context=config.runtime,
    )
    # The low-level evolutor also accepts paired one-dimensional E/eta
    # samples. A pipeline grid is instead always the Cartesian product, so
    # make both axes explicit even when N_E == N_eta.
    energy_grid = energy[:, None]
    eta_evaluation_grid = eta_grid[None, :]

    n_eta = eta_grid.numel()
    chunk_eta = config.earth.chunk_eta
    if chunk_eta is None or chunk_eta <= 0:
        chunk_eta = n_eta

    chunks = []
    diagnostic_chunks = []
    for start in range(0, n_eta, chunk_eta):
        result = earth_probability_state(
                nustate=state,
                profile_earth=profile,
                oscillation=config.oscillation,
                E_MeV=energy_grid,
                eta=eta_evaluation_grid[:, start:start + chunk_eta],
                depth_m=config.detector_depth_m,
                method=config.earth.method,
                massbasis=incident_basis == "mass",
                nsteps=config.earth.nsteps,
                ode_method=config.earth.ode_method,
                context=config.runtime,
                reunitarize=config.reunitarize_earth,
                analytic_eigenvalues=config.analytic_eigenvalues,
                return_diagnostics=return_diagnostics,
            )
        if return_diagnostics:
            probability_chunk, diagnostic_chunk = result
            chunks.append(probability_chunk)
            diagnostic_chunks.append(diagnostic_chunk)
        else:
            chunks.append(cast(torch.Tensor, result))
    probabilities = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
    diagnostics = None
    if return_diagnostics:
        diagnostics = EarthPerturbativeDiagnostics(
            *(
                torch.cat([getattr(item, field) for item in diagnostic_chunks], dim=1)
                for field in (
                    "max_first_order_norm",
                    "accumulated_first_order_norm",
                    "max_probability_correction",
                    "unitarity_defect",
                    "validity_code",
                )
            )
        )

    return EarthDetectorResult(
        incident_state=state,
        incident_basis=incident_basis,
        E_MeV=energy,
        profile=profile,
        eta=eta_grid,
        exposure=exposure,
        probabilities_eta=probabilities,
        probabilities_exposure=None,
        perturbative_diagnostics=diagnostics,
    )


@torch.no_grad()
def propagate_earth_to_detector_exposure(
    incident_state: torch.Tensor,
    *,
    E_MeV: TensorLike,
    config: PropagationConfig,
    incident_basis: Literal["flavour", "mass"] = "flavour",
    earth_profile: Optional[EarthProfile] = None,
    return_diagnostics: bool = False,
) -> EarthDetectorResult:
    """Propagate a state and average over the configured exposure.

    Requested perturbative diagnostics retain the worst value over the
    exposure angles for every energy; they are not exposure-averaged.
    """
    state, energy, profile = _earth_inputs(
        incident_state, E_MeV, config, earth_profile
    )
    exposure_result = earth_probability_exposure(
        nustate=state,
        profile_earth=profile,
        oscillation=config.oscillation,
        E_MeV=energy,
        depth_m=config.detector_depth_m,
        method=config.earth.method,
        massbasis=incident_basis == "mass",
        exposure=config.exposure,
        context=config.runtime,
        chunk_eta=config.earth.chunk_eta,
        reunitarize=config.reunitarize_earth,
        nsteps=config.earth.nsteps,
        ode_method=config.earth.ode_method,
        analytic_eigenvalues=config.analytic_eigenvalues,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        probabilities, diagnostics = exposure_result
    else:
        probabilities = exposure_result
        diagnostics = None
    return EarthDetectorResult(
        incident_state=state,
        incident_basis=incident_basis,
        E_MeV=energy,
        profile=profile,
        eta=None,
        exposure=None,
        probabilities_eta=None,
        probabilities_exposure=probabilities,
        perturbative_diagnostics=diagnostics,
    )
