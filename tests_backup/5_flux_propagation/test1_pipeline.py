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
tests for tpeanuts.flux_propagation.pipeline.
"""



from pathlib import Path

import torch

PACKAGE_DIR = Path(__file__).resolve().parents[3]


from tpeanuts.flux_propagation.pipeline_atmosphere import (
    integrate_initial_and_surface_fluxes,
    integrate_height_and_sum_flavours,
    propagate_atmosphere_coherent,
    propagate_earth_coherent,
    select_particle_angle_flux,
)
from tpeanuts.io.io_flux_propagation import (
    aggregate_detector_conversion_by_mode,
    aggregate_detector_flux_by_mode,
)
from tpeanuts.core.pmns import PMNS
from tpeanuts.io.io_earth import load_earth_density_from_csv
from tpeanuts.io.io_atmosphere import load_directory
from tpeanuts.util.test_utils import (
    assert_true,
    assert_close,
    run_test_suite,
)


DEVICE = "cpu"
DTYPE = torch.float64
mceq_run1_DIR = PACKAGE_DIR / "data" / "flux" / "mceq_run1"
EARTH_DENSITY_FILE = PACKAGE_DIR / "data" / "density" / "earth_density.csv"

RUN_REAL_mceq_run1_analysis = True

RUN1_NEUTRINO_PARTICLES = ("nue", "numu", "nutau")
RUN1_ANTINEUTRINO_PARTICLES = ("antinue", "antinumu", "antinutau")
RUN1_FINAL_FLAVOURS = ("nue", "numu", "nutau")
RUN1_PARTICLE_LABELS = {
    "nue": r"$\nu_e$",
    "numu": r"$\nu_\mu$",
    "nutau": r"$\nu_\tau$",
    "antinue": r"$\bar{\nu}_e$",
    "antinumu": r"$\bar{\nu}_\mu$",
    "antinutau": r"$\bar{\nu}_\tau$",
}

RUN1_THETA_MAX_COUNT = None
RUN1_ENERGY_MAX_COUNT = None
RUN1_HEIGHT_MAX_COUNT = None
RUN1_atmosphere_STEPS = 200
RUN1_TRAJECTORY_STEPS = 200


DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3


def make_flux_data():
    E = torch.tensor([1.0, 10.0], dtype=DTYPE)
    h = torch.tensor([0.0, 10.0, 20.0], dtype=DTYPE)
    theta = torch.tensor([0.0, 30.0], dtype=DTYPE)
    alpha = torch.tensor([0.0, 30.0], dtype=DTYPE)

    phi = torch.ones((2, 2, 3), dtype=DTYPE)
    phi[1] = 2.0

    return {
        "numu": {
            "paths": ["a.pt", "b.pt"],
            "metadata": [{"particle": "numu"}, {"particle": "numu"}],
            "entries": [{}, {}],
            "E_grid_GeV": E,
            "E_grid": E,
            "theta_grid_deg": theta,
            "alpha_grid_deg": alpha,
            "h_grid_km": h,
            "phi_E_theta_h": phi,
            "phi_E_theta": torch.trapezoid(phi, x=h, dim=2),
            "f_theta_E_h": torch.ones_like(phi) / 20.0,
        },
        "nue": {
            "paths": ["c.pt", "d.pt"],
            "metadata": [{"particle": "nue"}, {"particle": "nue"}],
            "entries": [{}, {}],
            "E_grid_GeV": E,
            "E_grid": E,
            "theta_grid_deg": theta,
            "alpha_grid_deg": alpha,
            "h_grid_km": h,
            "phi_E_theta_h": 0.5 * phi,
            "phi_E_theta": torch.trapezoid(0.5 * phi, x=h, dim=2),
            "f_theta_E_h": torch.ones_like(phi) / 20.0,
        },
    }


def test_select_particle_angle_flux():
    data = make_flux_data()

    selected = select_particle_angle_flux(
        data,
        "numu",
        alpha_deg=30.0,
        device=DEVICE,
        dtype=DTYPE,
    )

    print("selected keys:", selected.keys())

    assert_true(selected["particle"] == "numu")
    assert_true(selected["angle_index"] == 1)
    assert_close(selected["theta_deg"], 30.0)
    assert_close(selected["alpha_deg"], 30.0)
    assert_true(selected["phi_Eh"].shape == (2, 3))


def test_integrate_height_and_sum_flavours():
    data = make_flux_data()
    selected_numu = select_particle_angle_flux(data, "numu", alpha_deg=0.0)
    selected_nue = select_particle_angle_flux(data, "nue", alpha_deg=0.0)

    probs_numu = torch.zeros((2, 3, 3), dtype=DTYPE)
    probs_nue = torch.zeros((2, 3, 3), dtype=DTYPE)
    probs_numu[..., 1] = 1.0
    probs_nue[..., 0] = 1.0

    states_numu = torch.zeros((2, 3, 3), dtype=torch.complex128)
    states_nue = torch.zeros((2, 3, 3), dtype=torch.complex128)
    states_numu[..., 1] = 1.0 + 0.0j
    states_nue[..., 0] = 1.0 + 0.0j

    result = integrate_height_and_sum_flavours(
        {
            "numu": {
                "selected": selected_numu,
                "probabilities_beta_to_i": probs_numu,
                "detector_states": states_numu,
            },
            "nue": {
                "selected": selected_nue,
                "probabilities_beta_to_i": probs_nue,
                "detector_states": states_nue,
            },
        },
        device=DEVICE,
        dtype=DTYPE,
    )

    total = result["detector_flux_total_Ei"]

    print("total flux:", total)

    assert_true(total.shape == (2, 3))
    assert_close(float(total[0, 0].item()), 10.0)
    assert_close(float(total[0, 1].item()), 20.0)
    assert_close(float(total[0, 2].item()), 0.0)


def test_integrate_initial_and_surface_fluxes():
    data = make_flux_data()
    selected_numu = select_particle_angle_flux(data, "numu", alpha_deg=0.0)
    selected_nue = select_particle_angle_flux(data, "nue", alpha_deg=0.0)

    surface_numu = torch.zeros((2, 3, 3), dtype=DTYPE)
    surface_nue = torch.zeros((2, 3, 3), dtype=DTYPE)
    surface_numu[..., 1] = 1.0
    surface_nue[..., 0] = 1.0

    result = integrate_initial_and_surface_fluxes(
        {
            "numu": selected_numu,
            "nue": selected_nue,
        },
        {
            "numu": {"surface_probabilities": surface_numu},
            "nue": {"surface_probabilities": surface_nue},
        },
        device=DEVICE,
        dtype=DTYPE,
    )

    print("initial numu flux:", result["initial_flux_by_beta"]["numu"])
    print("surface numu flux:", result["surface_flux_by_beta"]["numu"])

    assert_true(result["initial_flux_by_beta"]["numu"].shape == (2, 3))
    assert_close(float(result["initial_flux_by_beta"]["numu"][0, 1].item()), 20.0)
    assert_close(float(result["surface_flux_by_beta"]["nue"][0, 0].item()), 10.0)


def make_detector_entry(particle, alpha_deg, theta_deg):
    alpha_value = float(alpha_deg)
    return {
        "particle": particle,
        "alpha_deg": torch.tensor(alpha_value, dtype=DTYPE),
        "theta_deg": torch.tensor(float(theta_deg), dtype=DTYPE),
        "E_grid_GeV": torch.tensor([1.0, 2.0], dtype=DTYPE),
        "h_grid_km": torch.tensor([0.0, 1.0], dtype=DTYPE),
        "initial_flux_Ei": torch.full((2, 3), alpha_value, dtype=DTYPE),
        "surface_flux_Ei": torch.full((2, 3), alpha_value, dtype=DTYPE),
        "detector_flux_Ei": torch.full((2, 3), alpha_value, dtype=DTYPE),
        "source_flux_E": torch.full((2,), alpha_value, dtype=DTYPE),
        "detector_probability_Ei": torch.full((2, 3), alpha_value, dtype=DTYPE),
        "surface_probability_Ei": torch.full((2, 3), alpha_value, dtype=DTYPE),
    }


def test_detector_aggregates_sort_by_alpha():
    data = {
        "nue": [
            make_detector_entry("nue", 30.0, 10.0),
            make_detector_entry("nue", 10.0, 20.0),
            make_detector_entry("nue", 20.0, 15.0),
        ],
    }

    flux = aggregate_detector_flux_by_mode(data)["nu"]
    conversion = aggregate_detector_conversion_by_mode(data)["nu"]

    expected = torch.tensor([10.0, 20.0, 30.0], dtype=DTYPE)

    print("sorted detector alpha:", flux["alpha_grid_deg"])
    print("aligned detector flux:", flux["detector_flux_alpha_Ei"][:, 0, 0])

    assert_close(flux["alpha_grid_deg"], expected)
    assert_close(flux["detector_flux_alpha_Ei"][:, 0, 0], expected)
    assert_close(conversion["alpha_grid_deg"], expected)


def run_pipeline_tests(verbose_traceback=False):
    tests = [
        test_select_particle_angle_flux,
        test_integrate_height_and_sum_flavours,
        test_integrate_initial_and_surface_fluxes,
        test_detector_aggregates_sort_by_alpha,
    ]

    return run_test_suite(
        tests,
        suite_name="flux PROPAGATION PIPELINE tests",
        verbose_traceback=verbose_traceback,
    )


def _evenly_spaced_indices(n_items, max_count=None):
    if max_count is None or max_count >= n_items:
        return torch.arange(n_items, dtype=torch.long)

    return torch.linspace(
        0,
        n_items - 1,
        int(max_count),
        dtype=torch.float64,
    ).round().to(torch.long).unique()


def _thin_selected_flux(selected, *, energy_indices, height_indices):
    thinned = dict(selected)

    E = selected["E_grid_GeV"][energy_indices]
    h = selected["h_grid_km"][height_indices]
    phi_Eh = selected["phi_Eh"][energy_indices][:, height_indices]

    thinned["E_grid_GeV"] = E
    thinned["h_grid_km"] = h
    thinned["phi_Eh"] = phi_Eh
    thinned["phi_E_theta"] = torch.trapezoid(phi_Eh, x=h, dim=-1)

    if selected.get("f_Eh", None) is not None:
        thinned["f_Eh"] = selected["f_Eh"][energy_indices][:, height_indices]

    return thinned


def _make_default_pmns(*, device=DEVICE, dtype=DTYPE):
    return PMNS(
        theta12=0.59,
        theta13=0.15,
        theta23=0.78,
        delta=1.20,
        device=device,
        real_dtype=dtype,
    )


def _sort_result_by_alpha(result):
    alpha = result["alpha_grid_deg"]
    order = torch.argsort(alpha)

    sorted_result = dict(result)

    for key in [
        "theta_grid_deg",
        "alpha_grid_deg",
        "source_flux_theta_E",
        "initial_flux_total_theta_Ei",
        "surface_flux_total_theta_Ei",
        "detector_flux_theta_Ei",
        "detector_probability_theta_Ei",
    ]:
        sorted_result[key] = result[key][order]

    for container_key in [
        "stage_flux_by_particle",
        "stage_probability_by_particle",
    ]:
        sorted_container = {}
        for particle, stages in result.get(container_key, {}).items():
            sorted_container[particle] = {
                stage: values[order]
                for stage, values in stages.items()
            }
        sorted_result[container_key] = sorted_container

    return sorted_result


@torch.no_grad()
def process_mceq_run1_detector_grid(
    *,
    data_dir=mceq_run1_DIR,
    detector_depth_m=0.0,
    theta_max_count=RUN1_THETA_MAX_COUNT,
    energy_max_count=RUN1_ENERGY_MAX_COUNT,
    height_max_count=RUN1_HEIGHT_MAX_COUNT,
    atmosphere_steps=RUN1_atmosphere_STEPS,
    trajectory_steps=RUN1_TRAJECTORY_STEPS,
    matter=True,
    device=DEVICE,
    dtype=DTYPE,
    debug=True,
):
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"mceq_run1 data directory not found: {data_dir}")

    flux_data = load_directory(
        str(data_dir),
        device=device,
        dtype=dtype,
        group_by="particle",
        verbose=debug,
    )

    pmns = _make_default_pmns(device=device, dtype=dtype)
    density = load_earth_density_from_csv(
        str(EARTH_DENSITY_FILE),
        device=device,
        dtype=dtype,
    )

    dm21 = torch.tensor(DM21_EV2, device=device, dtype=dtype)
    dm3l = torch.tensor(DM3L_EV2, device=device, dtype=dtype)

    mode_specs = {
        "nu": (RUN1_NEUTRINO_PARTICLES, False),
        "antinu": (RUN1_ANTINEUTRINO_PARTICLES, True),
    }
    results = {}

    for mode_label, (particles, antinu) in mode_specs.items():
        available_particles = [
            particle
            for particle in particles
            if particle in flux_data
        ]

        if len(available_particles) == 0:
            if debug:
                print(f"[{mode_label}] No particles found in {data_dir}")
            continue

        reference = flux_data[available_particles[0]]
        theta_grid = reference["theta_grid_deg"].to(device=device, dtype=dtype)
        alpha_grid = reference.get("alpha_grid_deg", theta_grid)
        alpha_grid = alpha_grid.to(device=device, dtype=dtype)

        theta_indices = _evenly_spaced_indices(theta_grid.numel(), theta_max_count)
        energy_indices = _evenly_spaced_indices(
            reference["E_grid_GeV"].numel(),
            energy_max_count,
        )
        height_indices = _evenly_spaced_indices(
            reference["h_grid_km"].numel(),
            height_max_count,
        )

        theta_values = theta_grid[theta_indices]
        alpha_values = alpha_grid[theta_indices]
        E_values = reference["E_grid_GeV"][energy_indices].to(
            device=device,
            dtype=dtype,
        )

        n_alpha = theta_indices.numel()
        n_E = E_values.numel()

        source_flux_theta_E = torch.zeros((n_alpha, n_E), device=device, dtype=dtype)
        initial_flux_total = torch.zeros((n_alpha, n_E, 3), device=device, dtype=dtype)
        surface_flux_total = torch.zeros_like(initial_flux_total)
        detector_flux_total = torch.zeros_like(initial_flux_total)
        detector_probability_theta_Ei = torch.zeros_like(initial_flux_total)

        stage_flux_by_particle = {
            particle: {
                "initial": torch.empty((n_alpha, n_E, 3), device=device, dtype=dtype),
                "surface": torch.empty((n_alpha, n_E, 3), device=device, dtype=dtype),
                "detector": torch.empty((n_alpha, n_E, 3), device=device, dtype=dtype),
            }
            for particle in available_particles
        }
        stage_probability_by_particle = {
            particle: {
                "initial": torch.empty((n_alpha, n_E, 3), device=device, dtype=dtype),
                "surface": torch.empty((n_alpha, n_E, 3), device=device, dtype=dtype),
                "detector": torch.empty((n_alpha, n_E, 3), device=device, dtype=dtype),
            }
            for particle in available_particles
        }

        for i_out, angle_index in enumerate(theta_indices.tolist()):
            selected_by_particle = {}
            atmosphere_by_particle = {}
            propagated = {}
            source_flux_total_E = None

            if debug:
                print(
                    f"[{mode_label}] angle {i_out + 1}/{n_alpha} | "
                    f"alpha={float(alpha_grid[angle_index].item()):.3f} deg | "
                    f"theta={float(theta_grid[angle_index].item()):.3f} deg"
                )

            for particle in available_particles:
                selected = select_particle_angle_flux(
                    flux_data,
                    particle,
                    angle_index=angle_index,
                    device=device,
                    dtype=dtype,
                )
                selected = _thin_selected_flux(
                    selected,
                    energy_indices=energy_indices,
                    height_indices=height_indices,
                )
                selected_by_particle[particle] = selected

                atmosphere = propagate_atmosphere_coherent(
                    selected,
                    pmns,
                    dm21,
                    dm3l,
                    detector_depth_m=detector_depth_m,
                    antinu=antinu,
                    matter=matter,
                    atmosphere_n_steps=atmosphere_steps,
                    trajectory_steps=trajectory_steps,
                    device=device,
                    dtype=dtype,
                    debug=False,
                )
                atmosphere_by_particle[particle] = atmosphere

                detector = propagate_earth_coherent(
                    atmosphere,
                    pmns,
                    dm21,
                    dm3l,
                    detector_depth_m=detector_depth_m,
                    density=density,
                    antinu=antinu,
                    device=device,
                    dtype=dtype,
                    debug=False,
                )
                propagated[particle] = detector

                source_flux = torch.trapezoid(
                    selected["phi_Eh"],
                    x=selected["h_grid_km"],
                    dim=-1,
                )
                source_flux_total_E = (
                    source_flux
                    if source_flux_total_E is None
                    else source_flux_total_E + source_flux
                )

            source_surface = integrate_initial_and_surface_fluxes(
                selected_by_particle,
                atmosphere_by_particle,
                device=device,
                dtype=dtype,
            )
            integrated = integrate_height_and_sum_flavours(
                propagated,
                device=device,
                dtype=dtype,
            )

            source_flux_theta_E[i_out] = source_flux_total_E
            detector_flux_total[i_out] = integrated["detector_flux_total_Ei"]
            denom = source_flux_total_E[:, None].clamp_min(torch.finfo(dtype).tiny)
            detector_probability_theta_Ei[i_out] = detector_flux_total[i_out] / denom

            initial_flux_total[i_out] = source_surface["initial_flux_total_Ei"]
            surface_flux_total[i_out] = source_surface["surface_flux_total_Ei"]

            for particle in available_particles:
                stage_flux_by_particle[particle]["initial"][i_out] = (
                    source_surface["initial_flux_by_beta"][particle]
                )
                stage_flux_by_particle[particle]["surface"][i_out] = (
                    source_surface["surface_flux_by_beta"][particle]
                )
                stage_flux_by_particle[particle]["detector"][i_out] = (
                    integrated["integrated_flux_by_beta"][particle]
                )

                stage_probability_by_particle[particle]["initial"][i_out] = (
                    source_surface["initial_probability_by_beta"][particle]
                )
                stage_probability_by_particle[particle]["surface"][i_out] = (
                    source_surface["surface_probability_by_beta"][particle]
                )

                source_particle = torch.sum(
                    source_surface["initial_flux_by_beta"][particle],
                    dim=-1,
                )
                stage_probability_by_particle[particle]["detector"][i_out] = (
                    integrated["integrated_flux_by_beta"][particle]
                    / source_particle[:, None].clamp_min(torch.finfo(dtype).tiny)
                )

        mode_result = {
            "particles": available_particles,
            "theta_grid_deg": theta_values.detach().cpu(),
            "alpha_grid_deg": alpha_values.detach().cpu(),
            "E_grid_GeV": E_values.detach().cpu(),
            "source_flux_theta_E": source_flux_theta_E.detach().cpu(),
            "initial_flux_total_theta_Ei": initial_flux_total.detach().cpu(),
            "surface_flux_total_theta_Ei": surface_flux_total.detach().cpu(),
            "detector_flux_theta_Ei": detector_flux_total.detach().cpu(),
            "detector_probability_theta_Ei": detector_probability_theta_Ei.detach().cpu(),
            "stage_flux_by_particle": {
                particle: {
                    stage: values.detach().cpu()
                    for stage, values in stages.items()
                }
                for particle, stages in stage_flux_by_particle.items()
            },
            "stage_probability_by_particle": {
                particle: {
                    stage: values.detach().cpu()
                    for stage, values in stages.items()
                }
                for particle, stages in stage_probability_by_particle.items()
            },
        }

        results[mode_label] = _sort_result_by_alpha(mode_result)

    return results


def _paper_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.1,
            "lines.linewidth": 1.8,
            "mathtext.fontset": "stix",
            "font.family": "serif",
        }
    )


def _particle_label(particle):
    return RUN1_PARTICLE_LABELS.get(particle, particle)


def _energy_index(E, requested=None):
    if requested is None:
        return int(E.numel() // 2)

    return int(max(0, min(int(requested), E.numel() - 1)))


def _stage_flux(result, stage):
    if stage == "initial":
        return result["initial_flux_total_theta_Ei"]

    if stage == "surface":
        return result["surface_flux_total_theta_Ei"]

    if stage == "detector":
        return result["detector_flux_theta_Ei"]

    raise ValueError("stage must be 'initial', 'surface', or 'detector'.")


def _stage_probability_total(result, stage):
    flux = _stage_flux(result, stage)
    source = result["source_flux_theta_E"].clamp_min(torch.finfo(flux.dtype).tiny)
    return flux / source[..., None]


def plot_particle_spectra_by_alpha(
    results,
    *,
    mode_label="nu",
    alpha_count=5,
    show=True,
):
    import matplotlib.pyplot as plt

    _paper_style()
    figures = []
    result = results[mode_label]
    E = result["E_grid_GeV"]
    alpha = result["alpha_grid_deg"]
    alpha_indices = _evenly_spaced_indices(alpha.numel(), alpha_count)

    for particle in result["particles"]:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        flux = result["stage_flux_by_particle"][particle]["initial"]
        source_flux = torch.sum(flux, dim=-1)
        particle_label = _particle_label(particle)

        for idx in alpha_indices.tolist():
            ax.loglog(
                E,
                source_flux[idx].clamp_min(torch.finfo(source_flux.dtype).tiny),
                marker="o",
                markersize=3.5,
                label=rf"$\alpha={float(alpha[idx]):.1f}^\circ$",
            )

        ax.set_xlabel(r"Energy $E$ [GeV]")
        ax.set_ylabel(r"Height-integrated origin flux")
        ax.set_title(rf"{mode_label}: {particle_label} origin spectrum")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        figures.append(fig)

    if show:
        plt.show()

    return figures


def plot_stage_probabilities_by_alpha(
    results,
    *,
    mode_label="nu",
    energy_index=None,
    show=True,
):
    import matplotlib.pyplot as plt

    _paper_style()
    figures = []
    result = results[mode_label]
    E = result["E_grid_GeV"]
    alpha = result["alpha_grid_deg"]
    iE = _energy_index(E, energy_index)
    stages = [
        ("initial", "Origin"),
        ("surface", "Surface"),
        ("detector", "Detector"),
    ]
    flavour_colors = {
        "nue": "tab:blue",
        "numu": "tab:orange",
        "nutau": "tab:green",
    }

    for stage, stage_title in stages:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        probabilities = _stage_probability_total(result, stage)

        for i_flavour, flavour in enumerate(RUN1_FINAL_FLAVOURS):
            flavour_label = _particle_label(flavour)
            ax.plot(
                alpha,
                probabilities[:, iE, i_flavour],
                color=flavour_colors[flavour],
                marker="o",
                markersize=3.5,
                label=flavour_label,
            )

        ax.set_xlabel(r"Detector angle $\alpha$ [deg]")
        ax.set_ylabel(r"flux-weighted probability")
        ax.set_title(
            f"{mode_label}: {stage_title} probabilities, "
            f"E={float(E[iE]):.3g} GeV"
        )
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        figures.append(fig)

    if show:
        plt.show()

    return figures


def plot_stage_fluxes_by_alpha(
    results,
    *,
    mode_label="nu",
    energy_index=None,
    show=True,
):
    import matplotlib.pyplot as plt

    _paper_style()
    figures = []
    result = results[mode_label]
    E = result["E_grid_GeV"]
    alpha = result["alpha_grid_deg"]
    iE = _energy_index(E, energy_index)
    stages = [
        ("initial", "Origin"),
        ("surface", "Surface"),
        ("detector", "Detector"),
    ]
    flavour_colors = {
        "nue": "tab:blue",
        "numu": "tab:orange",
        "nutau": "tab:green",
    }

    for stage, stage_title in stages:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        flux = _stage_flux(result, stage)

        for i_flavour, flavour in enumerate(RUN1_FINAL_FLAVOURS):
            flavour_label = _particle_label(flavour)
            ax.semilogy(
                alpha,
                flux[:, iE, i_flavour].clamp_min(torch.finfo(flux.dtype).tiny),
                color=flavour_colors[flavour],
                marker="o",
                markersize=3.5,
                label=flavour_label,
            )

        ax.set_xlabel(r"Detector angle $\alpha$ [deg]")
        ax.set_ylabel(r"Height-integrated flux")
        ax.set_title(
            f"{mode_label}: {stage_title} fluxes, "
            f"E={float(E[iE]):.3g} GeV"
        )
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        figures.append(fig)

    if show:
        plt.show()

    return figures


def plot_e_mu_probabilities_by_alpha(
    results,
    *,
    mode_label="nu",
    energy_index=None,
    show=True,
):
    import matplotlib.pyplot as plt

    _paper_style()
    result = results[mode_label]
    E = result["E_grid_GeV"]
    alpha = result["alpha_grid_deg"]
    iE = _energy_index(E, energy_index)
    stages = [
        ("initial", "Origin", "--"),
        ("surface", "Surface", "-."),
        ("detector", "Detector", "-"),
    ]
    flavours = [
        (0, "nue", "tab:blue"),
        (1, "numu", "tab:orange"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for stage, stage_label, linestyle in stages:
        probabilities = _stage_probability_total(result, stage)
        for i_flavour, flavour, color in flavours:
            flavour_label = _particle_label(flavour)
            ax.plot(
                alpha,
                probabilities[:, iE, i_flavour],
                linestyle=linestyle,
                color=color,
                marker="o",
                markersize=3.2,
                label=rf"{stage_label} {flavour_label}",
            )

    ax.set_xlabel(r"Detector angle $\alpha$ [deg]")
    ax.set_ylabel(r"flux-weighted probability")
    ax.set_title(
        f"{mode_label}: electron and muon flavour probabilities, "
        f"E={float(E[iE]):.3g} GeV"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    if show:
        plt.show()

    return [fig]


def plot_mceq_run1_publication_figures(
    results,
    *,
    mode_label="nu",
    energy_index=None,
    show=True,
):
    figures = []
    figures.extend(
        plot_particle_spectra_by_alpha(
            results,
            mode_label=mode_label,
            show=False,
        )
    )
    figures.extend(
        plot_stage_probabilities_by_alpha(
            results,
            mode_label=mode_label,
            energy_index=energy_index,
            show=False,
        )
    )
    figures.extend(
        plot_stage_fluxes_by_alpha(
            results,
            mode_label=mode_label,
            energy_index=energy_index,
            show=False,
        )
    )
    figures.extend(
        plot_e_mu_probabilities_by_alpha(
            results,
            mode_label=mode_label,
            energy_index=energy_index,
            show=False,
        )
    )

    if show:
        import matplotlib.pyplot as plt
        plt.show()

    return figures


def run_mceq_run1_detector_analysis(**kwargs):
    results = process_mceq_run1_detector_grid(**kwargs)
    figures = []

    for mode_label in results:
        figures.extend(
            plot_mceq_run1_publication_figures(
                results,
                mode_label=mode_label,
                show=False,
            )
        )

    import matplotlib.pyplot as plt
    plt.show()

    return results, figures


if __name__ == "__main__":
    run_pipeline_tests(verbose_traceback=True)

    if RUN_REAL_mceq_run1_analysis:
        run_mceq_run1_detector_analysis(debug=True)
