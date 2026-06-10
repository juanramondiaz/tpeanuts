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
Spyder-compatible tests for the new torch-based io.py.

No pytest required.
"""



import os
from pathlib import Path
import shutil

import torch

from tpeanuts.io.io_atmosphere import (
    angle_to_filename,
    build_angle_output_path,
    build_output_path,
    build_result_metadata,
    cast_tensor_tree,
    ensure_torch_extension,
    list_torch_files,
    load_directory,
    load_phi_Eh_alpha_theta_from_config,
    load_phi_Eh_theta_result,
    OutputConfig,
    safe_filename_name,
    save_phi_Eh_theta_result,
)

from tpeanuts.util.test_utils import (
    assert_true,
    assert_close,
    assert_raises,
    run_test_suite,
)


DEVICE = "cpu"
DTYPE = torch.float64

NOTEBOOK_STEM = Path(__file__).stem
DEFAULT_OUTPUT_ROOT = Path(r"V:\output")
OUTPUT_ROOT = Path(os.environ.get("TPEANUTS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
OUTPUT_TEST_ROOT = Path(OUTPUT_ROOT / "test")
OUTPUT_DIR = OUTPUT_TEST_ROOT / "mceq" / NOTEBOOK_STEM


def make_fake_result(
    theta_deg=30.0,
    alpha_deg=None,
    particle="numu",
    flavour_name="numu",
    n_E=5,
    n_h=20,
    n_X=10,
):
    E_grid = torch.logspace(-1, 2, n_E, device=DEVICE, dtype=DTYPE)
    h_grid = torch.linspace(0.0, 80.0, n_h, device=DEVICE, dtype=DTYPE)
    X_grid = torch.linspace(1.0, 1030.0, n_X, device=DEVICE, dtype=DTYPE)

    f_Eh = torch.ones((n_E, n_h), device=DEVICE, dtype=DTYPE)
    f_Eh = f_Eh / torch.trapezoid(f_Eh, x=h_grid, dim=1)[:, None]

    phi_E_obs = E_grid ** (-2.0)
    phi_Eh = phi_E_obs[:, None] * f_Eh

    flux_XE = torch.ones((n_X, n_E), device=DEVICE, dtype=DTYPE)

    result = {
        "theta_deg": torch.tensor(theta_deg, device=DEVICE, dtype=DTYPE),
        "particle": particle,
        "flavour_name": flavour_name,
        "E_grid_GeV": E_grid,
        "X_grid_gcm2": X_grid,
        "h_grid_km": h_grid,
        "flux_XE": flux_XE,
        "flux_smooth_XE": flux_XE.clone(),
        "dPhi_dX_XE": torch.zeros_like(flux_XE),
        "X_of_h_gcm2": torch.linspace(1030.0, 0.0, n_h, device=DEVICE, dtype=DTYPE),
        "dXdh_gcm2_per_km": torch.ones(n_h, device=DEVICE, dtype=DTYPE),
        "source_Eh": f_Eh.clone(),
        "f_Eh": f_Eh,
        "phi_E_obs": phi_E_obs,
        "phi_Eh": phi_Eh,
        "build_time_sec": 1.23,
    }

    if alpha_deg is not None:
        result["alpha_deg"] = torch.tensor(alpha_deg, device=DEVICE, dtype=DTYPE)

    return result


def make_temp_output_config(
    filename="phi_E_theta_h_from_mceq_profiles.pt",
    overwrite=True,
    dtype=torch.float64,
):
    tmpdir = OUTPUT_DIR

    config = OutputConfig(
        output_dir=tmpdir,
        filename=filename,
        dtype=dtype,
        compressed=True,
        overwrite=overwrite,
        save_intermediate=True,
    )

    return config, tmpdir


# ============================================================
# Filename tests
# ============================================================

def test_safe_filename_name():
    name = "nu/mu+ test:case"
    safe = safe_filename_name(name)

    print("original:", name)
    print("safe    :", safe)

    assert_true("/" not in safe)
    assert_true("+" not in safe)
    assert_true(" " not in safe)
    assert_true(":" not in safe)


def test_angle_to_filename():
    theta = 30.125
    safe = angle_to_filename(theta)

    print("theta:", theta)
    print("safe :", safe)

    assert_true(safe == "30p125")


def test_ensure_torch_extension_from_empty_extension():
    filename = ensure_torch_extension("flux_output")

    print("filename:", filename)

    assert_true(filename == "flux_output.pt")


def test_ensure_torch_extension_keeps_pt():
    filename = ensure_torch_extension("flux_output.pt")

    print("filename:", filename)

    assert_true(filename == "flux_output.pt")


def test_ensure_torch_extension_replaces_npz():
    filename = ensure_torch_extension("flux_output.npz")

    print("filename:", filename)

    assert_true(filename == "flux_output.pt")


def test_build_angle_output_path_with_theta():
    config = OutputConfig(
        output_dir=OUTPUT_DIR,
        filename="flux_output.npz",
    )

    path = build_angle_output_path(
        output_config=config,
        flavour_name="numu",
        theta_deg=45.0,
    )
    filename = os.path.basename(path)

    print("path:", path)
    print("filename:", filename)

    assert_true(path == os.path.join(OUTPUT_DIR, filename))
    assert_true(filename.endswith(".pt"))
    assert_true("numu" in filename)
    assert_true("theta_45p000deg" in filename)


def test_build_angle_output_path_with_alpha_theta():
    config = OutputConfig(
        output_dir=OUTPUT_DIR,
        filename="flux.npz",
    )

    path = build_angle_output_path(
        output_config=config,
        flavour_name="nue",
        particle="total_nue",
        alpha_deg=41.25,
        theta_deg=43.5,
    )
    filename = os.path.basename(path)

    print("path:", path)
    print("filename:", filename)

    assert_true(
        path == os.path.join(
            OUTPUT_DIR,
            "flux_nue_total_nue_alpha_41p250deg_theta_43p500deg.pt",
        )
    )
    assert_true(filename.endswith(".pt"))
    assert_true("nue" in filename)
    assert_true("total_nue" in filename)
    assert_true("alpha_41p250deg" in filename)
    assert_true("theta_43p500deg" in filename)


def test_build_output_path():
    path = build_output_path(
        output_dir=OUTPUT_DIR,
        filename="file.pt",
    )

    print("path:", path)

    assert_true(path == os.path.join(OUTPUT_DIR, "file.pt"))


# ============================================================
# Metadata tests
# ============================================================

def test_build_result_metadata():
    result = make_fake_result()

    metadata = build_result_metadata(
        result,
        flavour_name="numu",
    )

    print("metadata keys:", metadata.keys())
    print("tensor shapes:", metadata["tensor_shapes"])

    assert_true(metadata["particle"] == "numu")
    assert_true(metadata["flavour_name"] == "numu")
    assert_close(metadata["theta_deg"], 30.0)
    assert_true(metadata["format"] == "torch")
    assert_true(metadata["extension"] == ".pt")
    assert_true("tensor_shapes" in metadata)
    assert_true("phi_Eh" in metadata["tensor_shapes"])


def test_build_result_metadata_with_alpha_theta():
    result = make_fake_result(
        alpha_deg=41.25,
        theta_deg=43.5,
        particle="total_nue",
        flavour_name="nue",
    )

    metadata = build_result_metadata(
        result,
        flavour_name="nue",
    )

    print("metadata:", metadata)

    assert_true(metadata["particle"] == "total_nue")
    assert_true(metadata["flavour_name"] == "nue")
    assert_close(metadata["alpha_deg"], 41.25)
    assert_close(metadata["theta_deg"], 43.5)
    assert_true(metadata["angle_units"] == "deg")


def test_cast_tensor_tree_dtype_and_device():
    obj = {
        "a": torch.ones(3, dtype=torch.float64),
        "b": {
            "c": torch.ones(2, dtype=torch.float64),
            "d": "text",
        },
        "e": [torch.ones(1, dtype=torch.float64)],
    }

    out = cast_tensor_tree(
        obj,
        dtype=torch.float32,
        device="cpu",
    )

    print("dtype a:", out["a"].dtype)
    print("dtype b.c:", out["b"]["c"].dtype)
    print("dtype e[0]:", out["e"][0].dtype)

    assert_true(out["a"].dtype == torch.float32)
    assert_true(out["b"]["c"].dtype == torch.float32)
    assert_true(out["e"][0].dtype == torch.float32)
    assert_true(out["b"]["d"] == "text")


# ============================================================
# Save/load tests
# ============================================================

def test_save_phi_Eh_theta_result_creates_pt_file():
    config, tmpdir = make_temp_output_config()

    try:
        result = make_fake_result()

        path = save_phi_Eh_theta_result(
            result=result,
            output_config=config,
            flavour_name="numu",
        )

        print("saved path:", path)

        assert_true(os.path.exists(path))
        assert_true(path.endswith(".pt"))

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_save_with_npz_filename_still_creates_pt_file():
    config, tmpdir = make_temp_output_config(
        filename="phi_E_theta_h_from_mceq_profiles.npz"
    )

    try:
        result = make_fake_result()

        path = save_phi_Eh_theta_result(
            result=result,
            output_config=config,
            flavour_name="numu",
        )

        print("saved path:", path)

        assert_true(os.path.exists(path))
        assert_true(path.endswith(".pt"))

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_save_requires_theta_deg_key():
    config, tmpdir = make_temp_output_config()

    try:
        result = make_fake_result()
        result.pop("theta_deg")

        assert_raises(
            KeyError,
            save_phi_Eh_theta_result,
            result,
            config,
            flavour_name="numu",
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_phi_Eh_theta_result():
    config, tmpdir = make_temp_output_config()

    try:
        result = make_fake_result()

        path = save_phi_Eh_theta_result(
            result=result,
            output_config=config,
            flavour_name="numu",
        )

        data = load_phi_Eh_theta_result(
            input_path=path,
            map_location="cpu",
            dtype=DTYPE,
            device=DEVICE,
        )

        print("loaded keys:", data.keys())

        assert_true("E_grid_GeV" in data)
        assert_true("h_grid_km" in data)
        assert_true("phi_Eh" in data)
        assert_true("metadata" in data)
        assert_true("metadata_json" in data)

        assert_true(isinstance(data["E_grid_GeV"], torch.Tensor))
        assert_true(data["phi_Eh"].shape == result["phi_Eh"].shape)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_save_phi_Eh_theta_with_alpha_and_load_theta_result():
    config, tmpdir = make_temp_output_config(
        filename="flux.npz",
        dtype=torch.float64,
    )

    try:
        result = make_fake_result(
            alpha_deg=41.25,
            theta_deg=43.5,
            particle="total_nue",
            flavour_name="nue",
        )

        path = save_phi_Eh_theta_result(
            result=result,
            output_config=config,
            particle="total_nue",
            alpha_deg=41.25,
            theta_deg=43.5,
            flavour_name="nue",
        )

        print("saved path:", path)

        assert_true(os.path.exists(path))
        assert_true("alpha_41p250deg" in os.path.basename(path))
        assert_true("theta_43p500deg" in os.path.basename(path))

        data = load_phi_Eh_theta_result(
            input_path=path,
            map_location="cpu",
            dtype=torch.float64,
            device="cpu",
        )

        print("loaded metadata:", data["metadata"])

        assert_true("alpha_deg" in data)
        assert_close(float(data["alpha_deg"].item()), 41.25)
        assert_close(float(data["theta_deg"].item()), 43.5)
        assert_close(data["metadata"]["alpha_deg"], 41.25)
        assert_close(data["metadata"]["theta_deg"], 43.5)
        assert_true(data["metadata"]["particle"] == "total_nue")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_phi_Eh_alpha_theta_from_config():
    config, tmpdir = make_temp_output_config(
        filename="flux.npz",
        dtype=torch.float64,
    )

    try:
        result = make_fake_result(
            alpha_deg=41.25,
            theta_deg=43.5,
            particle="total_nue",
            flavour_name="nue",
        )

        save_phi_Eh_theta_result(
            result=result,
            output_config=config,
            particle="total_nue",
            alpha_deg=41.25,
            theta_deg=43.5,
            flavour_name="nue",
        )

        data = load_phi_Eh_alpha_theta_from_config(
            output_config=config,
            particle="total_nue",
            alpha_deg=41.25,
            theta_deg=43.5,
            flavour_name="nue",
            map_location="cpu",
            dtype=torch.float64,
            device="cpu",
        )

        print("loaded path metadata:", data["metadata"])

        assert_close(float(data["alpha_deg"].item()), 41.25)
        assert_close(float(data["theta_deg"].item()), 43.5)
        assert_true(data["metadata"]["particle"] == "total_nue")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_loaded_values_match_saved_values():
    config, tmpdir = make_temp_output_config(dtype=torch.float64)

    try:
        result = make_fake_result()

        path = save_phi_Eh_theta_result(
            result=result,
            output_config=config,
            flavour_name="numu",
        )

        data = load_phi_Eh_theta_result(
            input_path=path,
            map_location="cpu",
            dtype=torch.float64,
            device="cpu",
        )

        max_diff = torch.max(
            torch.abs(data["phi_Eh"] - result["phi_Eh"])
        ).item()

        print("max difference phi_Eh:", max_diff)

        assert_close(max_diff, 0.0, atol=1.0e-14, rtol=0.0)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_loaded_dtype_can_be_cast():
    config, tmpdir = make_temp_output_config(dtype=torch.float64)

    try:
        result = make_fake_result()

        path = save_phi_Eh_theta_result(
            result=result,
            output_config=config,
            flavour_name="numu",
        )

        data = load_phi_Eh_theta_result(
            input_path=path,
            map_location="cpu",
            dtype=torch.float32,
            device="cpu",
        )

        print("loaded dtype:", data["phi_Eh"].dtype)

        assert_true(data["phi_Eh"].dtype == torch.float32)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_missing_file_raises():
    assert_raises(
        FileNotFoundError,
        load_phi_Eh_theta_result,
        "missing_file_that_does_not_exist.pt",
    )


def test_overwrite_false_returns_existing_file():
    config, tmpdir = make_temp_output_config(overwrite=False)

    try:
        result = make_fake_result()

        path1 = save_phi_Eh_theta_result(
            result=result,
            output_config=config,
            flavour_name="numu",
        )

        path2 = save_phi_Eh_theta_result(
            result=result,
            output_config=config,
            flavour_name="numu",
        )

        print("path1:", path1)
        print("path2:", path2)

        assert_true(path1 == path2)
        assert_true(os.path.exists(path2))

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_list_torch_files():
    config, tmpdir = make_temp_output_config()

    try:
        result1 = make_fake_result(theta_deg=0.0)
        result2 = make_fake_result(theta_deg=45.0)

        save_phi_Eh_theta_result(result1, config, flavour_name="numu")
        save_phi_Eh_theta_result(result2, config, flavour_name="numu")

        files = list_torch_files(tmpdir)

        print("files:", files)

        assert_true(len(files) == 2)
        assert_true(all(f.endswith(".pt") for f in files))

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_directory():
    config, tmpdir = make_temp_output_config()

    try:
        result1 = make_fake_result(theta_deg=0.0)
        result2 = make_fake_result(theta_deg=45.0)
        result3 = make_fake_result(
            theta_deg=30.0,
            particle="total_nue",
            flavour_name="nue",
        )

        save_phi_Eh_theta_result(result1, config, flavour_name="numu")
        save_phi_Eh_theta_result(result2, config, flavour_name="numu")
        save_phi_Eh_theta_result(result3, config, flavour_name="nue")

        data = load_directory(
            tmpdir,
            map_location="cpu",
            dtype=DTYPE,
            device=DEVICE,
        )

        print("loaded groups:", data.keys())
        print("numu phi shape:", data["numu"]["phi_E_theta_h"].shape)

        assert_true(len(data) == 2)
        assert_true("numu" in data)
        assert_true("total_nue" in data)
        assert_true(data["numu"]["theta_grid_deg"].shape == (2,))
        assert_true(data["numu"]["phi_E_theta_h"].shape == (2, 5, 20))
        assert_true(data["numu"]["phi_E_theta"].shape == (2, 5))
        assert_true(data["numu"]["f_theta_E_h"].shape == (2, 5, 20))
        assert_close(float(data["numu"]["theta_grid_deg"][0].item()), 0.0)
        assert_close(float(data["numu"]["theta_grid_deg"][1].item()), 45.0)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_directory_can_group_by_flavour_name():
    config, tmpdir = make_temp_output_config()

    try:
        result1 = make_fake_result(
            theta_deg=0.0,
            particle="total_numu",
            flavour_name="numu",
        )
        result2 = make_fake_result(
            theta_deg=45.0,
            particle="total_numu",
            flavour_name="numu",
        )

        save_phi_Eh_theta_result(result1, config, flavour_name="numu")
        save_phi_Eh_theta_result(result2, config, flavour_name="numu")

        data = load_directory(
            tmpdir,
            map_location="cpu",
            dtype=DTYPE,
            device=DEVICE,
            group_by="flavour_name",
        )

        print("loaded groups:", data.keys())

        assert_true(list(data.keys()) == ["numu"])
        assert_true(data["numu"]["phi_E_theta_h"].shape == (2, 5, 20))

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_directory_keeps_alpha_grid_when_available():
    config, tmpdir = make_temp_output_config(filename="flux.npz")

    try:
        result1 = make_fake_result(
            alpha_deg=10.0,
            theta_deg=11.0,
            particle="total_nue",
            flavour_name="nue",
        )
        result2 = make_fake_result(
            alpha_deg=20.0,
            theta_deg=22.0,
            particle="total_nue",
            flavour_name="nue",
        )

        save_phi_Eh_theta_result(
            result1,
            config,
            particle="total_nue",
            alpha_deg=10.0,
            theta_deg=11.0,
            flavour_name="nue",
        )
        save_phi_Eh_theta_result(
            result2,
            config,
            particle="total_nue",
            alpha_deg=20.0,
            theta_deg=22.0,
            flavour_name="nue",
        )

        data = load_directory(
            tmpdir,
            map_location="cpu",
            dtype=DTYPE,
            device=DEVICE,
        )

        print("alpha grid:", data["total_nue"]["alpha_grid_deg"])

        assert_true("alpha_grid_deg" in data["total_nue"])
        assert_close(float(data["total_nue"]["alpha_grid_deg"][0].item()), 10.0)
        assert_close(float(data["total_nue"]["alpha_grid_deg"][1].item()), 20.0)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_directory_sorts_by_alpha_when_available():
    config, tmpdir = make_temp_output_config(filename="flux.npz")

    try:
        for alpha_deg, theta_deg in [(30.0, 10.0), (10.0, 20.0), (20.0, 15.0)]:
            result = make_fake_result(
                alpha_deg=alpha_deg,
                theta_deg=theta_deg,
                particle="total_nue",
                flavour_name="nue",
            )
            result["phi_Eh"] = torch.full_like(result["phi_Eh"], alpha_deg)
            result["phi_E_obs"] = torch.full_like(result["phi_E_obs"], alpha_deg)
            result["f_Eh"] = torch.full_like(result["f_Eh"], alpha_deg)

            save_phi_Eh_theta_result(
                result,
                config,
                particle="total_nue",
                alpha_deg=alpha_deg,
                theta_deg=theta_deg,
                flavour_name="nue",
            )

        data = load_directory(
            tmpdir,
            map_location="cpu",
            dtype=DTYPE,
            device=DEVICE,
        )

        group = data["total_nue"]
        print("sorted alpha grid:", group["alpha_grid_deg"])
        print("aligned phi values:", group["phi_E_theta_h"][:, 0, 0])

        assert_close(group["alpha_grid_deg"], torch.tensor([10.0, 20.0, 30.0], dtype=DTYPE))
        assert_close(group["phi_E_theta_h"][:, 0, 0], torch.tensor([10.0, 20.0, 30.0], dtype=DTYPE))

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_list_missing_directory_raises():
    assert_raises(
        FileNotFoundError,
        list_torch_files,
        "missing_directory_that_does_not_exist",
    )


# ============================================================
# Runner
# ============================================================

def run_io_tests(verbose_traceback=False):
    tests = [
        test_safe_filename_name,
        test_angle_to_filename,
        test_ensure_torch_extension_from_empty_extension,
        test_ensure_torch_extension_keeps_pt,
        test_ensure_torch_extension_replaces_npz,
        test_build_angle_output_path_with_theta,
        test_build_angle_output_path_with_alpha_theta,
        test_build_output_path,
        test_build_result_metadata,
        test_build_result_metadata_with_alpha_theta,
        test_cast_tensor_tree_dtype_and_device,
        test_save_phi_Eh_theta_result_creates_pt_file,
        test_save_with_npz_filename_still_creates_pt_file,
        test_save_requires_theta_deg_key,
        test_load_phi_Eh_theta_result,
        test_save_phi_Eh_theta_with_alpha_and_load_theta_result,
        test_load_phi_Eh_alpha_theta_from_config,
        test_loaded_values_match_saved_values,
        test_loaded_dtype_can_be_cast,
        test_load_missing_file_raises,
        test_overwrite_false_returns_existing_file,
        test_list_torch_files,
        test_load_directory,
        test_load_directory_can_group_by_flavour_name,
        test_load_directory_keeps_alpha_grid_when_available,
        test_load_directory_sorts_by_alpha_when_available,
        test_list_missing_directory_raises,
    ]

    return run_test_suite(
        tests,
        suite_name="IO TORCH tests",
        verbose_traceback=verbose_traceback,
    )


if __name__ == "__main__":
    run_io_tests(verbose_traceback=True)
