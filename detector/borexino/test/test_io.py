"""Tests for the Borexino hit-estimator to energy-density conversion."""

import pandas as pd
import torch

from tpeanuts.detector.borexino.io import load_low_energy_spectrum


def test_low_energy_loader_applies_hit_to_energy_jacobian(tmp_path):
    path = tmp_path / "spectrum.csv"
    pd.DataFrame(
        {
            "bin": [1, 2],
            "N_h": [92, 93],
            "energy_keV": [212.0, 214.5],
            "width_keV": [2.0, 3.0],
            "rate": [10.0, 12.0],
            "rate_error": [1.0, 1.2],
            "residual": [0.0, 0.0],
        }
    ).to_csv(path, index=False)

    observation = load_low_energy_spectrum(path, dtype=torch.float64)

    torch.testing.assert_close(
        observation.value * observation.bin_width_MeV,
        torch.tensor([10.0, 12.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        observation.sigma_plus * observation.bin_width_MeV,
        torch.tensor([1.0, 1.2], dtype=torch.float64),
    )
