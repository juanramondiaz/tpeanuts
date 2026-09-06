"""Regression tests for the Daya Bay published prompt-energy observable."""

import math

import torch

from tpeanuts.detector.dayabay.event_rate import real_observed_counts
from tpeanuts.detector.dayabay.inference_model import DayaBayDetectorModel
from tpeanuts.detector.dayabay.io import load_survival_probability_truth
from tpeanuts.detector.dayabay.parameters import FINAL_EREC_BIN_EDGES_MEV
from tpeanuts.inference.model_vacuum import VacuumOscillationModel
from tpeanuts.util.context import RuntimeContext


def test_ad11_predicted_prompt_shape_is_not_shifted_by_annihilation_energy():
    context = RuntimeContext.resolve("cpu", torch.float64)
    truth = load_survival_probability_truth()
    theta13 = 0.5 * math.asin(math.sqrt(truth["SinSq2Theta13"]))
    theta12 = 0.5 * math.asin(math.sqrt(truth["SinSq2Theta12"]))
    base, _ = VacuumOscillationModel.from_preset(
        "_SM_NUFIT61_NO", free=("theta13", "DeltamSq3l"), context=context,
    )
    oscillation_model = VacuumOscillationModel(
        context=context,
        free=("theta13", "DeltamSq3l"),
        fixed={
            "theta12": torch.tensor(theta12, dtype=context.dtype),
            "DeltamSq21": torch.tensor(truth["DeltaMSq21"], dtype=context.dtype),
        },
        theta23=base.theta23,
        delta13=base.delta13,
    )
    theta = torch.tensor(
        [theta13, truth["DeltaMSq32"] + truth["DeltaMSq21"], 1.0], dtype=context.dtype,
    )
    predicted = DayaBayDetectorModel(oscillation_model, "AD11").predict(theta)
    observed = real_observed_counts("AD11")
    centers = 0.5 * (FINAL_EREC_BIN_EDGES_MEV[:-1] + FINAL_EREC_BIN_EDGES_MEV[1:])
    predicted_mean = (predicted * centers).sum() / predicted.sum()
    observed_mean = (observed * centers).sum() / observed.sum()

    assert abs(float(predicted_mean - observed_mean)) < 0.15
    shape_l1 = (predicted / predicted.sum() - observed / observed.sum()).abs().sum()
    assert float(shape_l1) < 0.15
