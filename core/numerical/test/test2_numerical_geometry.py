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

"""Pytest-compatible checks for numerical trajectory geometry helpers."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.numerical.geometry import Trajectory, segment_sample_points
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor(value):
    return torch.as_tensor(value, device=DEVICE, dtype=DTYPE)


def test_segment_sample_points_midpoint_for_1d_grid():
    x = tensor([0.0, 0.2, 0.5, 1.0])

    sample = segment_sample_points(x, method="midpoint")

    assert sample.shape == (3,)
    assert_close(sample, tensor([0.1, 0.35, 0.75]), name="midpoint sample points")


def test_segment_sample_points_none_is_midpoint():
    x = tensor([0.0, 0.2, 0.5, 1.0])

    sample_none = segment_sample_points(x, method=None)
    sample_midpoint = segment_sample_points(x, method="midpoint")

    assert_close(sample_none, sample_midpoint, name="None sampling method equals midpoint")


def test_segment_sample_points_left_and_right_endpoints():
    x = tensor([0.0, 0.2, 0.5, 1.0])

    left = segment_sample_points(x, method="left")
    right = segment_sample_points(x, method="right")

    assert_close(left, tensor([0.0, 0.2, 0.5]), name="left sample points")
    assert_close(right, tensor([0.2, 0.5, 1.0]), name="right sample points")


def test_segment_sample_points_batched_grid_preserves_leading_shape():
    x = tensor(
        [
            [0.0, 0.2, 0.5, 1.0],
            [1.0, 1.5, 2.5, 4.0],
        ]
    )

    sample = segment_sample_points(x, method="midpoint")

    assert sample.shape == (2, 3)
    assert_close(
        sample,
        tensor(
            [
                [0.1, 0.35, 0.75],
                [1.25, 2.0, 3.25],
            ]
        ),
        name="batched midpoint sample points",
    )


def test_segment_sample_points_preserves_dtype_and_device():
    x = torch.tensor([0.0, 0.2, 0.5], device=DEVICE, dtype=torch.float32)

    sample = segment_sample_points(x)

    assert sample.device == x.device
    assert sample.dtype == x.dtype


def test_segment_sample_points_invalid_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        segment_sample_points(tensor([0.0, 1.0]), method="gauss")


def test_segment_sample_points_single_boundary_returns_empty_segments():
    x = tensor([0.0])

    sample = segment_sample_points(x)

    assert sample.shape == (0,)


def test_trajectory_container_preserves_geometry_and_metadata():
    x = tensor([0.0, 0.2, 0.5, 1.0])
    dx = x[1:] - x[:-1]
    sample = segment_sample_points(x)
    meta = {"medium": "synthetic", "n_segments": 3}

    trajectory = Trajectory(
        x=x,
        dx_evolution=dx,
        sample_x=sample,
        meta=meta,
    )

    assert trajectory.meta is meta
    assert_close(trajectory.dx_evolution, tensor([0.2, 0.3, 0.5]), name="trajectory dx")
    assert_close(trajectory.sample_x, tensor([0.1, 0.35, 0.75]), name="trajectory sample points")


def test_trajectory_container_accepts_batched_tensors():
    x = tensor(
        [
            [0.0, 0.2, 0.5, 1.0],
            [0.0, 0.1, 0.4, 0.9],
        ]
    )
    trajectory = Trajectory(
        x=x,
        dx_evolution=x[..., 1:] - x[..., :-1],
        sample_x=segment_sample_points(x, method="left"),
        meta={"batch": True},
    )

    assert trajectory.x.shape == (2, 4)
    assert trajectory.dx_evolution.shape == (2, 3)
    assert trajectory.sample_x.shape == (2, 3)
    assert_close(trajectory.sample_x, x[..., :-1], name="batched trajectory left samples")
