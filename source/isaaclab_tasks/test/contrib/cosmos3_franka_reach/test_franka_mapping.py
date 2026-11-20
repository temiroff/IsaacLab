# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka forward kinematics and its unified-v1 controls."""

import json
import math

import numpy as np
import pytest
import torch

from isaaclab_tasks.contrib.cosmos3_franka_reach.franka_mapping import (
    FrankaJointMapping,
    FrankaJointMappingCfg,
    hand_pose,
    tcp_position,
)

READY = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]
"""The Franka reach task's default joint positions."""

XDOF_YAM_VALID = [*range(9, 18), 33, *range(34, 43), 58]


@pytest.fixture
def mapping(tmp_path):
    """Identity normalization, except the gripper's width against the 0.082 m reference of the checkpoint."""
    q01, q99 = [-1.0] * 59, [1.0] * 59
    q01[33] = q01[58] = 0.0
    q99[33] = q99[58] = 0.082
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"global": {"q01": q01, "q99": q99}}))
    return FrankaJointMapping(FrankaJointMappingCfg(global_stats_path=str(stats)))


def test_the_hand_hangs_above_the_base_at_the_zero_pose():
    # With all joints at zero, the Franka documentation puts the flange 0.088 m forward and 0.926 m up, facing down.
    pose = hand_pose(torch.zeros(7, dtype=torch.float64))
    assert torch.allclose(pose[:3, 3], torch.tensor([0.088, 0.0, 0.926], dtype=torch.float64), atol=1e-9)
    assert torch.allclose(pose[:3, 2], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64), atol=1e-9)
    # The fingertips are 0.1034 m further down, and the first joint turns them about the vertical axis.
    turned = torch.tensor([math.pi / 2, *[0.0] * 8], dtype=torch.float64)
    assert torch.allclose(tcp_position(turned), torch.tensor([0.0, 0.088, 0.8226], dtype=torch.float64), atol=1e-9)


def test_a_still_arm_maps_to_the_identity_pose_and_its_finger_width(mapping):
    start = np.array(READY)
    controls = mapping.encode(np.tile(start, (16, 1)), start)
    raw = np.sinh(controls * np.arcsinh(1.0))
    # Zero translation and the identity rotation's first two columns on both wrists.
    for wrist in (slice(9, 18), slice(34, 43)):
        np.testing.assert_allclose(raw[:, wrist], np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0], (16, 1)), atol=1e-6)
    # The open 0.08 m Franka gripper against the 0.082 m reference; the left one is closed.
    np.testing.assert_allclose(controls[:, 33], np.arcsinh(0.08 / 0.041 - 1) / np.arcsinh(1.0), rtol=1e-6)
    np.testing.assert_allclose(controls[:, 58], -1.0, rtol=1e-6)
    invalid = np.ones(59, dtype=bool)
    invalid[XDOF_YAM_VALID] = False
    assert controls.shape == (16, 59) and controls.dtype == np.float32 and not controls[:, invalid].any()


def test_hand_motion_is_expressed_in_the_chunk_start_frame(mapping):
    start = np.array(READY)
    targets = np.tile(start, (16, 1))
    # Turning the base swings the hand sideways. In the start hand's frame the move keeps its length, the chord the
    # hand travels, and the rotation is the relative one.
    targets[:, 0] = np.linspace(0.0, 0.2, 16)
    raw = np.sinh(mapping.encode(targets, start) * np.arcsinh(1.0))
    poses = hand_pose(torch.from_numpy(targets[:, :7])).numpy()
    np.testing.assert_allclose(
        np.linalg.norm(raw[:, 9:12], axis=1), np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1), atol=1e-6
    )
    anchor = hand_pose(torch.from_numpy(start[:7])).numpy()
    np.testing.assert_allclose(raw[:, 12:15], (anchor[:3, :3].T @ poses[:, :3, :3])[:, :, 0], atol=1e-6)
