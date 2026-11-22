# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AgiBot forward kinematics and its unified-v1 controls."""

import json

import numpy as np
import pytest
import torch

from isaaclab_tasks.contrib.cosmos3_agibot_reach.agibot_mapping import (
    AGIBOT_JOINT_NAMES,
    AgibotJointMapping,
    AgibotJointMappingCfg,
    tcp_positions,
    wrist_poses,
)
from isaaclab_tasks.contrib.cosmos3_agibot_reach.agibot_reach_env_cfg import START_POSE

START = np.array([START_POSE[name] for name in AGIBOT_JOINT_NAMES])

AGIBOTWORLD_VALID = [*range(0, 18), 33, *range(34, 43), 58]


@pytest.fixture
def mapping(tmp_path):
    """Identity normalization, except the gripper's width against the 0.082 m reference of the checkpoint."""
    q01, q99 = [-1.0] * 59, [1.0] * 59
    q01[33] = q01[58] = 0.0
    q99[33] = q99[58] = 0.082
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"global": {"q01": q01, "q99": q99}}))
    return AgibotJointMapping(AgibotJointMappingCfg(global_stats_path=str(stats)))


def test_the_fingertips_of_the_episode_start_match_the_g1_description():
    # MuJoCo's forward kinematics of G1_omnipicker_calibrated.urdf at the same pose, plus the gripper's center link,
    # 0.3 m higher: the description's base link sits that far above the A2D's root.
    tcp = tcp_positions(torch.from_numpy(START))
    expected = torch.tensor([[0.715, 0.258, 0.971], [0.746, -0.137, 0.952]], dtype=torch.float64)
    assert torch.allclose(tcp, expected, atol=1e-3)


def test_a_still_robot_maps_to_identity_poses_and_open_grippers(mapping):
    controls = mapping.encode(np.tile(START, (16, 1)), START)
    raw = np.sinh(controls * np.arcsinh(1.0))
    # Zero translation and the identity rotation's first two columns on the head and both wrists.
    for pose in (slice(0, 9), slice(9, 18), slice(34, 43)):
        np.testing.assert_allclose(raw[:, pose], np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0], (16, 1)), atol=1e-6)
    # Both open grippers are 0.12 m wide, against the 0.082 m reference.
    np.testing.assert_allclose(controls[:, [33, 58]], np.arcsinh(0.12 / 0.041 - 1) / np.arcsinh(1.0), rtol=1e-6)
    invalid = np.ones(59, dtype=bool)
    invalid[AGIBOTWORLD_VALID] = False
    assert controls.shape == (16, 59) and controls.dtype == np.float32 and not controls[:, invalid].any()


def test_gripper_motion_is_expressed_in_the_chunk_start_frame_of_its_own_arm(mapping):
    targets = np.tile(START, (16, 1))
    # Turning the right shoulder swings the right gripper; the left one holds still. In the start gripper's frame the
    # move keeps its length, the chord the gripper travels, and the rotation is the relative one.
    targets[:, 7] += np.linspace(0.0, 0.2, 16)
    raw = np.sinh(mapping.encode(targets, START) * np.arcsinh(1.0))
    poses = wrist_poses(torch.from_numpy(targets[:, :14])).numpy()[:, 1]
    np.testing.assert_allclose(
        np.linalg.norm(raw[:, 9:12], axis=1), np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1), atol=1e-6
    )
    anchor = wrist_poses(torch.from_numpy(START[:14])).numpy()[1]
    np.testing.assert_allclose(raw[:, 12:15], (anchor[:3, :3].T @ poses[:, :3, :3])[:, :, 0], atol=1e-6)
    np.testing.assert_allclose(raw[:, 34:43], np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0], (16, 1)), atol=1e-6)
