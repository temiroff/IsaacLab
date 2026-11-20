# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka forward kinematics and its conversion to the ``normalized_unified_v1`` controls of Cosmos3 Sim-Bimanual.

The public ``nvidia/Cosmos3-Nano-Sim-Bimanual`` checkpoint was trained on wrist poses and gripper openness of
parallel-gripper arms. The Franka hand drives the right wrist of the ``xdof_yam`` domain; the left wrist, which the
domain also expects, holds still.
"""

from __future__ import annotations

import json
import math
from dataclasses import MISSING
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils import configclass

from isaaclab_neural.models.cosmos3 import JointMappingCfg

FRANKA_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
"""The arm's seven joints, then the two fingers: the order of the reset records and of the joint state."""

# Modified Denavit-Hartenberg parameters (a [m], d [m], alpha [rad]) of the seven joints and the flange, from the
# Franka documentation. The hand frame is the flange turned by -45 degrees about its z axis.
_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -math.pi / 2),
    (0.0, 0.316, math.pi / 2),
    (0.0825, 0.0, math.pi / 2),
    (-0.0825, 0.384, -math.pi / 2),
    (0.0, 0.0, math.pi / 2),
    (0.088, 0.0, math.pi / 2),
    (0.0, 0.107, 0.0),
)
_HAND_YAW = -math.pi / 4

TCP_OFFSET = 0.1034
"""Distance from the hand frame to the point between the fingertips, along the hand's z axis [m]."""

# The 59 unified-v1 controls: right wrist pose at 9-17 and openness at 33, left wrist pose at 34-42 and openness
# at 58. These are the controls the xdof_yam domain marks valid.
_RIGHT_WRIST, _RIGHT_OPENNESS, _LEFT_WRIST, _LEFT_OPENNESS = slice(9, 18), 33, slice(34, 43), 58
_VALID = np.zeros(59, dtype=bool)
_VALID[[_RIGHT_OPENNESS, _LEFT_OPENNESS]] = True
_VALID[_RIGHT_WRIST] = _VALID[_LEFT_WRIST] = True
_IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
"""Zero translation and the identity rotation in rot6d (its first two columns)."""


def _link(theta: torch.Tensor, a: float, d: float, alpha: float) -> torch.Tensor:
    """``RotX(alpha) TransX(a) RotZ(theta) TransZ(d)``, shape ``(..., 4, 4)``."""
    c, s = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    rows = (
        (c, -s, zero, a * one),
        (s * ca, c * ca, -sa * one, -d * sa * one),
        (s * sa, c * sa, ca * one, d * ca * one),
        (zero, zero, zero, one),
    )
    return torch.stack([torch.stack(row, -1) for row in rows], -2)


def hand_pose(arm_joint_pos: torch.Tensor) -> torch.Tensor:
    """Poses of the Franka hand in the robot's base frame, shape ``(..., 4, 4)``, from the arm's seven joint
    positions [rad], shape ``(..., 7)``. Runs on the joint positions' device."""
    angles = torch.cat((arm_joint_pos, torch.zeros_like(arm_joint_pos[..., :1])), dim=-1)  # the flange does not turn
    pose = torch.eye(4, dtype=arm_joint_pos.dtype, device=arm_joint_pos.device)
    for i, (a, d, alpha) in enumerate(_DH):
        pose = pose @ _link(angles[..., i], a, d, alpha)
    return pose @ _link(torch.full_like(angles[..., 0], _HAND_YAW), 0.0, 0.0, 0.0)


def tcp_position(joint_pos: torch.Tensor) -> torch.Tensor:
    """Position of the point between the fingertips in the robot's base frame [m], shape ``(..., 3)``, from the
    joint positions in the order of :data:`FRANKA_JOINT_NAMES`."""
    pose = hand_pose(joint_pos[..., :7])
    return pose[..., :3, 3] + TCP_OFFSET * pose[..., :3, 2]


@configclass
class FrankaJointMappingCfg(JointMappingCfg):
    """Joint mapping of the Franka arm to the ``xdof_yam`` controls of Cosmos3 Sim-Bimanual."""

    class_type: type | str = "{DIR}.franka_mapping:FrankaJointMapping"

    global_stats_path: str = MISSING
    """Normalization statistics of the checkpoint (``global_asinh_v1_backward_anchored_gripper_metric.json``),
    shipped in its ``cosmos3_nano_sim_bimanual_action_sources`` folder under its SHA-256."""


class FrankaJointMapping:
    """Convert Franka joint targets to the normalized ``unified_v1`` controls of Cosmos3 Sim-Bimanual.

    Row ``i`` of a chunk is the hand pose of target ``i`` relative to the hand pose at the start of the chunk
    (translation in the start pose's frame and the first two columns of the relative rotation), and the target's
    finger width [m] as the openness. The model's gripper statistics are widths against a 0.082 m reference, so the
    Franka's width (up to 0.08 m) needs no calibration. The left wrist holds still with closed fingers. Valid controls
    are normalized with the checkpoint's statistics; the others are zero.
    """

    def __init__(self, cfg: FrankaJointMappingCfg):
        stats = json.loads(Path(cfg.global_stats_path).read_text())["global"]
        low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        self.offset, self.scale = (low + high) / 2, np.maximum((high - low) / 2, 1e-8)

    def encode(self, joint_targets: np.ndarray, start_joint_state: np.ndarray) -> np.ndarray:
        """Return normalized controls shaped ``(T, 59)`` for ``T`` joint targets ``(T, 9)``."""
        targets = torch.from_numpy(np.asarray(joint_targets, dtype=np.float64))
        start = torch.from_numpy(np.asarray(start_joint_state, dtype=np.float64))
        anchor, poses = hand_pose(start[:7]).numpy(), hand_pose(targets[:, :7]).numpy()
        rotation = anchor[:3, :3].T @ poses[:, :3, :3]
        raw = np.zeros((len(poses), 59))
        raw[:, 9:12] = (poses[:, :3, 3] - anchor[:3, 3]) @ anchor[:3, :3]
        raw[:, 12:18] = rotation[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
        raw[:, _RIGHT_OPENNESS] = targets[:, 7:9].sum(-1).numpy()
        raw[:, _LEFT_WRIST] = _IDENTITY_POSE
        normalized = np.arcsinh((raw - self.offset) / self.scale) / np.arcsinh(1.0)
        normalized[:, ~_VALID] = 0.0
        return normalized.astype(np.float32)
