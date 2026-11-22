# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AgiBot A2D forward kinematics and its conversion to the ``normalized_unified_v1`` controls of Cosmos3 Sim-Bimanual.

The public ``nvidia/Cosmos3-Nano-Sim-Bimanual`` checkpoint was trained on AgiBot World episodes of the AgiBot G1. Its
``agibotworld`` controls are the poses of both gripper bases, from the forward kinematics of the G1 description that
Cosmos Framework ships (``robot_assets/G1_omnipicker_calibrated.urdf``), turned into the OpenCV convention, and the
gripper openness (``agibot_semantics.py`` and ``agibot_fk.py`` there). This module computes the same from the joint
targets of the A2D, which has the G1's arms.
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

AGIBOT_JOINT_NAMES = ([f"left_arm_joint{i}" for i in range(1, 8)] + [f"right_arm_joint{i}" for i in range(1, 8)]) + [
    "left_hand_joint1",
    "right_hand_joint1",
]
"""The left and right arms' seven joints, then the two gripper drives: the order of the reset records and of the
joint state."""

WAIST_LIFT, WAIST_PITCH = 0.196, 0.431
"""Waist lift [m] and pitch [rad] at the start of an AgiBot World episode. The task holds them."""

HEAD_YAW, HEAD_PITCH = 0.003, 0.371
"""Head yaw and pitch [rad] at the start of the same episode. The task holds them."""

GRIPPER_OPEN, GRIPPER_CLOSED = 85 / 120, 0.0
"""Positions of an A2D gripper drive that the training data counts as fully open and fully closed. The drive's
position is ``1 - g / 120`` for an AgiBot gripper reading ``g``, and the data counts readings of 35 and less as
fully open."""

BASE_HEIGHT = 0.3
"""Height of the G1 description's base link above the A2D's root [m]. The A2D matches the description's waist and
arms from there."""

TCP_OFFSET = 0.14308
"""Distance from a gripper base to the point between its fingertips, along the base's z axis [m]."""

_H, _PI = math.pi / 2, math.pi

# Fixed transforms of the G1 description as (xyz [m], rpy [rad]). The waist lifts along z, then every joint turns
# about the z axis of the frame that follows its transform.
_WAIST_LIFT_ORIGIN = ((0.0, 0.0, 0.3485), (0.0, 0.0, 0.0))
_WAIST_PITCH_ORIGIN = ((0.131, 0.0, 0.0), (-_H, 0.0, 0.0))
_ARM_BASE = ((0.0, -0.305, 0.0), (_H, 0.0, 0.0))
_ARMS = {
    "left": (
        ((0.0, 0.025, 0.0), (-_H, 0.0, 0.0)),
        (
            ((0.0, 0.0, 0.1859), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (-_H, 0.0, _H)),
            ((0.0, -0.305, 0.0), (-_H, _H, _PI)),
            ((0.0, 0.0, 0.0), (-_H, 0.0, _PI)),
            ((0.0, -0.1975, 0.0), (-_H, 0.0, -_PI)),
            ((0.0, 0.0, 0.0), (-_H, 0.0, 0.0)),
            ((0.0, -0.1805, 0.0), (_H, 0.0, 0.0)),
        ),
        ((0.0, 0.0, 0.0061), (0.0, 0.0, _H)),
    ),
    "right": (
        ((0.0, -0.025, 0.0), (-_H, 0.0, _PI)),
        (
            ((0.0, 0.0, 0.188), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (-_H, 0.0, -_H)),
            ((0.0, -0.305, 0.0), (-_H, -_H, _PI)),
            ((0.0, 0.0, 0.0), (-_H, 0.0, _PI)),
            ((0.0, -0.1975, 0.0), (-_H, 0.0, -_PI)),
            ((0.0, 0.0, 0.0), (-_H, 0.0, 0.0)),
            ((0.0, -0.1805, 0.0), (_H, 0.0, 0.0)),
        ),
        ((0.0, 0.0, 0.0061), (0.0, 0.0, -_H)),
    ),
}
"""Per arm: the mount on the arm base, the transforms before the seven joints, and the gripper base after the last."""

# The training data turns each gripper base about its z axis into the OpenCV convention (``agibot_fk.py``).
_TO_OPENCV = {
    "left": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "right": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}

# The 59 unified-v1 controls the agibotworld domain marks valid: the head camera's pose at 0-8, the right wrist pose at
# 9-17 and openness at 33, the left wrist pose at 34-42 and openness at 58.
_EGO, _RIGHT_WRIST, _RIGHT_OPENNESS, _LEFT_WRIST, _LEFT_OPENNESS = slice(0, 9), slice(9, 18), 33, slice(34, 43), 58
_VALID = np.zeros(59, dtype=bool)
_VALID[_EGO] = _VALID[_RIGHT_WRIST] = _VALID[_LEFT_WRIST] = True
_VALID[[_RIGHT_OPENNESS, _LEFT_OPENNESS]] = True
_IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
"""Zero translation and the identity rotation in rot6d (its first two columns)."""

OPEN_WIDTH = 0.120
"""Jaw-to-jaw width of an open AgiBot gripper [m], which scales the openness against the checkpoint's statistics."""


def _transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    """Homogeneous transform of a URDF origin: translation, then ``Rz(yaw) Ry(pitch) Rx(roll)``."""
    (cr, sr), (cp, sp), (cy, sy) = ((math.cos(a), math.sin(a)) for a in rpy)
    pose = np.eye(4)
    pose[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    pose[:3, 3] = xyz
    return pose


def _turn_z(angle: torch.Tensor) -> torch.Tensor:
    """Rotation about z by ``angle``, shape ``(..., 4, 4)``."""
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    rows = ((c, -s, zero, zero), (s, c, zero, zero), (zero, zero, one, zero), (zero, zero, zero, one))
    return torch.stack([torch.stack(row, -1) for row in rows], -2)


def _torso() -> np.ndarray:
    """Pose of the arm base in the robot's base frame at the held waist position."""
    lift = _transform((0.0, 0.0, BASE_HEIGHT + WAIST_LIFT), (0.0, 0.0, 0.0)) @ _transform(*_WAIST_LIFT_ORIGIN)
    pitch = _transform(*_WAIST_PITCH_ORIGIN) @ _transform((0.0, 0.0, 0.0), (0.0, 0.0, WAIST_PITCH))
    return lift @ pitch @ _transform(*_ARM_BASE)


def wrist_poses(arm_joint_pos: torch.Tensor) -> torch.Tensor:
    """Poses of the left and right gripper bases in the OpenCV convention of the training data, in the robot's base
    frame, shape ``(..., 2, 4, 4)``, from the 14 arm joint positions [rad] in the order of :data:`AGIBOT_JOINT_NAMES`,
    shape ``(..., 14)``. Runs on the joint positions' device."""
    poses = []
    for arm, (side, (mount, joints, gripper)) in enumerate(_ARMS.items()):
        tool = _transform(*gripper)
        tool[:3, :3] = tool[:3, :3] @ _TO_OPENCV[side]
        pose = torch.as_tensor(_torso() @ _transform(*mount), dtype=arm_joint_pos.dtype, device=arm_joint_pos.device)
        for i, origin in enumerate(joints):
            fixed = torch.as_tensor(_transform(*origin), dtype=arm_joint_pos.dtype, device=arm_joint_pos.device)
            pose = pose @ fixed @ _turn_z(arm_joint_pos[..., 7 * arm + i])
        poses.append(pose @ torch.as_tensor(tool, dtype=arm_joint_pos.dtype, device=arm_joint_pos.device))
    return torch.stack(poses, dim=-3)


def tcp_positions(joint_pos: torch.Tensor) -> torch.Tensor:
    """Positions of the points between the left and right fingertips in the robot's base frame [m], shape
    ``(..., 2, 3)``, from the joint positions in the order of :data:`AGIBOT_JOINT_NAMES`."""
    poses = wrist_poses(joint_pos[..., :14])
    return poses[..., :3, 3] + TCP_OFFSET * poses[..., :3, 2]


@configclass
class AgibotJointMappingCfg(JointMappingCfg):
    """Joint mapping of the AgiBot A2D to the ``agibotworld`` controls of Cosmos3 Sim-Bimanual."""

    class_type: type | str = "{DIR}.agibot_mapping:AgibotJointMapping"

    global_stats_path: str = MISSING
    """Normalization statistics of the checkpoint (``global_asinh_v1_backward_anchored_gripper_metric.json``),
    shipped in its ``cosmos3_nano_sim_bimanual_action_sources`` folder under its SHA-256."""


class AgibotJointMapping:
    """Convert AgiBot A2D joint targets to the normalized ``unified_v1`` controls of Cosmos3 Sim-Bimanual.

    Row ``i`` of a chunk holds, for each arm, the gripper base pose of target ``i`` relative to its pose at the start
    of the chunk (translation in the start pose's frame and the first two columns of the relative rotation), and the
    target's gripper openness as a width, :data:`OPEN_WIDTH` when fully open, as the checkpoint's statistics expect.
    The head holds still, so its relative pose is the identity. Valid controls are normalized with the checkpoint's
    statistics; the others are zero.
    """

    def __init__(self, cfg: AgibotJointMappingCfg):
        stats = json.loads(Path(cfg.global_stats_path).read_text())["global"]
        low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        self.offset, self.scale = (low + high) / 2, np.maximum((high - low) / 2, 1e-8)

    def encode(self, joint_targets: np.ndarray, start_joint_state: np.ndarray) -> np.ndarray:
        """Return normalized controls shaped ``(T, 59)`` for ``T`` joint targets ``(T, 16)``."""
        targets = np.asarray(joint_targets, dtype=np.float64)
        start = np.asarray(start_joint_state, dtype=np.float64)
        anchors = wrist_poses(torch.from_numpy(start[:14])).numpy()
        poses = wrist_poses(torch.from_numpy(targets[:, :14])).numpy()
        openness = np.clip((targets[:, 14:16] - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED), 0.0, 1.0)
        raw = np.zeros((len(targets), 59))
        raw[:, _EGO] = _IDENTITY_POSE
        for arm, (wrist, width) in enumerate(((_LEFT_WRIST, _LEFT_OPENNESS), (_RIGHT_WRIST, _RIGHT_OPENNESS))):
            anchor, pose = anchors[arm], poses[:, arm]
            raw[:, wrist.start : wrist.start + 3] = (pose[:, :3, 3] - anchor[:3, 3]) @ anchor[:3, :3]
            rotation = anchor[:3, :3].T @ pose[:, :3, :3]
            raw[:, wrist.start + 3 : wrist.stop] = rotation[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
            raw[:, width] = OPEN_WIDTH * openness[:, arm]
        normalized = np.arcsinh((raw - self.offset) / self.scale) / np.arcsinh(1.0)
        normalized[:, ~_VALID] = 0.0
        return normalized.astype(np.float32)
