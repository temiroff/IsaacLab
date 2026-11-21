# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action and reward of the world-model Franka reach task, from the joint state alone."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import JointPositionAction
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from .franka_mapping import tcp_position

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ResetPoseJointPositionAction(JointPositionAction):
    """Joint-position targets around the joint positions the episode started from.

    :class:`~isaaclab.envs.mdp.JointPositionAction` copies the default joint positions once, when it is built. The
    world-model articulation makes each reset record's state the default joint position when it resets, before the
    action terms reset, so this term copies them again then.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._offset[ids] = self._asset.data.default_joint_pos.torch[ids][:, self._joint_ids]


class TcpGoalDistance(ManagerTermBase):
    """Distance from the point between the Franka fingertips to a goal in the robot's base frame [m].

    The world model supplies no body poses, so the fingertips come from the forward kinematics of the joint state,
    on the device. With ``std``, the term is ``1 - tanh(distance / std)`` instead, as the reach task's fine-grained
    tracking reward.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self._goal = torch.tensor(cfg.params["goal"], dtype=torch.float32, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        goal: tuple[float, float, float],
        std: float | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        distance = torch.linalg.norm(tcp_position(self._asset.data.joint_pos.torch) - self._goal, dim=-1)
        return distance if std is None else 1.0 - torch.tanh(distance / std)
