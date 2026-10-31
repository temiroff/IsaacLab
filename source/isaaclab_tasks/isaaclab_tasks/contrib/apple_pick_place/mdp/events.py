# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_to_start_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    states_file: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
):
    """Reset the robot, apple and plate to saved starts of real demonstrations.

    Environment ``i`` uses start ``i`` modulo the number of starts, so a run is repeatable and, with at least as many
    environments as starts, covers all of them. Poses are relative to the environment origin, as ``xyzw`` quaternions.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        states_file: JSON file with ``joint_names`` and a list of ``states``. Each state has ``robot_root_pose``,
            ``joint_positions``, ``apple_pose`` and ``plate_pose``.
        robot_cfg: The robot.
        apple_cfg: The apple.
        plate_cfg: The plate.
    """
    cache = env.__dict__.setdefault("_apple_start_states", {})
    if states_file not in cache:
        with open(states_file) as f:
            data = json.load(f)
        states = data["states"]
        tensors = {
            key: torch.tensor([s[key] for s in states], dtype=torch.float32, device=env.device)
            for key in ("robot_root_pose", "joint_positions", "apple_pose", "plate_pose")
        }
        cache[states_file] = (data["joint_names"], tensors)
    joint_names, tensors = cache[states_file]

    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    index = env_ids % tensors["joint_positions"].shape[0]
    origins = env.scene.env_origins[env_ids]
    zero_velocity = torch.zeros(len(env_ids), 6, device=env.device)

    for cfg, key in ((apple_cfg, "apple_pose"), (plate_cfg, "plate_pose")):
        pose = tensors[key][index].clone()
        pose[:, :3] += origins
        asset = env.scene[cfg.name]
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)

    robot = env.scene[robot_cfg.name]
    root_pose = tensors["robot_root_pose"][index].clone()
    root_pose[:, :3] += origins
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)
    joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
    joint_pos = tensors["joint_positions"][index]
    robot.write_joint_position_to_sim_index(position=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), joint_ids=joint_ids, env_ids=env_ids)
    # No action drives the legs, so their PD targets must hold the start pose; targets start at zero otherwise.
    command = robot.actuators.target_command
    command.set_position_index(value=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    command.set_velocity_index(value=torch.zeros_like(joint_pos), joint_ids=joint_ids, env_ids=env_ids)
    command.set_effort_index(value=torch.zeros_like(joint_pos), joint_ids=joint_ids, env_ids=env_ids)


def reset_fixed_head_camera(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("head_camera"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "head_link",
):
    """Hold the head camera still at the pose the robot's head had in each environment's first reset.

    The real demonstrations were recorded from a camera that did not move while the robot did. The camera is not
    a child of the robot. At an environment's first reset this caches the world pose of the camera's offset from
    ``body_name``, and every reset restores it. Order this event after the one that sets the robot's start.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        camera_cfg: The camera, whose offset is relative to ``body_name``.
        robot_cfg: The robot.
        body_name: The robot body the camera's offset is relative to.
    """
    camera = env.scene[camera_cfg.name]
    robot = env.scene[robot_cfg.name]
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    if "_fixed_head_camera" not in env.__dict__:
        env._fixed_head_camera = {
            "ready": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "pos": torch.zeros(env.num_envs, 3, device=env.device),
            "quat": torch.zeros(env.num_envs, 4, device=env.device),
        }
    cached = env._fixed_head_camera
    first = env_ids[~cached["ready"][env_ids]]
    if len(first) > 0:
        env.sim.forward()
        body_pose = robot.data.body_link_pose_w.torch[first, robot.body_names.index(body_name)]
        offset = camera.cfg.offset
        pos, quat = math_utils.combine_frame_transforms(
            body_pose[:, :3],
            body_pose[:, 3:7],
            body_pose.new_tensor(offset.pos).expand(len(first), -1),
            body_pose.new_tensor(offset.rot).expand(len(first), -1),
        )
        cached["pos"][first] = pos
        cached["quat"][first] = quat
        cached["ready"][first] = True
    camera.set_world_poses(
        cached["pos"][env_ids], cached["quat"][env_ids], env_ids=env_ids, convention=camera.cfg.offset.convention
    )
