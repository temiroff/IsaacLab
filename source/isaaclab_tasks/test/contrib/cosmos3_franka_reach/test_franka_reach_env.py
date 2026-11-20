# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The world-model Franka reach task end to end, without Kit or Cosmos weights.

The real task configuration runs with two stand-ins: a world model that returns black frames and counts its chunks,
and a robot USD with the Franka's nine joints.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from isaaclab_neural.world_model import ModelCfg, WorldModel, WorldModelSpec

from isaaclab_tasks.contrib.cosmos3_franka_reach.franka_mapping import FRANKA_JOINT_NAMES, tcp_position
from isaaclab_tasks.contrib.cosmos3_franka_reach.franka_reach_env_cfg import CHUNK, FPS, GOAL, FrankaReachEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

H, W = 24, 32
READY = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]


class CountingModel(WorldModel):
    chunks = 0

    def spec(self):
        return WorldModelSpec(chunk_length=CHUNK, fps=FPS, image_size=(H, W))

    def reset(self, env_ids, images, prompts, seeds):
        pass

    def step(self, joint_targets, start_joint_state):
        CountingModel.chunks += 1
        return torch.zeros(self.num_envs, CHUNK, H, W, 3, dtype=torch.uint8, device=self.device)

    def close(self):
        pass


@configclass
class CountingModelCfg(ModelCfg):
    class_type: type = CountingModel


@pytest.fixture
def env(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(tmp_path / "franka.usda"))
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name in FRANKA_JOINT_NAMES:
        joint_type = UsdPhysics.PrismaticJoint if "finger" in name else UsdPhysics.RevoluteJoint
        joint_type.Define(stage, f"/Robot/{name}").CreateAxisAttr("Z")
    stage.Save()
    Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(tmp_path / "start.png")
    manifest = tmp_path / "resets.json"
    manifest.write_text(json.dumps([{"image": "start.png", "state": READY, "prompt": "reach", "split": "train"}]))

    CountingModel.chunks = 0
    cfg = resolve_presets(FrankaReachEnvCfg())
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(usd_path=str(tmp_path / "franka.usda"))
    cfg.scene.camera.height, cfg.scene.camera.width = H, W
    cfg.sim.device = "cpu"
    cfg.sim.physics.model = CountingModelCfg()
    cfg.sim.physics.reset_manifest = str(manifest)
    env = ManagerBasedRLEnv(cfg)
    yield env
    env.close()


def test_the_reach_reward_follows_the_hand_through_forward_kinematics(env):
    obs, _ = env.reset()
    assert obs["image"].shape == (1, 3, H, W) and obs["policy"].shape == (1, 9)
    action = torch.tensor([[0.4, -0.2, 0.1, 0.3, 0.0, -0.1, 0.2]])
    _, reward, *_ = env.step(action)
    # The arm moved to half the action away from the record's pose, the fingers stayed.
    joint_pos = torch.tensor(READY) + 0.5 * torch.cat((action[0], torch.zeros(2)))
    assert torch.allclose(env.scene["robot"].data.joint_pos.torch[0], joint_pos)
    distance = torch.linalg.norm(tcp_position(joint_pos) - torch.tensor(GOAL))
    expected = -0.2 * distance + 0.1 * (1 - torch.tanh(distance / 0.1)) - 1e-4 * action.square().sum()
    assert reward[0].item() == pytest.approx(expected.item() / FPS, rel=1e-4)
    for _ in range(CHUNK - 1):
        env.step(action)
    assert CountingModel.chunks == 1
