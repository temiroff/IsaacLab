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
START = [0.3, -0.7, -0.2, -2.6, 0.25, 3.1, 0.6, 0.04, 0.04]
"""The reset record's joint positions, away from the Franka's initial ones."""


class CountingModel(WorldModel):
    chunks = []

    def __init__(self, cfg, num_envs, device):
        self.num_envs, self.device = num_envs, device

    def spec(self):
        return WorldModelSpec(chunk_length=CHUNK, fps=FPS, image_size=(H, W))

    def reset(self, env_ids, images, prompts, seeds):
        pass

    def write(self, name, value):
        if name == "joint_targets":
            CountingModel.chunks.append(value.clone())

    def step(self):
        return {"camera": {"rgb": torch.zeros(self.num_envs, CHUNK, H, W, 3, dtype=torch.uint8, device=self.device)}}

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
    manifest.write_text(json.dumps([{"image": "start.png", "state": START, "prompt": "reach", "split": "train"}]))

    CountingModel.chunks = []
    cfg = resolve_presets(FrankaReachEnvCfg())
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(usd_path=str(tmp_path / "franka.usda"))
    cfg.scene.camera.height, cfg.scene.camera.width = H, W
    cfg.sim.device = "cpu"
    cfg.sim.physics.model_cfg = CountingModelCfg()
    cfg.sim.physics.reset_manifest = str(manifest)
    env = ManagerBasedRLEnv(cfg)
    yield env
    env.close()


def test_one_step_moves_the_arm_over_a_chunk_and_rewards_the_hand_through_forward_kinematics(env):
    obs, _ = env.reset()
    assert obs["image"].shape == (1, 3, H, W) and obs["policy"].shape == (1, 9)
    action = torch.tensor([[0.4, -0.2, 0.1, 0.3, 0.0, -0.1, 0.2]])
    _, reward, *_ = env.step(action)
    # The model ran once, on targets that reach half the action away from the record's pose in equal increments.
    (targets,) = CountingModel.chunks
    delta = 0.5 * torch.cat((action[0], torch.zeros(2)))
    fractions = torch.arange(1, CHUNK + 1) / CHUNK
    assert torch.allclose(targets[0], torch.tensor(START) + fractions[:, None] * delta)
    joint_pos = torch.tensor(START) + delta
    assert torch.allclose(env.scene["robot"].data.joint_pos.torch[0], joint_pos)
    distance = torch.linalg.norm(tcp_position(joint_pos) - torch.tensor(GOAL))
    expected = -0.2 * distance + 0.1 * (1 - torch.tanh(distance / 0.1)) - 1e-4 * action.square().sum()
    assert reward[0].item() == pytest.approx(expected.item() * CHUNK / FPS, rel=1e-4)
