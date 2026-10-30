# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The apple task's terms follow the evaluated GR00T policy's inputs and outputs."""

import gymnasium as gym
import pytest
from isaaclab_experimental.cosmos import CosmosTransferProcessorCfg, read_prompt_files

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.apple_pick_place.apple_pick_place_g1_dex3_env_cfg import (
    G1ApplePickPlaceCosmosEnvCfg,
    G1ApplePickPlaceEnvCfg,
)
from isaaclab_tasks.utils import parse_env_cfg


def test_apple_task_is_registered():
    spec = gym.spec("IsaacContrib-PickPlace-Apple-G1-Dex3")
    assert spec.kwargs["env_cfg_entry_point"].endswith(":G1ApplePickPlaceEnvPresetCfg")


def test_observations_and_actions_follow_the_policy_contract():
    cfg = G1ApplePickPlaceEnvCfg()
    policy = cfg.observations.policy
    # One term per policy input, not concatenated, so a pipeline can read each by name.
    terms = [name for name in vars(policy) if hasattr(getattr(policy, name), "func")]
    assert terms == ["ego_view", "left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
    assert not policy.concatenate_terms
    state = [name for term in terms[1:] for name in getattr(policy, term).params["asset_cfg"].joint_names]
    assert len(state) == len(set(state)) == 43
    # The policy's hand outputs list index and middle before the thumb; its hand inputs list the thumb first.
    assert policy.right_hand.params["asset_cfg"].joint_names[3] == "right_hand_index_0_joint"
    assert cfg.actions.joint_pos.joint_names[14] == "left_hand_index_0_joint"
    assert cfg.actions.joint_pos.joint_names == cfg.actions.joint_effort.joint_names
    assert len(cfg.actions.joint_pos.joint_names) == 31 and not any(
        "hip" in n for n in cfg.actions.joint_pos.joint_names
    )
    # 30 Hz policy actions on 240 Hz physics.
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1 / 30)


def test_cosmos_preset_generates_the_head_camera_image_for_whole_episodes():
    cfg = parse_env_cfg("IsaacContrib-PickPlace-Apple-G1-Dex3", overrides=("presets=cosmos",))
    assert isinstance(cfg, G1ApplePickPlaceCosmosEnvCfg)
    cfg.validate()
    camera = cfg.scene.head_camera
    # Cosmos renders depth on its 4:3 canvas; the policy still reads a 640 x 480 rgb image.
    assert (camera.width, camera.height) == (736, 544)
    assert camera.tensor_processors[-1].outputs == ["rgb"]
    assert cfg.observations.policy.ego_view.params["sensor_cfg"].name == "head_camera"
    model = next(p.backend for p in camera.tensor_processors if isinstance(p, CosmosTransferProcessorCfg))
    # One frame per 30 Hz step for a 20 s episode, plus the first frame, in the service's 1 + 4*k budget.
    assert model.max_episode_frames >= 601 and (model.max_episode_frames - 1) % 4 == 0
    # A longer episode gets a larger budget.
    longer = parse_env_cfg(
        "IsaacContrib-PickPlace-Apple-G1-Dex3", overrides=("presets=cosmos", "env.episode_length_s=30.0")
    )
    longer.validate()
    longer_model = next(
        p.backend for p in longer.scene.head_camera.tensor_processors if isinstance(p, CosmosTransferProcessorCfg)
    )
    assert longer_model.max_episode_frames >= 901
    # The prompt describes the real demonstrations' appearance.
    assert "apple" in read_prompt_files([model.prompt_files])[0]
    # The default task keeps the rendered image.
    assert not G1ApplePickPlaceEnvCfg().scene.head_camera.tensor_processors
