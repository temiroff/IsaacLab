# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The apple task's terms follow the evaluated GR00T policy's inputs and outputs."""

import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.apple_pick_place.apple_pick_place_g1_dex3_env_cfg import G1ApplePickPlaceEnvCfg


def test_apple_task_is_registered():
    spec = gym.spec("IsaacContrib-PickPlace-Apple-G1-Dex3")
    assert spec.kwargs["env_cfg_entry_point"].endswith(":G1ApplePickPlaceEnvCfg")


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
