# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the reset records of the world-model Franka reach task with OvPhysX and OVRTX, without Kit.

Each record is a start image of the task's camera, the Franka joint state the image shows and a prompt. The script
poses the arm at sampled joint positions around its default pose, lets OvPhysX hold it there, renders the camera with
OVRTX and writes the images and ``resets.json`` that ``WorldModelCfg.reset_manifest`` reads. It also checks the task's
forward kinematics against the simulated hand.

Usage:

.. code-block:: bash

    uv sync --extra ov
    uv run python scripts/tools/cosmos/render_franka_reset_records.py
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_train", type=int, default=4, help="Number of training records.")
parser.add_argument("--num_validation", type=int, default=1, help="Number of validation records.")
parser.add_argument("--seed", type=int, default=0, help="Seed of the sampled joint positions.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import numpy as np
import torch
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass, instantiate, replace

from isaaclab_tasks.contrib.cosmos3_franka_reach import franka_reach_env_cfg as task
from isaaclab_tasks.contrib.cosmos3_franka_reach.franka_mapping import hand_pose

PROMPT = (
    "The scene opens in a high-angle, wide landscape shot of a compact robot work cell, sharply focused from"
    " foreground to background. A single white Franka robotic arm with dark gray joint housings and a two-finger"
    " parallel gripper is mounted at the bottom center of the frame, its base partly cropped by the bottom edge, and"
    " reaches forward over a plain brown tabletop that fills the middle of the frame. A flat gray floor lies beyond"
    " the table. Even, diffuse overhead lighting casts soft shadows in a controlled laboratory mood. The arm moves its"
    " gripper smoothly to a point above the table."
)
"""Task description of every record, in the style of the checkpoint's YAM prompts."""

ARM_OFFSET = 0.3
"""Largest offset of each arm joint from the default pose [rad]."""


@configclass
class RenderSceneCfg(task.FrankaReachSceneCfg):
    """The task's robot and camera, with the table, floor and lights the start images show."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 1.3, 1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.3, 0.18), roughness=0.6),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1500.0)
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight", spawn=sim_utils.DistantLightCfg(intensity=2000.0, angle=10.0)
    )


def settle(sim, scene, steps: int) -> None:
    """Hold the joint targets for ``steps`` physics steps, rendering each one."""
    for _ in range(steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


def main():
    # OvPhysX and OVRTX run without Isaac Sim / Kit.
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device, physics=OvPhysxCfg())
    scene_cfg = RenderSceneCfg(num_envs=1, env_spacing=2.5)
    # The task's camera, rendered with OVRTX instead of imagined by the world model.
    scene_cfg.camera = replace(scene_cfg.camera, renderer_cfg=OVRTXRendererCfg(), frame_history_length=0)
    with launch_simulation((sim_cfg, scene_cfg), args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = instantiate(scene_cfg)
        sim.reset()
        robot, camera = scene["robot"], scene["camera"]
        if robot.joint_names != task.FRANKA_JOINT_NAMES:
            raise RuntimeError(f"Unexpected joint order {robot.joint_names}.")
        hand = robot.find_bodies("panda_hand")[0][0]
        default = robot.data.default_joint_pos.torch[0].clone()
        limits = robot.data.soft_joint_pos_limits.torch[0]
        generator = torch.Generator(device=default.device).manual_seed(args_cli.seed)
        # The first frames stream in the assets and textures.
        settle(sim, scene, 120)

        records, fk_error = [], 0.0
        splits = ["train"] * args_cli.num_train + ["validation"] * args_cli.num_validation
        counts = dict.fromkeys(splits, 0)
        for split in splits:
            q = default.clone()
            noise = torch.rand(7, generator=generator, device=default.device) * 2 - 1
            q[:7] = torch.clamp(q[:7] + ARM_OFFSET * noise, limits[:7, 0], limits[:7, 1])
            robot.write_joint_position_to_sim_index(position=q[None])
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q)[None])
            robot.set_joint_position_target_index(target=q[None])
            settle(sim, scene, 60)

            state = robot.data.joint_pos.torch[0]
            simulated = robot.data.body_link_pos_w.torch[0, hand] - robot.data.root_link_pos_w.torch[0]
            fk_error = max(fk_error, torch.linalg.norm(hand_pose(state[:7].double())[:3, 3].float() - simulated).item())
            name = f"{split}_{counts[split]:02d}.png"
            counts[split] += 1
            Image.fromarray(camera.data.output["rgb"][0, ..., :3].cpu().numpy()).save(task.ASSETS / name)
            records.append({"image": name, "state": state.cpu().tolist(), "prompt": PROMPT, "split": split})
            print(f"[INFO] {name}: joint state {np.round(records[-1]['state'], 3).tolist()}")

        (task.ASSETS / "resets.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"[INFO] Wrote {len(records)} reset records to {Path(task.ASSETS) / 'resets.json'}.")
        print(f"[INFO] Largest forward-kinematics error of the hand: {fk_error * 1000:.2f} mm.")
        if fk_error > 1e-3:
            raise RuntimeError("The task's forward kinematics does not match the simulated Franka hand.")


if __name__ == "__main__":
    main()
