# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the reset records of the world-model AgiBot reach task with OvPhysX and OVRTX, without Kit.

Each record is a start image of the task's head camera, the A2D joint state the image shows and a prompt. The script
poses the full A2D at the AgiBot World episode's waist and head, and its arms at sampled joint positions around the
task's start pose, lets OvPhysX hold it there, renders the camera with OVRTX and writes the images and ``resets.json``
that ``NeuralCfg.reset_manifest`` reads. It also checks the task's forward kinematics against the simulated grippers.

The A2D's head joints do not follow the G1 description the camera pose comes from: the camera sits where the
description's head camera is, inside the A2D's head, so the head links are hidden. The arm and gripper links get a
dark material, like the AgiBot World G1's black arms.

Usage:

.. code-block:: bash

    uv sync --extra ov
    uv run python scripts/tools/cosmos/render_agibot_reset_records.py
"""

import argparse
import json

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_train", type=int, default=4, help="Number of training records.")
parser.add_argument("--num_validation", type=int, default=1, help="Number of validation records.")
parser.add_argument("--seed", type=int, default=0, help="Seed of the sampled joint positions.")
parser.add_argument("--table_top", type=float, default=0.80, help="Height of the table top above the A2D's root [m].")
add_launcher_args(parser)
args_cli = parser.parse_args()

import numpy as np
import torch
from isaaclab_isaacsim.sim.schemas import PhysxRigidBodyCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from PIL import Image

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass, instantiate, replace
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_tasks.contrib.cosmos3_agibot_reach import agibot_reach_env_cfg as task
from isaaclab_tasks.contrib.cosmos3_agibot_reach.agibot_mapping import (
    AGIBOT_JOINT_NAMES,
    GRIPPER_OPEN,
    HEAD_PITCH,
    TCP_OFFSET,
    WAIST_LIFT,
    WAIST_PITCH,
    tcp_positions,
)

from isaaclab_assets import AGIBOT_A2D_CFG

PROMPT = (
    "The scene opens in an overhead-oblique, landscape tabletop shot from the robot's head camera with sharp focus,"
    " framing a light wood-grain work surface that fills most of the image while both robotic arms are cropped by the"
    " side edges. A pale gray and off-white checkered floor with small green markers extends beyond the far edge of"
    " the table. Slightly right of center, a turquoise plastic bottle-like block stands upright; left of it lies a"
    " flat blue-and-white packet, a small brown cardboard box sits in the right part of the table and a small red box"
    " rests near the upper-left tabletop. The left robotic arm is black with articulated linkages and a two-pronged"
    " black gripper; only its gripper and part of its wrist are visible at the left edge. The right robotic arm is"
    " black with black mounting blocks and an articulated two-pronged gripper; it is cropped by the right edge and"
    " faces inward toward the central objects. The bottle-like block is the tallest tabletop object and the packet"
    " lies flat; the left arm is nearest the camera along the left foreground edge, while the right arm is on the"
    " right side. Bright diffuse indoor lighting creates warm highlights on the tan wood and soft shadows from the"
    " arms and objects, with a palette of warm wood, turquoise, blue-white packaging, brown cardboard, red, black"
    " hardware and pale gray tile, giving the setup a controlled, technical demonstration quality. Both grippers move"
    " smoothly forward and down over the table."
)
"""Task description of every record, in the style of the checkpoint's AgiBot prompts."""

ARM_OFFSET = 0.15
"""Largest offset of each arm joint from the task's start pose [rad]."""

DARK, WOOD = (0.01, 0.01, 0.012), (0.55, 0.33, 0.16)
"""Diffuse colors of the arm and gripper links and of the table top."""

DARK_LINKS = (
    [f"Link{i}_{side}" for i in range(1, 8) for side in "lr"]
    + ["base_link_l", "base_link_r", "left_base_link", "right_base_link"]
    + [
        f"{side}_{finger}_{part}_Link"
        for side in ("left", "right")
        for finger in ("Left", "Right")
        for part in ("01", "00", "Support", "Pad", "2")
    ]
)
"""A2D links that get the dark material: the arms, the gripper bases and the fingers."""

OBJECTS = {
    "Packet": ((0.12, 0.08, 0.015), (0.97, 0.12), 25.0, (0.75, 0.82, 0.95)),
    "Bottle": ((0.05, 0.05, 0.14), (1.02, -0.04), 0.0, (0.1, 0.65, 0.62)),
    "Carton": ((0.14, 0.1, 0.08), (0.98, -0.36), -15.0, (0.55, 0.38, 0.22)),
    "RedBox": ((0.06, 0.06, 0.05), (1.12, 0.38), 10.0, (0.75, 0.08, 0.06)),
}
"""Small objects on the table: size [m], position in the robot's base frame [m], yaw [deg] and color."""


@configclass
class RenderSceneCfg(task.AgibotReachSceneCfg):
    """The task's camera and the full A2D, with the table, objects, floor and lights the start images show."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(color=(0.95, 0.93, 0.9), intensity=750.0)
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight", spawn=sim_utils.DistantLightCfg(intensity=600.0, angle=15.0)
    )


def add_table_and_objects(scene_cfg: RenderSceneCfg, top: float) -> None:
    """Add the wood table, its near edge 0.6 m in front of the robot, and the objects on it."""
    depth, width = 0.7, 1.6
    scene_cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6 + depth / 2, 0.0, top / 2)),
        spawn=sim_utils.CuboidCfg(
            size=(depth, width, top), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=WOOD, roughness=0.6)
        ),
    )
    for name, (size, (x, y), yaw, color) in OBJECTS.items():
        half = np.radians(yaw) / 2
        setattr(
            scene_cfg,
            name.lower(),
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(x, y, top + size[2] / 2), rot=(0.0, 0.0, float(np.sin(half)), float(np.cos(half)))
                ),
                spawn=sim_utils.CuboidCfg(
                    size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.5)
                ),
            ),
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
    # The full A2D, root fixed at the env origin, without gravity so that it holds the recorded pose exactly.
    scene_cfg.robot = replace(
        AGIBOT_A2D_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=replace(AGIBOT_A2D_CFG.spawn, rigid_props=PhysxRigidBodyCfg(disable_gravity=True)),
        init_state=replace(AGIBOT_A2D_CFG.init_state, pos=(0.0, 0.0, 0.0)),
    )
    # The task's camera, rendered with OVRTX instead of imagined by the world model.
    scene_cfg.camera = replace(scene_cfg.camera, renderer_cfg=OVRTXRendererCfg(), frame_history_length=0)
    add_table_and_objects(scene_cfg, args_cli.table_top)
    with launch_simulation((sim_cfg, scene_cfg), args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = instantiate(scene_cfg)
        stage = sim_utils.get_current_stage()
        robot_path = "/World/envs/env_0/Robot"
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=DARK, roughness=0.45, metallic=0.1)
        dark.func("/World/Looks/DarkArm", dark)
        for link in DARK_LINKS:
            sim_utils.bind_visual_material(f"{robot_path}/{link}", "/World/Looks/DarkArm", stage=stage)
        for link in ("link_pitch_head", "link_yaw_head"):
            UsdGeom.Imageable(stage.GetPrimAtPath(f"{robot_path}/{link}")).MakeInvisible()
        sim.reset()
        robot, camera = scene["robot"], scene["camera"]
        names = list(robot.joint_names)
        task_ids = [names.index(name) for name in AGIBOT_JOINT_NAMES]
        supports = ("Left_Support_Joint", "Right_Support_Joint")
        grippers = [names.index(f"{side}_{joint}") for side in ("left", "right") for joint in supports]
        bases = [robot.find_bodies(f"{side}_base_link")[0][0] for side in ("left", "right")]
        limits = robot.data.soft_joint_pos_limits.torch[0]
        start = robot.data.default_joint_pos.torch[0].clone()
        held = {"joint_lift_body": WAIST_LIFT, "joint_body_pitch": WAIST_PITCH, "joint_head_yaw": 0.0}
        # The A2D's head pitch turns the other way round than the G1 description's.
        held["joint_head_pitch"] = -HEAD_PITCH
        for name, value in {**task.START_POSE, **held}.items():
            start[names.index(name)] = value
        # The support joints follow the gripper drives, which START_POSE opens.
        start[grippers] = GRIPPER_OPEN
        generator = torch.Generator(device=start.device).manual_seed(args_cli.seed)

        def hold(q: torch.Tensor, steps: int) -> None:
            robot.write_joint_position_to_sim_index(position=q[None])
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q)[None])
            robot.set_joint_position_target_index(target=q[None])
            settle(sim, scene, steps)

        # The first frames stream in the assets and textures.
        hold(start, 120)

        records, fk_error = [], 0.0
        task.ASSETS.mkdir(exist_ok=True)
        splits = ["train"] * args_cli.num_train + ["validation"] * args_cli.num_validation
        counts = dict.fromkeys(splits, 0)
        for split in splits:
            q = start.clone()
            arms = task_ids[:14]
            noise = torch.rand(14, generator=generator, device=start.device) * 2 - 1
            q[arms] = torch.clamp(q[arms] + ARM_OFFSET * noise, limits[arms, 0], limits[arms, 1])
            hold(q, 60)

            state = robot.data.joint_pos.torch[0, task_ids]
            root_pos, root_quat = robot.data.root_link_pos_w.torch[0], robot.data.root_link_quat_w.torch[0]
            # The point between the fingertips lies TCP_OFFSET along each gripper base's z axis, in the root frame.
            base_z = quat_apply(robot.data.body_link_quat_w.torch[0, bases], torch.eye(3, device=start.device)[[2, 2]])
            tcp_w = robot.data.body_link_pos_w.torch[0, bases] + TCP_OFFSET * base_z
            simulated = quat_apply_inverse(root_quat.expand(2, 4), tcp_w - root_pos)
            error = torch.linalg.norm(tcp_positions(state.double()).float() - simulated, dim=-1)
            fk_error = max(fk_error, error.max().item())
            name = f"{split}_{counts[split]:02d}.png"
            counts[split] += 1
            Image.fromarray(camera.data.output["rgb"][0, ..., :3].cpu().numpy()).save(task.ASSETS / name)
            records.append({"image": name, "state": state.cpu().tolist(), "prompt": PROMPT, "split": split})
            print(
                f"[INFO] {name}: joint state {np.round(records[-1]['state'], 3).tolist()}; TCP error left"
                f" {error[0] * 1000:.1f} mm, right {error[1] * 1000:.1f} mm"
            )

        (task.ASSETS / "resets.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"[INFO] Wrote {len(records)} reset records to {task.ASSETS / 'resets.json'}.")
        print(f"[INFO] Largest forward-kinematics error of the fingertip centres: {fk_error * 1000:.2f} mm.")
        if fk_error > 1e-2:
            raise RuntimeError("The task's forward kinematics does not match the simulated A2D grippers.")


if __name__ == "__main__":
    main()
