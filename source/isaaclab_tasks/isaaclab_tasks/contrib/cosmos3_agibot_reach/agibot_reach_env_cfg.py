# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AgiBot A2D bimanual reach, with the public Cosmos3 Sim-Bimanual checkpoint imagining the head camera's view instead
of physics.

The task is written like any manager-based task: the A2D, its head camera, joint-position actions on both arms, joint
and image observations and a reach reward per gripper. One environment step is one chunk: the policy acts once per
chunk of frames. The reward follows the grippers from the joint state with forward kinematics, so it does not depend
on the generated frames. The ``physics=`` preset selects the world-model backend; no physics engine runs.
Its reset records are start images that Isaac Lab rendered with OvPhysX and OVRTX, without Kit
(``scripts/tools/cosmos/render_agibot_reset_records.py``).
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg, action_rate_l2, image_rgb, joint_pos_rel, time_out
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.utils import configclass, replace

from isaaclab_neural.models.cosmos3 import Cosmos3Cfg
from isaaclab_neural.physics import NeuralCfg
from isaaclab_neural.renderers import NeuralRendererCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets import AGIBOT_A2D_CFG

from .agibot_mapping import AGIBOT_JOINT_NAMES, GRIPPER_OPEN, AgibotJointMappingCfg
from .mdp import ResetPoseJointPositionAction, TcpGoalDistance

ASSETS = Path(__file__).with_name("assets")
CHECKPOINT = ".pretrained_checkpoints/Cosmos3-Nano-Sim-Bimanual"
"""Where ``hf download nvidia/Cosmos3-Nano-Sim-Bimanual --local-dir`` puts the checkpoint, relative to the working
directory."""

CHUNK = 16
"""Frames the world model generates, and joint targets it consumes, per model step."""

FPS = 30
IMAGE_SIZE = (352, 640)
"""Height and width of the start images and of the generated frames: the checkpoint's 16:9 AgiBot examples, at the
height its video tokenizer generates."""

START_POSE = {
    **{f"left_arm_joint{i + 1}": q for i, q in enumerate((-1.006, 0.509, 0.285, -1.134, 0.730, 1.238, -0.220))},
    **{f"right_arm_joint{i + 1}": q for i, q in enumerate((1.748, -0.735, -1.570, 0.814, -0.134, -1.244, -0.177))},
    "left_hand_joint1": GRIPPER_OPEN,
    "right_hand_joint1": GRIPPER_OPEN,
}
"""Arm joint positions [rad] at the start of an AgiBot World episode, with both grippers open."""

LEFT_GOAL, RIGHT_GOAL = (0.80, 0.25, 0.82), (0.80, -0.15, 0.82)
"""Positions the points between the left and right fingertips reach for, in the robot's base frame [m]: just above
the table, forward of where they start."""


@configclass
class AgibotReachPhysicsCfg(PresetCfg):
    """Backends of the task. Only the world model drives it, so it is also the default.

    The Cosmos3 service serves the checkpoint in session mode (``--mode session``), which keeps the distilled
    checkpoint's causal history and samples with its fixed four-step schedule; it serves one environment.
    """

    default: NeuralCfg = NeuralCfg(
        model_cfg=Cosmos3Cfg(
            domain_name="agibotworld",
            joint_mapping=AgibotJointMappingCfg(
                global_stats_path=(
                    f"{CHECKPOINT}/cosmos3_nano_sim_bimanual_action_sources/"
                    "ae695e2c643a5626b3c3564f00705aa3a83ab0da7640549c7340db71172594b0.json"
                )
            ),
        ),
        reset_manifest=str(ASSETS / "resets.json"),
    )


@configclass
class AgibotReachSceneCfg(InteractiveSceneCfg):
    """The A2D and the head camera whose view the world model imagines.

    The camera sits where the calibrated head camera of the AgiBot World G1 is at the episode's waist and head pose
    (``head_camera_link`` of ``G1_omnipicker_calibrated.urdf``), looking 46 degrees down at the table between the
    grippers. Its 95-degree horizontal field of view lines the rendered grippers up with a frame of the episode.
    ``render_agibot_reset_records.py`` renders the start images with this robot and camera.
    """

    robot: ArticulationCfg = replace(
        AGIBOT_A2D_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        joint_ordering=AGIBOT_JOINT_NAMES,
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=START_POSE),
    )
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=IMAGE_SIZE[0],
        width=IMAGE_SIZE[1],
        data_types=["rgb"],
        frame_history_length=CHUNK,
        spawn=PinholeCameraCfg(focal_length=9.6, clipping_range=(0.01, 20.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.491, -0.0157, 1.2486), rot=(-0.6502, 0.659, -0.2617, 0.2728), convention="ros"
        ),
        renderer_cfg=NeuralRendererCfg(),
    )


@configclass
class ActionsCfg:
    """Joint-position targets of both arms around the reset record's pose, over one chunk.

    The policy sets one target per joint, which the arms reach in equal increments over the chunk's frames. With
    ``interpolate=False`` it sets all 16 targets of the chunk instead. The grippers stay open.
    """

    arms = JointPositionActionCfg(
        class_type=ResetPoseJointPositionAction,
        asset_name="robot",
        joint_names=["left_arm_joint.*", "right_arm_joint.*"],
        scale=0.5,
        use_default_offset=True,
        chunk_length=CHUNK,
        interpolate=True,
    )


@configclass
class ObservationsCfg:
    """Joint positions and the camera's image, in separate groups so an RL library can encode the image."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=joint_pos_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False

    @configclass
    class ImageCfg(ObsGroup):
        rgb = ObsTerm(
            func=image_rgb,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb", "channel_first": True},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    image: ImageCfg = ImageCfg()


@configclass
class EventCfg:
    """No reset event: the world-model articulation starts each episode from a reset record when it resets.

    This replaces the default ``reset_scene_to_default``, which also resets the root pose, and the world model has
    none."""


@configclass
class RewardsCfg:
    """The reach rewards of the reach task for each gripper, on the forward kinematics of the joint state."""

    left_tcp_distance = RewTerm(func=TcpGoalDistance, weight=-0.2, params={"goal": LEFT_GOAL, "arm": 0})
    left_tcp_distance_tanh = RewTerm(func=TcpGoalDistance, weight=0.1, params={"goal": LEFT_GOAL, "arm": 0, "std": 0.1})
    right_tcp_distance = RewTerm(func=TcpGoalDistance, weight=-0.2, params={"goal": RIGHT_GOAL, "arm": 1})
    right_tcp_distance_tanh = RewTerm(
        func=TcpGoalDistance, weight=0.1, params={"goal": RIGHT_GOAL, "arm": 1, "std": 0.1}
    )
    action_rate = RewTerm(func=action_rate_l2, weight=-1.0e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class AgibotReachEnvCfg(ManagerBasedRLEnvCfg):
    """Move both A2D grippers to :data:`LEFT_GOAL` and :data:`RIGHT_GOAL` while Cosmos3 imagines the head camera's
    view."""

    scene: AgibotReachSceneCfg = AgibotReachSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # One environment step is one chunk: a frame and a joint target per physics step, then the model runs once.
        self.decimation = CHUNK
        self.sim.dt = 1.0 / FPS
        self.sim.render_interval = self.decimation
        # 8 chunks. Every step ends on a chunk boundary, where the world model can be reset.
        self.episode_length_s = 8 * CHUNK / FPS
        self.sim.physics = AgibotReachPhysicsCfg()
