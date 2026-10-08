# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a policy in an Isaac Lab task whose camera observation comes from an image transfer model.

The task is created with ``gym.make``. Its policy camera renders at the model canvas and runs the camera
modifier chain controls -> image transfer -> PPISP -> resize, so the policy observes generated images at
the resolution it expects. Episodes end and reset as usual; each reset starts a new model episode.

By default the model is the stand-in worker, which tints edge images, so the example runs anywhere:

.. code-block:: bash

    python examples/sensors/image_transfer_policy.py --output output/image-transfer-policy --episode-seconds 2

Pass ``--worker-command`` with a JSON argv list to use another worker, for example a learned model in its
own environment. Without ``--checkpoint`` the policy takes uniform random actions.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

STAND_IN_WORKER = Path(__file__).resolve().parent / "image_transfer_stand_in_worker.py"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--output", type=Path, required=True, help="New directory for the run.")
parser.add_argument("--worker-command", type=Path, help="JSON argv list starting the worker; defaults to the stand-in.")
parser.add_argument("--task", default="Isaac-Reorient-KukaAllegro-Camera")
parser.add_argument("--presets", default="cube,single_camera,newton_mjwarp,newton_renderer,rgb64")
parser.add_argument("--camera", default="base_camera", help="Scene camera whose rgb the policy observes.")
parser.add_argument("--checkpoint", type=Path, help="RSL-RL checkpoint; random actions without it.")
parser.add_argument("--episodes", type=int, default=3, help="Completed episodes to play.")
parser.add_argument("--episode-seconds", type=float, help="Override the task's episode length [s].")
parser.add_argument("--prompt", help="Text description of the images to generate.")
parser.add_argument("--initial-frames", type=int, default=1, help="Captures in the first chunk of an episode.")
parser.add_argument("--update-frames", type=int, default=4, help="Captures in each later chunk.")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True


def display_ppisp_cfg(exposure_ev: float = 0.3, vignetting: float = -0.5):
    """PPISP settings with a display gamma of 1/2.2 in the camera response, for sRGB-encoded output."""
    from isaaclab_ppisp import PpispCfg

    # The response applies gamma = 0.1 + softplus(raw); this raw value gives 1/2.2.
    gamma_raw = math.log(math.exp(1.0 / 2.2 - 0.1) - 1.0)
    inputs = {"exposureOffset": exposure_ev}
    for channel in "RGB":
        inputs[f"crfGamma{channel}"] = gamma_raw
        inputs[f"vignettingAlpha1{channel}"] = vignetting
    return PpispCfg(inputs=inputs)


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_experimental.image_transfer import (
        ImageTransferModifierCfg,
        WorkerModelCfg,
        image_transfer_camera,
        srgb_to_linear,
    )
    from isaaclab_ppisp import PpispModifierCfg
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.sim import SimulationContext
    from isaaclab.utils import to_dict
    from isaaclab.utils.modifiers import ModifierCfg

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    args.output.mkdir(parents=True, exist_ok=False)
    env_cfg, agent_cfg = resolve_task_config(
        args.task, "rsl_rl_cfg_entry_point", play_mode=True, overrides=[f"presets={args.presets}"]
    )
    env_cfg.scene.num_envs = 1
    if args.episode_seconds is not None:
        env_cfg.episode_length_s = args.episode_seconds
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    steps_per_episode = math.ceil(env_cfg.episode_length_s / step_dt)
    command = (
        json.loads(args.worker_command.read_text()) if args.worker_command else [sys.executable, str(STAND_IN_WORKER)]
    )
    worker = WorkerModelCfg(
        command=command,
        prompt=args.prompt,
        control="edge",
        # One capture per environment step, plus the reset step, rounded up to whole chunks.
        max_episode_frames=args.initial_frames + args.update_frames * math.ceil(steps_per_episode / args.update_frames),
        log_path=str(args.output / "worker.log"),
        record_dir=str(args.output / "generated_chunks"),
    )
    transfer = ImageTransferModifierCfg(
        backend=worker, seed=42, initial_frames=args.initial_frames, update_frames=args.update_frames
    )
    camera = getattr(env_cfg.scene, args.camera)
    policy_shape = (camera.height, camera.width, 3)
    setattr(
        env_cfg.scene,
        args.camera,
        image_transfer_camera(
            camera,
            transfer,
            control="edge",
            post_modifiers=[ModifierCfg(func=srgb_to_linear), PpispModifierCfg(isp_cfg=display_ppisp_cfg())],
        ),
    )

    with launch_simulation(env_cfg, args):
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        if args.checkpoint is not None:
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(str(args.checkpoint))
            policy = runner.get_inference_policy(device=env.unwrapped.device)
        else:
            generator = torch.Generator(device=env.unwrapped.device).manual_seed(0)

            def policy(_obs):
                shape = (env.num_envs, env.num_actions)
                return torch.rand(shape, generator=generator, device=env.unwrapped.device) * 2 - 1

        sensor = env.unwrapped.scene[args.camera]
        obs = env.get_observations()
        observed, episode_ends, step_seconds = [], [], []
        while len(episode_ends) < args.episodes:
            started = time.perf_counter()
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            if hasattr(policy, "reset"):
                policy.reset(dones)
            step_seconds.append(time.perf_counter() - started)
            observed.append(sensor.data.output["rgb"].torch[0].cpu().numpy().copy())
            if bool(dones[0]):
                episode_ends.append(len(observed))
                print(f"Episode {len(episode_ends)}/{args.episodes} ended at step {len(observed)}", flush=True)

        stream = SimulationContext.instance().get_or_create_backend(worker).streams[0]
        observed = np.stack(observed)
        if observed.shape[1:] != policy_shape:
            raise AssertionError(f"The policy observed {observed.shape[1:]}, expected {policy_shape}.")
        if stream.resets < args.episodes - 1:
            raise AssertionError(f"Expected at least {args.episodes - 1} model episode resets, got {stream.resets}.")
        np.savez_compressed(args.output / "policy_observations.npz", rgb=observed, episode_ends=episode_ends)
        report = {
            "status": "passed",
            "task": args.task,
            "presets": args.presets,
            "worker": command,
            "policy": str(args.checkpoint) if args.checkpoint else "uniform random actions",
            "episodes": len(episode_ends),
            "episode_end_steps": episode_ends,
            "model_episode_resets": stream.resets,
            "generated_frames": stream.recorded_frames,
            "policy_image_shape": list(policy_shape),
            "mean_step_seconds": float(np.mean(step_seconds)),
            "chunk_timings": stream.chunk_timings,
        }
        (args.output / "policy-validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({key: value for key, value in report.items() if key != "chunk_timings"}, indent=2))
        env.close()


if __name__ == "__main__":
    main()
