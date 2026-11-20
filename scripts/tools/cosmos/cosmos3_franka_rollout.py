# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the world-model Franka reach task and record what the world model sees and generates.

The arm follows a scripted sweep around each reset record's pose, or a policy trained with
``isaaclab train --rl_library rsl_rl``. The video shows each episode's start image, which Isaac Lab rendered with
RTX, followed by the frames Cosmos3 generated; episodes time out every 8 chunks, so a long run passes through resets.
The script prints the model's time per chunk and writes it to ``timing.json``.

Start the Cosmos3 service first, in the Cosmos Framework environment, with
``isaaclab-cosmos-world-model-server --checkpoint .pretrained_checkpoints/Cosmos3-Nano-Sim-Bimanual --mode session``.

Usage:

.. code-block:: bash

    uv sync --extra video  # imageio, through moviepy
    uv run python scripts/tools/cosmos/cosmos3_franka_rollout.py --output outputs/cosmos3/franka_rollout
"""

import argparse
import json
import math
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-Cosmos3-Franka-Reach-v0", help="Name of the task.")
parser.add_argument("--chunks", type=int, default=20, help="Number of model chunks to run.")
parser.add_argument("--checkpoint", type=Path, help="rsl_rl checkpoint of a trained policy; a scripted sweep if unset.")
parser.add_argument("--output", type=Path, required=True, help="Directory for the video and the timing.")
add_launcher_args(parser)
args_cli = parser.parse_args()

import imageio.v2 as imageio
import torch

from isaaclab.utils import instantiate, to_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def record_model(model, writer, timing: list[float]) -> None:
    """Write environment 0's start images and generated chunks to ``writer``, and time each chunk."""
    reset, step = model.reset, model.step

    def recording_reset(env_ids, images, prompts, seeds):
        if 0 in env_ids:
            writer.append_data(images[env_ids.index(0)].cpu().numpy())
        reset(env_ids, images, prompts, seeds)

    def recording_step(joint_targets, start_joint_state):
        start = time.perf_counter()
        frames = step(joint_targets, start_joint_state)
        recorded = frames[0].cpu().numpy()
        timing.append(time.perf_counter() - start)
        for frame in recorded:
            writer.append_data(frame)
        return frames

    model.reset, model.step = recording_reset, recording_step


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)
    with launch_simulation(env_cfg, args_cli):
        env = instantiate(env_cfg)
        model = env.sim.backend_manager.model
        spec = model.spec()
        args_cli.output.mkdir(parents=True)
        timing = []
        with imageio.get_writer(args_cli.output / "rollout.mp4", fps=spec.fps, macro_block_size=1) as writer:
            record_model(model, writer, timing)
            if args_cli.checkpoint is None:
                obs, _ = env.reset()

                def policy(_obs):
                    # Sweep the base and the elbow over two seconds; actions are offsets from the record's pose.
                    phase = 2 * math.pi * env.episode_length_buf.float()[:, None] * env.step_dt / 2.0
                    action = torch.zeros_like(env.action_manager.action)
                    action[:, 0] = 0.8 * torch.sin(phase[:, 0])
                    action[:, 3] = 0.6 * torch.sin(2 * phase[:, 0])
                    return action

            else:
                from rsl_rl.runners import OnPolicyRunner

                from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

                agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=env.unwrapped.device)
                runner.load(str(args_cli.checkpoint))
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                obs = env.get_observations()
            with torch.inference_mode():
                for _ in range(args_cli.chunks * spec.chunk_length):
                    obs = env.step(policy(obs))[0]
        env.close()

    # The first chunk includes the session's start, so it is reported apart.
    steady = timing[1:] or timing
    summary = {
        "chunk_length": spec.chunk_length,
        "fps": spec.fps,
        "chunks": len(timing),
        "first_chunk_s": timing[0],
        "mean_chunk_s": sum(steady) / len(steady),
        "chunk_s": timing,
    }
    (args_cli.output / "timing.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[INFO] {len(timing)} chunks of {spec.chunk_length} frames: first {timing[0]:.2f} s, then"
        f" {summary['mean_chunk_s']:.2f} s per chunk ({summary['mean_chunk_s'] / spec.chunk_length * 1000:.0f} ms per"
        f" frame). Video: {args_cli.output / 'rollout.mp4'}"
    )


if __name__ == "__main__":
    main()
