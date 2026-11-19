# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics backend that replaces a physics engine with a learned world model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat
from isaaclab.sim.views import FrameView, UsdFrameView
from isaaclab.utils.configclass import instantiate

from ..world_model import load_reset_records

if TYPE_CHECKING:
    from ..world_model import WorldModel, WorldModelSpec
    from .world_model_manager_cfg import WorldModelCfg

# There is no physics engine to move prims, so frame prims such as cameras are read from the USD stage. Registering
# the class here lets ``FrameView`` resolve it for this backend without a backend-specific views package.
FrameView.register("neural", UsdFrameView)


class WorldModelManager(PhysicsManager):
    """Own a world model and each environment's joint state.

    The joint state is the reset record's state at reset and the last commanded joint target after that; the model
    never outputs joint state. The model runs once per chunk: every :meth:`step` buffers one joint target per
    environment, and the step that completes a chunk predicts all of the chunk's frames at once. Environments can
    therefore only reset between chunks.

    Joint state, targets and frames are torch tensors on the simulation device. The articulation reads and writes
    :attr:`joint_state` and :attr:`joint_target` in place, so an environment step copies nothing to the host.
    """

    backend_name = "neural"
    _cfg: WorldModelCfg

    def __init__(self, cfg: WorldModelCfg):
        super().__init__(cfg)
        self.model: WorldModel | None = None
        self.spec: WorldModelSpec | None = None
        self.num_envs = 0
        self.num_joints = 0
        self.joint_state: torch.Tensor
        """Joint positions [m or rad, depending on joint type], shape ``(N, J)``."""
        self.joint_target: torch.Tensor
        """Joint position targets applied by the next :meth:`step`, shape ``(N, J)``."""
        self.frame_history: torch.Tensor
        """RGB frames of the last completed chunk, shape ``(N, T, H, W, 3)``, ``uint8``. After a reset, every frame
        is the start frame."""
        self.frame_serial: torch.Tensor
        """Per-environment counter that increases whenever the environment's frames change, shape ``(N,)``."""
        self.frame_version = 0
        """Host counter that increases whenever any environment's frames change, so readers skip steps without new
        frames without reading the device."""
        self.new_frames = False
        """Whether the last :meth:`step` completed a chunk."""
        self._records = load_reset_records(cfg.reset_manifest)
        self._split_records = [r for r in self._records if r.split == cfg.split]
        self._rng = np.random.default_rng(cfg.seed)
        self._cursor = 0
        self._draws = 0
        self._record_states: torch.Tensor
        self._targets: torch.Tensor
        self._chunk_start: torch.Tensor
        self._steps_in_chunk = 0
        self._scene_data: SceneDataBackend

    def bind(self, num_envs: int, num_joints: int) -> None:
        """Size the manager for a scene. Called by the articulation when it initializes."""
        if num_envs % self._cfg.envs_per_reset_record:
            raise ValueError(f"envs_per_reset_record={self._cfg.envs_per_reset_record} must divide num_envs={num_envs}")
        self.model = instantiate(self._cfg.model, num_envs, self._device)
        self.spec = self.model.spec()
        if not math.isclose(self._sim.cfg.dt, 1.0 / self.spec.fps):
            raise ValueError(f"sim.dt must equal 1 / fps = {1.0 / self.spec.fps} s, got {self._sim.cfg.dt}")
        self.num_envs, self.num_joints = num_envs, num_joints
        device = self._device
        self.joint_state = torch.zeros((num_envs, num_joints), device=device)
        self.joint_target = torch.zeros_like(self.joint_state)
        self.frame_history = torch.zeros(
            (num_envs, self.spec.chunk_length, *self.spec.image_size, 3), dtype=torch.uint8, device=device
        )
        self.frame_serial = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self._record_states = torch.from_numpy(np.stack([r.joint_state for r in self._split_records])).to(device)
        self._targets = torch.zeros((num_envs, self.spec.chunk_length, num_joints), device=device)
        self._chunk_start = torch.zeros_like(self.joint_state)

    def initialize(self, sim_context) -> None:
        super().initialize(sim_context)
        # World-model frames are not rigid-body poses, so there is no scene data to publish.
        self._scene_data = _EmptySceneData(self._device)

    def reset(self, soft: bool = False) -> None:
        """Dispatch the lifecycle events. Episodes start with :meth:`reset_envs`, called by the articulation's reset."""
        if not soft:
            self.dispatch_event(PhysicsEvent.MODEL_INIT)
        self.dispatch_event(PhysicsEvent.PHYSICS_READY)

    def reset_envs(self, env_ids: Sequence[int]) -> None:
        """Start new episodes in the selected environments from sampled reset records.

        Records are image files and prompts on the host, so the environments are given as host integers.

        Raises:
            RuntimeError: If called in the middle of a chunk.
        """
        if self._steps_in_chunk:
            raise RuntimeError(
                f"Environments can only reset on chunk boundaries, but {self._steps_in_chunk} of "
                f"{self.spec.chunk_length} steps of the chunk have been taken."
            )
        env_ids = list(env_ids)
        indices = self._sample(env_ids)
        records = [self._split_records[i] for i in indices]
        self._draws += 1
        seeds = [(self._cfg.seed + 997 * self._draws + env) % 2**32 for env in env_ids]
        frames = self.model.reset(env_ids, [r.image for r in records], [r.prompt for r in records], seeds)
        rows = torch.tensor(env_ids, device=self._device)
        self.frame_history[rows] = frames[:, None]
        self.frame_serial[rows] += 1
        self.frame_version += 1
        state = self._record_states[torch.tensor(indices, device=self._device)]
        self.joint_state[rows] = state
        self.joint_target[rows] = state
        self.new_frames = False

    def step(self) -> None:
        """Advance one frame; on the last frame of a chunk, predict the chunk with the model."""
        if not self._steps_in_chunk:
            self._chunk_start.copy_(self.joint_state)
        self._targets[:, self._steps_in_chunk] = self.joint_target
        self.joint_state.copy_(self.joint_target)
        self._steps_in_chunk += 1
        self.new_frames = self._steps_in_chunk == self.spec.chunk_length
        if self.new_frames:
            self.frame_history.copy_(self.model.step(self._targets, self._chunk_start))
            self.frame_serial += 1
            self.frame_version += 1
            self._steps_in_chunk = 0
        self._sim_time += self.get_physics_dt()

    def forward(self) -> None:
        pass

    def get_scene_data_backend(self) -> SceneDataBackend:
        return self._scene_data

    def close(self) -> None:
        super().close()
        if self.model is not None:
            self.model.close()

    def _sample(self, env_ids: list[int]) -> list[int]:
        """Pick a reset record per environment and return their indices. Environments of one reset-record group share
        a draw, so a group-relative algorithm compares rollouts of the same scene. Training draws at random; other
        splits are visited in order."""
        group_draws: dict[int, int] = {}
        for group in sorted({env // self._cfg.envs_per_reset_record for env in env_ids}):
            if self._cfg.split == "train":
                group_draws[group] = int(self._rng.integers(len(self._split_records)))
            else:
                group_draws[group] = self._cursor % len(self._split_records)
                self._cursor += 1
        return [group_draws[env // self._cfg.envs_per_reset_record] for env in env_ids]


class _EmptySceneData(SceneDataBackend):
    """Scene data without transforms."""

    def __init__(self, device: str):
        self.transforms_timestamp = 0
        self._transforms = SceneDataFormat.Transform()
        self._transforms.transforms = wp.empty(0, dtype=wp.transformf, device=device)

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        return self._transforms

    @property
    def transform_count(self) -> int:
        return 0

    @property
    def transform_paths(self) -> list[str]:
        return []
