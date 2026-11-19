# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos3 behind the :class:`~isaaclab_neural.world_model.WorldModel` interface, served by a separate process.

The Cosmos service (``isaaclab-cosmos-world-model-server``) loads the Cosmos Framework in its own environment, so
Isaac Lab does not install it. This client maps joint targets to controls with the task's
:class:`~.cosmos3_cfg.JointMappingCfg` and exchanges controls and frames with the service.
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab_experimental.cosmos._protocol import check_reply, connect, receive_message, request, send_message

from isaaclab.utils.configclass import instantiate

from ...world_model import WorldModel, WorldModelSpec

if TYPE_CHECKING:
    from pathlib import Path

    from .cosmos3_cfg import Cosmos3Cfg


class Cosmos3Model(WorldModel):
    """Client of a Cosmos3 world-model service; one session serves all environments."""

    cfg: Cosmos3Cfg

    def __init__(self, cfg: Cosmos3Cfg, num_envs: int, device: str):
        super().__init__(cfg, num_envs, device)
        self.mapping = None
        self._spec: WorldModelSpec | None = None
        self._socket: socket.socket | None = None

    def spec(self) -> WorldModelSpec:
        """Read the chunk length, frame rate and image size from the service."""
        if self._spec is None:
            reply, _ = request(self.cfg.endpoint, {"op": "status"}, timeout=self.cfg.timeout)
            capabilities = reply.get("capabilities", {})
            if capabilities.get("kind") != "world_model":
                raise RuntimeError(f"The Cosmos service at {self.cfg.endpoint} does not serve a world model.")
            self._spec = WorldModelSpec(
                chunk_length=capabilities["chunk_length"],
                fps=capabilities["fps"],
                image_size=tuple(capabilities["image_size"]),
                max_envs=capabilities["max_envs"],
            )
        return self._spec

    def reset(self, env_ids: list[int], images: list[Path], prompts: list[str], seeds: list[int]) -> torch.Tensor:
        """Send the start images and prompts of the selected environments; return their first frames."""
        from PIL import Image

        if self._socket is None:
            self._open()
        start = np.stack([np.asarray(Image.open(path).convert("RGB")) for path in images])
        message = {"op": "reset", "env_ids": list(env_ids), "prompts": list(prompts), "seeds": list(seeds)}
        return self._exchange(message, start)

    def step_controls(self, controls: np.ndarray) -> torch.Tensor:
        """Predict the next chunk from normalized ``unified_v1`` controls shaped ``(N, T, 59)``.

        Replay of recorded controls uses this directly; the manager goes through :meth:`step`.
        """
        return self._exchange({"op": "step"}, np.ascontiguousarray(controls, dtype=np.float32))

    def step(self, joint_targets: torch.Tensor, start_joint_state: torch.Tensor) -> torch.Tensor:
        """Convert joint targets to controls with the task's joint mapping, then predict the chunk.

        The mapping runs on the host and the controls cross to the service there, so the targets are read once per
        chunk.
        """
        if self.mapping is None:
            self.mapping = instantiate(self.cfg.joint_mapping)
        targets, start = joint_targets.cpu().numpy(), start_joint_state.cpu().numpy()
        return self.step_controls(np.stack([self.mapping.encode(targets[i], start[i]) for i in range(self.num_envs)]))

    def close(self) -> None:
        """Close the session; the service keeps the weights loaded for the next client."""
        if self._socket is None:
            return
        try:
            send_message(self._socket, {"op": "close"})
            receive_message(self._socket)
        except (OSError, ValueError):
            pass
        finally:
            self._socket.close()
            self._socket = None

    def _open(self) -> None:
        self._socket = connect(self.cfg.endpoint, self.cfg.timeout)
        try:
            send_message(self._socket, {"op": "open", "num_envs": self.num_envs})
            check_reply(receive_message(self._socket)[0])
        except Exception:
            self._socket.close()
            self._socket = None
            raise

    def _exchange(self, message: dict, array: np.ndarray) -> torch.Tensor:
        """Send one request with one array and return the frames of the reply on :attr:`device`.

        A failure closes the session: the service may already have advanced the model's state.
        """
        if self._socket is None:
            raise RuntimeError("The Cosmos world-model session is closed or failed; reset the environments again.")
        try:
            send_message(self._socket, message, [array])
            reply, frames = receive_message(self._socket)
            check_reply(reply)
            if len(frames) != 1 or frames[0].dtype != np.uint8:
                raise RuntimeError("The Cosmos world model must reply with one uint8 frame array.")
            return torch.from_numpy(frames[0]).to(self.device)
        except Exception:
            self._socket.close()
            self._socket = None
            raise
