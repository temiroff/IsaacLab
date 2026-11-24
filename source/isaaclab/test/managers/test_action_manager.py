# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Tests for action chunks in the action manager. They need no Isaac Sim launch."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import ActionManager, ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.unit

NUM_ENVS = 2
DECIMATION = 4


class _Command(ActionTerm):
    """A term that logs when each command is processed, with the environment's physics step, and what it applies."""

    is_graphable = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(env.num_envs, cfg.dim)
        self._processed = torch.zeros_like(self._raw)
        self.processed_at: list[tuple[int, float]] = []
        self.applied: list[float] = []

    @property
    def action_dim(self) -> int:
        return self.cfg.dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._processed[:] = 2 * actions
        self.processed_at.append((self._env.physics_step, actions[0, 0].item()))

    def apply_actions(self):
        self.applied.append(self._processed[0, 0].item())


@configclass
class _CommandCfg(ActionTermCfg):
    class_type: type[ActionTerm] = _Command
    asset_name: str = "robot"
    dim: int = 1


def _manager(**chunk_lengths: int) -> ActionManager:
    """An action manager with one term per keyword; its value is the term's chunk length and its dim is 2 for arm."""
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        device="cpu",
        cfg=SimpleNamespace(decimation=DECIMATION),
        scene={"robot": object()},
        sim=SimpleNamespace(is_playing=lambda: True),
        physics_step=0,
    )
    cfg = {
        name: _CommandCfg(dim=2 if name == "arm" else 1, chunk_length=length) for name, length in chunk_lengths.items()
    }
    manager = ActionManager(SimpleNamespace(**cfg), env)
    manager._test_env = env
    return manager


def _env_step(manager: ActionManager, action: list[float]) -> None:
    """One environment step: process the action once, then apply it on every physics step."""
    manager._test_env.physics_step = 0
    manager.process_action(torch.tensor([action] * NUM_ENVS))
    for step in range(DECIMATION):
        manager._test_env.physics_step = step
        manager.apply_action()


def test_terms_apply_their_commands_in_order_and_restart_with_the_next_action() -> None:
    """Each term processes its next command at the physics step where it starts; the next action restarts the chunks."""
    manager = _manager(arm=2, gripper=4, plain=1)
    # the arm takes 2 commands of 2 values, the gripper 4 of 1 value, and the plain term 1 command of 1 value
    assert manager.action_term_dim == [4, 4, 1]
    assert manager.total_action_dim == 9
    arm, gripper, plain = (manager.get_term(name) for name in ("arm", "gripper", "plain"))

    _env_step(manager, [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 20.0])
    # each command is processed when it starts, so a term that reads the simulation sees it at that physics step
    assert arm.processed_at == [(0, 1.0), (2, 3.0)]
    assert gripper.processed_at == [(0, 10.0), (1, 11.0), (2, 12.0), (3, 13.0)]
    assert plain.processed_at == [(0, 20.0)]
    assert arm.applied == [2.0, 2.0, 6.0, 6.0]
    assert gripper.applied == [20.0, 22.0, 24.0, 26.0]
    assert plain.applied == [40.0] * 4

    _env_step(manager, [5.0, 6.0, 7.0, 8.0, 30.0, 31.0, 32.0, 33.0, 40.0])
    assert arm.applied[4:] == [10.0, 10.0, 14.0, 14.0]
    assert gripper.applied[4:] == [60.0, 62.0, 64.0, 66.0]
    assert plain.applied[4:] == [80.0] * 4
    assert manager.get_active_iterable_terms(0)[0] == ("arm", [5.0, 6.0, 7.0, 8.0])


def test_a_chunk_keeps_the_apply_step_in_the_environment_loop() -> None:
    """A chunk is processed between physics steps, which a recorded apply step cannot do."""
    assert _manager(arm=1, gripper=1).is_graphable
    assert not _manager(arm=1, gripper=2).is_graphable


@pytest.mark.parametrize("chunk_length", [0, -2, 3, 2.0])
def test_chunk_length_must_be_a_positive_divisor_of_the_decimation(chunk_length) -> None:
    with pytest.raises(ValueError, match="positive divisor"):
        _manager(arm=chunk_length)
