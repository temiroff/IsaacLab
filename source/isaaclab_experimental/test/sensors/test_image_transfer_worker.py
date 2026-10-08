# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Worker-process image transfer models, exercised through the public stand-in worker."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_experimental.image_transfer import ImageTransferModifierCfg, WorkerModelCfg, depth_to_control
from isaaclab_experimental.image_transfer import modifier as modifier_module

from isaaclab.utils import instantiate
from isaaclab.utils.modifiers import ModifierCfg, ModifierChain

pytestmark = pytest.mark.unit

STAND_IN_WORKER = Path(__file__).resolve().parents[4] / "examples/sensors/image_transfer_stand_in_worker.py"


@pytest.fixture(autouse=True)
def no_sim(monkeypatch):
    monkeypatch.setattr(modifier_module, "SimulationContext", SimpleNamespace(instance=lambda: None))


def _chain(worker: WorkerModelCfg) -> ModifierChain:
    transfer = ImageTransferModifierCfg(backend=worker, seed=42, initial_frames=1, update_frames=4)
    return ModifierChain([ModifierCfg(func=depth_to_control, params={"near": 1.0, "far": 5.0}), transfer], "cpu")


def test_worker_receives_settings_chunks_and_episode_resets(tmp_path):
    """The camera-side model sends settings once, streams chunks, restarts episodes, and the worker exits cleanly."""
    worker = WorkerModelCfg(
        command=[sys.executable, str(STAND_IN_WORKER)],
        prompt="A robot hand turns a cube.",
        control="depth",
        max_episode_frames=9,
        log_path=str(tmp_path / "worker.log"),
        record_dir=str(tmp_path / "chunks"),
    )
    chain = _chain(worker)
    depth = torch.full((1, 2, 3, 1), 3.0)
    outputs = [chain(depth).clone() for _ in range(5)]
    chain.reset([0])
    outputs.append(chain(depth).clone())
    model = chain._instances[0]._owned_model
    stream = model.streams[0]
    chain.close()

    assert stream.process.returncode == 0
    assert stream.resets == 1
    # Depth 3 m between 1 m and 5 m is a 128 control; the stand-in tints it to (128, 77, 26).
    assert all(output[0, 0, 0].tolist() == [128, 77, 26] for output in outputs)
    assert [len(np.load(chunk)) for chunk in sorted((tmp_path / "chunks").glob("*.npy"))] == [1, 4, 1]
    log = (tmp_path / "worker.log").read_text()
    expected = '{"control": "depth", "max_episode_frames": 9, "prompt": "A robot hand turns a cube.", "seed": 42}'
    assert f"stand-in worker settings: {expected}" in log
    # One view: the next episode starts from the initial seed plus the number of views.
    assert "stand-in worker reset: seeds [43]" in log


def test_a_worker_that_exits_fails_the_chunk_instead_of_hanging(tmp_path):
    chain = _chain(WorkerModelCfg(command=[sys.executable, "-c", "pass"], log_path=str(tmp_path / "worker.log")))
    with pytest.raises(RuntimeError, match="exited"):
        chain(torch.full((1, 2, 3, 1), 3.0))
    chain.close()


def test_worker_model_rejects_settings_it_cannot_serve():
    with pytest.raises(ValueError, match="command"):
        instantiate(WorkerModelCfg())
    model_cfg = WorkerModelCfg(command=[sys.executable, str(STAND_IN_WORKER)])
    with pytest.raises(ValueError, match="one camera view"):
        instantiate(model_cfg).open_stream(2, (0, 1))
