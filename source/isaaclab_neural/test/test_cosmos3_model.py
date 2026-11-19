# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Cosmos3 client against the world-model service, with a fake model in place of the Cosmos Framework."""

import socket
import threading
import time

import numpy as np
import pytest
import torch
from isaaclab_experimental.cosmos import _cuda_ipc, _protocol
from isaaclab_experimental.cosmos.server.service import serve
from isaaclab_experimental.cosmos.server.world_model import NUM_CONTROLS, handle_world_model_connection
from PIL import Image

from isaaclab.utils import configclass

from isaaclab_neural.models.cosmos3 import Cosmos3Cfg, Cosmos3Model, JointMappingCfg

T, H, W = 4, 6, 8


class Mapping:
    def __init__(self, cfg):
        pass

    def encode(self, joint_targets, start_joint_state):
        controls = np.zeros((len(joint_targets), NUM_CONTROLS), np.float32)
        controls[:, 0] = joint_targets[:, 0] + start_joint_state[0]
        return controls


@configclass
class MappingCfg(JointMappingCfg):
    class_type: type = Mapping


class FakeSession:
    def __init__(self, model, num_envs):
        self.model, self.num_envs = model, num_envs

    def reset(self, env_ids, images, prompts, seeds):
        self.model.resets.append((env_ids, prompts, seeds))
        return torch.from_numpy(images.copy())

    def step(self, controls):
        if self.model.fail:
            raise RuntimeError("generation failed")
        self.model.controls.append(controls.copy())
        shape = (self.num_envs, T, H, W, 3)
        return torch.full(shape, len(self.model.controls), dtype=torch.uint8, device=self.model.capabilities["device"])

    def close(self):
        self.model.closed_sessions += 1


class FakeWorldModel:
    """The service side of a world model, as the Cosmos3 service exposes it."""

    def __init__(self, device="cpu"):
        self.capabilities = {
            "kind": "world_model",
            "device": device,
            "chunk_length": T,
            "fps": 30,
            "image_size": [H, W],
            "max_envs": None,
        }
        self.resets, self.controls, self.closed_sessions, self.fail = [], [], 0, False

    def warmup(self):
        pass

    def open_session(self, num_envs):
        return FakeSession(self, num_envs)

    def close(self):
        pass


@pytest.fixture
def service(request):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{probe.getsockname()[1]}"
    model = FakeWorldModel(getattr(request, "param", "cpu"))
    thread = threading.Thread(
        target=serve, args=(model, endpoint), kwargs={"handler": handle_world_model_connection}, daemon=True
    )
    thread.start()
    for _ in range(100):
        try:
            _protocol.request(endpoint, {"op": "status"}, timeout=1)
            break
        except ConnectionError:
            time.sleep(0.05)
    yield model, endpoint
    _protocol.request(endpoint, {"op": "shutdown"}, timeout=2)
    thread.join(timeout=5)


def start_images(tmp_path, count):
    paths = []
    for i in range(count):
        path = tmp_path / f"start{i}.png"
        Image.fromarray(np.full((H, W, 3), 100 + i, np.uint8)).save(path)
        paths.append(path)
    return paths


def test_the_client_reads_the_spec_resets_from_start_images_and_steps_with_mapped_controls(service, tmp_path):
    fake, endpoint = service
    model = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5, joint_mapping=MappingCfg()), 2, "cpu")
    assert (model.spec().chunk_length, model.spec().image_size) == (T, (H, W))
    first = model.reset([0, 1], start_images(tmp_path, 2), ["unfold", "fold"], [3, 4])
    assert first.shape == (2, H, W, 3) and first[:, 0, 0, 0].tolist() == [100, 101]
    assert fake.resets == [([0, 1], ["unfold", "fold"], [3, 4])]
    targets = torch.arange(2 * T * 3, dtype=torch.float32).reshape(2, T, 3)
    frames = model.step(targets, torch.tensor([[1.0, 0, 0], [5.0, 0, 0]]))
    assert frames.shape == (2, T, H, W, 3) and frames.dtype == torch.uint8 and (frames == 1).all()
    assert fake.controls[0].shape == (2, T, NUM_CONTROLS) and fake.controls[0][1, 0, 0] == targets[1, 0, 0] + 5
    model.close()
    assert fake.closed_sessions == 1


def test_a_failed_step_ends_the_session_and_frees_the_service(service, tmp_path):
    fake, endpoint = service
    model = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5), 1, "cpu")
    model.reset([0], start_images(tmp_path, 1), ["unfold"], [0])
    fake.fail = True
    with pytest.raises(RuntimeError, match="generation failed"):
        model.step_controls(np.zeros((1, T, NUM_CONTROLS), np.float32))
    with pytest.raises(RuntimeError, match="closed or failed"):
        model.step_controls(np.zeros((1, T, NUM_CONTROLS), np.float32))
    # The service closes the failed session, so a new client can open one.
    fake.fail = False
    for _ in range(100):
        if not _protocol.request(endpoint, {"op": "status"}, timeout=1)[0]["session_active"]:
            break
        time.sleep(0.05)
    other = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5), 1, "cpu")
    other.reset([0], start_images(tmp_path, 1), ["unfold"], [0])
    assert other.step_controls(np.zeros((1, T, NUM_CONTROLS), np.float32)).shape == (1, T, H, W, 3)
    other.close()


def test_cuda_ipc_is_refused_with_its_reason_when_unavailable(service, tmp_path):
    _, endpoint = service
    model = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5, transport="cuda_ipc"), 1, "cpu")
    with pytest.raises(RuntimeError, match="cuda_ipc is unavailable: the simulation does not run on a CUDA device"):
        model.reset([0], start_images(tmp_path, 1), ["unfold"], [0])


@pytest.mark.skipif(not (torch.cuda.is_available() and _cuda_ipc.available()), reason="needs CUDA IPC (Linux)")
@pytest.mark.parametrize("service", ["cuda:0"], indirect=True)
def test_chunks_reach_the_simulation_device_through_cuda_ipc(service, tmp_path):
    fake, endpoint = service
    model = Cosmos3Model(Cosmos3Cfg(endpoint=endpoint, timeout=5, transport="cuda_ipc"), 2, "cuda:0")
    model.reset([0, 1], start_images(tmp_path, 2), ["unfold", "fold"], [3, 4])
    assert model.transport == "cuda_ipc"
    for chunk in (1, 2):
        frames = model.step_controls(np.zeros((2, T, NUM_CONTROLS), np.float32))
        assert frames.device == torch.device("cuda:0") and (frames == chunk).all()
    model.close()
    assert fake.closed_sessions == 1
