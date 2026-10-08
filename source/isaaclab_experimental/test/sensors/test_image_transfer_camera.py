# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Putting image transfer on a task camera, and the control and resize functions it uses."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_experimental.image_transfer import (
    ImageTransferModifierCfg,
    WorkerModelCfg,
    center_crop_resize,
    edge_control,
    image_transfer_camera,
    region_control,
)
from isaaclab_experimental.image_transfer import modifier as modifier_module

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg, ModifierChain

pytestmark = pytest.mark.unit

POLICY_CAMERA = CameraCfg(
    prim_path="/World/Camera", spawn=sim_utils.PinholeCameraCfg(), data_types=["rgb"], width=64, height=64
)


class InvertStream:
    def step(self, controls, reset_rows, seeds):
        return [255 - control for control in controls]

    def close(self):
        pass


class InvertModel:
    def __init__(self, cfg):
        pass

    def open_stream(self, num_views, seeds):
        return InvertStream()

    def close(self):
        pass


@configclass
class InvertModelCfg(BackendCfg):
    class_type: type = InvertModel


def test_policy_camera_renders_the_model_canvas_and_returns_policy_sized_images(monkeypatch):
    """Observation terms keep their shape: the chain renders 832 x 480 and publishes 64 x 64 ``rgb``."""
    monkeypatch.setattr(modifier_module, "SimulationContext", SimpleNamespace(instance=lambda: None))
    camera = image_transfer_camera(
        POLICY_CAMERA,
        ImageTransferModifierCfg(backend=InvertModelCfg()),
        control="depth",
        control_params={"near": 1.0, "far": 5.0},
        post_modifiers=[ModifierCfg(func=lambda data: data // 2)],
    )
    assert (camera.width, camera.height) == (832, 480)
    assert camera.modifier_outputs() == {"distance_to_image_plane": "rgb"}
    assert camera.render_data_types() == ["distance_to_image_plane"]

    chain = ModifierChain(camera.modifiers["distance_to_image_plane"], "cpu")
    depth = torch.full((1, 480, 832, 1), 3.0)
    image = chain(depth)
    chain.close()
    # Depth control 128 -> model 127 -> post-modifier 63, resized to the policy size.
    assert image.shape == (1, 64, 64, 3) and image.dtype == torch.uint8
    assert torch.all(image == 63)


def test_policy_camera_rejects_mismatched_control_or_output():
    with pytest.raises(ValueError, match="expects 'depth'"):
        image_transfer_camera(POLICY_CAMERA, ImageTransferModifierCfg(backend=WorkerModelCfg(control="depth")))
    with pytest.raises(ValueError, match="publish 'rgb'"):
        image_transfer_camera(
            POLICY_CAMERA,
            ImageTransferModifierCfg(backend=InvertModelCfg()),
            post_modifiers=[ImageTransferModifierCfg(backend=InvertModelCfg(), output="rgba")],
        )
    depth_camera = CameraCfg(
        prim_path="/World/Camera", spawn=sim_utils.PinholeCameraCfg(), data_types=["depth"], width=64, height=64
    )
    with pytest.raises(ValueError, match="'rgb'"):
        image_transfer_camera(depth_camera, ImageTransferModifierCfg(backend=InvertModelCfg()))


def test_center_crop_resize_keeps_the_center_and_the_aspect_ratio():
    images = torch.zeros((1, 480, 832, 3), dtype=torch.uint8)
    images[:, :, 416:] = 200
    images[:, :, :176] = 255  # Outside the centered 480 x 480 crop.
    resized = center_crop_resize(images, 64, 64)
    assert resized.shape == (1, 64, 64, 3)
    assert resized[0, :, :32].max() == 0 and torch.all(resized[0, :, 32:] == 200)


def test_edge_and_region_controls_produce_three_channel_uint8_images():
    pytest.importorskip("cv2")
    rgb = torch.zeros((1, 16, 16, 3), dtype=torch.uint8)
    rgb[:, :, 8:] = 255
    edges = edge_control(rgb)
    assert edges.shape == (1, 16, 16, 3) and edges.dtype == torch.uint8
    assert edges[0, :, 7:9].any() and not edges[0, :, :6].any()

    ids = torch.tensor([[[[0], [2]], [[3], [1]]]], dtype=torch.int32)
    regions = region_control(ids, palette={2: (255, 0, 0), 3: (0, 255, 0)})
    assert regions[0, 0, 1].tolist() == [255, 0, 0] and regions[0, 1, 0].tolist() == [0, 255, 0]
    assert regions[0, 0, 0].tolist() == [0, 0, 0] and regions[0, 1, 1].tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="without a palette color"):
        region_control(ids, palette={2: (255, 0, 0)})
