# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP applied after image transfer in one camera processor chain."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_experimental.image_transfer import (
    FunctionProcessorCfg,
    ImageTransferProcessorCfg,
    depth_to_control,
    srgb_to_linear,
)
from isaaclab_experimental.image_transfer import processor as processor_module

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.warp import ProxyArray

ppisp = pytest.importorskip("isaaclab_ppisp")


class _PassThroughStream:
    def step(self, controls, reset_rows, seeds):
        return controls

    def close(self):
        pass


class _PassThroughModel:
    def __init__(self, cfg):
        pass

    def open_stream(self, num_views, seeds):
        return _PassThroughStream()

    def close(self):
        pass


@configclass
class _PassThroughModelCfg(BackendCfg):
    class_type: type = _PassThroughModel


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="The PPISP kernel runs on CUDA devices."),
]


@pytest.fixture(autouse=True)
def no_sim(monkeypatch):
    monkeypatch.setattr(processor_module, "SimulationContext", SimpleNamespace(instance=lambda: None))


def _generate(exposure_offset: float) -> torch.Tensor:
    depth = torch.linspace(1.0, 9.0, 8, device="cuda:0").view(1, 1, 8, 1).expand(2, 4, 8, 1).contiguous()
    buffers = {"distance_to_image_plane": ProxyArray(wp.from_torch(depth))}
    cfgs = [
        FunctionProcessorCfg(
            function=depth_to_control,
            params={"near": 1.0, "far": 9.0},
            inputs=["distance_to_image_plane"],
            outputs=["control"],
        ),
        ImageTransferProcessorCfg(backend=_PassThroughModelCfg(), outputs=["generated"]),
        FunctionProcessorCfg(function=srgb_to_linear, inputs=["generated"], outputs=["generated_linear"]),
        ppisp.PpispProcessorCfg(
            isp_cfg=ppisp.PpispCfg(inputs={"exposureOffset": exposure_offset}), inputs=["generated_linear"]
        ),
    ]
    # Construct and run in order, as a camera does: each processor reads the outputs of the earlier ones.
    for cfg in cfgs:
        buffers.update(cfg.func(cfg, buffers)(buffers))
    return buffers["rgb"].torch.clone()


def test_ppisp_after_image_transfer_produces_8bit_rgb_with_relative_exposure():
    baseline = _generate(0.0)
    brighter = _generate(1.0)

    assert baseline.shape == (2, 4, 8, 3) and baseline.dtype == torch.uint8
    assert brighter.float().mean() > baseline.float().mean()
