# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Color transfer contract at the real image-transfer-to-PPISP pipeline boundary."""

import pytest
import torch
import warp as wp
from isaaclab_ppisp import PpispCfg, PpispProcessorCfg

from isaaclab.sensors.post_processing import CameraPostProcessorContext, SensorPostProcessingPipeline
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_contrib.image_transfer import DepthImageProcessorCfg, SrgbToLinearProcessorCfg


@pytest.mark.unit
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_srgb_conversion_feeds_ppisp_without_changing_source(device):
    """Known sRGB values linearize correctly and PPISP preserves the comparison image."""
    conversion = SrgbToLinearProcessorCfg()
    pipeline = SensorPostProcessingPipeline(
        [conversion, PpispProcessorCfg(isp_cfg=PpispCfg())],
        CameraPostProcessorContext(None, (), 1, 1, 3, device),
        conversion.inputs,
        ["rgb", "source_rgb", "rgb_radiance"],
    )
    try:
        outputs = pipeline.allocate()
        source = pipeline.render_outputs["rgb"].torch
        values = torch.tensor([0, 128, 255], dtype=torch.uint8, device=device).view(1, 1, 3, 1).expand(1, 1, 3, 3)
        source.copy_(values)
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device)), sync_enter=True, sync_exit=True):
            pipeline.process(wp.array([True], dtype=wp.bool, device=device))
        expected = torch.tensor([0.0, 0.2158605, 1.0], device=device)
        torch.testing.assert_close(outputs["rgb_radiance"].torch[0, 0, :, 0], expected)
        torch.testing.assert_close(source, values)
        torch.testing.assert_close(outputs["source_rgb"].torch, values)
        ppisp = outputs["rgb"].torch[0, 0, :, 0].int()
        assert ppisp[0] < ppisp[1] < ppisp[2]
    finally:
        pipeline.close()


@pytest.mark.unit
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_image_transfer_torch_default_stream_finishes_before_warp_ppisp(device):
    """A delayed model result must reach the Warp stage during the same capture."""

    class DelayedModel:
        seeds = (42,)

        def __init__(self):
            self.value = 0

        def step(self, controls, reset_rows, seeds):
            self.value += 1
            torch.cuda._sleep(10_000_000)
            return [torch.full_like(controls[0], self.value * 64)]

        def close(self):
            pass

    cfg = DepthImageProcessorCfg(backend=DelayedModel())
    pipeline = SensorPostProcessingPipeline(
        [cfg, SrgbToLinearProcessorCfg(), PpispProcessorCfg(isp_cfg=PpispCfg())],
        CameraPostProcessorContext(None, (), 1, 2, 2, device),
        cfg.inputs,
        ["rgb", "source_rgb"],
    )
    try:
        with torch.cuda.stream(torch.cuda.default_stream(device)):
            with wp.ScopedStream(
                wp.stream_from_torch(torch.cuda.current_stream(device)), sync_enter=True, sync_exit=True
            ):
                outputs = pipeline.allocate()
                pipeline.render_outputs["distance_to_image_plane"].torch.fill_(3)
            mask = wp.array([True], dtype=wp.bool, device=device)
            for _ in range(2):
                with wp.ScopedStream(
                    wp.stream_from_torch(torch.cuda.current_stream(device)), sync_enter=True, sync_exit=True
                ):
                    pipeline.process(mask)
            torch.cuda.synchronize(device)
            assert torch.all(outputs["source_rgb"].torch == 128)
    finally:
        pipeline.close()
