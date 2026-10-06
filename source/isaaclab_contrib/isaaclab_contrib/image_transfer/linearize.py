# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit display-sRGB approximation for applying PPISP to display RGB."""

import warp as wp

from isaaclab.sensors.post_processing import CameraPostProcessorContext, SensorPostProcessor


@wp.kernel
def _linearize(
    source: wp.array4d(dtype=wp.uint8), linear: wp.array4d(dtype=wp.float32), saved: wp.array4d(dtype=wp.uint8)
):
    n, y, x, c = wp.tid()
    value = wp.float32(source[n, y, x, c]) / 255.0
    result = value / 12.92
    if value > 0.04045:
        result = wp.pow((value + 0.055) / 1.055, 2.4)
    linear[n, y, x, c] = result
    saved[n, y, x, c] = source[n, y, x, c]


def resolve_srgb_to_linear(cfg, context: CameraPostProcessorContext) -> SensorPostProcessor:
    """Bind persistent image buffers on the caller's Warp stream."""
    buffers = {}

    def initialize(inputs, outputs):
        buffers.update(source=inputs["rgb"].warp, linear=outputs["rgb_radiance"].warp, saved=outputs["source_rgb"].warp)

    def process(env_mask):
        wp.launch(
            _linearize,
            dim=(context.num_views, context.height, context.width, 3),
            inputs=[buffers["source"], buffers["linear"], buffers["saved"]],
            device=context.device,
        )

    return SensorPostProcessor(
        inputs=cfg.inputs, outputs=cfg.outputs, initialize=initialize, process=process, close=buffers.clear
    )
