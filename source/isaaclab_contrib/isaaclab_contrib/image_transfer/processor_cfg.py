# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for application-owned image transfer and color conversion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field

import torch
import warp as wp

from isaaclab.renderers import RenderBufferSpec
from isaaclab.sensors.post_processing import SensorPostProcessorCfg
from isaaclab.utils import configclass

from .backend import ImageTransferBackend


@configclass
class ImageTransferProcessorCfg(SensorPostProcessorCfg):
    """Prepare named camera inputs for an application-owned image backend.

    ``prepare_control`` receives selected views as named NHWC Torch tensors
    and returns uint8 NHWC three-channel controls on the same device. It must
    be stateless across calls; temporal state belongs to the backend. Inputs
    are isolated from renderer buffers before preparation. A callable may
    combine multiple inputs or delegate conversion to an optional library.

    With no callable, a single three-channel uint8 input passes through.
    The image representation is chosen by the application, not this package.
    This interface does not imply a backend supports every input modality.

    Configuration copying borrows ``backend``; one processor exclusively owns
    its stream lifecycle. Output holds the latest image between chunks.
    Resets clear selected queues and outputs, then notify the backend before
    their next generation. A stalled view causes bounded backpressure.
    """

    func = "{DIR}.processor:resolve_image_transfer_processor"
    inputs: dict[str, RenderBufferSpec] = {"rgb": RenderBufferSpec(3, wp.uint8, color_space="srgb")}
    outputs: dict[str, RenderBufferSpec] = {"rgba": RenderBufferSpec(4, wp.uint8, color_space="srgb")}
    backend: ImageTransferBackend | None = field(default=None, metadata={"copy": False})
    prepare_control: Callable[[dict[str, torch.Tensor]], torch.Tensor] | None = None
    """Application-owned conversion from selected named inputs to image controls."""
    initial_frames: int = 1
    """Captures required for the first chunk of each episode."""
    update_frames: int = 1
    """Captures required for subsequent chunks."""
    max_pending_frames: int = 8
    """Maximum queued captures per view before backpressure raises an error."""


@configclass
class DepthImageProcessorCfg(ImageTransferProcessorCfg):
    """Convenience configuration for fixed-range metric-depth conversion.

    Near hits become white; far, missing and nonpositive hits become black.
    Use :class:`ImageTransferProcessorCfg` for other control preparations.
    """

    func = "{DIR}.processor:resolve_depth_image_processor"
    inputs: dict[str, RenderBufferSpec] = {"distance_to_image_plane": RenderBufferSpec(1, wp.float32)}
    near: float = 0.1
    """White depth endpoint in meters."""
    far: float = 10.0
    """Black depth endpoint in meters."""


@configclass
class SrgbToLinearProcessorCfg(SensorPostProcessorCfg):
    """Approximate linear input for relative PPISP effects on display sRGB.

    Inverse sRGB does not recover scene HDR, exposure, or a baked camera
    response. Use native renderer radiance for calibrated camera processing.
    The input image is retained as ``source_rgb`` for comparison.
    """

    func = "{DIR}.linearize:resolve_srgb_to_linear"
    inputs: dict[str, RenderBufferSpec] = {"rgb": RenderBufferSpec(3, wp.uint8, color_space="srgb")}
    outputs: dict[str, RenderBufferSpec] = {
        "rgb_radiance": RenderBufferSpec(3, wp.float32, color_space="scene_linear"),
        "source_rgb": RenderBufferSpec(3, wp.uint8, color_space="srgb"),
    }
