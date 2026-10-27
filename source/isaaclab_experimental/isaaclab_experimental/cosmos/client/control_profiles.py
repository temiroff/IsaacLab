# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos camera control recipe as camera tensor processors.

The factory returns processors for :attr:`CameraCfg.tensor_processors`: the control processor reading the camera's
depth, followed by the Cosmos transfer processor producing ``rgb``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab.utils.tensor_processors import TensorProcessorCfg

from isaaclab_experimental.image_transfer import FunctionProcessorCfg, depth_to_control

from .cosmos_processor_cfg import CosmosTransferProcessorCfg

if TYPE_CHECKING:
    from .cosmos_model_cfg import CosmosModelCfg


def depth_processor(
    backend: CosmosModelCfg, *, near: float = 1.2, far: float = 3.5, **cosmos: Any
) -> list[TensorProcessorCfg]:
    """Depth-guided Cosmos processors on the camera's ``distance_to_image_plane`` output.

    Near hits become white and far hits black. Set the metric range to match the camera's scene.
    """
    control = FunctionProcessorCfg(
        function=depth_to_control,
        params={"near": near, "far": far},
        inputs=["distance_to_image_plane"],
        outputs=["control"],
    )
    return [control, CosmosTransferProcessorCfg(backend=backend, **cosmos)]
