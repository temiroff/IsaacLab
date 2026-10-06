# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the image processor on CPU with an illustrative application backend."""

import argparse
from pathlib import Path

import numpy as np
import torch
import warp as wp

from isaaclab.sensors.post_processing import CameraPostProcessorContext, SensorPostProcessingPipeline

from isaaclab_contrib.image_transfer import DepthImageProcessorCfg


class TintBackend:
    """Apply a fixed tint; this example does not run a learned model."""

    seeds = (7,)

    def step(self, controls, reset_rows, seeds):
        tint = torch.tensor([1.0, 0.6, 0.2])
        return [pixels.float().mul(tint).round().to(torch.uint8) for pixels in controls]

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    cfg = DepthImageProcessorCfg(backend=TintBackend(), near=1.0, far=9.0)
    pipeline = SensorPostProcessingPipeline(
        [cfg], CameraPostProcessorContext(None, (), 1, 64, 96, "cpu"), cfg.inputs, ["rgb"]
    )
    try:
        output = pipeline.allocate()["rgb"].torch
        depth = pipeline.render_outputs["distance_to_image_plane"].torch
        depth.copy_(torch.linspace(1, 9, 96).view(1, 1, 96, 1).expand_as(depth))
        pipeline.process(wp.array([True], dtype=wp.bool, device="cpu"))
        assert output[0, 0, 0].tolist() == [255, 153, 51]
        assert output[0, 0, -1].tolist() == [0, 0, 0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, depth=depth.numpy(), rgb=output.numpy())
        print(f"Saved depth {tuple(depth.shape)} and RGB {tuple(output.shape)} to {args.output}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
