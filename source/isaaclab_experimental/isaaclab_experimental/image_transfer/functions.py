# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Function modifiers around image transfer: control preparation, color conversion and resizing.

Use them in a camera modifier chain through :class:`~isaaclab.utils.modifiers.ModifierCfg`, for example
``ModifierCfg(func=edge_control, params={"lower": 100, "upper": 200})``.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def depth_to_control(data: torch.Tensor, near: float = 0.1, far: float = 10.0) -> torch.Tensor:
    """Map metric depth to uint8 three-channel controls.

    Hits at ``near`` become white and hits at ``far`` black. Missing, nonfinite, and nonpositive
    depth becomes black.

    Args:
        data: Depth [m] of shape ``(N, H, W, 1)``.
        near: White depth endpoint [m].
        far: Black depth endpoint [m].

    Returns:
        Controls of shape ``(N, H, W, 3)``, dtype uint8.
    """
    if not (math.isfinite(near) and math.isfinite(far) and 0 < near < far):
        raise ValueError("Depth bounds must be finite and satisfy 0 < near < far.")
    valid = torch.isfinite(data) & (data > 0)
    metric = torch.where(valid, data, far)
    pixels = ((far - metric) / (far - near)).clamp(0, 1).mul(255).round().to(torch.uint8)
    return pixels.expand(*pixels.shape[:-1], 3)


def edge_control(data: torch.Tensor, lower: int = 100, upper: int = 200) -> torch.Tensor:
    """Map sRGB images to uint8 three-channel Canny edge controls.

    Edges are white on black. The conversion runs on the CPU with OpenCV
    (``opencv-python-headless``), which is imported on the first call.

    Args:
        data: sRGB of shape ``(N, H, W, 3)``, dtype uint8.
        lower: Lower hysteresis threshold of the Canny detector.
        upper: Upper hysteresis threshold of the Canny detector.

    Returns:
        Controls of shape ``(N, H, W, 3)``, dtype uint8.
    """
    if not 0 <= lower < upper:
        raise ValueError("Edge thresholds must satisfy 0 <= lower < upper.")
    try:
        import cv2
    except ImportError as error:
        raise ImportError("edge_control needs OpenCV: pip install opencv-python-headless") from error
    frames = data.cpu().numpy()
    edges = np.stack([cv2.Canny(np.ascontiguousarray(frame), lower, upper) for frame in frames])
    return torch.from_numpy(edges).to(data.device).unsqueeze(-1).expand(-1, -1, -1, 3)


def region_control(data: torch.Tensor, palette: dict[int, tuple[int, int, int]]) -> torch.Tensor:
    """Color raw segmentation IDs with a fixed palette.

    IDs 0 and 1 (background and unlabeled) become black. Keep the palette fixed for an episode so each
    region keeps its color.

    Args:
        data: Uncolorized segmentation IDs of shape ``(N, H, W, 1)``, dtype int32.
        palette: Color ``(r, g, b)`` in [0, 255] per segmentation ID greater than 1.

    Returns:
        Controls of shape ``(N, H, W, 3)``, dtype uint8.

    Raises:
        ValueError: If the palette is invalid or an ID in ``data`` has no color.
    """
    ids = data[..., 0]
    output = torch.zeros((*ids.shape, 3), dtype=torch.uint8, device=ids.device)
    known = ids <= 1
    for label, color in palette.items():
        label = int(label)
        if label < 2 or len(color) != 3 or any(not 0 <= value <= 255 for value in color):
            raise ValueError("Palette entries need IDs greater than 1 and three channels in [0, 255].")
        selected = ids == label
        output[selected] = torch.tensor(tuple(color), dtype=torch.uint8, device=ids.device)
        known |= selected
    if not bool(known.all()):
        raise ValueError("The segmentation contains IDs without a palette color.")
    return output


def srgb_to_linear(data: torch.Tensor) -> torch.Tensor:
    """Approximate scene-linear color from display sRGB, for example before PPISP.

    The inverse sRGB transfer function does not recover scene HDR, exposure, or a baked camera
    response. Use the renderer's ``rgb_radiance`` for calibrated camera processing.

    Args:
        data: sRGB of shape ``(N, H, W, 3)``, dtype uint8.

    Returns:
        Linear color of shape ``(N, H, W, 3)``, dtype float32.
    """
    value = data.to(torch.float32) / 255.0
    return torch.where(value > 0.04045, ((value + 0.055) / 1.055) ** 2.4, value / 12.92)


def center_crop_resize(data: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Crop uint8 images to the target aspect ratio around the center, then area-resize them.

    Use it at the end of a chain to return a large generated image to the resolution a policy expects.

    Args:
        data: Images of shape ``(N, H, W, C)``, dtype uint8.
        width: Output width [px].
        height: Output height [px].

    Returns:
        Images of shape ``(N, height, width, C)``, dtype uint8.
    """
    _, rows, cols, _ = data.shape
    if cols * height > rows * width:
        crop = rows * width // height
        data = data[:, :, (cols - crop) // 2 : (cols - crop) // 2 + crop]
    else:
        crop = cols * height // width
        data = data[:, (rows - crop) // 2 : (rows - crop) // 2 + crop]
    pixels = torch.nn.functional.interpolate(data.permute(0, 3, 1, 2).float(), size=(height, width), mode="area")
    return pixels.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
