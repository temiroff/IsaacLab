# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos image generation applied to camera observations."""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import replace

from isaaclab_experimental.image_transfer import ImageTransferModifier

if TYPE_CHECKING:
    from isaaclab.sensors import SensorBase

    from .cosmos_modifier_cfg import CosmosTransferModifierCfg


logger = logging.getLogger(__name__)


class CosmosTransferModifier(ImageTransferModifier):
    """Generate AR control chunks or independently augment current camera RGB.

    The shared image transfer modifier owns control queues, episode resets, and publication of the
    latest generated image in AR mode. Single-frame mode instead reads raw RGB, prepares same-capture
    controls on demand, and publishes raw RGB whenever generation is skipped.
    """

    def __init__(self, cfg: CosmosTransferModifierCfg, data_dim: tuple[int, ...], device: str):
        """Validate the Cosmos cadence and initialize the camera's image transfer stream.

        Args:
            cfg: Cosmos camera modifier configuration.
            data_dim: Control shape ``(N, H, W, 3)``.
            device: Device of the controls and generated output.

        Raises:
            ValueError: If the configured cadence is incompatible with Cosmos Sim-Transfer.
        """
        self._single_frame = cfg.backend.generation_mode == "single_frame"
        if type(cfg.decision_interval) is not int or cfg.decision_interval < 1:
            raise ValueError("decision_interval must be a positive integer.")
        if not math.isfinite(cfg.augmentation_probability) or not 0 <= cfg.augmentation_probability <= 1:
            raise ValueError("augmentation_probability must lie in [0, 1].")
        if not self._single_frame and (cfg.decision_interval != 1 or cfg.augmentation_probability != 1):
            raise ValueError("Decision scheduling requires single_frame generation.")
        if self._single_frame:
            cfg = replace(cfg, initial_frames=1, update_frames=1)
        elif (cfg.initial_frames, cfg.update_frames) != (1, 4):
            raise ValueError("Cosmos requires initial_frames=1 and update_frames=4.")
        super().__init__(cfg, data_dim, device)
        self._capture_index = 0
        self._rng = random.Random(cfg.seed)
        self._last_capture_time: float | None = None
        self._gaps = {"count": 0, "sum": 0.0, "max": 0.0}
        self._gap_warned = False

    def bind_sensor(self, sensor: SensorBase) -> None:
        """Queue controls for the views ``sensor`` captures; a camera sharing a session also takes its views there."""
        super().bind_sensor(sensor)
        bind = getattr(self._stream, "bind_sensor", None)
        if bind is not None:
            bind(sensor)

    @property
    def capture_gap_s(self) -> dict[str, float]:
        """Simulated time between the camera captures Cosmos received: ``count``, ``mean`` and ``max`` [s]."""
        count = self._gaps["count"]
        return {"count": count, "mean": self._gaps["sum"] / count if count else 0.0, "max": self._gaps["max"]}

    def _sensor_time(self) -> float | None:
        """Simulated time of the current capture from the bound camera's timestamp, or None when unavailable."""
        stamp = getattr(getattr(self, "_sensor", None), "_timestamp", None)
        if stamp is None:
            return None
        try:
            if not isinstance(stamp, torch.Tensor):
                import warp as wp

                stamp = wp.to_torch(stamp)
            return float(stamp.max())
        except Exception:  # an unusual sensor must never break image generation
            return None

    def _track_capture_gap(self) -> None:
        """Measure the time between captures and warn once when they are too far apart for Cosmos.

        Frames sent to Cosmos further apart than ``max_capture_gap_s`` made fast-moving objects deform in the generated
        video (a 0.133 s gap pinched a grasped apple; 0.033 s did not).
        """
        now = self._sensor_time()
        if now is None:
            return
        last, self._last_capture_time = self._last_capture_time, now
        if last is None or now <= last:  # the first capture, or a reset restarted the camera's clock
            return
        gap = now - last
        self._gaps["count"] += 1
        self._gaps["sum"] += gap
        self._gaps["max"] = max(self._gaps["max"], gap)
        limit = self._cfg.max_capture_gap_s
        if gap > limit + 1e-6 and not self._gap_warned:
            self._gap_warned = True
            period = getattr(getattr(self._sensor, "cfg", None), "update_period", None)
            logger.warning(
                "Cosmos camera frames are %.3f s apart (camera update_period=%s). Frames further apart than %.3f s can "
                "make fast-moving objects deform in the generated video. Set the camera's update_period to at most "
                "%.3f s (1/30 s is recommended) and render at least that often, or raise max_capture_gap_s to silence "
                "this warning.",
                gap,
                period,
                limit,
                limit,
            )

    def close(self) -> None:
        """Close the stream and report how far apart the captures were."""
        if not self._closed and self._gaps["count"]:
            stats = self.capture_gap_s
            logger.info(
                "Cosmos capture gap: mean %.3f s, max %.3f s over %d captures.",
                stats["mean"],
                stats["max"],
                stats["count"],
            )
        super().close()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Publish current raw RGB when skipping; independent generation never waits for an AR chunk."""
        if not self._single_frame:
            self._track_capture_gap()
            return super().__call__(data)
        if self._closed or self._failed:
            raise RuntimeError("Image transfer modifier is closed or failed; recreate it.")
        if tuple(data.shape) != self._data_dim or data.dtype != torch.uint8:
            raise ValueError(f"Single-frame RGB must be uint8 with shape {self._data_dim}.")
        decision = self._capture_index % self._cfg.decision_interval == 0
        self._capture_index += 1
        if not decision or self._rng.random() >= self._cfg.augmentation_probability:
            return data
        controls = self._single_frame_controls(data)
        resets = tuple(sorted(self._reset_rows)) if self._started else ()
        count = len(self._seeds)
        seeds = tuple((self._seeds[row] + self._episodes[row] * count) % 2**31 for row in resets)
        try:
            images = self._stream.step([frame.unsqueeze(0) for frame in controls], resets, seeds)
            if len(images) != count or any(
                image.dtype != torch.uint8 or tuple(image.shape) != (1, *self._data_dim[1:]) for image in images
            ):
                raise ValueError("Cosmos must return one uint8 RGB image per current view.")
            result = torch.stack([image[0] for image in images])
        except Exception:
            self._failed = True
            raise
        self._started = True
        self._reset_rows.clear()
        return result

    def _single_frame_controls(self, rgb: torch.Tensor) -> torch.Tensor:
        modality = self._cfg.backend.modality
        if modality == "depth":
            from isaaclab_experimental.image_transfer import depth_to_control

            if self._sensor is None:
                raise ValueError("Single-frame depth generation needs a bound camera rendering depth and RGB.")
            depth = self._sensor._render_outputs["distance_to_image_plane"].torch
            return depth_to_control(depth, self._cfg.depth_near, self._cfg.depth_far)
        if modality == "edge":
            from .control_profiles import edge_control

            return edge_control(rgb, *self._cfg.edge_thresholds)
        if modality == "blur":
            return rgb
        raise ValueError("Single-frame cameras support depth, edge, or blur guidance.")

    def reset(self, env_ids=None):
        """Make the first post-reset capture a decision point and discard prior episode state."""
        super().reset(env_ids)
        if env_ids is None or len(env_ids):
            self._capture_index = 0
