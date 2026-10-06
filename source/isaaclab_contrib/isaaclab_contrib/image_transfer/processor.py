# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Buffered image observations through the sensor post-processing contract."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import warp as wp

from isaaclab.sensors.post_processing import CameraPostProcessorContext, SensorPostProcessor
from isaaclab.utils.warp import ProxyArray

from .processor_cfg import DepthImageProcessorCfg, ImageTransferProcessorCfg


def resolve_image_transfer_processor(
    cfg: ImageTransferProcessorCfg, context: CameraPostProcessorContext
) -> SensorPostProcessor:
    """Bind prepared camera controls to persistent, separately owned RGB outputs."""
    _validate_config(cfg, context.num_views)
    backend = cfg.backend
    pending: list[list[torch.Tensor]] = [[] for _ in range(context.num_views)]
    fresh = [True] * context.num_views
    episodes = [0] * context.num_views
    reset_pending: set[int] = set()
    camera_inputs: dict[str, torch.Tensor] = {}
    rgba: torch.Tensor | None = None
    started = False
    closed = False
    failed = False

    @contextmanager
    def torch_stream():
        if not wp.get_device(context.device).is_cuda:
            yield
            return
        render_stream = wp.get_stream(context.device)
        model_stream = wp.stream_to_torch(render_stream)
        # A null handle can denote different default-stream semantics in Torch
        # and Warp builds. Explicit handles share ordering; null handles need
        # a completion boundary in both directions before another library runs.
        if not render_stream.cuda_stream:
            wp.synchronize_stream(render_stream)
        with torch.cuda.stream(model_stream):
            try:
                yield
            finally:
                if not render_stream.cuda_stream:
                    model_stream.synchronize()

    def initialize(inputs: dict[str, ProxyArray], outputs: dict[str, ProxyArray]) -> None:
        nonlocal rgba
        camera_inputs.update({name: value.torch for name, value in inputs.items()})
        rgba = outputs["rgba"].torch
        with torch_stream():
            rgba.zero_()
            rgba[..., 3] = 255

    def process_frames(env_mask: wp.array) -> None:
        nonlocal started, failed
        if closed or failed or not camera_inputs or rgba is None:
            raise RuntimeError("Image processing processor is closed, failed, or not initialized.")
        rows = wp.to_torch(env_mask).nonzero().flatten().tolist()
        if not rows:
            return
        if any(len(pending[row]) >= cfg.max_pending_frames for row in rows):
            raise RuntimeError("Image processing control queue is full; camera rows must advance at compatible rates.")
        with torch.no_grad():
            # Advanced indexing isolates preparation from borrowed camera buffers.
            selected = {name: value[rows] for name, value in camera_inputs.items()}
            pixels = cfg.prepare_control(selected) if cfg.prepare_control is not None else next(iter(selected.values()))
            if (
                not isinstance(pixels, torch.Tensor)
                or pixels.shape != (len(rows), context.height, context.width, 3)
                or pixels.dtype != torch.uint8
                or pixels.device != rgba.device
            ):
                raise ValueError("Prepared controls must be uint8 NHWC with three channels on the camera device.")
            for index, row in enumerate(rows):
                pending[row].append(pixels[index].clone())
            counts = [cfg.initial_frames if value else cfg.update_frames for value in fresh]
            if not all(len(queue) >= count for queue, count in zip(pending, counts, strict=True)):
                return
            controls = [torch.stack(queue[:count]) for queue, count in zip(pending, counts, strict=True)]
            resets = tuple(sorted(reset_pending)) if started else ()
            seeds = tuple((int(backend.seeds[row]) + episodes[row] * context.num_views) % (2**31) for row in resets)
            try:
                generated = backend.step(controls, resets, seeds)
                if len(generated) != context.num_views:
                    raise ValueError("Image processing returned the wrong number of camera rows.")
                for video, count in zip(generated, counts, strict=True):
                    if video.shape != (count, context.height, context.width, 3) or video.dtype != torch.uint8:
                        raise ValueError("Image processing must return uint8 THWC RGB matching each control chunk.")
                # Validate the complete result before publishing any row.
                for row, (video, count) in enumerate(zip(generated, counts, strict=True)):
                    rgba[row, ..., :3].copy_(video[-1])
                    del pending[row][:count]
                    fresh[row] = False
                started = True
                reset_pending.clear()
            except Exception:
                # A backend step may already have advanced its state. Never retry it
                # against unchanged input queues and silently desynchronize histories.
                failed = True
                raise

    def process(env_mask: wp.array) -> None:
        with torch_stream():
            process_frames(env_mask)

    def reset(env_mask: wp.array) -> None:
        if closed or failed:
            raise RuntimeError("Cannot reset a closed or failed Image processing processor.")
        for row in wp.to_torch(env_mask).nonzero().flatten().tolist():
            pending[row].clear()
            fresh[row] = True
            # Resets before the first model call have no previous episode to discard.
            if started:
                episodes[row] += 1
                reset_pending.add(row)
            if rgba is not None:
                with torch_stream():
                    rgba[row, ..., :3].zero_()

    def close() -> None:
        nonlocal closed, rgba
        if closed:
            return
        closed = True
        rgba = None
        camera_inputs.clear()
        pending.clear()
        backend.close()

    return SensorPostProcessor(
        inputs=cfg.inputs, outputs=cfg.outputs, initialize=initialize, process=process, reset=reset, close=close
    )


def resolve_depth_image_processor(
    cfg: DepthImageProcessorCfg, context: CameraPostProcessorContext
) -> SensorPostProcessor:
    """Use the common image processor with a stateless metric-depth transform."""
    if not (math.isfinite(cfg.near) and math.isfinite(cfg.far) and 0 < cfg.near < cfg.far):
        raise ValueError("Depth bounds must be finite and satisfy 0 < near < far.")
    if cfg.prepare_control is not None:
        raise ValueError("Use ImageTransferProcessorCfg for custom control preparation.")

    def prepare_depth(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        depth = inputs["distance_to_image_plane"]
        valid = torch.isfinite(depth) & (depth > 0)
        metric = torch.where(valid, depth, cfg.far)
        pixels = ((cfg.far - metric) / (cfg.far - cfg.near)).clamp(0, 1).mul(255).round().to(torch.uint8)
        return pixels.expand(-1, -1, -1, 3)

    from isaaclab.utils import replace

    return resolve_image_transfer_processor(replace(cfg, prepare_control=prepare_depth), context)


def _validate_config(cfg: ImageTransferProcessorCfg, num_views: int) -> None:
    if cfg.backend is None or len(cfg.backend.seeds) != num_views:
        raise ValueError("Supply one exclusive Image processing backend with one seed per camera view.")
    if not cfg.inputs:
        raise ValueError("Declare at least one camera input.")
    if cfg.prepare_control is None and len(cfg.inputs) != 1:
        raise ValueError("Multiple camera inputs require a prepare_control callable.")
    if cfg.prepare_control is not None and not callable(cfg.prepare_control):
        raise ValueError("prepare_control must be callable or None.")
    sizes = (cfg.initial_frames, cfg.update_frames, cfg.max_pending_frames)
    if any(type(value) is not int or value < 1 for value in sizes):
        raise ValueError("Chunk sizes and queue capacity must be positive integers.")
    if cfg.max_pending_frames < max(cfg.initial_frames, cfg.update_frames):
        raise ValueError("Queue capacity must accommodate both configured chunk sizes.")
