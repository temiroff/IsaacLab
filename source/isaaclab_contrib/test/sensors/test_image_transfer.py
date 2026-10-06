# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image-transfer scheduling and ownership at the real post-processing pipeline boundary."""

import pytest
import torch
import warp as wp

from isaaclab.sensors.post_processing import CameraPostProcessorContext, SensorPostProcessingPipeline
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import clone

from isaaclab_contrib.image_transfer import DepthImageProcessorCfg

pytestmark = pytest.mark.unit


class RecordingStream:
    """Record model requests without implementing frame gating or episode bookkeeping."""

    seeds = (11, 22)

    def __init__(self):
        self.calls = []
        self.closes = 0
        self.bad_output = False

    def step(self, controls, reset_rows, seeds):
        self.calls.append(([control.clone() for control in controls], reset_rows, seeds))
        # A distinct model output, unrelated to depth normalization or scheduling.
        return [torch.full_like(control, 100 + len(self.calls)) for control in controls][: 1 if self.bad_output else 2]

    def close(self):
        self.closes += 1


@pytest.fixture
def setup_pipeline():
    backend = RecordingStream()
    cfg = DepthImageProcessorCfg(backend=backend, near=1.0, far=5.0, update_frames=3)
    assert clone(cfg).backend is backend
    pipeline = SensorPostProcessingPipeline(
        [cfg], CameraPostProcessorContext(None, (), 2, 2, 3, "cpu"), cfg.inputs, ["rgb"]
    )
    outputs = pipeline.allocate()
    depth = pipeline.render_outputs["distance_to_image_plane"].torch
    depth.fill_(3)
    yield pipeline, backend, depth, outputs["rgb"].torch
    pipeline.close()


def mask(*values):
    return wp.array(values, dtype=wp.bool, device="cpu")


def test_chunk_cadence_normalization_and_borrowed_buffers(setup_pipeline):
    """Only fresh captures count; queued depth is owned and outputs hold between chunks."""
    pipeline, backend, depth, rgb = setup_pipeline
    depth[0, ..., 0] = torch.tensor([[1, 3, 5], [float("inf"), float("nan"), 0]])
    source = depth.clone()
    pointer = rgb.data_ptr()
    pipeline.process(mask(True, True))
    expected = torch.tensor([[255, 128, 0], [0, 0, 0]], dtype=torch.uint8)
    torch.testing.assert_close(backend.calls[0][0][0][0, ..., 0], expected)
    torch.testing.assert_close(depth, source, equal_nan=True)
    assert torch.all(rgb == 101)
    pipeline.process(mask(False, False))
    assert len(backend.calls) == 1
    for distance in (1, 2):
        depth.fill_(distance)
        pipeline.process(mask(True, True))
        assert torch.all(rgb == 101)
    depth.fill_(5)
    pipeline.process(mask(True, True))
    assert rgb.data_ptr() == pointer
    assert torch.all(rgb == 102)
    assert backend.calls[1][0][0].shape[0] == 3
    assert backend.calls[1][0][0][:, 0, 0, 0].tolist() == [255, 191, 0]
    assert backend.calls[1][1:] == ((), ())


def test_partial_reset_discards_pending_episode_without_resetting_other_row(setup_pipeline):
    """Reset affects only the selected row, including its queued controls and seed."""
    pipeline, backend, depth, rgb = setup_pipeline
    pipeline.process(mask(True, True))
    depth.fill_(1)
    pipeline.process(mask(True, True))
    pipeline.process(mask(True, True))
    pipeline.reset(mask(False, True))
    assert torch.all(rgb[0] == 101) and torch.count_nonzero(rgb[1]) == 0
    depth[1].fill_(5)
    pipeline.process(mask(False, True))
    assert len(backend.calls) == 1
    pipeline.process(mask(True, False))
    controls, rows, seeds = backend.calls[-1]
    assert rows == (1,) and seeds == (24,)
    assert [control.shape[0] for control in controls] == [3, 1]
    assert torch.all(controls[0] == 255) and torch.count_nonzero(controls[1]) == 0
    assert torch.all(rgb == 102)
    pipeline.close()
    pipeline.close()
    assert backend.closes == 1


def test_model_error_never_publishes_partial_results_or_retries_advanced_state(setup_pipeline):
    """A malformed model result poisons the stream instead of advancing only some rows."""
    pipeline, backend, _, rgb = setup_pipeline
    backend.bad_output = True
    with pytest.raises(ValueError, match="number of camera rows"):
        pipeline.process(mask(True, True))
    assert torch.count_nonzero(rgb) == 0
    with pytest.raises(RuntimeError, match="failed"):
        pipeline.process(mask(True, True))
    assert len(backend.calls) == 1


def test_stalled_row_has_bounded_queue_and_no_silent_frame_loss(setup_pipeline):
    """A stalled environment produces explicit backpressure instead of unbounded memory."""
    pipeline, backend, _, _ = setup_pipeline
    for _ in range(8):
        pipeline.process(mask(True, False))
    with pytest.raises(RuntimeError, match="queue is full"):
        pipeline.process(mask(True, False))
    assert not backend.calls
    pipeline.process(mask(False, True))
    assert len(backend.calls) == 1


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_cuda_processor_orders_model_work_on_the_render_stream(device):
    """Torch inference honors the active Warp render stream, even off the default stream."""

    class StreamObserver(RecordingStream):
        def step(self, controls, reset_rows, seeds):
            self.observed_stream = torch.cuda.current_stream(device).cuda_stream
            return super().step(controls, reset_rows, seeds)

    backend = StreamObserver()
    cfg = DepthImageProcessorCfg(backend=backend)
    pipeline = SensorPostProcessingPipeline(
        [cfg], CameraPostProcessorContext(None, (), 2, 2, 3, device), cfg.inputs, ["rgb"]
    )
    render_stream = torch.cuda.Stream(device=device)
    try:
        with wp.ScopedStream(wp.stream_from_torch(render_stream), sync_enter=True, sync_exit=True):
            outputs = pipeline.allocate()
            with torch.cuda.stream(render_stream):
                pipeline.render_outputs["distance_to_image_plane"].torch.fill_(1)
            pipeline.process(wp.ones(2, dtype=wp.bool, device=device))
        render_stream.synchronize()
        assert backend.observed_stream == render_stream.cuda_stream
        assert torch.all(outputs["rgb"].torch == 101)
    finally:
        pipeline.close()


def test_configured_initial_chunk_waits_for_multiple_captures():
    """An application can choose a multi-frame first chunk independently of updates."""
    backend = RecordingStream()
    cfg = DepthImageProcessorCfg(backend=backend, initial_frames=2, update_frames=3, max_pending_frames=3)
    pipeline = SensorPostProcessingPipeline(
        [cfg], CameraPostProcessorContext(None, (), 2, 2, 3, "cpu"), cfg.inputs, ["rgb"]
    )
    try:
        output = pipeline.allocate()["rgb"].torch
        pipeline.render_outputs["distance_to_image_plane"].torch.fill_(2)
        pipeline.process(mask(True, True))
        assert not backend.calls and torch.count_nonzero(output) == 0
        pipeline.process(mask(True, True))
        assert [len(c) for c in backend.calls[0][0]] == [2, 2]
        for _ in range(2):
            pipeline.process(mask(True, True))
            assert len(backend.calls) == 1
        pipeline.process(mask(True, True))
        assert [len(c) for c in backend.calls[1][0]] == [3, 3]
        assert torch.all(output == 102)
    finally:
        pipeline.close()


def test_named_inputs_prepare_only_fresh_views_and_keep_renderer_storage_owned():
    """Custom multi-input preparation preserves view order and cannot mutate camera buffers."""
    from isaaclab.renderers import RenderBufferSpec

    from isaaclab_contrib.image_transfer import ImageTransferProcessorCfg

    seen = []

    def prepare(inputs):
        seen.append(inputs["label"].flatten().tolist())
        output = inputs["signal"].to(torch.uint8).expand(-1, -1, -1, 3).clone()
        output[..., 1] = inputs["label"][..., 0]
        inputs["signal"].zero_()
        return output

    backend = RecordingStream()
    cfg = ImageTransferProcessorCfg(
        backend=backend,
        inputs={"signal": RenderBufferSpec(1, wp.float32), "label": RenderBufferSpec(1, wp.int32)},
        prepare_control=prepare,
    )
    pipeline = SensorPostProcessingPipeline(
        [cfg], CameraPostProcessorContext(None, (), 2, 1, 1, "cpu"), cfg.inputs, ["rgb"]
    )
    try:
        rgb = pipeline.allocate()["rgb"].torch
        signal = pipeline.render_outputs["signal"].torch
        signal[:, 0, 0, 0] = torch.tensor([7.0, 9.0])
        pipeline.render_outputs["label"].torch[:, 0, 0, 0] = torch.tensor([30, 40])
        pipeline.process(mask(False, True))
        assert not backend.calls
        signal[1].fill_(99)
        pipeline.process(mask(True, False))
        assert seen == [[40], [30]]
        assert [c.flatten().tolist() for c in backend.calls[0][0]] == [[7, 30, 7], [9, 40, 9]]
        assert signal.flatten().tolist() == [7, 99]
        assert torch.all(rgb == 101)
    finally:
        pipeline.close()


def test_invalid_preparation_does_not_queue_or_advance_the_backend():
    """A malformed custom transform fails before any control is queued or generated."""
    from isaaclab_contrib.image_transfer import ImageTransferProcessorCfg

    malformed = True

    def prepare(inputs):
        return inputs["rgb"].float() if malformed else inputs["rgb"]

    backend = RecordingStream()
    cfg = ImageTransferProcessorCfg(backend=backend, prepare_control=prepare)
    pipeline = SensorPostProcessingPipeline(
        [cfg], CameraPostProcessorContext(None, (), 2, 1, 1, "cpu"), cfg.inputs, ["rgb"]
    )
    try:
        pipeline.allocate()
        pipeline.render_outputs["rgb"].torch.fill_(50)
        with pytest.raises(ValueError, match="Prepared controls"):
            pipeline.process(mask(True, True))
        assert not backend.calls
        malformed = False
        pipeline.render_outputs["rgb"].torch.fill_(80)
        pipeline.process(mask(True, True))
        assert all(torch.all(control == 80) for control in backend.calls[0][0])
    finally:
        pipeline.close()
