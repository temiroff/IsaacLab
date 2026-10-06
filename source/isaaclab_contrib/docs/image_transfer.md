# External image processing for camera observations

`isaaclab_contrib.image_transfer` connects an application-owned image backend
to camera post-processing chains. It does not load models, download assets, or
select a network transport.

## Backend contract

Implement `ImageTransferBackend` in your application. It provides one initial
seed per camera view, `step(controls, reset_rows, seeds)`, and idempotent `close()`.
Each control is an owned uint8 THWC three-channel image sequence. Applications
choose the control representation through named inputs and `prepare_control`.
Returned images must be
uint8 sRGB with the same sequence length, camera dimensions, and channel count.
There is no implicit crop or resize.

Calls run synchronously on the processor's Torch stream. The backend must order
its returned tensors on that stream. Each chain exclusively owns its backend's
stream lifecycle. Model residency and implementation remain application concerns.

`initial_frames` and `update_frames` configure the first and subsequent chunk
sizes; both default to one. Configure them for your backend. Output publishes
the last returned image and holds it until the next complete chunk, so larger
chunks add observation age as well as processing latency. All camera rows must
have a complete chunk before a batch advances. `max_pending_frames` bounds each
row's queue and raises on overrun instead of silently dropping captures.

Resetting a view clears its pending controls and visible RGB. Its next chunk
uses the initial size, and `step` receives the row index and a new deterministic
seed. Reset seeds advance by the number of views per episode, modulo `2**31`.
The backend must discard all episode state for those rows. An exception or
invalid result poisons the processor: recreate the chain instead of retrying
against potentially advanced backend state.

## Pluggable control preparation

`ImageTransferProcessorCfg` accepts any declared camera input buffers and a
stateless `prepare_control(inputs)` callable. It receives selected fresh views
as a dictionary of NHWC Torch tensors and returns one uint8 NHWC three-channel
control on the camera device. Selected inputs are isolated from renderer storage;
queued controls are copied. The backend owns temporal state and episode resets.
There is no modality enum or model-specific import in this interface.

For example, an application can prepare RGB with missing depth hits blacked out:

```python
import torch
import warp as wp
from isaaclab.renderers import RenderBufferSpec
from isaaclab_contrib.image_transfer import ImageTransferProcessorCfg

def prepare_visible_rgb(inputs):
    depth = inputs["distance_to_image_plane"]
    valid = torch.isfinite(depth) & (depth > 0)
    return torch.where(valid, inputs["rgb"], 0)

processor = ImageTransferProcessorCfg(
    backend=backend,
    inputs={
        "rgb": RenderBufferSpec(3, wp.uint8, color_space="srgb"),
        "distance_to_image_plane": RenderBufferSpec(1, wp.float32),
    },
    prepare_control=prepare_visible_rgb,
)
```

A single three-channel uint8 input can pass through without a callable.
An application may prepare edges, region colors, or another image representation
without changing the scheduler. Optional preprocessing libraries remain application
dependencies. Inputs must be supported by the selected camera renderer; accepting
them here does not imply a backend understands them or can combine modalities.

`DepthImageProcessorCfg` is a convenience configuration using the same scheduler:
it converts metric `distance_to_image_plane` with fixed near-white/far-black
bounds. Nonfinite and nonpositive hits become black. Existing depth configurations
continue to work. Use the generic configuration for custom preparation.

## Camera integration

Create the chain after spawning the camera and before simulation reset:

```python
from isaaclab.sensors.post_processing import CameraPostProcessingChain
from isaaclab_contrib.image_transfer import DepthImageProcessorCfg

chain = CameraPostProcessingChain(
    camera,
    [DepthImageProcessorCfg(
        backend=backend, near=0.2, far=12.0,
        initial_frames=2, update_frames=3, max_pending_frames=6,
    )],
    ["rgb"],
    num_views=env.num_envs, device=env.device, stage=env.sim.stage,
)

# After a fresh camera capture:
chain.update()
image = chain.outputs["rgb"].torch
```

Repeated reads of one capture do not advance the backend. Forward environment
resets with `chain.reset(env_ids)` and release the chain with `chain.close()`.
For managed observations, pass the same processor list to `mdp.processed_image`.

## Optional PPISP composition

Append `SrgbToLinearProcessorCfg()` and `PpispProcessorCfg(isp_cfg=...)` to the
processor list to apply relative camera effects to the backend's output.
The converter preserves the original image as `source_rgb` and applies the
inverse sRGB transfer function for PPISP's linear input. This approximation
does not recover scene HDR or remove an already baked camera response. Use
native renderer radiance for physically calibrated PPISP processing.

The core image processor does not depend on PPISP. Applications using that
stage must install the corresponding package separately.

## Local example and validation

`examples/sensors/depth_image_processor.py` exercises the real pipeline on CPU
with a small illustrative tint backend. It requires no model or remote service:

```bash
uv run python examples/sensors/depth_image_processor.py --output output/image-transfer-demo.npz
uv run python -m pytest source/isaaclab_contrib/test/sensors/test_image_transfer.py
```

The example saves the source depth and processed RGB. Unit tests cover named
multi-input preparation, malformed controls, input ownership, chunk scheduling, partial resets, bounded queues, failure handling,
and Torch/Warp ordering. PPISP-specific tests additionally exercise the color
conversion and downstream consumer on CUDA.
