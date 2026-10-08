# Image transfer for camera images

`isaaclab_experimental.image_transfer` runs an image-generation model on a camera, live during simulation,
the same way PPISP does: as a camera modifier chain that runs once per captured image. Observation terms
read the generated images from `camera.data.output` like any rendered image. The package is
model-agnostic; it does not load models, download weights, or choose a network transport.

```
renderer (rgb or depth at the model canvas)
  -> control preparation          edge_control / depth_to_control / region_control
  -> image transfer               ImageTransferModifierCfg: chunks, held frame, resets -> model
  -> post-processing (optional)   srgb_to_linear -> PpispModifierCfg
  -> center_crop_resize           back to the camera's original size
  -> camera.data.output["rgb"]    -> observation terms -> policy
```

## Package layout

| Module | Contents |
|---|---|
| `cfg.py` | `ImageTransferModifierCfg` (the camera modifier), `WorkerModelCfg` (a model in a worker process) |
| `modifier.py` | `ImageTransferModifier`: per-view queues, chunk scheduling, held output, resets and seeds |
| `model.py` | `ImageTransferModel` / `ImageTransferStream`: the contract a model implements |
| `worker.py` | `WorkerModel`: the worker-process model and its line protocol |
| `functions.py` | Function modifiers: `edge_control`, `depth_to_control`, `region_control`, `srgb_to_linear`, `center_crop_resize` |
| `camera.py` | `image_transfer_camera`: puts the whole chain on a task camera |

## Use it on a task camera

`image_transfer_camera` returns a copy of a camera whose `rgb` output is generated. The copy renders at the
model canvas (832 x 480 by default) and scales the result back to the camera's size, so observation terms and
policies keep their input shape and the task needs no other change:

```python
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab_experimental.image_transfer import (
    ImageTransferModifierCfg, WorkerModelCfg, image_transfer_camera, srgb_to_linear,
)
from isaaclab_ppisp import PpispModifierCfg

worker = WorkerModelCfg(command=["python3", "/path/to/worker.py"], prompt="A robot hand turns a cube.", control="edge")
env_cfg.scene.base_camera = image_transfer_camera(
    env_cfg.scene.base_camera,
    ImageTransferModifierCfg(backend=worker, initial_frames=1, update_frames=4),
    control="edge",
    post_modifiers=[ModifierCfg(func=srgb_to_linear), PpispModifierCfg(isp_cfg=isp_cfg)],
)
```

The chain can also be written directly in `CameraCfg.modifiers`, keyed by the camera output it reads:

```python
CameraCfg(
    data_types=["rgb"],
    modifiers={
        "distance_to_image_plane": [
            ModifierCfg(func=depth_to_control, params={"near": 0.2, "far": 12.0}),
            ImageTransferModifierCfg(backend=MyModelCfg(), initial_frames=1, update_frames=4),
        ]
    },
)
```

Repeated reads of one capture do not advance the model.

## Models

`ImageTransferModifierCfg.backend` is a `BackendCfg` whose `class_type` builds an `ImageTransferModel`.
Equal configurations share one model through `SimulationContext.get_or_create_backend`, so the model holds
only shared resources such as weights. Each modifier opens its own `ImageTransferStream` with
`open_stream(num_views, seeds)`; the stream holds the temporal state of its views. Without a simulation
context, the modifier builds and owns the model.

`ImageTransferStream.step(controls, reset_rows, seeds)` receives one owned uint8 THWC three-channel control
sequence per view and returns uint8 sRGB images with the same sequence lengths and image size. There is no
implicit crop or resize. Calls run on the current Torch stream; returned tensors must be ready on it.
`close()` must be idempotent. The camera closes its modifiers, and with them their streams, when it is
released.

### Models in a worker process

Models that need their own Python environment, GPU, or machine use `WorkerModelCfg`. The camera starts the
worker with `command` (locally, or for example through `ssh host python3 worker.py`) when it first needs an
image, and stops it when the camera closes. A worker serves one camera view. The protocol is one JSON
message per line:

1. `{"open": {"prompt", "control", "seed", "max_episode_frames", "max_chunks"}}` once, before any control.
2. `{"control": <base64 .npy, uint8 (T, H, W, 3)>}` per chunk; a chunk that starts a new episode also carries
   `"reset_rows": [0]` and `"seeds": [seed]`.
3. The worker answers each chunk on stdout with `COSMOS_RESULT {"video": <base64 .npy, uint8 (T, H, W, 3)>}`.
   Other output is logged to `log_path`.

`examples/sensors/image_transfer_stand_in_worker.py` is a reference worker that tints the controls instead of
running a model.

## Controls

Controls are prepared by modifiers earlier in the chain and must be uint8 `(N, H, W, 3)`:

- `edge_control`: Canny edges of the rendered sRGB image (needs `opencv-python-headless`).
- `depth_to_control`: metric depth, near white and far black; missing, nonfinite, and nonpositive depth is black.
- `region_control`: raw segmentation IDs colored with a fixed palette.

Any other function or class modifier with the same output can take their place.

## Scheduling and resets

`initial_frames` and `update_frames` set the first and subsequent chunk sizes; both default to one. The output
publishes the last generated image and holds it until the next chunk, so larger chunks add observation age as
well as latency. Every view needs a complete chunk before a chunk is generated, and `max_pending_frames` bounds
each view's queue.

Resetting a view, for example when its environment resets, clears its queued controls and its visible image.
Its next chunk uses the initial size, and `step` receives the view index and a new deterministic seed: view `i`
starts from `seed + i`, and each reset advances it by the number of views, modulo `2**31`. The stream must
discard all episode state for those views. An exception or invalid result marks the modifier as failed;
recreate the camera instead of retrying against a stream that may already have advanced.

## PPISP after image transfer

Generated images are display sRGB. To apply a camera look, append `ModifierCfg(func=srgb_to_linear)` and
`PpispModifierCfg`. For sRGB-encoded output, give the PPISP camera response a display gamma of 1/2.2 (see
`display_ppisp_cfg` in `examples/sensors/image_transfer_policy.py`). The inverse sRGB transfer function does not
recover scene HDR or remove a baked camera response; use the renderer's `rgb_radiance` for calibrated PPISP
processing. The image transfer package does not depend on PPISP.

## Examples and validation

```bash
# Chain on CPU with an in-process tint model:
uv run python examples/sensors/depth_image_processor.py --output output/image-transfer-demo.npz
# Kuka Allegro camera task, policy loop with resets, stand-in worker (random actions without --checkpoint):
uv run python examples/sensors/image_transfer_policy.py --output output/image-transfer-policy --episode-seconds 2
uv run python -m pytest source/isaaclab_experimental/test/sensors
```

The policy example writes `policy-validation.json` (episodes, model resets, generated frames, policy image
shape, timings), the observed images, and every generated chunk.

## Limits

A worker serves one camera view, so worker-backed chains run with one environment. Simulation waits for each
chunk, so throughput is bounded by the model.
