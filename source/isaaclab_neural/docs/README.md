# Isaac Lab: Neural simulators

`isaaclab_neural` lets a learned **world model** replace the physics engine. A task keeps its ordinary articulation
and cameras; the world model supplies the joint state and the camera frames through the same interfaces.

```text
Policy / trainer  ->  Task  ->  Isaac Lab  ->  NeuralManager  ->  WorldModel
```

The world model only generates images. The manager keeps each environment's joint state itself (the reset record's
state at reset, then the last joint target), so joint observations and actions behave as on a physics backend.

## Components

| Component | Role |
| --- | --- |
| `NeuralCfg` | `PhysicsCfg` selected as the task's `physics=` preset. Holds the model configuration and the reset records. |
| `NeuralManager` | Physics backend (`backend_name = "neural"`). Buffers one joint target per step, calls the model once per chunk, and keeps the predicted frames of each view. |
| `WorldModel` | Abstract model: `spec()`, `reset()`, `write()`, `step()`, `close()`. |
| `NeuralRendererCfg` | Selected as a camera's `renderer_cfg`, with the model view the camera shows. Copies that view's frames into the camera, including its frame history. |
| `Articulation` | Virtual articulation. Shares the manager's joint state and targets, so the task's joint observation and action terms work unchanged. Its reset starts an episode from a reset record. Everything that needs a body raises. |
| `at_chunk_boundary` | Makes a termination function fire only where environments can reset. |
| `Cosmos3Cfg`, `Cosmos3Model` | Client of the Cosmos3 world-model service (`isaaclab_neural.models.cosmos3`). |

## Using a task's configuration

The task declares an `ArticulationCfg` and a `CameraCfg` as it would for a physics backend, and selects the world
model with `NeuralCfg(model_cfg=..., reset_manifest=...)` and `NeuralRendererCfg()`. The camera needs
`frame_history_length` equal to the model's chunk length and the model's image size.

* **Views**: a model predicts one or more named camera views (`WorldModelSpec.views`). Each world-model camera shows
  one of them, set with `NeuralRendererCfg(view=...)`; for a model with one view, `view` can stay `None`. Several
  cameras of a multi-view model each select their own view.
* **Outputs**: a model predicts one or more camera outputs per view (`WorldModelSpec.outputs`, default `("rgb",)`),
  named as camera data types. A camera requests them in `data_types`, and `NeuralRendererCfg.outputs` declares their
  layout, RGB by default; for depth, for example, add `RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32)`.
* **Joints and limits** come from the spawned USD asset (a URDF spawner converts to USD): the movable joints in
  traversal order, or in the order of `ArticulationCfg.joint_ordering`, which must match the reset records. Joint
  limits are read from the USD, and `soft_joint_pos_limit_factor` applies as usual.
* **Observations and actions**: `joint_pos` terms read the joint state and `JointPositionAction` writes the target.
  Velocity and effort targets, poses, forces and physical properties raise `NotImplementedError`.
* **Resets**: `Articulation.reset`, which the scene calls before the reset events, samples a reset record (`image`,
  `state`, `prompt`, `split` in the JSON manifest) and starts the episode from it: the record's start image becomes the
  camera's frames (`image` is a path, `{view: path}`, or `{view: {output: path}}` with `.npy` arrays for outputs other
  than RGB), and its joint state the joint
  state and the default joint position. The model restarts the environment from the record's image and prompt at the end
  of the next chunk, before it predicts that chunk. The standard `reset_joints_by_offset` and `reset_joints_by_scale`
  events therefore start from the record; a non-zero range moves the joints away from the start image the model sees, so
  randomize through the records instead. `reset_scene_to_default`, the default reset event, also resets the root pose,
  which the world model does not have, so a task replaces the default event configuration. Training samples records at
  random; other splits visit them in order. `envs_per_reset_record` consecutive environments share a record (default 1);
  a group-relative algorithm such as GRPO sets it to its group size so that a group compares rollouts of one scene.
* **Device**: joint state, targets, frames and the start images of the split's records stay on the simulation device.
  Resets and the steps inside a chunk read nothing back to the host. The step that ends a chunk, which waits for the
  model anyway, reads which environments reset, since the records' prompts are host strings.

## Timing

A model predicts `chunk_length` frames from `chunk_length` joint targets. Use `sim.dt = 1 / fps` and either:

* `decimation = 1`: one environment step is one frame, and the model runs on the last step of a chunk. The frames of
  a chunk arrive together, so a reward on them is paid in the step that completes the chunk.
* `decimation = chunk_length` with `JointPositionActionCfg(chunk_length=chunk_length)`: one environment step is
  one chunk, so the policy acts once per chunk and sees each new chunk. With `interpolate=True` the policy sets one
  target per joint and the arm reaches it in equal increments over the chunk; otherwise it sets every target of the
  chunk.

Isaac Lab resets an environment in any step in which a termination fires, but the model can only restart between
chunks, so a reset in the middle of a chunk raises when the chunk ends.

* Make `episode_length_s` a multiple of the chunk duration, so that the time-out falls on a boundary.
* A termination you add must fire only on boundaries. Wrap its function with `at_chunk_boundary`:

  ```python
  done = DoneTerm(func=at_chunk_boundary(my_condition), params={...})
  ```

  The condition is judged on the step that completes a chunk. A condition that stays true, such as a success or a
  failure, is caught at the next boundary, at most one chunk late; one that holds only briefly inside a chunk is
  missed.
* Each environment resets at its own next boundary; it does not wait for the others.

## Add a world model

Implement the `WorldModel` interface, subclass `ModelCfg` and set its `class_type`. The manager builds the model as
`class_type(cfg, num_envs, device)` with the simulation device, and the model keeps what it needs of those:

| Method | Responsibility |
| --- | --- |
| `spec()` | Chunk length, frame rate, image size, the largest number of environments, the inputs it needs, and the names of the views and outputs. |
| `reset(env_ids, images, prompts, seeds)` | Restart the selected environments (host lists) from their start frames, `{view: {output: (K, H, W, C)}}`. Called right before `step()`, for the environments that reset since the last chunk. |
| `write(name, value)` | Set one of the inputs the spec lists, named by `WorldModelAttribute`: the chunk's joint targets `(N, T, J)` (`JOINT_TARGETS`) or the joint state before the chunk `(N, J)` (`START_JOINT_STATE`). Called before every `step()`. |
| `step()` | Predict the next chunk from the inputs written, as frames `{view: {output: (N, T, H, W, C)}}`. |
| `close()` | Release resources. |

Frames and inputs are torch tensors on the device. Each written input is the manager's buffer, reused for the next
chunk, so a model that keeps it copies it. A new input needs the backend to provide it as a `WorldModelAttribute`; the
manager rejects a model whose spec needs one it does not provide. Construction is cheap; load the weights on the first call.

## Cosmos3

Cosmos3 runs in its own process, in the Cosmos Framework's environment, so Isaac Lab does not install the Framework.
`Cosmos3Model` is the client: it reads the chunk length, frame rate and image size from the service, sends start
images and controls, and receives frames. The service is the world-model counterpart of the Cosmos Transfer service
in [`cosmos_transfer.md`](../../isaaclab_experimental/docs/cosmos_transfer.md) and uses the same transport.

In a server terminal, create the Framework environment at commit `3a84cd3b67f60520f523e184fa7b82011bc75cb4` of
[cosmos-framework](https://github.com/NVIDIA/cosmos-framework), install the Cosmos service package into it, and start
the service with a `unified_v1` checkpoint:

```bash
cd "$COSMOS_FRAMEWORK_ROOT"
uv sync --locked --all-extras --group=cu130-train
uv run --no-sync uv pip install --editable "$ISAAC_LAB_ROOT/source/isaaclab_experimental/isaaclab_experimental/cosmos"
export LD_LIBRARY_PATH=
uv run --no-sync isaaclab-cosmos-world-model-server --checkpoint "$COSMOS3_CHECKPOINT"
```

Wait for `Cosmos ready at ...`, then run the task in the Isaac Lab environment. A task selects the model with
`NeuralCfg(model_cfg=Cosmos3Cfg(joint_mapping=...))`. `Cosmos3Model.step` converts joint targets to 59 normalized
controls with the task's `JointMappingCfg`, so the robot description and the normalization stay with the task.

The service's `--mode window` (default) serves any number of environments, one after the other; `--mode session`
serves one. With the service on the same machine and a GPU this process can access directly, chunks stay on the GPU
(CUDA IPC, Linux); otherwise they go through the socket, which carries about 16 environments of 16 frames at
480 x 640 per message. `Cosmos3Cfg.transport` forces either one.

### Example: AgiBot reach on a public checkpoint

`Isaac-Cosmos3-AgiBot-Reach-v0` shows the whole workflow with public weights only, the
[`nvidia/Cosmos3-Nano-Sim-Bimanual`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual) checkpoint, and no
Isaac Sim or Kit at any step. The robot is Isaac Lab's AgiBot A2D, the robot of the checkpoint's `agibotworld` domain,
seen from its head camera.

**What the world model replaces, and what stays Isaac Lab.** The world model is the physics and the camera: from the
start image and both arms' joint targets it predicts the next 16 camera frames. Everything else is an ordinary
manager-based task: the scene, the joint-position actions, the observations, the reward and terminations, the policy and
PPO training with `rsl_rl`. The joint state comes from the joint targets, so joint observations and a reward computed
from the joints are exact; only the image is generated.

**What a task provides.** For another robot or another model, these are the pieces to replace:

| Piece | In the example | What it does |
| --- | --- | --- |
| World model | `Cosmos3Cfg(domain_name="agibotworld")` in `NeuralCfg(model_cfg=...)` | The model, here the Cosmos3 service. Any `WorldModel` fits (see "Add a world model"). |
| Actions | `JointPositionActionCfg(chunk_length=16, interpolate=True)` | One environment step is one chunk: the policy sets the 14 arm joint targets once per chunk, around the reset record's pose, and the arms reach them in 16 equal increments. The task's `ResetPoseJointPositionAction` reads that pose again at every reset; `JointPositionAction` would keep the A2D's initial pose. `env.actions.arms.interpolate=false` lets the policy set all 16 targets (224 numbers). |
| Joint mapping | `AgibotJointMappingCfg` | Converts a chunk of joint targets into the model's action space. Here: forward kinematics to each gripper's pose relative to the chunk's start pose, plus its opening, as the two wrists of the checkpoint's `agibotworld` domain, normalized with the statistics the checkpoint ships. |
| Reset records | `assets/resets.json` and its PNGs | Start image, joint state and prompt of each episode start. `render_agibot_reset_records.py` renders them with OvPhysX and OVRTX, without Kit, and checks the mapping's forward kinematics against the simulated grippers. |
| Camera | `CameraCfg(renderer_cfg=NeuralRendererCfg(), frame_history_length=16)` | Shows the generated frames; placed where the AgiBot World head camera is and where the start images were rendered from, at 352 x 640. |
| Reward | `TcpGoalDistance` | Distance from each gripper's fingertips to its goal, from the forward kinematics of the joint state. |

**Run it.** In the Cosmos Framework environment, download the checkpoint and start the service at the task's image
size. The checkpoint is distilled to a fixed four-step schedule, which the service samples in `--mode session`;
session mode serves one environment.

```bash
cd "$ISAAC_LAB_ROOT"
uv run --project "$COSMOS_FRAMEWORK_ROOT" --no-sync hf download nvidia/Cosmos3-Nano-Sim-Bimanual \
    --local-dir .pretrained_checkpoints/Cosmos3-Nano-Sim-Bimanual
export LD_LIBRARY_PATH=
uv run --project "$COSMOS_FRAMEWORK_ROOT" --no-sync isaaclab-cosmos-world-model-server \
    --checkpoint .pretrained_checkpoints/Cosmos3-Nano-Sim-Bimanual --mode session --image-size 352 640
```

Then, in the Isaac Lab environment:

```bash
uv sync --extra ov --extra rsl-rl --extra video
# Optional: render new start images (the task includes a set). OvPhysX poses the A2D and OVRTX renders the camera.
uv run python scripts/tools/cosmos/render_agibot_reset_records.py
# Train with PPO and record the task's camera.
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cosmos3-AgiBot-Reach-v0 --max_iterations 10 \
    --video sensor:camera --video_length 16
# Run the trained policy (the latest checkpoint of the experiment) and record it.
uv run isaaclab play --rl_library rsl_rl --task Isaac-Cosmos3-AgiBot-Reach-v0 \
    --video sensor:camera --video_length 16
```

* `--rl_library rsl_rl` selects the task's `rsl_rl` PPO configuration: a CNN on the generated image plus the joint
  positions, one episode (8 steps of one chunk each) per rollout.
* `--video sensor:camera` records the task's camera, which is the image the policy sees. No viewer is started. The
  camera shows the last frame of each generated chunk (all 16 frames are in its frame history), so the video has
  one frame per step, per chunk. `--video_length 16` is 16 chunks, two episodes, so the video passes a reset to a new
  start image.
* A service on another endpoint is selected with the override `env.sim.physics.model_cfg.endpoint=unix:///path` or
  `env.sim.physics.model_cfg.endpoint=tcp://host:port`.

**What to expect.** On an RTX PRO 6000, each chunk of 16 frames takes about 0.8 s once the service has opened its
session, and a PPO iteration (one episode of 8 chunks) about 7 s. With scripted joint targets that move both grippers
to their goals, the generated grippers follow their paths and the A2D keeps its shape; the model gives the arms the
silver forearms of the AgiBot World robots as they come into view. A short PPO run shows that the loop runs (start
images, generated chunks, resets, updates); it does not produce a useful policy.

#### What a model needs from the robot

**What Cosmos3 gets per chunk.** One image (the start image at a reset, then its own last frame), 16 actions (one
per frame), the prompt and a seed. It returns 16 frames and no joint state.

**How joints become actions.** The public checkpoint accepts only the action format it was trained on: per arm, a
gripper pose and opening (`unified_v1`, 59 numbers). It has no input for joint angles. The task's joint mapping
therefore converts each arm's 7 joint targets with forward kinematics into a gripper pose relative to the start of
the chunk. It is a direct translation, with no other logic. The forward kinematics and the camera follow the AgiBot
World G1 description the checkpoint's data was recorded with, so the actions mean what they meant in training.

**Other robots.** A robot the checkpoint never saw, such as a Franka, keeps its first frame and then drifts: the model
shortens or bends the arm or detaches the gripper. A robot outside the trained format, such as a Shadow Hand, cannot
be expressed as a gripper pose and opening at all. Both need Cosmos3 fine-tuned on videos of that robot recorded with
its joint actions, and a joint mapping that passes the joints through unchanged. Nothing else in Isaac Lab changes: the
task, the reward, PPO and the commands stay the same.

## Tests

```bash
uv run python -m pytest source/isaaclab_neural/test -q
```

The tests use a fake model: no weights, GPU or simulation app.
