* Added :mod:`isaaclab_experimental.image_transfer` for application-owned image generation from camera outputs as
  camera tensor processors: :class:`~isaaclab_experimental.image_transfer.ImageTransferProcessorCfg` with chunked
  per-view queues that follow the camera's ``env_mask``, per-view resets with deterministic seeds, and a model shared
  through the simulation context, plus :class:`~isaaclab_experimental.image_transfer.FunctionProcessorCfg` and the
  ``depth_to_control`` function.
