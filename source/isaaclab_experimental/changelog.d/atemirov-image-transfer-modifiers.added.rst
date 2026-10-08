* Added :mod:`isaaclab_experimental.image_transfer` for image generation from camera outputs as camera modifiers:
  :class:`~isaaclab_experimental.image_transfer.ImageTransferModifierCfg` with chunked per-view queues, per-view
  resets with deterministic seeds, and a model shared through the simulation context;
  :class:`~isaaclab_experimental.image_transfer.WorkerModelCfg` for models served by a worker process;
  :func:`~isaaclab_experimental.image_transfer.image_transfer_camera` to put the chain on a task camera; and the
  ``edge_control``, ``depth_to_control``, ``region_control``, ``srgb_to_linear`` and ``center_crop_resize``
  function modifiers.
