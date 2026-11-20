* Added ``Cosmos3Model``, a client of the Cosmos3 world-model service ``isaaclab-cosmos-world-model-server``. The
  Cosmos Framework runs in the service's own environment, so Isaac Lab does not install it; on the service's machine,
  chunks reach the simulation device through CUDA IPC. ``Cosmos3Cfg.domain_name`` and ``num_inference_steps`` select
  the action domain and the window-mode sampling steps of the session.
