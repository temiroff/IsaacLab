* Added ``isaaclab-cosmos-world-model-server``, which serves Cosmos3 ``unified_v1`` world-model sessions to
  ``isaaclab_neural``: resets from start images and prompts, and steps with chunks of normalized controls. Chunks
  reach a client on the same machine through CUDA IPC. Each session names its action domain and window-mode sampling
  steps; session mode steps a distilled checkpoint in its own number of controls per step.
