* Added the ``isaaclab_neural`` package. Its ``WorldModelManager`` physics backend, ``WorldModelRenderer`` and virtual
  ``Articulation`` let a learned world model supply the joint state and camera frames of a task instead of a physics
  engine, through the task's ``physics=`` and ``renderer=`` presets.
* Added the ``WorldModel`` interface, reset records that the virtual ``Articulation`` starts episodes from, and
  ``at_chunk_boundary`` for terminations that fire where environments can reset.
* Added ``Cosmos3Model``, a client of the Cosmos3 world-model service, and a replay script. The Cosmos Framework
  runs in the service's own environment, so Isaac Lab does not install it.
