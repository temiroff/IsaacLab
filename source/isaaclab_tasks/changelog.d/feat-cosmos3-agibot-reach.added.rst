* Added ``Isaac-Cosmos3-AgiBot-Reach-v0``, an AgiBot A2D bimanual reach task whose head-camera frames come from the
  public Cosmos3 Sim-Bimanual checkpoint through the world model of ``isaaclab_neural``, without Isaac Sim or Kit. Its
  start images are rendered with OvPhysX and OVRTX. Its joint mapping encodes both grippers' forward kinematics as the
  wrists of the ``agibotworld`` action domain, its reach reward reads the same forward kinematics from the joint state,
  and it includes an ``rsl_rl`` PPO configuration with a CNN image encoder. Its policy acts once per chunk of 16 frames
  through the ``chunk_length`` of ``JointPositionActionCfg``.
