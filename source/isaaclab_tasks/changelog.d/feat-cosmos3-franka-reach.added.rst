* Added ``Isaac-Cosmos3-Franka-Reach-v0``, a Franka reach task whose camera frames come from the public Cosmos3
  Sim-Bimanual checkpoint through the world model of ``isaaclab_neural``, starting from reset images rendered with
  PhysX and RTX. Its joint mapping encodes the hand's forward kinematics as the right wrist of the ``xdof_yam`` action
  domain, its reach reward reads the same forward kinematics from the joint state, and it includes an ``rsl_rl`` PPO
  configuration with a CNN image encoder.
