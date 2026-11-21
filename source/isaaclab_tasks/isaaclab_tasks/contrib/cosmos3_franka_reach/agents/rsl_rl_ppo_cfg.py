# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO from the generated camera image and the joint positions."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

# Global average pooling keeps the latent small at the model's 480 x 640 frames.
CNN_CFG = RslRlCNNModelCfg.CNNCfg(
    output_channels=[16, 32, 64],
    kernel_size=[8, 4, 3],
    stride=[4, 2, 2],
    activation="elu",
    global_pool="avg",
)


@configclass
class FrankaReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Actor and critic both see the image and the joint positions."""

    # One episode of 8 chunks per rollout.
    num_steps_per_env = 8
    # Episodes start at step 0, so their time-outs fall on chunk boundaries, where the world model can reset.
    init_at_random_ep_len = False
    max_iterations = 1000
    save_interval = 50
    experiment_name = "cosmos3_franka_reach"
    obs_groups = {"actor": ["policy", "image"], "critic": ["policy", "image"]}
    actor = RslRlCNNModelCfg(
        obs_normalization=True,
        hidden_dims=[256, 128],
        activation="elu",
        cnn_cfg=CNN_CFG,
        # A small initial spread keeps the generated motion smooth enough for the world model.
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    critic = RslRlCNNModelCfg(obs_normalization=True, hidden_dims=[256, 128], activation="elu", cnn_cfg=CNN_CFG)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
    )
    # The NaN scan reads every observation back to the host each step.
    check_for_nan = False
