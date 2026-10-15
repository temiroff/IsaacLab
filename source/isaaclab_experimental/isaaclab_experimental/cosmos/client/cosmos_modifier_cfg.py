# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Cosmos camera modifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from isaaclab_experimental.image_transfer import ImageTransferModifierCfg

if TYPE_CHECKING:
    from .cosmos_modifier import CosmosTransferModifier


@configclass
class CosmosTransferModifierCfg(ImageTransferModifierCfg):
    """Generate one initial frame, then four frames per Cosmos Sim-Transfer chunk."""

    func: type[CosmosTransferModifier] | str = "{DIR}.cosmos_modifier:CosmosTransferModifier"
    """The Cosmos camera modifier."""

    initial_frames: int = 1
    """Captures required for the first chunk of an episode."""

    update_frames: int = 4
    """Captures required for subsequent chunks."""

    seed: int = 42
    """Initial generation seed; full camera resets advance it deterministically."""

    decision_interval: int = 1
    """Single-frame inference opportunities every N camera captures, starting at reset (e.g. 16)."""

    augmentation_probability: float = 1.0
    """Probability of inference at a single-frame decision point; otherwise publish current raw RGB.

    One draw applies to the camera batch. Shared cameras must use identical scheduling settings.
    """

    max_capture_gap_s: float = 0.05
    """Largest simulated time between two camera captures that is expected to give a clean generated video [s].

    A warning is logged once when the captures Cosmos receives are further apart, which usually means the camera's
    ``update_period`` (or the environment step) is too long. A 0.033 s gap (30 Hz) stayed clean; 0.133 s did not.
    """

    depth_near: float = 0.1
    """Single-frame depth control's white endpoint [m]."""

    depth_far: float = 2.0
    """Single-frame depth control's black endpoint [m]."""

    edge_thresholds: tuple[int, int] = (100, 200)
    """Single-frame Canny thresholds."""
