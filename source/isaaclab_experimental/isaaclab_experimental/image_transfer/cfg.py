# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for image transfer: the camera modifier and the worker-process model."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg

if TYPE_CHECKING:
    from .modifier import ImageTransferModifier
    from .worker import WorkerModel


@configclass
class ImageTransferModifierCfg(ModifierCfg):
    """Generate images from uint8 three-channel controls with an application-owned model.

    Place it after a modifier that prepares controls, for example :func:`depth_to_control`, in a
    camera modifier chain so that it runs once per captured image:

    .. code-block:: python

        CameraCfg(
            data_types=["rgb"],
            modifiers={
                "distance_to_image_plane": [
                    ModifierCfg(func=depth_to_control, params={"near": 0.1, "far": 10.0}),
                    ImageTransferModifierCfg(backend=MyModelCfg(), update_frames=4),
                ]
            },
        )

    The output holds the latest generated image between chunks. Resets discard the reset views'
    queued controls and episode state.
    """

    func: type[ImageTransferModifier] | str = "{DIR}.modifier:ImageTransferModifier"
    """Image transfer modifier class."""

    backend: BackendCfg = MISSING
    """Model configuration. Equal configurations share one model through the simulation context; the
    constructed model must implement :class:`ImageTransferModel`."""

    seed: int = 0
    """Initial seed of view 0. View ``i`` starts from ``seed + i``; each reset advances a view's seed by
    the number of views, modulo ``2**31``."""

    initial_frames: int = 1
    """Captures required for the first chunk of each episode."""

    update_frames: int = 1
    """Captures required for subsequent chunks."""

    max_pending_frames: int = 8
    """Maximum queued captures per view before backpressure raises an error."""

    output: str = "rgb"
    """Camera output produced from the controls."""


@configclass
class WorkerModelCfg(BackendCfg):
    """Image transfer model served by a worker process; see :mod:`.worker` for the protocol.

    Use it when the model needs its own Python environment, GPU or machine. The camera starts the worker
    through :attr:`command` (locally, or over SSH) when it first needs an image, sends the settings below,
    and stops the worker when the camera closes. A worker serves one camera view.
    """

    class_type: type[WorkerModel] | str = "{DIR}.worker:WorkerModel"
    """Worker model class."""

    command: list[str] = []
    """Argv that starts the worker, for example ``["python3", "/path/to/worker.py"]``."""

    prompt: str | None = None
    """Text description of the images to generate. Defaults to None, which uses the worker's default."""

    control: str | None = None
    """Control type the camera chain prepares, for example ``"edge"`` or ``"depth"``, so the worker can
    select its matching model. Defaults to None, which uses the worker's default."""

    max_episode_frames: int | None = None
    """Longest episode in camera frames; environments must reset before it. Defaults to None, which uses
    the worker's default."""

    max_chunks: int | None = None
    """Model calls allowed over the stream's lifetime, across episodes. Defaults to None, which uses the
    worker's default."""

    log_path: str | None = None
    """File receiving the worker's log. Defaults to None, which forwards it to this process's stderr."""

    record_dir: str | None = None
    """New directory receiving every generated chunk as ``.npy``. Defaults to None, which records nothing."""

    timeout_s: float = 600.0
    """Longest wait for one chunk [s], including model start-up for the first chunk."""
