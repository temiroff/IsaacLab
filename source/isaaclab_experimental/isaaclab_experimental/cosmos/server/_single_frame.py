# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent-frame sessions using the resident Sim-Transfer weights."""

from __future__ import annotations


class SingleFrameStream:
    """Generate a fresh bootstrap for every call, releasing its KV/VAE state immediately.

    Episode seeds and prompts persist as CPU metadata. A reset changes only the selected views;
    it never carries image history into a new episode. The service serializes model access.
    """

    def __init__(self, owner, seeds, prompts, modality, height, width, budget):
        self._owner = owner
        self._seeds = list(seeds)
        self._prompts = list(prompts)
        self._counts = [0] * len(seeds)
        self._modality = modality
        self._height = height
        self._width = width
        self._budget = budget
        self._closed = False

    def step(self, controls, reset_rows, seeds, **episode):
        """Return one current image per view; optional prompts correspond to reset_rows."""
        if self._closed:
            raise RuntimeError("The Cosmos camera session is closed.")
        count = len(self._seeds)
        if (
            len(controls) != count
            or any(tuple(control.shape) != (1, self._height, self._width, 3) for control in controls)
        ):
            raise ValueError("Single-frame Cosmos requires exactly one frame per view.")
        if (
            list(reset_rows) != sorted(set(reset_rows))
            or any(type(row) is not int or not 0 <= row < count for row in reset_rows)
            or len(seeds) != len(reset_rows)
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
        ):
            raise ValueError("Resets require distinct ordered views and one valid seed each.")
        if set(episode) - {"prompt"}:
            raise ValueError("Unsupported single-frame episode settings.")
        if "prompt" in episode:
            prompts = episode["prompt"]
            prompts = prompts if isinstance(prompts, list) else [prompts] * len(reset_rows)
            if not reset_rows or len(prompts) != len(reset_rows) or any(
                text is not None and not isinstance(text, str) for text in prompts
            ):
                raise ValueError("Supply one text prompt or None per reset view.")
            for row, prompt in zip(reset_rows, prompts):
                self._prompts[row] = prompt
        for row, seed in zip(reset_rows, seeds):
            self._seeds[row] = seed
            self._counts[row] = 0
        if any(frames >= self._budget for frames in self._counts):
            raise RuntimeError("Cosmos episode horizon exceeded; reset before continuing.")
        owner = self._owner
        child = None
        owner._stream = None
        try:
            child = owner.open_stream(
                num_views=count,
                seeds=tuple((seed + frame) % 2**31 for seed, frame in zip(self._seeds, self._counts)),
                prompt=self._prompts,
                modality=self._modality,
                height=self._height,
                width=self._width,
                max_episode_frames=1,
            )
            images = child.step(controls, (), ())
            self._counts = [frame + 1 for frame in self._counts]
            return images
        finally:
            if child is not None:
                child.close()
            owner.release_generation_state()
            owner._stream = self

    def close(self):
        """Release this logical session; no temporal GPU state survives a step."""
        self._closed = True
        if self._owner._stream is self:
            self._owner._stream = None
