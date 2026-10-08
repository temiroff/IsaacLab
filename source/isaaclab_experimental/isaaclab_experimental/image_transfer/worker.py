# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image transfer model served by a worker process, for models that need their own environment.

The worker is any program that speaks this line protocol on stdin/stdout. Each message is one JSON line.

1. Isaac Lab sends ``{"open": {"prompt": ..., "control": ..., "seed": ..., "max_episode_frames": ...,
   "max_chunks": ...}}`` once, before any control. Keys whose value is None are omitted.
2. For every chunk, Isaac Lab sends ``{"control": <base64 .npy of uint8 (T, H, W, 3)>}``. A chunk that starts
   a new episode also carries ``"reset_rows": [0]`` and ``"seeds": [seed]``.
3. The worker answers each chunk with ``COSMOS_RESULT {"video": <base64 .npy of uint8 (T, H, W, 3)>}`` on
   stdout, optionally with ``"model_seconds"``. Other stdout lines and stderr are logged.
4. Closing stdin ends the worker.

The ``COSMOS_RESULT`` prefix is kept for compatibility with existing workers.
"""

from __future__ import annotations

import base64
import io
import json
import queue
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .cfg import WorkerModelCfg

RESULT_PREFIX = "COSMOS_RESULT "


class WorkerStream:
    """One worker process generating images for a single camera view."""

    def __init__(self, cfg: WorkerModelCfg, seed: int):
        """Start the worker and send its settings.

        Args:
            cfg: Worker configuration.
            seed: Seed of the first episode.
        """
        self.cfg = cfg
        self.record_dir = Path(cfg.record_dir) if cfg.record_dir is not None else None
        self.recorded_frames = 0
        self.resets = 0
        self.chunk_timings: list[dict] = []
        self.closed = False
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=False)
        self._resources = ExitStack()
        log = self._resources.enter_context(open(cfg.log_path, "w", encoding="utf-8")) if cfg.log_path else None  # noqa: SIM115
        self._log = log if log is not None else sys.stderr
        self.process = subprocess.Popen(
            list(cfg.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log if log is not None else None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._responses: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        settings = {
            "prompt": cfg.prompt,
            "control": cfg.control,
            "seed": int(seed),
            "max_episode_frames": cfg.max_episode_frames,
            "max_chunks": cfg.max_chunks,
        }
        self._send({"open": {key: value for key, value in settings.items() if value is not None}})

    def step(
        self, controls: list[torch.Tensor], reset_rows: tuple[int, ...], seeds: tuple[int, ...]
    ) -> list[torch.Tensor]:
        """Send one control chunk and return the generated images.

        Args:
            controls: One uint8 ``(T, H, W, 3)`` chunk for the single view.
            reset_rows: ``(0,)`` when this chunk starts a new episode, otherwise empty.
            seeds: Seed of the new episode, when ``reset_rows`` is ``(0,)``.

        Returns:
            The generated uint8 ``(T, H, W, 3)`` images.
        """
        if len(controls) != 1 or tuple(reset_rows) not in ((), (0,)):
            raise ValueError("A worker stream serves exactly one camera view.")
        started = time.perf_counter()
        payload = io.BytesIO()
        np.save(payload, controls[0].cpu().numpy(), allow_pickle=False)
        request = {"control": base64.b64encode(payload.getvalue()).decode("ascii")}
        if reset_rows:
            self.resets += 1
            request.update(reset_rows=[0], seeds=[int(seeds[0])])
        self._send(request)
        try:
            response = self._responses.get(timeout=self.cfg.timeout_s)
        except queue.Empty as error:
            raise RuntimeError(f"The image transfer worker timed out; see {self._log_name()}.") from error
        if response is None:
            raise RuntimeError(f"The image transfer worker exited; see {self._log_name()}.")
        images = np.load(io.BytesIO(base64.b64decode(response["video"])), allow_pickle=False)
        if images.dtype != np.uint8 or images.shape != tuple(controls[0].shape):
            raise ValueError("The worker must return one uint8 RGB image per control frame.")
        self.chunk_timings.append(
            {
                "frames": len(images),
                "round_trip_seconds": time.perf_counter() - started,
                "model_seconds": response.get("model_seconds"),
            }
        )
        if self.record_dir is not None:
            np.save(self.record_dir / f"chunk-{self.recorded_frames:06d}.npy", images, allow_pickle=False)
            self.recorded_frames += len(images)
        return [torch.from_numpy(images.copy()).to(controls[0].device)]

    def close(self) -> None:
        """Close stdin, wait for the worker to exit and release the log. Repeated calls are safe."""
        if self.closed:
            return
        self.closed = True
        with suppress(OSError):
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self._reader.join(timeout=10)
        self.process.stdout.close()
        self._resources.close()

    def _send(self, message: dict) -> None:
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except OSError as error:
            raise RuntimeError(f"The image transfer worker exited; see {self._log_name()}.") from error

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                if line.startswith(RESULT_PREFIX):
                    self._responses.put(json.loads(line[len(RESULT_PREFIX) :]))
                else:
                    self._log.write(line)
                    self._log.flush()
        finally:
            self._responses.put(None)

    def _log_name(self) -> str:
        return self.cfg.log_path or "the worker's stderr"


class WorkerModel:
    """:class:`~isaaclab_experimental.image_transfer.ImageTransferModel` whose streams are worker processes."""

    def __init__(self, cfg: WorkerModelCfg):
        """Validate the configuration. Workers start when streams open.

        Args:
            cfg: Worker configuration.

        Raises:
            ValueError: If the command, prompt or episode limits are invalid.
        """
        if not cfg.command:
            raise ValueError("WorkerModelCfg.command must start the worker, for example ['python3', 'worker.py'].")
        if cfg.prompt is not None and not cfg.prompt.strip():
            raise ValueError("The prompt must not be empty; use None for the worker's default.")
        if cfg.max_episode_frames is not None and cfg.max_episode_frames < 1:
            raise ValueError("max_episode_frames must be positive.")
        if cfg.max_chunks is not None and cfg.max_chunks < 1:
            raise ValueError("max_chunks must be positive.")
        self.cfg = cfg
        self.streams: list[WorkerStream] = []

    def open_stream(self, num_views: int, seeds: tuple[int, ...]) -> WorkerStream:
        """Start a worker for one camera view.

        Args:
            num_views: Number of views; must be 1.
            seeds: Seed of the view's first episode.

        Returns:
            The stream.
        """
        if num_views != 1:
            raise ValueError("A worker serves one camera view; use one camera per worker.")
        stream = WorkerStream(self.cfg, seeds[0])
        self.streams.append(stream)
        return stream

    def close(self) -> None:
        """Close every stream. Repeated calls are safe."""
        streams, self.streams = self.streams, []
        for stream in streams:
            stream.close()
