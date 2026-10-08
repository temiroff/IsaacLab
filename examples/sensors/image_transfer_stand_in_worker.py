# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stand-in image transfer worker: a reference implementation of the worker protocol.

It tints each control frame instead of running a learned model, so the full camera, worker and reset
path can be checked on any machine. A real worker loads its model where this one prints its settings,
and generates images where this one tints them. See :mod:`isaaclab_experimental.image_transfer.worker`.

Start it through :class:`~isaaclab_experimental.image_transfer.WorkerModelCfg`, for example
``WorkerModelCfg(command=[sys.executable, "examples/sensors/image_transfer_stand_in_worker.py"])``.
"""

import base64
import io
import json
import sys

import numpy as np

TINT = np.array([1.0, 0.6, 0.2])


def main():
    for line in sys.stdin:
        request = json.loads(line)
        if "open" in request:
            # A real worker loads its model here, using the prompt, control type and seed.
            print(
                "stand-in worker settings: " + json.dumps(request["open"], sort_keys=True), file=sys.stderr, flush=True
            )
            continue
        if request.get("reset_rows"):
            # A real worker restarts its episode state here, with the new seed.
            print("stand-in worker reset: seeds " + json.dumps(request["seeds"]), file=sys.stderr, flush=True)
        control = np.load(io.BytesIO(base64.b64decode(request["control"])), allow_pickle=False)
        payload = io.BytesIO()
        np.save(payload, np.rint(control * TINT).astype(np.uint8), allow_pickle=False)
        video = base64.b64encode(payload.getvalue()).decode("ascii")
        print("COSMOS_RESULT " + json.dumps({"video": video}), flush=True)


if __name__ == "__main__":
    main()
