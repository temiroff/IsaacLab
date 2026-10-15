# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load Cosmos once, then accept camera sessions independently of Isaac Lab."""

from __future__ import annotations

import argparse
import logging
import os

from isaaclab_experimental.cosmos._protocol import DEFAULT_ENDPOINT
from isaaclab_experimental.cosmos.server._framework import CosmosInferenceModel
from isaaclab_experimental.cosmos.server.service import serve


def main(args: list[str] | None = None) -> None:
    """Run the streaming service in the Cosmos Framework environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"unix:///path (same machine, owner-only) or tcp://host:port (default {DEFAULT_ENDPOINT}).",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Incoming control frame rate for temporal conditioning."
    )
    parser.add_argument(
        "--emphasize-control-in-prompt", action="store_true", help="Request precise control alignment in the caption."
    )
    parser.add_argument(
        "--deterministic-cudnn",
        action="store_true",
        help="Disable convolution autotuning and select deterministic cuDNN algorithms after loading the model.",
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--kv-window",
        type=int,
        default=30,
        help="Generation history in latent frames (Sim-Transfer recipe: 30); shorter is faster but remembers less.",
    )
    parser.add_argument(
        "--attention-sink", type=int, default=3, help="Earliest latent frames always kept in the history window."
    )
    options = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["COSMOS_TRAINING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

    def load_model():
        model = CosmosInferenceModel(
            options.checkpoint,
            options.device,
            use_compile=not options.no_compile,
            kv_window=options.kv_window,
            attention_sink=options.attention_sink,
            fps=options.fps,
            emphasize_control_in_prompt=options.emphasize_control_in_prompt,
        )
        if options.deterministic_cudnn:
            import torch

            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        return model

    serve(load_model, options.endpoint, warmup=options.warmup)


if __name__ == "__main__":
    main()
