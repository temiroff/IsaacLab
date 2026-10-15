# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect an independently running Cosmos service."""

from __future__ import annotations

import argparse
import json
import sys

from isaaclab_experimental.cosmos._protocol import DEFAULT_ENDPOINT, request


def main(args: list[str] | None = None) -> int:
    """Run the ``isaaclab cosmos status`` command."""
    parser = argparse.ArgumentParser(description=__doc__, prog="isaaclab cosmos")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("status", "Inspect the service."),
        ("offload", "Move idle model weights to CPU between rollouts."),
        ("activate", "Restore model weights to the service GPU."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Cosmos service endpoint.")
        command.add_argument("--timeout", type=float, default=600.0, help="Operation timeout [s].")
    options = parser.parse_args(args)
    try:
        reply, _ = request(options.endpoint, {"op": options.command}, timeout=options.timeout)
        print(json.dumps(reply, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Cosmos: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
