# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Application-owned image processing backend contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


class ImageTransferBackend(Protocol):
    """One exclusive batched stream, with one initial seed per camera view.

    Inputs are owned uint8 THWC three-channel controls prepared by the application.
    The processor configuration determines initial and subsequent chunk sizes.
    Outputs must be uint8 THWC sRGB with matching sizes. Calls run on the
    processor's Torch stream; returned tensors must be ready on that stream.
    Reset rows discard all backend-owned episode state and use the given seeds.
    """

    seeds: Sequence[int]

    def step(
        self, controls: list[torch.Tensor], reset_rows: tuple[int, ...], seeds: tuple[int, ...]
    ) -> list[torch.Tensor]:
        """Consume one chunk per camera row and return the corresponding images."""
        ...

    def close(self) -> None:
        """Release stream state; repeated calls must be safe."""
        ...
