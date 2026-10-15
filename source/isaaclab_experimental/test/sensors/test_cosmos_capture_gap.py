# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Cosmos camera modifier measures the time between captures and warns when it is too long."""

import logging
from types import SimpleNamespace

import pytest
import torch
from isaaclab_experimental.cosmos.client.cosmos_modifier import CosmosTransferModifier

pytestmark = pytest.mark.unit


def _modifier(limit=0.05, update_period=0.1):
    """A modifier with only the state the gap tracking reads, so no Cosmos service is needed."""
    modifier = CosmosTransferModifier.__new__(CosmosTransferModifier)
    modifier._cfg = SimpleNamespace(max_capture_gap_s=limit)
    modifier._last_capture_time = None
    modifier._gaps = {"count": 0, "sum": 0.0, "max": 0.0}
    modifier._gap_warned = False
    modifier._sensor = SimpleNamespace(_timestamp=torch.zeros(1), cfg=SimpleNamespace(update_period=update_period))
    return modifier


def _capture_at(modifier, seconds):
    modifier._sensor._timestamp = torch.tensor([seconds])
    modifier._track_capture_gap()


def test_gap_at_30_hz_is_clean_and_measured(caplog):
    modifier = _modifier()
    with caplog.at_level(logging.WARNING):
        for step in range(5):
            _capture_at(modifier, step / 30)
    assert not caplog.records
    stats = modifier.capture_gap_s
    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(1 / 30, abs=1e-6)
    assert stats["max"] == pytest.approx(1 / 30, abs=1e-6)


def test_sparse_captures_warn_once_with_the_update_period(caplog):
    modifier = _modifier(update_period=0.1)
    with caplog.at_level(logging.WARNING):
        for step in range(6):
            _capture_at(modifier, step * 0.1)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "0.100 s apart" in warnings[0].getMessage()
    assert "update_period=0.1" in warnings[0].getMessage()
    assert modifier.capture_gap_s["max"] == pytest.approx(0.1, abs=1e-6)


def test_a_reset_restarts_the_clock_without_counting_a_gap(caplog):
    modifier = _modifier()
    with caplog.at_level(logging.WARNING):
        for seconds in (0.0, 1 / 30, 2 / 30, 0.0, 1 / 30):  # the camera's clock restarts at the reset
            _capture_at(modifier, seconds)
    assert not caplog.records
    assert modifier.capture_gap_s["count"] == 3


def test_a_sensor_without_a_timestamp_is_ignored():
    modifier = _modifier()
    modifier._sensor = SimpleNamespace()
    modifier._track_capture_gap()
    assert modifier.capture_gap_s["count"] == 0


def test_the_limit_is_configurable(caplog):
    modifier = _modifier(limit=0.2)
    with caplog.at_level(logging.WARNING):
        for step in range(4):
            _capture_at(modifier, step * 0.1)
    assert not caplog.records
