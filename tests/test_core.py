# Copyright 2022 Danijar Hafner.
"""`core.py`のテスト."""

from typing import get_args

import numpy as np
import pytest

from pinpad.core import LAYOUTS, PinPad


@pytest.mark.parametrize("layout", get_args(LAYOUTS))
def test_e2e(layout: str) -> None:
    """`PinPad`の初期化と環境のリセットをテスト."""
    try:
        env = PinPad.make(layout=layout)  # type: ignore[arg-type]
        env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            env.step(action=action)
    except Exception:  # noqa: BLE001
        pytest.fail(f"Failed to test layout: {layout}")


def test_reset_seed() -> None:
    """`reset`の`seed`オプションをテスト."""
    env = PinPad.make(layout="three")
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.array_equal(obs1, obs2)


def test_reset_options() -> None:
    """`reset`の`options`オプションをテスト."""
    env = PinPad.make(layout="three")
    obs1, _ = env.reset(options={"player": (10, 10)}, seed=42)
    obs2, _ = env.reset(options={"player": (10, 10)}, seed=0)
    assert np.array_equal(obs1, obs2)
