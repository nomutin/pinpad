from typing import get_args

import pytest

from src.pinpad import PinPad
from src.pinpad.types import LAYOUTS


@pytest.mark.parametrize("layout", get_args(LAYOUTS))
def test_e2e(layout: str) -> None:
    try:
        env = PinPad.make(layout=layout)  # type: ignore
        env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            env.step(action=action)
        assert True
    except Exception:
        assert False


def test_reset_seed() -> None:
    env = PinPad.make(layout="three")
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert (obs1 == obs2).all()


def test_reset_options() -> None:
    env = PinPad.make(layout="three")
    obs1, _ = env.reset(options={"player": (10, 10)}, seed=42)
    obs2, _ = env.reset(options={"player": (10, 10)}, seed=0)
    assert (obs1 == obs2).all()
